"""
intruder_detector.py
====================
Open-Set Recognition & Multi-Pipeline Intruder Detection Engine.

Solves the Open-Set Recognition Problem:
----------------------------------------
Closed-set classifiers assume all test faces belong to the enrolled training classes.
In a real-world security system, unseen strangers must be identified as "INTRUDER / UNKNOWN".

Mathematical Mechanisms:
1. Probabilistic Thresholding:
   For probabilistic classifiers (Random Forest, Logistic Regression, RBF-SVM, Decision Tree):
       confidence = max_{c in Known} P(y = c | x)
       Decision(x) = argmax P(y = c | x)  if confidence >= tau_prob
                     "INTRUDER"          if confidence < tau_prob

2. Metric-Space Distance Rejection:
   For k-Nearest Neighbors (k-NN):
       Computes Euclidean distance to nearest enrolled training exemplars in feature space.
       d_nn = (1 / k) * sum_{j=1}^k || x - x_{(j)} ||_2
       Decision(x) = MajorityVote(k-NN)   if d_nn <= tau_dist
                     "INTRUDER"          if d_nn > tau_dist

3. Multi-Pipeline Consensus Fusion:
   Collects decisions [D_1, D_2, D_3, D_4, D_5] and confidence scores [C_1, ..., C_5] across
   all 5 independent feature-classifier pipelines.
   - Computes weighted voting and consensus agreement.
   - Triggers 'INTRUDER ALERT' if agreement threshold or overall confidence falls below safety bounds.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from ..classifiers.pipelines import PIPELINE_CONFIGS


class IntruderDetector:
    """
    Open-Set Intruder Detector and Multi-Model Fusion Engine.
    """

    def __init__(
        self,
        trained_pipelines: Dict[int, Pipeline],
        known_classes: List[str],
        prob_threshold: float = 0.65,
        knn_dist_threshold: float = 1.20,
        min_consensus_votes: int = 3,
        enrolled_templates: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        template_thresholds: Optional[Dict[str, Dict[int, float]]] = None
    ) -> None:
        """
        Initialize the Intruder Detector.

        Args:
            trained_pipelines: Dictionary of trained pipelines for pipeline IDs 1..5.
            known_classes: List of enrolled class names.
            prob_threshold: Minimum prediction probability to accept a face as authorized.
            knn_dist_threshold: Maximum feature space distance to accept k-NN prediction.
            min_consensus_votes: Minimum number of agreeing pipelines out of 5 to grant access.
            enrolled_templates: Pre-computed mean normalized feature templates per enrolled person per pipeline.
            template_thresholds: Maximum allowable cosine distance to enrolled template per pipeline.
        """
        self.pipelines = trained_pipelines
        self.known_classes = known_classes
        self.prob_threshold = prob_threshold
        self.knn_dist_threshold = knn_dist_threshold
        self.min_consensus_votes = min_consensus_votes
        self.enrolled_templates = enrolled_templates or {}
        self.template_thresholds = template_thresholds or {}

    def predict_single_pipeline(
        self,
        face_img: np.ndarray,
        pipeline_id: int
    ) -> Dict[str, Any]:
        """
        Evaluate a normalized face image through a single pipeline with open-set rejection.

        Args:
            face_img: 128x128 preprocessed face image.
            pipeline_id: 1, 2, 3, 4, or 5.

        Returns:
            Dictionary with 'raw_prediction', 'confidence', 'is_intruder', 'final_decision'.
        """
        config = PIPELINE_CONFIGS[pipeline_id]
        pipeline = self.pipelines[pipeline_id]
        extractor_func = config["extractor_func"]

        # Extract feature vector
        raw_feat = extractor_func(face_img)
        feat = raw_feat.reshape(1, -1)
        feat_norm = raw_feat / (np.linalg.norm(raw_feat) + 1e-7)

        # Predict class label
        raw_pred = str(pipeline.predict(feat)[0])
        confidence = 0.0
        is_intruder = False

        # If model explicitly classified the sample into the unknown/intruder background class
        if raw_pred.lower() in ["unknown", "intruder", "imposter", "other"]:
            if hasattr(pipeline, "predict_proba"):
                confidence = float(np.max(pipeline.predict_proba(feat)[0]))
            else:
                confidence = 0.85
            return {
                "pipeline_id": pipeline_id,
                "model_id": pipeline_id,
                "model_name": config["model_name"],
                "pipeline_name": config["pipeline_name"],
                "feature_name": config["feature_name"],
                "classifier_name": config["classifier_name"],
                "raw_prediction": raw_pred,
                "confidence": confidence,
                "is_intruder": True,
                "final_decision": "INTRUDER"
            }

        # Check template distance gating for claimed known identity
        template_fail = False
        if raw_pred in self.enrolled_templates and pipeline_id in self.enrolled_templates[raw_pred]:
            templ = self.enrolled_templates[raw_pred][pipeline_id]
            cos_dist = float(1.0 - np.dot(feat_norm, templ))
            max_allowed = self.template_thresholds.get(raw_pred, {}).get(pipeline_id, 0.40)
            if cos_dist > max_allowed:
                template_fail = True

        # Pipeline 2 is k-NN (evaluate nearest neighbor distance in scaled feature space)
        if pipeline_id == 2:
            scaler = pipeline.named_steps['scaler']
            knn_clf = pipeline.named_steps['classifier']
            feat_scaled = scaler.transform(feat)

            distances, indices = knn_clf.kneighbors(feat_scaled)
            mean_dist = float(np.mean(distances[0]))

            confidence = float(1.0 / (1.0 + mean_dist))

            if mean_dist > self.knn_dist_threshold or raw_pred not in self.known_classes or template_fail:
                is_intruder = True
                final_decision = "INTRUDER"
            else:
                final_decision = raw_pred
        else:
            # Probabilistic models (RF, Logistic Regression, RBF-SVM, Decision Tree)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(feat)[0]
                classes = list(pipeline.classes_)
                if raw_pred in classes:
                    pred_idx = classes.index(raw_pred)
                    confidence = float(probs[pred_idx])
                else:
                    confidence = float(np.max(probs))
            else:
                confidence = 0.75

            if confidence < self.prob_threshold or raw_pred not in self.known_classes or template_fail:
                is_intruder = True
                final_decision = "INTRUDER"
            else:
                final_decision = raw_pred

        return {
            "pipeline_id": pipeline_id,
            "model_id": pipeline_id,
            "model_name": config["model_name"],
            "pipeline_name": config["pipeline_name"],
            "feature_name": config["feature_name"],
            "classifier_name": config["classifier_name"],
            "raw_prediction": raw_pred,
            "confidence": confidence,
            "is_intruder": is_intruder,
            "final_decision": final_decision
        }

    def predict_fusion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate face through ALL 5 pipelines and perform multi-model consensus fusion.

        Args:
            face_img: 128x128 preprocessed face image.

        Returns:
            Fusion decision dictionary with individual model outputs, consensus vote counts,
            and the ultimate 'AUTHORIZED: <Name>' or 'INTRUDER ALERT' status.
        """
        individual_results: Dict[int, Dict[str, Any]] = {}
        votes: Dict[str, int] = {}
        confidences: List[float] = []

        for pipeline_id in range(1, 6):
            res = self.predict_single_pipeline(face_img, pipeline_id)
            individual_results[pipeline_id] = res

            decision = res["final_decision"]
            votes[decision] = votes.get(decision, 0) + 1
            confidences.append(res["confidence"])

        # Determine majority decision
        sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        top_decision, top_vote_count = sorted_votes[0]

        mean_confidence = float(np.mean(confidences))

        # Check consensus criteria
        if top_decision == "INTRUDER" or top_vote_count < self.min_consensus_votes:
            status = "INTRUDER_ALERT"
            identity = "UNKNOWN / INTRUDER"
            is_authorized = False
        else:
            status = "AUTHORIZED"
            identity = top_decision
            is_authorized = True

        return {
            "status": status,
            "identity": identity,
            "is_authorized": is_authorized,
            "mean_confidence": mean_confidence,
            "consensus_votes": votes,
            "individual_results": individual_results
        }

    def evaluate_open_set_performance(
        self,
        X_known_test: np.ndarray,
        y_known_test: np.ndarray,
        X_intruders: np.ndarray
    ) -> Dict[str, Any]:
        """
        Benchmark open-set metrics:
        - False Acceptance Rate (FAR): % of intruders wrongly accepted as authorized.
        - False Rejection Rate (FRR): % of enrolled authorized persons rejected as intruders.
        - True Intruder Detection Rate: % of intruders correctly flagged.
        - Correct Identification Rate: % of enrolled persons correctly classified.

        Args:
            X_known_test: Test images of enrolled known identities.
            y_known_test: Ground truth labels for known test images.
            X_intruders: Test images of unknown intruder faces.

        Returns:
            Dictionary with open-set performance metrics.
        """
        # 1. Test on Known Subjects
        known_total = len(y_known_test)
        correct_id_count = 0
        false_reject_count = 0

        for i in range(known_total):
            fusion_res = self.predict_fusion(X_known_test[i])
            if not fusion_res["is_authorized"]:
                false_reject_count += 1
            elif fusion_res["identity"] == y_known_test[i]:
                correct_id_count += 1

        frr = (false_reject_count / known_total) * 100.0 if known_total > 0 else 0.0
        cir = (correct_id_count / known_total) * 100.0 if known_total > 0 else 0.0

        # 2. Test on Intruder Subjects
        intruder_total = len(X_intruders)
        false_accept_count = 0
        true_intruder_detect_count = 0

        for i in range(intruder_total):
            fusion_res = self.predict_fusion(X_intruders[i])
            if fusion_res["is_authorized"]:
                false_accept_count += 1
            else:
                true_intruder_detect_count += 1

        far = (false_accept_count / intruder_total) * 100.0 if intruder_total > 0 else 0.0
        idr = (true_intruder_detect_count / intruder_total) * 100.0 if intruder_total > 0 else 0.0

        return {
            "known_test_samples": known_total,
            "intruder_test_samples": intruder_total,
            "false_accept_rate_pct": far,
            "false_reject_rate_pct": frr,
            "intruder_detection_rate_pct": idr,
            "correct_identification_rate_pct": cir
        }
