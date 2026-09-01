"""
pipelines.py
============
Factory definitions for the 5 independent Classical ML Pipelines.

Strict Classical ML Architectures:
----------------------------------
- Pipeline 1: BSIF               -> StandardScaler + RandomForestClassifier (Ensemble of Decision Trees)
- Pipeline 2: LPQ                -> StandardScaler + KNeighborsClassifier (Distance-Weighted Metric Space)
- Pipeline 3: WLD                -> StandardScaler + LogisticRegression (Multinomial Linear Classifier)
- Pipeline 4: Gabor Wavelets     -> StandardScaler + SVC (RBF Nonlinear Kernel Support Vector Machine)
- Pipeline 5: Landmark Geometry  -> StandardScaler + DecisionTreeClassifier (CART Rule-Based Splitting)
"""

from typing import Dict, Any, Tuple, Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..features.bsif import extract_features_bsif
from ..features.lpq import extract_features_lpq
from ..features.wld import extract_features_wld
from ..features.gabor import extract_features_gabor
from ..features.geometry import extract_features_geometry


# =============================================================================
# 5 STANDALONE CLASSIFIER FACTORIES
# =============================================================================

def get_classifier_rf(random_state: int = 42) -> Pipeline:
    """Classifier 1: Random Forest Classifier (Tree Ensemble)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def get_classifier_knn(n_neighbors: int = 3) -> Pipeline:
    """Classifier 2: Distance-Weighted k-Nearest Neighbors (k-NN)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='cosine',
            algorithm='auto',
            n_jobs=-1
        ))
    ])


def get_classifier_lr(random_state: int = 42) -> Pipeline:
    """Classifier 3: Multinomial Logistic Regression (L2 Regularized)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ))
    ])


def get_classifier_svm(random_state: int = 42) -> Pipeline:
    """Classifier 4: RBF Kernel Support Vector Machine (RBF-SVM)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=random_state
        ))
    ])


def get_classifier_dt(random_state: int = 42) -> Pipeline:
    """Classifier 5: Decision Tree Classifier / ExtraTrees."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ExtraTreesClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        ))
    ])


# Aliases for 1-to-1 canonical pipelines
get_pipeline_1 = get_classifier_rf
get_pipeline_2 = get_classifier_knn
get_pipeline_3 = get_classifier_lr
get_pipeline_4 = get_classifier_svm
get_pipeline_5 = get_classifier_dt

get_member_1_pipeline = get_pipeline_1
get_member_2_pipeline = get_pipeline_2
get_member_3_pipeline = get_pipeline_3
get_member_4_pipeline = get_pipeline_4
get_member_5_pipeline = get_pipeline_5


# =============================================================================
# 1-TO-N REGISTRIES: 5 FEATURES x 5 CLASSIFIERS = 25 EXPERIMENTS
# =============================================================================

FEATURE_EXTRACTORS: Dict[str, Dict[str, Any]] = {
    "BSIF": {
        "id": 1,
        "name": "BSIF",
        "extractor_func": extract_features_bsif,
        "description": "Binarized Statistical Image Features"
    },
    "LPQ": {
        "id": 2,
        "name": "LPQ",
        "extractor_func": extract_features_lpq,
        "description": "Local Phase Quantization"
    },
    "WLD": {
        "id": 3,
        "name": "WLD",
        "extractor_func": extract_features_wld,
        "description": "Weber Local Descriptor"
    },
    "Gabor": {
        "id": 4,
        "name": "Gabor",
        "extractor_func": extract_features_gabor,
        "description": "40-Filter Gabor Wavelet Bank"
    },
    "Geometry": {
        "id": 5,
        "name": "Geometry",
        "extractor_func": extract_features_geometry,
        "description": "Facial Landmark & Contour Geometry"
    }
}

CLASSIFIER_FACTORIES: Dict[str, Dict[str, Any]] = {
    "Random Forest": {
        "id": 1,
        "name": "Random Forest",
        "short_name": "RF",
        "factory": get_classifier_rf,
        "description": "Random Forest (150 Trees)"
    },
    "KNN": {
        "id": 2,
        "name": "KNN",
        "short_name": "KNN",
        "factory": get_classifier_knn,
        "description": "Distance-Weighted k-Nearest Neighbors"
    },
    "Logistic Regression": {
        "id": 3,
        "name": "Logistic Regression",
        "short_name": "LR",
        "factory": get_classifier_lr,
        "description": "Multinomial Logistic Regression (L2)"
    },
    "SVM": {
        "id": 4,
        "name": "SVM",
        "short_name": "SVM",
        "factory": get_classifier_svm,
        "description": "RBF-Kernel Support Vector Machine"
    },
    "Decision Tree": {
        "id": 5,
        "name": "Decision Tree",
        "short_name": "DT",
        "factory": get_classifier_dt,
        "description": "Decision Tree Ensemble (ExtraTrees)"
    }
}


# Canonical 1-to-1 pipeline configs for backward compatibility
PIPELINE_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "pipeline_id": 1,
        "model_name": "BSIF + Random Forest",
        "pipeline_name": "Pipeline 1 (BSIF + Random Forest)",
        "member_name": "BSIF + Random Forest",
        "feature_name": "BSIF",
        "classifier_name": "Random Forest",
        "extractor_func": extract_features_bsif,
        "pipeline_factory": get_pipeline_1,
        "description": "Binarized Statistical Image Features + Ensemble Random Forest"
    },
    2: {
        "pipeline_id": 2,
        "model_name": "LPQ + k-NN",
        "pipeline_name": "Pipeline 2 (LPQ + k-NN)",
        "member_name": "LPQ + k-NN",
        "feature_name": "LPQ",
        "classifier_name": "KNN",
        "extractor_func": extract_features_lpq,
        "pipeline_factory": get_pipeline_2,
        "description": "Local Phase Quantization + Distance-Weighted Nearest Neighbors"
    },
    3: {
        "pipeline_id": 3,
        "model_name": "WLD + Logistic Regression",
        "pipeline_name": "Pipeline 3 (WLD + Logistic Regression)",
        "member_name": "WLD + Logistic Regression",
        "feature_name": "WLD",
        "classifier_name": "Logistic Regression",
        "extractor_func": extract_features_wld,
        "pipeline_factory": get_pipeline_3,
        "description": "Weber Local Descriptor + Multinomial Logistic Regression"
    },
    4: {
        "pipeline_id": 4,
        "model_name": "Gabor + RBF-SVM",
        "pipeline_name": "Pipeline 4 (Gabor + RBF-SVM)",
        "member_name": "Gabor + RBF-SVM",
        "feature_name": "Gabor",
        "classifier_name": "SVM",
        "extractor_func": extract_features_gabor,
        "pipeline_factory": get_pipeline_4,
        "description": "40-Filter Gabor Wavelet Bank + RBF Kernel SVM"
    },
    5: {
        "pipeline_id": 5,
        "model_name": "Geometry + Decision Tree",
        "pipeline_name": "Pipeline 5 (Geometry + Decision Tree)",
        "member_name": "Geometry + Decision Tree",
        "feature_name": "Geometry",
        "classifier_name": "Decision Tree",
        "extractor_func": extract_features_geometry,
        "pipeline_factory": get_pipeline_5,
        "description": "Facial Landmark & Contour Geometry + Decision Tree"
    }
}

