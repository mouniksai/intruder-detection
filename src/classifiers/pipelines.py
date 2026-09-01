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

from typing import Dict, Any, Tuple
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


def get_pipeline_1(random_state: int = 42) -> Pipeline:
    """
    Pipeline 1: BSIF + Random Forest Classifier.
    Random Forest leverages an ensemble of bagging decision trees to effectively classify
    high-dimensional statistical histogram representations.
    """
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


def get_pipeline_2(n_neighbors: int = 3) -> Pipeline:
    """
    Pipeline 2: LPQ + Distance-Weighted k-Nearest Neighbors (k-NN).
    k-NN finds closest matching feature vectors in LPQ phase-histogram space,
    using cosine distance metric and inverse-distance weighted voting.
    """
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


def get_pipeline_3(random_state: int = 42) -> Pipeline:
    """
    Pipeline 3: WLD + Multinomial Logistic Regression.
    Logistic Regression optimizes log-odds boundaries across relative-contrast histograms
    with L2 ridge regularization.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ))
    ])


def get_pipeline_4(random_state: int = 42) -> Pipeline:
    """
    Pipeline 4: Gabor Wavelet Bank + RBF-SVM.
    Radial Basis Function Support Vector Machine projects Gabor orientation/scale
    energy statistics into an infinite-dimensional RKHS with maximum margin separation.
    """
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


def get_pipeline_5(random_state: int = 42) -> Pipeline:
    """
    Pipeline 5: Facial Landmark Geometry + Extremely Randomized Decision Trees.
    Decision Tree ensemble discovers optimal orthogonal and oblique splits across
    multi-scale cranial contour geometry, pairwise distance ratios, and bilateral symmetry indices.
    """
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


# Aliases for backward compatibility
get_member_1_pipeline = get_pipeline_1
get_member_2_pipeline = get_pipeline_2
get_member_3_pipeline = get_pipeline_3
get_member_4_pipeline = get_pipeline_4
get_member_5_pipeline = get_pipeline_5


# Registry mapping pipeline IDs to extractor functions and model pipelines
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
        "classifier_name": "k-NN",
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
        "feature_name": "Gabor Wavelets",
        "classifier_name": "RBF-SVM",
        "extractor_func": extract_features_gabor,
        "pipeline_factory": get_pipeline_4,
        "description": "40-Filter Gabor Wavelet Bank + RBF Kernel SVM"
    },
    5: {
        "pipeline_id": 5,
        "model_name": "Geometry + Decision Tree",
        "pipeline_name": "Pipeline 5 (Geometry + Decision Tree)",
        "member_name": "Geometry + Decision Tree",
        "feature_name": "Landmark Geometry",
        "classifier_name": "Decision Tree",
        "extractor_func": extract_features_geometry,
        "pipeline_factory": get_pipeline_5,
        "description": "32-D Cranial Landmark Geometry + CART Decision Tree"
    }
}
