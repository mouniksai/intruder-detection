"""
Classifiers Package
-------------------
Contains pipelines and evaluation harnesses for the 5 independent classical CV classifiers.
"""

from .pipelines import (
    PIPELINE_CONFIGS,
    get_pipeline_1,
    get_pipeline_2,
    get_pipeline_3,
    get_pipeline_4,
    get_pipeline_5,
    get_member_1_pipeline,
    get_member_2_pipeline,
    get_member_3_pipeline,
    get_member_4_pipeline,
    get_member_5_pipeline
)
from .train_eval import Review1Evaluator

__all__ = [
    "PIPELINE_CONFIGS",
    "get_pipeline_1",
    "get_pipeline_2",
    "get_pipeline_3",
    "get_pipeline_4",
    "get_pipeline_5",
    "get_member_1_pipeline",
    "get_member_2_pipeline",
    "get_member_3_pipeline",
    "get_member_4_pipeline",
    "get_member_5_pipeline",
    "Review1Evaluator"
]

