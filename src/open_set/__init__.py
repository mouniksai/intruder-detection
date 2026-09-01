"""
Open-Set Intruder Recognition Package
-------------------------------------
Handles confidence thresholding, metric-space neighbor distance gating,
and multi-pipeline consensus fusion for flagging unauthorized intruders.
"""

from .intruder_detector import IntruderDetector

__all__ = ["IntruderDetector"]
