import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Stage 1: Real-time Motion, Pose & Object Scanner Package
from .pose_engine import PoseEngine
from .florence_scanner import FlorenceScanner
from .zone_manager import ZoneManager
