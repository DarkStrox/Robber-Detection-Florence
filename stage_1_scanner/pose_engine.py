import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from ultralytics import YOLO

class PoseEngine:
    """
    Stage 1: Ultra-fast Pose Engine using YOLOv8-pose (~100+ FPS).
    Tracks 17 human skeletal keypoints:
    0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
    5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow
    9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip
    13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    """
    def __init__(self, model_name="yolov8n-pose.pt"):
        print("[Stage 1 PoseEngine] Loading YOLOv8-Pose model...")
        self.model = YOLO(model_name)
        
    def analyze_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        pose_data = []
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return pose_data
            
        keypoints_data = results.keypoints.data.cpu().numpy()
        boxes_data = results.boxes.xyxy.cpu().numpy()
        
        for i in range(len(keypoints_data)):
            kp = keypoints_data[i]
            box = boxes_data[i]
            
            hands_up = self._check_hands_up(kp)
            waistband_reach = self._check_waistband_reach(kp)
            arm_aiming = self._check_arm_aiming(kp)
            
            pose_data.append({
                "bbox": box.tolist(),
                "keypoints": kp,
                "hands_up": hands_up,
                "waistband_reach": waistband_reach,
                "arm_aiming": arm_aiming
            })
            
        return pose_data

    def _check_hands_up(self, kp):
        l_shoulder_y, r_shoulder_y = kp[5][1], kp[6][1]
        l_wrist_y, r_wrist_y = kp[9][1], kp[10][1]
        l_conf, r_conf = kp[9][2], kp[10][2]
        
        l_hands_up = (l_conf > 0.3) and (l_wrist_y < l_shoulder_y - 20)
        r_hands_up = (r_conf > 0.3) and (r_wrist_y < r_shoulder_y - 20)
        return l_hands_up or r_hands_up

    def _check_waistband_reach(self, kp):
        l_hip, r_hip = kp[11], kp[12]
        l_wrist, r_wrist = kp[9], kp[10]
        
        if l_wrist[2] > 0.3 and l_hip[2] > 0.3:
            if np.linalg.norm(l_wrist[:2] - l_hip[:2]) < 60:
                return True
                
        if r_wrist[2] > 0.3 and r_hip[2] > 0.3:
            if np.linalg.norm(r_wrist[:2] - r_hip[:2]) < 60:
                return True
                
        return False

    def _check_arm_aiming(self, kp):
        for shoulder_idx, elbow_idx, wrist_idx in [(5, 7, 9), (6, 8, 10)]:
            s, e, w = kp[shoulder_idx], kp[elbow_idx], kp[wrist_idx]
            if s[2] > 0.3 and e[2] > 0.3 and w[2] > 0.3:
                y_diff = abs(s[1] - w[1])
                x_diff = abs(s[0] - w[0])
                if x_diff > 80 and y_diff < 40:
                    return True
        return False
