import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from ultralytics import YOLO

class PoseEngine:
    """
    Ultra-fast Pose Engine using YOLOv8-pose (~100+ FPS).
    Tracks 17 human skeletal keypoints:
    0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
    5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow
    9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip
    13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    """
    def __init__(self, model_name="yolov8n-pose.pt"):
        print("[PoseEngine] Loading YOLOv8-Pose model...")
        self.model = YOLO(model_name)
        
    def analyze_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        pose_data = []
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return pose_data
            
        keypoints_data = results.keypoints.data.cpu().numpy() # [N, 17, 3]
        boxes_data = results.boxes.xyxy.cpu().numpy()
        
        for i in range(len(keypoints_data)):
            kp = keypoints_data[i]
            box = boxes_data[i]
            
            # Check key gestures
            hands_up = self._check_hands_up(kp, box)
            waistband_reach = self._check_waistband_reach(kp, box)
            arm_aiming = self._check_arm_aiming(kp, box)
            
            pose_data.append({
                "bbox": box.tolist(),
                "keypoints": kp,
                "hands_up": hands_up,
                "waistband_reach": waistband_reach,
                "arm_aiming": arm_aiming
            })
            
        return pose_data

    def _check_hands_up(self, kp, box):
        """Returns True if either wrist is clearly above shoulder or nose height."""
        # 0: Nose, 5: L-Shoulder, 6: R-Shoulder, 9: L-Wrist, 10: R-Wrist
        l_shoulder_y, r_shoulder_y = kp[5][1], kp[6][1]
        l_wrist_y, r_wrist_y = kp[9][1], kp[10][1]
        
        l_conf, r_conf = kp[9][2], kp[10][2]
        box_h = max(1.0, box[3] - box[1])
        margin = max(15.0, box_h * 0.05)
        
        # In image coordinates, smaller Y means HIGHER up on screen
        l_hands_up = (l_conf > 0.25) and (l_wrist_y < l_shoulder_y - margin)
        r_hands_up = (r_conf > 0.25) and (r_wrist_y < r_shoulder_y - margin)
        
        return l_hands_up or r_hands_up

    def _check_waistband_reach(self, kp, box):
        """Returns True if wrist keypoint is close to hip/waistband level or reaching down."""
        # 11: L-Hip, 12: R-Hip, 9: L-Wrist, 10: R-Wrist
        l_hip, r_hip = kp[11], kp[12]
        l_wrist, r_wrist = kp[9], kp[10]
        
        box_h = max(1.0, box[3] - box[1])
        max_reach_dist = max(70.0, box_h * 0.25) # Scale with person height
        
        if l_wrist[2] > 0.25 and l_hip[2] > 0.25:
            dist_left = np.linalg.norm(l_wrist[:2] - l_hip[:2])
            if dist_left < max_reach_dist:
                return True
                
        if r_wrist[2] > 0.25 and r_hip[2] > 0.25:
            dist_right = np.linalg.norm(r_wrist[:2] - r_hip[:2])
            if dist_right < max_reach_dist:
                return True
                
        return False

    def _check_arm_aiming(self, kp, box):
        """Returns True if shoulder, elbow, and wrist align horizontally (pointing/aiming)."""
        # 5: L-Shoulder, 7: L-Elbow, 9: L-Wrist
        box_w = max(1.0, box[2] - box[0])
        box_h = max(1.0, box[3] - box[1])
        
        for shoulder_idx, elbow_idx, wrist_idx in [(5, 7, 9), (6, 8, 10)]:
            s, e, w = kp[shoulder_idx], kp[elbow_idx], kp[wrist_idx]
            if s[2] > 0.25 and e[2] > 0.25 and w[2] > 0.25:
                y_diff = abs(s[1] - w[1])
                x_diff = abs(s[0] - w[0])
                if x_diff > box_w * 0.35 and y_diff < box_h * 0.20:
                    return True
        return False
