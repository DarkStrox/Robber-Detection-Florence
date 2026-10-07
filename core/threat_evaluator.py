import time
from collections import deque

class ThreatEvaluator:
    """
    Evaluates dynamic multi-factor threat scores and maintains a 3-frame persistence buffer.
    Threat Score Formula:
    Score = S_weapon (+0.85) + S_mask (+0.70 inside zone, +0.30 outside) + S_pose (+0.40) + S_zone (+0.30)
    """
    def __init__(self, buffer_size=3, threat_threshold=0.70):
        self.buffer_size = buffer_size
        self.threat_threshold = threat_threshold
        self.history = deque(maxlen=buffer_size)

    def evaluate(self, stage1_detections, pose_data, zone_engine):
        score = 0.0
        reasons = []

        labels = [d["label"] for d in stage1_detections]

        # 1. Weapon Detection
        has_weapon = any(w in labels for w in ["gun", "handgun", "rifle", "knife", "blade", "weapon"])
        if has_weapon:
            score += 0.85
            reasons.append("WEAPON DETECTED")

        # 2. Mask Detection
        has_mask = any(m in labels for m in ["ski mask", "balaclava", "mask", "face cover"])
        if has_mask:
            # Check if any masked person is in restricted zone
            masked_in_zone = False
            for det in stage1_detections:
                if det["label"] in ["ski mask", "balaclava", "mask", "person"]:
                    if zone_engine.is_inside_zone(det["bbox"], "cash_counter"):
                        masked_in_zone = True
                        break

            if masked_in_zone:
                score += 0.70
                reasons.append("MASKED IN COUNTER ZONE")
            else:
                score += 0.30
                reasons.append("MASK DETECTED")

        # 3. Pose Gestures (Hands-Up, Waistband Reach, Aiming)
        for pose in pose_data:
            if pose["hands_up"]:
                score += 0.40
                reasons.append("HANDS-UP SURRENDER POSE")
            if pose["waistband_reach"]:
                score += 0.35
                reasons.append("REACHING WAISTBAND/POCKET")
            if pose["arm_aiming"]:
                score += 0.50
                reasons.append("WEAPON AIMING GESTURE")

        # Normalize score max to 1.0
        score = min(1.0, score)
        self.history.append(score)

        # Persistence Buffer: Require average of last N frames to exceed threshold
        avg_score = sum(self.history) / len(self.history) if self.history else 0.0
        is_triggered = avg_score >= self.threat_threshold

        return {
            "threat_score": score,
            "buffered_score": avg_score,
            "is_triggered": is_triggered,
            "reasons": list(set(reasons))
        }
