from collections import deque

class ThreatEvaluator:
    """
    Stage 3: Multi-Factor Threat Evaluator with 3-Frame Smoothing Buffer.
    Realistic Threat Formula:
    Score = S_weapon (+0.85) + S_mask (+0.50 inside zone) + S_pose (+0.40) + S_zone (+0.20)
    Rules:
    - Weapons (+0.85) require pose gesture (waistband reach / arm aiming) or hand proximity.
    - Person in zone (+0.20) requires an actual 'person' inside cash counter polygon (NOT static furniture).
    - Mask in zone (+0.50) requires person inside cash counter polygon with full face cover.
    """
    def __init__(self, buffer_size=3, threat_threshold=0.70):
        self.buffer_size = buffer_size
        self.threat_threshold = threat_threshold
        self.history = deque(maxlen=buffer_size)

    def evaluate(self, stage1_detections, pose_data, zone_manager):
        score = 0.0
        reasons = []

        # 1. Pose Gestures Analysis
        hands_up = any(p["hands_up"] for p in pose_data)
        waistband_reach = any(p["waistband_reach"] for p in pose_data)
        arm_aiming = any(p["arm_aiming"] for p in pose_data)

        if hands_up:
            score += 0.45
            reasons.append("HANDS-UP SURRENDER POSE")
        if waistband_reach:
            score += 0.40
            reasons.append("REACHING WAISTBAND/POCKET")
        if arm_aiming:
            score += 0.55
            reasons.append("WEAPON AIMING GESTURE")

        # 2. Person in Counter Restricted Zone Check (Strictly 'person' label ONLY)
        person_in_zone = False
        for det in stage1_detections:
            if det["label"] == "person":
                if zone_manager.is_inside_zone(det["bbox"], "cash_counter"):
                    person_in_zone = True
                    break

        if person_in_zone:
            score += 0.20
            reasons.append("COUNTER RESTRICTED ZONE BREACH")

        # 3. Weapon Detection (Requires Threat Gesture or Hand Proximity)
        raw_labels = [d["label"].lower() for d in stage1_detections]
        has_weapon_label = any(w in raw_labels for w in ["gun", "handgun", "pistol", "knife", "blade"])

        if has_weapon_label and (waistband_reach or arm_aiming or person_in_zone):
            score += 0.85
            reasons.append("WEAPON DETECTED")

        # 4. Strict Full-Face Mask Detection (Ski mask / Balaclava inside zone)
        has_mask_label = any(m in raw_labels for m in ["ski mask", "balaclava", "full face mask"])
        if has_mask_label and person_in_zone:
            score += 0.50
            reasons.append("FULL FACE MASK IN COUNTER ZONE")

        score = min(1.0, score)
        self.history.append(score)

        avg_score = sum(self.history) / len(self.history) if self.history else 0.0
        is_triggered = avg_score >= self.threat_threshold

        return {
            "threat_score": score,
            "buffered_score": avg_score,
            "is_triggered": is_triggered,
            "reasons": list(set(reasons))
        }
