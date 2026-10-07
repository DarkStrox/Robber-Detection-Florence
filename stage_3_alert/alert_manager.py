import os
import time
import json
import cv2

class AlertManager:
    """
    Stage 3: Alert Manager for handling audio chimes, visual alerts, 
    logging JSON incident reports, and saving evidence frames.
    """
    def __init__(self, log_dir="incident_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_alert_time = 0.0
        self.cooldown_seconds = 5.0  # Avoid spamming alerts within 5 seconds

    def trigger_alert(self, frame_bgr, threat_score, reasons):
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown_seconds:
            return None  # In cooldown

        self.last_alert_time = current_time
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Save Evidence Image Frame
        image_name = f"threat_{timestamp_str}.jpg"
        image_path = os.path.join(self.log_dir, image_name)
        cv2.imwrite(image_path, frame_bgr)

        # 2. Save Incident Log JSON
        log_data = {
            "timestamp": timestamp_str,
            "threat_score": threat_score,
            "triggers": reasons,
            "evidence_image": image_path
        }
        json_path = os.path.join(self.log_dir, f"threat_{timestamp_str}.json")
        with open(json_path, "w") as f:
            json.dump(log_data, f, indent=4)

        print(f"\n🚨 [ALERT MANAGER] THREAT INCIDENT RECORDED!")
        print(f" -> Score: {threat_score*100:.0f}%")
        print(f" -> Triggers: {', '.join(reasons)}")
        print(f" -> Saved Evidence: {image_path}\n")

        return log_data
