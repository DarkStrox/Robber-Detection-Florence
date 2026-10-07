import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
import cv2
import torch
import numpy as np

# Clean Modular Imports from Stage Folders
from stage_1_scanner import PoseEngine, FlorenceScanner, ZoneManager
from stage_2_verifier import LocateAnythingVerifier
from stage_3_alert import ThreatEvaluator, AlertManager
from video_batch_processor import VideoBatchProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class RobberDetectorApp:
    def __init__(self):
        print("==================================================")
        print("   REAL-TIME ROBBER & THREAT DETECTION SYSTEM")
        print("==================================================")
        
        print("\n--- Initializing Stage 1 Scanner ---")
        self.pose_engine = PoseEngine()
        self.florence_scanner = FlorenceScanner()
        self.zone_manager = ZoneManager(os.path.join(CURRENT_DIR, "zones.json"))
        
        print("\n--- Initializing Stage 2 Verifier ---")
        self.locate_verifier = LocateAnythingVerifier()
        
        print("\n--- Initializing Stage 3 Alert System ---")
        self.threat_evaluator = ThreatEvaluator(buffer_size=3, threat_threshold=0.70)
        self.alert_manager = AlertManager()
        
        print("\nAll Stages Successfully Initialized!\n")

    def run_image_test(self, image_path):
        """Runs complete multi-stage threat evaluation on a single image and prints status report."""
        if not os.path.exists(image_path):
            print(f"Test image not found at {image_path}")
            return

        print(f"==================================================")
        print(f"  RUNNING THREAT VERIFICATION ON TEST IMAGE")
        print(f"==================================================")
        print(f"Target Image: {os.path.basename(image_path)}")
        
        img_bgr = cv2.imread(image_path)
        
        # 1. Pose Engine
        pose_start = time.time()
        pose_data = self.pose_engine.analyze_frame(img_bgr)
        pose_ms = (time.time() - pose_start) * 1000

        # 2. Stage 1 Florence Scanner
        stage1_detections, s1_ms = self.florence_scanner.scan_frame(
            img_bgr, 
            target_prompts="person, gun, knife, ski mask, balaclava, bag, counter"
        )

        # 3. Stage 3 Threat Evaluator
        eval_result = self.threat_evaluator.evaluate(stage1_detections, pose_data, self.zone_manager)
        threat_score = eval_result["buffered_score"]
        is_triggered = eval_result["is_triggered"]
        reasons = eval_result["reasons"]

        # 4. Stage 2 Verifier (If Threat Threshold Triggered)
        s2_confirmed = False
        s2_ms = 0.0
        if is_triggered:
            for det in stage1_detections:
                if det["label"] in ["gun", "knife", "ski mask", "balaclava", "person"]:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    h, w, _ = img_bgr.shape
                    crop_x1, crop_y1 = max(0, x1 - 20), max(0, y1 - 20)
                    crop_x2, crop_y2 = min(w, x2 + 20), min(h, y2 + 20)
                    crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if self.locate_verifier.is_server_online():
                        confirmations, s2_ms = self.locate_verifier.verify_crop(crop, "gun, knife, weapon, mask")
                        if len(confirmations) > 0:
                            s2_confirmed = True
                            break

        # Display Detailed Report
        print("\n--- PERFORMANCE & INFERENCE SPEED ---")
        print(f" • Stage 1 Florence Scanner Speed : {s1_ms:.2f} ms")
        print(f" • Stage 1 YOLO-Pose Engine Speed  : {pose_ms:.2f} ms")
        if s2_ms > 0:
            print(f" • Stage 2 NVIDIA Locate Speed    : {s2_ms:.2f} ms")

        print("\n--- DETECTED OBJECTS & GESTURES ---")
        detected_labels = [d['label'] for d in stage1_detections]
        print(f" • Objects Detected : {detected_labels if detected_labels else 'None'}")
        
        hands_up_count = sum(1 for p in pose_data if p['hands_up'])
        reach_count = sum(1 for p in pose_data if p['waistband_reach'])
        print(f" • Surrender Poses  : {hands_up_count}")
        print(f" • Waistband Reach  : {reach_count}")

        print("\n--------------------------------------------------")
        print(f" THREAT SCORE: {threat_score * 100:.0f}% ({threat_score:.2f} / 1.00)")
        
        if is_triggered or s2_confirmed:
            print(" STATUS      : 🚨 THREAT ALARM TRIGGERED!")
            print(f" TRIGGERS    : {' | '.join(reasons)}")
            self.alert_manager.trigger_alert(img_bgr, threat_score, reasons)
        else:
            print(" STATUS      : ✅ SYSTEM SAFE (No Active Threat Detected)")
        print("==================================================\n")

    def run_video(self, video_path=None, save_output=True):
        input_dir = os.path.join(CURRENT_DIR, "input_videos")
        output_dir = os.path.join(CURRENT_DIR, "output_videos")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if video_path is None:
            video_path = os.path.join(input_dir, "combined_output.mp4")

        processor = VideoBatchProcessor(florence_scanner=self.florence_scanner, pose_engine=self.pose_engine)
        out_path = os.path.join(output_dir, "combined_output_threat_annotated.mp4")
        return processor.process_video(input_video_path=video_path, output_video_path=out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Robbery & Threat Detection AI")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Gradio Web Dashboard")
    parser.add_argument("--video", type=str, default=None, help="Process a video file directly")
    args = parser.parse_args()

    if args.ui:
        from app_ui import build_gradio_app
        print("\n🚀 Launching Gradio Threat Detection Dashboard at http://127.0.0.1:7860...")
        demo = build_gradio_app()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    elif args.video:
        app = RobberDetectorApp()
        app.run_video(args.video)
    else:
        app = RobberDetectorApp()
        combined_path = os.path.join(CURRENT_DIR, "input_videos", "combined_output.mp4")
        if os.path.exists(combined_path):
            print(f"\nProcessing default video: {combined_path}")
            app.run_video(combined_path)
        else:
            sample_img = os.path.join(CURRENT_DIR, "test_security_scene.png")
            app.run_image_test(sample_img)
