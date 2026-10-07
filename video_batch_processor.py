import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from stage_1_scanner import PoseEngine, FlorenceScanner, ZoneManager
from stage_2_verifier import LocateAnythingVerifier
from stage_3_alert import ThreatEvaluator, AlertManager

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEOS_DIR = os.path.join(CURRENT_DIR, "input_videos")
OUTPUT_VIDEOS_DIR = os.path.join(CURRENT_DIR, "output_videos")
os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

DEFAULT_INPUT_VIDEO = os.path.join(INPUT_VIDEOS_DIR, "combined_output.mp4")
DEFAULT_OUTPUT_VIDEO = os.path.join(OUTPUT_VIDEOS_DIR, "combined_output_threat_annotated.mp4")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)

class VideoBatchProcessor:
    def __init__(self, florence_scanner=None, pose_engine=None):
        self.pose_engine = pose_engine if pose_engine else PoseEngine()
        self.florence_scanner = florence_scanner if florence_scanner else FlorenceScanner()
        self.zone_manager = ZoneManager(os.path.join(CURRENT_DIR, "zones.json"))
        self.locate_verifier = LocateAnythingVerifier()
        self.threat_evaluator = ThreatEvaluator(buffer_size=3, threat_threshold=0.70)
        self.alert_manager = AlertManager()

    def process_video(self, input_video_path=DEFAULT_INPUT_VIDEO, output_video_path=DEFAULT_OUTPUT_VIDEO, progress_callback=None, scan_interval=3):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video file not found at: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure output filename ends in .mp4
        if output_video_path.endswith(".avi"):
            output_video_path = output_video_path.rsplit(".", 1)[0] + ".mp4"

        # Setup Video Writer with web-compatible H.264 / avc1 codec
        writer = None
        for codec_str in ['avc1', 'H264', 'mp4v']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                w = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if w.isOpened():
                    writer = w
                    print(f"[VideoBatchProcessor] Initialized VideoWriter with codec: '{codec_str}'")
                    break
            except Exception:
                pass
                
        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print(f"=== Batch Processing Started: {input_video_path} ===")
        print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

        frame_index = 0
        max_threat_score = 0.0
        alarm_frame_count = 0
        start_proc_time = time.time()

        analytics_records = []
        incident_events = []
        cached_detections = []

        is_s2_online = self.locate_verifier.is_server_online()
        print(f"Stage 2 NVIDIA Verifier Online Status: {is_s2_online}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            timestamp_sec = round(frame_index / fps, 2)
            loop_start = time.time()

            try:
                # 1. Pose Engine analysis on every frame
                pose_data = self.pose_engine.analyze_frame(frame)

                # 2. Florence Scanner on every Nth frame to maximize FPS
                s1_ms = 0.0
                if frame_index == 1 or frame_index % scan_interval == 0:
                    cached_detections, s1_ms = self.florence_scanner.scan_frame(
                        frame,
                        target_prompts="person. gun. handgun. firearm. knife. weapon. ski mask. balaclava. full face mask. face mask. bag. counter. cash register."
                    )

                # 3. Stage 3 Threat Evaluation
                eval_result = self.threat_evaluator.evaluate(cached_detections, pose_data, self.zone_manager)
                raw_threat_score = eval_result["threat_score"]
                buffered_score = eval_result["buffered_score"]
                is_triggered = eval_result["is_triggered"]
                reasons = eval_result["reasons"]

                if buffered_score > max_threat_score:
                    max_threat_score = buffered_score

                if is_triggered:
                    alarm_frame_count += 1

                # 4. Stage 2 Verifier trigger on high threat score
                s2_confirmed = False
                s2_latency_ms = 0.0
                if is_triggered and is_s2_online:
                    for det in cached_detections:
                        if det["label"] in ["gun", "knife", "weapon", "ski mask", "balaclava", "full face mask", "person"]:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            crop_x1, crop_y1 = max(0, x1 - 15), max(0, y1 - 15)
                            crop_x2, crop_y2 = min(width, x2 + 15), min(height, y2 + 15)
                            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                confirmations, s2_latency_ms = self.locate_verifier.verify_crop(crop, "gun, knife, weapon, mask")
                                if len(confirmations) > 0:
                                    s2_confirmed = True
                                    reasons.append(f"STAGE 2 CONFIRMED: {', '.join(confirmations)}")
                                    break

                # If threat triggered, record incident event
                if is_triggered:
                    self.alert_manager.trigger_alert(frame, buffered_score, reasons)
                    if len(incident_events) == 0 or (timestamp_sec - incident_events[-1]["timestamp_sec"] > 1.0):
                        incident_events.append({
                            "frame_index": int(frame_index),
                            "timestamp_sec": float(timestamp_sec),
                            "threat_score": float(round(buffered_score, 3)),
                            "reasons": [str(r) for r in set(reasons)],
                            "stage2_confirmed": bool(s2_confirmed)
                        })

                loop_ms = (time.time() - loop_start) * 1000
                current_fps = 1000.0 / loop_ms if loop_ms > 0 else fps

                # 5. Draw crisp resolution-adaptive visual HUD overlay
                annotated_frame = self.draw_overlay(frame, cached_detections, pose_data, buffered_score, is_triggered, s2_confirmed, reasons, current_fps)

                # Write annotated frame
                writer.write(annotated_frame)

                # Collect frame metric analytics
                analytics_records.append({
                    "frame": int(frame_index),
                    "timestamp_sec": float(timestamp_sec),
                    "raw_threat_score": float(round(raw_threat_score, 3)),
                    "buffered_threat_score": float(round(buffered_score, 3)),
                    "is_triggered": bool(is_triggered),
                    "stage2_confirmed": bool(s2_confirmed),
                    "reasons": [str(r) for r in set(reasons)],
                    "detected_objects": [str(d["label"]) for d in cached_detections],
                    "hands_up_count": int(sum(1 for p in pose_data if p["hands_up"])),
                    "waistband_reach_count": int(sum(1 for p in pose_data if p["waistband_reach"]))
                })
            except Exception as e:
                print(f"Error on frame {frame_index}: {e}")

            # Progress update
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{total_frames} frames...")
            if progress_callback and frame_index % 10 == 0:
                progress_callback(frame_index / total_frames, f"Frame {frame_index}/{total_frames} ({int((frame_index/total_frames)*100)}%)")

        cap.release()
        writer.release()
        total_time_sec = time.time() - start_proc_time
        avg_fps = frame_index / total_time_sec if total_time_sec > 0 else 0.0

        # Save Analytics JSON
        analytics_file = os.path.join(OUTPUT_VIDEOS_DIR, "combined_output_analytics.json")
        summary_data = {
            "input_video": str(input_video_path),
            "output_video": str(output_video_path),
            "total_frames": int(frame_index),
            "duration_sec": float(round(frame_index / fps, 2)),
            "max_threat_score": float(round(max_threat_score, 3)),
            "alarm_frame_count": int(alarm_frame_count),
            "alarm_ratio_percent": float(round((alarm_frame_count / max(1, frame_index)) * 100, 2)),
            "avg_fps": float(round(avg_fps, 2)),
            "incident_count": int(len(incident_events)),
            "incidents": incident_events,
            "timeline": analytics_records
        }

        try:
            with open(analytics_file, "w") as f:
                json.dump(summary_data, f, cls=NumpyEncoder, indent=4)
            print(f"Analytics JSON saved successfully: {analytics_file}")
        except Exception as e:
            print(f"Error saving analytics JSON: {e}")

        print(f"=== Batch Processing Finished ===")
        print(f"Output Video: {output_video_path}")
        print(f"Max Threat Score: {max_threat_score * 100:.1f}% | Total Incidents: {len(incident_events)}")

        return summary_data

    def draw_overlay(self, frame, detections, pose_data, threat_score, is_triggered, s2_confirmed, reasons, fps):
        """Renders crisp resolution-adaptive HUD overlay using PIL."""
        h, w, _ = frame.shape
        
        # 1. Draw Zones via OpenCV
        self.zone_manager.draw_zones(frame)

        # Convert to PIL for crisp text rendering
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Adaptive font sizing for low (320x240) and high (720p/1080p) resolutions
        font_size_hud = max(11, int(h * 0.05))
        font_size_label = max(9, int(h * 0.038))

        try:
            font_hud = ImageFont.truetype("arial.ttf", font_size_hud)
            font_label = ImageFont.truetype("arial.ttf", font_size_label)
        except Exception:
            font_hud = ImageFont.load_default()
            font_label = ImageFont.load_default()

        box_width = 1 if w <= 400 else 2

        # 2. Draw Poses
        for pose in pose_data:
            x1, y1, x2, y2 = map(int, pose["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            box_color = (34, 197, 94) # Green
            pose_label = None

            if pose["hands_up"]:
                box_color = (249, 115, 22) # Orange
                pose_label = "SURRENDER POSE"
            elif pose["waistband_reach"]:
                box_color = (234, 179, 8) # Yellow
                pose_label = "WAISTBAND REACH"
            elif pose["arm_aiming"]:
                box_color = (239, 68, 68) # Red
                pose_label = "ARM AIMING"

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            if pose_label:
                tb = draw.textbbox((x1, max(0, y1 - font_size_label - 3)), pose_label, font=font_label)
                badge_bg_right = min(w, tb[2] + 4)
                badge_bg_bottom = min(h, max(0, y1))
                badge_bg_top = max(0, y1 - font_size_label - 4)
                draw.rectangle([x1, badge_bg_top, badge_bg_right, badge_bg_bottom], fill=box_color)
                draw.text((x1 + 2, badge_bg_top + 1), pose_label, fill=(0, 0, 0), font=font_label)

        # 3. Draw Detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            label = det["label"].upper()

            det_color = (59, 130, 246) # Blue
            if label in ["GUN", "KNIFE", "WEAPON", "SKI MASK", "BALACLAVA", "FULL FACE MASK"]:
                det_color = (239, 68, 68) # Red

            draw.rectangle([x1, y1, x2, y2], outline=det_color, width=box_width)
            
            tb = draw.textbbox((x1, max(0, y1 - font_size_label - 3)), label, font=font_label)
            badge_bg_right = min(w, tb[2] + 4)
            badge_bg_bottom = min(h, max(0, y1))
            badge_bg_top = max(0, y1 - font_size_label - 4)
            draw.rectangle([x1, badge_bg_top, badge_bg_right, badge_bg_bottom], fill=det_color)
            draw.text((x1 + 2, badge_bg_top + 1), label, fill=(255, 255, 255), font=font_label)

        # 4. Top Banner HUD
        hud_bar_height = max(22, int(h * 0.10))
        draw.rectangle([0, 0, w, hud_bar_height], fill=(15, 17, 23))

        status_str = f"STATUS: SAFE ({threat_score*100:.0f}%)"
        text_color = (34, 197, 94)

        if threat_score >= 0.5:
            status_str = f"WARNING: Threat ({threat_score*100:.0f}%)"
            text_color = (249, 115, 22)

        if is_triggered or s2_confirmed:
            status_str = f"[ALARM] THREAT DETECTED! ({threat_score*100:.0f}%)"
            text_color = (239, 68, 68)

        draw.text((6, 3), status_str, fill=text_color, font=font_hud)

        fps_str = f"FPS: {fps:.1f}"
        tb_fps = draw.textbbox((0, 0), fps_str, font=font_hud)
        fps_width = tb_fps[2] - tb_fps[0]
        draw.text((max(10, w - fps_width - 8), 3), fps_str, fill=(156, 163, 175), font=font_hud)

        # 5. Bottom Alert Reason Bar
        if reasons:
            bottom_bar_h = max(18, int(h * 0.08))
            draw.rectangle([0, h - bottom_bar_h, w, h], fill=(15, 17, 23))
            reasons_str = f"Alerts: {' | '.join(set(reasons))}"
            reason_color = (239, 68, 68) if is_triggered else (234, 179, 8)
            draw.text((6, h - bottom_bar_h + 2), reasons_str, fill=reason_color, font=font_label)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    processor = VideoBatchProcessor()
    processor.process_video("combined_output.mp4", "combined_output_threat_annotated.mp4")
