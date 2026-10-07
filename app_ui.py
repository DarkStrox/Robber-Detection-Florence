import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from video_batch_processor import VideoBatchProcessor, CURRENT_DIR
from stage_1_scanner import PoseEngine, FlorenceScanner, ZoneManager
from stage_2_verifier import LocateAnythingVerifier
from stage_3_alert import ThreatEvaluator, AlertManager

# Global initialized instances (Lazy / Shared)
_FLORENCE_SCANNER = None
_POSE_ENGINE = None

_LIVE_THREAT_EVALUATOR = None
_LIVE_ZONE_MANAGER = None
_LIVE_FRAME_COUNTER = 0
_LIVE_CACHED_DETECTIONS = []

_STOP_REALTIME_STREAM = False

def get_florence_scanner():
    global _FLORENCE_SCANNER
    if _FLORENCE_SCANNER is None:
        print("[App UI] Lazy initializing FlorenceScanner...")
        _FLORENCE_SCANNER = FlorenceScanner()
    return _FLORENCE_SCANNER

def get_pose_engine():
    global _POSE_ENGINE
    if _POSE_ENGINE is None:
        print("[App UI] Lazy initializing PoseEngine...")
        _POSE_ENGINE = PoseEngine()
    return _POSE_ENGINE

def get_live_threat_evaluator():
    global _LIVE_THREAT_EVALUATOR
    if _LIVE_THREAT_EVALUATOR is None:
        _LIVE_THREAT_EVALUATOR = ThreatEvaluator(buffer_size=3, threat_threshold=0.70)
    return _LIVE_THREAT_EVALUATOR

def get_live_zone_manager():
    global _LIVE_ZONE_MANAGER
    if _LIVE_ZONE_MANAGER is None:
        _LIVE_ZONE_MANAGER = ZoneManager(os.path.join(CURRENT_DIR, "zones.json"))
    return _LIVE_ZONE_MANAGER

def check_server_status():
    verifier = LocateAnythingVerifier()
    is_online = verifier.is_server_online()
    if is_online:
        return "🟢 NVIDIA LocateAnything 3B Server: ONLINE (Port 8085)"
    else:
        return "🟠 NVIDIA LocateAnything 3B Server: STANDBY / OFFLINE (Port 8085)"

INPUT_VIDEOS_DIR = os.path.join(CURRENT_DIR, "input_videos")
OUTPUT_VIDEOS_DIR = os.path.join(CURRENT_DIR, "output_videos")
os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

def extract_video_path(video_input):
    """Safely extracts valid absolute file path from Gradio Video inputs (str, dict, FileData, or None)."""
    default_path = os.path.join(INPUT_VIDEOS_DIR, "combined_output.mp4")
    if video_input is None:
        return default_path

    raw_path = None
    if isinstance(video_input, str):
        raw_path = video_input
    elif isinstance(video_input, dict):
        raw_path = video_input.get("path") or video_input.get("name")
    elif hasattr(video_input, "name") and video_input.name:
        raw_path = video_input.name
    elif hasattr(video_input, "path") and video_input.path:
        raw_path = video_input.path

    if raw_path:
        candidate = raw_path if os.path.isabs(raw_path) else os.path.join(CURRENT_DIR, raw_path)
        if os.path.exists(candidate):
            return candidate

    return default_path

def stop_realtime_stream():
    """Triggers stop signal to halt live real-time video stream detection loop."""
    global _STOP_REALTIME_STREAM
    _STOP_REALTIME_STREAM = True
    print("[App UI] Stop real-time video stream requested by user.")
    return "### ⏹️ STREAM STOPPED BY USER"

def format_nv_status(florence_scanner):
    """Formats NVIDIA LocateAnything 3B verification status for UI telemetry."""
    is_online = florence_scanner.last_nv_online
    verified_items = florence_scanner.last_nv_verified_items
    nv_ms = florence_scanner.last_nv_latency_ms

    if is_online:
        if verified_items:
            return f"🟢 Verified Threat Objects: `{verified_items}` ({nv_ms:.1f} ms)"
        else:
            return f"🟢 `None` (NVIDIA 3B Crop Check Passed — No Threat Detected)"
    else:
        return f"🟠 `Standby / Offline`"

def stream_realtime_video_detection(video_file, scan_interval=3):
    """Streams real-time threat detection frame-by-frame live on a video file with user stop support."""
    global _STOP_REALTIME_STREAM
    _STOP_REALTIME_STREAM = False

    video_path = extract_video_path(video_file)
    print(f"[App UI] Starting real-time video stream detection on: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "### ❌ Error Opening Video File", 0, "Failed to open video source."
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    florence = get_florence_scanner()
    pose_eng = get_pose_engine()
    zone_mgr = get_live_zone_manager()
    threat_eval = ThreatEvaluator(buffer_size=3, threat_threshold=0.70)
    processor = VideoBatchProcessor(florence_scanner=florence, pose_engine=pose_eng)

    frame_idx = 0
    cached_dets = []
    
    while True:
        if _STOP_REALTIME_STREAM:
            print("[App UI] Halting real-time stream loop on user request.")
            yield annotated_rgb if 'annotated_rgb' in locals() else None, "### ⏹️ STREAM STOPPED BY USER", 0, f"Halted at frame {frame_idx}/{total_frames}."
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # 1. Pose Engine on every frame (Ultra-Fast ~100 FPS)
        pose_data = pose_eng.analyze_frame(frame)
        pose_ms = (time.time() - t0) * 1000

        # Check if threat pose gesture is active (waistband reach, arm aiming, or hands up)
        has_threat_gesture = any(p["waistband_reach"] or p["arm_aiming"] or p["hands_up"] for p in pose_data)

        # 2. Florence Scanner on interval OR instant override whenever a threat gesture is active!
        s1_ms = 0.0
        if frame_idx == 1 or has_threat_gesture or frame_idx % int(scan_interval) == 0 or len(cached_dets) == 0:
            t_s1 = time.time()
            cached_dets, s1_ms = florence.scan_frame(frame)

        # 3. Stage 3 Threat Evaluation
        eval_res = threat_eval.evaluate(cached_dets, pose_data, zone_mgr)
        threat_score = eval_res["buffered_score"]
        is_triggered = eval_res["is_triggered"]
        reasons = eval_res["reasons"]

        current_fps = 1000.0 / max(1.0, pose_ms + s1_ms)
        annotated_bgr = processor.draw_overlay(frame, cached_dets, pose_data, threat_score, is_triggered, False, reasons, current_fps)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        score_pct = round(threat_score * 100, 1)

        if is_triggered:
            status_md = f"### 🚨 THREAT ALARM TRIGGERED! ({score_pct}%)"
        elif threat_score >= 0.5:
            status_md = f"### ⚠️ WARNING: Elevated Threat ({score_pct}%)"
        else:
            status_md = f"### 🟢 STATUS: SAFE ({score_pct}%)"

        reasons_str = ", ".join(reasons) if reasons else "None"
        progress_pct = int((frame_idx / max(1, total_frames)) * 100)
        nv_status_str = format_nv_status(florence)
        
        details_md = f"""
        ### ⏱️ Video Stream Progress: {progress_pct}% (Frame {frame_idx}/{total_frames})
        - **NVIDIA LocateAnything 3B Verified**: {nv_status_str}
        - **Active Triggers**: `{reasons_str}`
        - **Objects Detected**: `{[d['label'] for d in cached_dets]}`
        - **Surrender Gestures**: `{sum(1 for p in pose_data if p['hands_up'])}`
        - **Waistband Reaches**: `{sum(1 for p in pose_data if p['waistband_reach'])}`
        - **YOLO-Pose Speed**: `{pose_ms:.1f} ms`
        - **Florence-2 Speed**: `{s1_ms:.1f} ms`
        """

        yield annotated_rgb, status_md, score_pct, details_md

    cap.release()

def process_live_stream_frame(image):
    """Processes incoming streaming frame from webcam or live IP camera."""
    global _LIVE_FRAME_COUNTER, _LIVE_CACHED_DETECTIONS
    if image is None:
        return None, "### ⚪ CAMERA OFFLINE", 0, "No camera input detected."

    # Convert RGB image (from Gradio webcam input) to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _LIVE_FRAME_COUNTER += 1

    t0 = time.time()
    pose_engine = get_pose_engine()
    florence_scanner = get_florence_scanner()
    threat_evaluator = get_live_threat_evaluator()
    zone_manager = get_live_zone_manager()

    # 1. Pose Engine analysis on every frame
    pose_data = pose_engine.analyze_frame(frame_bgr)
    pose_ms = (time.time() - t0) * 1000

    has_threat_gesture = any(p["waistband_reach"] or p["arm_aiming"] or p["hands_up"] for p in pose_data)

    # 2. Florence Scanner (Instant scan on threat gesture OR every 3 frames)
    s1_ms = 0.0
    if has_threat_gesture or _LIVE_FRAME_COUNTER % 3 == 0 or len(_LIVE_CACHED_DETECTIONS) == 0:
        _LIVE_CACHED_DETECTIONS, s1_ms = florence_scanner.scan_frame(frame_bgr)

    # 3. Stage 3 Threat Evaluation
    eval_result = threat_evaluator.evaluate(_LIVE_CACHED_DETECTIONS, pose_data, zone_manager)
    threat_score = eval_result["buffered_score"]
    is_triggered = eval_result["is_triggered"]
    reasons = eval_result["reasons"]

    # Annotate Frame with resolution-adaptive HUD
    processor = VideoBatchProcessor(florence_scanner=florence_scanner, pose_engine=pose_engine)
    annotated_bgr = processor.draw_overlay(frame_bgr, _LIVE_CACHED_DETECTIONS, pose_data, threat_score, is_triggered, False, reasons, 1000.0 / max(1.0, pose_ms + s1_ms))

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    score_pct = round(threat_score * 100, 1)

    if is_triggered:
        status_md = f"### 🚨 THREAT ALARM TRIGGERED! ({score_pct}%)"
    elif threat_score >= 0.5:
        status_md = f"### ⚠️ WARNING: Elevated Threat ({score_pct}%)"
    else:
        status_md = f"### 🟢 STATUS: SAFE ({score_pct}%)"

    reasons_str = ", ".join(reasons) if reasons else "None"
    nv_status_str = format_nv_status(florence_scanner)

    details_md = f"""
    - **NVIDIA LocateAnything 3B Verified**: {nv_status_str}
    - **Active Triggers**: `{reasons_str}`
    - **Objects Detected**: `{[d['label'] for d in _LIVE_CACHED_DETECTIONS]}`
    - **Surrender Gestures**: `{sum(1 for p in pose_data if p['hands_up'])}`
    - **Waistband Reaches**: `{sum(1 for p in pose_data if p['waistband_reach'])}`
    - **YOLO-Pose Speed**: `{pose_ms:.1f} ms`
    - **Florence-2 Speed**: `{s1_ms:.1f} ms`
    """

    return annotated_rgb, status_md, score_pct, details_md

def run_batch_test(video_file, scan_interval, progress=gr.Progress()):
    video_path = extract_video_path(video_file)
    print(f"[App UI] Processing video target: {video_path}")

    output_video_path = os.path.join(OUTPUT_VIDEOS_DIR, "combined_output_threat_annotated.mp4")
    
    florence = get_florence_scanner()
    pose = get_pose_engine()

    processor = VideoBatchProcessor(florence_scanner=florence, pose_engine=pose)

    def progress_cb(pct, msg):
        progress(pct, desc=msg)

    analytics = processor.process_video(
        input_video_path=video_path,
        output_video_path=output_video_path,
        progress_callback=progress_cb,
        scan_interval=int(scan_interval)
    )

    # Generate Matplotlib Threat Score Chart
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    timestamps = [r["timestamp_sec"] for r in analytics["timeline"]]
    buffered_scores = [r["buffered_threat_score"] * 100 for r in analytics["timeline"]]
    raw_scores = [r["raw_threat_score"] * 100 for r in analytics["timeline"]]

    ax.plot(timestamps, raw_scores, color="#718096", alpha=0.5, label="Raw Frame Threat %")
    ax.plot(timestamps, buffered_scores, color="#E53E3E", linewidth=2.5, label="Buffered Threat Score %")
    ax.axhline(y=70, color="#DD6B20", linestyle="--", linewidth=1.5, label="Alarm Threshold (70%)")

    ax.set_title(f"Real-Time Threat Score Timeline — {os.path.basename(video_path)}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Video Time (Seconds)", fontsize=10)
    ax.set_ylabel("Threat Score (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    chart_file = os.path.join(OUTPUT_VIDEOS_DIR, "threat_timeline_chart.png")
    fig.savefig(chart_file)
    plt.close(fig)

    # Format Incident Dataframe
    incidents = analytics.get("incidents", [])
    if incidents:
        df_incidents = pd.DataFrame(incidents)
        df_incidents["reasons"] = df_incidents["reasons"].apply(lambda r: ", ".join(r))
        df_incidents.columns = ["Frame #", "Time (s)", "Threat Score", "Trigger Reasons", "Stage 2 Confirmed"]
    else:
        df_incidents = pd.DataFrame(columns=["Frame #", "Time (s)", "Threat Score", "Trigger Reasons", "Stage 2 Confirmed"])

    nv_status_str = format_nv_status(florence)

    kpi_summary = f"""
    ### 📊 Security Analytics Report
    - **Video Source**: `{os.path.basename(video_path)}`
    - **NVIDIA LocateAnything 3B Status**: {nv_status_str}
    - **Total Duration**: {analytics['duration_sec']}s ({analytics['total_frames']} frames)
    - **Peak Threat Score**: **{analytics['max_threat_score'] * 100:.1f}%**
    - **Alarm Frames Triggered**: {analytics['alarm_frame_count']} ({analytics['alarm_ratio_percent']}%)
    - **Total Security Incidents**: {analytics['incident_count']}
    - **Avg Inference Speed**: {analytics['avg_fps']} FPS
    """

    return output_video_path, chart_file, kpi_summary, df_incidents

def inspect_single_frame(video_file, frame_number):
    video_path = extract_video_path(video_file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(max(1, int(frame_number)), max(1, total_frames))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None, "Failed to read frame."

    florence = get_florence_scanner()
    pose_eng = get_pose_engine()
    zone_mgr = ZoneManager(os.path.join(CURRENT_DIR, "zones.json"))
    threat_eval = ThreatEvaluator()

    # Inference
    t0 = time.time()
    pose_data = pose_eng.analyze_frame(frame)
    pose_ms = (time.time() - t0) * 1000

    detections, s1_ms = florence.scan_frame(frame)
    eval_res = threat_eval.evaluate(detections, pose_data, zone_mgr)
    
    threat_score = eval_res["buffered_score"]
    is_triggered = eval_res["is_triggered"]
    reasons = eval_res["reasons"]

    # Annotate Frame using VideoBatchProcessor overlay
    processor = VideoBatchProcessor(florence_scanner=florence, pose_engine=pose_eng)
    annotated_bgr = processor.draw_overlay(frame, detections, pose_data, threat_score, is_triggered, False, reasons, 30.0)

    rgb_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    status_badge = "🚨 THREAT ALARM TRIGGERED!" if is_triggered else "✅ SAFE (No Threat)"
    nv_status_str = format_nv_status(florence)

    details = f"""
    ### Frame {target_frame} / {total_frames} Inspection
    - **Status**: {status_badge}
    - **NVIDIA LocateAnything 3B Verified**: {nv_status_str}
    - **Threat Score**: **{threat_score*100:.1f}%**
    - **Triggers**: {', '.join(reasons) if reasons else 'None'}
    - **Objects Detected**: {[d['label'] for d in detections]}
    - **Hands-Up Count**: {sum(1 for p in pose_data if p['hands_up'])}
    - **Waistband Reach**: {sum(1 for p in pose_data if p['waistband_reach'])}
    - **Florence-2 Latency**: {s1_ms:.1f} ms
    - **YOLO-Pose Latency**: {pose_ms:.1f} ms
    """

    return rgb_frame, details

def get_zone_config_text():
    zone_file = os.path.join(CURRENT_DIR, "zones.json")
    if os.path.exists(zone_file):
        with open(zone_file, "r") as f:
            return f.read()
    return "{}"

def save_zone_config_text(json_text):
    try:
        data = json.loads(json_text)
        zone_file = os.path.join(CURRENT_DIR, "zones.json")
        with open(zone_file, "w") as f:
            json.dump(data, f, indent=4)
        return "✅ Zones updated successfully!"
    except Exception as e:
        return f"❌ JSON Error: {e}"

def build_gradio_app():
    theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate",
        neutral_hue="slate"
    )

    with gr.Blocks(theme=theme, title="Real-Time Threat Detection Center") as demo:
        gr.Markdown(
            """
            # 🛡️ Real-Time Robbery & Threat Detection AI Center
            ### Cascaded Dual-Stage Architecture (Florence-2 Open-Vocab Scanner + YOLOv8-Pose + NVIDIA LocateAnything 3B)
            """
        )

        server_status_markdown = gr.Markdown(check_server_status())

        with gr.Tabs():

            # TAB 1: REAL-TIME VIDEO FILE DETECTOR
            with gr.TabItem("▶️ Real-Time Video File Threat Detector"):
                gr.Markdown("### 🎥 Stream & Detect Threats Live in Any Video File")
                with gr.Row():
                    with gr.Column(scale=1):
                        realtime_video_input = gr.Video(
                            value=os.path.join(INPUT_VIDEOS_DIR, "combined_output.mp4"),
                            label="Target Video File (default: combined_output.mp4)"
                        )
                        realtime_scan_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Florence-2 Scan Interval (Every N Frames)",
                            info="Lower value = higher detection precision; Higher value = faster FPS"
                        )
                        with gr.Row():
                            start_realtime_btn = gr.Button("▶️ Start Real-Time Detection", variant="primary")
                            stop_realtime_btn = gr.Button("⏹️ Stop Stream Midway", variant="stop")

                        realtime_meter_slider = gr.Slider(
                            minimum=0, maximum=100, value=0,
                            label="Real-Time Threat Score (%)",
                            interactive=False
                        )

                    with gr.Column(scale=2):
                        realtime_hud_image = gr.Image(label="Real-Time Security HUD Overlay")
                        realtime_status_md = gr.Markdown("### 🟢 STATUS: SAFE (0%)")
                        realtime_details_md = gr.Markdown("Select a video and click '▶️ Start Real-Time Detection' to watch live stream.")

                start_realtime_btn.click(
                    fn=stream_realtime_video_detection,
                    inputs=[realtime_video_input, realtime_scan_slider],
                    outputs=[realtime_hud_image, realtime_status_md, realtime_meter_slider, realtime_details_md]
                )

                stop_realtime_btn.click(
                    fn=stop_realtime_stream,
                    outputs=[realtime_status_md]
                )

            # TAB 2: REAL-TIME LIVE WEBCAM & CAMERA MONITOR
            with gr.TabItem("🔴 Real-Time Live Webcam & Camera Monitor"):
                gr.Markdown("### 🎥 Live Stream Camera Monitoring & Instant Threat Detection")
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Live Webcam / Camera Feed"
                        )
                        live_meter_slider = gr.Slider(
                            minimum=0, maximum=100, value=0,
                            label="Live Threat Score (%)",
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        live_hud_output = gr.Image(label="Real-Time Security HUD Overlay")
                        live_status_md = gr.Markdown("### 🟢 STATUS: SAFE (0%)")
                        live_details_md = gr.Markdown("Enable webcam to start live streaming detection.")

                webcam_input.stream(
                    fn=process_live_stream_frame,
                    inputs=[webcam_input],
                    outputs=[live_hud_output, live_status_md, live_meter_slider, live_details_md]
                )

            # TAB 3: BATCH VIDEO TEST & FULL ANALYTICS EXPORTER
            with gr.TabItem("🎬 Video Batch Exporter & Analytics"):
                gr.Markdown("### Export Full MP4 Video & Offline Analytics Report")
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            value=os.path.join(INPUT_VIDEOS_DIR, "combined_output.mp4"),
                            label="Target Video File (default: combined_output.mp4)"
                        )
                        scan_interval_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Florence-2 Scan Interval (Every N Frames)",
                            info="Lower value = higher detection precision; Higher value = faster FPS"
                        )
                        run_batch_btn = gr.Button("🚀 Run Full Offline Batch Export", variant="primary")
                        
                    with gr.Column(scale=2):
                        kpi_output = gr.Markdown("Click 'Run Full Offline Batch Export' to generate metrics and export MP4 video.")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📹 Processed Threat-Annotated Video")
                        annotated_video_output = gr.Video(label="Annotated Video Output")
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📈 Dynamic Threat Score Timeline")
                        timeline_chart_output = gr.Image(label="Threat Score Curve (%)")

                gr.Markdown("#### 📋 Logged Security Incidents")
                incidents_table_output = gr.Dataframe(headers=["Frame #", "Time (s)", "Threat Score", "Trigger Reasons", "Stage 2 Confirmed"])

                run_batch_btn.click(
                    fn=run_batch_test,
                    inputs=[video_input, scan_interval_slider],
                    outputs=[annotated_video_output, timeline_chart_output, kpi_output, incidents_table_output]
                )

            # TAB 4: SINGLE FRAME INSPECTION
            with gr.TabItem("🔍 Frame-by-Frame Inspector"):
                gr.Markdown("### Inspect Individual Video Frames & Threat Factors")
                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_video_input = gr.Video(
                            value=os.path.join(INPUT_VIDEOS_DIR, "combined_output.mp4"),
                            label="Source Video"
                        )
                        frame_num_input = gr.Number(value=30, label="Frame Number to Inspect", precision=0)
                        inspect_btn = gr.Button("🔍 Inspect Frame", variant="secondary")

                    with gr.Column(scale=2):
                        frame_image_output = gr.Image(label="Annotated Frame View")
                        frame_details_output = gr.Markdown("Select a frame and click 'Inspect Frame'.")

                inspect_btn.click(
                    fn=inspect_single_frame,
                    inputs=[inspect_video_input, frame_num_input],
                    outputs=[frame_image_output, frame_details_output]
                )

            # TAB 5: RESTRICTED ZONE & SYSTEM SETTINGS
            with gr.TabItem("⚙️ Zone Manager & Config"):
                gr.Markdown("### Restricted Zone Polygon Configuration")
                with gr.Row():
                    with gr.Column(scale=1):
                        zone_json_input = gr.Code(
                            value=get_zone_config_text(),
                            language="json",
                            label="zones.json Configuration"
                        )
                        save_zone_btn = gr.Button("💾 Save Polygon Zones", variant="primary")
                        zone_status_msg = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                            #### Threat Score Formula Parameters
                            $$\\text{Threat Score} = S_{\\text{weapon}} + S_{\\text{mask}} + S_{\\text{zone}} + S_{\\text{pose}}$$

                            - **$S_{\\text{weapon}} = +0.85$**: Gun, knife, blade detection.
                            - **$S_{\\text{mask}} = +0.50$**: Full face mask (ski mask / balaclava) inside zone.
                            - **$S_{\\text{pose}} = +0.40$**: Hands-up surrender (+0.45), waistband reach (+0.40), or arm aiming (+0.55).
                            - **$S_{\\text{zone}} = +0.20$**: Cash counter area polygon breach by a person.
                            - **Buffer Size = 3 frames**: Smoothing buffer to eliminate flicker.
                            """
                        )

                save_zone_btn.click(
                    fn=save_zone_config_text,
                    inputs=[zone_json_input],
                    outputs=[zone_status_msg]
                )

    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
