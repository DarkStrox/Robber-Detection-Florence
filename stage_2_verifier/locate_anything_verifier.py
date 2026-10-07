import os
import time
import base64
import re
import requests
import cv2

class LocateAnythingVerifier:
    """
    Stage 2: High-Precision Verifier using NVIDIA Locate Anything 3B GGUF.
    Communicates via HTTP with local server on http://127.0.0.1:8085.
    Performs deep zero-shot inspection on cropped ROIs.
    """
    def __init__(self, server_url="http://127.0.0.1:8085/v1/chat/completions"):
        self.server_url = server_url
        print(f"[Stage 2 LocateAnythingVerifier] Connected to NVIDIA server at {self.server_url}")

    def is_server_online(self):
        try:
            health_url = self.server_url.replace("/v1/chat/completions", "/health")
            res = requests.get(health_url, timeout=2)
            if res.status_code == 200:
                return True
            models_url = self.server_url.replace("/v1/chat/completions", "/v1/models")
            res_m = requests.get(models_url, timeout=2)
            return res_m.status_code == 200
        except Exception:
            return False

    def verify_crop(self, crop_bgr, query_prompt="gun, knife, weapon"):
        if crop_bgr is None or crop_bgr.size == 0:
            return [], 0.0

        # Upscale crop 2x for low-resolution security frames
        h, w, _ = crop_bgr.shape
        if h < 100 or w < 100:
            crop_bgr = cv2.resize(crop_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        _, buffer = cv2.imencode('.jpg', crop_bgr)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{base64_str}"

        payload = {
            "model": "LocateAnything-3B-Q4_K_M.gguf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<image>\nLocate all instances matching: {query_prompt}."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 120
        }

        start_time = time.time()
        try:
            res = requests.post(self.server_url, json=payload, timeout=10)
            latency_ms = (time.time() - start_time) * 1000
            
            if res.status_code != 200:
                return [], latency_ms
                
            data = res.json()
            response_text = data["choices"][0]["message"]["content"]
            low_text = response_text.lower()
            
            # 1. Reject explicit None responses
            if "<box>none</box>" in low_text or "<box>[none]</box>" in low_text:
                return [], latency_ms

            # 2. Reject whole-image fallback boxes <0><0><1000><1000> or <0><0><999><999>
            if "<0><0><1000><1000>" in response_text or "<0><0><999><999>" in response_text:
                return [], latency_ms

            # 3. Parse valid localized bounding box detections
            detections = []
            regex = r'(?:<ref>([\s\S]*?)<\/ref>)|(?:<box>([\s\S]*?)<\/box>)'
            current_label = "weapon"
            
            for match in re.finditer(regex, response_text):
                if match.group(1) is not None:
                    lbl = match.group(1).strip().lower()
                    if any(w in lbl for w in ["gun", "handgun", "pistol"]):
                        current_label = "gun"
                    elif "knife" in lbl or "blade" in lbl:
                        current_label = "knife"
                    else:
                        current_label = "weapon"
                elif match.group(2) is not None:
                    coords_str = match.group(2).strip()
                    if coords_str.lower() not in ['none', '[none]', '']:
                        detections.append(current_label)
                        
            return list(set(detections)), latency_ms
        except Exception as e:
            print(f"[Stage 2 Verifier] Verification request failed: {e}")
            return [], 0.0
