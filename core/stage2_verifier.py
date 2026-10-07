import os
import time
import base64
import re
import requests
import cv2
from PIL import Image
import io

class Stage2Verifier:
    """
    Stage 2 High-Precision Verifier using NVIDIA Locate Anything 3B GGUF.
    Communicates via HTTP with local server on http://127.0.0.1:8085.
    Performs deep zero-shot inspection on cropped ROIs (weapons vs wallets/phones, ski mask vs medical mask).
    """
    def __init__(self, server_url="http://127.0.0.1:8085/v1/chat/completions"):
        self.server_url = server_url
        print(f"[Stage2Verifier] Connected to NVIDIA Locate Anything server at {self.server_url}")

    def is_server_online(self):
        try:
            health_url = self.server_url.replace("/v1/chat/completions", "/props")
            res = requests.get(health_url, timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def verify_crop(self, crop_bgr, query_prompt="gun, knife, weapon"):
        """
        Sends cropped BGR image region to NVIDIA Locate Anything 3B.
        Returns parsed detections and latency in milliseconds.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return [], 0.0

        # Encode crop to JPEG base64
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
                print(f"[Stage2Verifier] Server returned HTTP {res.status_code}")
                return [], latency_ms
                
            data = res.json()
            response_text = data["choices"][0]["message"]["content"]
            
            # Parse bounding box responses
            detections = []
            regex = r'(?:<ref>([\s\S]*?)<\/ref>)|(?:<box>([\s\S]*?)<\/box>)'
            current_label = query_prompt
            
            for match in re.finditer(regex, response_text):
                if match.group(1) is not None:
                    current_label = match.group(1).strip()
                elif match.group(2) is not None:
                    coords_str = match.group(2).strip()
                    if coords_str.lower() != 'none':
                        detections.append(current_label.lower())
                        
            return detections, latency_ms
        except Exception as e:
            print(f"[Stage2Verifier] Verification call failed: {e}")
            return [], 0.0
