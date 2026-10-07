import os
import time
import base64
import re
import requests
from PIL import Image, ImageDraw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://127.0.0.1:8085/v1/chat/completions"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        if ext == "jpg": ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"

def detect_with_locate_anything(image_path, query_text):
    print(f"\n--- Running NVIDIA Locate Anything 3B ---")
    print(f"Query: '{query_text}'")
    
    base64_image = encode_image_to_base64(image_path)
    
    payload = {
        "model": "LocateAnything-3B-Q4_K_M.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<image>\nLocate all the instances that matches the following description: {query_text}."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 120
    }
    
    start_time = time.time()
    response = requests.post(SERVER_URL, json=payload, timeout=30)
    inference_ms = (time.time() - start_time) * 1000
    
    if response.status_code != 200:
        print(f"Error from server: {response.status_code} - {response.text}")
        return [], inference_ms
        
    data = response.json()
    response_text = data["choices"][0]["message"]["content"]
    print(f"Inference Time: {inference_ms:.2f} ms")
    print(f"Raw Response: {response_text}")
    
    # Parse bounding boxes
    detections = []
    regex = r'(?:<ref>([\s\S]*?)<\/ref>)|(?:<box>([\s\S]*?)<\/box>)'
    current_label = query_text
    
    img = Image.open(image_path)
    w, h = img.size
    
    for match in re.finditer(regex, response_text):
        if match.group(1) is not None:
            current_label = match.group(1).strip()
        elif match.group(2) is not None:
            coords_str = match.group(2).strip()
            if coords_str.lower() == 'none':
                continue
                
            coords = None
            if '<' in coords_str and '>' in coords_str:
                num_matches = re.findall(r'<(\d+)>', coords_str)
                if len(num_matches) == 4:
                    coords = [int(m) for m in num_matches]
            else:
                cleaned = coords_str.replace('[', '').replace(']', '')
                parts = [float(n) for n in re.split(r'[\s,]+', cleaned) if n]
                if len(parts) == 4:
                    coords = parts
                    
            if coords:
                ymin, xmin, ymax, xmax = coords
                # Scale 0-1000 coordinates to actual pixel dimensions
                pixel_bbox = [
                    (xmin / 1000.0) * w,
                    (ymin / 1000.0) * h,
                    (xmax / 1000.0) * w,
                    (ymax / 1000.0) * h
                ]
                detections.append((current_label, pixel_bbox))
                
    return detections, inference_ms

if __name__ == "__main__":
    test_img = os.path.join(CURRENT_DIR, "test_security_scene.png")
    detections, ms = detect_with_locate_anything(test_img, "person, counter, bag, cap")
    
    print(f"\nParsed {len(detections)} detection bounding boxes:")
    for label, bbox in detections:
        print(f" - [{label}]: {bbox}")
