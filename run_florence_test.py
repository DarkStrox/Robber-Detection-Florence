import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import transformers.dynamic_module_utils

# Monkey-patch check_imports for Windows compatibility
orig_check_imports = transformers.dynamic_module_utils.check_imports
def custom_check_imports(filename):
    try:
        imports = orig_check_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    except ImportError as e:
        if "flash_attn" in str(e):
            return []
        raise e

transformers.dynamic_module_utils.check_imports = custom_check_imports

from transformers import AutoProcessor, AutoModelForCausalLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.environ["HF_HOME"] = MODELS_DIR

MODEL_ID = "microsoft/Florence-2-base"

print("==================================================")
print("  LOADING FLORENCE-2 REAL-TIME TEST ENGINE")
print("==================================================")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# Load processor and model
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=MODELS_DIR
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=MODELS_DIR
).to(device)

print("Model loaded successfully!\n")

def run_inference(image_path, prompt_text):
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    start_time = time.time()
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=1
    )
    inference_ms = (time.time() - start_time) * 1000

    result_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    task_name = prompt_text.split(" ")[0] if " " in prompt_text else prompt_text
    parsed_result = processor.post_process_generation(
        result_text, 
        task=task_name, 
        image_size=(image.width, image.height)
    )
    
    return parsed_result, inference_ms, image

def test_image(image_path):
    print(f"\n--- Testing Image: {os.path.basename(image_path)} ---")
    
    # 1. Detailed Scene Caption
    print("\n1. [Task: Detailed Caption]")
    caption_result, caption_ms, img = run_inference(image_path, "<DETAILED_CAPTION>")
    caption = caption_result.get("<DETAILED_CAPTION>", "")
    print(f"   Inference Time: {caption_ms:.2f} ms")
    print(f"   Description: {caption}")

    # 2. Open Vocabulary Detection
    query_targets = "person, counter, cash register, bag, gun, mask"
    print(f"\n2. [Task: Open Vocabulary Detection] -> targets: '{query_targets}'")
    detect_prompt = f"<OPEN_VOCABULARY_DETECTION> {query_targets}"
    detect_result, detect_ms, _ = run_inference(image_path, detect_prompt)
    print(f"   Inference Time: {detect_ms:.2f} ms")
    
    bboxes = detect_result.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes", [])
    labels = detect_result.get("<OPEN_VOCABULARY_DETECTION>", {}).get("labels", [])
    
    print(f"   Detected Objects Count: {len(bboxes)}")
    
    # Draw detections onto image
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for bbox, label in zip(bboxes, labels):
        print(f"    - Found '{label}' at bbox: {bbox}")
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), label, fill="yellow")
    
    output_path = os.path.join(CURRENT_DIR, "result_detected.png")
    draw_img.save(output_path)
    print(f"\nSaved visualization with detections to: {output_path}")

if __name__ == "__main__":
    sample_image = os.path.join(CURRENT_DIR, "test_security_scene.png")
    test_image(sample_image)
