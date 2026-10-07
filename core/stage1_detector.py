import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import cv2
from PIL import Image
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

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.environ["HF_HOME"] = MODELS_DIR

class Stage1Detector:
    r"""
    Stage 1 Fast Frame Scanner using Microsoft Florence-2-base (~120 ms).
    Loaded locally from E:\Aitronix Projects\Robber Detection\models.
    """
    def __init__(self, model_id="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stage1Detector] Loading Florence-2-base on {self.device.upper()}...")
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True,
            cache_dir=MODELS_DIR
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=MODELS_DIR
        ).to(self.device)
        
        print("[Stage1Detector] Florence-2-base loaded successfully!")

    def scan_frame(self, frame_bgr, target_prompts="person, gun, knife, ski mask, balaclava, bag, counter"):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        prompt_text = f"<OPEN_VOCABULARY_DETECTION> {target_prompts}"
        
        inputs = self.processor(text=prompt_text, images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        start_time = time.time()
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=1
        )
        latency_ms = (time.time() - start_time) * 1000

        result_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result_text, 
            task="<OPEN_VOCABULARY_DETECTION>", 
            image_size=(pil_img.width, pil_img.height)
        )

        detections = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])
        
        results = []
        for bbox, label in zip(bboxes, labels):
            results.append({
                "label": label.strip().lower(),
                "bbox": bbox  # [x1, y1, x2, y2]
            })
            
        return results, latency_ms
