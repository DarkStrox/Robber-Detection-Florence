import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import cv2
from PIL import Image
import transformers.dynamic_module_utils

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
from stage_2_verifier import LocateAnythingVerifier

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.environ["HF_HOME"] = MODELS_DIR

def is_valid_weapon_geometry(bbox, person_boxes):
    """
    Filters out static counter fixtures (monitor poles, stand bars) and detached merchandise.
    - Rejects extreme thin stick/pole geometries (aspect ratio > 2.8).
    - Requires proximity to a person or person's hands.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    aspect_ratio = max(w / h, h / w)

    # 1. Filter out extreme thin poles / sticks (e.g. monitor arms, metal stand bars)
    if aspect_ratio > 2.8:
        return False

    # 2. Must be associated with / near a person (within 60px of a person box)
    if not person_boxes:
        return False

    near_person = False
    for (px1, py1, px2, py2) in person_boxes:
        if not (x2 < px1 - 60 or x1 > px2 + 60 or y2 < py1 - 60 or y1 > py2 + 60):
            near_person = True
            break

    return near_person

class FlorenceScanner:
    r"""
    Stage 1: Fast Open-Vocabulary Frame Scanner using Florence-2-base (~120 ms).
    Uses Florence-2 <CAPTION_TO_PHRASE_GROUNDING> + Stage 2 Verifier confirmation to eliminate false positive sticks/poles & hoodie masks.
    """
    def __init__(self, model_id="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stage 1 FlorenceScanner] Loading model on {self.device.upper()}...")
        
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

        # Stage 2 NVIDIA Verifier instance
        self.verifier = LocateAnythingVerifier()

        # Telemetry tracking for NVIDIA LocateAnything 3B
        self.last_nv_online = False
        self.last_nv_verified_items = []
        self.last_nv_latency_ms = 0.0
        
        print("[Stage 1 FlorenceScanner] Model ready!")

    def scan_frame(self, frame_bgr, target_prompts="person, gun, handgun, pistol, firearm, weapon, knife, blade, ski mask, balaclava, full face mask, bag, counter, cash register"):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        width, height = pil_img.size
        
        # 1. Florence-2 Phrase Grounding
        prompt_text = f"<CAPTION_TO_PHRASE_GROUNDING> {target_prompts}"
        
        inputs = self.processor(text=prompt_text, images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        start_time = time.time()
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=1
        )
        latency_ms = (time.time() - start_time) * 1000

        result_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result_text, 
            task="<CAPTION_TO_PHRASE_GROUNDING>", 
            image_size=(width, height)
        )

        detections = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])
        
        person_boxes = [bbox for bbox, label in zip(bboxes, labels) if "person" in label.lower()]

        results = []
        is_nv_online = self.verifier.is_server_online()
        nv_verified_items = []
        nv_latency_total = 0.0

        for bbox, label in zip(bboxes, labels):
            lbl_clean = label.strip().lower()
            is_weapon = any(w in lbl_clean for w in ["gun", "handgun", "pistol", "firearm", "weapon", "knife", "blade"])
            is_mask = any(m in lbl_clean for m in ["ski mask", "balaclava", "full face mask"])
            
            if is_weapon:
                # A. Apply geometric & person-proximity filtering to discard sticks/poles & counter equipment
                if not is_valid_weapon_geometry(bbox, person_boxes):
                    continue

                # B. Stage 2 NVIDIA Verifier Crop Check (if online)
                if is_nv_online:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop = frame_bgr[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    nv_dets, nv_ms = self.verifier.verify_crop(crop, query_prompt="handgun, pistol, knife")
                    nv_latency_total += nv_ms
                    if not nv_dets:
                        continue  # NVIDIA Verifier rejected the candidate crop as false positive
                    else:
                        nv_verified_items.extend(nv_dets)

                lbl_clean = "gun" if any(w in lbl_clean for w in ["gun", "handgun", "pistol", "firearm", "weapon"]) else "knife"

            elif is_mask:
                # Stage 2 NVIDIA Verifier Mask Check (reject hoodies/hats)
                if is_nv_online:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop = frame_bgr[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    nv_dets, nv_ms = self.verifier.verify_crop(crop, query_prompt="ski mask, balaclava")
                    nv_latency_total += nv_ms
                    if not nv_dets:
                        continue  # NVIDIA Verifier confirmed it's just a hoodie/hat, NOT a ski mask!
                    else:
                        nv_verified_items.extend(nv_dets)

                lbl_clean = "ski mask"

            results.append({
                "label": lbl_clean,
                "bbox": bbox
            })

        self.last_nv_online = is_nv_online
        self.last_nv_verified_items = list(set(nv_verified_items))
        self.last_nv_latency_ms = nv_latency_total

        return results, latency_ms
