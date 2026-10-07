import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import cv2
import torch
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.environ["HF_HOME"] = MODELS_DIR

MODEL_ID = "microsoft/Florence-2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Florence-2 on {device.upper()} for Video Testing...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=MODELS_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=MODELS_DIR
).to(device)

def process_video_or_webcam(video_source=0, target_prompts="person, gun, knife, ski mask"):
    """
    Process video stream or file frame-by-frame with Florence-2 Open Vocabulary Detection.
    Pass video_source = 0 for live webcam, or video_source = 'path/to/video.mp4' for file.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        return

    print(f"\n--- Starting Real-Time Florence-2 Stream (Source: {video_source}) ---")
    print(f"Targeting: '{target_prompts}' | Press 'q' to stop.")
    
    prompt = f"<OPEN_VOCABULARY_DETECTION> {target_prompts}"
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream finished.")
            break

        frame_count += 1
        
        # Process every 5th frame for high FPS stream
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(device)
            if device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            start = time.time()
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=1
            )
            fps = 1.0 / (time.time() - start)

            result_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                result_text, 
                task="<OPEN_VOCABULARY_DETECTION>", 
                image_size=(pil_img.width, pil_img.height)
            )

            detections = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
            bboxes = detections.get("bboxes", [])
            labels = detections.get("labels", [])

            # Draw boxes on frame
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"Florence-2 FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Florence-2 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change to a video filepath like 'robbery_test.mp4' or 0 for live webcam
    process_video_or_webcam(0)
