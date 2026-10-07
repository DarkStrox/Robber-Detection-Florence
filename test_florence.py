import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import transformers.dynamic_module_utils

# Monkey-patch check_imports to ignore optional flash_attn on Windows
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

# Set custom models folder inside current workspace directory (E: drive)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Point Hugging Face cache to the local project folder on E: drive
os.environ["HF_HOME"] = MODELS_DIR

MODEL_ID = "microsoft/Florence-2-base"

print(f"Downloading & Loading {MODEL_ID} directly into local project directory:")
print(f" -> Path: {MODELS_DIR}")

# Load Processor & Model into local directory
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=MODELS_DIR
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=MODELS_DIR
).to(device)

print("SUCCESS: Model successfully downloaded and saved inside E: drive project folder!")
