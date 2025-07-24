import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# ----- 1. Load fine-tuned model -----
MODEL_DIR = "git-finetuned"  # ← directory where you saved model
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----- 2. Load test image -----
img_path = "/home/nvidia/jetson-inference/python/training/classification/data/final_project/caption_data/senticap_images/COCO_val2014_000000011182.jpg"  # ← change as needed
image = Image.open(img_path).convert("RGB").resize((224, 224))

# ----- 3. Preprocess input -----
inputs = processor(images=image, return_tensors="pt").to(device)

# ----- 4. Generate caption -----
generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
)

# ----- 5. Decode and print result -----
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated Caption:", caption)
