# finetune_git_captioning_fixed.py
import os, torch
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import os
from datasets import Dataset

CSV_PATH  = "caption_data/senticap.csv"
IMAGE_DIR = "caption_data/senticap_images"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# df = df.sample(30, random_state=42)  # ↓ Try smaller values (e.g. 50) if needed

# Construct full image paths
df["image_path"] = df["filename"].apply(lambda x: os.path.join(IMAGE_DIR, x))

# Use only the image path and raw caption
df["caption"] = df["raw"]
ds = Dataset.from_pandas(df[["image_path", "caption"]])


# ------------------------------------------------------------------
# 1.  Model + processor  -------------------------------------------
# ------------------------------------------------------------------
MODEL_ID  = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Freeze the vision encoder (saves VRAM / avoids over-fitting)
for name, param in model.named_parameters():
    if name.startswith("git.vision") or "vision_model" in name or "vision_encoder" in name:
        param.requires_grad = False

num_total      = sum(p.numel() for p in model.parameters())
num_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {num_trainable} / {num_total} "
      f"({100 * num_trainable / num_total:.2f}%)")

# ------------------------------------------------------------------
# 2.  Pre-processing  ------------------------------------------------
# ------------------------------------------------------------------
def preprocess(example):
    img = Image.open(example["image_path"]).convert("RGB").resize((224, 224))

    enc = processor(
        images      = img,
        text        = example["caption"],
        padding     = "max_length",
        truncation  = True,
        return_tensors = "pt",
    )

    # Mask out padding tokens in the labels
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100

    return {
        "pixel_values"   : enc["pixel_values"].squeeze(),
        "input_ids"      : enc["input_ids"].squeeze(),
        "attention_mask" : enc["attention_mask"].squeeze(),
        "labels"         : labels.squeeze(),
    }

ds = ds.map(preprocess)  # num_proc>1 for faster preprocessing
ds.set_format(type="torch",
              columns=["pixel_values", "input_ids", "attention_mask", "labels"])

# ------------------------------------------------------------------
# 3.  Training  -----------------------------------------------------
# ------------------------------------------------------------------
args = TrainingArguments(
    output_dir                   = "git-finetuned",
    fp16                         = torch.cuda.is_available(),
    per_device_train_batch_size  = 1,
    gradient_accumulation_steps  = 1,
    num_train_epochs             = 20,
    learning_rate                = 2e-5,
    lr_scheduler_type            = "linear",
    warmup_ratio                 = 0.1,
    logging_steps                = 10,
    save_strategy                = "epoch",
    save_total_limit             = 1,
    remove_unused_columns        = False,
    gradient_checkpointing       = True,
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = ds,
    tokenizer       = processor,
)

trainer.train()

# ------------------------------------------------------------------
# 4.  Save  ---------------------------------------------------------
# ------------------------------------------------------------------
model.save_pretrained("git-finetuned")
processor.save_pretrained("git-finetuned")
