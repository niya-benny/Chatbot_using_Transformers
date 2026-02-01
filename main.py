# main.py
# Transformer-based Chatbot using T5 (PyTorch)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "t5-small"
DATA_PATH = "data/conversations.txt"
SAVE_DIR = "chatbot_model"

EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 64
LEARNING_RATE = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATASET
# ----------------------------
class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    inp, out = line.strip().split("\t")
                    self.pairs.append((inp, out))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]

        source = self.tokenizer(
            "chat: " + input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }

# ----------------------------
# TRAINING
# ----------------------------
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    dataset = ChatDataset(tokenizer, DATA_PATH, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("✅ Model saved to:", SAVE_DIR)

# ----------------------------
# INFERENCE
# ----------------------------
def load_model():
    if not os.path.exists(SAVE_DIR):
        raise FileNotFoundError(
            "❌ chatbot_model not found. Run train() first."
        )

    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model

def chat(tokenizer, model, text):
    inputs = tokenizer(
        "chat: " + text,
        return_tensors="pt",
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            top_p=0.9,
            temperature=0.7,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # Run this ONCE
    train()

    tokenizer, model = load_model()
    print(chat(tokenizer, model, "Hello"))
