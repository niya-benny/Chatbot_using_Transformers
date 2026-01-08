# main.py
# Transformer-based Chatbot using T5

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW
)
from tqdm import tqdm


# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "t5-small"
DATA_PATH = "data/conversations.txt"
EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 64
LEARNING_RATE = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
