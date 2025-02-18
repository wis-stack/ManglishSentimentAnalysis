from transformers import AdamW
from model import SemanticClassifier
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch

BATCH_SIZE = 32
NUM_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SemanticClassifier()
model.to(device)
optimizer = AdamW(model.parameters(),lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
total_steps = len(dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)