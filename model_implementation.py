import torch
from transformers import BertTokenizer
from preprocessing import Preprocessing
from model import SemanticClassifier
import torch.serialization

torch.serialization.add_safe_globals([SemanticClassifier])
labels = ["Negative", "Neutral", "Positive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
preprocessing_instance = Preprocessing()
model = torch.load(
    "data/semantic_classifier.pth", map_location=device, weights_only=False
)
model.to(device)
model.eval()


async def preprocess_text(text, max_length=128, device="cuda"):
    processed_text = await preprocessing_instance.preprocessing_pipeline(text)
    processed_text = str(processed_text)
    encoding = tokenizer(
        processed_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


async def predict(text):
    inputs = await preprocess_text(text, device=device)

    with torch.no_grad():
        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    predicted_label = torch.argmax(output, dim=1).cpu().item()
    return labels[int(predicted_label)]
