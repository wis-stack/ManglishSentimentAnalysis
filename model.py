from transformers import BertModel
import torch.nn as nn

class SemanticClassifier(nn.Module):
    def __init__(self,num_classes=3):
        super(SemanticClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        for param in self.bert.encoder.layer[:-2].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits