from transformers import BertModel
import torch.nn as nn

class SemanticClassifier(nn.Module):
    def __init__(self,num_classes=3):
        super(SemanticClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes) 
        )

        for param in self.bert.encoder.layer[:-4].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits