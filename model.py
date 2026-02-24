import torch
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的输出作为整个序列的表示
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def create_model(bert_model_name, num_classes=2):
    """创建模型实例"""
    model = TextClassifier(bert_model_name, num_classes)
    return model