import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    """创建数据加载器"""
    dataset = TextDataset(
        texts=df.text.values,
        targets=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def load_sample_data():
    """加载示例数据"""
    # 创建示例数据
    data = {
        'text': [
            'I love this movie, it is fantastic!',
            'This film is terrible, I hate it.',
            'Amazing acting and great story.',
            'Boring and poorly made.',
            'Best movie I have ever seen!',
            'Not good, very disappointing.',
            'Excellent cinematography and direction.',
            'Waste of time, do not watch.',
            'Brilliant performance by the actors.',
            'Awful script and bad acting.'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    }
    return pd.DataFrame(data)