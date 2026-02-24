import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import Config
from data_preprocessing import create_data_loader, load_sample_data
from model import create_model

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions)

def train_model():
    """主训练函数"""
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_PATH)
    Config.TOKENIZER = tokenizer
    
    # 加载数据
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    
    # 划分训练集和验证集
    # random_state=42 确保每次运行时数据划分保持一致，便于实验复现
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
    
    # 创建数据加载器
    train_data_loader = create_data_loader(
        df_train, tokenizer, Config.MAX_LEN, Config.TRAIN_BATCH_SIZE
    )
    
    val_data_loader = create_data_loader(
        df_val, tokenizer, Config.MAX_LEN, Config.VALID_BATCH_SIZE, shuffle=False
    )
    
    # 创建模型
    device = torch.device(Config.DEVICE)
    model = create_model(Config.BERT_PATH).to(device)
    
    # 定义优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    best_accuracy = 0
    
    # 训练循环
    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        print('-' * 10)
        
        train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            scheduler,
            device
        )
        
        val_accuracy = eval_model(model, val_data_loader, device)
        
        print(f'Train Loss: {train_loss:.4f} Val Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            torch.save(model.state_dict(), Config.MODEL_PATH)
            best_accuracy = val_accuracy
    
    print(f'Best validation accuracy: {best_accuracy:.4f}')

if __name__ == "__main__":
    train_model()