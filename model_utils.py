import torch
from transformers import AutoTokenizer, AutoConfig
import os

from config import Config
from model import create_model

def export_model(model, tokenizer, export_path):
    """
    导出模型和tokenizer
    
    Args:
        model: 训练好的模型
        tokenizer: 对应的tokenizer
        export_path: 导出路径
    """
    # 创建导出目录
    os.makedirs(export_path, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(export_path, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # 保存tokenizer
    tokenizer_path = os.path.join(export_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    # 保存模型配置
    config_path = os.path.join(export_path, "config.json")
    model.bert.config.to_json_file(config_path)
    
    print(f"Model exported to {export_path}")

def load_model(export_path, device):
    """
    从指定路径加载模型和tokenizer
    
    Args:
        export_path: 导出路径
        device: 设备 (cpu 或 cuda)
        
    Returns:
        model, tokenizer
    """
    # 加载tokenizer
    tokenizer_path = os.path.join(export_path, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 从预训练的BERT模型创建模型（使用配置）
    config_path = os.path.join(export_path, "config.json")
    if os.path.exists(config_path):
        # 如果存在配置文件，使用它来创建模型
        model = create_model(Config.BERT_PATH)
    else:
        # 否则使用默认的BERT路径
        model = create_model(Config.BERT_PATH)
    
    # 加载模型权重
    model_path = os.path.join(export_path, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {export_path}")
    return model, tokenizer

def get_model_info(export_path):
    """
    获取模型信息
    
    Args:
        export_path: 导出路径
        
    Returns:
        dict: 包含模型信息的字典
    """
    info = {}
    model_path = os.path.join(export_path, "model.pth")
    tokenizer_path = os.path.join(export_path, "tokenizer")
    
    if os.path.exists(model_path):
        model_stats = os.stat(model_path)
        info['model_size_mb'] = round(model_stats.st_size / (1024 * 1024), 2)
        info['model_path'] = model_path
    
    if os.path.exists(tokenizer_path):
        info['tokenizer_path'] = tokenizer_path
        # 计算tokenizer文件数量
        tokenizer_files = os.listdir(tokenizer_path)
        info['tokenizer_files'] = tokenizer_files
    
    return info