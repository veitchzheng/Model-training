"""
完整的文本分类示例
展示模型训练、预测、导出和加载的全过程
"""

import torch
from transformers import AutoTokenizer

from config import Config
from data_preprocessing import load_sample_data
from model import create_model
from train import train_model
from predict import batch_predict
from model_utils import export_model, load_model, get_model_info

def run_complete_example():
    """运行完整的示例"""
    print("=== 文本分类模型完整示例 ===\n")
    
    # 1. 模型训练
    print("1. 开始模型训练...")
    train_model()
    print("模型训练完成!\n")
    
    # 2. 加载训练好的模型进行预测
    print("2. 加载训练好的模型进行预测...")
    device = torch.device(Config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_PATH)
    model = create_model(Config.BERT_PATH)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    
    # 测试预测
    test_texts = [
        "This is an amazing movie with great acting!",
        "Terrible film, waste of time.",
        "I really enjoyed watching this.",
        "Boring and uninteresting plot."
    ]
    
    results = batch_predict(test_texts, model, tokenizer, device)
    print("预测结果:")
    for result in results:
        print(f"  文本: {result['text']}")
        print(f"  预测: {result['prediction']} (置信度: {result['confidence']:.4f})")
    print()
    
    # 3. 导出模型
    print("3. 导出模型...")
    export_path = "exported_model"
    export_model(model, tokenizer, export_path)
    
    # 显示模型信息
    info = get_model_info(export_path)
    print(f"模型大小: {info.get('model_size_mb', 'N/A')} MB")
    print(f"模型文件: {info.get('model_path', 'N/A')}")
    print()
    
    # 4. 从导出目录加载模型并再次预测
    print("4. 从导出目录加载模型并预测...")
    loaded_model, loaded_tokenizer = load_model(export_path, device)
    
    # 使用加载的模型进行预测
    new_texts = [
        "Fantastic storyline and brilliant performances!",
        "Not worth watching, very poor quality."
    ]
    
    loaded_results = batch_predict(new_texts, loaded_model, loaded_tokenizer, device)
    print("加载模型后的预测结果:")
    for result in loaded_results:
        print(f"  文本: {result['text']}")
        print(f"  预测: {result['prediction']} (置信度: {result['confidence']:.4f})")
    print()
    
    print("=== 示例完成 ===")

if __name__ == "__main__":
    run_complete_example()