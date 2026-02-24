import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

from config import Config
from model import create_model

def predict_text(text, model, tokenizer, device):
    """对单个文本进行预测"""
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    
    return predicted.item(), confidence.item()

def batch_predict(texts, model, tokenizer, device):
    """批量预测文本"""
    results = []
    for text in texts:
        pred, confidence = predict_text(text, model, tokenizer, device)
        results.append({
            'text': text,
            'prediction': 'positive' if pred == 1 else 'negative',
            'confidence': confidence
        })
    return results

if __name__ == "__main__":
    # 这个脚本需要先运行train.py训练模型
    # 然后可以在这里添加测试代码
    
    # 示例用法:
    texts = [
        "This movie is great!",
        "I hate this film.",
        "Amazing acting and great story.",
        "Boring and poorly made."
    ]
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_PATH)
    model = create_model(Config.BERT_PATH)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    device = torch.device(Config.DEVICE)
    model.to(device)
    
    # 批量预测
    results = batch_predict(texts, model, tokenizer, device)
    print("Prediction Results:")
    print("=" * 50)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        print("-" * 50)