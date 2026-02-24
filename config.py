class Config:
    # 数据相关配置
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    
    # 模型相关配置
    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "model.bin"
    
    # 训练相关配置
    DEVICE = "cpu"  # 可以改为 "cuda" 如果有GPU
    TOKENIZER = None