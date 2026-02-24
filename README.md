# 文本分类模型训练示例

这是一个完整的文本分类模型训练示例，展示了从数据预处理、模型训练、预测到模型导出和加载的全过程。

## 目录结构

```
.
├── config.py              # 配置文件
├── data_preprocessing.py  # 数据预处理模块
├── model.py               # 模型定义
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── model_utils.py         # 模型导出和加载工具
├── example.py             # 完整示例
├── requirements.txt       # 依赖包
└── README.md              # 说明文档
```

## 环境安装

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 运行完整示例

直接运行完整示例，它会自动完成训练、预测、导出和加载的全过程：

```bash
python example.py
```

### 2. 分步执行

#### 模型训练
```bash
python train.py
```

#### 使用训练好的模型进行预测
```bash
python predict.py
```
(需要先修改脚本中的注释部分来添加实际的预测代码)

### 3. 模型导出和加载

模型导出和加载功能在 `model_utils.py` 中实现：
- `export_model()`: 导出模型和tokenizer
- `load_model()`: 从指定路径加载模型和tokenizer

## 代码说明

### config.py
包含模型训练的各种超参数配置：
- MAX_LEN: 文本最大长度
- TRAIN_BATCH_SIZE/VALID_BATCH_SIZE: 训练/验证批次大小
- EPOCHS: 训练轮数
- BERT_PATH: 使用的BERT模型路径
- MODEL_PATH: 模型保存路径

### data_preprocessing.py
数据预处理模块：
- `TextDataset`: 自定义数据集类
- `create_data_loader`: 创建PyTorch数据加载器
- `load_sample_data`: 加载示例数据

### model.py
模型定义：
- `TextClassifier`: 基于BERT的文本分类模型

### train.py
训练脚本：
- `train_epoch`: 训练一个epoch
- `eval_model`: 评估模型
- `train_model`: 主训练函数

### predict.py
预测脚本：
- `predict_text`: 对单个文本进行预测
- `batch_predict`: 批量预测文本

### model_utils.py
模型工具：
- `export_model`: 导出模型
- `load_model`: 加载模型
- `get_model_info`: 获取模型信息

### example.py
完整示例，展示了以下流程：
1. 模型训练
2. 使用训练好的模型进行预测
3. 导出模型
4. 从导出目录加载模型并再次预测

## 自定义数据

要使用自己的数据，只需修改 `data_preprocessing.py` 中的 `load_sample_data()` 函数，使其加载你的数据即可。确保数据格式为包含'text'和'label'列的DataFrame。

## 注意事项

1. 默认使用CPU训练，如果有GPU可以修改 `config.py` 中的 DEVICE 为 "cuda"
2. 示例中使用了较小的数据集，实际使用时建议使用更大的数据集
3. 可以根据需要调整BERT模型类型和其他超参数