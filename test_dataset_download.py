from data.data_enhancer import DataEnhancer
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

# 创建增强器实例
config = {
    'total_data_limit': 200000,
    'max_length': 16
}
enhancer = DataEnhancer(config)
enhancer.load_tokenizer('bert-base-uncased')

# 测试直接添加本地模拟数据集
print("测试直接添加本地模拟数据集...")

# 创建模拟数据
mock_data = {
    "text": ["这是一个积极的评论。", "这是一个消极的评论。", "这个产品非常好！", "我不喜欢这个服务。"],
    "label": [1, 0, 1, 0]
}

# 创建数据集
dataset = Dataset.from_dict(mock_data)

# 直接将数据集添加到增强器中
enhancer.datasets.append({
    "name": "mock_local_dataset",
    "data": dataset,
    "text_column": "text",
    "label_column": "label"
})
enhancer.dataset_weights.append(1.0)

print("数据集添加成功！")
print(f"当前增强器包含 {len(enhancer.datasets)} 个数据集")

# 获取合并后的数据集
combined_dataset = enhancer.balance_and_sample(method="weighted")
print(f"合并后的数据集包含 {len(combined_dataset)} 个样本")

# 测试预处理
input_ids, attention_mask, labels = enhancer.preprocess_for_training(combined_dataset, max_length=16)
print(f"预处理后数据: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}, labels={len(labels)}")
print(f"单个输入形状: {input_ids[0].shape}")

print("\n===== 测试完成 =====")
