import os
import torch
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

class DataEnhancer:
    """
    数据增强和筛选器，用于下载多个数据集并控制总数据量
    确保数据多样性的同时保持可控的数据规模
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.total_data_limit = config.get('total_data_limit', 200000)  # 默认总数据量限制
        self.datasets = []
        self.dataset_weights = []
        
    def load_tokenizer(self, model_name: str = "bert-base-uncased"):
        """加载分词器用于数据处理"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer
    
    def add_dataset(self, dataset_name: str, split: str = "train", 
                   text_column: str = "text", label_column: str = "label",
                   weight: float = 1.0, max_samples: Optional[int] = None):
        """
        添加数据集到增强器
        
        Args:
            dataset_name: Hugging Face数据集名称
            split: 数据分割
            text_column: 文本列名
            label_column: 标签列名
            weight: 数据集权重（影响采样比例）
            max_samples: 该数据集的最大样本数限制
        """
        try:
            # 配置禁用SSL验证以解决证书问题
            import os
            import requests
            
            # 禁用requests的SSL验证
            requests.packages.urllib3.disable_warnings()
            session = requests.Session()
            session.verify = False
            
            # 使用环境变量配置SSL验证
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''
            
            # 检查本地是否已有数据集
            local_data_dir = os.path.join("data", dataset_name.replace("/", "_"))
            
            if os.path.exists(local_data_dir):
                print(f"从本地加载数据集: {dataset_name}...")
                dataset = load_dataset(local_data_dir, split=split)
            else:
                print(f"正在下载并保存数据集: {dataset_name}...")
                # 下载数据集并保存到本地
                dataset = load_dataset(dataset_name, split=split)
                dataset.save_to_disk(local_data_dir)
                print(f"数据集已保存到本地: {local_data_dir}")
            
            # 重命名列以保持一致性
            if text_column != "text":
                dataset = dataset.rename_column(text_column, "text")
            if label_column != "label":
                dataset = dataset.rename_column(label_column, "label")
            
            # 限制样本数
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            self.datasets.append({
                "name": dataset_name,
                "data": dataset,
                "text_column": "text",
                "label_column": "label"
            })
            self.dataset_weights.append(weight)
            
            print(f"成功加载数据集 {dataset_name}，包含 {len(dataset)} 个样本")
            return True
            
        except Exception as e:
            print(f"加载数据集 {dataset_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def balance_and_sample(self, method: str = "weighted"):
        """
        根据权重平衡采样数据集，控制总数据量
        
        Args:
            method: 采样方法 (weighted: 按权重采样, uniform: 均匀采样)
            
        Returns:
            平衡采样后的数据集
        """
        if not self.datasets:
            raise ValueError("没有加载任何数据集")
        
        # 计算总权重和每个数据集的采样比例
        total_weight = sum(self.dataset_weights)
        if method == "weighted":
            proportions = [w / total_weight for w in self.dataset_weights]
        else:  # uniform
            proportions = [1.0 / len(self.datasets)] * len(self.datasets)
        
        # 计算每个数据集的目标采样数量
        target_counts = []
        remaining = self.total_data_limit
        
        for i, (dataset_info, proportion) in enumerate(zip(self.datasets, proportions)):
            available = len(dataset_info["data"])
            target = int(self.total_data_limit * proportion)
            
            # 如果该数据集不够，使用全部并调整剩余
            if target > available:
                target_counts.append(available)
                remaining -= available
            else:
                target_counts.append(target)
                remaining -= target
        
        # 分配剩余的样本额度
        if remaining > 0:
            # 按比例分配剩余样本
            for i in range(len(target_counts)):
                if remaining <= 0:
                    break
                    
                dataset_len = len(self.datasets[i]["data"])
                if target_counts[i] < dataset_len:
                    add_count = min(remaining, dataset_len - target_counts[i])
                    target_counts[i] += add_count
                    remaining -= add_count
        
        # 采样每个数据集
        sampled_datasets = []
        for dataset_info, target_count in zip(self.datasets, target_counts):
            if target_count <= 0:
                continue
                
            # 随机采样
            indices = random.sample(range(len(dataset_info["data"])), target_count)
            sampled = dataset_info["data"].select(indices)
            sampled_datasets.append(sampled)
            
            print(f"从 {dataset_info['name']} 采样 {target_count} 个样本")
        
        # 合并数据集
        if len(sampled_datasets) == 1:
            combined = sampled_datasets[0]
        else:
            combined = concatenate_datasets(sampled_datasets)
        
        # 打乱数据顺序
        combined = combined.shuffle(seed=42)
        
        print(f"\n最终合并数据集: {len(combined)} 个样本")
        return combined
    
    def preprocess_for_training(self, dataset, max_length: int = 32):
        """
        预处理数据集用于训练
        
        Args:
            dataset: 输入数据集
            max_length: 最大序列长度
            
        Returns:
            预处理后的输入和标签张量
        """
        if not self.tokenizer:
            raise ValueError("请先加载分词器")
        
        def tokenize_function(examples):
            # 简单的序列到序列预处理，使用相同的输入输出
            model_inputs = self.tokenizer(examples["text"], max_length=max_length,
                                         padding="max_length", truncation=True)
            labels = self.tokenizer(examples["text"], max_length=max_length,
                                   padding="max_length", truncation=True)["input_ids"]
            model_inputs["labels"] = labels
            return model_inputs
        
        # 应用分词
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 转换为PyTorch张量
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # 提取输入和标签
        input_ids = tokenized_dataset["input_ids"]
        attention_mask = tokenized_dataset["attention_mask"]
        labels = tokenized_dataset["labels"]
        
        return input_ids, attention_mask, labels

# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        "total_data_limit": 50000,  # 控制总数据量
        "model_name": "bert-base-uncased"
    }
    
    # 创建增强器
    enhancer = DataEnhancer(config)
    enhancer.load_tokenizer()
    
    # 添加多个数据集
    datasets_to_load = [
        {"name": "imdb", "split": "train", "text_column": "text", "label_column": "label", "weight": 2.0, "max_samples": 20000},
        {"name": "glue/sst2", "split": "train", "text_column": "sentence", "label_column": "label", "weight": 1.0, "max_samples": 10000},
        {"name": "ag_news", "split": "train", "text_column": "text", "label_column": "label", "weight": 1.5, "max_samples": 15000},
    ]
    
    for dataset_config in datasets_to_load:
        enhancer.add_dataset(**dataset_config)
    
    # 平衡采样
    balanced_dataset = enhancer.balance_and_sample(method="weighted")
    
    # 预处理
    input_ids, attention_mask, labels = enhancer.preprocess_for_training(balanced_dataset, max_length=32)
    
    print(f"\n预处理完成:")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  注意力掩码形状: {attention_mask.shape}")
    print(f"  标签形状: {labels.shape}")
