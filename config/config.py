import os
import yaml
from typing import Dict, Any

class ConfigManager:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化"""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "project": {
                "name": "linai plus",
                "description": "增强版AI系统"
            },
            "data": {
                "source": "local",
                "data_type": "text",
                "output_path": "data",
                "hf": {
                    "dataset_name": None,  # Hugging Face数据集名称，例如: "imdb"
                    "split": "train",      # 数据集分割，例如: "train", "test", "validation"
                    "data_column": "text",  # 数据列名称，例如: "text", "sentence"
                    "api_token": None,      # Hugging Face API令牌，用于私有数据集或提高API限制
                    "use_auth_token": False # 是否使用身份验证令牌（用于私有数据集）
                },
                "image": {
                    "size": [224, 224],
                    "grayscale": False,
                    "formats": ["jpg", "jpeg", "png", "bmp"]
                },
                "video": {
                    "fps": 30,
                    "size": [224, 224],
                    "num_frames": 16,
                    "formats": ["mp4", "avi", "mov", "mkv"]
                }
            },
            "preprocess": {
                "max_length": 256,
                "remove_stopwords": True,
                "lower_case": True,
                "image": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "augment": True
                },
                "video": {
                    "sampling_method": "uniform",
                    "augment": True
                }
            },
            "model": {
                "model_type": "transformer",
                "num_epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.0005,
                "image": {
                    "architecture": "vision_transformer",
                    "hidden_dim": 768,
                    "classifier_dim": 256
                },
                "video": {
                    "architecture": "transformer",
                    "hidden_dim": 768,
                    "classifier_dim": 256,
                    "time_steps": 16
                },
                "memory": {
                    "enabled": True,
                    "memory_dir": "data/memory",
                    "max_memory_size": 10000,
                    "embedding_dim": 256,
                    "similarity_threshold": 0.7
                }
            }
        }