import os
import sys
import yaml
import torch
import pandas as pd
from datasets import Dataset

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 测试数据增强功能
def test_data_enhancer():
    print("\n=== 测试数据增强功能 ===")
    
    try:
        from data.data_enhancer import DataEnhancer
        
        # 配置
        config = {
            "total_data_limit": 1000,  # 小批量测试
            "tokenizer_name": "bert-base-uncased"
        }
        
        # 创建增强器
        enhancer = DataEnhancer(config)
        enhancer.load_tokenizer()
        
        # 使用本地IMDB数据集测试
        print("使用本地IMDB数据集测试...")
        
        # 模拟数据集结构
        mock_dataset = Dataset.from_dict({
            "text": ["This movie is great!", "I hated this film.", "An amazing experience.", "Terrible acting.", "Best movie ever!"] * 200,
            "label": [1, 0, 1, 0, 1] * 200
        })
        
        # 直接添加到增强器
        enhancer.datasets.append({"name": "mock_imdb", "data": mock_dataset, "text_column": "text", "label_column": "label"})
        enhancer.dataset_weights.append(1.0)
        
        # 平衡采样
        balanced_dataset = enhancer.balance_and_sample(method="weighted")
        print(f"采样后数据集大小: {len(balanced_dataset)}")
        
        # 预处理
        input_ids, attention_mask, labels = enhancer.preprocess_for_training(balanced_dataset, max_length=16)
        
        print(f"预处理完成:")
        print(f"  输入数量: {len(input_ids)}")
        print(f"  注意力掩码数量: {len(attention_mask)}")
        print(f"  标签数量: {len(labels)}")
        
        # 检查第一个样本的形状
        print(f"  单个输入形状: {input_ids[0].shape if hasattr(input_ids[0], 'shape') else 'N/A'}")
        
        print("数据增强功能测试通过！")
        return True
        
    except Exception as e:
        print(f"数据增强功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 测试知识蒸馏功能
def test_knowledge_distillation():
    print("\n=== 测试知识蒸馏功能 ===")
    
    try:
        from train.trainer import KnowledgeDistiller
        import torch.nn as nn
        
        # 创建简单的教师和学生模型
        class SimpleModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 创建模型
        input_dim = 10
        hidden_dim_teacher = 20  # 教师模型更复杂
        hidden_dim_student = 10  # 学生模型更简单
        output_dim = 5
        
        teacher_model = SimpleModel(input_dim, hidden_dim_teacher, output_dim)
        student_model = SimpleModel(input_dim, hidden_dim_student, output_dim)
        
        # 创建蒸馏器
        distiller = KnowledgeDistiller(student_model, teacher_model, temperature=3.0, alpha=0.5)
        
        # 测试蒸馏损失计算
        batch_size = 8
        x = torch.randn(batch_size, input_dim)
        labels = torch.randint(0, output_dim, (batch_size,))
        
        student_outputs = student_model(x)
        teacher_outputs = teacher_model(x)
        
        loss = distiller.compute_distillation_loss(student_outputs, teacher_outputs, labels)
        
        print(f"蒸馏损失: {loss.item()}")
        print(f"学生模型输出形状: {student_outputs.shape}")
        print(f"教师模型输出形状: {teacher_outputs.shape}")
        
        print("知识蒸馏功能测试通过！")
        return True
        
    except Exception as e:
        print(f"知识蒸馏功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 测试配置文件加载
def test_config():
    print("\n=== 测试配置文件 ===")
    
    try:
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("配置文件加载成功")
        
        # 检查数据增强配置
        if 'enhanced' in config['data']['source']:
            print("数据增强配置:")
            print(f"  总数据限制: {config['data']['total_data_limit']}")
            print(f"  采样方法: {config['data']['sampling_method']}")
            print(f"  数据集数量: {len(config['data']['datasets'])}")
        
        # 检查知识蒸馏配置
        if config['model'].get('use_distillation', False):
            print("知识蒸馏配置:")
            print(f"  温度: {config['model']['distillation_temperature']}")
            print(f"  Alpha: {config['model']['distillation_alpha']}")
            print(f"  教师模型: {config['model']['teacher_model_name']}")
        
        print("配置文件测试通过！")
        return True
        
    except Exception as e:
        print(f"配置文件测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试增强功能...")
    
    # 运行所有测试
    test_config()
    test_data_enhancer()
    test_knowledge_distillation()
    
    print("\n=== 所有测试完成 ===")
