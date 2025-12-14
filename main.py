import sys
import os
import torch
import traceback
import datasets

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 限制PyTorch的GPU内存使用，为系统预留30%内存
torch.cuda.set_per_process_memory_fraction(0.7)

from config.config import ConfigManager
from data.data_fetcher import DataFetcher
from preprocess.preprocessor import DataPreprocessor
from models.model_def import ModelFactory
from train.trainer import ModelTrainer

def main():
    """主程序入口"""
    print("===== 全自动AI模型训练系统 ======")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"程序根目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    try:
        # 1. 加载配置
        print("1. 加载配置...")
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")
        print(f"配置文件路径: {config_path}")
        print(f"配置文件是否存在: {os.path.exists(config_path)}")
        if os.path.exists(config_path):
            print(f"配置文件权限: {oct(os.stat(config_path).st_mode)[-3:]}")
        
        config_manager = ConfigManager()
        config = config_manager.config
        print(f"项目名称: {config['project']['name']}")
        print(f"数据来源: {config['data']['source']}")
        
        # 2. 获取数据
        print("\n2. 获取数据...")
        data = []
        preprocessed_data = []
        train_loader = None
        data_config = {}
        
        if config['data']['source'] == "web":
            print("从网络获取数据...")
            data_fetcher = DataFetcher(config)
            data = data_fetcher.fetch_from_web()
        elif config['data']['source'] == "local":
            print("从本地获取数据...")
            data_fetcher = DataFetcher(config)
            data_path = config['data'].get('local', {}).get('path', 'data')
            absolute_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
            print(f"本地数据路径: {absolute_data_path}")
            print(f"本地数据路径是否存在: {os.path.exists(absolute_data_path)}")
            if os.path.exists(absolute_data_path):
                print(f"本地数据路径权限: {oct(os.stat(absolute_data_path).st_mode)[-3:]}")
            data = data_fetcher.fetch_from_local()
        elif config['data']['source'] == "hf":
            print("从Hugging Face获取数据...")
            data_fetcher = DataFetcher(config)
            # 检查是否配置了API令牌
            hf_config = config['data'].get('hf', {})
            if hf_config.get('use_auth_token') and not hf_config.get('api_token'):
                print("提示: 如果需要访问私有Hugging Face数据集，请在配置文件中设置hf.api_token")
                print("您可以将config/config.yaml.example重命名为config.yaml并填写您的API令牌")
            data = data_fetcher.fetch_from_hf()
        elif config['data']['source'] == "enhanced":
            print("使用数据增强器加载多个数据集...")
            from data.data_enhancer import DataEnhancer
            
            enhancer = DataEnhancer(config['data'])
            
            # 加载分词器
            tokenizer = enhancer.load_tokenizer(model_name=config['data'].get('tokenizer_name', 'bert-base-uncased'))
            
            # 添加配置的数据集
            datasets_config = config['data'].get('datasets', [])
            for dataset_config in datasets_config:
                enhancer.add_dataset(
                    dataset_name=dataset_config['name'],
                    split=dataset_config.get('split', 'train'),
                    text_column=dataset_config.get('text_column', 'text'),
                    label_column=dataset_config.get('label_column', 'label'),
                    weight=dataset_config.get('weight', 1.0),
                    max_samples=dataset_config.get('max_samples', None)
                )
            
            # 平衡采样数据集
            balanced_dataset = enhancer.balance_and_sample(method=config['data'].get('sampling_method', 'weighted'))
            
            # 数据预处理
            input_ids, attention_mask, labels = enhancer.preprocess_for_training(
                balanced_dataset, 
                max_length=config.get('preprocess', {}).get('max_length', 32)
            )
            
            # 创建预处理后的数据格式（假设与DataPreprocessor的输出兼容）
            preprocessed_data = list(zip(input_ids, attention_mask, labels))
            print(f"预处理后的数据数量: {len(preprocessed_data)}")
            
            # 准备训练数据
            print("\n4. 准备训练数据...")
            trainer = ModelTrainer(config)
            train_loader, data_config = trainer.prepare_data(preprocessed_data)
        
        # 处理非增强数据源
        if not train_loader and config['data']['source'] != "enhanced":
            if not data:
                print("未获取到数据，程序退出。")
                return
            
            print(f"获取到 {len(data)} 条数据")
            
            # 3. 预处理数据
            print("\n3. 预处理数据...")
            preprocessor = DataPreprocessor(config)
            preprocessed_data = preprocessor.preprocess(data)
            
            if not preprocessed_data:
                print("预处理后的数据为空，程序退出。")
                return
            
            print(f"预处理后的数据数量: {len(preprocessed_data)}")
            
            # 4. 准备训练数据
            print("\n4. 准备训练数据...")
            trainer = ModelTrainer(config)
            train_loader, data_config = trainer.prepare_data(preprocessed_data)
        
        vocab = data_config.get("vocab", {})
        print(f"词汇表大小: {len(vocab)}")
        print(f"训练批次数量: {len(train_loader)}")
        
        # 5. 创建模型
        print("\n5. 创建模型...")
        
        # 根据模型类型设置不同的配置
        model_type = config.get("model", {}).get("model_type", "transformer")
        
        if model_type == "seq2seq_transformer":
            model_config = {
                "vocab_size": len(vocab),
                "embedding_dim": 64,       # 进一步减少嵌入维度以提高速度
                "hidden_dim": 128,         # 进一步减少隐藏层维度以提高速度
                "max_length": config.get("preprocess", {}).get("max_length", 32),  # 保持与预处理一致的序列长度
                "num_heads": 2,            # 减少注意力头数量
                "num_layers": 1,           # 减少到1层Transformer以大幅提高速度
                "dropout": 0.05,           # 降低Dropout率，减少计算开销
                "num_emotions": 5          # 情感类别数量
            }
        else:
            model_config = {
                "vocab_size": len(vocab),
                "embedding_dim": 128,  # 减少嵌入维度以降低显存占用
                "hidden_dim": 256,      # 减少隐藏层维度以降低显存占用
                "num_classes": 2,
                "max_length": config.get("preprocess", {}).get("max_length", 128),
                "num_heads": 4,         # 减少Transformer头数量
                "num_layers": 2,        # 减少Transformer层数
                "dropout": 0.1          # Dropout率
            }
        
        # 添加LoRA参数（如果配置了）
        lora_config = config.get("model", {}).get("lora", {})
        if lora_config:
            model_config["lora"] = lora_config
        model = ModelFactory.create_model(model_type, model_config)
        print(f"模型类型: {type(model).__name__}")
        
        # 6. 开始训练...
        print("\n6. 开始训练...")
        
        # 检查是否使用知识蒸馏
        use_distillation = config['model'].get('use_distillation', False)
        teacher_model = None
        
        if use_distillation:
            print("加载教师模型用于知识蒸馏...")
            # 这里可以根据配置加载预训练的教师模型
            # 例如：from transformers import AutoModelForSequenceClassification
            # teacher_model = AutoModelForSequenceClassification.from_pretrained(config['model']['teacher_model_name'])
            
        # 训练模型（支持知识蒸馏）
        history = trainer.train(
            model, 
            train_loader,
            use_distillation=use_distillation,
            teacher_model=teacher_model,
            temperature=config['model'].get('distillation_temperature', 5.0),
            alpha=config['model'].get('distillation_alpha', 0.7)
        )
        
        # 7. 保存模型
        print("\n7. 保存模型...")
        output_path = "output/models"
        absolute_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
        print(f"模型输出路径: {absolute_output_path}")
        print(f"模型输出路径是否存在: {os.path.exists(absolute_output_path)}")
        if os.path.exists(absolute_output_path):
            print(f"模型输出路径权限: {oct(os.stat(absolute_output_path).st_mode)[-3:]}")
        trainer.save_model(model, config, vocab)
        
        print("\n===== 训练完成！=====")
        
    except PermissionError as e:
        print(f"\n发生PermissionError错误: {e}")
        print(f"错误文件路径: {e.filename}")
        print(f"错误操作: {e.strerror}")
        print("\n详细错误信息:")
        traceback.print_exc()
    except Exception as e:
        print(f"\n发生其他错误: {type(e).__name__}: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main()