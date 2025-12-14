import torch
import yaml
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_def import TransformerSeq2Seq

def test_multitoken_prediction():
    """测试多token预测功能"""
    print("=== 测试多token预测功能 ===")
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # 创建一个小型测试模型
    vocab_size = 10002
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=model_config.get('embedding_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.1),
        max_length=model_config.get('max_length', 128),
        use_lora=model_config.get('use_lora', False),
        lora_r=model_config.get('lora_r', 8),
        lora_alpha=model_config.get('lora_alpha', 16),
        lora_dropout=model_config.get('lora_dropout', 0.05)
    )
    
    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 测试用的输入序列
    input_seq = torch.tensor([[1, 2, 3, 4, 5]]).to(device)  # 简单的输入序列
    emotion_label = torch.tensor([0]).to(device)  # 测试情感标签
    
    print(f"输入序列: {input_seq}")
    print(f"设备: {device}")
    print(f"模型是否使用LoRA: {model.use_lora}")
    
    # 测试不同的tokens_per_step值
    for tokens_per_step in [1, 2, 3]:
        print(f"\n--- 测试 tokens_per_step = {tokens_per_step} ---")
        try:
            # 生成序列
            generated = model.generate(
                input_seq, 
                emotion_label, 
                max_length=20,
                tokens_per_step=tokens_per_step
            )
            
            print(f"生成结果: {generated}")
            print(f"生成序列长度: {len(generated[0])}")
            
        except Exception as e:
            print(f"生成时出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_multitoken_prediction()
