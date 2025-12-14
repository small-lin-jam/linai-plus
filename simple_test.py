import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.abspath('d:/linai'))

from client.model_loader import ModelLoader

# 模型路径
model_path = "D:\linai\output\models\model.pt"

# 加载模型
print("加载模型中...")
model_loader = ModelLoader(model_path)
print(f"模型加载成功！")
print(f"模型类型: {model_loader.model_type}")
print(f"数据类型: {model_loader.data_type}")
print(f"配置信息: {model_loader.config}")

# 检查词汇表
print(f"\n词汇表检查:")
print(f"config中的vocab存在: {'vocab' in model_loader.config}")
print(f"self.vocab存在: {hasattr(model_loader, 'vocab') and model_loader.vocab is not None}")

# 测试文本生成
print("\n测试文本生成:")
input_texts = ["你好", "今天天气真好", "我想学习", "这是一个测试"]

for input_text in input_texts:
    print(f"\n输入: {input_text}")
    
    # 预处理输入
    preprocessed = model_loader.preprocess_input(input_text)
    print(f"预处理结果: {preprocessed}")
    
    # 直接调用模型的generate_with_emotion方法
    vocab = model_loader.config.get('vocab', {})
    max_length = model_loader.config.get('max_length', 100)
    
    # 将文本转换为索引
    vector = [vocab.get(token, vocab.get('<UNK>', 1)) for token in preprocessed[0]]
    
    # 截断或填充到最大长度
    if len(vector) < max_length:
        vector += [vocab.get('<PAD>', 0)] * (max_length - len(vector))
    else:
        vector = vector[:max_length]
    
    input_tensor = torch.tensor([vector], dtype=torch.long)
    
    # 使用正确的特殊标记
    start_token = 2  # <SOS>
    end_token = 3    # <EOS>
    
    # 尝试不同的生成参数
    for temperature in [0.5, 1.0, 1.5]:
        for top_k in [5, 10, 20]:
            print(f"\n参数: temperature={temperature}, top_k={top_k}")
            generated_ids = model_loader.model.generate_with_emotion(
                input_tensor, 
                emotion=0, 
                start_token=start_token, 
                end_token=end_token,
                temperature=temperature,
                top_k=top_k
            )
            print(f"生成的ID序列: {generated_ids[0].tolist()}")
            
            # 转换为文本
            if hasattr(model_loader, 'id_to_text'):
                generated_text = model_loader.id_to_text(generated_ids[0].tolist())
                print(f"转换后的文本: '{generated_text}'")

# 不需要关闭模型，因为PyTorch会自动管理资源
