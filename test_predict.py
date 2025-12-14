import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.abspath('d:/linai'))

from client.model_loader import ModelLoader

def test_model_predict():
    """测试模型预测功能"""
    # 模型路径
    model_path = (input("请输入模型路径: ") or "D:\\linai\\output\\models\\model.pt").strip('"').strip("'")
    
    try:
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
        
        if hasattr(model_loader, 'vocab') and model_loader.vocab is not None:
            vocab = model_loader.vocab
            print(f"vocab长度: {len(vocab)}")
            # 打印前10个词汇表条目
            print(f"vocab前10项: {list(vocab.items())[:10]}")
            
            # 检查特殊标记
            special_tokens = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
            print(f"\n特殊标记检查:")
            for token, expected_id in special_tokens.items():
                if token in vocab:
                    print(f"{token}: {vocab[token]} (预期: {expected_id})")
                else:
                    print(f"{token}: 未找到")
        elif 'vocab' in model_loader.config:
            vocab = model_loader.config['vocab']
            print(f"vocab长度: {len(vocab)}")
            print(f"vocab前10项: {list(vocab.items())[:10]}")
        
        # 测试输入
        test_inputs = ["你好", "今天天气怎么样", "中国的首都是哪里"]
        
        for test_input in test_inputs:
            print(f"\n{'='*50}")
            print(f"测试输入: {test_input}")
            print(f"{'='*50}")
            
            # 预处理输入
            preprocessed = model_loader.preprocess_input(test_input)
            print(f"预处理结果: {preprocessed}")
            
            # 如果是文本数据，展示向量化过程
            if model_loader.data_type == 'text':
                vocab = getattr(model_loader, 'vocab', model_loader.config.get('vocab', {}))
                max_length = model_loader.config.get('max_length', 100)
                if vocab:
                    vector = [vocab.get(token, vocab.get('<UNK>', 1)) for token in preprocessed[0]]
                    if len(vector) < max_length:
                        vector += [vocab.get('<PAD>', 0)] * (max_length - len(vector))
                    else:
                        vector = vector[:max_length]
                    print(f"向量化结果: {vector[:20]}... (长度: {len(vector)})")
            
            # 执行预测
            print("\n执行预测...")
            result = model_loader.predict(test_input)
            print(f"预测结果: {result}")
            
            # 检查生成文本
            if 'generated_text' in result:
                print(f"生成文本: {result['generated_text']}")
            if 'emotional_text' in result:
                print(f"情感化文本: {result['emotional_text']}")
            if 'reply' in result:
                print(f"最终回复: {result['reply']}")
        
        # 测试不同情感
        print(f"\n{'='*50}")
        print(f"测试不同情感")
        print(f"{'='*50}")
        
        test_input = "你好"
        emotions = {0: "积极", 1: "消极", 2: "愤怒", 3: "惊讶", 4: "中性"}
        
        for emotion_id, emotion_name in emotions.items():
            result = model_loader.predict(test_input, emotion=emotion_id)
            print(f"\n{emotion_name}情感回复: {result.get('emotional_text', '无')}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_predict()