import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocessor import DataPreprocessor
from models.model_def import ModelFactory
from data.data_fetcher import DataFetcher
import yaml
import os
import argparse
from tqdm import tqdm

# 简单的Seq2Seq数据集类
class Seq2SeqDataset(Dataset):
    def __init__(self, data, vocab, max_length=100):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token = vocab.get('<PAD>', 0)
        self.unk_token = vocab.get('<UNK>', 1)
        self.sos_token = vocab.get('<SOS>', 2) if '<SOS>' in vocab else 2
        self.eos_token = vocab.get('<EOS>', 3) if '<EOS>' in vocab else 3
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # 将源序列转换为索引
        src_tokens = src.split()[:self.max_length-2]  # 留出空间给SOS和EOS
        src_indices = [self.vocab.get(token, self.unk_token) for token in src_tokens]
        src_indices = [self.sos_token] + src_indices + [self.eos_token]
        
        # 将目标序列转换为索引
        tgt_tokens = tgt.split()[:self.max_length-2]  # 留出空间给SOS和EOS
        tgt_indices = [self.vocab.get(token, self.unk_token) for token in tgt_tokens]
        tgt_indices = [self.sos_token] + tgt_indices + [self.eos_token]
        
        # 填充到最大长度
        src_indices = src_indices + [self.pad_token] * (self.max_length - len(src_indices))
        tgt_indices = tgt_indices + [self.pad_token] * (self.max_length - len(tgt_indices))
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def build_vocab(data, vocab_size=10000):
    """构建词汇表"""
    from collections import Counter
    
    word_counts = Counter()
    for src, tgt in data:
        word_counts.update(src.split())
        word_counts.update(tgt.split())
    
    # 保留出现频率最高的词汇
    most_common = word_counts.most_common(vocab_size - 4)  # 留出空间给特殊标记
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab

def main():
    parser = argparse.ArgumentParser(description='训练TransformerSeq2Seq模型')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output_path', type=str, default='output/models/seq2seq_model.pt', help='模型输出路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取训练数据
    # 这里使用IMDB数据集的样本来创建简单的问答对
    # 实际应用中应该使用真实的对话数据集
    print('加载训练数据...')
    data_fetcher = DataFetcher(config)
    
    # 创建简单的训练数据
    # 实际应用中应该使用真实的对话数据集
    train_data = []
    
    # 示例数据：问题-回答对
    sample_pairs = [
        ("你好", "你好！我是你的AI助手，很高兴为你服务。"),
        ("今天天气怎么样", "今天天气很好，适合外出活动。"),
        ("你会做什么", "我可以回答你的问题，提供信息和帮助。"),
        ("谢谢", "不客气，有什么问题随时问我。"),
        ("再见", "再见！祝你有美好的一天。"),
        ("你是谁", "我是一个基于Transformer的AI助手。"),
        ("世界上最高的山峰是什么", "世界上最高的山峰是珠穆朗玛峰，海拔约8848米。"),
        ("中国的首都是哪里", "中国的首都是北京。"),
        ("水的化学式是什么", "水的化学式是H₂O。"),
        ("地球的年龄大约是多少", "地球的年龄大约是46亿年。")
    ]
    
    # 重复示例数据以增加训练量
    for _ in range(100):
        train_data.extend(sample_pairs)
    
    print(f'训练数据大小: {len(train_data)}')
    
    # 构建词汇表
    print('构建词汇表...')
    vocab = build_vocab(train_data, vocab_size=10000)
    vocab_size = len(vocab)
    print(f'词汇表大小: {vocab_size}')
    
    # 创建数据集和数据加载器
    max_length = config.get('model', {}).get('max_length', 100)
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    train_dataset = Seq2SeqDataset(train_data, vocab, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    print('创建模型...')
    model_config = {
        'model_type': 'transformer_seq2seq',
        'vocab_size': vocab_size,
        'embedding_dim': config.get('model', {}).get('embedding_dim', 256),
        'hidden_dim': config.get('model', {}).get('hidden_dim', 512),
        'num_heads': config.get('model', {}).get('num_heads', 8),
        'num_layers': config.get('model', {}).get('num_layers', 4),
        'dropout': config.get('model', {}).get('dropout', 0.1),
        'max_length': max_length,
        'num_emotions': 5
    }
    
    model_factory = ModelFactory()
    model = model_factory.create_model('transformer_seq2seq', model_config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 设置优化器和损失函数
    lr = config.get('training', {}).get('learning_rate', 0.0005)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get('<PAD>', 0))
    
    # 训练模型
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    
    print(f'开始训练，使用设备: {device}')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for src_batch, tgt_batch in train_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                # 准备目标序列的输入和输出
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(src_batch, tgt_input)
                
                # 计算损失
                loss = criterion(outputs.view(-1, vocab_size), tgt_output.view(-1))
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # 保存模型
    print(f'保存模型到 {args.output_path}')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'vocab': vocab,
        'data_type': 'text'
    }
    
    torch.save(checkpoint, args.output_path)
    
    print('训练完成！')

if __name__ == '__main__':
    main()
