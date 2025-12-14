from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from .memory_module import AIMemoryModule, MemoryEnhancedModel

# LoRA实现（Low-Rank Adaptation）
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16, dropout=0.0):
        """初始化LoRA层
        
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            r: LoRA秩
            alpha: 缩放因子
            dropout: dropout率
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        
        # LoRA参数
        self.A = nn.Parameter(torch.zeros((r, in_dim)))
        self.B = nn.Parameter(torch.zeros((out_dim, r)))
        
        # 初始化参数
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scaling = alpha / r
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, in_dim)
            
        Returns:
            LoRA输出 (batch_size, seq_len, out_dim)
        """
        # 转置以适应矩阵乘法 (batch_size, in_dim, seq_len)
        x = x.transpose(1, 2)
        
        # LoRA前向传播
        x = self.dropout(x)
        x = (self.B @ (self.A @ x)) * self.scaling
        
        # 转置回原始形状 (batch_size, seq_len, out_dim)
        return x.transpose(1, 2)

# 实现位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int = 512):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TransformerTextClassifier(nn.Module):
    """基于Transformer的文本分类模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 4, num_classes: int = 2, 
                 dropout: float = 0.1, max_length: int = 256):
        """初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量
            num_layers: Transformer层数
            num_classes: 类别数量
            dropout: dropout率
            max_length: 最大序列长度
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length)
        
        # Transformer编码器层，设置batch_first=True以提高性能
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                                  dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len)
            
        Returns:
            模型输出
        """
        # 词嵌入 (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Transformer编码（batch_first=True，输入输出都是 (batch_size, seq_len, embedding_dim)）
        encoded = self.transformer_encoder(embedded)
        sentence_rep = torch.mean(encoded, dim=1)
        
        # 分类
        out = self.fc(sentence_rep)
        
        return out

class SimpleTextClassifier(nn.Module):
    """简单的文本分类模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        """初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数量
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        #  dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        # 词嵌入
        embedded = self.embedding(x)
        
        # 计算句子嵌入（取平均值）
        embedded = torch.mean(embedded, dim=1)
        
        # 全连接层
        out = self.fc1(embedded)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class CNNImageClassifier(nn.Module):
    """基于CNN的图像分类模型"""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """初始化模型
        
        Args:
            num_classes: 类别数量
            dropout: dropout率
        """
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)  # 假设输入图像大小为224x224
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, C, H, W)
            
        Returns:
            模型输出
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # 展平
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class VisionTransformer(nn.Module):
    """基于Vision Transformer的图像分类模型"""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embedding_dim: int = 768, num_heads: int = 12, num_layers: int = 12, 
                 num_classes: int = 2, dropout: float = 0.1):
        """初始化模型
        
        Args:
            image_size: 输入图像大小
            patch_size: Patch大小
            in_channels: 输入通道数
            embedding_dim: 嵌入维度
            num_heads: 注意力头数量
            num_layers: Transformer层数
            num_classes: 类别数量
            dropout: dropout率
        """
        super().__init__()
        
        # 检查图像大小是否能被patch大小整除
        assert image_size % patch_size == 0, "图像大小必须能被patch大小整除"
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embedding = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                                  dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, C, H, W)
            
        Returns:
            模型输出
        """
        # Patch嵌入
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加class token
        batch_size = x.size(0)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 分类
        x = x[:, 0]  # 使用class token的输出
        x = self.fc(x)
        
        return x

class CNN3DVideoClassifier(nn.Module):
    """基于3D CNN的视频分类模型"""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """初始化模型
        
        Args:
            num_classes: 类别数量
            dropout: dropout率
        """
        super().__init__()
        
        # 3D卷积层
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # 3D池化层
        self.pool = nn.MaxPool3d((2, 2, 2))
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 14 * 14, 1024)  # 假设输入视频大小为 (16, 224, 224)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, F, C, H, W) -> 转换为 (batch_size, C, F, H, W)
            
        Returns:
            模型输出
        """
        # 转换通道顺序
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # 展平
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class TransformerVideoClassifier(nn.Module):
    """基于Transformer的视频分类模型"""
    
    def __init__(self, num_frames: int = 16, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embedding_dim: int = 768, num_heads: int = 12, 
                 num_layers: int = 12, num_classes: int = 2, dropout: float = 0.1):
        """初始化模型
        
        Args:
            num_frames: 输入帧数
            image_size: 输入图像大小
            patch_size: Patch大小
            in_channels: 输入通道数
            embedding_dim: 嵌入维度
            num_heads: 注意力头数量
            num_layers: Transformer层数
            num_classes: 类别数量
            dropout: dropout率
        """
        super().__init__()
        
        # 图像模型
        self.image_model = VisionTransformer(image_size, patch_size, in_channels, 
                                            embedding_dim, num_heads, num_layers, 
                                            dropout=dropout)
        
        # 时间Transformer
        self.time_transformer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                                          dim_feedforward=embedding_dim * 4, 
                                                          dropout=dropout, batch_first=True)
        
        # 分类头
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据 (batch_size, F, C, H, W)
            
        Returns:
            模型输出
        """
        batch_size, num_frames, C, H, W = x.size()
        
        # 处理每一帧
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            feature = self.image_model.patch_embedding(frame)
            feature = feature.flatten(2).transpose(1, 2)
            
            # 添加位置编码
            feature = feature + self.image_model.pos_embedding[:, 1:, :]
            feature = self.dropout(feature)
            
            # Transformer编码
            feature = self.image_model.transformer_encoder(feature)
            
            # 使用平均池化获取帧特征
            frame_feature = torch.mean(feature, dim=1)
            frame_features.append(frame_feature)
        
        # 堆叠帧特征 (batch_size, num_frames, embedding_dim)
        video_features = torch.stack(frame_features, dim=1)
        
        # 时间Transformer编码
        video_features = self.time_transformer(video_features)
        
        # 时间平均池化
        video_features = torch.mean(video_features, dim=1)
        
        # 分类
        out = self.fc(video_features)
        
        return out

class ModelFactory:
    """模型工厂类，用于创建不同类型的模型"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """创建模型
        
        Args:
            model_type: 模型类型
            config: 模型配置
            
        Returns:
            创建的模型
        """
        if model_type == "simple_classifier":
            # 优先使用config中的vocab_size字段
            vocab_size = config.get("vocab_size", 10000)
            print(f"使用vocab_size: {vocab_size}")
            embedding_dim = config.get("embedding_dim", 128)
            hidden_dim = config.get("hidden_dim", 256)
            num_classes = config.get("num_classes", 2)
            
            return SimpleTextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
        elif model_type == "transformer":
            # 优先使用config中的vocab_size字段
            vocab_size = config.get("vocab_size", 10000)
            print(f"使用vocab_size: {vocab_size}")
            embedding_dim = config.get("embedding_dim", 256)
            hidden_dim = config.get("hidden_dim", 512)
            num_heads = config.get("num_heads", 8)
            num_layers = config.get("num_layers", 4)
            num_classes = config.get("num_classes", 2)
            dropout = config.get("dropout", 0.1)
            max_length = config.get("max_length", 256)
            
            return TransformerTextClassifier(vocab_size, embedding_dim, hidden_dim, 
                                            num_heads, num_layers, num_classes, dropout, max_length)
        elif model_type == "seq2seq_transformer":
            # 优先使用config中的vocab_size字段
            vocab_size = config.get("vocab_size", 10000)
            print(f"使用vocab_size: {vocab_size}")
            embedding_dim = config.get("embedding_dim", 128)
            hidden_dim = config.get("hidden_dim", 256)
            num_heads = config.get("num_heads", 4)
            num_layers = config.get("num_layers", 2)
            dropout = config.get("dropout", 0.1)
            max_length = config.get("max_length", 128)
            num_emotions = config.get("num_emotions", 5)
            
            # LoRA配置
            lora_config = config.get("lora", {})
            use_lora = lora_config.get("use_lora", False)
            lora_r = lora_config.get("lora_r", 8)
            lora_alpha = lora_config.get("lora_alpha", 16)
            lora_dropout = lora_config.get("lora_dropout", 0.05)
            
            return TransformerSeq2Seq(vocab_size, embedding_dim, hidden_dim, 
                                     num_heads, num_layers, dropout, max_length, 
                                     num_emotions, use_lora, lora_r, lora_alpha, lora_dropout)
        elif model_type == "cnn_image":
            num_classes = config.get("num_classes", 2)
            dropout = config.get("dropout", 0.5)
            
            return CNNImageClassifier(num_classes, dropout)
        elif model_type == "vit":
            image_size = config.get("image_size", 224)
            patch_size = config.get("patch_size", 16)
            in_channels = config.get("in_channels", 3)
            embedding_dim = config.get("embedding_dim", 768)
            num_heads = config.get("num_heads", 12)
            num_layers = config.get("num_layers", 12)
            num_classes = config.get("num_classes", 2)
            dropout = config.get("dropout", 0.1)
            
            return VisionTransformer(image_size, patch_size, in_channels, 
                                    embedding_dim, num_heads, num_layers, 
                                    num_classes, dropout)
        elif model_type == "cnn3d_video":
            num_classes = config.get("num_classes", 2)
            dropout = config.get("dropout", 0.5)
            
            return CNN3DVideoClassifier(num_classes, dropout)
        elif model_type == "transformer_video":
            num_frames = config.get("num_frames", 16)
            image_size = config.get("image_size", 224)
            patch_size = config.get("patch_size", 16)
            in_channels = config.get("in_channels", 3)
            embedding_dim = config.get("embedding_dim", 768)
            num_heads = config.get("num_heads", 12)
            num_layers = config.get("num_layers", 12)
            num_classes = config.get("num_classes", 2)
            dropout = config.get("dropout", 0.1)
            
            return TransformerVideoClassifier(num_frames, image_size, patch_size, 
                                            in_channels, embedding_dim, num_heads, 
                                            num_layers, num_classes, dropout)
        elif model_type == "seq2seq_transformer":
            # 优先使用config中的vocab_size字段
            vocab_size = config.get("vocab_size", 10000)
            print(f"使用vocab_size: {vocab_size}")
            embedding_dim = config.get("embedding_dim", 256)
            hidden_dim = config.get("hidden_dim", 512)
            num_heads = config.get("num_heads", 8)
            num_layers = config.get("num_layers", 4)
            dropout = config.get("dropout", 0.1)
            max_length = config.get("max_length", 256)
            num_emotions = config.get("num_emotions", 5)
            
            return TransformerSeq2Seq(vocab_size, embedding_dim, hidden_dim, 
                                     num_heads, num_layers, dropout, max_length, 
                                     num_emotions)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class TransformerSeq2Seq(nn.Module):
    """基于Transformer的Seq2Seq模型，用于问答和文本生成任务，支持情感控制、LoRA和高效多token预测"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, max_length: int = 128, 
                 num_emotions: int = 5, use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        """初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量
            num_layers: Transformer层数
            dropout: dropout率
            max_length: 最大序列长度
            num_emotions: 情感类别数量
            use_lora: 是否使用LoRA技术
            lora_r: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: LoRA dropout率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_emotions = num_emotions
        self.use_lora = use_lora
        
        # 词嵌入层（共享编码器和解码器）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 情感嵌入层
        self.emotion_embedding = nn.Embedding(num_emotions, embedding_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                                  dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                                  dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 梯度检查点支持
        self.use_gradient_checkpointing = False
        
        # LoRA适配器
        if self.use_lora:
            print(f"启用LoRA技术，r={lora_r}, alpha={lora_alpha}")
            # 编码器输出LoRA
            self.encoder_lora = LoRALayer(embedding_dim, embedding_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            # 解码器输出LoRA
            self.decoder_lora = LoRALayer(embedding_dim, embedding_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        
        # 输出层
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 情感到词汇的映射表（用于添加情感词汇）
        self.emotion_vocab = {
            0: ["开心", "快乐", "高兴", "愉快", "愉悦", "兴奋", "欣喜", "满意", "欢乐", "雀跃", "喜悦", "欢欣"],  # 积极
            1: ["悲伤", "难过", "伤心", "痛苦", "沮丧", "失落", "不幸", "悲哀", "悲痛", "忧郁", "哀伤", "伤感"],  # 消极
            2: ["愤怒", "生气", "恼火", "恼怒", "气愤", "暴怒", "愤恨", "愤慨", "震怒", "狂怒", "盛怒", "发火"],  # 愤怒
            3: ["惊讶", "吃惊", "震惊", "诧异", "意外", "愕然", "惊叹", "讶异", "惊奇", "惊愕", "骇然", "诧异"],  # 惊讶
            4: ["平静", "平和", "淡定", "冷静", "宁静", "沉稳", "泰然", "从容", "平和", "安详", "平稳", "镇静"]   # 中性
        }
    
    def forward(self, src, tgt, emotion=None):
        """前向传播（训练时使用）
        
        Args:
            src: 源序列 (batch_size, src_seq_len)
            tgt: 目标序列 (batch_size, tgt_seq_len)
            emotion: 情感标签 (batch_size,)
            
        Returns:
            模型输出 (batch_size, tgt_seq_len, vocab_size)
        """
        # 词嵌入和位置编码
        src_embedded = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = self.dropout(src_embedded)
        
        tgt_embedded = self.embedding(tgt) * np.sqrt(self.embedding.embedding_dim)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # 如果提供了情感标签，将情感嵌入添加到源序列和解码器输入中
        if emotion is not None:
            emotion_embedded = self.emotion_embedding(emotion).unsqueeze(1)  # (batch_size, 1, embedding_dim)
            
            # 将情感嵌入添加到每个源序列的每个词嵌入中
            src_embedded = src_embedded + emotion_embedded
            
            # 将情感嵌入添加到解码器的每个输入词嵌入中
            tgt_embedded = tgt_embedded + emotion_embedded
        
        # 创建解码器的掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
        # Transformer编码
        if self.use_gradient_checkpointing:
            # 使用梯度检查点进行编码
            memory = checkpoint_sequential(
                self.transformer_encoder.layers, 
                len(self.transformer_encoder.layers) // 2 + 1, 
                src_embedded,
                use_reentrant=False  # 添加明确的use_reentrant参数
            )
        else:
            memory = self.transformer_encoder(src_embedded)
        
        # 应用LoRA到编码器输出
        if self.use_lora:
            memory = memory + self.encoder_lora(memory)
        
        # Transformer解码
        if self.use_gradient_checkpointing:
            # 使用梯度检查点进行解码
            # 为了解码器，我们需要创建一个包装函数
            def decoder_forward(decoder_input, memory):
                return self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)
            
            output = checkpoint(decoder_forward, tgt_embedded, memory, use_reentrant=False)  # 添加明确的use_reentrant参数
        else:
            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        
        # 应用LoRA到解码器输出
        if self.use_lora:
            output = output + self.decoder_lora(output)
        
        # 输出层
        output = self.fc(output)
        
        return output
    
    def generate(self, src, start_token=2, end_token=3, max_length=None, emotion=None, 
                temperature=0.7, top_k=50, top_p=0.9, sample_strategy="top_k", beam_size=1, tokens_per_step=1):
        """生成目标序列（推理时使用），支持情感控制
        
        Args:
            src: 源序列 (batch_size, src_seq_len)
            start_token: 开始标记的索引，默认<SOS>标记
            end_token: 结束标记的索引，默认<EOS>标记
            max_length: 最大生成长度
            emotion: 情感标签 (batch_size,) 或单个情感值
            temperature: 生成温度，控制输出的多样性
            top_k: top-k采样中保留的最高概率词数
            top_p: top-p采样中保留的累积概率阈值
            sample_strategy: 采样策略，可选值为 "top_k", "top_p", "greedy"
            beam_size: 束搜索的束大小，当beam_size>1时使用束搜索
            tokens_per_step: 每一步生成的token数量，增强多token预测能力
            
        Returns:
            生成的序列 (batch_size, generated_seq_len)
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = src.size(0)

        # 设置起始和结束标记
        start_token = 2  # <SOS> 开始标记
        end_token = 3    # <EOS> 结束标记

        # 如果束大小大于1，使用束搜索
        if beam_size > 1:
            return self._beam_search(src, start_token, end_token, max_length, emotion, 
                                    temperature, top_k, top_p, sample_strategy, beam_size)

        # 初始化生成序列
        generated = torch.full((batch_size, 1), int(start_token), device=src.device)
        
        # 词嵌入和位置编码
        src_embedded = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = self.dropout(src_embedded)
        
        # 如果提供了情感标签，将情感嵌入添加到源序列
        if emotion is not None:
            if isinstance(emotion, int):
                emotion = torch.tensor([emotion] * batch_size, device=src.device)
            elif not isinstance(emotion, torch.Tensor):
                emotion = torch.tensor(emotion, device=src.device)
            
            emotion_embedded = self.emotion_embedding(emotion).unsqueeze(1)  # (batch_size, 1, embedding_dim)
            # 将情感嵌入添加到每个源序列的每个词嵌入中
            src_embedded = src_embedded + emotion_embedded
        
        # Transformer编码
        memory = self.transformer_encoder(src_embedded)
        
        # 逐词生成（支持多token每步）
        current_length = 1
        while current_length < max_length:
            # 词嵌入和位置编码
            tgt_embedded = self.embedding(generated) * np.sqrt(self.embedding.embedding_dim)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            tgt_embedded = self.dropout(tgt_embedded)
            
            # 如果提供了情感标签，将情感嵌入添加到解码器输入中
            if emotion is not None:
                tgt_embedded = tgt_embedded + emotion_embedded
            
            # 创建解码器的掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=src.device)
            
            # Transformer解码
            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
            
            # 输出层
            output = self.fc(output[:, -1, :])
            
            # 使用温度缩放控制输出多样性
            output = output / temperature
            
            # 选择下一个词
            next_word = self._select_next_word(output, temperature, top_k, top_p, sample_strategy)
            
            # 将新生成的词添加到序列中
            generated = torch.cat([generated, next_word], dim=1)
            current_length += 1
            
            # 如果启用多token生成且还有空间，继续生成更多token
            for step in range(1, tokens_per_step):
                if current_length >= max_length:
                    break
                    
                # 词嵌入和位置编码（使用新扩展的tgt）
                tgt_embedded = self.embedding(generated) * np.sqrt(self.embedding.embedding_dim)
                tgt_embedded = self.pos_encoder(tgt_embedded)
                tgt_embedded = self.dropout(tgt_embedded)
                
                if emotion is not None:
                    tgt_embedded = tgt_embedded + emotion_embedded
                
                # 更新解码器掩码
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=src.device)
                
                # 再次解码
                output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                
                # 输出层
                output = self.fc(output[:, -1, :])
                
                # 温度缩放
                output = output / temperature
                
                # 选择下一个词
                next_word = self._select_next_word(output, temperature, top_k, top_p, sample_strategy)
                
                # 添加到目标序列
                generated = torch.cat([generated, next_word], dim=1)
                current_length += 1
            
            # 如果所有序列都生成了结束标记，则停止
            if (generated[:, -1] == end_token).all():
                break
        
        return generated
    
    def _select_next_word(self, output, temperature, top_k=50, top_p=0.9, sample_strategy="top_k"):
        """选择下一个词，支持不同的采样策略
        
        Args:
            output: 模型输出 (batch_size, vocab_size)
            temperature: 生成温度
            top_k: top-k采样中保留的最高概率词数
            top_p: top-p采样中保留的累积概率阈值
            sample_strategy: 采样策略，可选值为 "top_k", "top_p", "greedy"
            
        Returns:
            选择的词索引 (batch_size, 1)
        """
        if sample_strategy == "greedy":
            # 贪婪采样，选择概率最高的词
            next_word = torch.argmax(output, dim=-1).unsqueeze(1)
        else:
            # 温度缩放
            logits = output / temperature
            
            if sample_strategy == "top_k":
                # top-k采样
                filtered_logits = self._top_k_logits(logits, top_k)
            elif sample_strategy == "top_p":
                # top-p采样（核采样）
                filtered_logits = self._top_p_logits(logits, top_p)
            else:
                # 默认使用top-k
                filtered_logits = self._top_k_logits(logits, top_k)
            
            # 使用softmax获取概率分布
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            # 基于概率分布采样
            next_word = torch.multinomial(probabilities, 1)
        
        return next_word
    
    def _top_k_logits(self, logits, k):
        """保留概率最高的k个词，其余设为-∞
        
        Args:
            logits: 模型输出的logits (batch_size, vocab_size)
            k: 保留的词数量
            
        Returns:
            过滤后的logits
        """
        if k == 0 or k >= logits.size(-1):
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1).expand_as(logits)
        return torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)
    
    def _top_p_logits(self, logits, p):
        """保留累积概率达到p的最高概率词，其余设为-∞（核采样）
        
        Args:
            logits: 模型输出的logits (batch_size, vocab_size)
            p: 累积概率阈值
            
        Returns:
            过滤后的logits
        """
        if p >= 1.0:
            return logits
        
        # 对logits进行排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # 计算softmax概率
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 创建一个掩码，保留累积概率<=p的词，并且至少保留一个词
        mask = cumulative_probs <= p
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=-1)
        
        # 将不在掩码中的词的logits设为-∞
        filtered_logits = torch.full_like(logits, -float('inf'))
        filtered_logits.scatter_(1, sorted_indices, sorted_logits)
        filtered_logits = torch.where(mask.scatter(1, sorted_indices, mask), filtered_logits, torch.full_like(filtered_logits, -float('inf')))
        
        return filtered_logits
    
    def _beam_search(self, src, start_token=2, end_token=3, max_length=None, emotion=None, 
                    temperature=0.7, top_k=50, top_p=0.9, sample_strategy="top_k", beam_size=5):
        """使用束搜索生成目标序列
        
        Args:
            src: 源序列 (batch_size, src_seq_len)
            start_token: 开始标记的索引
            end_token: 结束标记的索引
            max_length: 最大生成长度
            emotion: 情感标签 (batch_size,) 或单个情感值
            temperature: 生成温度，控制输出的多样性
            top_k: top-k采样中保留的最高概率词数
            top_p: top-p采样中保留的累积概率阈值
            sample_strategy: 采样策略
            beam_size: 束大小
            
        Returns:
            生成的序列 (batch_size, generated_seq_len)
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = src.size(0)
        device = src.device
        
        # 初始化束
        beams = [{"sequence": torch.tensor([[start_token]], device=device), "score": 0.0} for _ in range(batch_size * beam_size)]
        
        # 词嵌入和位置编码
        src_embedded = self.embedding(src.repeat_interleave(beam_size, dim=0)) * np.sqrt(self.embedding.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = self.dropout(src_embedded)
        
        # 如果提供了情感标签，将情感嵌入添加到源序列
        if emotion is not None:
            if isinstance(emotion, int):
                emotion = torch.tensor([emotion] * batch_size, device=device)
            elif not isinstance(emotion, torch.Tensor):
                emotion = torch.tensor(emotion, device=device)
            
            emotion = emotion.repeat_interleave(beam_size, dim=0)
            emotion_embedded = self.emotion_embedding(emotion).unsqueeze(1)  # (batch_size*beam_size, 1, embedding_dim)
            # 将情感嵌入添加到每个源序列的每个词嵌入中
            src_embedded = src_embedded + emotion_embedded
        
        # Transformer编码
        memory = self.transformer_encoder(src_embedded)
        
        # 逐词生成
        for i in range(max_length - 1):
            candidates = []
            
            for beam_idx, beam in enumerate(beams):
                seq = beam["sequence"]
                score = beam["score"]
                
                # 如果序列已经结束，直接添加到候选列表
                if seq[0, -1] == end_token:
                    candidates.append(beam)
                    continue
                
                # 词嵌入和位置编码
                tgt_embedded = self.embedding(seq) * np.sqrt(self.embedding.embedding_dim)
                tgt_embedded = self.pos_encoder(tgt_embedded)
                tgt_embedded = self.dropout(tgt_embedded)
                
                # 如果提供了情感标签，将情感嵌入添加到解码器输入中
                if emotion is not None:
                    tgt_embedded = tgt_embedded + emotion_embedded[beam_idx:beam_idx+1]
                
                # 创建解码器的掩码
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
                
                # Transformer解码
                output = self.transformer_decoder(tgt_embedded, memory[beam_idx:beam_idx+1], tgt_mask=tgt_mask)
                
                # 输出层
                output = self.fc(output[:, -1, :])
                
                # 温度缩放
                logits = output / temperature
                
                # 获取top-k个候选词
                top_k_probs, top_k_indices = torch.topk(F.softmax(logits, dim=-1), k=beam_size)
                
                # 扩展束
                for j in range(beam_size):
                    next_token = top_k_indices[0, j].unsqueeze(0).unsqueeze(0)
                    next_score = score + torch.log(top_k_probs[0, j]).item()
                    next_seq = torch.cat([seq, next_token], dim=1)
                    
                    candidates.append({"sequence": next_seq, "score": next_score})
            
            # 按分数排序并选择前beam_size个束
            candidates.sort(key=lambda x: x["score"], reverse=True)
            beams = candidates[:batch_size * beam_size]
            
            # 检查是否所有束都已结束
            if all(beam["sequence"][0, -1] == end_token for beam in beams):
                break
        
        # 选择每个batch中分数最高的序列
        generated = []
        for i in range(batch_size):
            batch_beams = beams[i*beam_size:(i+1)*beam_size]
            best_beam = max(batch_beams, key=lambda x: x["score"])
            generated.append(best_beam["sequence"])
        
        return torch.cat(generated, dim=0)
    
    def generate_with_emotion(self, src, emotion=0, start_token=2, end_token=3, max_length=None, 
                             temperature=0.7, top_k=50, top_p=0.9, sample_strategy="top_k", beam_size=1):
        """使用指定情感生成回复
        
        Args:
            src: 源序列 (batch_size, src_seq_len)
            emotion: 情感类别 (0-4)
            start_token: 开始标记的索引
            end_token: 结束标记的索引
            max_length: 最大生成长度
            temperature: 生成温度
            top_k: top-k采样中保留的最高概率词数
            top_p: top-p采样中保留的累积概率阈值
            sample_strategy: 采样策略
            beam_size: 束搜索的束大小
            
        Returns:
            生成的序列 (batch_size, generated_seq_len)
        """
        return self.generate(src, start_token, end_token, max_length, emotion, 
                           temperature, top_k, top_p, sample_strategy, beam_size)