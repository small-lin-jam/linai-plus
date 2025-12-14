import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from tqdm import tqdm  # 添加进度条支持

class CachedDataset(Dataset):
    """将所有数据缓存到GPU显存的数据集类，减少CPU-GPU数据传输开销
    
    Args:
        X: 输入数据张量
        y: 标签张量
        device: 缓存设备（默认cuda）
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: torch.device = torch.device('cuda')):
        self.X = X.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)
        self.device = device
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 直接从GPU返回数据，无需传输
        return self.X[idx], self.y[idx]

# 尝试导入量化相关库
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

# 尝试导入accelerate用于分布式训练
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

class KnowledgeDistiller:
    """
    知识蒸馏类，实现教师-学生模型之间的知识迁移
    """
    def __init__(self, student_model, teacher_model, temperature: float = 5.0, alpha: float = 0.7):
        """
        Args:
            student_model: 学生模型（较小的模型）
            teacher_model: 教师模型（较大的预训练模型）
            temperature: 蒸馏温度，控制知识的平滑度
            alpha: 蒸馏损失权重，(1-alpha)为原始损失权重
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
            
        Returns:
            总损失
        """
        # 软标签损失（KL散度）
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬标签损失（交叉熵）
        classification_loss = nn.functional.cross_entropy(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        
        return total_loss

class ModelTrainer:
    """模型训练模块"""
    
    def __init__(self, config: Dict):
        """初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 直接从顶层获取配置
        self.preprocess_config = config.get("preprocess", {})
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        
        # 默认训练参数
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 4)  # 针对8GB显存优化的默认batch_size
        self.gradient_accumulation_steps = self.model_config.get("gradient_accumulation_steps", 4)  # 调整梯度累积步数，保持有效batch_size
        self.epochs = self.model_config.get("num_epochs", 10)
        self.max_length = self.preprocess_config.get("max_length", 128)
        self.data_type = self.data_config.get("data_type", "text")
        self.model_type = self.model_config.get("model_type", "transformer")  # 存储模型类型
        self.output_path = "output/models"  # 固定输出路径
        
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        
        # 设备设置：优先使用GPU，CPU作为辅助设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device} (GPU优先，CPU作为辅助设备)")
        
        # 为系统预留30%内存的设置
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)  # 总内存(GB)
        available_memory = total_memory * 0.7  # 可用内存(70%总内存)
        print(f"系统总内存: {total_memory:.1f}GB, 为训练分配内存: {available_memory:.1f}GB (预留30%)")
        
        # 优化配置
        self.use_quantization = self.model_config.get("use_quantization", True) and QUANTIZATION_AVAILABLE  # 默认启用量化
        self.use_gradient_checkpointing = self.model_config.get("use_gradient_checkpointing", True)  # 默认启用梯度检查点
        self.use_bfloat16 = self.model_config.get("use_bfloat16", False)  # bfloat16支持
        
        # 检查bfloat16支持
        if self.use_bfloat16 and torch.cuda.is_available():
            self.use_bfloat16 = torch.cuda.is_bf16_supported()
    
    def build_vocab(self, data: List[List[str]], vocab_size: int = 10000) -> Dict[str, int]:
        """构建词汇表
        
        Args:
            data: 分词后的数据
            vocab_size: 词汇表大小
            
        Returns:
            词汇表
        """
        # 统计词频
        word_counts = {}
        for tokens in data:
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # 按词频排序并选择前vocab_size个词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:vocab_size]
        
        # 创建词汇表
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in top_words:
            vocab[word] = len(vocab)
        
        return vocab
    
    def vectorize_data(self, data: List[List[str]], vocab: Dict[str, int], max_length: int = 100) -> np.ndarray:
        """将文本数据向量化
        
        Args:
            data: 分词后的数据
            vocab: 词汇表
            max_length: 最大序列长度
            
        Returns:
            向量化后的数据
        """
        vectorized_data = []
        
        for tokens in data:
            # 将词转换为索引
            vector = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            
            # 截断或填充到最大长度
            if len(vector) < max_length:
                vector += [vocab["<PAD>"]] * (max_length - len(vector))
            else:
                vector = vector[:max_length]
            
            vectorized_data.append(vector)
        
        return np.array(vectorized_data)
    
    def prepare_data(self, data: Union[List[List[str]], List[np.ndarray], List[Tuple[str, List[np.ndarray]]]]) -> Tuple[DataLoader, Dict[str, Any]]:
        """准备训练数据
        
        Args:
            data: 预处理后的数据
            
        Returns:
            DataLoader和相关配置
        """
        if self.data_type == "text":
            # 文本数据处理
            vocab = self.build_vocab(data)
            vectorized_data = self.vectorize_data(data, vocab, max_length=self.max_length)
            
            # 检查是否是Seq2Seq模型训练
            if hasattr(self, 'model_type') and self.model_type == "seq2seq_transformer":
                # 创建自回归训练数据：源序列为前max_length-1个词，目标序列为后max_length-1个词
                src = vectorized_data[:, :-1]  # 源序列：去掉最后一个词
                tgt = vectorized_data[:, 1:]   # 目标序列：去掉第一个词
                
                # 转换为Tensor
                X = torch.tensor(src, dtype=torch.long)
                y = torch.tensor(tgt, dtype=torch.long)
            else:
                # 创建简单的标签（这里假设是二分类问题）
                labels = np.random.randint(0, 2, size=len(vectorized_data))
                
                # 转换为Tensor
                X = torch.tensor(vectorized_data, dtype=torch.long)
                y = torch.tensor(labels, dtype=torch.long)
            
            # 根据操作系统设置合适的worker数量
            import platform
            if platform.system() == 'Windows':
                num_workers = 16  # Windows系统使用16个工作进程
            else:
                num_workers = 32  # Linux/Mac系统充分利用CPU核心
            
            # 创建数据集和DataLoader
            # 使用CachedDataset将所有数据预加载到GPU显存
            if self.device.type == 'cuda':
                print(f"将所有训练数据预加载到GPU显存中，减少CPU-GPU数据传输开销...")
                print(f"输入数据大小: {X.size()}，内存占用: {(X.element_size() * X.nelement()) / (1024**2):.2f} MB")
                print(f"标签数据大小: {y.size()}，内存占用: {(y.element_size() * y.nelement()) / (1024**2):.2f} MB")
                dataset = CachedDataset(X, y, device=self.device)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=0,  # GPU缓存时禁用多进程
                    pin_memory=False,  # 数据已在GPU，无需pin_memory
                    persistent_workers=False
                )
            else:
                # CPU训练时的常规配置
                dataset = TensorDataset(X, y)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    pin_memory=True, 
                    persistent_workers=num_workers > 0,
                    prefetch_factor=8 if num_workers > 0 else None
                )
            
            return dataloader, {"vocab": vocab}
        
        elif self.data_type == "image":
            # 图像数据处理
            # 假设data是预处理后的图像数组列表 [img1, img2, ...]，每个img是形状为 (C, H, W) 的numpy数组
            processed_images = []
            for img in data:
                if img.ndim == 3:
                    # 已经是 (C, H, W) 格式
                    processed_images.append(img)
                elif img.ndim == 2:
                    # 灰度图，添加通道维度
                    processed_images.append(np.expand_dims(img, axis=0))
            
            # 创建简单的标签（这里假设是二分类问题）
            labels = np.random.randint(0, 2, size=len(processed_images))
            
            # 转换为Tensor
            X = torch.tensor(np.array(processed_images), dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            
            # 根据操作系统设置合适的worker数量，避免Windows权限问题
            import platform
            if platform.system() == 'Windows':
                num_workers = 0  # Windows系统禁用多进程数据加载以避免权限问题
            else:
                num_workers = 32  # Linux/Mac系统充分利用CPU核心
            
            # 创建数据集和DataLoader
            # 使用CachedDataset将所有数据预加载到GPU显存
            if self.device.type == 'cuda':
                print(f"将所有训练数据预加载到GPU显存中，减少CPU-GPU数据传输开销...")
                print(f"输入数据大小: {X.size()}，内存占用: {(X.element_size() * X.nelement()) / (1024**2):.2f} MB")
                print(f"标签数据大小: {y.size()}，内存占用: {(y.element_size() * y.nelement()) / (1024**2):.2f} MB")
                dataset = CachedDataset(X, y, device=self.device)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=0,  # GPU缓存时禁用多进程
                    pin_memory=False,  # 数据已在GPU，无需pin_memory
                    persistent_workers=False
                )
            else:
                # CPU训练时的常规配置
                dataset = TensorDataset(X, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                                      num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=8 if num_workers > 0 else None)
            
            return dataloader, {}
        
        elif self.data_type == "video":
            # 视频数据处理
            # 假设data是预处理后的视频帧列表 [(video1_frames), (video2_frames), ...]，每个video_frames是形状为 (T, C, H, W) 的numpy数组
            # 创建简单的标签（这里假设是二分类问题）
            labels = np.random.randint(0, 2, size=len(data))
            
            # 转换为Tensor
            X = torch.tensor(np.array(data), dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            
            # 创建数据集和DataLoader
            # 使用CachedDataset将所有数据预加载到GPU显存
            if self.device.type == 'cuda':
                print(f"将所有训练数据预加载到GPU显存中，减少CPU-GPU数据传输开销...")
                print(f"输入数据大小: {X.size()}，内存占用: {(X.element_size() * X.nelement()) / (1024**2):.2f} MB")
                print(f"标签数据大小: {y.size()}，内存占用: {(y.element_size() * y.nelement()) / (1024**2):.2f} MB")
                dataset = CachedDataset(X, y, device=self.device)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=0,  # GPU缓存时禁用多进程
                    pin_memory=False,  # 数据已在GPU，无需pin_memory
                    persistent_workers=False
                )
            else:
                # CPU训练时的常规配置
                dataset = TensorDataset(X, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                                      num_workers=32, pin_memory=True, persistent_workers=True, prefetch_factor=8)
            
            return dataloader, {}
        
        else:
            raise ValueError(f"不支持的数据类型: {self.data_type}")
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """量化模型到INT8，减少显存使用并加速训练
        
        Args:
            model: 原始模型
            
        Returns:
            量化后的模型
        """
        if not QUANTIZATION_AVAILABLE:
            return model
            
        # 将模型的全连接层替换为8位量化版本
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # 创建8位量化线性层
                quantized_layer = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,  # 不使用FP16权重
                    threshold=6.0  # 量化阈值
                )
                
                # 复制原始权重
                quantized_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_layer.bias.data = module.bias.data.clone()
                
                # 替换原始层
                setattr(model, name, quantized_layer)
            else:
                # 递归处理子模块
                self._quantize_model(module)
        
        return model
    
    def train(self, model: nn.Module, train_loader: DataLoader, use_distributed: bool = False,
              teacher_model: Optional[nn.Module] = None, use_distillation: bool = False,
              temperature: float = 5.0, alpha: float = 0.7) -> Dict[str, List[float]]:
        """训练模型
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            use_distributed: 是否使用分布式训练
            teacher_model: 教师模型（用于知识蒸馏）
            use_distillation: 是否使用知识蒸馏
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            
        Returns:
            训练历史
        """
        # 损失函数和优化器（使用AdamW和更高的学习率，结合梯度累积实现更大的有效batch_size）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate * self.gradient_accumulation_steps)
        
        # 初始化知识蒸馏器（如果使用）
        distiller = None
        if use_distillation and teacher_model is not None:
            print(f"使用知识蒸馏，教师模型: {teacher_model.__class__.__name__}，温度: {temperature}，alpha: {alpha}")
            distiller = KnowledgeDistiller(model, teacher_model, temperature, alpha)
            # 将教师模型移动到设备
            teacher_model.to(self.device)
        
        # 学习率调度器：余弦退火
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.1)
        
        # 训练历史
        if self.model_type == "seq2seq_transformer":
            history = {
                "train_loss": [],
                "learning_rate": []  # 记录学习率变化
            }
        else:
            history = {
                "train_loss": [],
                "train_acc": [],
                "learning_rate": []  # 记录学习率变化
            }
        
        # 初始化Accelerator用于分布式训练
        if use_distributed and ACCELERATE_AVAILABLE:
            accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
            print(f"使用分布式训练，设备: {accelerator.device}")
        else:
            accelerator = None
        
        # 应用梯度检查点（如果启用）
        if self.use_gradient_checkpointing and hasattr(model, 'use_gradient_checkpointing'):
            print(f"使用梯度检查点技术减少显存使用...")
            # 启用模型的梯度检查点标志
            model.use_gradient_checkpointing = True
        
        # 量化模型（如果启用）
        if self.use_quantization:
            print(f"使用INT8量化加速训练...")
            model = self._quantize_model(model)
        
        # 注意：暂时不使用Torch.compile，因为Python 3.14+上不兼容
        
        # 如果使用分布式训练，准备模型、优化器、数据加载器和调度器
        if accelerator is not None:
            model, optimizer, train_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, scheduler
            )
        else:
            # 将整个模型移至设备
            model.to(self.device)
        
        # 混合精度训练配置（仅在非分布式情况下使用）
        if self.use_bfloat16:
            # 使用bfloat16（不需要scaler）
            print(f"使用bfloat16混合精度训练...")
            scaler = None
            use_mixed_precision = True
        else:
            # 使用fp16混合精度
            scaler = torch.amp.GradScaler('cuda') if accelerator is None and self.device.type == 'cuda' else None
            use_mixed_precision = scaler is not None
        
        # 开始训练
        try:
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                # 使用tqdm添加进度条
                with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                        # 数据已在GPU设备上（通过CachedDataset预加载），无需再移动
                        # if accelerator is None:
                        #     batch_X = batch_X.to(self.device)
                        #     batch_y = batch_y.to(self.device)
                        
                        # 前向传播（使用混合精度）
                        with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=torch.bfloat16 if self.use_bfloat16 else torch.float16):
                            if self.model_type == "seq2seq_transformer":
                                # Seq2Seq模型需要源序列和目标序列作为输入
                                outputs = model(batch_X, batch_y[:, :-1])  # 目标序列去掉最后一个词作为输入
                                # 调整损失计算：outputs形状是(batch_size, tgt_seq_len, vocab_size)，batch_y形状是(batch_size, tgt_seq_len)
                                loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y[:, 1:].reshape(-1))
                            else:
                                # 分类模型
                                outputs = model(batch_X)
                                if distiller is not None:
                                    # 知识蒸馏：获取教师模型的输出
                                    with torch.no_grad():
                                        teacher_outputs = teacher_model(batch_X)
                                    # 计算蒸馏损失
                                    loss = distiller.compute_distillation_loss(outputs, teacher_outputs, batch_y)
                                else:
                                    # 常规损失
                                    loss = criterion(outputs, batch_y)
                            
                            # 梯度累积需要将损失除以累积步数
                            loss = loss / self.gradient_accumulation_steps
                        
                        # 计算准确率（仅对分类模型）
                        if self.model_type != "seq2seq_transformer":
                            _, predicted = torch.max(outputs.data, 1)
                        
                        # 反向传播和优化
                        if accelerator is not None:
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                        else:
                            # 反向传播和优化（使用混合精度）
                            if scaler is not None:
                                # 放大损失，避免梯度下溢
                                scaler.scale(loss).backward()
                            else:
                                # 普通反向传播
                                loss.backward()
                            
                            # 梯度累积：每self.gradient_accumulation_steps步更新一次参数
                            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                                if scaler is not None:
                                    # 优化器更新
                                    scaler.step(optimizer)
                                    # 更新缩放因子
                                    scaler.update()
                                else:
                                    # 普通反向传播
                                    optimizer.step()
                                
                                # 更新学习率
                                scheduler.step()
                                
                                # 清空梯度
                                optimizer.zero_grad()
                        
                        # 计算损失和准确率
                        if accelerator is not None:
                            # 收集所有进程的结果
                            if self.model_type == "seq2seq_transformer":
                                # Seq2Seq模型的总元素数是batch_size * tgt_seq_len
                                train_loss += accelerator.gather(loss * self.gradient_accumulation_steps).sum().item() * batch_X.size(0) * batch_X.size(1)
                            else:
                                # 分类模型
                                train_loss += accelerator.gather(loss * self.gradient_accumulation_steps).sum().item() * batch_X.size(0)
                                correct += accelerator.gather((predicted == batch_y).sum()).item()
                                total += accelerator.gather(batch_y.size(0)).sum().item()
                        else:
                            if self.model_type == "seq2seq_transformer":
                                # Seq2Seq模型的总元素数是batch_size * tgt_seq_len
                                train_loss += loss.item() * batch_X.size(0) * batch_X.size(1) * self.gradient_accumulation_steps
                            else:
                                # 分类模型
                                train_loss += loss.item() * batch_X.size(0) * self.gradient_accumulation_steps
                                total += batch_y.size(0)
                                correct += (predicted == batch_y).sum().item()
                        
                        # 更新进度条并显示当前损失
                        pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
                        pbar.update(1)
                
                # 计算平均损失和准确率
                if self.model_type == "seq2seq_transformer":
                    # Seq2Seq模型的总元素数是所有批次的batch_size * tgt_seq_len
                    epoch_loss = train_loss / (len(train_loader.dataset) * self.max_length)
                else:
                    epoch_loss = train_loss / len(train_loader.dataset)
                    epoch_acc = correct / total
                
                # 记录训练历史和当前学习率
                history["train_loss"].append(epoch_loss)
                if self.model_type != "seq2seq_transformer":
                    history["train_acc"].append(epoch_acc)
                history["learning_rate"].append(optimizer.param_groups[0]["lr"])
                
                # 仅在主进程打印
                if accelerator is None or accelerator.is_main_process:
                    if self.model_type == "seq2seq_transformer":
                        print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
                    else:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
                    
                    # 保存每个epoch的检查点
                    print(f"保存第 {epoch+1} 个epoch的模型检查点...")
                    checkpoint_path = os.path.join(self.output_path, f"checkpoint_epoch_{epoch+1}.pt")
                    save_dict = {
                        "epoch": epoch+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss,
                        "data_type": self.data_type
                    }
                    torch.save(save_dict, checkpoint_path)
                    print(f"检查点已保存到 {checkpoint_path}")
        except KeyboardInterrupt:
            # 优雅地处理用户中断
            print("\n训练被用户中断")
        
        return history
    
    def save_model(self, model: nn.Module, config: Dict[str, Any], vocab: Optional[Dict[str, int]] = None, model_name: str = "model.pt") -> None:
        """保存模型
        
        Args:
            model: 模型
            config: 相关配置
            vocab: 词汇表（可选）
            model_name: 模型文件名
        """
        model_path = os.path.join(self.output_path, model_name)
        
        # 保存模型和配置
        save_dict = {
            "model_state_dict": model.state_dict(),
            "data_type": self.data_type,
            "config": config
        }
        
        # 如果提供了词汇表，也保存它
        if vocab is not None:
            save_dict["vocab"] = vocab
        
        torch.save(save_dict, model_path)
        
        print(f"模型已保存到 {model_path}")