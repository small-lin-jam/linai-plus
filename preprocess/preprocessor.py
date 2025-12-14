import os
import re
import hashlib
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Union, Tuple

class DataPreprocessor:
    """数据预处理模块"""
    
    def __init__(self, config: Dict):
        """初始化数据预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocess_config = config.get("preprocess", {})
        self.tokenizer_name = self.preprocess_config.get("tokenizer", "bert")
        self.cache_dir = self.preprocess_config.get("cache_dir", "data/cache")
        
        # 图像预处理配置
        self.image_config = self.preprocess_config.get("image", {})
        self.image_size = self.image_config.get("size", (224, 224))
        self.image_normalize = self.image_config.get("normalize", True)
        self.mean = self.image_config.get("mean", [0.485, 0.456, 0.406])
        self.std = self.image_config.get("std", [0.229, 0.224, 0.225])
        
        # 视频预处理配置
        self.video_config = self.preprocess_config.get("video", {})
        self.video_frames = self.video_config.get("frames", 16)
        self.video_size = self.video_config.get("size", (224, 224))
        self.video_normalize = self.video_config.get("normalize", True)
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """文本清洗
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 优化的正则表达式处理顺序
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 转换为小写并处理空格
        text = text.lower()
        
        # 移除特殊字符和多余空格
        text = re.sub(r'[^\w\s]', '', text)  # 只保留字母、数字、下划线和空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 文本
            
        Returns:
            分词结果
        """
        # 简单的空格分词
        return text.split()
    
    def _get_cache_key(self, text: str) -> str:
        """生成文本的缓存键
        
        Args:
            text: 文本
            
        Returns:
            缓存键
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def preprocess(self, data: List[str]) -> List[List[str]]:
        """预处理数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            预处理后的数据列表
        """
        preprocessed_data = []
        
        # 批量处理优化
        for text in data:
            if not text:
                continue
                
            # 文本清洗
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                # 分词
                tokens = self.tokenize(cleaned_text)
                
                if tokens:
                    preprocessed_data.append(tokens)
        
        return preprocessed_data
    
    def preprocess(self, data: Union[List[str], List[np.ndarray], List[Tuple[str, List[np.ndarray]]]]) -> Union[List[List[str]], List[torch.Tensor], List[torch.Tensor]]:
        """预处理数据
        
        Args:
            data: 原始数据列表，可以是文本、图像或视频
            
        Returns:
            预处理后的数据列表
        """
        if not data:
            return []
        
        # 根据数据类型选择预处理方法
        first_item = data[0]
        
        if isinstance(first_item, str):
            return self.preprocess_text(data)
        elif isinstance(first_item, np.ndarray):
            return self.preprocess_images(data)
        elif isinstance(first_item, tuple):
            return self.preprocess_videos(data)
        else:
            raise ValueError(f"不支持的数据类型: {type(first_item)}")
    
    def preprocess_text(self, data: List[str]) -> List[List[str]]:
        """预处理文本数据
        
        Args:
            data: 原始文本数据列表
            
        Returns:
            预处理后的数据列表
        """
        preprocessed_data = []
        cache = {}
        seen_processed_hashes = set()  # 用于存储已见过的预处理后数据的哈希，实现去重
        
        # 检查缓存
        for text in data:
            if not text:
                continue
                
            # 尝试从缓存获取
            cache_key = self._get_cache_key(text)
            if cache_key in cache:
                preprocessed_data.append(cache[cache_key])
                continue
                
            # 文本清洗
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                # 分词
                tokens = self.tokenize(cleaned_text)
                
                if tokens:
                    # 生成预处理后数据的哈希
                    processed_key = hashlib.md5(str(tokens).encode()).hexdigest()
                    if processed_key not in seen_processed_hashes:
                        seen_processed_hashes.add(processed_key)
                        # 缓存结果
                        cache[cache_key] = tokens
                        preprocessed_data.append(tokens)
        
        print(f"预处理去重完成：原始数据数量: {len(data)}, 预处理后数量: {len(preprocessed_data)}")
        return preprocessed_data
    
    def preprocess_images(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """预处理图像数据
        
        Args:
            images: 图像数据列表
            
        Returns:
            预处理后的图像张量列表
        """
        if not images:
            return []
            
        # 创建变换管道
        import torchvision.transforms as transforms
        image_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std) if self.image_normalize else transforms.Lambda(lambda x: x)
        ])
        
        preprocessed_images = []
        for image in images:
            if image is None:
                continue
                
            # 转换为PIL图像
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                
            pil_image = Image.fromarray(image)
            
            # 应用变换
            try:
                image_tensor = image_transforms(pil_image)
                preprocessed_images.append(image_tensor)
            except Exception as e:
                print(f"图像预处理失败: {e}")
                continue
        
        return preprocessed_images
    
    def preprocess_videos(self, videos: List[Tuple[str, List[np.ndarray]]]) -> List[torch.Tensor]:
        """预处理视频数据
        
        Args:
            videos: 视频数据列表，每个视频是一个元组 (视频名, 帧列表)
            
        Returns:
            预处理后的视频张量列表
        """
        if not videos:
            return []
            
        # 创建变换管道
        import torchvision.transforms as transforms
        frame_transforms = transforms.Compose([
            transforms.Resize(self.video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std) if self.video_normalize else transforms.Lambda(lambda x: x)
        ])
        
        preprocessed_videos = []
        
        for video_name, frames in videos:
            if not frames:
                continue
                
            # 均匀采样帧
            sampled_frames = self._sample_frames(frames, self.video_frames)
            
            preprocessed_frames = []
            for frame in sampled_frames:
                if frame is None:
                    continue
                    
                # 转换为PIL图像
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                    
                pil_frame = Image.fromarray(frame)
                
                # 应用变换
                try:
                    frame_tensor = frame_transforms(pil_frame)
                    preprocessed_frames.append(frame_tensor)
                except Exception as e:
                    print(f"视频帧预处理失败: {e}")
                    continue
            
            if preprocessed_frames:
                # 将帧堆叠为视频张量 (F, C, H, W)
                try:
                    video_tensor = torch.stack(preprocessed_frames)
                    preprocessed_videos.append(video_tensor)
                except Exception as e:
                    print(f"视频帧堆叠失败: {e}")
                    continue
        
        return preprocessed_videos
    
    def _sample_frames(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        """均匀采样帧
        
        Args:
            frames: 帧列表
            num_frames: 采样的帧数
            
        Returns:
            采样后的帧列表
        """
        if len(frames) <= num_frames:
            return frames
        
        step = len(frames) // num_frames
        sampled_frames = [frames[i * step] for i in range(num_frames)]
        
        # 如果有剩余帧，添加最后一帧
        if len(sampled_frames) < num_frames:
            sampled_frames.append(frames[-1])
        
        return sampled_frames
    
    def save_preprocessed_data(self, data: Union[List[List[str]], List[torch.Tensor]], 
                              output_path: str = "data/preprocessed", 
                              data_type: str = "text") -> None:
        """保存预处理后的数据
        
        Args:
            data: 预处理后的数据
            output_path: 输出路径
            data_type: 数据类型 (text, image, video)
        """
        os.makedirs(output_path, exist_ok=True)
        
        if data_type == "text":
            file_path = os.path.join(output_path, "preprocessed_data.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for tokens in data:
                    f.write(" ".join(tokens) + "\n")
            
            print(f"已保存 {len(data)} 条预处理文本数据到 {file_path}")
        
        elif data_type in ["image", "video"]:
            # 保存为PyTorch张量
            file_path = os.path.join(output_path, f"preprocessed_{data_type}s.pt")
            torch.save(data, file_path)
            
            print(f"已保存 {len(data)} 条预处理{data_type}数据到 {file_path}")
        
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")