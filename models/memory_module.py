import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Union
from datetime import datetime
import faiss

class AIMemoryModule:
    """AI记忆模块，用于存储和检索模型的知识和经验"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化记忆模块
        
        Args:
            config: 配置字典
        """
        self.memory_config = config.get("memory", {})
        self.memory_dir = self.memory_config.get("memory_dir", "data/memory")
        self.max_memory_size = self.memory_config.get("max_memory_size", 10000)
        self.embedding_dim = self.memory_config.get("embedding_dim", 256)
        self.similarity_threshold = self.memory_config.get("similarity_threshold", 0.7)
        
        # 创建记忆目录
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 记忆存储
        self.memories = []
        self.memory_embeddings = None
        self.memory_index = None
        
        # 加载现有记忆
        self._load_memories()
        
        # 初始化FAISS索引
        self._init_index()
    
    def _load_memories(self):
        """加载现有记忆"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                self.memories = json.load(f)
            
            # 加载嵌入
            embedding_file = os.path.join(self.memory_dir, "memory_embeddings.npy")
            if os.path.exists(embedding_file):
                self.memory_embeddings = np.load(embedding_file)
    
    def _save_memories(self):
        """保存记忆"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
        
        # 保存嵌入
        if self.memory_embeddings is not None:
            embedding_file = os.path.join(self.memory_dir, "memory_embeddings.npy")
            np.save(embedding_file, self.memory_embeddings)
    
    def _init_index(self):
        """初始化FAISS索引"""
        if self.memory_embeddings is not None and len(self.memory_embeddings) > 0:
            self.memory_index = faiss.IndexFlatIP(self.embedding_dim)
            self.memory_index.add(self.memory_embeddings)
    
    def add_memory(self, content: Any, embedding: Union[torch.Tensor, np.ndarray], 
                   memory_type: str = "text", metadata: Dict[str, Any] = None):
        """添加记忆
        
        Args:
            content: 记忆内容
            embedding: 记忆的嵌入向量
            memory_type: 记忆类型 (text, image, video)
            metadata: 记忆的元数据
        """
        # 转换嵌入为numpy数组
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().detach().numpy()
        
        # 确保嵌入是二维数组
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # 创建记忆条目
        memory = {
            "id": len(self.memories) + 1,
            "content": content,
            "embedding": embedding.tolist(),
            "memory_type": memory_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        # 添加到记忆列表
        self.memories.append(memory)
        
        # 更新嵌入矩阵
        if self.memory_embeddings is None:
            self.memory_embeddings = embedding
        else:
            self.memory_embeddings = np.vstack((self.memory_embeddings, embedding))
        
        # 更新FAISS索引
        if self.memory_index is None:
            self._init_index()
        else:
            self.memory_index.add(embedding)
        
        # 限制记忆大小
        if len(self.memories) > self.max_memory_size:
            # 删除最旧的记忆
            self.memories.pop(0)
            self.memory_embeddings = self.memory_embeddings[1:]
            
            # 重新初始化索引
            self._init_index()
        
        # 保存记忆
        self._save_memories()
    
    def retrieve_memories(self, query_embedding: Union[torch.Tensor, np.ndarray], 
                         top_k: int = 5, threshold: float = None) -> List[Dict[str, Any]]:
        """检索相关记忆
        
        Args:
            query_embedding: 查询的嵌入向量
            top_k: 返回的记忆数量
            threshold: 相似度阈值
            
        Returns:
            相关记忆列表
        """
        if self.memory_index is None or len(self.memories) == 0:
            return []
        
        # 转换嵌入为numpy数组
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().detach().numpy()
        
        # 确保嵌入是二维数组
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 设置相似度阈值
        threshold = threshold or self.similarity_threshold
        
        # 检索相似的记忆
        similarities, indices = self.memory_index.search(query_embedding, top_k)
        
        # 过滤并返回相关记忆
        relevant_memories = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= threshold:
                memory = self.memories[idx].copy()
                memory["similarity"] = float(similarity)
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def update_memory(self, memory_id: int, content: Any = None, 
                     embedding: Union[torch.Tensor, np.ndarray] = None, 
                     metadata: Dict[str, Any] = None):
        """更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新的记忆内容
            embedding: 新的记忆嵌入
            metadata: 新的元数据
        """
        # 查找记忆
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                # 更新内容
                if content is not None:
                    memory["content"] = content
                
                # 更新嵌入
                if embedding is not None:
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().detach().numpy()
                    
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)
                    
                    memory["embedding"] = embedding.tolist()
                    self.memory_embeddings[i] = embedding
                
                # 更新元数据
                if metadata is not None:
                    memory["metadata"].update(metadata)
                
                # 更新时间
                memory["updated_at"] = datetime.now().isoformat()
                
                # 保存记忆
                self._save_memories()
                
                # 重新初始化索引
                self._init_index()
                
                return True
        
        return False
    
    def delete_memory(self, memory_id: int) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        # 查找记忆
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                # 删除记忆
                self.memories.pop(i)
                self.memory_embeddings = np.delete(self.memory_embeddings, i, axis=0)
                
                # 保存记忆
                self._save_memories()
                
                # 重新初始化索引
                self._init_index()
                
                return True
        
        return False
    
    def clear_memories(self):
        """清除所有记忆"""
        self.memories = []
        self.memory_embeddings = None
        self.memory_index = None
        
        # 删除记忆文件
        memory_file = os.path.join(self.memory_dir, "memories.json")
        if os.path.exists(memory_file):
            os.remove(memory_file)
        
        embedding_file = os.path.join(self.memory_dir, "memory_embeddings.npy")
        if os.path.exists(embedding_file):
            os.remove(embedding_file)
    
    def get_memory_count(self) -> int:
        """获取记忆数量
        
        Returns:
            记忆数量
        """
        return len(self.memories)
    
    def get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """按类型获取记忆
        
        Args:
            memory_type: 记忆类型
            
        Returns:
            记忆列表
        """
        return [memory for memory in self.memories if memory["memory_type"] == memory_type]

class MemoryEnhancedModel(nn.Module):
    """记忆增强模型基类"""
    
    def __init__(self, base_model: nn.Module, memory_module: AIMemoryModule):
        """初始化记忆增强模型
        
        Args:
            base_model: 基础模型
            memory_module: 记忆模块
        """
        super().__init__()
        self.base_model = base_model
        self.memory_module = memory_module
    
    def forward_with_memory(self, x: Any, use_memory: bool = True) -> Any:
        """使用记忆进行前向传播
        
        Args:
            x: 输入数据
            use_memory: 是否使用记忆
            
        Returns:
            模型输出
        """
        # 基础模型预测
        base_output = self.base_model(x)
        
        if not use_memory:
            return base_output
        
        # 获取输入嵌入
        input_embedding = self._get_input_embedding(x)
        
        # 检索相关记忆
        relevant_memories = self.memory_module.retrieve_memories(input_embedding)
        
        # 如果有相关记忆，调整输出
        if relevant_memories:
            # 这里可以实现更复杂的记忆融合逻辑
            # 例如，使用记忆来调整模型输出或作为额外特征
            adjusted_output = self._fuse_memory_with_output(base_output, relevant_memories)
            return adjusted_output
        
        return base_output
    
    def _get_input_embedding(self, x: Any) -> Union[torch.Tensor, np.ndarray]:
        """获取输入的嵌入向量
        
        Args:
            x: 输入数据
            
        Returns:
            嵌入向量
        """
        # 子类需要实现这个方法
        raise NotImplementedError("_get_input_embedding方法需要在子类中实现")
    
    def _fuse_memory_with_output(self, output: Any, memories: List[Dict[str, Any]]) -> Any:
        """融合记忆和模型输出
        
        Args:
            output: 模型输出
            memories: 相关记忆
            
        Returns:
            融合后的输出
        """
        # 默认实现：返回基础输出
        return output
    
    def add_memory(self, content: Any, embedding: Union[torch.Tensor, np.ndarray], 
                  memory_type: str = "text", metadata: Dict[str, Any] = None):
        """添加记忆
        
        Args:
            content: 记忆内容
            embedding: 记忆嵌入
            memory_type: 记忆类型
            metadata: 元数据
        """
        self.memory_module.add_memory(content, embedding, memory_type, metadata)
    
    def retrieve_memories(self, query_embedding: Union[torch.Tensor, np.ndarray], 
                         top_k: int = 5, threshold: float = None) -> List[Dict[str, Any]]:
        """检索记忆
        
        Args:
            query_embedding: 查询嵌入
            top_k: 返回数量
            threshold: 相似度阈值
            
        Returns:
            相关记忆
        """
        return self.memory_module.retrieve_memories(query_embedding, top_k, threshold)
