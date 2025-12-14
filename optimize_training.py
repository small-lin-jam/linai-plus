#!/usr/bin/env python3
"""
优化训练配置，充分利用64核心CPU和RTX3070 GPU的协同运作
"""

import torch
import os
import platform
from typing import Dict, Any

def get_optimized_config() -> Dict[str, Any]:
    """
    获取针对当前硬件配置（64核心CPU+64GB内存，RTX3070笔记本GPU+8G显存）的优化训练配置
    
    Returns:
        优化后的配置字典
    """
    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 基础配置
    config = {
        "device": device,
        "use_gpu": use_gpu,
        "use_cpu": not use_gpu,
        "model_type": "seq2seq_transformer",
        "epochs": 10,
    }
    
    if use_gpu:
        # GPU训练优化配置
        config.update({
            # 模型配置：在8GB显存下保持合理大小
            "model_config": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_heads": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "max_length": 32,
                "vocab_size": 10002,
            },
            
            # 训练配置
            "training_config": {
                # 批量大小：根据显存调整
                "batch_size": 32,  # 增大批量大小，充分利用GPU显存
                
                # 梯度累积：进一步增大有效批量大小
                "gradient_accumulation_steps": 2,  # 有效批量大小 = 32 * 2 = 64
                
                # 学习率：适当提高学习率以匹配更大的批量大小
                "learning_rate": 0.001,  # 原学习率的2倍，因为批量大小增大了2倍
                
                # 混合精度训练：使用bfloat16加速训练并减少显存使用
                "use_bfloat16": True,
                
                # 梯度检查点：减少显存使用
                "use_gradient_checkpointing": True,
                
                # 量化：使用INT8量化加速训练
                "use_quantization": True,
            },
            
            # 数据加载配置：CPU和GPU协同
            "dataloader_config": {
                # GPU缓存：将数据预加载到GPU显存
                "use_gpu_cache": True,
                
                # 数据加载器工作进程数：充分利用64核心CPU
                "num_workers": 32,  # 使用32个工作进程，避免过度竞争
                
                # 预取因子：预取更多批次
                "prefetch_factor": 16,  # 预取16个批次
                
                # 固定内存：固定数据到内存，加速数据传输
                "pin_memory": True,
                
                # 持久化工作进程：减少进程创建开销
                "persistent_workers": True,
            },
        })
    else:
        # CPU训练优化配置
        config.update({
            "model_config": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_heads": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "max_length": 32,
                "vocab_size": 10002,
            },
            
            "training_config": {
                "batch_size": 128,  # CPU训练可以使用更大的批量大小
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0005,
                "use_bfloat16": False,  # CPU通常不支持bfloat16
                "use_gradient_checkpointing": False,  # CPU不需要梯度检查点
                "use_quantization": True,  # 量化仍可加速CPU训练
            },
            
            "dataloader_config": {
                "use_gpu_cache": False,
                "num_workers": 32,  # 使用32个工作进程
                "prefetch_factor": 16,
                "pin_memory": False,  # CPU训练不需要pin_memory
                "persistent_workers": True,
            },
        })
    
    return config

def print_hardware_info():
    """
    打印当前硬件信息
    """
    print("=" * 50)
    print("硬件信息")
    print("=" * 50)
    
    # CPU信息
    print(f"CPU核心数: {os.cpu_count()}")
    
    # 内存信息
    if platform.system() == 'Windows':
        import psutil
        mem = psutil.virtual_memory()
        print(f"内存总量: {mem.total / (1024**3):.2f} GB")
    else:
        # Linux/Mac
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_total = int(line.split()[1]) * 1024  # 转换为字节
                    print(f"内存总量: {mem_total / (1024**3):.2f} GB")
                    break
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            device = torch.device(f'cuda:{i}')
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            gpu_sm = f"{torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}"
            gpu_bf16 = torch.cuda.is_bf16_supported()
            
            print(f"GPU {i}: {gpu_name}")
            print(f"  显存: {gpu_mem:.2f} GB")
            print(f"  SM版本: {gpu_sm}")
            print(f"  bfloat16支持: {gpu_bf16}")
    else:
        print("未检测到GPU")
    
    print("=" * 50)

def main():
    """
    主函数，展示优化配置
    """
    print_hardware_info()
    
    # 获取优化配置
    optimized_config = get_optimized_config()
    
    print("\n优化训练配置")
    print("=" * 50)
    
    # 打印模型配置
    print("模型配置:")
    for key, value in optimized_config["model_config"].items():
        print(f"  {key}: {value}")
    
    # 打印训练配置
    print("\n训练配置:")
    for key, value in optimized_config["training_config"].items():
        print(f"  {key}: {value}")
    
    # 打印数据加载配置
    print("\n数据加载配置:")
    for key, value in optimized_config["dataloader_config"].items():
        print(f"  {key}: {value}")
    
    # 计算有效批量大小
    if "training_config" in optimized_config:
        batch_size = optimized_config["training_config"]["batch_size"]
        grad_accum = optimized_config["training_config"]["gradient_accumulation_steps"]
        effective_batch_size = batch_size * grad_accum
        print(f"\n有效批量大小: {effective_batch_size} ({batch_size} * {grad_accum})")
    
    print("\n" + "=" * 50)
    print("优化建议")
    print("=" * 50)
    print("1. 使用GPU训练：RTX3070的并行计算能力远强于CPU")
    print("2. 启用bfloat16混合精度训练：加速训练并减少显存使用")
    print("3. 使用梯度检查点：进一步减少显存使用")
    print("4. 增加批量大小和梯度累积：提高训练稳定性和效率")
    print("5. 充分利用CPU核心进行数据加载：使用多进程数据加载")
    print("6. 启用数据预取：减少GPU等待CPU数据的时间")
    print("7. 考虑使用分布式训练：如果有多个GPU可用")

if __name__ == "__main__":
    main()