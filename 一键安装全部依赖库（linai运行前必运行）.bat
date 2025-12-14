@echo off
title Python库自动安装脚本 
echo 正在安装AI项目依赖库（使用清华镜像源）...
echo.
 
:: 核心依赖安装列表 
set packages=^
requests beautifulsoup4 huggingface_hub paramiko ^
torch torchvision torchaudio ^
numpy pandas matplotlib tqdm ^
transformers langchain textblob ^
nltk wandb tensorboardX ^
fastapi uvicorn python-dotenv 
 
:: 批量安装（使用清华镜像源）
.venv\Scripts\pip install %packages% -i https://pypi.tuna.tsinghua.edu.cn/simple  
 
:: 特殊库处理（根据CUDA版本安装cupy）
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" (
    for /f "tokens=*" %%i in ('dir /b "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"') do (
        set cuda_ver=%%i 
    )
    set cuda_ver=!cuda_ver:~1!
    .venv\Scripts\pip install cupy-cuda!cuda_ver! -i https://pypi.tuna.tsinghua.edu.cn/simple  
) else (
    echo 未检测到CUDA，跳过cupy安装 
)
 
:: 下载NLTK数据
echo.
echo 正在下载NLTK情感分析数据...
.venv\Scripts\python -c "import nltk; nltk.download('punkt',  quiet=True); nltk.download('vader_lexicon',  quiet=True)"
 
:: 环境配置 
echo.
echo 创建项目目录结构...
mkdir downloads 2>nul 
mkdir data 2>nul 
mkdir logs 2>nul 

echo 配置HuggingFace镜像...
setx HF_ENDPOINT "https://hf-mirror.com/" 
 
echo.
echo ✅ 所有依赖安装完成！
echo 请运行: .venv\Scripts\python linai.py  启动项目 
pause