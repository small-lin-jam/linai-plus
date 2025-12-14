import os
import ssl
import urllib3
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Union, Tuple
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image

# 基本的SSL验证禁用设置

# 设置环境变量以禁用SSL验证
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# 配置huggingface_hub的缓存目录
os.environ['HF_HUB_CACHE'] = './cache'
os.environ['HF_DATASETS_CACHE'] = './cache'

# 禁用SSL证书验证（解决证书问题）
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
ssl._create_default_https_context = lambda: context

# 配置urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置requests库默认禁用SSL验证
requests.packages.urllib3.disable_warnings()

# 配置huggingface_hub库的SSL设置 - 移除有问题的配置
try:
    from huggingface_hub import configure_http_backend
    # 新版本huggingface_hub不再支持这样的配置
    # 直接删除或使用正确的参数格式
    pass
except ImportError:
    pass

class DataFetcher:
    """数据获取模块，支持从网络获取训练数据"""
    
    def __init__(self, config: Dict):
        """初始化数据获取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.output_path = self.data_config.get("output_path", "data/raw")
        self.data_type = self.data_config.get("data_type", "text")  # text, image, video
        
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        
        # 根据数据类型创建子目录
        if self.data_type == "image":
            self.image_path = os.path.join(self.output_path, "images")
            os.makedirs(self.image_path, exist_ok=True)
        elif self.data_type == "video":
            self.video_path = os.path.join(self.output_path, "videos")
            os.makedirs(self.video_path, exist_ok=True)
    

    
    def _crawl_url(self, url: str, depth: int, visited_urls: set, max_pages: int) -> List[str]:
        """爬取指定URL
        
        Args:
            url: URL地址
            depth: 爬取深度
            visited_urls: 已访问的URL集合
            max_pages: 最大页面数
            
        Returns:
            爬取的数据列表
        """
        if depth <= 0 or url in visited_urls or max_pages <= 0:
            return []
        
        visited_urls.add(url)
        collected_data = []
        
        try:
            # 发送请求
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 提取文本内容
            text = soup.get_text(separator=" ", strip=True)
            if text:
                collected_data.append(text)
                
                # 爬取子页面
                if depth > 1:
                    links = soup.find_all("a", href=True)
                    for link in links:
                        if len(collected_data) >= max_pages:
                            break
                        
                        next_url = link["href"]
                        # 处理相对URL
                        if not next_url.startswith("http"):
                            from urllib.parse import urljoin
                            next_url = urljoin(url, next_url)
                        
                        sub_data = self._crawl_url(next_url, depth - 1, visited_urls, max_pages - len(collected_data))
                        collected_data.extend(sub_data)
        
        except Exception as e:
            print(f"爬取URL失败 {url}: {e}")
        
        return collected_data
    
    def _save_data(self, data: List[str]) -> None:
        """保存数据到文件
        
        Args:
            data: 数据列表
        """
        file_path = os.path.join(self.output_path, "web_data.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(item + "\n\n" + "="*50 + "\n\n")
        
        print(f"已保存 {len(data)} 条数据到 {file_path}")
    
    def fetch_from_local(self) -> Union[List[str], List[np.ndarray], List[Tuple[str, List[np.ndarray]]]]:
        """从本地获取数据
        
        Returns:
            数据列表：文本数据返回字符串列表，图像数据返回numpy数组列表，视频数据返回(视频路径, 帧列表)元组列表
        """
        import hashlib
        
        local_config = self.data_config.get("local", {})
        data_path = local_config.get("path", "data")
        
        collected_data = []
        seen_hashes = set()  # 用于存储已见过的数据哈希，实现去重
        
        if self.data_type == "text":
            # 读取所有文本文件
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content:
                                    # 如果文件很大，按行分割
                                    lines = content.split('\n')
                                    for line in lines:
                                        line = line.strip()
                                        if line:
                                            # 计算文本哈希值
                                            line_hash = hashlib.md5(line.encode()).hexdigest()
                                            if line_hash not in seen_hashes:
                                                seen_hashes.add(line_hash)
                                                collected_data.append(line)
                        except Exception as e:
                            print(f"读取文件失败 {file_path}: {e}")
        elif self.data_type == "image":
            # 读取所有图像文件
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
            for root, _, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            # 使用PIL读取图像
                            with Image.open(file_path) as img:
                                img_array = np.array(img)
                                # 计算图像哈希值（使用前1000个像素点的MD5）
                                img_flat = img_array.flatten()[:1000]  # 使用前1000个像素点
                                img_hash = hashlib.md5(img_flat.tobytes()).hexdigest()
                                if img_hash not in seen_hashes:
                                    seen_hashes.add(img_hash)
                                    collected_data.append(img_array)
                                    print(f"已读取图像: {file_path}, 形状: {img_array.shape}")
                        except Exception as e:
                            print(f"读取图像失败 {file_path}: {e}")
        elif self.data_type == "video":
            # 读取所有视频文件
            import cv2
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
            for root, _, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            # 使用OpenCV读取视频
                            cap = cv2.VideoCapture(file_path)
                            frames = []
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frames.append(frame)
                            cap.release()
                            
                            if frames:
                                # 计算视频哈希值（使用前3帧的哈希）
                                video_hash = hashlib.md5()
                                for frame in frames[:3]:  # 使用前3帧
                                    frame_flat = frame.flatten()[:1000]  # 使用前1000个像素点
                                    video_hash.update(frame_flat.tobytes())
                                video_hash_val = video_hash.hexdigest()
                                
                                if video_hash_val not in seen_hashes:
                                    seen_hashes.add(video_hash_val)
                                    collected_data.append((file_path, frames))
                                    print(f"已读取视频: {file_path}, 帧数: {len(frames)}")
                        except Exception as e:
                            print(f"读取视频失败 {file_path}: {e}")
        
        print(f"去重完成：原始数据数量: {len(seen_hashes)}, 去重后数量: {len(collected_data)}")
        return collected_data
    
    def fetch_from_hf(self) -> List[str]:
        """从Hugging Face获取数据集
        
        Returns:
            数据列表
        """
        try:
            import json
            import glob
            import os
            
            hf_config = self.data_config.get("hf", {})
            dataset_name = hf_config.get("dataset_name")
            split = hf_config.get("split", "train")
            data_column = hf_config.get("data_column", "text")
            api_token = hf_config.get("api_token")
            use_auth_token = hf_config.get("use_auth_token", False)
            
            if not dataset_name:
                print("请在配置中指定Hugging Face数据集名称 (hf.dataset_name)")
                return []
            
            print(f"正在从Hugging Face下载数据集: {dataset_name} ({split})")
            
            # 使用huggingface_hub直接下载数据集
            download_path = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",  # 明确指定是数据集仓库
                token=api_token if api_token else use_auth_token,
                ignore_patterns=[".git", "*.git*", "*.md", "*.jsonl", "*.tar.gz"],
                # 这里不能直接使用verify=False，但我们已经全局禁用了SSL验证
            )
            
            print(f"数据集下载到: {download_path}")
            
            # 手动解析下载的数据集文件
            collected_data = []
            
            # 查找并读取数据文件
            data_files = []
            
            # 首先查看目录结构
            print(f"\n下载目录结构:")
            for root, dirs, files in os.walk(download_path):
                level = root.replace(download_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
                    # 收集所有可能的数据文件
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in [".json", ".jsonl", ".txt", ".csv"]:
                        data_files.append(file_path)
            
            # 如果还是没找到文件，尝试直接获取所有文件
            if not data_files:
                print("\n未找到标准数据文件，尝试获取所有文件")
                data_files = [os.path.join(root, file) 
                              for root, dirs, files in os.walk(download_path)
                              for file in files
                              if not file.startswith(".")]  # 排除隐藏文件
            
            if not data_files:
                print("未找到任何数据文件")
                return []
            
            print(f"\n找到 {len(data_files)} 个数据文件")
            
            # 读取数据文件
            for file_path in data_files:
                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    if ext == ".json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and data_column in item:
                                        collected_data.append(item[data_column])
                                    elif isinstance(item, str):
                                        collected_data.append(item)
                    elif ext == ".jsonl":
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    item = json.loads(line)
                                    if isinstance(item, dict) and data_column in item:
                                        collected_data.append(item[data_column])
                                    elif isinstance(item, str):
                                        collected_data.append(item)
                    elif ext == ".txt":
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            lines = content.split("\n")
                            collected_data.extend([line.strip() for line in lines if line.strip()])
                    elif ext == ".parquet":
                        # 使用pandas读取Parquet文件
                        try:
                            import pandas as pd
                            
                            # 读取Parquet文件
                            df = pd.read_parquet(file_path)
                            
                            # 检查是否包含所需的数据列
                            if data_column in df.columns:
                                # 获取指定列的数据
                                text_data = df[data_column].tolist()
                                collected_data.extend([str(text) for text in text_data if text])
                                print(f"从Parquet文件 {os.path.basename(file_path)} 中读取了 {len(text_data)} 条数据")
                            else:
                                # 打印可用的列名
                                print(f"Parquet文件 {os.path.basename(file_path)} 不包含列 {data_column}")
                                print(f"可用列: {', '.join(df.columns.tolist())}")
                                
                                # 尝试使用text列作为后备
                                if 'text' in df.columns:
                                    text_data = df['text'].tolist()
                                    collected_data.extend([str(text) for text in text_data if text])
                                    print(f"使用后备列 'text' 读取了 {len(text_data)} 条数据")
                        except ImportError:
                            print(f"需要安装pandas和pyarrow来读取Parquet文件: {file_path}")
                            print("请运行: pip install pandas pyarrow")
                        except Exception as e:
                            print(f"读取Parquet文件失败 {file_path}: {e}")
                except Exception as e:
                    print(f"读取数据文件失败 {file_path}: {e}")
                    continue
            
            # 如果没有找到特定列的数据，尝试读取plain_text目录
            if not collected_data:
                plain_text_path = os.path.join(download_path, "plain_text")
                if os.path.exists(plain_text_path):
                    text_files = glob.glob(os.path.join(plain_text_path, "*.txt"), recursive=True)
                    for file_path in text_files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                lines = content.split("\n")
                                collected_data.extend([line.strip() for line in lines if line.strip()])
                        except Exception as e:
                            print(f"读取文本文件失败 {file_path}: {e}")
                            continue
            
            if not collected_data:
                print(f"未提取到数据，请检查数据列 '{data_column}' 是否存在于数据文件中")
                return []
            
            print(f"从Hugging Face获取了 {len(collected_data)} 条数据")
            
            # 保存数据到文件
            self._save_data(collected_data)
            
            return collected_data
            
        except Exception as e:
            import traceback
            print(f"从Hugging Face获取数据失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return []
    
    def fetch_from_web(self) -> Union[List[str], List[np.ndarray]]:
        """从网络获取数据
        
        Returns:
            数据列表：文本数据返回字符串列表，图像数据返回numpy数组列表
        """
        web_config = self.data_config.get("web", {})
        urls = web_config.get("urls", [])
        crawling_depth = web_config.get("crawling_depth", 2)
        max_pages = web_config.get("max_pages", 100)
        
        collected_data = []
        visited_urls = set()
        
        if self.data_type == "text":
            # 文本数据爬取（原有逻辑）
            for url in urls:
                if len(collected_data) >= max_pages:
                    break
                
                data = self._crawl_url(url, crawling_depth, visited_urls, max_pages - len(collected_data))
                collected_data.extend(data)
            
            # 保存数据到文件
            self._save_data(collected_data)
        
        elif self.data_type == "image":
            # 图像数据爬取
            for url in urls:
                if len(collected_data) >= max_pages:
                    break
                
                # 获取页面上的所有图像URL
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, "html.parser")
                    img_tags = soup.find_all("img")
                    
                    for img_tag in img_tags:
                        if len(collected_data) >= max_pages:
                            break
                        
                        img_url = img_tag.get("src")
                        if not img_url:
                            continue
                        
                        # 处理相对URL
                        if not img_url.startswith("http"):
                            from urllib.parse import urljoin
                            img_url = urljoin(url, img_url)
                        
                        # 下载图像
                        try:
                            img_response = requests.get(img_url, timeout=5)
                            img_response.raise_for_status()
                            
                            # 保存图像到文件
                            import io
                            img = Image.open(io.BytesIO(img_response.content))
                            img_array = np.array(img)
                            collected_data.append(img_array)
                            
                            # 保存图像到本地
                            img_name = f"image_{len(collected_data)}.jpg"
                            img_path = os.path.join(self.image_path, img_name)
                            img.save(img_path)
                            print(f"已下载图像: {img_url} 保存到: {img_path}")
                            
                        except Exception as e:
                            print(f"下载图像失败 {img_url}: {e}")
                except Exception as e:
                    print(f"处理URL失败 {url}: {e}")
        
        return collected_data
    
    def fetch(self) -> Union[List[str], List[np.ndarray], List[Tuple[str, List[np.ndarray]]]]:
        """获取数据的主方法
        
        Returns:
            数据列表：文本数据返回字符串列表，图像数据返回numpy数组列表，视频数据返回(视频路径, 帧列表)元组列表
        """
        source = self.data_config.get("source", "local")
        
        if source == "web":
            return self.fetch_from_web()
        elif source == "local":
            return self.fetch_from_local()
        elif source == "hf":
            return self.fetch_from_hf()
        else:
            print(f"不支持的数据来源: {source}")
            return []