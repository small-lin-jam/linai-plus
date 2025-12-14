import torch
import torch.nn as nn
from models.model_def import ModelFactory
from preprocess.preprocessor import DataPreprocessor
from typing import Dict, Any, Union, List
import numpy as np

class ModelLoader:
    """模型加载器，用于加载和运行训练好的模型"""
    
    def __init__(self, model_path: str):
        """初始化模型加载器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.data_type = None
        self.config = None
        self.preprocessor = None
        self.emotion_lexicon = None
        
        # 加载情感词汇表
        self.load_emotion_lexicon()
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            # 加载模型文件
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 打印检查点中的键列表
            print(f"检查点中的键列表: {list(checkpoint.keys())}")
            print(f"检查点中是否包含vocab: {'vocab' in checkpoint}")
            if 'vocab' in checkpoint:
                print(f"词汇表大小: {len(checkpoint['vocab'])}")
                # 打印前10个词汇表条目
                print(f"词汇表前10项: {list(checkpoint['vocab'].items())[:10]}")
            else:
                print("检查点中不包含词汇表")
            
            # 获取模型配置
            self.data_type = checkpoint.get('data_type', 'text')
            self.config = checkpoint.get('config', {})
            
            # 确保vocab被正确加载，无论是从checkpoint直接获取还是从config中获取
            vocab_loaded = False
            if 'vocab' in checkpoint:
                self.config['vocab'] = checkpoint['vocab']
                self.vocab = checkpoint['vocab']  # 保存到self.vocab供id_to_text使用
                # 设置正确的vocab_size，包含特殊标记
                self.config['vocab_size'] = len(self.config['vocab'])
                print(f"从检查点加载了vocab，长度为: {self.config['vocab_size']}")
                print(f"词汇表前10项: {list(self.vocab.items())[:10]}")
                vocab_loaded = True
            elif 'vocab' in checkpoint.get('config', {}):
                # 如果vocab在checkpoint的config中
                self.config['vocab'] = checkpoint['config']['vocab']
                self.vocab = self.config['vocab']
                self.config['vocab_size'] = len(self.config['vocab'])
                print(f"从检查点的config中加载了vocab，长度为: {self.config['vocab_size']}")
                print(f"词汇表前10项: {list(self.vocab.items())[:10]}")
                vocab_loaded = True
            else:
                # 寻找其他可能包含vocab的键
                for key in checkpoint.keys():
                    # 跳过键为'vocab'的项，避免嵌套字典问题
                    if key == 'vocab':
                        continue
                    if isinstance(checkpoint[key], dict) and '<PAD>' in checkpoint[key]:
                        # 过滤掉所有键不是字符串或整数的项，避免嵌套字典问题
                        filtered_vocab = {k: v for k, v in checkpoint[key].items() if isinstance(k, (str, int))}
                        self.config['vocab'] = filtered_vocab
                        self.vocab = self.config['vocab']
                        self.config['vocab_size'] = len(self.config['vocab'])
                        print(f"从检查点的{key}中加载了vocab，长度为: {self.config['vocab_size']}")
                        print(f"词汇表前10项: {list(self.vocab.items())[:10]}")
                        vocab_loaded = True
                        break
                
                # 如果仍然没有找到vocab，尝试从模型权重中获取正确的vocab_size
                if not vocab_loaded:
                    if 'model_state_dict' in checkpoint and 'embedding.weight' in checkpoint['model_state_dict']:
                        self.config['vocab_size'] = checkpoint['model_state_dict']['embedding.weight'].shape[0]
                        print(f"从模型权重获取vocab_size: {self.config['vocab_size']}")
                    else:
                        # 设置一个默认值
                        self.config['vocab_size'] = 10000
                        print(f"使用默认vocab_size: {self.config['vocab_size']}")
            
            # 根据数据类型创建预处理模块
            if self.data_type == 'text':
                self.preprocessor = DataPreprocessor({'preprocess': {'max_length': self.config.get('max_length', 100)}})
            elif self.data_type in ['image', 'video']:
                self.preprocessor = DataPreprocessor({'preprocess': {'image': {'size': self.config.get('image_size', (224, 224))}}})
            
            # 打印配置信息，调试模型加载问题
            print(f"配置信息: {self.config}")
            print(f"配置中是否包含vocab: {'vocab' in self.config}")
            if 'vocab' in self.config:
                print(f"vocab长度: {len(self.config['vocab'])}")
            print(f"配置中是否包含vocab_size: {'vocab_size' in self.config}")
            if 'vocab_size' in self.config:
                print(f"vocab_size: {self.config['vocab_size']}")
            
            # 检测模型类型
            model_type = self.config.get('model_type', None)
            
            # 如果配置中没有model_type，尝试从模型状态字典中检测
            if model_type is None and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # 检查是否是Seq2Seq Transformer模型（包含encoder和decoder）
                if any(key.startswith('transformer_encoder') for key in state_dict.keys()) and any(key.startswith('transformer_decoder') for key in state_dict.keys()):
                    model_type = 'seq2seq_transformer'
                    print(f"从模型状态字典检测到模型类型: {model_type}")
                # 检查是否是普通Transformer模型（只包含encoder）
                elif any(key.startswith('transformer_encoder') for key in state_dict.keys()):
                    model_type = 'transformer'
                    print(f"从模型状态字典检测到模型类型: {model_type}")
                else:
                    model_type = 'simple_classifier'
                    print(f"无法从模型状态字典确定模型类型，使用默认类型: {model_type}")
            elif model_type is None:
                model_type = 'simple_classifier'
                print(f"没有找到模型类型信息，使用默认类型: {model_type}")
            
            # 保存model_type到实例变量
            self.model_type = model_type
            
            # 确保嵌入维度与保存的模型匹配
            if 'model_state_dict' in checkpoint and 'embedding.weight' in checkpoint['model_state_dict']:
                embedding_dim = checkpoint['model_state_dict']['embedding.weight'].shape[1]
                self.config['embedding_dim'] = embedding_dim
                print(f"从模型状态字典获取embedding_dim: {embedding_dim}")
            
            # 从模型状态字典中提取其他配置参数
            if 'model_state_dict' in checkpoint and model_type in ['transformer', 'seq2seq_transformer']:
                state_dict = checkpoint['model_state_dict']
                
                # 提取max_length（从pos_encoder.pe的形状中获取）
                if 'pos_encoder.pe' in state_dict:
                    max_length = state_dict['pos_encoder.pe'].shape[0]
                    self.config['max_length'] = max_length
                    print(f"从模型状态字典获取max_length: {max_length}")
                
                # 提取hidden_dim（从linear1.weight的形状中获取）
                if 'transformer_encoder.layers.0.linear1.weight' in state_dict:
                    hidden_dim = state_dict['transformer_encoder.layers.0.linear1.weight'].shape[0]
                    self.config['hidden_dim'] = hidden_dim
                    print(f"从模型状态字典获取hidden_dim: {hidden_dim}")
                
                # 提取num_classes或vocab_size（从fc.weight的形状中获取）
                if 'fc.weight' in state_dict:
                    if model_type == 'seq2seq_transformer':
                        # 对于序列到序列模型，fc.weight.shape[0]是vocab_size
                        vocab_size = state_dict['fc.weight'].shape[0]
                        self.config['vocab_size'] = vocab_size
                        print(f"从模型状态字典获取vocab_size: {vocab_size}")
                    else:
                        # 对于分类模型，fc.weight.shape[0]是num_classes
                        num_classes = state_dict['fc.weight'].shape[0]
                        self.config['num_classes'] = num_classes
                        print(f"从模型状态字典获取num_classes: {num_classes}")
                
                # 提取num_layers（从transformer_encoder.layers的数量中获取）
                layer_count = 0
                while f'transformer_encoder.layers.{layer_count}.self_attn.in_proj_weight' in state_dict:
                    layer_count += 1
                if layer_count > 0:
                    self.config['num_layers'] = layer_count
                    print(f"从模型状态字典获取num_layers: {layer_count}")
                
                # 提取num_heads（从self_attn.in_proj_weight的形状中获取）
                if 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict:
                    # in_proj_weight的形状是[3 * embed_dim, embed_dim]
                    # 其中embed_dim是embedding_dim
                    in_proj_weight_shape = state_dict['transformer_encoder.layers.0.self_attn.in_proj_weight'].shape[0]
                    if in_proj_weight_shape % embedding_dim == 0:
                        num_heads = in_proj_weight_shape // (3 * embedding_dim)
                        self.config['num_heads'] = num_heads
                        print(f"从模型状态字典获取num_heads: {num_heads}")
            
            # 创建模型
            model_factory = ModelFactory()
            self.model = model_factory.create_model(model_type, self.config)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"模型加载成功！数据类型: {self.data_type}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def preprocess_input(self, input_data: Union[str, np.ndarray]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """预处理输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            预处理后的数据
        """
        if self.data_type == 'text':
            # 文本数据预处理
            preprocessed = self.preprocessor.preprocess_text([input_data])
            return preprocessed
        elif self.data_type == 'image':
            # 图像数据预处理
            preprocessed = self.preprocessor.preprocess_images([input_data])
            return preprocessed[0] if preprocessed else None
        elif self.data_type == 'video':
            # 视频数据预处理
            # 这里简化处理，假设视频是帧列表
            preprocessed = self.preprocessor.preprocess_videos([('temp_video', [input_data])])
            return preprocessed[0] if preprocessed else None
        else:
            raise ValueError(f"不支持的数据类型: {self.data_type}")
    
    def predict(self, input_data: Union[str, np.ndarray], emotion: int = 0) -> Dict[str, Any]:
        """使用模型进行预测，生成适合问答交互的输出，支持情感控制
        
        Args:
            input_data: 输入数据，可以是文本或图像
            emotion: 情感类别 (0-4)，0=积极，1=消极，2=愤怒，3=惊讶，4=中性
            
        Returns:
            预测结果，包含对话式回复
        """
        try:
            # 预处理数据
            preprocessed_data = self.preprocess_input(input_data)
            
            if preprocessed_data is None:
                return {'error': '数据预处理失败'}
            
            # 准备模型输入
            if self.data_type == 'text':
                # 文本数据需要额外处理
                vocab = self.config.get('vocab', {})
                max_length = self.config.get('max_length', 100)
                
                # 将文本转换为索引
                vector = [vocab.get(token, vocab.get('<UNK>', 1)) for token in preprocessed_data[0]]
                
                # 截断或填充到最大长度
                if len(vector) < max_length:
                    vector += [vocab.get('<PAD>', 0)] * (max_length - len(vector))
                else:
                    vector = vector[:max_length]
                
                input_tensor = torch.tensor([vector], dtype=torch.long)
                
                # 检查模型类型
                if hasattr(self.model, 'generate'):
                    # 生成式模型
                    # 使用正确的特殊标记作为开始和结束标记
                    start_token = 2  # <SOS>
                    end_token = 3    # <EOS>
                    
                    # 生成回复，支持情感控制
                    if hasattr(self.model, 'generate_with_emotion'):
                        generated_ids = self.model.generate_with_emotion(input_tensor, emotion=emotion, start_token=start_token, end_token=end_token)
                    else:
                        generated_ids = self.model.generate(input_tensor, start_token=start_token, end_token=end_token)
                    
                    # 调试信息：查看生成的ID序列
                    print(f"生成的ID序列: {generated_ids[0].tolist()}")
                    print(f"self.vocab存在性: {hasattr(self, 'vocab') and self.vocab is not None}")
                    if hasattr(self, 'vocab') and self.vocab is not None:
                        print(f"词汇表大小: {len(self.vocab)}")
                        # 检查特殊标记的ID
                        print(f"<PAD> ID: {self.vocab.get('<PAD>')}")
                        print(f"<UNK> ID: {self.vocab.get('<UNK>')}")
                        print(f"<SOS> ID: {self.vocab.get('<SOS>')}")
                        print(f"<EOS> ID: {self.vocab.get('<EOS>')}")
                    
                    # 将生成的ID转换为文本
                    generated_text = self.id_to_text(generated_ids[0].tolist())
                    print(f"转换后的文本: '{generated_text}'")
                    
                    # 添加情感词汇，使回复更有感情
                    emotional_text = self.add_emotion_to_text(generated_text, emotion)
                    
                    return {
                        "generated_text": generated_text,
                        "emotional_text": emotional_text,
                        "reply": emotional_text,
                        "emotion": emotion
                    }
                else:
                    # 分类模型
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0, predicted_class].item()
                    
                    # 生成对话式回复
                    reply = f"根据分析，我预测该输入属于类别 {predicted_class}，置信度为 {confidence:.2f}。"
                    
                    return {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist()[0],
                        "reply": reply
                    }
            else:
                # 图像或视频数据
                if not isinstance(preprocessed_data, torch.Tensor):
                    return {'error': '数据预处理失败'}
                input_tensor = preprocessed_data.unsqueeze(0)
                
                # 进行预测
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                # 生成对话式回复
                reply = f"根据图像分析，我预测该图像属于类别 {predicted_class}，置信度为 {confidence:.2f}。"
                
                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist()[0],
                    "reply": reply
                }
            
        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
    
    def id_to_text(self, ids: List[int]) -> str:
        """将id序列转换为文本
        
        Args:
            ids: id列表
            
        Returns:
            文本
        """
        # 创建反向词汇表
        if hasattr(self, 'vocab') and self.vocab is not None:
            try:
                # 检查vocab结构是否正常
                if isinstance(self.vocab, dict) and all(isinstance(k, (str, int)) for k in self.vocab.keys()):
                    reverse_vocab = {v: k for k, v in self.vocab.items()}
                    
                    # 转换id为文本
                    text = []
                    for idx in ids:
                        # 跳过特殊标记
                        if idx in reverse_vocab and reverse_vocab[idx] not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
                            text.append(reverse_vocab[idx])
                    
                    return " ".join(text)
                else:
                    print(f"警告：词汇表结构异常，类型: {type(self.vocab)}, 键类型: {[type(k) for k in list(self.vocab.keys())[:5]]}")
                    # 如果词汇表结构异常，尝试从嵌入层权重获取词汇表大小
                    if hasattr(self, 'config') and 'vocab_size' in self.config:
                        print(f"使用vocab_size: {self.config['vocab_size']}")
            except Exception as e:
                print(f"创建反向词汇表时出错: {e}")
                print(f"词汇表类型: {type(self.vocab)}")
                # 打印词汇表的一些内容，以便调试
                if hasattr(self.vocab, 'items'):
                    items = list(self.vocab.items())[:10]
                    print(f"词汇表前10项: {items}")
        
        # 如果没有词汇表或创建反向词汇表失败，简单地返回id的字符串表示
        return " ".join([str(id) for id in ids])
    
    def add_emotion_to_text(self, text: str, emotion: int = 0) -> str:
        """为文本添加情感词汇，使回复更有感情
        
        Args:
            text: 原始文本
            emotion: 情感类别 (0-4)
            
        Returns:
            添加了情感的文本
        """
        if not text:
            return text
            
        # 情感前缀
        emotion_prefixes = {
            0: ["😊 很高兴地告诉你，", "😄 开心地说，", "🥰 愉悦地分享，", "😊 喜悦地告知，", "😄 兴奋地说，"],
            1: ["😔 遗憾地说，", "😢 难过地表示，", "😞 悲伤地回答，", "😔 惋惜地说，", "😢 沮丧地回应，"],
            2: ["😠 愤怒地指出，", "😤 恼火地回应，", "😡 气愤地说，", "😠 生气地表示，", "😤 恼怒地说，"],
            3: ["😮 惊讶地发现，", "😲 吃惊地表示，", "😯 意外地说，", "😮 震惊地回应，", "😲 愕然地说，"],
            4: ["😐 平静地告诉你，", "😌 淡定地表示，", "🤔 理性地分析，", "😐 客观地说，", "😌 平和地回应，"]
        }
        
        # 情感后缀
        emotion_suffixes = {
            0: ["！😊", "哦！😄", "呀！🥰", "呢！😊", "哦！🥰"],
            1: ["。😔", "...😢", "呢。😞", "哦。😔", "唉...😢"],
            2: ["！😠", "！😤", "！😡", "！😠", "！😤"],
            3: ["！😮", "！😲", "！😯", "呢！😮", "哦！😲"],
            4: ["。😐", "。😌", "。🤔", "哦。😐", "呢。😌"]
        }
        
        # 随机选择前缀和后缀
        import random
        prefix = random.choice(emotion_prefixes.get(emotion, [""]))
        suffix = random.choice(emotion_suffixes.get(emotion, [""]))
        
        # 如果文本已经有标点符号结尾，去掉后添加情感后缀
        if text and text[-1] in ["。", "！", "？", "..."]:
            text = text[:-1]
        
        return f"{prefix}{text}{suffix}"
    
    def load_emotion_lexicon(self):
        """加载情感词汇表"""
        lexicon_path = "d:\\linai\\data\\emotion_lexicon.txt"
        self.emotion_lexicon = {
            0: [],  # 积极
            1: [],  # 消极
            2: [],  # 愤怒
            3: [],  # 惊讶
            4: []   # 中性
        }
        
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    # 文件格式是：情感类别,词汇,权重
                    emotion = int(parts[0].strip())
                    word = parts[1].strip()
                    weight = float(parts[2].strip())
                    if emotion in self.emotion_lexicon:
                        self.emotion_lexicon[emotion].append((word, weight))
            print(f"情感词汇表加载成功，包含 {sum(len(words) for words in self.emotion_lexicon.values())} 个词汇")
        except Exception as e:
            print(f"加载情感词汇表失败: {e}")
            # 如果加载失败，使用默认的词汇表
            self.emotion_lexicon = {
                0: [("好", 1.0), ("棒", 1.0), ("优秀", 1.0), ("喜欢", 1.0), ("开心", 1.0), ("快乐", 1.0), ("高兴", 1.0), ("满意", 1.0), ("赞", 1.0), ("精彩", 1.0)],
                1: [("坏", 1.0), ("差", 1.0), ("糟糕", 1.0), ("讨厌", 1.0), ("悲伤", 1.0), ("难过", 1.0), ("生气", 1.0), ("失望", 1.0)],
                2: [("愤怒", 1.0), ("生气", 1.0), ("恼火", 1.0), ("恼怒", 1.0), ("气愤", 1.0), ("暴怒", 1.0), ("愤恨", 1.0), ("愤慨", 1.0)],
                3: [("惊讶", 1.0), ("吃惊", 1.0), ("震惊", 1.0), ("诧异", 1.0), ("意外", 1.0), ("愕然", 1.0), ("惊叹", 1.0), ("讶异", 1.0)],
                4: [("一般", 1.0), ("普通", 1.0), ("正常", 1.0), ("平常", 1.0)]
            }
    
    def analyze_emotion(self, text: str) -> int:
        """分析文本的情感
        
        Args:
            text: 输入文本
            
        Returns:
            情感类别 (0-4)
        """
        if not text:
            return 4  # 中性
            
        # 如果情感词汇表已加载，使用词汇表进行分析
        if self.emotion_lexicon:
            emotion_scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            # 预处理文本
            text_lower = text.lower()
            tokens = text_lower.split()
            
            # 计算每个情感类别的得分
            for emotion, words_with_weights in self.emotion_lexicon.items():
                for word, weight in words_with_weights:
                    if word in text_lower:
                        emotion_scores[emotion] += weight
            
            # 考虑否定词的影响
            negative_words = ["不", "没", "无", "否", "非", "未", "别", "莫", "休", "勿"]
            for i, token in enumerate(tokens):
                if token in negative_words and i < len(tokens) - 1:
                    next_token = tokens[i + 1]
                    # 检查下一个词是否在任何情感类别中
                    for emotion, words_with_weights in self.emotion_lexicon.items():
                        for word, weight in words_with_weights:
                            if word == next_token:
                                # 否定词反转情感：积极变消极，消极变积极，保持其他情感不变
                                if emotion == 0:  # 积极 -> 消极
                                    emotion_scores[1] += weight
                                    emotion_scores[0] -= weight
                                elif emotion == 1:  # 消极 -> 积极
                                    emotion_scores[0] += weight
                                    emotion_scores[1] -= weight
            
            # 找出得分最高的情感类别
            max_score = max(emotion_scores.values())
            if max_score > 0:
                return max(emotion_scores, key=emotion_scores.get)
            else:
                return 4  # 中性
        else:
            # 使用简单的关键词匹配（备用方案），考虑否定词
            positive_keywords = ["好", "喜欢", "棒", "优秀", "高兴", "快乐", "满意", "赞", "精彩", "完美", "开心", "愉悦"]
            negative_keywords = ["不好", "不喜欢", "差", "糟糕", "生气", "难过", "失望", "坏", "恶心", "讨厌", "痛苦", "悲伤"]
            angry_keywords = ["愤怒", "生气", "恼火", "恼怒", "气愤", "暴怒", "愤恨", "愤慨"]
            surprise_keywords = ["惊讶", "吃惊", "震惊", "诧异", "意外", "愕然", "惊叹"]
            negative_words = ["不", "没", "无", "否", "非", "未", "别", "莫", "休", "勿"]
            
            text_lower = text.lower()
            
            # 检查是否有否定词
            has_negative = any(word in text_lower for word in negative_words)
            
            if has_negative:
                # 如果有否定词，反转积极/消极判断
                if any(keyword in text_lower for keyword in negative_keywords):
                    return 0  # 有否定词 + 消极关键词 = 积极
                elif any(keyword in text_lower for keyword in positive_keywords):
                    return 1  # 有否定词 + 积极关键词 = 消极
                elif any(keyword in text_lower for keyword in angry_keywords):
                    return 2  # 愤怒不受否定词影响
                elif any(keyword in text_lower for keyword in surprise_keywords):
                    return 3  # 惊讶不受否定词影响
                else:
                    return 4  # 中性
            else:
                # 没有否定词，正常判断
                if any(keyword in text_lower for keyword in angry_keywords):
                    return 2  # 愤怒
                elif any(keyword in text_lower for keyword in surprise_keywords):
                    return 3  # 惊讶
                elif any(keyword in text_lower for keyword in positive_keywords):
                    return 0  # 积极
                elif any(keyword in text_lower for keyword in negative_keywords):
                    return 1  # 消极
                else:
                    return 4  # 中性
