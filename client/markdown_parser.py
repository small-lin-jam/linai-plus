import re
import emoji

class MarkdownParser:
    """Markdown解析器，用于处理Markdown格式的用户输入和表情"""
    
    def __init__(self):
        """初始化Markdown解析器"""
        self.emoji_map = self._load_emoji_map()
    
    def _load_emoji_map(self) -> dict:
        """加载表情符号映射表
        
        Returns:
            表情符号映射表
        """
        # 基础表情符号映射
        return {
            'smile': ':)',
            'laugh': '😂',
            'sad': '😢',
            'angry': '😠',
            'surprised': '😮',
            'heart': '❤️',
            'thumbsup': '👍',
            'thumbsdown': '👎',
            'clap': '👏',
            'fire': '🔥',
            'star': '⭐',
            'thinking': '🤔',
            'cool': '😎',
            'love': '😍',
            'cry': '😢',
            'happy': '😊',
            'wink': '😉',
            'confused': '😕',
            'tired': '😴',
            'excited': '🎉',
            'nerd': '🤓'
        }
    
    def parse_markdown(self, text: str) -> str:
        """解析Markdown格式的文本
        
        Args:
            text: Markdown格式的文本
            
        Returns:
            解析后的纯文本
        """
        # 移除Markdown标记
        
        # 移除标题标记 (#)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 移除加粗标记 (**)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # 移除斜体标记 (*)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # 移除代码块标记 (```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 移除行内代码标记 (`)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # 移除链接标记 [text](url)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # 移除图片标记 ![alt](url)
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        
        # 移除列表标记 (-)
        text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
        
        # 移除编号列表标记 (1., 2., etc.)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # 移除引用标记 (>)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # 移除水平分隔线 (---, ***)
        text = re.sub(r'^(-{3}|\*{3})\s*$', '', text, flags=re.MULTILINE)
        
        # 移除多余空行
        text = re.sub(r'\n{3,}', r'\n\n', text)
        
        return text.strip()
    
    def parse_emojis(self, text: str) -> str:
        """解析表情符号
        
        Args:
            text: 包含表情符号的文本
            
        Returns:
            解析后的文本
        """
        # 解析命名表情符号 (如 :smile: -> 😊)
        for name, emoji_char in self.emoji_map.items():
            text = text.replace(f':{name}:', emoji_char)
        
        # 解析Unicode表情符号
        text = emoji.emojize(emoji.demojize(text))
        
        return text
    
    def process_input(self, input_text: str) -> str:
        """处理用户输入的文本
        
        Args:
            input_text: 用户输入的文本
            
        Returns:
            处理后的文本
        """
        # 解析Markdown
        text = self.parse_markdown(input_text)
        
        # 解析表情符号
        text = self.parse_emojis(text)
        
        return text
