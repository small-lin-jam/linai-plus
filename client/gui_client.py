import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import os
import sys
import numpy as np
from PIL import Image, ImageTk
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.abspath('d:/linai'))

from client.model_loader import ModelLoader
from client.markdown_parser import MarkdownParser

class ModelClient:
    """模型客户端GUI应用"""
    
    def __init__(self, root):
        """初始化客户端GUI
        
        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("AI模型客户端")
        self.root.geometry("800x600")
        
        # 初始化变量
        self.model_loader = None
        self.markdown_parser = MarkdownParser()
        self.model_path = None
        self.current_image = None
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 顶部模型加载区域
        model_frame = ttk.LabelFrame(self.root, text="模型管理")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        self.model_status_var = tk.StringVar(value="未加载模型")
        ttk.Label(model_frame, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=10)
        
        # 情感选择区域
        emotion_frame = ttk.LabelFrame(self.root, text="情感设置")
        emotion_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 情感类型
        self.emotion_var = tk.IntVar(value=0)  # 默认积极
        emotion_types = [
            (0, "😊 积极"),
            (1, "😔 消极"),
            (2, "😠 愤怒"),
            (3, "😮 惊讶"),
            (4, "😐 中性")
        ]
        
        # 自动检测复选框
        self.auto_detect_emotion = tk.BooleanVar(value=True)
        ttk.Checkbutton(emotion_frame, text="自动检测用户情感", variable=self.auto_detect_emotion).pack(side=tk.LEFT, padx=10)
        
        # 情感选择下拉菜单
        ttk.Label(emotion_frame, text="回复情感:").pack(side=tk.LEFT, padx=10)
        self.emotion_combobox = ttk.Combobox(emotion_frame, textvariable=self.emotion_var, values=[emotion[0] for emotion in emotion_types], 
                                            state="readonly", width=10)
        self.emotion_combobox.pack(side=tk.LEFT, padx=5)
        # 设置显示值
        self.emotion_combobox.bind("<<ComboboxSelected>>", self.on_emotion_change)
        self.emotion_combobox.config(values=[emotion[1] for emotion in emotion_types])
        self.emotion_combobox.current(0)
        
        # 问答区域（聊天界面）
        chat_frame = ttk.LabelFrame(self.root, text="问答界面")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 聊天记录显示区域
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=80, height=30, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 设置字体和颜色
        self.chat_display.tag_configure("user", foreground="blue", font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("ai", foreground="green", font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("system", foreground="gray", font=('Arial', 9, 'italic'))
        self.chat_display.tag_configure("message", font=('Arial', 10))
        
        # 输入区域
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=70, height=3)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.insert(tk.END, "请输入您的问题...")
        # 绑定回车键发送
        self.input_text.bind("<Return>", lambda event: self.send_question())
        self.input_text.bind("<Shift-Return>", lambda event: self.input_text.insert(tk.END, "\n"))
        
        # 发送按钮
        ttk.Button(input_frame, text="发送", command=self.send_question).pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)
    
    def load_model(self):
        """加载模型文件"""
        # 如果没有指定路径，打开文件选择对话框
        if not self.model_path_var.get():
            file_path = filedialog.askopenfilename(
                filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
            )
            if not file_path:
                return
            self.model_path_var.set(file_path)
        
        try:
            # 加载模型
            self.model_path = self.model_path_var.get()
            self.model_loader = ModelLoader(self.model_path)
            
            # 更新状态
            self.model_status_var.set(f"已加载模型: {os.path.basename(self.model_path)}")
            self.add_message("系统", f"模型加载成功！数据类型: {self.model_loader.data_type}")
            
        except Exception as e:
            self.model_status_var.set(f"模型加载失败: {str(e)}")
            self.add_message("系统", f"错误: {str(e)}")
    
    def on_emotion_change(self, event):
        """情感选择变化处理"""
        # 这个方法在选择变化时被调用，但由于我们使用IntVar绑定，实际值会自动更新
        pass
    
    def send_question(self):
        """发送问题并获取回答"""
        if not self.model_loader:
            self.add_message("系统", "请先加载模型")
            return
        
        try:
            # 获取输入文本
            input_text = self.input_text.get("1.0", tk.END).strip()
            if not input_text:
                self.add_message("系统", "请输入您的问题")
                return
            
            # 清空输入框
            self.input_text.delete("1.0", tk.END)
            
            # 添加用户问题到聊天记录
            self.add_message("用户", input_text)
            
            # 处理输入文本
            processed_text = self.markdown_parser.process_input(input_text)
            
            # 确定使用的情感
            if self.auto_detect_emotion.get():
                # 自动检测用户输入的情感
                detected_emotion = self.model_loader.analyze_emotion(input_text)
                emotion = detected_emotion
                emotion_name = ["积极", "消极", "愤怒", "惊讶", "中性"][emotion]
                self.add_message("系统", f"检测到您的情感：{emotion_name} ({['😊','😔','😠','😮','😐'][emotion]})")
            else:
                # 使用选择的情感
                emotion = self.emotion_var.get()
            
            # 进行预测
            result = self.model_loader.predict(processed_text, emotion=emotion)
            
            # 显示结果
            if "error" in result:
                self.add_message("AI", f"预测失败: {result['error']}")
            else:
                # 使用模型返回的对话式回复
                if 'reply' in result:
                    self.add_message("AI", result['reply'])
                    # 如果有原始生成文本，显示为系统消息（用于调试）
                    if 'generated_text' in result and result['generated_text'] != result['reply']:
                        self.add_message("系统", f"原始生成: {result['generated_text']}")
                else:
                    # 回退到之前的格式
                    if result['predicted_class'] == 0:
                        sentiment = "负面"
                    else:
                        sentiment = "正面"
                    
                    answer = f"情感分析结果：{sentiment}\n置信度：{result['confidence']:.4f}"
                    self.add_message("AI", answer)
                
        except Exception as e:
            self.add_message("系统", f"处理失败: {str(e)}")
    
    def add_message(self, sender, message):
        """添加消息到聊天记录
        
        Args:
            sender: 发送者（"用户", "AI", "系统"）
            message: 消息内容
        """
        self.chat_display.config(state=tk.NORMAL)
        
        # 插入发送者名称，使用不同的标签
        self.chat_display.insert(tk.END, f"{sender}: ", sender.lower())
        
        # 插入消息内容
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        
        # 滚动到底部
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

def main():
    """主函数"""
    root = tk.Tk()
    app = ModelClient(root)
    root.mainloop()

if __name__ == "__main__":
    main()
