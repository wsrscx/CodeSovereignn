import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import queue
import threading

class UIManager:
    def __init__(self, root, message_queue, start_callback, folder_callback, settings_callback):
        """初始化UI管理器
        
        Args:
            root: tkinter根窗口
            message_queue: 消息队列，用于线程间通信
            start_callback: 开始生成项目的回调函数
            folder_callback: 选择文件夹的回调函数
            settings_callback: 更新设置的回调函数
        """
        self.root = root
        self.message_queue = message_queue
        self.start_callback = start_callback
        self.folder_callback = folder_callback
        self.settings_callback = settings_callback
        
        # 创建UI组件
        self._create_ui()
        
        # 设置默认值
        self.api_url_var.set("http://localhost:11434/api")
        self.model_var.set("deepseek-coder-v2:latest")
    
    def _create_ui(self):
        """创建UI组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧和右侧面板
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # ===== 左侧面板 =====
        # 设置部分
        settings_frame = ttk.LabelFrame(left_frame, text="Ollama 设置")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # API URL
        api_frame = ttk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_frame, text="API URL:").pack(side=tk.LEFT)
        self.api_url_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.api_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 模型选择
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="模型名称:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 应用设置按钮
        ttk.Button(
            settings_frame, 
            text="应用设置", 
            command=self._apply_settings
        ).pack(fill=tk.X, pady=5)
        
        # 项目路径部分
        path_frame = ttk.LabelFrame(left_frame, text="项目路径")
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(fill=tk.X, pady=5)
        
        self.path_var = tk.StringVar()
        ttk.Entry(path_input_frame, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            path_input_frame, 
            text="浏览...", 
            command=self.folder_callback
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # 需求输入部分
        req_frame = ttk.LabelFrame(left_frame, text="项目需求")
        req_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.requirement_text = scrolledtext.ScrolledText(req_frame, wrap=tk.WORD, height=10)
        self.requirement_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 项目类型选择
        type_frame = ttk.Frame(left_frame)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.project_type_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(
            type_frame, 
            text="创建新项目", 
            variable=self.project_type_var, 
            value=True
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            type_frame, 
            text="修改现有项目", 
            variable=self.project_type_var, 
            value=False
        ).pack(side=tk.LEFT)
        
        # 开始按钮
        ttk.Button(
            left_frame, 
            text="开始生成", 
            command=self._start_generation
        ).pack(fill=tk.X, ipady=5)
        
        # ===== 右侧面板 =====
        # 状态显示
        status_frame = ttk.LabelFrame(right_frame, text="状态")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(pady=5)
        
        # 计划显示
        plan_frame = ttk.LabelFrame(right_frame, text="AI计划")
        plan_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.plan_var = tk.StringVar(value="等待开始...")
        ttk.Label(plan_frame, textvariable=self.plan_var, wraplength=400).pack(pady=5)
        
        # 日志显示
        log_frame = ttk.LabelFrame(right_frame, text="执行日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)
    
    def _apply_settings(self):
        """应用Ollama设置"""
        api_url = self.api_url_var.get().strip()
        model = self.model_var.get().strip()
        
        if not api_url or not model:
            self.update_error("API URL和模型名称不能为空")
            return
        
        self.settings_callback(api_url, model)
    
    def _start_generation(self):
        """开始生成项目"""
        requirement = self.requirement_text.get("1.0", tk.END).strip()
        is_new_project = self.project_type_var.get()
        
        if not requirement:
            self.update_error("请输入项目需求")
            return
        
        self.start_callback(requirement, is_new_project)
    
    def update_project_path(self, path):
        """更新项目路径
        
        Args:
            path: 新的项目路径
        """
        self.path_var.set(path)
    
    def update_status(self, message):
        """更新状态信息
        
        Args:
            message: 状态消息
        """
        self.status_var.set(message)
        self._append_log(f"[状态] {message}")
    
    def update_error(self, message):
        """更新错误信息
        
        Args:
            message: 错误消息
        """
        self.status_var.set(f"错误: {message}")
        self._append_log(f"[错误] {message}", "error")
    
    def update_file_log(self, message):
        """更新文件操作日志
        
        Args:
            message: 文件操作消息
        """
        self._append_log(f"[文件] {message}", "file")
    
    def update_plan(self, message):
        """更新AI计划
        
        Args:
            message: 计划消息
        """
        self.plan_var.set(message)
        self._append_log(f"[计划] {message}", "plan")
    
    def clear_logs(self):
        """清空日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _append_log(self, message, log_type="info"):
        """添加日志
        
        Args:
            message: 日志消息
            log_type: 日志类型，可以是info、error、file或plan
        """
        self.log_text.config(state=tk.NORMAL)
        
        # 设置标签
        if log_type == "error":
            tag = "error"
        elif log_type == "file":
            tag = "file"
        elif log_type == "plan":
            tag = "plan"
        else:
            tag = "info"
        
        # 添加日志
        end_pos = self.log_text.index(tk.END)
        self.log_text.insert(tk.END, message + "\n")
        line_start = float(end_pos.split(".")[0])
        line_end = float(self.log_text.index(tk.END).split(".")[0])
        
        # 应用标签
        for i in range(int(line_start), int(line_end)):
            self.log_text.tag_add(tag, f"{i}.0", f"{i}.end")
        
        # 配置标签样式
        if not hasattr(self, "_tags_configured"):
            self.log_text.tag_configure("error", foreground="red")
            self.log_text.tag_configure("file", foreground="blue")
            self.log_text.tag_configure("plan", foreground="green")
            self.log_text.tag_configure("info", foreground="black")
            self._tags_configured = True
        
        # 滚动到底部
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)