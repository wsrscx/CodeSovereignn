import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import queue
import json
import time

# 导入自定义模块
from ollama_client import OllamaClient
from project_manager import ProjectManager
from ui_manager import UIManager

class AIProjectBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("AI项目构建器")
        self.root.geometry("1000x700")
        
        # 创建队列用于线程间通信
        self.message_queue = queue.Queue()
        
        # 初始化组件
        self.ollama_client = OllamaClient()
        self.project_manager = ProjectManager()
        self.ui_manager = UIManager(self.root, self.message_queue, self.start_project_generation, 
                                   self.select_project_folder, self.update_ollama_settings)
        
        # 设置默认值
        self.project_path = ""
        self.requirement = ""
        self.model_name = "deepseek-coder-v2:latest"
        self.ollama_api_url = "http://localhost:11434/api"
        
        # 启动UI更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self.update_ui_from_queue)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def select_project_folder(self):
        """选择项目文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            self.project_path = folder
            self.ui_manager.update_project_path(folder)
            self.message_queue.put({"type": "status", "message": f"已选择项目路径: {folder}"})
    
    def update_ollama_settings(self, api_url, model_name):
        """更新Ollama设置"""
        self.ollama_api_url = api_url
        self.model_name = model_name
        self.ollama_client.update_settings(api_url, model_name)
        self.message_queue.put({"type": "status", "message": f"已更新Ollama设置 - API: {api_url}, 模型: {model_name}"})
    
    def start_project_generation(self, requirement, is_new_project):
        """启动项目生成过程"""
        self.requirement = requirement
        
        # 检查项目路径
        if not self.project_path:
            self.message_queue.put({"type": "error", "message": "请先选择项目文件夹"})
            return
        
        # 检查需求
        if not requirement.strip():
            self.message_queue.put({"type": "error", "message": "请输入项目需求"})
            return
        
        # 清空日志并显示开始信息
        self.ui_manager.clear_logs()
        self.message_queue.put({"type": "status", "message": "开始处理项目需求..."})
        
        # 在新线程中运行项目生成
        generation_thread = threading.Thread(
            target=self.run_project_generation,
            args=(requirement, is_new_project)
        )
        generation_thread.daemon = True
        generation_thread.start()
    
    def run_project_generation(self, requirement, is_new_project):
        """在单独的线程中运行项目生成"""
        try:
            # 准备项目上下文
            if is_new_project:
                self.message_queue.put({"type": "status", "message": "创建新项目..."})
                os.makedirs(self.project_path, exist_ok=True)
            else:
                self.message_queue.put({"type": "status", "message": "分析现有项目..."})
                project_files = self.project_manager.scan_project(self.project_path)
                self.message_queue.put({"type": "status", "message": f"找到 {len(project_files)} 个文件"})
            
            # 构建提示词
            prompt = self.build_prompt(requirement, is_new_project)
            
            # 调用AI生成代码
            self.message_queue.put({"type": "status", "message": "正在使用AI生成代码..."})
            self.message_queue.put({"type": "plan", "message": "分析需求并规划项目结构"})
            
            # 分段处理大型上下文
            responses = self.ollama_client.generate_with_context(prompt, self.project_path)
            
            # 处理AI响应
            for i, response in enumerate(responses):
                self.process_ai_response(response, i+1, len(responses))
            
            self.message_queue.put({"type": "status", "message": "项目生成完成!"})
            self.message_queue.put({"type": "plan", "message": "所有任务已完成"})
            
        except Exception as e:
            self.message_queue.put({"type": "error", "message": f"错误: {str(e)}"})
    
    def build_prompt(self, requirement, is_new_project):
        """构建提示词"""
        if is_new_project:
            prompt = (
                "你是一个专业的软件开发助手，精通各种编程语言和框架。\n\n"
                "# 任务\n"
                "根据用户的需求，创建一个完整的项目，包括所有必要的代码文件、配置文件和文档。\n\n"
                "# 要求\n"
                "1. 分析用户需求，确定合适的项目结构和技术栈\n"
                "2. 创建所有必要的文件，包括源代码、配置文件和文档\n"
                "3. 确保代码质量高，没有明显的bug和错误\n"
                "4. 提供清晰的注释和文档\n"
                "5. 考虑代码的可维护性和可扩展性\n\n"
                "# 输出格式\n"
                "对于每个文件，请使用以下格式：\n"
                "```file:<文件路径>\n"
                "<文件内容>\n"
                "```\n\n"
                "在文件之间，你可以添加解释说明，说明你的设计决策和下一步计划。\n\n"
                "# 用户需求\n"
                f"{requirement}\n"
            )
        else:
            prompt = (
                "你是一个专业的软件开发助手，精通各种编程语言和框架。\n\n"
                "# 任务\n"
                "根据用户的需求，修改现有项目，添加新功能或修复问题。\n\n"
                "# 要求\n"
                "1. 分析用户需求和现有代码\n"
                "2. 修改、添加或删除必要的文件\n"
                "3. 确保代码质量高，没有明显的bug和错误\n"
                "4. 保持代码风格的一致性\n"
                "5. 考虑代码的可维护性和可扩展性\n\n"
                "# 输出格式\n"
                "对于每个需要创建或修改的文件，请使用以下格式：\n"
                "```file:<文件路径>\n"
                "<文件内容>\n"
                "```\n\n"
                "在文件之间，你可以添加解释说明，说明你的设计决策和下一步计划。\n\n"
                "# 用户需求\n"
                f"{requirement}\n\n"
                "# 现有项目文件\n"
            )
            # 添加项目文件列表和内容摘要
            project_files = self.project_manager.scan_project(self.project_path)
            for file_path in project_files[:10]:  # 限制文件数量，避免超出上下文限制
                relative_path = os.path.relpath(file_path, self.project_path)
                prompt += f"- {relative_path}\n"
            
            prompt += "\n如果需要查看更多文件内容，请在回复中说明。\n"
        
        return prompt
    
    def process_ai_response(self, response, current_part, total_parts):
        """处理AI的响应，提取文件并保存"""
        self.message_queue.put({"type": "status", "message": f"处理AI响应 ({current_part}/{total_parts})..."})
        
        # 解析响应中的文件
        file_pattern = r"```file:(.+?)\n([\s\S]+?)```"
        import re
        matches = re.finditer(file_pattern, response)
        
        files_processed = 0
        for match in matches:
            file_path = match.group(1).strip()
            file_content = match.group(2)
            
            # 确保文件路径是绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.project_path, file_path)
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            files_processed += 1
            relative_path = os.path.relpath(file_path, self.project_path)
            self.message_queue.put({"type": "file", "message": f"已保存文件: {relative_path}"})
        
        # 提取计划信息
        plans = re.finditer(r"(?:下一步计划|接下来我将|计划如下|我的计划是)[:：]?\s*(.+?)(?=\n\n|$)", response)
        for plan in plans:
            plan_text = plan.group(1).strip()
            if plan_text:
                self.message_queue.put({"type": "plan", "message": plan_text})
        
        if files_processed == 0:
            self.message_queue.put({"type": "status", "message": "此部分响应中没有找到文件"})
    
    def update_ui_from_queue(self):
        """从队列更新UI"""
        while self.running:
            try:
                # 非阻塞方式获取消息
                message = self.message_queue.get(block=False)
                message_type = message.get("type", "")
                message_text = message.get("message", "")
                
                if message_type == "status":
                    self.ui_manager.update_status(message_text)
                elif message_type == "error":
                    self.ui_manager.update_error(message_text)
                elif message_type == "file":
                    self.ui_manager.update_file_log(message_text)
                elif message_type == "plan":
                    self.ui_manager.update_plan(message_text)
                
                self.message_queue.task_done()
            except queue.Empty:
                pass
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.1)
            
            # 更新UI
            try:
                self.root.update()
            except tk.TclError:
                # 窗口已关闭
                self.running = False
                break
    
    def on_closing(self):
        """窗口关闭时的处理"""
        self.running = False
        self.root.destroy()
        sys.exit(0)

def main():
    root = tk.Tk()
    app = AIProjectBuilder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()