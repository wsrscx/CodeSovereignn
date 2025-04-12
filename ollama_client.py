import requests
import json
import time
import os

class OllamaClient:
    def __init__(self, api_url="http://localhost:11434/api", model_name="deepseek-coder-v2:latest"):
        """初始化Ollama客户端
        
        Args:
            api_url: Ollama API的URL
            model_name: 使用的模型名称
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = 10000  # 设置超时为10000秒
    
    def update_settings(self, api_url, model_name):
        """更新API设置
        
        Args:
            api_url: 新的API URL
            model_name: 新的模型名称
        """
        self.api_url = api_url
        self.model_name = model_name
    
    def generate(self, prompt):
        """生成单个响应
        
        Args:
            prompt: 提示词
            
        Returns:
            生成的文本响应
        """
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 4096,  # 增加生成的token数量
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(error_msg)
                return f"生成失败: {error_msg}"
        
        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            print(error_msg)
            return f"生成失败: {error_msg}"
    
    def generate_with_context(self, prompt, project_path):
        """分段处理大型上下文
        
        Args:
            prompt: 基础提示词
            project_path: 项目路径，用于读取文件内容
            
        Returns:
            生成的响应列表
        """
        responses = []
        
        # 第一阶段：项目规划和结构设计
        planning_prompt = prompt + "\n\n首先，请分析需求并提供项目的整体规划和结构设计。"
        planning_response = self.generate(planning_prompt)
        responses.append(planning_response)
        
        # 从响应中提取计划信息
        planned_files = self._extract_planned_files(planning_response)
        
        # 第二阶段：逐个实现文件
        if planned_files:
            # 如果AI已经规划了文件，按计划实现
            for file_group in self._group_files(planned_files, 3):  # 每组最多3个文件
                files_prompt = prompt + "\n\n请实现以下文件:\n" + "\n".join([f"- {f}" for f in file_group])
                
                # 如果是修改现有项目，添加现有文件内容
                for file_path in file_group:
                    full_path = os.path.join(project_path, file_path) if not os.path.isabs(file_path) else file_path
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            files_prompt += f"\n\n现有文件 {file_path} 的内容:\n```\n{file_content}\n```"
                        except Exception as e:
                            files_prompt += f"\n\n无法读取文件 {file_path}: {str(e)}"
                
                implementation_response = self.generate(files_prompt)
                responses.append(implementation_response)
        else:
            # 如果AI没有明确规划文件，分批次生成代码
            implementation_prompt = prompt + "\n\n请开始实现项目的核心文件。"
            implementation_response = self.generate(implementation_prompt)
            responses.append(implementation_response)
            
            # 继续生成其他必要文件
            followup_prompt = prompt + "\n\n请继续实现项目的其他必要文件，包括配置文件、辅助模块和文档。"
            followup_response = self.generate(followup_prompt)
            responses.append(followup_response)
        
        # 第三阶段：完善和测试
        finalization_prompt = prompt + "\n\n请检查项目的完整性，确保所有必要的文件都已创建，并提供如何运行和测试项目的说明。"
        finalization_response = self.generate(finalization_prompt)
        responses.append(finalization_response)
        
        return responses
    
    def _extract_planned_files(self, response):
        """从响应中提取计划创建的文件列表
        
        Args:
            response: AI的响应文本
            
        Returns:
            计划创建的文件路径列表
        """
        import re
        
        # 尝试从文件模式中提取
        file_pattern = r"```file:(.+?)\n"
        file_matches = re.findall(file_pattern, response)
        
        if file_matches:
            return [path.strip() for path in file_matches]
        
        # 尝试从项目结构描述中提取
        structure_pattern = r"(?:项目结构|文件结构|目录结构)[:：]?\s*(?:\n|.)*?(?:```|\n\n)"
        structure_match = re.search(structure_pattern, response)
        
        if structure_match:
            structure_text = structure_match.group(0)
            # 查找结构中的文件路径
            path_pattern = r"[\w\-\.]+\.[\w]+"  # 简单的文件名模式
            return re.findall(path_pattern, structure_text)
        
        return []
    
    def _group_files(self, files, group_size):
        """将文件列表分组
        
        Args:
            files: 文件路径列表
            group_size: 每组的大小
            
        Returns:
            分组后的列表
        """
        return [files[i:i+group_size] for i in range(0, len(files), group_size)]