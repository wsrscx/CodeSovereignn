import os
import fnmatch

class ProjectManager:
    def __init__(self):
        """初始化项目管理器"""
        # 忽略的文件和目录模式
        self.ignore_patterns = [
            "__pycache__", 
            "*.pyc", 
            ".git", 
            ".vscode", 
            "venv", 
            "env",
            "*.exe",
            "*.dll",
            "*.so",
            "*.dylib",
            "node_modules",
            "*.zip",
            "*.tar.gz",
            "*.rar"
        ]
        
        # 支持的文本文件扩展名
        self.text_extensions = [
            ".py", ".js", ".html", ".css", ".json", ".xml", ".md", ".txt",
            ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bat",
            ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".ts", ".jsx", ".tsx"
        ]
        
        # 最大文件大小（字节）
        self.max_file_size = 1024 * 1024  # 1MB
    
    def scan_project(self, project_path):
        """扫描项目文件夹，返回所有文本文件的路径
        
        Args:
            project_path: 项目文件夹路径
            
        Returns:
            文本文件路径列表
        """
        text_files = []
        
        for root, dirs, files in os.walk(project_path):
            # 过滤掉忽略的目录
            dirs[:] = [d for d in dirs if not self._should_ignore(d)]
            
            for file in files:
                if self._should_ignore(file):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # 检查文件大小
                try:
                    if os.path.getsize(file_path) > self.max_file_size:
                        continue
                except OSError:
                    continue
                
                # 检查是否为文本文件
                if self._is_text_file(file_path):
                    text_files.append(file_path)
        
        return text_files
    
    def read_file(self, file_path):
        """读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容，如果读取失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"无法读取文件 {file_path}: {str(e)}")
            try:
                # 尝试使用其他编码
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception:
                return None
    
    def save_file(self, file_path, content):
        """保存文件内容
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"无法保存文件 {file_path}: {str(e)}")
            return False
    
    def _should_ignore(self, name):
        """检查文件或目录是否应该被忽略
        
        Args:
            name: 文件或目录名
            
        Returns:
            是否应该被忽略
        """
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.ignore_patterns)
    
    def _is_text_file(self, file_path):
        """检查是否为文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为文本文件
        """
        # 检查扩展名
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.text_extensions:
            return True
        
        # 如果没有扩展名或扩展名不在列表中，尝试读取文件的前几个字节
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                # 检查是否包含空字节（二进制文件通常包含空字节）
                if b'\x00' in sample:
                    return False
                # 尝试解码为文本
                try:
                    sample.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    try:
                        sample.decode('gbk')
                        return True
                    except UnicodeDecodeError:
                        return False
        except Exception:
            return False