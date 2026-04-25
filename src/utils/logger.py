import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

class LLMLogger:
    """LLM交互日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        """初始化日志记录器
        
        Args:
            log_dir: 日志文件存储目录
        """
        self.log_dir = log_dir
        self._ensure_log_dir()
        self.current_log_file = self._create_log_file()
        
        # 设置普通日志记录器
        self.logger = logging.getLogger("KGConstructor")
        self.logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f"kg_constructor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _ensure_log_dir(self):
        """确保日志目录存在"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _create_log_file(self) -> str:
        """创建新的日志文件
        
        Returns:
            日志文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"llm_interaction_{timestamp}.jsonl")
        
    def log_interaction(self, 
                       prompt: str,
                       response: str,
                       system_prompt: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """记录一次LLM交互
        
        Args:
            prompt: 输入的提示词
            response: LLM的响应
            system_prompt: 系统提示词（如果有）
            metadata: 额外的元数据信息
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "system_prompt": system_prompt,
            "metadata": metadata or {}
        }
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    def start_new_session(self):
        """开始新的会话，创建新的日志文件"""
        self.current_log_file = self._create_log_file()
        
    def debug(self, message: str, *args, **kwargs):
        """记录调试信息"""
        self.logger.debug(message, *args, **kwargs)
        
    def info(self, message: str, *args, **kwargs):
        """记录普通信息"""
        self.logger.info(message, *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs):
        """记录警告信息"""
        self.logger.warning(message, *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs):
        """记录错误信息"""
        self.logger.error(message, *args, **kwargs)
        
    def critical(self, message: str, *args, **kwargs):
        """记录严重错误信息"""
        self.logger.critical(message, *args, **kwargs) 