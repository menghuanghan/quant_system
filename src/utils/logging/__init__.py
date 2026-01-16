"""
日志配置模块
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志"""
        if not hasattr(self, 'initialized'):
            self.logger = None
            self.initialized = False
    
    def setup(self, 
              name: str = 'quant_system',
              level: str = 'INFO',
              log_dir: Optional[str] = None,
              console: bool = True,
              file: bool = True):
        """
        配置日志
        
        Args:
            name: 日志名称
            level: 日志级别
            log_dir: 日志目录
            console: 是否输出到控制台
            file: 是否输出到文件
        """
        if self.initialized:
            return self.logger
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件handler
        if file:
            if log_dir is None:
                log_dir = 'logs'
            
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # 按日期创建日志文件
            log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.initialized = True
        return self.logger
    
    def get_logger(self) -> logging.Logger:
        """
        获取logger实例
        
        Returns:
            Logger实例
        """
        if not self.initialized:
            self.setup()
        return self.logger


# 创建全局logger实例
logger_instance = Logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger
    
    Args:
        name: logger名称
        
    Returns:
        Logger实例
    """
    if name:
        return logging.getLogger(name)
    return logger_instance.get_logger()
