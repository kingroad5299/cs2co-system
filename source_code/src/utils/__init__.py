"""
工具函数模块
提供通用的工具函数和辅助类

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__all__ = [
    "setup_logger",
    "ConfigManager",
    "DatabaseManager",
    "CacheManager",
    "DateUtil",
    "FileUtil",
    "ValidationUtil",
    "PerformanceTimer"
]

# 模块描述
__description__ = """
工具函数模块提供CS2CO系统所需的通用工具和辅助功能，包括：
1. 日志管理：灵活的日志配置和输出
2. 配置管理：配置文件的加载和解析
3. 数据库管理：数据库连接和会话管理
4. 缓存管理：内存和Redis缓存
5. 日期工具：日期时间处理函数
6. 文件工具：文件读写和路径处理
7. 验证工具：数据验证和清洗
8. 性能计时：代码性能分析和优化
"""

# 版本信息
__version__ = "1.0.0"

# 导入工具类
try:
    from .logger import setup_logger
    from .config_manager import ConfigManager
    from .database_manager import DatabaseManager
    from .cache_manager import CacheManager
    from .date_util import DateUtil
    from .file_util import FileUtil
    from .validation_util import ValidationUtil
    from .performance_timer import PerformanceTimer
except ImportError as e:
    print(f"Warning: Some utility modules may not be available: {e}")

    # 提供占位函数
    def setup_logger(name: str = "cs2co"):
        """日志设置占位函数"""
        import logging
        return logging.getLogger(name)

    class ConfigManager:
        """配置管理器占位类"""
        def __init__(self, config_path: str = None):
            self.config = {}

    class DatabaseManager:
        """数据库管理器占位类"""
        pass

    class CacheManager:
        """缓存管理器占位类"""
        pass

    class DateUtil:
        """日期工具占位类"""
        @staticmethod
        def get_current_time():
            import datetime
            return datetime.datetime.now()

    class FileUtil:
        """文件工具占位类"""
        @staticmethod
        def read_file(file_path: str):
            with open(file_path, 'r') as f:
                return f.read()

    class ValidationUtil:
        """验证工具占位类"""
        @staticmethod
        def validate_email(email: str) -> bool:
            import re
            return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

    class PerformanceTimer:
        """性能计时器占位类"""
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

def get_utils_info() -> dict:
    """获取工具模块信息"""
    return {
        "module": "utils",
        "version": __version__,
        "description": __description__.strip(),
        "utilities": [
            "setup_logger（日志管理）",
            "ConfigManager（配置管理）",
            "DatabaseManager（数据库管理）",
            "CacheManager（缓存管理）",
            "DateUtil（日期工具）",
            "FileUtil（文件工具）",
            "ValidationUtil（验证工具）",
            "PerformanceTimer（性能计时）"
        ]
    }