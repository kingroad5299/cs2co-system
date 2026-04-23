"""
日志管理模块
提供灵活、可配置的日志系统

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import os
import sys
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# 第三方日志库
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    loguru_logger = None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 默认配置
DEFAULT_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_path": "logs/cs2co_system.log",
    "max_file_size_mb": 100,
    "backup_count": 10,
    "console_output": True,
    "json_format": False,
    "use_loguru": False,
    "use_structlog": False
}


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if hasattr(record, 'structured_data'):
            # 结构化日志
            structured_data = getattr(record, 'structured_data', {})
            message = record.getMessage()

            log_entry = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "name": record.name,
                "message": message,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                **structured_data
            }

            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_entry, ensure_ascii=False)
        else:
            # 传统格式
            return super().format(record)


def setup_logger(
    name: str = "cs2co",
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    设置和配置日志系统

    Args:
        name: 日志器名称
        config: 配置字典
        log_dir: 日志目录

    Returns:
        配置好的日志器
    """
    # 合并配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)

    # 确保日志目录存在
    if log_dir:
        final_config["file_path"] = os.path.join(log_dir, "cs2co_system.log")

    log_file_path = final_config["file_path"]
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(final_config["level"].upper(), logging.INFO))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    if final_config["console_output"]:
        console_handler = logging.StreamHandler(sys.stdout)
        if final_config["json_format"]:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = logging.Formatter(
                final_config["format"],
                datefmt=final_config["date_format"]
            )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=final_config["max_file_size_mb"] * 1024 * 1024,
        backupCount=final_config["backup_count"],
        encoding='utf-8'
    )

    if final_config["json_format"]:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            final_config["format"],
            datefmt=final_config["date_format"]
        )

    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 添加额外信息
    logger.propagate = False

    return logger


def setup_loguru_logger(
    name: str = "cs2co",
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None
):
    """
    使用loguru设置日志系统

    Args:
        name: 日志器名称
        config: 配置字典
        log_dir: 日志目录

    Returns:
        loguru日志器
    """
    if not LOGURU_AVAILABLE:
        raise ImportError("loguru is not installed. Please install it with: pip install loguru")

    # 合并配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)

    # 确保日志目录存在
    if log_dir:
        final_config["file_path"] = os.path.join(log_dir, "cs2co_system_{time}.log")

    log_file_path = final_config["file_path"]
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 配置loguru
    loguru_config = {
        "handlers": []
    }

    # 控制台输出
    if final_config["console_output"]:
        loguru_config["handlers"].append({
            "sink": sys.stdout,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "level": final_config["level"]
        })

    # 文件输出
    loguru_config["handlers"].append({
        "sink": log_file_path,
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        "level": final_config["level"],
        "rotation": f"{final_config['max_file_size_mb']} MB",
        "retention": f"{final_config['backup_count']} days",
        "compression": "zip",
        "encoding": "utf-8"
    })

    # 应用配置
    loguru_logger.remove()
    for handler in loguru_config["handlers"]:
        loguru_logger.add(**handler)

    return loguru_logger.bind(name=name)


def setup_structlog_logger(
    name: str = "cs2co",
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None
):
    """
    使用structlog设置结构化日志系统

    Args:
        name: 日志器名称
        config: 配置字典
        log_dir: 日志目录

    Returns:
        structlog日志器
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError("structlog is not installed. Please install it with: pip install structlog")

    # 合并配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)

    # 确保日志目录存在
    if log_dir:
        final_config["file_path"] = os.path.join(log_dir, "cs2co_system.log")

    log_file_path = final_config["file_path"]
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 配置structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if final_config["json_format"]:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(name)

    # 配置标准logging以写入文件
    std_logger = logging.getLogger(name)
    std_logger.setLevel(LOG_LEVELS.get(final_config["level"].upper(), logging.INFO))

    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=final_config["max_file_size_mb"] * 1024 * 1024,
        backupCount=final_config["backup_count"],
        encoding='utf-8'
    )

    if final_config["json_format"]:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            final_config["format"],
            datefmt=final_config["date_format"]
        )

    file_handler.setFormatter(file_formatter)
    std_logger.addHandler(file_handler)

    return logger


class LoggerManager:
    """日志管理器"""

    def __init__(self, name: str = "cs2co", config: Optional[Dict[str, Any]] = None):
        """
        初始化日志管理器

        Args:
            name: 日志器名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
        self.logger = None
        self.loguru_logger = None
        self.structlog_logger = None

    def setup(self, log_type: str = "standard", log_dir: Optional[str] = None):
        """
        设置日志系统

        Args:
            log_type: 日志类型 - "standard"、"loguru"、"structlog"
            log_dir: 日志目录
        """
        if log_type == "loguru" and LOGURU_AVAILABLE:
            self.loguru_logger = setup_loguru_logger(self.name, self.config, log_dir)
            self.logger = None
        elif log_type == "structlog" and STRUCTLOG_AVAILABLE:
            self.structlog_logger = setup_structlog_logger(self.name, self.config, log_dir)
            self.logger = None
        else:
            self.logger = setup_logger(self.name, self.config, log_dir)
            self.loguru_logger = None
            self.structlog_logger = None

    def log(self, level: str, message: str, **kwargs):
        """
        记录日志

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外字段
        """
        if self.loguru_logger:
            getattr(self.loguru_logger, level.lower())(message, **kwargs)
        elif self.structlog_logger:
            getattr(self.structlog_logger, level.lower())(message, **kwargs)
        elif self.logger:
            log_method = getattr(self.logger, level.lower())

            # 添加结构化数据
            if kwargs:
                extra = kwargs.copy()
                log_method(message, extra={"structured_data": extra})
            else:
                log_method(message)

    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.log("critical", message, **kwargs)

    def exception(self, message: str, exc_info: bool = True, **kwargs):
        """记录异常日志"""
        if self.loguru_logger:
            self.loguru_logger.exception(message, **kwargs)
        elif self.structlog_logger:
            self.structlog_logger.exception(message, **kwargs)
        elif self.logger:
            self.logger.exception(message, exc_info=exc_info, extra={"structured_data": kwargs})

    def get_logger(self):
        """获取底层日志器"""
        if self.loguru_logger:
            return self.loguru_logger
        elif self.structlog_logger:
            return self.structlog_logger
        else:
            return self.logger


# 全局默认日志器
_default_logger_manager = None


def get_logger(name: str = "cs2co", config: Optional[Dict[str, Any]] = None) -> LoggerManager:
    """
    获取全局日志器

    Args:
        name: 日志器名称
        config: 配置字典

    Returns:
        日志管理器实例
    """
    global _default_logger_manager

    if _default_logger_manager is None:
        _default_logger_manager = LoggerManager(name, config)
        _default_logger_manager.setup()

    return _default_logger_manager


def log_performance(start_time: datetime, operation: str, logger: Optional[LoggerManager] = None):
    """
    记录操作性能

    Args:
        start_time: 开始时间
        operation: 操作名称
        logger: 日志器，如为None则使用默认日志器
    """
    if logger is None:
        logger = get_logger()

    duration = (datetime.now() - start_time).total_seconds()

    if duration > 10:
        level = "warning"
    elif duration > 5:
        level = "info"
    else:
        level = "debug"

    getattr(logger, level)(
        f"Performance: {operation} completed in {duration:.2f} seconds",
        operation=operation,
        duration_seconds=duration,
        start_time=start_time.isoformat(),
        end_time=datetime.now().isoformat()
    )


# 便捷函数
def debug(message: str, **kwargs):
    """记录调试日志（便捷函数）"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """记录信息日志（便捷函数）"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """记录警告日志（便捷函数）"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """记录错误日志（便捷函数）"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """记录严重错误日志（便捷函数）"""
    get_logger().critical(message, **kwargs)


def exception(message: str, exc_info: bool = True, **kwargs):
    """记录异常日志（便捷函数）"""
    get_logger().exception(message, exc_info=exc_info, **kwargs)