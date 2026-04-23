"""
CS2CO系统 - 泛工业高可用备件时空联合运营系统
核心模块初始化文件

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "CS2CO Team"
__email__ = "support@cs2co.example.com"

# 导出核心模块
from .core import *
from .models import *
from .services import *
from .api import *
from .utils import *

# 项目描述
__description__ = """
CS2CO（Universal Cold-Start to Continuous Optimization System）是一个面向
工业设备维修备件管理的智能决策支持系统。系统基于系统动态脆弱性（SDVS）模型、
MacDec-POMDP多智能体强化学习框架和安全演化引擎，解决备件管理中的冷启动难题。
"""

# 主要功能模块
__modules__ = {
    "core": "核心算法模块",
    "models": "数据模型模块",
    "services": "业务服务模块",
    "api": "API接口模块",
    "utils": "工具函数模块"
}

def get_version() -> str:
    """获取系统版本"""
    return __version__

def get_system_info() -> dict:
    """获取系统信息"""
    return {
        "name": "CS2CO System",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__.strip(),
        "modules": __modules__
    }