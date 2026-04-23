"""
数据模型模块
定义系统核心数据结构和数据库模型

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__all__ = [
    "ComponentNode",
    "SparePart",
    "SystemState",
    "MaintenanceRecord",
    "InventoryRecord",
    "ComponentStatus",
    "SupplyChainStatus"
]

# 数据模型描述
__description__ = """
数据模型模块定义了CS2CO系统的核心数据结构，包括：
1. 设备组件模型：设备拓扑网络和健康状态
2. 备件模型：库存、采购、成本信息
3. 系统状态模型：实时运行状态和历史记录
4. 维护记录模型：设备维护和故障历史
5. 供应链模型：供应商、采购提前期、库存策略
"""

# 版本信息
__version__ = "1.0.0"

# 导入枚举类型
try:
    from .database_models import (
        ComponentNode,
        SparePart,
        SystemState,
        MaintenanceRecord,
        InventoryRecord
    )
    from ..core.sdvs.sdvs_model import ComponentStatus, SupplyChainStatus
except ImportError as e:
    print(f"Warning: Database models may not be available: {e}")

    # 定义占位类
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List
    from enum import Enum

    class ComponentStatus(Enum):
        """设备组件状态枚举"""
        NORMAL = "normal"
        WARNING = "warning"
        CRITICAL = "critical"
        FAILED = "failed"

    class SupplyChainStatus(Enum):
        """供应链状态枚举"""
        AVAILABLE = "available"
        DELAYED = "delayed"
        OUT_OF_STOCK = "out_of_stock"
        DISCONTINUED = "discontinued"

    @dataclass
    class ComponentNode:
        """设备组件节点占位类"""
        id: str = ""
        name: str = ""
        component_type: str = ""
        criticality: float = 0.0
        downtime_cost: float = 0.0

    @dataclass
    class SparePart:
        """维修备件占位类"""
        id: str = ""
        name: str = ""
        component_id: str = ""
        unit_cost: float = 0.0

    @dataclass
    class SystemState:
        """系统状态占位类"""
        timestamp: datetime = datetime.now()
        overall_vulnerability: float = 0.0

    @dataclass
    class MaintenanceRecord:
        """维护记录占位类"""
        component_id: str = ""
        maintenance_type: str = ""
        cost: float = 0.0

    @dataclass
    class InventoryRecord:
        """库存记录占位类"""
        spare_part_id: str = ""
        quantity: int = 0
        unit_price: float = 0.0

def get_models_info() -> dict:
    """获取数据模型信息"""
    return {
        "module": "models",
        "version": __version__,
        "description": __description__.strip(),
        "models": [
            "ComponentNode（设备组件）",
            "SparePart（维修备件）",
            "SystemState（系统状态）",
            "MaintenanceRecord（维护记录）",
            "InventoryRecord（库存记录）",
            "ComponentStatus（组件状态枚举）",
            "SupplyChainStatus（供应链状态枚举）"
        ]
    }