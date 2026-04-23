"""
业务服务模块
提供系统核心业务逻辑和数据处理服务

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__all__ = [
    "SDVSService",
    "OptimizationService",
    "InventoryService",
    "MaintenanceService",
    "ForecastingService"
]

# 业务服务描述
__description__ = """
业务服务模块实现了CS2CO系统的核心业务逻辑，包括：
1. SDVS服务：动态脆弱性评估与计算
2. 优化服务：库存策略优化和调度
3. 库存服务：备件库存管理和监控
4. 维护服务：设备维护计划和执行
5. 预测服务：故障预测和需求预测
"""

# 版本信息
__version__ = "1.0.0"

# 导入服务类
try:
    from .sdvs_service import SDVSService
    from .optimization_service import OptimizationService
    from .inventory_service import InventoryService
    from .maintenance_service import MaintenanceService
    from .forecasting_service import ForecastingService
except ImportError as e:
    print(f"Warning: Some service modules may not be available: {e}")

    # 提供占位类
    class SDVSService:
        """SDVS服务占位类"""
        pass

    class OptimizationService:
        """优化服务占位类"""
        pass

    class InventoryService:
        """库存服务占位类"""
        pass

    class MaintenanceService:
        """维护服务占位类"""
        pass

    class ForecastingService:
        """预测服务占位类"""
        pass

def get_services_info() -> dict:
    """获取业务服务信息"""
    return {
        "module": "services",
        "version": __version__,
        "description": __description__.strip(),
        "services": [
            "SDVSService（动态脆弱性评估）",
            "OptimizationService（库存优化调度）",
            "InventoryService（备件库存管理）",
            "MaintenanceService（设备维护管理）",
            "ForecastingService（故障需求预测）"
        ]
    }