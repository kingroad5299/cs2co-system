"""
核心算法模块
包含SDVS模型、强化学习框架、安全演化引擎等核心算法

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__all__ = [
    "SDVSModel",
    "MacDecPOMDPEnvironment",
    "SafetyEvolutionEngine",
    "IndustryAdapter",
    "ColdStartOptimizer"
]

# 核心功能描述
__description__ = """
核心算法模块提供CS2CO系统的智能决策能力，包括：
1. SDVS模型：系统动态脆弱性评估
2. MacDec-POMDP：宏动作去中心化部分可观测马尔可夫博弈
3. 安全强化学习：CMDP约束优化与安全护盾
4. 迁移学习：零数据冷启动支持
5. 在线自适应：概念漂移检测与适应
"""

# 版本信息
__version__ = "1.0.0"

# 导入核心算法类
try:
    from .sdvs.sdvs_model import SDVSModel
    from .rl.macdec_pomdp_env import MacDecPOMDPEnvironment
    from .rl.safety_evolution_engine import SafetyEvolutionEngine
    from .adapters.industry_adapters import IndustryAdapter
    from .optimizers.cold_start_optimizer import ColdStartOptimizer
except ImportError as e:
    print(f"Warning: Some core modules may not be available: {e}")
    # 提供占位类
    class SDVSModel:
        """系统动态脆弱性模型占位类"""
        pass

    class MacDecPOMDPEnvironment:
        """MacDec-POMDP环境占位类"""
        pass

    class SafetyEvolutionEngine:
        """安全演化引擎占位类"""
        pass

    class IndustryAdapter:
        """行业适配器占位类"""
        pass

    class ColdStartOptimizer:
        """冷启动优化器占位类"""
        pass

def get_core_info() -> dict:
    """获取核心模块信息"""
    return {
        "module": "core",
        "version": __version__,
        "description": __description__.strip(),
        "features": [
            "SDVS模型（系统动态脆弱性评估）",
            "MacDec-POMDP（多智能体强化学习）",
            "安全演化引擎（约束优化）",
            "行业适配器（多行业支持）",
            "冷启动优化器（零数据学习）"
        ]
    }