"""
系统动态脆弱性标量（SDVS）模型
基于设备拓扑网络、供应链阻力和设备健康状态的动态风险评估模型
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime, timedelta
from scipy.stats import weibull_min
import warnings


class ComponentStatus(Enum):
    """设备组件状态"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class SupplyChainStatus(Enum):
    """供应链状态"""
    AVAILABLE = "available"
    DELAYED = "delayed"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"


@dataclass
class ComponentNode:
    """设备组件节点"""
    id: str
    name: str
    component_type: str
    criticality: float  # 重要性系数 [0, 1]
    downtime_cost: float  # 停机成本（元/小时）
    maintenance_interval: int  # 维护间隔（小时）
    last_maintenance: datetime
    status: ComponentStatus = ComponentStatus.NORMAL
    health_score: float = 1.0  # 健康分数 [0, 1]
    weibull_shape: float = 2.0  # 威布尔形状参数β
    weibull_scale: float = 10000.0  # 威布尔尺度参数η（小时）
    environmental_factor: float = 1.0  # 环境恶劣因子κ

    def __post_init__(self):
        if not 0 <= self.criticality <= 1:
            raise ValueError("criticality must be between 0 and 1")
        if self.downtime_cost < 0:
            raise ValueError("downtime_cost must be non-negative")
        if self.maintenance_interval <= 0:
            raise ValueError("maintenance_interval must be positive")
        if not 0 <= self.health_score <= 1:
            raise ValueError("health_score must be between 0 and 1")
        if self.weibull_shape <= 0:
            raise ValueError("weibull_shape must be positive")
        if self.weibull_scale <= 0:
            raise ValueError("weibull_scale must be positive")
        if self.environmental_factor <= 0:
            raise ValueError("environmental_factor must be positive")


@dataclass
class SparePart:
    """维修备件"""
    id: str
    name: str
    component_id: str  # 对应的设备组件ID
    unit_cost: float  # 单位成本（元）
    holding_cost_rate: float  # 持有成本率（%/年）
    ordering_cost: float  # 订购成本（元/次）
    lead_time_mean: float  # 平均采购提前期（天）
    lead_time_std: float  # 采购提前期标准差（天）
    substitutable_parts: List[str]  # 可替代备件ID列表
    current_inventory: int = 0
    safety_stock: int = 0
    reorder_point: int = 0
    economic_order_quantity: int = 0

    def __post_init__(self):
        if self.unit_cost < 0:
            raise ValueError("unit_cost must be non-negative")
        if self.holding_cost_rate < 0:
            raise ValueError("holding_cost_rate must be non-negative")
        if self.ordering_cost < 0:
            raise ValueError("ordering_cost must be non-negative")
        if self.lead_time_mean <= 0:
            raise ValueError("lead_time_mean must be positive")
        if self.lead_time_std < 0:
            raise ValueError("lead_time_std must be non-negative")
        if self.current_inventory < 0:
            raise ValueError("current_inventory must be non-negative")
        if self.safety_stock < 0:
            raise ValueError("safety_stock must be non-negative")
        if self.reorder_point < 0:
            raise ValueError("reorder_point must be non-negative")
        if self.economic_order_quantity < 0:
            raise ValueError("economic_order_quantity must be non-negative")


class SDVSModel:
    """
    系统动态脆弱性标量模型
    基于论文：面向系统冷启动的维修备件时空联合运营与持续优化研究

    核心方程：
    𝒱_i(t) = 1 - exp[-(𝒞_i · 𝒮_i · ℋ_i(t))^γ]

    其中：
    - 𝒞_i: 拓扑后果标量
    - 𝒮_i: 供应链阻力标量
    - ℋ_i(t): 动态健康因子
    - γ: 形状参数
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, mu: float = 0.5):
        """
        初始化SDVS模型

        Args:
            alpha: 拓扑重要性与经济后果的权重系数 [0, 1]
            gamma: 脆弱性对输入变化的敏感度形状参数 (>0)
            mu: 可替代性衰减系数 (>0)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")

        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.graph = nx.DiGraph()
        self.components: Dict[str, ComponentNode] = {}
        self.spare_parts: Dict[str, SparePart] = {}
        self.component_to_spare: Dict[str, str] = {}  # 组件ID -> 备件ID映射

    def add_component(self, component: ComponentNode) -> None:
        """添加设备组件"""
        self.components[component.id] = component
        self.graph.add_node(component.id,
                           criticality=component.criticality,
                           downtime_cost=component.downtime_cost)

    def add_dependency(self, from_component_id: str, to_component_id: str,
                      weight: float = 1.0) -> None:
        """
        添加组件依赖关系

        Args:
            from_component_id: 依赖方组件ID
            to_component_id: 被依赖方组件ID
            weight: 依赖权重
        """
        if from_component_id not in self.components:
            raise ValueError(f"Component {from_component_id} not found")
        if to_component_id not in self.components:
            raise ValueError(f"Component {to_component_id} not found")

        self.graph.add_edge(from_component_id, to_component_id, weight=weight)

    def add_spare_part(self, spare_part: SparePart) -> None:
        """添加维修备件"""
        if spare_part.component_id not in self.components:
            raise ValueError(f"Component {spare_part.component_id} not found")

        self.spare_parts[spare_part.id] = spare_part
        self.component_to_spare[spare_part.component_id] = spare_part.id

    def calculate_betweenness_centrality(self, normalized: bool = True) -> Dict[str, float]:
        """
        计算介数中心性

        Args:
            normalized: 是否进行归一化

        Returns:
            各节点的介数中心性
        """
        if len(self.graph) == 0:
            return {}

        betweenness = nx.betweenness_centrality(self.graph, weight='weight', normalized=normalized)
        return betweenness

    def calculate_topological_consequence(self, component_id: str,
                                        betweenness: Dict[str, float]) -> float:
        """
        计算拓扑后果标量𝒞_i

        𝒞_i = α · BC_norm(e(i)) + (1-α) · C_e(i)^d / max_j C_e(j)^d

        Args:
            component_id: 组件ID
            betweenness: 介数中心性字典

        Returns:
            拓扑后果标量 [0, 1]
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")

        component = self.components[component_id]

        # 归一化介数中心性
        if component_id in betweenness:
            bc_norm = betweenness[component_id]
        else:
            bc_norm = 0.0

        # 归一化停机经济惩罚
        downtime_costs = [c.downtime_cost for c in self.components.values()]
        max_downtime_cost = max(downtime_costs) if downtime_costs else 1.0

        if max_downtime_cost == 0:
            downtime_norm = 0
        else:
            downtime_norm = component.downtime_cost / max_downtime_cost

        # 凸组合
        topological_consequence = (self.alpha * bc_norm +
                                 (1 - self.alpha) * downtime_norm)

        return max(0.0, min(1.0, topological_consequence))

    def calculate_supply_chain_resistance(self, spare_part_id: str) -> float:
        """
        计算供应链阻力标量𝒮_i

        𝒮_i = L_i^Q / L_max^Q · exp(-μ · N_sub(i))

        Args:
            spare_part_id: 备件ID

        Returns:
            供应链阻力标量 [0, 1]
        """
        if spare_part_id not in self.spare_parts:
            raise ValueError(f"Spare part {spare_part_id} not found")

        spare_part = self.spare_parts[spare_part_id]

        # 计算最大采购提前期
        lead_times = [sp.lead_time_mean for sp in self.spare_parts.values()]
        max_lead_time = max(lead_times) if lead_times else 1.0

        if max_lead_time == 0:
            lead_time_norm = 0
        else:
            lead_time_norm = spare_part.lead_time_mean / max_lead_time

        # 计算可替代性衰减
        n_substitutable = len(spare_part.substitutable_parts)
        substitutability_decay = math.exp(-self.mu * n_substitutable)

        # 供应链阻力标量
        supply_resistance = lead_time_norm * substitutability_decay

        return max(0.0, min(1.0, supply_resistance))

    def calculate_health_factor(self, component_id: str,
                              current_time: datetime) -> float:
        """
        计算动态健康因子ℋ_i(t)

        基于威布尔分布和贝叶斯更新的设备健康状态评估

        Args:
            component_id: 组件ID
            current_time: 当前时间

        Returns:
            动态健康因子 [0, 1]
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")

        component = self.components[component_id]

        # 计算运行时间（小时）
        if component.last_maintenance:
            operation_hours = (current_time - component.last_maintenance).total_seconds() / 3600
        else:
            operation_hours = 0

        # 修正威布尔参数
        shape_param = component.weibull_shape
        scale_param = component.weibull_scale * component.environmental_factor

        if scale_param <= 0:
            warnings.warn(f"Scale parameter is non-positive for component {component_id}")
            scale_param = 1000.0

        # 计算瞬时故障率
        if operation_hours <= 0:
            instantaneous_failure_rate = 0
        else:
            try:
                instantaneous_failure_rate = (shape_param / scale_param) * \
                                           math.pow(operation_hours / scale_param, shape_param - 1)
            except (ValueError, ZeroDivisionError):
                instantaneous_failure_rate = 0

        # 计算累积故障概率（动态健康因子）
        try:
            # 使用威布尔分布CDF
            health_factor = weibull_min.cdf(operation_hours, shape_param, scale=scale_param)
        except:
            # 备用计算方法
            if operation_hours <= 0:
                health_factor = 0
            else:
                health_factor = 1 - math.exp(-math.pow(operation_hours / scale_param, shape_param))

        # 结合实时健康分数
        final_health_factor = health_factor * component.health_score

        return max(0.0, min(1.0, final_health_factor))

    def calculate_sdvs(self, spare_part_id: str, current_time: datetime) -> float:
        """
        计算系统动态脆弱性标量𝒱_i(t)

        𝒱_i(t) = 1 - exp[-(𝒞_i · 𝒮_i · ℋ_i(t))^γ]

        Args:
            spare_part_id: 备件ID
            current_time: 当前时间

        Returns:
            系统动态脆弱性标量 [0, 1]
        """
        if spare_part_id not in self.spare_parts:
            raise ValueError(f"Spare part {spare_part_id} not found")

        spare_part = self.spare_parts[spare_part_id]
        component_id = spare_part.component_id

        # 计算介数中心性
        betweenness = self.calculate_betweenness_centrality(normalized=True)

        # 计算三个维度标量
        topological_consequence = self.calculate_topological_consequence(
            component_id, betweenness)

        supply_resistance = self.calculate_supply_chain_resistance(spare_part_id)

        health_factor = self.calculate_health_factor(component_id, current_time)

        # 计算SDVS
        product = topological_consequence * supply_resistance * health_factor

        if product <= 0:
            vulnerability = 0.0
        else:
            try:
                vulnerability = 1.0 - math.exp(-math.pow(product, self.gamma))
            except (ValueError, OverflowError):
                vulnerability = 1.0  # 极端情况取最大值

        return max(0.0, min(1.0, vulnerability))

    def batch_calculate_sdvs(self, current_time: datetime) -> Dict[str, float]:
        """
        批量计算所有备件的SDVS值

        Args:
            current_time: 当前时间

        Returns:
            备件ID到SDVS值的映射
        """
        results = {}
        betweenness = self.calculate_betweenness_centrality(normalized=True)

        for spare_part_id, spare_part in self.spare_parts.items():
            try:
                component_id = spare_part.component_id

                # 计算三个维度标量
                topological_consequence = self.calculate_topological_consequence(
                    component_id, betweenness)

                supply_resistance = self.calculate_supply_chain_resistance(spare_part_id)

                health_factor = self.calculate_health_factor(component_id, current_time)

                # 计算SDVS
                product = topological_consequence * supply_resistance * health_factor

                if product <= 0:
                    vulnerability = 0.0
                else:
                    try:
                        vulnerability = 1.0 - math.exp(-math.pow(product, self.gamma))
                    except (ValueError, OverflowError):
                        vulnerability = 1.0

                results[spare_part_id] = max(0.0, min(1.0, vulnerability))

            except Exception as e:
                warnings.warn(f"Error calculating SDVS for {spare_part_id}: {e}")
                results[spare_part_id] = 0.0

        return results

    def classify_vulnerability_level(self, sdvs_value: float) -> str:
        """
        根据SDVS值分类脆弱性等级

        Args:
            sdvs_value: SDVS值 [0, 1]

        Returns:
            脆弱性等级描述
        """
        if sdvs_value < 0.1:
            return "极低"
        elif sdvs_value < 0.3:
            return "低"
        elif sdvs_value < 0.5:
            return "中"
        elif sdvs_value < 0.7:
            return "高"
        elif sdvs_value < 0.9:
            return "极高"
        else:
            return "紧急"

    def get_critical_components(self, threshold: float = 0.7,
                              current_time: Optional[datetime] = None) -> List[Tuple[str, float]]:
        """
        获取高脆弱性备件

        Args:
            threshold: 脆弱性阈值
            current_time: 当前时间，如为None则使用当前系统时间

        Returns:
            高脆弱性备件列表（备件ID, SDVS值）
        """
        if current_time is None:
            current_time = datetime.now()

        all_sdvs = self.batch_calculate_sdvs(current_time)
        critical_components = [(sp_id, sdvs) for sp_id, sdvs in all_sdvs.items()
                              if sdvs >= threshold]

        # 按SDVS值降序排序
        critical_components.sort(key=lambda x: x[1], reverse=True)

        return critical_components

    def get_system_vulnerability_summary(self, current_time: Optional[datetime] = None) -> Dict:
        """
        获取系统脆弱性摘要

        Args:
            current_time: 当前时间

        Returns:
            系统脆弱性摘要信息
        """
        if current_time is None:
            current_time = datetime.now()

        all_sdvs = self.batch_calculate_sdvs(current_time)

        if not all_sdvs:
            return {
                "mean_vulnerability": 0.0,
                "max_vulnerability": 0.0,
                "min_vulnerability": 0.0,
                "std_vulnerability": 0.0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "total_count": 0
            }

        sdvs_values = list(all_sdvs.values())

        summary = {
            "mean_vulnerability": float(np.mean(sdvs_values)),
            "max_vulnerability": float(np.max(sdvs_values)),
            "min_vulnerability": float(np.min(sdvs_values)),
            "std_vulnerability": float(np.std(sdvs_values)),
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "total_count": len(sdvs_values)
        }

        # 统计各等级数量
        for sdvs_value in sdvs_values:
            if sdvs_value >= 0.7:
                summary["critical_count"] += 1
            elif sdvs_value >= 0.5:
                summary["high_count"] += 1
            elif sdvs_value >= 0.3:
                summary["medium_count"] += 1
            else:
                summary["low_count"] += 1

        return summary

    def export_model_parameters(self) -> Dict:
        """导出模型参数"""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "mu": self.mu,
            "component_count": len(self.components),
            "spare_part_count": len(self.spare_parts),
            "dependency_count": self.graph.number_of_edges(),
            "graph_density": nx.density(self.graph),
            "graph_diameter": nx.diameter(self.graph) if nx.is_strongly_connected(self.graph) else None
        }


# 示例使用
if __name__ == "__main__":
    # 创建SDVS模型实例
    sdvs_model = SDVSModel(alpha=0.5, gamma=1.0, mu=0.5)

    # 添加设备组件
    components = [
        ComponentNode(
            id="comp_001",
            name="采煤机截割部",
            component_type="采煤机",
            criticality=0.9,
            downtime_cost=25000.0,  # 2.5万元/小时
            maintenance_interval=500,
            last_maintenance=datetime.now() - timedelta(days=30),
            health_score=0.85,
            weibull_shape=2.5,
            weibull_scale=8000.0,
            environmental_factor=1.2
        ),
        ComponentNode(
            id="comp_002",
            name="液压支架阀组",
            component_type="液压支架",
            criticality=0.7,
            downtime_cost=15000.0,
            maintenance_interval=800,
            last_maintenance=datetime.now() - timedelta(days=45),
            health_score=0.92,
            weibull_shape=2.0,
            weibull_scale=10000.0,
            environmental_factor=1.0
        ),
        ComponentNode(
            id="comp_003",
            name="刮板输送机链轮",
            component_type="输送机",
            criticality=0.6,
            downtime_cost=12000.0,
            maintenance_interval=600,
            last_maintenance=datetime.now() - timedelta(days=20),
            health_score=0.78,
            weibull_shape=1.8,
            weibull_scale=6000.0,
            environmental_factor=1.3
        )
    ]

    for comp in components:
        sdvs_model.add_component(comp)

    # 添加依赖关系
    dependencies = [
        ("comp_002", "comp_001", 1.0),  # 液压支架依赖采煤机
        ("comp_003", "comp_001", 0.8),  # 输送机依赖采煤机
        ("comp_003", "comp_002", 0.5)   # 输送机依赖液压支架
    ]

    for from_comp, to_comp, weight in dependencies:
        sdvs_model.add_dependency(from_comp, to_comp, weight)

    # 添加维修备件
    spare_parts = [
        SparePart(
            id="sp_001",
            name="截割部齿轮",
            component_id="comp_001",
            unit_cost=5000.0,
            holding_cost_rate=0.15,
            ordering_cost=500.0,
            lead_time_mean=15.0,
            lead_time_std=3.0,
            substitutable_parts=["sp_002"],
            current_inventory=2,
            safety_stock=1,
            reorder_point=3,
            economic_order_quantity=5
        ),
        SparePart(
            id="sp_002",
            name="液压阀芯",
            component_id="comp_002",
            unit_cost=3000.0,
            holding_cost_rate=0.12,
            ordering_cost=300.0,
            lead_time_mean=10.0,
            lead_time_std=2.0,
            substitutable_parts=[],
            current_inventory=5,
            safety_stock=2,
            reorder_point=4,
            economic_order_quantity=8
        ),
        SparePart(
            id="sp_003",
            name="链轮轴承",
            component_id="comp_003",
            unit_cost=800.0,
            holding_cost_rate=0.10,
            ordering_cost=200.0,
            lead_time_mean=20.0,
            lead_time_std=5.0,
            substitutable_parts=["sp_004"],
            current_inventory=10,
            safety_stock=3,
            reorder_point=6,
            economic_order_quantity=12
        )
    ]

    for sp in spare_parts:
        sdvs_model.add_spare_part(sp)

    # 计算SDVS值
    current_time = datetime.now()

    print("=== SDVS模型演示 ===")
    print(f"模型参数: alpha={sdvs_model.alpha}, gamma={sdvs_model.gamma}, mu={sdvs_model.mu}")
    print(f"组件数量: {len(sdvs_model.components)}")
    print(f"备件数量: {len(sdvs_model.spare_parts)}")
    print(f"依赖关系: {sdvs_model.graph.number_of_edges()}")
    print()

    # 计算单个备件的SDVS
    for spare_part_id in ["sp_001", "sp_002", "sp_003"]:
        sdvs_value = sdvs_model.calculate_sdvs(spare_part_id, current_time)
        level = sdvs_model.classify_vulnerability_level(sdvs_value)
        print(f"备件 {spare_part_id}: SDVS = {sdvs_value:.4f} ({level})")

    print()

    # 批量计算
    all_sdvs = sdvs_model.batch_calculate_sdvs(current_time)
    print("所有备件SDVS值:")
    for spare_part_id, sdvs_value in all_sdvs.items():
        level = sdvs_model.classify_vulnerability_level(sdvs_value)
        print(f"  {spare_part_id}: {sdvs_value:.4f} ({level})")

    print()

    # 获取高脆弱性备件
    critical_components = sdvs_model.get_critical_components(threshold=0.5, current_time=current_time)
    print(f"高脆弱性备件 (阈值≥0.5): {len(critical_components)}个")
    for spare_part_id, sdvs_value in critical_components:
        level = sdvs_model.classify_vulnerability_level(sdvs_value)
        print(f"  {spare_part_id}: {sdvs_value:.4f} ({level})")

    print()

    # 系统摘要
    summary = sdvs_model.get_system_vulnerability_summary(current_time)
    print("系统脆弱性摘要:")
    print(f"  平均脆弱性: {summary['mean_vulnerability']:.4f}")
    print(f"  最大脆弱性: {summary['max_vulnerability']:.4f}")
    print(f"  最小脆弱性: {summary['min_vulnerability']:.4f}")
    print(f"  标准差: {summary['std_vulnerability']:.4f}")
    print(f"  紧急/高/中/低: {summary['critical_count']}/{summary['high_count']}/"
          f"{summary['medium_count']}/{summary['low_count']}")

    print()

    # 模型参数
    params = sdvs_model.export_model_parameters()
    print("模型参数摘要:")
    for key, value in params.items():
        print(f"  {key}: {value}")