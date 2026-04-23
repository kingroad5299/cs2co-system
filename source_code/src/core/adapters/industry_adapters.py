"""
行业物理适配器模块
实现通用理论模型到具体行业的参数映射
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import json
import yaml
from datetime import datetime, timedelta
import warnings


class IndustryType(Enum):
    """行业类型"""
    COAL_MINING = "coal_mining"          # 煤炭开采
    CONSTRUCTION_MACHINERY = "construction_machinery"  # 工程机械
    AUTOMOTIVE_AFTERSALES = "automotive_aftersales"    # 汽车售后
    HEAVY_INDUSTRY = "heavy_industry"    # 重工业
    POWER_GENERATION = "power_generation"  # 电力发电


class EquipmentCategory(Enum):
    """设备类别"""
    MINING = "mining"                    # 采矿设备
    EXCAVATION = "excavation"            # 挖掘设备
    HAULING = "hauling"                  # 运输设备
    PROCESSING = "processing"            # 加工设备
    AUXILIARY = "auxiliary"              # 辅助设备


@dataclass
class IndustryParameterMapping:
    """行业参数映射配置"""
    industry_type: IndustryType
    equipment_categories: List[EquipmentCategory]

    # 拓扑参数映射
    topology_alpha: float = 0.5  # 拓扑权重系数
    network_type: str = "directed"  # 网络类型

    # 经济参数映射
    currency_unit: str = "CNY"  # 货币单位
    cost_multiplier: float = 1.0  # 成本乘数
    downtime_cost_base: float = 10000.0  # 停机成本基数

    # 时间参数映射
    time_unit: str = "hours"  # 时间单位
    planning_horizon: int = 365  # 规划周期（天）

    # 可靠性参数映射
    weibull_shape_range: Tuple[float, float] = (1.5, 3.0)  # 威布尔形状参数范围
    weibull_scale_range: Tuple[float, float] = (2000.0, 10000.0)  # 威布尔尺度参数范围
    environmental_factor_range: Tuple[float, float] = (0.8, 1.5)  # 环境因子范围

    # 供应链参数映射
    lead_time_multiplier: float = 1.0  # 采购提前期乘数
    substitutability_factor: float = 0.8  # 可替代性因子

    # 空间参数映射
    distance_unit: str = "kilometers"  # 距离单位
    average_speed: float = 15.0  # 平均速度（km/h）

    # 性能参数映射
    availability_target: float = 0.95  # 可用性目标
    response_time_target: float = 24.0  # 响应时间目标（小时）

    # 自定义参数
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquipmentTemplate:
    """设备模板"""
    equipment_id: str
    name: str
    category: EquipmentCategory
    critical_components: List[str]  # 关键组件列表
    typical_downtime_cost: float  # 典型停机成本（元/小时）
    maintenance_interval: int  # 维护间隔（小时）
    reliability_parameters: Dict[str, float]  # 可靠性参数

    def __post_init__(self):
        if self.typical_downtime_cost < 0:
            raise ValueError("typical_downtime_cost must be non-negative")
        if self.maintenance_interval <= 0:
            raise ValueError("maintenance_interval must be positive")


@dataclass
class SparePartTemplate:
    """备件模板"""
    part_id: str
    name: str
    equipment_id: str
    unit_cost_range: Tuple[float, float]  # 单位成本范围
    lead_time_range: Tuple[float, float]  # 采购提前期范围（天）
    substitutability: float  # 可替代性 [0, 1]
    criticality_level: int  # 关键性等级 1-5

    def __post_init__(self):
        if not 0 <= self.substitutability <= 1:
            raise ValueError("substitutability must be between 0 and 1")
        if not 1 <= self.criticality_level <= 5:
            raise ValueError("criticality_level must be between 1 and 5")


class IndustryAdapter:
    """
    行业物理适配器基类
    实现通用参数到行业具体参数的映射
    """

    def __init__(self, mapping: IndustryParameterMapping):
        self.mapping = mapping
        self.equipment_templates: Dict[str, EquipmentTemplate] = {}
        self.spare_part_templates: Dict[str, SparePartTemplate] = {}

    def add_equipment_template(self, template: EquipmentTemplate):
        """添加设备模板"""
        self.equipment_templates[template.equipment_id] = template

    def add_spare_part_template(self, template: SparePartTemplate):
        """添加备件模板"""
        self.spare_part_templates[template.part_id] = template

    def map_general_to_specific(self,
                               general_parameter: str,
                               context: Optional[Dict] = None) -> Any:
        """
        通用参数映射到具体行业参数

        Args:
            general_parameter: 通用参数名称
            context: 上下文信息

        Returns:
            行业具体参数值
        """
        mapping_rules = self._get_mapping_rules()

        if general_parameter in mapping_rules:
            rule = mapping_rules[general_parameter]
            return rule(general_parameter, context)
        else:
            warnings.warn(f"No mapping rule for parameter: {general_parameter}")
            return None

    def _get_mapping_rules(self) -> Dict[str, callable]:
        """获取参数映射规则"""
        return {
            # 拓扑参数映射
            "topology_weight": self._map_topology_weight,
            "network_density": self._map_network_density,

            # 经济参数映射
            "cost_parameter": self._map_cost_parameter,
            "downtime_cost": self._map_downtime_cost,

            # 时间参数映射
            "time_parameter": self._map_time_parameter,
            "planning_period": self._map_planning_period,

            # 可靠性参数映射
            "failure_rate": self._map_failure_rate,
            "maintenance_interval": self._map_maintenance_interval,

            # 供应链参数映射
            "lead_time": self._map_lead_time,
            "substitutability": self._map_substitutability,

            # 空间参数映射
            "distance": self._map_distance,
            "transport_speed": self._map_transport_speed,

            # 性能参数映射
            "availability_target": self._map_availability_target,
            "response_time": self._map_response_time
        }

    def _map_topology_weight(self, param: str, context: Dict) -> float:
        """映射拓扑权重"""
        base_weight = 0.5
        industry_factor = self._get_industry_factor("topology_complexity")
        return base_weight * industry_factor

    def _map_network_density(self, param: str, context: Dict) -> float:
        """映射网络密度"""
        # 不同行业的设备依赖关系复杂度不同
        industry_density_map = {
            IndustryType.COAL_MINING: 0.7,
            IndustryType.CONSTRUCTION_MACHINERY: 0.6,
            IndustryType.AUTOMOTIVE_AFTERSALES: 0.5,
            IndustryType.HEAVY_INDUSTRY: 0.8,
            IndustryType.POWER_GENERATION: 0.75
        }
        return industry_density_map.get(self.mapping.industry_type, 0.6)

    def _map_cost_parameter(self, param: str, context: Dict) -> float:
        """映射成本参数"""
        base_cost = context.get("base_cost", 1.0) if context else 1.0
        return base_cost * self.mapping.cost_multiplier

    def _map_downtime_cost(self, param: str, context: Dict) -> float:
        """映射停机成本"""
        equipment_type = context.get("equipment_type") if context else None
        base_cost = self.mapping.downtime_cost_base

        if equipment_type and equipment_type in self.equipment_templates:
            template = self.equipment_templates[equipment_type]
            return template.typical_downtime_cost
        else:
            # 根据设备类别调整
            equipment_category = context.get("equipment_category") if context else None
            category_multiplier = self._get_category_multiplier(equipment_category)
            return base_cost * category_multiplier

    def _map_time_parameter(self, param: str, context: Dict) -> float:
        """映射时间参数"""
        # 转换时间单位
        time_value = context.get("time_value", 1.0) if context else 1.0
        time_unit = context.get("time_unit", "hours") if context else "hours"

        if time_unit == "days" and self.mapping.time_unit == "hours":
            return time_value * 24
        elif time_unit == "hours" and self.mapping.time_unit == "days":
            return time_value / 24
        else:
            return time_value

    def _map_planning_period(self, param: str, context: Dict) -> int:
        """映射规划周期"""
        return self.mapping.planning_horizon

    def _map_failure_rate(self, param: str, context: Dict) -> float:
        """映射故障率"""
        base_rate = context.get("base_rate", 0.001) if context else 0.001
        environmental_factor = context.get("environmental_factor", 1.0) if context else 1.0

        # 根据行业调整
        industry_factor = self._get_industry_factor("operational_harshness")
        return base_rate * environmental_factor * industry_factor

    def _map_maintenance_interval(self, param: str, context: Dict) -> int:
        """映射维护间隔"""
        equipment_type = context.get("equipment_type") if context else None

        if equipment_type and equipment_type in self.equipment_templates:
            template = self.equipment_templates[equipment_type]
            return template.maintenance_interval
        else:
            # 默认维护间隔
            return 500  # 小时

    def _map_lead_time(self, param: str, context: Dict) -> float:
        """映射采购提前期"""
        part_type = context.get("part_type") if context else None

        if part_type and part_type in self.spare_part_templates:
            template = self.spare_part_templates[part_type]
            lead_time_min, lead_time_max = template.lead_time_range
            # 返回范围中值
            return (lead_time_min + lead_time_max) / 2
        else:
            base_lead_time = 10.0  # 天
            return base_lead_time * self.mapping.lead_time_multiplier

    def _map_substitutability(self, param: str, context: Dict) -> float:
        """映射可替代性"""
        part_type = context.get("part_type") if context else None

        if part_type and part_type in self.spare_part_templates:
            template = self.spare_part_templates[part_type]
            return template.substitutability
        else:
            return self.mapping.substitutability_factor

    def _map_distance(self, param: str, context: Dict) -> float:
        """映射距离"""
        distance_value = context.get("distance_value", 1.0) if context else 1.0
        distance_unit = context.get("distance_unit", "km") if context else "km"

        # 单位转换（简化）
        if distance_unit != self.mapping.distance_unit:
            warnings.warn(f"Distance unit conversion not implemented: {distance_unit} to {self.mapping.distance_unit}")

        return distance_value

    def _map_transport_speed(self, param: str, context: Dict) -> float:
        """映射运输速度"""
        return self.mapping.average_speed

    def _map_availability_target(self, param: str, context: Dict) -> float:
        """映射可用性目标"""
        return self.mapping.availability_target

    def _map_response_time(self, param: str, context: Dict) -> float:
        """映射响应时间"""
        return self.mapping.response_time_target

    def _get_industry_factor(self, factor_type: str) -> float:
        """获取行业因子"""
        factor_map = {
            "topology_complexity": {
                IndustryType.COAL_MINING: 1.2,
                IndustryType.CONSTRUCTION_MACHINERY: 1.0,
                IndustryType.AUTOMOTIVE_AFTERSALES: 0.8,
                IndustryType.HEAVY_INDUSTRY: 1.3,
                IndustryType.POWER_GENERATION: 1.1
            },
            "operational_harshness": {
                IndustryType.COAL_MINING: 1.5,
                IndustryType.CONSTRUCTION_MACHINERY: 1.2,
                IndustryType.AUTOMOTIVE_AFTERSALES: 1.0,
                IndustryType.HEAVY_INDUSTRY: 1.4,
                IndustryType.POWER_GENERATION: 1.3
            },
            "cost_sensitivity": {
                IndustryType.COAL_MINING: 1.1,
                IndustryType.CONSTRUCTION_MACHINERY: 1.0,
                IndustryType.AUTOMOTIVE_AFTERSALES: 0.9,
                IndustryType.HEAVY_INDUSTRY: 1.2,
                IndustryType.POWER_GENERATION: 1.1
            }
        }

        factors = factor_map.get(factor_type, {})
        return factors.get(self.mapping.industry_type, 1.0)

    def _get_category_multiplier(self, category: Optional[EquipmentCategory]) -> float:
        """获取设备类别乘数"""
        if category is None:
            return 1.0

        category_multiplier_map = {
            EquipmentCategory.MINING: 2.5,      # 采矿设备：高停机成本
            EquipmentCategory.EXCAVATION: 2.0,   # 挖掘设备
            EquipmentCategory.HAULING: 1.5,      # 运输设备
            EquipmentCategory.PROCESSING: 1.8,   # 加工设备
            EquipmentCategory.AUXILIARY: 1.0     # 辅助设备
        }

        return category_multiplier_map.get(category, 1.0)

    def generate_sdvs_parameters(self,
                                equipment_id: str,
                                spare_part_id: str) -> Dict[str, Any]:
        """
        生成SDVS模型参数

        Args:
            equipment_id: 设备ID
            spare_part_id: 备件ID

        Returns:
            SDVS参数字典
        """
        parameters = {}

        # 获取设备模板
        equipment_template = self.equipment_templates.get(equipment_id)
        spare_part_template = self.spare_part_templates.get(spare_part_id)

        if equipment_template:
            # 拓扑重要性
            parameters["criticality"] = self._calculate_criticality(equipment_template)

            # 停机经济惩罚
            parameters["downtime_cost"] = equipment_template.typical_downtime_cost

            # 维护间隔
            parameters["maintenance_interval"] = equipment_template.maintenance_interval

            # 可靠性参数
            reliability_params = equipment_template.reliability_parameters
            parameters["weibull_shape"] = reliability_params.get("shape", 2.0)
            parameters["weibull_scale"] = reliability_params.get("scale", 5000.0)
            parameters["environmental_factor"] = reliability_params.get("environmental_factor", 1.0)

        if spare_part_template:
            # 采购提前期
            lead_time_min, lead_time_max = spare_part_template.lead_time_range
            parameters["lead_time_mean"] = (lead_time_min + lead_time_max) / 2
            parameters["lead_time_std"] = (lead_time_max - lead_time_min) / 4

            # 可替代性
            parameters["substitutability"] = spare_part_template.substitutability

            # 单位成本
            cost_min, cost_max = spare_part_template.unit_cost_range
            parameters["unit_cost"] = (cost_min + cost_max) / 2

            # 关键性等级
            parameters["criticality_level"] = spare_part_template.criticality_level

        # 行业通用参数
        parameters.update({
            "industry_type": self.mapping.industry_type.value,
            "topology_alpha": self.mapping.topology_alpha,
            "currency_unit": self.mapping.currency_unit,
            "time_unit": self.mapping.time_unit,
            "distance_unit": self.mapping.distance_unit
        })

        return parameters

    def _calculate_criticality(self, template: EquipmentTemplate) -> float:
        """计算设备关键性"""
        base_criticality = 0.5

        # 根据设备类别调整
        category_weights = {
            EquipmentCategory.MINING: 0.9,
            EquipmentCategory.EXCAVATION: 0.8,
            EquipmentCategory.HAULING: 0.7,
            EquipmentCategory.PROCESSING: 0.75,
            EquipmentCategory.AUXILIARY: 0.5
        }

        category_weight = category_weights.get(template.category, 0.6)

        # 根据关键组件数量调整
        component_factor = min(len(template.critical_components) / 10, 1.0)

        # 根据停机成本调整
        cost_factor = min(template.typical_downtime_cost / 50000.0, 1.0)

        criticality = base_criticality * 0.3 + category_weight * 0.3 + \
                     component_factor * 0.2 + cost_factor * 0.2

        return min(max(criticality, 0.0), 1.0)

    def export_configuration(self, filepath: str, format: str = "json"):
        """导出适配器配置"""
        config = {
            "industry_type": self.mapping.industry_type.value,
            "parameter_mapping": self.mapping.__dict__,
            "equipment_templates": {
                tid: template.__dict__
                for tid, template in self.equipment_templates.items()
            },
            "spare_part_templates": {
                pid: template.__dict__
                for pid, template in self.spare_part_templates.items()
            }
        }

        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"适配器配置已导出到: {filepath}")

    def load_configuration(self, filepath: str, format: str = "json"):
        """加载适配器配置"""
        if format.lower() == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif format.lower() == "yaml":
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # 恢复行业类型
        industry_type = IndustryType(config["industry_type"])

        # 恢复参数映射
        mapping_data = config["parameter_mapping"]
        self.mapping = IndustryParameterMapping(
            industry_type=industry_type,
            equipment_categories=[EquipmentCategory(ec) for ec in mapping_data.get("equipment_categories", [])],
            topology_alpha=mapping_data.get("topology_alpha", 0.5),
            network_type=mapping_data.get("network_type", "directed"),
            currency_unit=mapping_data.get("currency_unit", "CNY"),
            cost_multiplier=mapping_data.get("cost_multiplier", 1.0),
            downtime_cost_base=mapping_data.get("downtime_cost_base", 10000.0),
            time_unit=mapping_data.get("time_unit", "hours"),
            planning_horizon=mapping_data.get("planning_horizon", 365),
            weibull_shape_range=tuple(mapping_data.get("weibull_shape_range", (1.5, 3.0))),
            weibull_scale_range=tuple(mapping_data.get("weibull_scale_range", (2000.0, 10000.0))),
            environmental_factor_range=tuple(mapping_data.get("environmental_factor_range", (0.8, 1.5))),
            lead_time_multiplier=mapping_data.get("lead_time_multiplier", 1.0),
            substitutability_factor=mapping_data.get("substitutability_factor", 0.8),
            distance_unit=mapping_data.get("distance_unit", "kilometers"),
            average_speed=mapping_data.get("average_speed", 15.0),
            availability_target=mapping_data.get("availability_target", 0.95),
            response_time_target=mapping_data.get("response_time_target", 24.0),
            custom_parameters=mapping_data.get("custom_parameters", {})
        )

        # 恢复设备模板
        self.equipment_templates.clear()
        for tid, template_data in config.get("equipment_templates", {}).items():
            template = EquipmentTemplate(
                equipment_id=template_data["equipment_id"],
                name=template_data["name"],
                category=EquipmentCategory(template_data["category"]),
                critical_components=template_data["critical_components"],
                typical_downtime_cost=template_data["typical_downtime_cost"],
                maintenance_interval=template_data["maintenance_interval"],
                reliability_parameters=template_data["reliability_parameters"]
            )
            self.equipment_templates[tid] = template

        # 恢复备件模板
        self.spare_part_templates.clear()
        for pid, template_data in config.get("spare_part_templates", {}).items():
            template = SparePartTemplate(
                part_id=template_data["part_id"],
                name=template_data["name"],
                equipment_id=template_data["equipment_id"],
                unit_cost_range=tuple(template_data["unit_cost_range"]),
                lead_time_range=tuple(template_data["lead_time_range"]),
                substitutability=template_data["substitutability"],
                criticality_level=template_data["criticality_level"]
            )
            self.spare_part_templates[pid] = template

        print(f"适配器配置已从 {filepath} 加载")


class CoalMiningAdapter(IndustryAdapter):
    """煤炭行业适配器"""

    def __init__(self):
        mapping = IndustryParameterMapping(
            industry_type=IndustryType.COAL_MINING,
            equipment_categories=[
                EquipmentCategory.MINING,
                EquipmentCategory.HAULING,
                EquipmentCategory.AUXILIARY
            ],
            topology_alpha=0.6,  # 煤炭行业更注重拓扑重要性
            downtime_cost_base=25000.0,  # 煤炭行业停机成本高
            weibull_shape_range=(1.8, 3.2),  # 磨损更严重
            weibull_scale_range=(1500.0, 6000.0),  # 寿命较短
            environmental_factor_range=(1.2, 2.0),  # 环境更恶劣
            lead_time_multiplier=1.2,  # 采购更困难
            average_speed=15.0,  # 巷道运输速度
            availability_target=0.97,  # 高可用性要求
            response_time_target=12.0,  # 快速响应要求
            custom_parameters={
                "mining_depth_factor": 1.2,
                "geological_complexity": "high",
                "safety_requirement": "extremely_high"
            }
        )

        super().__init__(mapping)

        # 初始化煤炭行业特定模板
        self._initialize_coal_mining_templates()

    def _initialize_coal_mining_templates(self):
        """初始化煤炭行业设备模板"""

        # 采煤机
        self.add_equipment_template(EquipmentTemplate(
            equipment_id="shearer",
            name="采煤机",
            category=EquipmentCategory.MINING,
            critical_components=[
                "截割部", "牵引部", "电气系统", "液压系统"
            ],
            typical_downtime_cost=30000.0,  # 3万元/小时
            maintenance_interval=480,  # 480小时
            reliability_parameters={
                "shape": 2.5,
                "scale": 4000.0,
                "environmental_factor": 1.5,
                "wear_rate": 0.8
            }
        ))

        # 液压支架
        self.add_equipment_template(EquipmentTemplate(
            equipment_id="hydraulic_support",
            name="液压支架",
            category=EquipmentCategory.MINING,
            critical_components=[
                "立柱", "阀组", "控制系统"
            ],
            typical_downtime_cost=15000.0,  # 1.5万元/小时
            maintenance_interval=720,
            reliability_parameters={
                "shape": 2.0,
                "scale": 6000.0,
                "environmental_factor": 1.3,
                "corrosion_rate": 0.6
            }
        ))

        # 刮板输送机
        self.add_equipment_template(EquipmentTemplate(
            equipment_id="scraper_conveyor",
            name="刮板输送机",
            category=EquipmentCategory.HAULING,
            critical_components=[
                "链轮", "链条", "减速器", "电动机"
            ],
            typical_downtime_cost=12000.0,  # 1.2万元/小时
            maintenance_interval=600,
            reliability_parameters={
                "shape": 1.8,
                "scale": 5000.0,
                "environmental_factor": 1.4,
                "abrasion_rate": 0.9
            }
        ))

        # 初始化煤炭行业备件模板
        self._initialize_coal_mining_spare_parts()

    def _initialize_coal_mining_spare_parts(self):
        """初始化煤炭行业备件模板"""

        # 采煤机备件
        self.add_spare_part_template(SparePartTemplate(
            part_id="shearer_cutting_drum",
            name="截割滚筒",
            equipment_id="shearer",
            unit_cost_range=(80000.0, 120000.0),
            lead_time_range=(30.0, 60.0),  # 30-60天
            substitutability=0.3,
            criticality_level=5
        ))

        self.add_spare_part_template(SparePartTemplate(
            part_id="shearer_gear",
            name="截割部齿轮",
            equipment_id="shearer",
            unit_cost_range=(5000.0, 15000.0),
            lead_time_range=(15.0, 30.0),
            substitutability=0.5,
            criticality_level=4
        ))

        # 液压支架备件
        self.add_spare_part_template(SparePartTemplate(
            part_id="support_valve",
            name="液压阀组",
            equipment_id="hydraulic_support",
            unit_cost_range=(3000.0, 8000.0),
            lead_time_range=(10.0, 20.0),
            substitutability=0.4,
            criticality_level=4
        ))

        # 刮板输送机备件
        self.add_spare_part_template(SparePartTemplate(
            part_id="conveyor_chain",
            name="输送链条",
            equipment_id="scraper_conveyor",
            unit_cost_range=(2000.0, 5000.0),
            lead_time_range=(7.0, 14.0),
            substitutability=0.7,
            criticality_level=3
        ))

    def get_mining_specific_parameters(self, mine_depth: float,
                                      geological_condition: str) -> Dict[str, Any]:
        """
        获取采矿特定参数

        Args:
            mine_depth: 矿井深度（米）
            geological_condition: 地质条件

        Returns:
            采矿特定参数
        """
        # 深度因子
        depth_factor = 1.0 + (mine_depth / 1000.0) * 0.2

        # 地质条件因子
        geology_factors = {
            "simple": 1.0,
            "medium": 1.2,
            "complex": 1.5,
            "extremely_complex": 2.0
        }
        geology_factor = geology_factors.get(geological_condition.lower(), 1.2)

        # 环境恶劣因子
        environmental_harshness = depth_factor * geology_factor

        parameters = {
            "environmental_harshness": environmental_harshness,
            "mine_depth": mine_depth,
            "geological_condition": geological_condition,
            "depth_factor": depth_factor,
            "geology_factor": geology_factor,
            "recommended_safety_factor": 1.5 * environmental_harshness,
            "inspection_frequency": max(168, 168 / environmental_harshness)  # 小时
        }

        return parameters


class ConstructionMachineryAdapter(IndustryAdapter):
    """工程机械行业适配器"""

    def __init__(self):
        mapping = IndustryParameterMapping(
            industry_type=IndustryType.CONSTRUCTION_MACHINERY,
            equipment_categories=[
                EquipmentCategory.EXCAVATION,
                EquipmentCategory.HAULING,
                EquipmentCategory.PROCESSING
            ],
            topology_alpha=0.4,  # 工程机械更注重经济后果
            downtime_cost_base=8000.0,
            weibull_shape_range=(1.6, 2.8),
            weibull_scale_range=(3000.0, 8000.0),
            environmental_factor_range=(1.0, 1.8),
            lead_time_multiplier=1.0,
            average_speed=25.0,  # 公路运输速度
            availability_target=0.92,
            response_time_target=48.0,
            custom_parameters={
                "project_based": True,
                "mobility_requirement": "high",
                "operating_environment": "variable"
            }
        )

        super().__init__(mapping)
        self._initialize_construction_machinery_templates()

    def _initialize_construction_machinery_templates(self):
        """初始化工程机械设备模板"""
        # 实现类似煤炭行业的模板初始化
        pass


class AutomotiveAftersalesAdapter(IndustryAdapter):
    """汽车售后行业适配器"""

    def __init__(self):
        mapping = IndustryParameterMapping(
            industry_type=IndustryType.AUTOMOTIVE_AFTERSALES,
            equipment_categories=[
                EquipmentCategory.PROCESSING,
                EquipmentCategory.AUXILIARY
            ],
            topology_alpha=0.3,
            downtime_cost_base=2000.0,
            weibull_shape_range=(1.4, 2.5),
            weibull_scale_range=(5000.0, 12000.0),
            environmental_factor_range=(0.9, 1.3),
            lead_time_multiplier=0.8,
            average_speed=30.0,
            availability_target=0.90,
            response_time_target=72.0,
            custom_parameters={
                "customer_facing": True,
                "service_level_agreement": "standard",
                "part_variety": "high"
            }
        )

        super().__init__(mapping)
        self._initialize_automotive_templates()


# 示例使用
if __name__ == "__main__":
    print("=== 行业物理适配器演示 ===")

    # 创建煤炭行业适配器
    coal_adapter = CoalMiningAdapter()

    print(f"行业类型: {coal_adapter.mapping.industry_type.value}")
    print(f"设备类别: {[cat.value for cat in coal_adapter.mapping.equipment_categories]}")
    print(f"设备模板数量: {len(coal_adapter.equipment_templates)}")
    print(f"备件模板数量: {len(coal_adapter.spare_part_templates)}")
    print()

    # 演示参数映射
    print("参数映射演示:")

    # 映射停机成本
    context = {
        "equipment_type": "shearer",
        "equipment_category": EquipmentCategory.MINING
    }
    downtime_cost = coal_adapter.map_general_to_specific("downtime_cost", context)
    print(f"采煤机停机成本: {downtime_cost:.2f} 元/小时")

    # 映射采购提前期
    context = {"part_type": "shearer_cutting_drum"}
    lead_time = coal_adapter.map_general_to_specific("lead_time", context)
    print(f"截割滚筒采购提前期: {lead_time:.1f} 天")

    # 映射可替代性
    substitutability = coal_adapter.map_general_to_specific("substitutability", context)
    print(f"截割滚筒可替代性: {substitutability:.2f}")
    print()

    # 生成SDVS参数
    print("SDVS参数生成:")
    sdvs_params = coal_adapter.generate_sdvs_parameters("shearer", "shearer_cutting_drum")
    for key, value in sdvs_params.items():
        print(f"  {key}: {value}")
    print()

    # 获取采矿特定参数
    print("采矿特定参数:")
    mining_params = coal_adapter.get_mining_specific_parameters(
        mine_depth=800.0,
        geological_condition="complex"
    )
    for key, value in mining_params.items():
        print(f"  {key}: {value}")
    print()

    # 导出配置
    coal_adapter.export_configuration("coal_mining_adapter_config.json", "json")
    print("配置已导出到 coal_mining_adapter_config.json")

    print("=== 演示结束 ===")