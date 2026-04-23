"""
数据库模型定义
定义CS2CO系统的核心数据表和ORM模型

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, JSON, ForeignKey, Enum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.mysql import MEDIUMTEXT, LONGTEXT

# SQLAlchemy基类
Base = declarative_base()

class ComponentStatus(enum.Enum):
    """设备组件状态枚举"""
    NORMAL = "normal"      # 正常
    WARNING = "warning"    # 预警
    CRITICAL = "critical"  # 危险
    FAILED = "failed"      # 故障

class SupplyChainStatus(enum.Enum):
    """供应链状态枚举"""
    AVAILABLE = "available"        # 可用
    DELAYED = "delayed"            # 延迟
    OUT_OF_STOCK = "out_of_stock"  # 缺货
    DISCONTINUED = "discontinued"  # 停产

class MaintenanceType(enum.Enum):
    """维护类型枚举"""
    PREVENTIVE = "preventive"      # 预防性维护
    CORRECTIVE = "corrective"      # 纠正性维护
    PREDICTIVE = "predictive"      # 预测性维护
    EMERGENCY = "emergency"        # 紧急维护

class ComponentNode(Base):
    """
    设备组件节点表
    存储工业设备的组件信息
    """
    __tablename__ = "component_nodes"

    # 主键和基础字段
    id = Column(String(64), primary_key=True, comment="组件唯一标识")
    name = Column(String(128), nullable=False, comment="组件名称")
    component_type = Column(String(64), nullable=False, comment="组件类型")
    description = Column(Text, comment="组件描述")

    # 重要性和成本
    criticality = Column(Float, nullable=False, default=0.0, comment="重要性系数 [0, 1]")
    downtime_cost = Column(Float, nullable=False, default=0.0, comment="停机成本（元/小时）")
    replacement_cost = Column(Float, nullable=False, default=0.0, comment="更换成本（元）")

    # 维护参数
    maintenance_interval = Column(Integer, nullable=False, default=500, comment="维护间隔（小时）")
    last_maintenance = Column(DateTime, nullable=False, comment="最后维护时间")
    next_maintenance = Column(DateTime, comment="下次计划维护时间")

    # 健康状态
    status = Column(Enum(ComponentStatus), nullable=False, default=ComponentStatus.NORMAL, comment="组件状态")
    health_score = Column(Float, nullable=False, default=1.0, comment="健康分数 [0, 1]")
    failure_probability = Column(Float, nullable=False, default=0.0, comment="故障概率 [0, 1]")

    # 可靠性参数
    weibull_shape = Column(Float, nullable=False, default=2.0, comment="威布尔形状参数β")
    weibull_scale = Column(Float, nullable=False, default=10000.0, comment="威布尔尺度参数η（小时）")
    environmental_factor = Column(Float, nullable=False, default=1.0, comment="环境恶劣因子κ")

    # 位置和所属信息
    location = Column(String(256), comment="组件位置")
    equipment_id = Column(String(64), comment="所属设备ID")
    plant_id = Column(String(64), comment="所属工厂ID")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关系
    spare_parts = relationship("SparePart", back_populates="component", cascade="all, delete-orphan")
    maintenance_records = relationship("MaintenanceRecord", back_populates="component", cascade="all, delete-orphan")
    dependencies = relationship("ComponentDependency", foreign_keys="[ComponentDependency.from_component_id]", back_populates="from_component")

    # 索引
    __table_args__ = (
        Index("idx_component_type", "component_type"),
        Index("idx_component_status", "status"),
        Index("idx_equipment_id", "equipment_id"),
        Index("idx_plant_id", "plant_id"),
        Index("idx_last_maintenance", "last_maintenance"),
    )


class ComponentDependency(Base):
    """
    组件依赖关系表
    存储设备组件间的依赖关系
    """
    __tablename__ = "component_dependencies"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="依赖关系ID")
    from_component_id = Column(String(64), ForeignKey("component_nodes.id", ondelete="CASCADE"), nullable=False, comment="依赖方组件ID")
    to_component_id = Column(String(64), ForeignKey("component_nodes.id", ondelete="CASCADE"), nullable=False, comment="被依赖方组件ID")
    dependency_type = Column(String(32), nullable=False, default="functional", comment="依赖类型：functional/structural/operational")
    weight = Column(Float, nullable=False, default=1.0, comment="依赖权重 [0, 1]")
    description = Column(Text, comment="依赖关系描述")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关系
    from_component = relationship("ComponentNode", foreign_keys=[from_component_id], back_populates="dependencies")
    to_component = relationship("ComponentNode", foreign_keys=[to_component_id])

    # 唯一约束和索引
    __table_args__ = (
        UniqueConstraint("from_component_id", "to_component_id", name="uq_dependency_pair"),
        Index("idx_dependency_from", "from_component_id"),
        Index("idx_dependency_to", "to_component_id"),
    )


class SparePart(Base):
    """
    维修备件表
    存储备件库存和采购信息
    """
    __tablename__ = "spare_parts"

    # 主键和基础字段
    id = Column(String(64), primary_key=True, comment="备件唯一标识")
    name = Column(String(128), nullable=False, comment="备件名称")
    part_number = Column(String(64), comment="备件编号")
    manufacturer = Column(String(128), comment="制造商")

    # 关联组件
    component_id = Column(String(64), ForeignKey("component_nodes.id", ondelete="CASCADE"), nullable=False, comment="对应组件ID")

    # 成本和库存
    unit_cost = Column(Float, nullable=False, default=0.0, comment="单位成本（元）")
    holding_cost_rate = Column(Float, nullable=False, default=0.15, comment="持有成本率（%/年）")
    ordering_cost = Column(Float, nullable=False, default=500.0, comment="订购成本（元/次）")

    # 库存状态
    current_inventory = Column(Integer, nullable=False, default=0, comment="当前库存量")
    safety_stock = Column(Integer, nullable=False, default=0, comment="安全库存量")
    reorder_point = Column(Integer, nullable=False, default=0, comment="再订购点")
    economic_order_quantity = Column(Integer, nullable=False, default=0, comment="经济订购批量")
    max_inventory = Column(Integer, comment="最大库存容量")

    # 采购信息
    lead_time_mean = Column(Float, nullable=False, default=15.0, comment="平均采购提前期（天）")
    lead_time_std = Column(Float, nullable=False, default=3.0, comment="采购提前期标准差（天）")
    supplier_id = Column(String(64), comment="主要供应商ID")

    # 可替代性和特性
    substitutable_parts = Column(JSON, default=list, comment="可替代备件ID列表")
    weight_kg = Column(Float, comment="重量（千克）")
    dimensions = Column(String(128), comment="尺寸（长×宽×高）")
    shelf_life_months = Column(Integer, comment="保质期（月）")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    last_ordered = Column(DateTime, comment="最后订购时间")

    # 关系
    component = relationship("ComponentNode", back_populates="spare_parts")
    inventory_records = relationship("InventoryRecord", back_populates="spare_part", cascade="all, delete-orphan")
    order_records = relationship("OrderRecord", back_populates="spare_part", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index("idx_spare_component", "component_id"),
        Index("idx_spare_part_number", "part_number"),
        Index("idx_spare_manufacturer", "manufacturer"),
        Index("idx_spare_inventory", "current_inventory"),
    )


class SystemState(Base):
    """
    系统状态表
    存储系统整体状态和关键指标
    """
    __tablename__ = "system_states"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="状态记录ID")
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, comment="记录时间戳")

    # 系统级指标
    overall_vulnerability = Column(Float, nullable=False, default=0.0, comment="系统总体脆弱性 [0, 1]")
    system_availability = Column(Float, nullable=False, default=1.0, comment="系统可用性 [0, 1]")
    total_downtime_cost = Column(Float, nullable=False, default=0.0, comment="总停机成本（元）")
    total_inventory_cost = Column(Float, nullable=False, default=0.0, comment="总库存成本（元）")

    # 设备状态统计
    component_counts = Column(JSON, nullable=False, default=dict, comment="组件状态统计")
    critical_component_count = Column(Integer, nullable=False, default=0, comment="高危组件数量")

    # 库存状态统计
    inventory_statistics = Column(JSON, nullable=False, default=dict, comment="库存统计")
    low_stock_count = Column(Integer, nullable=False, default=0, comment="低库存备件数量")
    out_of_stock_count = Column(Integer, nullable=False, default=0, comment="缺货备件数量")

    # 维护状态统计
    maintenance_statistics = Column(JSON, nullable=False, default=dict, comment="维护统计")
    overdue_maintenance_count = Column(Integer, nullable=False, default=0, comment="逾期维护数量")

    # 性能指标
    response_time_ms = Column(Float, nullable=False, default=0.0, comment="系统响应时间（毫秒）")
    cpu_usage_percent = Column(Float, nullable=False, default=0.0, comment="CPU使用率（%）")
    memory_usage_percent = Column(Float, nullable=False, default=0.0, comment="内存使用率（%）")
    disk_usage_percent = Column(Float, nullable=False, default=0.0, comment="磁盘使用率（%）")

    # 备注
    notes = Column(Text, comment="状态备注")

    # 索引
    __table_args__ = (
        Index("idx_state_timestamp", "timestamp"),
        Index("idx_state_vulnerability", "overall_vulnerability"),
    )


class MaintenanceRecord(Base):
    """
    维护记录表
    存储设备维护历史和详情
    """
    __tablename__ = "maintenance_records"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="维护记录ID")
    component_id = Column(String(64), ForeignKey("component_nodes.id", ondelete="CASCADE"), nullable=False, comment="维护组件ID")

    # 维护详情
    maintenance_type = Column(Enum(MaintenanceType), nullable=False, comment="维护类型")
    maintenance_date = Column(DateTime, nullable=False, default=datetime.utcnow, comment="维护日期")
    duration_minutes = Column(Integer, nullable=False, default=0, comment="维护时长（分钟）")

    # 成本和资源
    cost = Column(Float, nullable=False, default=0.0, comment="维护成本（元）")
    labor_cost = Column(Float, nullable=False, default=0.0, comment="人工成本（元）")
    material_cost = Column(Float, nullable=False, default=0.0, comment="材料成本（元）")
    spare_parts_used = Column(JSON, default=list, comment="使用的备件列表")

    # 维护人员
    technician_id = Column(String(64), comment="技术人员ID")
    team_id = Column(String(64), comment="维护团队ID")

    # 维护前后状态
    pre_maintenance_status = Column(Enum(ComponentStatus), nullable=False, comment="维护前状态")
    post_maintenance_status = Column(Enum(ComponentStatus), nullable=False, comment="维护后状态")
    pre_health_score = Column(Float, nullable=False, default=0.0, comment="维护前健康分数")
    post_health_score = Column(Float, nullable=False, default=1.0, comment="维护后健康分数")

    # 故障信息（如适用）
    failure_description = Column(Text, comment="故障描述")
    failure_cause = Column(String(256), comment="故障原因")
    corrective_actions = Column(Text, comment="纠正措施")

    # 审核和批准
    approved_by = Column(String(64), comment="批准人")
    approval_date = Column(DateTime, comment="批准日期")
    status = Column(String(32), nullable=False, default="completed", comment="记录状态：planned/in_progress/completed/cancelled")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关系
    component = relationship("ComponentNode", back_populates="maintenance_records")

    # 索引
    __table_args__ = (
        Index("idx_maintenance_component", "component_id"),
        Index("idx_maintenance_date", "maintenance_date"),
        Index("idx_maintenance_type", "maintenance_type"),
        Index("idx_maintenance_status", "status"),
    )


class InventoryRecord(Base):
    """
    库存记录表
    存储库存变动历史
    """
    __tablename__ = "inventory_records"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="库存记录ID")
    spare_part_id = Column(String(64), ForeignKey("spare_parts.id", ondelete="CASCADE"), nullable=False, comment="备件ID")

    # 库存变动
    transaction_type = Column(String(32), nullable=False, comment="交易类型：in/out/adjust/transfer")
    quantity_change = Column(Integer, nullable=False, comment="数量变化（+为入库，-为出库）")
    unit_price = Column(Float, nullable=False, comment="单价（元）")
    total_value = Column(Float, nullable=False, comment="总价值（元）")

    # 变动前后
    previous_quantity = Column(Integer, nullable=False, comment="变动前数量")
    new_quantity = Column(Integer, nullable=False, comment="变动后数量")

    # 相关单据
    reference_number = Column(String(64), comment="参考单号")
    reference_type = Column(String(32), comment="参考类型：order/maintenance/transfer/adjustment")
    reference_id = Column(String(64), comment="参考ID")

    # 位置信息
    warehouse_id = Column(String(64), comment="仓库ID")
    location_code = Column(String(64), comment="货位编码")

    # 操作信息
    operator_id = Column(String(64), comment="操作员ID")
    operation_date = Column(DateTime, nullable=False, default=datetime.utcnow, comment="操作日期")

    # 备注
    remarks = Column(Text, comment="备注")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")

    # 关系
    spare_part = relationship("SparePart", back_populates="inventory_records")

    # 索引
    __table_args__ = (
        Index("idx_inventory_spare_part", "spare_part_id"),
        Index("idx_inventory_transaction_date", "operation_date"),
        Index("idx_inventory_reference", "reference_type", "reference_id"),
        Index("idx_inventory_warehouse", "warehouse_id"),
    )


class OrderRecord(Base):
    """
    采购订单表
    存储备件采购订单信息
    """
    __tablename__ = "order_records"

    id = Column(String(64), primary_key=True, comment="订单唯一标识")
    spare_part_id = Column(String(64), ForeignKey("spare_parts.id", ondelete="CASCADE"), nullable=False, comment="备件ID")

    # 订单详情
    order_date = Column(DateTime, nullable=False, default=datetime.utcnow, comment="订购日期")
    order_quantity = Column(Integer, nullable=False, comment="订购数量")
    unit_price = Column(Float, nullable=False, comment="单价（元）")
    total_amount = Column(Float, nullable=False, comment="总金额（元）")

    # 供应商信息
    supplier_id = Column(String(64), nullable=False, comment="供应商ID")
    supplier_name = Column(String(128), nullable=False, comment="供应商名称")
    contact_person = Column(String(64), comment="联系人")
    contact_phone = Column(String(32), comment="联系电话")

    # 物流信息
    expected_delivery_date = Column(DateTime, comment="预计交货日期")
    actual_delivery_date = Column(DateTime, comment="实际交货日期")
    shipping_method = Column(String(32), comment="运输方式")
    tracking_number = Column(String(64), comment="物流追踪号")

    # 订单状态
    status = Column(String(32), nullable=False, default="pending", comment="订单状态：pending/confirmed/shipped/delivered/cancelled")
    supply_chain_status = Column(Enum(SupplyChainStatus), nullable=False, default=SupplyChainStatus.AVAILABLE, comment="供应链状态")

    # 财务信息
    payment_status = Column(String(32), default="unpaid", comment="付款状态：unpaid/partial/paid")
    payment_date = Column(DateTime, comment="付款日期")
    invoice_number = Column(String(64), comment="发票号码")

    # 备注
    notes = Column(Text, comment="订单备注")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关系
    spare_part = relationship("SparePart", back_populates="order_records")

    # 索引
    __table_args__ = (
        Index("idx_order_spare_part", "spare_part_id"),
        Index("idx_order_date", "order_date"),
        Index("idx_order_supplier", "supplier_id"),
        Index("idx_order_status", "status"),
        Index("idx_order_delivery", "expected_delivery_date"),
    )


class SDVSRecord(Base):
    """
    SDVS记录表
    存储系统动态脆弱性评估结果
    """
    __tablename__ = "sdvs_records"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="SDVS记录ID")
    spare_part_id = Column(String(64), ForeignKey("spare_parts.id", ondelete="CASCADE"), nullable=False, comment="备件ID")

    # 计算参数
    calculation_time = Column(DateTime, nullable=False, default=datetime.utcnow, comment="计算时间")
    model_version = Column(String(32), nullable=False, default="1.0.0", comment="模型版本")

    # SDVS值和各维度
    sdvs_value = Column(Float, nullable=False, comment="SDVS值 [0, 1]")
    topological_consequence = Column(Float, nullable=False, comment="拓扑后果标量𝒞_i [0, 1]")
    supply_resistance = Column(Float, nullable=False, comment="供应链阻力标量𝒮_i [0, 1]")
    health_factor = Column(Float, nullable=False, comment="动态健康因子ℋ_i(t) [0, 1]")

    # 脆弱性等级
    vulnerability_level = Column(String(16), nullable=False, comment="脆弱性等级：极低/低/中/高/极高/紧急")
    is_critical = Column(Boolean, nullable=False, default=False, comment="是否高危")

    # 输入参数
    input_parameters = Column(JSON, nullable=False, default=dict, comment="输入参数")
    calculation_details = Column(JSON, nullable=False, default=dict, comment="计算详情")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")

    # 关系
    spare_part = relationship("SparePart")

    # 索引
    __table_args__ = (
        Index("idx_sdvs_spare_part", "spare_part_id"),
        Index("idx_sdvs_calculation_time", "calculation_time"),
        Index("idx_sdvs_value", "sdvs_value"),
        Index("idx_sdvs_level", "vulnerability_level"),
        Index("idx_sdvs_critical", "is_critical"),
    )


class ModelParameter(Base):
    """
    模型参数表
    存储各种算法的参数配置
    """
    __tablename__ = "model_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="参数ID")
    model_name = Column(String(64), nullable=False, comment="模型名称")
    parameter_name = Column(String(128), nullable=False, comment="参数名称")

    # 参数值
    parameter_value = Column(JSON, nullable=False, comment="参数值")
    data_type = Column(String(32), nullable=False, comment="数据类型：int/float/str/bool/list/dict")

    # 参数范围和约束
    min_value = Column(Float, comment="最小值")
    max_value = Column(Float, comment="最大值")
    default_value = Column(JSON, comment="默认值")

    # 参数描述
    description = Column(Text, comment="参数描述")
    unit = Column(String(32), comment="参数单位")

    # 版本和状态
    version = Column(String(32), nullable=False, default="1.0.0", comment="参数版本")
    is_active = Column(Boolean, nullable=False, default=True, comment="是否启用")

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 唯一约束和索引
    __table_args__ = (
        UniqueConstraint("model_name", "parameter_name", "version", name="uq_model_parameter"),
        Index("idx_model_name", "model_name"),
        Index("idx_parameter_name", "parameter_name"),
        Index("idx_model_version", "version"),
    )


# 定义所有模型的列表
ALL_MODELS = [
    ComponentNode,
    ComponentDependency,
    SparePart,
    SystemState,
    MaintenanceRecord,
    InventoryRecord,
    OrderRecord,
    SDVSRecord,
    ModelParameter
]

def create_all_tables(engine):
    """创建所有数据库表"""
    Base.metadata.create_all(engine)

def drop_all_tables(engine):
    """删除所有数据库表"""
    Base.metadata.drop_all(engine)

def get_model_info() -> Dict[str, Any]:
    """获取数据库模型信息"""
    return {
        "models": [model.__tablename__ for model in ALL_MODELS],
        "total_tables": len(ALL_MODELS),
        "version": "1.0.0",
        "created_at": datetime.utcnow().isoformat()
    }