"""
SDVS服务模块
提供系统动态脆弱性评估的完整服务

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import asyncio
import yaml
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import networkx as nx
from scipy.stats import weibull_min
import warnings

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..models.database_models import (
    ComponentNode, SparePart, SDVSRecord, SystemState,
    ComponentStatus, SupplyChainStatus
)
from ..core.sdvs.sdvs_model import SDVSModel, ComponentNode as CoreComponent, SparePart as CoreSparePart


class SDVSCalculationMode(Enum):
    """SDVS计算模式"""
    REALTIME = "realtime"      # 实时计算
    BATCH = "batch"           # 批量计算
    INCREMENTAL = "incremental"  # 增量计算
    SCHEDULED = "scheduled"   # 计划计算


@dataclass
class SDVSCalculationResult:
    """SDVS计算结果"""
    spare_part_id: str
    sdvs_value: float
    topological_consequence: float
    supply_resistance: float
    health_factor: float
    vulnerability_level: str
    is_critical: bool
    calculation_time: datetime
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["calculation_time"] = self.calculation_time.isoformat()
        return result


class SDVSService:
    """
    SDVS服务类
    提供系统动态脆弱性评估的完整服务
    """

    def __init__(self, config_path: str = None):
        """
        初始化SDVS服务

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_path)
        self.sdvs_model = SDVSModel(
            alpha=self.config.get("alpha", 0.5),
            gamma=self.config.get("gamma", 1.0),
            mu=self.config.get("mu", 0.5)
        )
        self.vulnerability_thresholds = self.config.get("vulnerability_thresholds", {})
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = timedelta(hours=1)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "alpha": 0.5,
            "gamma": 1.0,
            "mu": 0.5,
            "vulnerability_thresholds": {
                "critical": 0.7,
                "high": 0.5,
                "medium": 0.3,
                "low": 0.1
            },
            "cache_ttl_hours": 1,
            "batch_size": 100,
            "max_retries": 3
        }

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # 合并配置
                default_config.update(config.get("sdvs_model", {}))
            except Exception as e:
                warnings.warn(f"Failed to load config from {config_path}: {e}")

        return default_config

    def load_components_from_database(self, db: Session) -> int:
        """
        从数据库加载组件数据

        Args:
            db: 数据库会话

        Returns:
            加载的组件数量
        """
        # 清空当前模型
        self.sdvs_model.components.clear()
        self.sdvs_model.spare_parts.clear()
        self.sdvs_model.graph.clear()

        # 加载组件
        components = db.query(ComponentNode).all()
        for comp in components:
            core_comp = CoreComponent(
                id=comp.id,
                name=comp.name,
                component_type=comp.component_type,
                criticality=comp.criticality,
                downtime_cost=comp.downtime_cost,
                maintenance_interval=comp.maintenance_interval,
                last_maintenance=comp.last_maintenance,
                status=ComponentStatus(comp.status.value),
                health_score=comp.health_score,
                weibull_shape=comp.weibull_shape,
                weibull_scale=comp.weibull_scale,
                environmental_factor=comp.environmental_factor
            )
            self.sdvs_model.add_component(core_comp)

        # 加载依赖关系
        # TODO: 需要实现ComponentDependency的查询

        # 加载备件
        spare_parts = db.query(SparePart).all()
        for sp in spare_parts:
            core_sp = CoreSparePart(
                id=sp.id,
                name=sp.name,
                component_id=sp.component_id,
                unit_cost=sp.unit_cost,
                holding_cost_rate=sp.holding_cost_rate,
                ordering_cost=sp.ordering_cost,
                lead_time_mean=sp.lead_time_mean,
                lead_time_std=sp.lead_time_std,
                substitutable_parts=sp.substitutable_parts,
                current_inventory=sp.current_inventory,
                safety_stock=sp.safety_stock,
                reorder_point=sp.reorder_point,
                economic_order_quantity=sp.economic_order_quantity
            )
            self.sdvs_model.add_spare_part(core_sp)

        return len(components)

    async def calculate_sdvs_async(self, spare_part_id: str,
                                 current_time: Optional[datetime] = None,
                                 use_cache: bool = True) -> SDVSCalculationResult:
        """
        异步计算SDVS值

        Args:
            spare_part_id: 备件ID
            current_time: 当前时间，如为None则使用当前系统时间
            use_cache: 是否使用缓存

        Returns:
            SDVS计算结果
        """
        if current_time is None:
            current_time = datetime.now()

        # 检查缓存
        cache_key = f"{spare_part_id}_{current_time.date()}"
        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            cache_time = self.cache_expiry.get(cache_key)
            if cache_time and cache_time > current_time:
                return cache_entry

        try:
            # 计算SDVS值
            sdvs_value = self.sdvs_model.calculate_sdvs(spare_part_id, current_time)

            # 获取各维度值
            betweenness = self.sdvs_model.calculate_betweenness_centrality(normalized=True)
            spare_part = self.sdvs_model.spare_parts[spare_part_id]
            component_id = spare_part.component_id

            topological_consequence = self.sdvs_model.calculate_topological_consequence(
                component_id, betweenness)
            supply_resistance = self.sdvs_model.calculate_supply_chain_resistance(spare_part_id)
            health_factor = self.sdvs_model.calculate_health_factor(component_id, current_time)

            # 确定脆弱性等级
            vulnerability_level = self.sdvs_model.classify_vulnerability_level(sdvs_value)
            is_critical = sdvs_value >= self.vulnerability_thresholds.get("critical", 0.7)

            # 构建结果
            result = SDVSCalculationResult(
                spare_part_id=spare_part_id,
                sdvs_value=sdvs_value,
                topological_consequence=topological_consequence,
                supply_resistance=supply_resistance,
                health_factor=health_factor,
                vulnerability_level=vulnerability_level,
                is_critical=is_critical,
                calculation_time=current_time,
                details={
                    "component_id": component_id,
                    "model_parameters": self.sdvs_model.export_model_parameters(),
                    "cache_hit": False
                }
            )

            # 更新缓存
            if use_cache:
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = current_time + self.cache_ttl

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to calculate SDVS for {spare_part_id}: {e}")

    async def batch_calculate_sdvs(self, spare_part_ids: List[str],
                                 current_time: Optional[datetime] = None,
                                 max_concurrent: int = 10) -> List[SDVSCalculationResult]:
        """
        批量计算SDVS值

        Args:
            spare_part_ids: 备件ID列表
            current_time: 当前时间
            max_concurrent: 最大并发计算数

        Returns:
            SDVS计算结果列表
        """
        if current_time is None:
            current_time = datetime.now()

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def calculate_with_semaphore(spare_part_id: str):
            async with semaphore:
                try:
                    result = await self.calculate_sdvs_async(spare_part_id, current_time, use_cache=True)
                    return result
                except Exception as e:
                    warnings.warn(f"Failed to calculate SDVS for {spare_part_id}: {e}")
                    return None

        # 创建任务列表
        tasks = [calculate_with_semaphore(sp_id) for sp_id in spare_part_ids]

        # 并发执行
        task_results = await asyncio.gather(*tasks, return_exceptions=False)

        # 过滤有效结果
        for result in task_results:
            if result is not None:
                results.append(result)

        return results

    def save_calculation_result(self, db: Session, result: SDVSCalculationResult) -> SDVSRecord:
        """
        保存计算结果到数据库

        Args:
            db: 数据库会话
            result: 计算结果

        Returns:
            保存的SDVS记录
        """
        sdvs_record = SDVSRecord(
            spare_part_id=result.spare_part_id,
            calculation_time=result.calculation_time,
            model_version="1.0.0",
            sdvs_value=result.sdvs_value,
            topological_consequence=result.topological_consequence,
            supply_resistance=result.supply_resistance,
            health_factor=result.health_factor,
            vulnerability_level=result.vulnerability_level,
            is_critical=result.is_critical,
            input_parameters=result.details.get("model_parameters", {}),
            calculation_details=result.details or {}
        )

        db.add(sdvs_record)
        db.commit()
        db.refresh(sdvs_record)

        return sdvs_record

    def get_historical_sdvs(self, db: Session, spare_part_id: str,
                          start_time: datetime, end_time: datetime,
                          limit: int = 100) -> List[SDVSRecord]:
        """
        获取历史SDVS记录

        Args:
            db: 数据库会话
            spare_part_id: 备件ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大记录数

        Returns:
            SDVS记录列表
        """
        records = db.query(SDVSRecord).filter(
            SDVSRecord.spare_part_id == spare_part_id,
            SDVSRecord.calculation_time >= start_time,
            SDVSRecord.calculation_time <= end_time
        ).order_by(SDVSRecord.calculation_time.desc()).limit(limit).all()

        return records

    def analyze_vulnerability_trend(self, db: Session, spare_part_id: str,
                                  window_days: int = 30) -> Dict[str, Any]:
        """
        分析脆弱性趋势

        Args:
            db: 数据库会话
            spare_part_id: 备件ID
            window_days: 时间窗口（天）

        Returns:
            趋势分析结果
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=window_days)

        records = self.get_historical_sdvs(db, spare_part_id, start_time, end_time, limit=1000)

        if not records:
            return {
                "spare_part_id": spare_part_id,
                "window_days": window_days,
                "record_count": 0,
                "message": "No historical data found"
            }

        # 提取数据
        timestamps = [r.calculation_time for r in records]
        sdvs_values = [r.sdvs_value for r in records]
        topological_values = [r.topological_consequence for r in records]
        supply_values = [r.supply_resistance for r in records]
        health_values = [r.health_factor for r in records]

        # 计算趋势
        from scipy import stats
        if len(sdvs_values) > 1:
            x = np.arange(len(sdvs_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, sdvs_values)
            trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        else:
            slope = intercept = r_value = p_value = std_err = 0
            trend = "insufficient_data"

        # 统计指标
        analysis = {
            "spare_part_id": spare_part_id,
            "window_days": window_days,
            "record_count": len(records),
            "time_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            },
            "summary": {
                "mean_sdvs": float(np.mean(sdvs_values)),
                "std_sdvs": float(np.std(sdvs_values)),
                "min_sdvs": float(np.min(sdvs_values)),
                "max_sdvs": float(np.max(sdvs_values)),
                "current_sdvs": float(sdvs_values[-1]) if sdvs_values else 0.0
            },
            "trend_analysis": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "trend": trend
            },
            "component_breakdown": {
                "mean_topological": float(np.mean(topological_values)),
                "mean_supply": float(np.mean(supply_values)),
                "mean_health": float(np.mean(health_values)),
                "contributions": {
                    "topological": float(np.mean(topological_values) / max(np.mean(sdvs_values), 0.001)),
                    "supply": float(np.mean(supply_values) / max(np.mean(sdvs_values), 0.001)),
                    "health": float(np.mean(health_values) / max(np.mean(sdvs_values), 0.001))
                }
            },
            "vulnerability_distribution": self._analyze_vulnerability_distribution(records)
        }

        return analysis

    def _analyze_vulnerability_distribution(self, records: List[SDVSRecord]) -> Dict[str, Any]:
        """分析脆弱性分布"""
        if not records:
            return {}

        levels = {}
        for record in records:
            level = record.vulnerability_level
            levels[level] = levels.get(level, 0) + 1

        total = len(records)
        distribution = {}
        for level, count in levels.items():
            distribution[level] = {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0
            }

        return distribution

    def identify_critical_components(self, db: Session,
                                   threshold: float = None,
                                   limit: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        识别高脆弱性组件

        Args:
            db: 数据库会话
            threshold: 脆弱性阈值，如为None则使用配置阈值
            limit: 返回数量限制

        Returns:
            高脆弱性组件列表，每个元素为(备件ID, SDVS值, 详细信息)
        """
        if threshold is None:
            threshold = self.vulnerability_thresholds.get("critical", 0.7)

        # 获取最新计算结果
        # 这里需要先计算当前所有备件的SDVS值
        current_time = datetime.now()
        all_spare_parts = list(self.sdvs_model.spare_parts.keys())

        # 批量计算
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.batch_calculate_sdvs(all_spare_parts, current_time)
            )
        except:
            # 同步计算
            results = []
            for sp_id in all_spare_parts:
                try:
                    result = asyncio.run(self.calculate_sdvs_async(sp_id, current_time))
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Failed to calculate SDVS for {sp_id}: {e}")

        # 过滤和排序
        critical_results = []
        for result in results:
            if result.sdvs_value >= threshold:
                details = {
                    "topological_consequence": result.topological_consequence,
                    "supply_resistance": result.supply_resistance,
                    "health_factor": result.health_factor,
                    "vulnerability_level": result.vulnerability_level
                }
                critical_results.append((result.spare_part_id, result.sdvs_value, details))

        # 按SDVS值降序排序
        critical_results.sort(key=lambda x: x[1], reverse=True)

        # 限制数量
        return critical_results[:limit]

    def generate_vulnerability_report(self, db: Session,
                                    report_type: str = "daily") -> Dict[str, Any]:
        """
        生成脆弱性报告

        Args:
            db: 数据库会话
            report_type: 报告类型：daily/weekly/monthly

        Returns:
            脆弱性报告
        """
        current_time = datetime.now()

        if report_type == "daily":
            window_days = 1
        elif report_type == "weekly":
            window_days = 7
        elif report_type == "monthly":
            window_days = 30
        else:
            window_days = 1

        # 获取系统总体状态
        system_state = db.query(SystemState).order_by(SystemState.timestamp.desc()).first()

        # 识别关键组件
        critical_components = self.identify_critical_components(db, limit=10)

        # 生成报告
        report = {
            "report_type": report_type,
            "generation_time": current_time.isoformat(),
            "time_window_days": window_days,
            "system_summary": {
                "overall_vulnerability": system_state.overall_vulnerability if system_state else 0.0,
                "critical_component_count": len(critical_components),
                "system_availability": system_state.system_availability if system_state else 1.0
            },
            "critical_components": [
                {
                    "spare_part_id": sp_id,
                    "sdvs_value": sdvs_value,
                    "details": details
                }
                for sp_id, sdvs_value, details in critical_components
            ],
            "recommendations": self._generate_recommendations(critical_components),
            "metrics": {
                "calculation_count": len(self.cache),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "model_parameters": self.sdvs_model.export_model_parameters()
            }
        }

        return report

    def _generate_recommendations(self, critical_components: List[Tuple[str, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """生成推荐建议"""
        recommendations = []

        for sp_id, sdvs_value, details in critical_components:
            rec = {
                "spare_part_id": sp_id,
                "priority": "high" if sdvs_value >= 0.7 else "medium" if sdvs_value >= 0.5 else "low",
                "actions": []
            }

            # 根据各个维度生成具体建议
            if details["topological_consequence"] >= 0.7:
                rec["actions"].append({
                    "type": "topology",
                    "description": "组件拓扑重要性高，建议增加冗余备份",
                    "urgency": "high"
                })

            if details["supply_resistance"] >= 0.7:
                rec["actions"].append({
                    "type": "supply_chain",
                    "description": "供应链阻力大，建议寻找替代供应商或增加库存",
                    "urgency": "high"
                })

            if details["health_factor"] >= 0.7:
                rec["actions"].append({
                    "type": "maintenance",
                    "description": "设备健康状态差，建议立即进行维护检查",
                    "urgency": "high"
                })

            if rec["actions"]:
                recommendations.append(rec)

        return recommendations

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not self.cache_expiry:
            return 0.0

        current_time = datetime.now()
        valid_cache_count = sum(
            1 for expiry in self.cache_expiry.values()
            if expiry > current_time
        )
        total_cache_count = len(self.cache_expiry)

        return valid_cache_count / total_cache_count if total_cache_count > 0 else 0.0

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "service": "SDVSService",
            "status": "running",
            "model_loaded": len(self.sdvs_model.components) > 0,
            "component_count": len(self.sdvs_model.components),
            "spare_part_count": len(self.sdvs_model.spare_parts),
            "cache_size": len(self.cache),
            "cache_valid_count": sum(
                1 for expiry in self.cache_expiry.values()
                if expiry > datetime.now()
            ),
            "config": self.config,
            "last_update": datetime.now().isoformat()
        }