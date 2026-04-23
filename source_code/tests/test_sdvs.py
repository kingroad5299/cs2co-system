"""
CS2CO系统SDVS模块测试
测试系统动态脆弱性评估功能

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import os
import sys
import pytest
import asyncio
import tempfile
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.sdvs.sdvs_model import SDVSModel, ComponentNode, SparePart, ComponentStatus
    from src.services.sdvs_service import SDVSService, SDVSCalculationResult
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Error importing CS2CO modules for testing: {e}")
    sys.exit(1)


# 设置测试日志器
logger_manager = get_logger("test_sdvs")
logger = logger_manager.get_logger()


class TestSDVSModel:
    """SDVS模型测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.sdvs_model = SDVSModel(alpha=0.5, gamma=1.0, mu=0.5)
        self.current_time = datetime.now()

        # 创建测试组件
        self.test_components = [
            ComponentNode(
                id="test_comp_001",
                name="测试组件1",
                component_type="test",
                criticality=0.8,
                downtime_cost=10000.0,
                maintenance_interval=500,
                last_maintenance=self.current_time - timedelta(days=30),
                status=ComponentStatus.NORMAL,
                health_score=0.9,
                weibull_shape=2.0,
                weibull_scale=8000.0,
                environmental_factor=1.0
            ),
            ComponentNode(
                id="test_comp_002",
                name="测试组件2",
                component_type="test",
                criticality=0.6,
                downtime_cost=8000.0,
                maintenance_interval=600,
                last_maintenance=self.current_time - timedelta(days=20),
                status=ComponentStatus.NORMAL,
                health_score=0.8,
                weibull_shape=1.8,
                weibull_scale=10000.0,
                environmental_factor=1.2
            )
        ]

        # 创建测试备件
        self.test_spare_parts = [
            SparePart(
                id="test_sp_001",
                name="测试备件1",
                component_id="test_comp_001",
                unit_cost=5000.0,
                holding_cost_rate=0.15,
                ordering_cost=500.0,
                lead_time_mean=15.0,
                lead_time_std=3.0,
                substitutable_parts=["test_sp_002"],
                current_inventory=5,
                safety_stock=2,
                reorder_point=3,
                economic_order_quantity=10
            ),
            SparePart(
                id="test_sp_002",
                name="测试备件2",
                component_id="test_comp_002",
                unit_cost=3000.0,
                holding_cost_rate=0.12,
                ordering_cost=300.0,
                lead_time_mean=10.0,
                lead_time_std=2.0,
                substitutable_parts=[],
                current_inventory=8,
                safety_stock=3,
                reorder_point=4,
                economic_order_quantity=12
            )
        ]

        # 添加组件和备件到模型
        for comp in self.test_components:
            self.sdvs_model.add_component(comp)

        for sp in self.test_spare_parts:
            self.sdvs_model.add_spare_part(sp)

        # 添加依赖关系
        self.sdvs_model.add_dependency("test_comp_002", "test_comp_001", weight=1.0)

    def test_model_initialization(self):
        """测试模型初始化"""
        assert self.sdvs_model.alpha == 0.5
        assert self.sdvs_model.gamma == 1.0
        assert self.sdvs_model.mu == 0.5
        assert len(self.sdvs_model.components) == 2
        assert len(self.sdvs_model.spare_parts) == 2
        assert self.sdvs_model.graph.number_of_edges() == 1

    def test_add_component(self):
        """测试添加组件"""
        new_component = ComponentNode(
            id="test_comp_003",
            name="测试组件3",
            component_type="test",
            criticality=0.7,
            downtime_cost=6000.0,
            maintenance_interval=700,
            last_maintenance=self.current_time - timedelta(days=15),
            status=ComponentStatus.NORMAL,
            health_score=0.85
        )

        self.sdvs_model.add_component(new_component)
        assert "test_comp_003" in self.sdvs_model.components
        assert self.sdvs_model.graph.has_node("test_comp_003")

    def test_add_dependency(self):
        """测试添加依赖关系"""
        self.sdvs_model.add_dependency("test_comp_001", "test_comp_002", weight=0.8)
        assert self.sdvs_model.graph.has_edge("test_comp_001", "test_comp_002")

    def test_add_spare_part(self):
        """测试添加备件"""
        new_spare = SparePart(
            id="test_sp_003",
            name="测试备件3",
            component_id="test_comp_001",
            unit_cost=4000.0,
            holding_cost_rate=0.14,
            ordering_cost=400.0,
            lead_time_mean=12.0,
            lead_time_std=2.5
        )

        self.sdvs_model.add_spare_part(new_spare)
        assert "test_sp_003" in self.sdvs_model.spare_parts
        assert self.sdvs_model.component_to_spare["test_comp_001"] == "test_sp_001"

    def test_calculate_betweenness_centrality(self):
        """测试计算介数中心性"""
        betweenness = self.sdvs_model.calculate_betweenness_centrality(normalized=True)
        assert isinstance(betweenness, dict)
        assert len(betweenness) == 2
        assert all(0 <= value <= 1 for value in betweenness.values())

    def test_calculate_topological_consequence(self):
        """测试计算拓扑后果标量"""
        betweenness = self.sdvs_model.calculate_betweenness_centrality(normalized=True)
        consequence = self.sdvs_model.calculate_topological_consequence("test_comp_001", betweenness)

        assert 0 <= consequence <= 1
        # test_comp_001 具有更高的关键性和停机成本，应该具有更高的拓扑后果
        consequence2 = self.sdvs_model.calculate_topological_consequence("test_comp_002", betweenness)
        assert consequence >= consequence2

    def test_calculate_supply_chain_resistance(self):
        """测试计算供应链阻力标量"""
        resistance = self.sdvs_model.calculate_supply_chain_resistance("test_sp_001")
        assert 0 <= resistance <= 1

        # test_sp_001 具有更长的提前期，应该具有更高的供应链阻力
        resistance2 = self.sdvs_model.calculate_supply_chain_resistance("test_sp_002")
        assert resistance >= resistance2

    def test_calculate_health_factor(self):
        """测试计算健康因子"""
        health = self.sdvs_model.calculate_health_factor("test_comp_001", self.current_time)
        assert 0 <= health <= 1

    def test_calculate_sdvs(self):
        """测试计算SDVS值"""
        sdvs_value = self.sdvs_model.calculate_sdvs("test_sp_001", self.current_time)
        assert 0 <= sdvs_value <= 1
        logger.info(f"SDVS value for test_sp_001: {sdvs_value}")

    def test_batch_calculate_sdvs(self):
        """测试批量计算SDVS值"""
        results = self.sdvs_model.batch_calculate_sdvs(self.current_time)
        assert isinstance(results, dict)
        assert len(results) == 2
        assert all(0 <= value <= 1 for value in results.values())

    def test_classify_vulnerability_level(self):
        """测试脆弱性等级分类"""
        test_cases = [
            (0.05, "极低"),
            (0.2, "低"),
            (0.4, "中"),
            (0.6, "高"),
            (0.8, "极高"),
            (0.95, "紧急")
        ]

        for value, expected_level in test_cases:
            level = self.sdvs_model.classify_vulnerability_level(value)
            assert level == expected_level

    def test_get_critical_components(self):
        """测试获取高脆弱性组件"""
        critical = self.sdvs_model.get_critical_components(threshold=0.1, current_time=self.current_time)
        assert isinstance(critical, list)
        # 所有备件的SDVS值都应该大于0.1
        assert len(critical) == 2

    def test_get_system_vulnerability_summary(self):
        """测试获取系统脆弱性摘要"""
        summary = self.sdvs_model.get_system_vulnerability_summary(self.current_time)
        assert isinstance(summary, dict)
        assert "mean_vulnerability" in summary
        assert "critical_count" in summary
        assert summary["total_count"] == 2

    def test_export_model_parameters(self):
        """测试导出模型参数"""
        params = self.sdvs_model.export_model_parameters()
        assert isinstance(params, dict)
        assert params["component_count"] == 2
        assert params["spare_part_count"] == 2
        assert params["dependency_count"] == 1


class TestSDVSService:
    """SDVS服务测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        config = {
            "sdvs_model": {
                "alpha": 0.5,
                "gamma": 1.0,
                "mu": 0.5,
                "vulnerability_thresholds": {
                    "critical": 0.7,
                    "high": 0.5,
                    "medium": 0.3,
                    "low": 0.1
                }
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        # 创建SDVS服务实例
        self.sdvs_service = SDVSService(self.config_path)

        # 创建测试数据
        self.current_time = datetime.now()
        self.test_spare_part_ids = ["test_sp_001", "test_sp_002"]

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 删除临时文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_service_initialization(self):
        """测试服务初始化"""
        assert self.sdvs_service is not None
        assert self.sdvs_service.sdvs_model is not None
        assert "critical" in self.sdvs_service.vulnerability_thresholds

    @pytest.mark.asyncio
    async def test_calculate_sdvs_async(self):
        """测试异步计算SDVS"""
        # 首先需要添加一些测试数据到模型
        # 这里我们直接测试错误情况（因为没有实际数据）
        try:
            result = await self.sdvs_service.calculate_sdvs_async("non_existent_spare_part", self.current_time)
            # 如果没有抛出异常，结果应该为空
            assert result is None
        except Exception as e:
            # 预期会抛出异常
            assert "not found" in str(e) or "Failed" in str(e)

    @pytest.mark.asyncio
    async def test_batch_calculate_sdvs(self):
        """测试批量计算SDVS"""
        results = await self.sdvs_service.batch_calculate_sdvs(self.test_spare_part_ids, self.current_time)
        assert isinstance(results, list)
        # 由于没有实际数据，结果列表应该为空
        assert len(results) == 0

    def test_analyze_vulnerability_trend(self):
        """测试分析脆弱性趋势"""
        # 创建模拟数据库会话
        class MockSession:
            def query(self, *args):
                return self

            def filter(self, *args):
                return self

            def order_by(self, *args):
                return self

            def limit(self, *args):
                return self

            def all(self):
                return []

        mock_db = MockSession()
        analysis = self.sdvs_service.analyze_vulnerability_trend(mock_db, "test_sp_001", 30)

        assert isinstance(analysis, dict)
        assert analysis["spare_part_id"] == "test_sp_001"
        assert analysis["record_count"] == 0

    def test_identify_critical_components(self):
        """测试识别关键组件"""
        # 创建模拟数据库会话
        class MockSession:
            pass

        mock_db = MockSession()
        critical = self.sdvs_service.identify_critical_components(mock_db, threshold=0.7, limit=10)

        assert isinstance(critical, list)
        # 由于没有实际数据，结果应该为空
        assert len(critical) == 0

    def test_generate_vulnerability_report(self):
        """测试生成脆弱性报告"""
        # 创建模拟数据库会话
        class MockSession:
            def query(self, *args):
                return self

            def order_by(self, *args):
                return self

            def first(self):
                return None

        mock_db = MockSession()
        report = self.sdvs_service.generate_vulnerability_report(mock_db, "daily")

        assert isinstance(report, dict)
        assert report["report_type"] == "daily"
        assert "system_summary" in report
        assert "critical_components" in report

    def test_get_service_status(self):
        """测试获取服务状态"""
        status = self.sdvs_service.get_service_status()

        assert isinstance(status, dict)
        assert status["service"] == "SDVSService"
        assert status["status"] == "running"
        assert "component_count" in status
        assert "spare_part_count" in status


class TestSDVSCalculationResult:
    """SDVS计算结果测试类"""

    def test_result_initialization(self):
        """测试结果初始化"""
        current_time = datetime.now()
        result = SDVSCalculationResult(
            spare_part_id="test_sp_001",
            sdvs_value=0.65,
            topological_consequence=0.7,
            supply_resistance=0.8,
            health_factor=0.9,
            vulnerability_level="高",
            is_critical=True,
            calculation_time=current_time,
            details={"test": "data"}
        )

        assert result.spare_part_id == "test_sp_001"
        assert result.sdvs_value == 0.65
        assert result.topological_consequence == 0.7
        assert result.supply_resistance == 0.8
        assert result.health_factor == 0.9
        assert result.vulnerability_level == "高"
        assert result.is_critical is True
        assert result.calculation_time == current_time
        assert result.details == {"test": "data"}

    def test_to_dict(self):
        """测试转换为字典"""
        current_time = datetime.now()
        result = SDVSCalculationResult(
            spare_part_id="test_sp_001",
            sdvs_value=0.65,
            topological_consequence=0.7,
            supply_resistance=0.8,
            health_factor=0.9,
            vulnerability_level="高",
            is_critical=True,
            calculation_time=current_time
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["spare_part_id"] == "test_sp_001"
        assert result_dict["sdvs_value"] == 0.65
        assert result_dict["calculation_time"] == current_time.isoformat()


def test_integration():
    """集成测试"""
    # 创建完整的SDVS模型
    sdvs_model = SDVSModel()

    # 添加测试数据
    current_time = datetime.now()

    components = [
        ComponentNode(
            id="integration_comp_001",
            name="集成测试组件1",
            component_type="integration",
            criticality=0.9,
            downtime_cost=20000.0,
            maintenance_interval=400,
            last_maintenance=current_time - timedelta(days=25),
            health_score=0.88
        ),
        ComponentNode(
            id="integration_comp_002",
            name="集成测试组件2",
            component_type="integration",
            criticality=0.7,
            downtime_cost=15000.0,
            maintenance_interval=550,
            last_maintenance=current_time - timedelta(days=15),
            health_score=0.92
        )
    ]

    spare_parts = [
        SparePart(
            id="integration_sp_001",
            name="集成测试备件1",
            component_id="integration_comp_001",
            unit_cost=6000.0,
            lead_time_mean=18.0,
            lead_time_std=4.0
        ),
        SparePart(
            id="integration_sp_002",
            name="集成测试备件2",
            component_id="integration_comp_002",
            unit_cost=4000.0,
            lead_time_mean=12.0,
            lead_time_std=2.0
        )
    ]

    for comp in components:
        sdvs_model.add_component(comp)

    for sp in spare_parts:
        sdvs_model.add_spare_part(sp)

    sdvs_model.add_dependency("integration_comp_002", "integration_comp_001", weight=1.0)

    # 执行集成测试
    assert len(sdvs_model.components) == 2
    assert len(sdvs_model.spare_parts) == 2

    # 计算SDVS值
    sdvs_values = sdvs_model.batch_calculate_sdvs(current_time)
    assert len(sdvs_values) == 2
    assert all(0 <= v <= 1 for v in sdvs_values.values())

    # 获取系统摘要
    summary = sdvs_model.get_system_vulnerability_summary(current_time)
    assert summary["total_count"] == 2
    assert 0 <= summary["mean_vulnerability"] <= 1

    logger.info("Integration test completed successfully")
    logger.info(f"SDVS values: {sdvs_values}")
    logger.info(f"System summary: {summary}")


if __name__ == "__main__":
    """直接运行测试"""
    import sys

    # 运行所有测试
    test_classes = [
        TestSDVSModel(),
        TestSDVSService(),
        TestSDVSCalculationResult()
    ]

    success_count = 0
    fail_count = 0

    for test_class in test_classes:
        test_class.setup_method()

        # 运行所有测试方法
        for method_name in dir(test_class):
            if method_name.startswith("test_") and callable(getattr(test_class, method_name)):
                try:
                    method = getattr(test_class, method_name)

                    # 处理异步测试
                    if asyncio.iscoroutinefunction(method):
                        asyncio.run(method())
                    else:
                        method()

                    print(f"✓ {method_name} passed")
                    success_count += 1
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
                    fail_count += 1

        test_class.teardown_method()

    # 运行集成测试
    try:
        test_integration()
        print("✓ Integration test passed")
        success_count += 1
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        fail_count += 1

    print(f"\nTest Results: {success_count} passed, {fail_count} failed")

    if fail_count > 0:
        sys.exit(1)
    else:
        print("All tests passed!")