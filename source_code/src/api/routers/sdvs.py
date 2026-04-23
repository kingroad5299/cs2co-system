"""
SDVS API路由
提供系统动态脆弱性评估的API端点

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel, Field

from src.api.main import success_response, error_response
from src.services.sdvs_service import SDVSService, SDVSCalculationResult

# 创建路由
router = APIRouter(
    prefix="/api/v1/sdvs",
    tags=["SDVS"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# 安全验证
security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证API密钥"""
    # 实际项目中应该从数据库或配置中读取有效的API密钥
    valid_keys = ["cs2co-api-key-12345", "test-api-key"]
    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials


# 请求/响应模型
class SDVSCalculationRequest(BaseModel):
    """SDVS计算请求"""
    spare_part_ids: List[str] = Field(..., min_items=1, max_items=100, description="备件ID列表")
    calculation_mode: str = Field(default="realtime", description="计算模式：realtime/batch/incremental")
    include_details: bool = Field(default=True, description="是否包含详细信息")
    use_cache: bool = Field(default=True, description="是否使用缓存")

    class Config:
        schema_extra = {
            "example": {
                "spare_part_ids": ["SP001", "SP002", "SP003"],
                "calculation_mode": "realtime",
                "include_details": True,
                "use_cache": True
            }
        }


class SDVSCalculationResponse(BaseModel):
    """SDVS计算响应"""
    calculation_id: str = Field(..., description="计算ID")
    calculation_time: datetime = Field(..., description="计算时间")
    total_calculated: int = Field(..., description="计算总数")
    mode: str = Field(..., description="计算模式")
    results: List[Dict[str, Any]] = Field(..., description="计算结果列表")


class SDVSHistoryRequest(BaseModel):
    """SDVS历史请求"""
    spare_part_id: str = Field(..., description="备件ID")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    limit: int = Field(default=100, ge=1, le=1000, description="最大记录数")


class SDVSReportRequest(BaseModel):
    """SDVS报告请求"""
    report_type: str = Field(default="daily", description="报告类型：daily/weekly/monthly")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="时间窗口（天）")
    include_recommendations: bool = Field(default=True, description="是否包含推荐建议")


class SDVSTrendAnalysisRequest(BaseModel):
    """SDVS趋势分析请求"""
    spare_part_id: str = Field(..., description="备件ID")
    window_days: int = Field(default=30, ge=1, le=365, description="分析窗口（天）")
    include_forecast: bool = Field(default=False, description="是否包含预测")


# 依赖项
def get_sdvs_service():
    """获取SDVS服务实例"""
    # 在实际应用中，应该从应用状态或依赖注入容器中获取
    from src.api.main import app
    return app.state.sdvs_service


@router.post("/calculate", response_model=Dict[str, Any])
async def calculate_sdvs(
    request: SDVSCalculationRequest,
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    计算SDVS值

    根据提供的备件ID列表计算系统动态脆弱性标量值。
    """
    try:
        # 生成计算ID
        import uuid
        calculation_id = str(uuid.uuid4())[:8]
        calculation_time = datetime.now()

        # 执行计算
        results = await sdvs_service.batch_calculate_sdvs(
            spare_part_ids=request.spare_part_ids,
            current_time=calculation_time,
            max_concurrent=10
        )

        # 转换为字典格式
        results_dict = [result.to_dict() for result in results]

        # 构建响应
        response_data = SDVSCalculationResponse(
            calculation_id=calculation_id,
            calculation_time=calculation_time,
            total_calculated=len(results_dict),
            mode=request.calculation_mode,
            results=results_dict
        )

        return success_response(
            data=response_data.dict(),
            message=f"Successfully calculated SDVS for {len(results_dict)} spare parts"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SDVS calculation failed: {str(e)}"
        )


@router.get("/history/{spare_part_id}", response_model=Dict[str, Any])
async def get_sdvs_history(
    spare_part_id: str,
    days: int = Query(default=30, ge=1, le=365, description="历史天数"),
    limit: int = Query(default=100, ge=1, le=1000, description="最大记录数"),
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    获取SDVS历史记录

    获取指定备件在指定时间范围内的SDVS历史记录。
    """
    try:
        # 在实际应用中，这里应该从数据库查询历史记录
        # 为了示例，我们返回模拟数据
        import random
        from datetime import timedelta

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 生成模拟历史数据
        history_data = []
        for i in range(min(limit, 100)):
            timestamp = start_time + timedelta(days=i * (days / min(limit, 100)))
            sdvs_value = random.uniform(0.1, 0.9)

            # 计算各维度值
            topological = random.uniform(0.3, 0.9)
            supply = random.uniform(0.2, 0.8)
            health = random.uniform(0.4, 1.0)

            # 确定脆弱性等级
            if sdvs_value >= 0.7:
                level = "极高"
            elif sdvs_value >= 0.5:
                level = "高"
            elif sdvs_value >= 0.3:
                level = "中"
            else:
                level = "低"

            history_data.append({
                "timestamp": timestamp.isoformat(),
                "sdvs_value": round(sdvs_value, 4),
                "topological_consequence": round(topological, 4),
                "supply_resistance": round(supply, 4),
                "health_factor": round(health, 4),
                "vulnerability_level": level,
                "is_critical": sdvs_value >= 0.7
            })

        response_data = {
            "spare_part_id": spare_part_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "days": days,
            "limit": limit,
            "record_count": len(history_data),
            "history": history_data
        }

        return success_response(
            data=response_data,
            message=f"SDVS history retrieved for spare part {spare_part_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get SDVS history: {str(e)}"
        )


@router.post("/report", response_model=Dict[str, Any])
async def generate_sdvs_report(
    request: SDVSReportRequest,
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    生成SDVS报告

    生成指定类型的SDVS报告，包括系统脆弱性摘要和关键组件分析。
    """
    try:
        # 在实际应用中，这里应该从数据库生成报告
        # 为了示例，我们返回模拟报告
        import random
        from datetime import datetime, timedelta

        # 确定时间窗口
        if request.time_window_days:
            window_days = request.time_window_days
        elif request.report_type == "daily":
            window_days = 1
        elif request.report_type == "weekly":
            window_days = 7
        else:  # monthly
            window_days = 30

        # 生成模拟报告数据
        report_data = {
            "report_id": f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_type": request.report_type,
            "time_window_days": window_days,
            "generation_time": datetime.now().isoformat(),
            "summary": {
                "total_components_analyzed": random.randint(50, 200),
                "critical_components": random.randint(1, 15),
                "high_vulnerability_components": random.randint(5, 30),
                "medium_vulnerability_components": random.randint(20, 60),
                "low_vulnerability_components": random.randint(10, 40),
                "average_vulnerability": round(random.uniform(0.2, 0.6), 4),
                "max_vulnerability": round(random.uniform(0.7, 0.95), 4),
                "min_vulnerability": round(random.uniform(0.05, 0.2), 4),
                "system_availability": round(random.uniform(0.95, 0.99), 4)
            },
            "critical_components": [
                {
                    "spare_part_id": f"SP{random.randint(1000, 9999)}",
                    "component_name": f"组件{random.randint(1, 100)}",
                    "sdvs_value": round(random.uniform(0.7, 0.95), 4),
                    "vulnerability_level": "极高",
                    "topological_consequence": round(random.uniform(0.6, 0.9), 4),
                    "supply_resistance": round(random.uniform(0.5, 0.8), 4),
                    "health_factor": round(random.uniform(0.7, 0.95), 4)
                }
                for _ in range(random.randint(3, 8))
            ],
            "trend_analysis": {
                "overall_trend": random.choice(["上升", "下降", "稳定"]),
                "trend_strength": round(random.uniform(0.3, 0.9), 4),
                "predicted_vulnerability": round(random.uniform(0.3, 0.7), 4),
                "confidence_level": round(random.uniform(0.7, 0.95), 4)
            }
        }

        if request.include_recommendations:
            report_data["recommendations"] = [
                {
                    "id": 1,
                    "priority": "高",
                    "type": "库存优化",
                    "description": "增加高风险备件的安全库存",
                    "action_items": ["调整安全库存水平", "优化订购批量"],
                    "estimated_impact": "降低缺货风险20-30%"
                },
                {
                    "id": 2,
                    "priority": "中",
                    "type": "维护计划",
                    "description": "对健康状态差的设备进行预防性维护",
                    "action_items": ["制定维护计划", "分配维护资源"],
                    "estimated_impact": "提高设备可用性10-15%"
                },
                {
                    "id": 3,
                    "priority": "低",
                    "type": "供应链优化",
                    "description": "优化采购提前期较长的备件供应链",
                    "action_items": ["寻找替代供应商", "建立战略合作关系"],
                    "estimated_impact": "缩短采购提前期15-25%"
                }
            ]

        return success_response(
            data=report_data,
            message=f"{request.report_type.capitalize()} SDVS report generated successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SDVS report: {str(e)}"
        )


@router.post("/trend-analysis", response_model=Dict[str, Any])
async def analyze_sdvs_trend(
    request: SDVSTrendAnalysisRequest,
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    分析SDVS趋势

    分析指定备件的SDVS历史趋势，提供趋势分析和预测。
    """
    try:
        # 在实际应用中，这里应该使用实际的历史数据分析趋势
        # 为了示例，我们返回模拟分析结果
        import random
        import numpy as np
        from datetime import datetime, timedelta

        # 生成模拟历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=request.window_days)

        # 生成时间序列
        dates = [start_time + timedelta(days=i) for i in range(request.window_days)]
        # 生成SDVS值（带有一些趋势和噪声）
        base_trend = np.linspace(0.3, 0.6, request.window_days)
        noise = np.random.normal(0, 0.05, request.window_days)
        sdvs_values = np.clip(base_trend + noise, 0, 1)

        # 计算趋势指标
        if len(sdvs_values) > 1:
            x = np.arange(len(sdvs_values))
            slope = np.polyfit(x, sdvs_values, 1)[0]
            if slope > 0.01:
                trend = "上升"
                trend_strength = min(abs(slope) * 100, 1.0)
            elif slope < -0.01:
                trend = "下降"
                trend_strength = min(abs(slope) * 100, 1.0)
            else:
                trend = "稳定"
                trend_strength = 0.0
        else:
            trend = "数据不足"
            trend_strength = 0.0

        # 构建分析结果
        analysis_data = {
            "spare_part_id": request.spare_part_id,
            "analysis_window_days": request.window_days,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "statistics": {
                "mean_vulnerability": round(float(np.mean(sdvs_values)), 4),
                "std_vulnerability": round(float(np.std(sdvs_values)), 4),
                "min_vulnerability": round(float(np.min(sdvs_values)), 4),
                "max_vulnerability": round(float(np.max(sdvs_values)), 4),
                "current_vulnerability": round(float(sdvs_values[-1]), 4) if len(sdvs_values) > 0 else 0.0
            },
            "trend_analysis": {
                "trend_direction": trend,
                "trend_strength": round(trend_strength, 4),
                "slope": round(float(slope) if 'slope' in locals() else 0.0, 6),
                "volatility": round(float(np.std(np.diff(sdvs_values))) if len(sdvs_values) > 1 else 0.0, 4)
            },
            "historical_data": [
                {
                    "date": date.isoformat(),
                    "sdvs_value": round(value, 4),
                    "vulnerability_level": "极高" if value >= 0.7 else "高" if value >= 0.5 else "中" if value >= 0.3 else "低"
                }
                for date, value in zip(dates, sdvs_values)
            ][-50:]  # 只返回最近50个数据点
        }

        # 如果请求包含预测
        if request.include_forecast and len(sdvs_values) > 10:
            # 简单的线性预测
            forecast_days = 7
            if 'slope' in locals():
                forecast_values = []
                last_value = sdvs_values[-1]
                for i in range(forecast_days):
                    forecast_value = last_value + slope * (i + 1)
                    forecast_values.append(max(0, min(1, forecast_value)))

                analysis_data["forecast"] = {
                    "forecast_days": forecast_days,
                    "predicted_values": [round(v, 4) for v in forecast_values],
                    "confidence_interval": [
                        round(max(0, v - 0.1), 4) for v in forecast_values
                    ],
                    "confidence_interval_high": [
                        round(min(1, v + 0.1), 4) for v in forecast_values
                    ]
                }

        return success_response(
            data=analysis_data,
            message=f"SDVS trend analysis completed for {request.spare_part_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze SDVS trend: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_sdvs_service_status(
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    获取SDVS服务状态

    获取SDVS服务的当前状态和性能指标。
    """
    try:
        status_data = sdvs_service.get_service_status()
        return success_response(
            data=status_data,
            message="SDVS service status retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get SDVS service status: {str(e)}"
        )


@router.get("/vulnerability-levels", response_model=Dict[str, Any])
async def get_vulnerability_levels(
    sdvs_service: SDVSService = Depends(get_sdvs_service),
    api_key: str = Depends(verify_api_key)
):
    """
    获取脆弱性等级定义

    获取SDVS脆弱性等级的定义和阈值。
    """
    try:
        levels = {
            "critical": {
                "threshold": 0.7,
                "description": "紧急 - 需要立即采取措施",
                "color": "#ff4d4f",
                "actions": ["立即维护", "增加库存", "寻找替代"]
            },
            "high": {
                "threshold": 0.5,
                "description": "高 - 需要优先处理",
                "color": "#ffa940",
                "actions": ["计划维护", "监控库存", "评估风险"]
            },
            "medium": {
                "threshold": 0.3,
                "description": "中 - 需要关注",
                "color": "#ffec3d",
                "actions": ["定期检查", "优化库存", "跟踪状态"]
            },
            "low": {
                "threshold": 0.1,
                "description": "低 - 正常监控",
                "color": "#73d13d",
                "actions": ["常规监控", "统计分析", "记录跟踪"]
            },
            "very_low": {
                "threshold": 0.0,
                "description": "极低 - 风险可接受",
                "color": "#389e0d",
                "actions": ["保持现状", "定期评估", "文档记录"]
            }
        }

        return success_response(
            data=levels,
            message="Vulnerability levels retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vulnerability levels: {str(e)}"
        )


# 健康检查端点
@router.get("/health", response_model=Dict[str, Any])
async def sdvs_health_check(sdvs_service: SDVSService = Depends(get_sdvs_service)):
    """SDVS服务健康检查"""
    try:
        status_data = sdvs_service.get_service_status()

        health_status = {
            "service": "SDVSService",
            "status": "healthy" if status_data.get("model_loaded", False) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "model_loaded": status_data.get("model_loaded", False),
                "component_count": status_data.get("component_count", 0),
                "spare_part_count": status_data.get("spare_part_count", 0),
                "cache_status": "active" if status_data.get("cache_size", 0) > 0 else "inactive"
            }
        }

        return success_response(
            data=health_status,
            message="SDVS service health check completed"
        )

    except Exception as e:
        return error_response(
            message=f"SDVS service health check failed: {str(e)}",
            error_code="SDVS_HEALTH_CHECK_FAILED",
            status_code=500
        )