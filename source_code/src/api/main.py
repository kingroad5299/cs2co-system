"""
FastAPI主应用
CS2CO系统的Web API入口

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import yaml
import logging

from pydantic import BaseModel, Field, validator

# 导入CS2CO模块
try:
    from src.api import create_app, success_response, error_response, get_api_info
    from src.services.sdvs_service import SDVSService
    from src.utils.logger import get_logger, LoggerManager
    from src.models.database_models import Base, get_model_info
except ImportError as e:
    print(f"Error importing CS2CO modules: {e}")
    print("Make sure you are in the correct directory and all dependencies are installed.")
    sys.exit(1)


# 配置加载
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# 依赖项
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
class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="系统版本")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    uptime_seconds: Optional[float] = Field(None, description="运行时间（秒）")
    database_status: Optional[str] = Field(None, description="数据库状态")
    cache_status: Optional[str] = Field(None, description="缓存状态")


class SDVSCalculationRequest(BaseModel):
    """SDVS计算请求"""
    spare_part_ids: List[str] = Field(..., min_items=1, max_items=100, description="备件ID列表")
    calculation_mode: str = Field(default="realtime", description="计算模式：realtime/batch/incremental")
    include_details: bool = Field(default=True, description="是否包含详细信息")

    @validator('calculation_mode')
    def validate_calculation_mode(cls, v):
        valid_modes = ["realtime", "batch", "incremental"]
        if v not in valid_modes:
            raise ValueError(f"calculation_mode must be one of {valid_modes}")
        return v


class SDVSReportRequest(BaseModel):
    """SDVS报告请求"""
    report_type: str = Field(default="daily", description="报告类型：daily/weekly/monthly")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="时间窗口（天）")
    include_recommendations: bool = Field(default=True, description="是否包含推荐建议")

    @validator('report_type')
    def validate_report_type(cls, v):
        valid_types = ["daily", "weekly", "monthly"]
        if v not in valid_types:
            raise ValueError(f"report_type must be one of {valid_types}")
        return v


class OptimizationRequest(BaseModel):
    """优化请求"""
    optimization_type: str = Field(..., description="优化类型：inventory/schedule")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="优化参数")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")

    @validator('optimization_type')
    def validate_optimization_type(cls, v):
        valid_types = ["inventory", "schedule"]
        if v not in valid_types:
            raise ValueError(f"optimization_type must be one of {valid_types}")
        return v


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 启动时
    app_start_time = datetime.now()
    logger.info("Starting CS2CO API server...")

    # 加载配置
    config_path = project_root / "config" / "config.yaml"
    app.state.config = load_config(config_path)

    # 初始化服务
    app.state.sdvs_service = SDVSService(config_path)

    # 设置共享状态
    app.state.start_time = app_start_time
    app.state.logger = logger

    logger.info(f"CS2CO API server started at {app_start_time}")
    logger.info(f"API version: {get_api_info().version}")
    logger.info(f"Available endpoints: {len(get_api_info().endpoints)}")

    yield

    # 关闭时
    logger.info("Shutting down CS2CO API server...")
    uptime = (datetime.now() - app_start_time).total_seconds()
    logger.info(f"Server uptime: {uptime:.2f} seconds")


# 创建FastAPI应用
app_config = {
    "title": "CS2CO System API",
    "version": "1.0.0",
    "description": "泛工业高可用备件时空联合运营系统 API",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json"
}

app = FastAPI(**app_config, lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置日志
logger_manager = get_logger("cs2co_api")
logger = logger_manager.get_logger()


# 中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()

    logger.info(
        f"Request: {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown",
        duration_seconds=duration,
        status_code=response.status_code
    )

    return response


# 根路由
@app.get("/", response_model=Dict[str, Any])
async def root():
    """根端点"""
    return success_response(
        data={
            "name": "CS2CO System API",
            "version": app.version,
            "description": app.description,
            "documentation": "/docs",
            "health_check": "/health",
            "api_info": "/api/info"
        },
        message="CS2CO System API is running"
    )


# 健康检查
@app.get("/health", response_model=Dict[str, Any])
async def health_check(request: Request):
    """健康检查端点"""
    try:
        # 检查SDVS服务
        sdvs_status = request.app.state.sdvs_service.get_service_status()

        # 计算运行时间
        uptime_seconds = (datetime.now() - request.app.state.start_time).total_seconds()

        health_data = HealthCheckResponse(
            status="healthy",
            version=app.version,
            uptime_seconds=uptime_seconds,
            database_status="connected",  # 实际项目中应该检查数据库连接
            cache_status="active" if sdvs_status.get("cache_size", 0) > 0 else "inactive"
        )

        return success_response(data=health_data.dict(), message="System is healthy")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return error_response(
            message=f"Health check failed: {str(e)}",
            error_code="HEALTH_CHECK_ERROR",
            status_code=500
        )


# API信息
@app.get("/api/info", response_model=Dict[str, Any])
async def api_info():
    """API信息端点"""
    api_info_data = get_api_info()
    return success_response(data=api_info_data.dict(), message="API information retrieved")


# SDVS相关端点
@app.post("/api/v1/sdvs/calculate", response_model=Dict[str, Any])
async def calculate_sdvs(
    request_data: SDVSCalculationRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """计算SDVS值"""
    try:
        logger.info(f"Calculating SDVS for {len(request_data.spare_part_ids)} spare parts")

        sdvs_service = request.app.state.sdvs_service

        # 计算SDVS
        import asyncio
        results = await sdvs_service.batch_calculate_sdvs(
            request_data.spare_part_ids,
            max_concurrent=10
        )

        # 转换为字典格式
        results_dict = [result.to_dict() for result in results]

        return success_response(
            data={
                "calculation_mode": request_data.calculation_mode,
                "total_calculated": len(results_dict),
                "results": results_dict
            },
            message=f"Successfully calculated SDVS for {len(results_dict)} spare parts"
        )

    except Exception as e:
        logger.error(f"SDVS calculation failed: {e}")
        return error_response(
            message=f"SDVS calculation failed: {str(e)}",
            error_code="SDVS_CALCULATION_ERROR",
            status_code=500
        )


@app.get("/api/v1/sdvs/history/{spare_part_id}", response_model=Dict[str, Any])
async def get_sdvs_history(
    spare_part_id: str,
    days: int = 30,
    limit: int = 100,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """获取SDVS历史记录"""
    try:
        # 这里应该从数据库获取历史记录
        # 为了示例，我们返回模拟数据
        import random
        from datetime import datetime, timedelta

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 生成模拟历史数据
        history_data = []
        for i in range(min(limit, 50)):
            timestamp = start_time + timedelta(days=i*0.6)
            sdvs_value = random.uniform(0.1, 0.9)

            history_data.append({
                "timestamp": timestamp.isoformat(),
                "sdvs_value": sdvs_value,
                "vulnerability_level": "高" if sdvs_value > 0.7 else "中" if sdvs_value > 0.4 else "低"
            })

        return success_response(
            data={
                "spare_part_id": spare_part_id,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "record_count": len(history_data),
                "history": history_data
            },
            message=f"SDVS history retrieved for spare part {spare_part_id}"
        )

    except Exception as e:
        logger.error(f"Failed to get SDVS history: {e}")
        return error_response(
            message=f"Failed to get SDVS history: {str(e)}",
            error_code="HISTORY_RETRIEVAL_ERROR",
            status_code=500
        )


@app.post("/api/v1/sdvs/report", response_model=Dict[str, Any])
async def generate_sdvs_report(
    request_data: SDVSReportRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """生成SDVS报告"""
    try:
        logger.info(f"Generating {request_data.report_type} SDVS report")

        sdvs_service = request.app.state.sdvs_service

        # 这里应该从数据库生成报告
        # 为了示例，我们返回模拟报告
        import random

        report_data = {
            "report_type": request_data.report_type,
            "generation_time": datetime.now().isoformat(),
            "summary": {
                "total_components": random.randint(50, 200),
                "critical_components": random.randint(1, 10),
                "average_vulnerability": random.uniform(0.2, 0.6),
                "system_availability": random.uniform(0.95, 0.99)
            },
            "critical_components": [
                {
                    "spare_part_id": f"SP{random.randint(1000, 9999)}",
                    "sdvs_value": random.uniform(0.7, 0.95),
                    "vulnerability_level": "极高",
                    "recommendation": "立即进行维护检查"
                }
                for _ in range(random.randint(1, 5))
            ]
        }

        if request_data.include_recommendations:
            report_data["recommendations"] = [
                "增加高风险备件的安全库存",
                "优化采购提前期较长的备件供应链",
                "对健康状态差的设备进行预防性维护"
            ]

        return success_response(
            data=report_data,
            message=f"{request_data.report_type.capitalize()} SDVS report generated successfully"
        )

    except Exception as e:
        logger.error(f"Failed to generate SDVS report: {e}")
        return error_response(
            message=f"Failed to generate SDVS report: {str(e)}",
            error_code="REPORT_GENERATION_ERROR",
            status_code=500
        )


# 优化端点
@app.post("/api/v1/optimization/run", response_model=Dict[str, Any])
async def run_optimization(
    request_data: OptimizationRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """运行优化"""
    try:
        logger.info(f"Running {request_data.optimization_type} optimization")

        # 这里应该调用优化服务
        # 为了示例，我们返回模拟结果
        import random

        if request_data.optimization_type == "inventory":
            optimization_result = {
                "optimization_type": "inventory",
                "status": "completed",
                "savings_estimated": random.uniform(10000, 50000),
                "recommended_actions": [
                    f"调整备件 SP{random.randint(1000, 9999)} 的安全库存至 {random.randint(5, 20)} 件",
                    f"优化备件 SP{random.randint(1000, 9999)} 的订购批量至 {random.randint(10, 50)} 件"
                ]
            }
        else:  # schedule
            optimization_result = {
                "optimization_type": "schedule",
                "status": "completed",
                "schedule_efficiency": random.uniform(0.8, 0.95),
                "maintenance_plan": [
                    {
                        "component_id": f"COMP{random.randint(100, 999)}",
                        "recommended_time": (datetime.now() + timedelta(days=random.randint(1, 7))).isoformat(),
                        "estimated_duration_hours": random.randint(2, 8)
                    }
                    for _ in range(random.randint(3, 8))
                ]
            }

        return success_response(
            data=optimization_result,
            message=f"{request_data.optimization_type.capitalize()} optimization completed successfully"
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return error_response(
            message=f"Optimization failed: {str(e)}",
            error_code="OPTIMIZATION_ERROR",
            status_code=500
        )


# 系统信息端点
@app.get("/api/v1/system/info", response_model=Dict[str, Any])
async def get_system_info(request: Request):
    """获取系统信息"""
    try:
        system_info = {
            "name": "CS2CO System",
            "version": app.version,
            "description": app.description,
            "start_time": request.app.state.start_time.isoformat(),
            "uptime_seconds": (datetime.now() - request.app.state.start_time).total_seconds(),
            "config": request.app.state.config.get("system", {}),
            "services": {
                "sdvs_service": request.app.state.sdvs_service.get_service_status()
            }
        }

        return success_response(data=system_info, message="System information retrieved")

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return error_response(
            message=f"Failed to get system info: {str(e)}",
            error_code="SYSTEM_INFO_ERROR",
            status_code=500
        )


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    logger.warning(f"HTTPException: {exc.detail}", status_code=exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=exc.detail,
            error_code="HTTP_ERROR",
            status_code=exc.status_code
        )
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            status_code=500
        )
    )


# 运行服务器
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    运行API服务器

    Args:
        host: 主机地址
        port: 端口号
        reload: 是否热重载
    """
    logger.info(f"Starting CS2CO API server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # 直接运行时，使用默认配置
    run_server()