"""
API模块
提供系统的Web API接口

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

__all__ = [
    "create_app",
    "get_api_info",
    "APIVersion",
    "APIResponse"
]

# 模块描述
__description__ = """
API模块提供CS2CO系统的RESTful API接口，包括：
1. SDVS接口：动态脆弱性评估相关API
2. 优化接口：库存优化和调度API
3. 监控接口：系统状态和性能监控API
4. 数据接口：设备、备件、维护数据管理API
5. 行业接口：行业适配和参数配置API
"""

# 版本信息
__version__ = "1.0.0"

# 导入FastAPI相关类
try:
    from fastapi import FastAPI, Depends, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn

    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Any
    from datetime import datetime

    # 定义API响应模型
    class APIResponse(BaseModel):
        """API统一响应模型"""
        success: bool = Field(..., description="请求是否成功")
        message: str = Field(..., description="响应消息")
        data: Optional[Any] = Field(None, description="响应数据")
        error_code: Optional[str] = Field(None, description="错误代码")
        timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")

        class Config:
            json_encoders = {
                datetime: lambda dt: dt.isoformat()
            }

    class APIVersion(BaseModel):
        """API版本信息"""
        version: str = Field(..., description="API版本号")
        name: str = Field(..., description="API名称")
        description: str = Field(..., description="API描述")
        endpoints: List[str] = Field(..., description="可用端点列表")

except ImportError as e:
    print(f"Warning: FastAPI and related modules may not be available: {e}")

    # 提供占位类
    class APIResponse:
        """API响应占位类"""
        def __init__(self, success=True, message="", data=None, error_code=None, timestamp=None):
            self.success = success
            self.message = message
            self.data = data
            self.error_code = error_code
            self.timestamp = timestamp or datetime.now()

    class APIVersion:
        """API版本占位类"""
        def __init__(self, version="1.0.0", name="CS2CO API", description="", endpoints=None):
            self.version = version
            self.name = name
            self.description = description
            self.endpoints = endpoints or []

    def create_app():
        """创建应用占位函数"""
        class MockApp:
            def __init__(self):
                self.title = "CS2CO API"
                self.version = "1.0.0"
        return MockApp()

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    创建FastAPI应用

    Args:
        config: 应用配置

    Returns:
        FastAPI应用实例
    """
    # 默认配置
    default_config = {
        "title": "CS2CO System API",
        "version": __version__,
        "description": __description__.strip(),
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "cors_origins": ["http://localhost:3000"],
        "cors_allow_credentials": True,
        "cors_allow_methods": ["*"],
        "cors_allow_headers": ["*"],
        "debug": False
    }

    if config:
        default_config.update(config)

    # 创建应用
    app = FastAPI(
        title=default_config["title"],
        version=default_config["version"],
        description=default_config["description"],
        docs_url=default_config["docs_url"] if not default_config["debug"] else None,
        redoc_url=default_config["redoc_url"] if not default_config["debug"] else None,
        openapi_url=default_config["openapi_url"]
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=default_config["cors_origins"],
        allow_credentials=default_config["cors_allow_credentials"],
        allow_methods=default_config["cors_allow_methods"],
        allow_headers=default_config["cors_allow_headers"],
    )

    return app


def get_api_info() -> APIVersion:
    """获取API信息"""
    endpoints = [
        "/api/v1/sdvs/calculate",        # SDVS计算
        "/api/v1/sdvs/history",          # SDVS历史
        "/api/v1/sdvs/report",           # SDVS报告
        "/api/v1/optimization/inventory", # 库存优化
        "/api/v1/optimization/schedule",  # 调度优化
        "/api/v1/monitoring/health",      # 健康检查
        "/api/v1/monitoring/metrics",     # 系统指标
        "/api/v1/data/components",        # 组件管理
        "/api/v1/data/spare_parts",       # 备件管理
        "/api/v1/data/maintenance",       # 维护记录
        "/api/v1/industry/adapters",      # 行业适配器
        "/api/v1/config/parameters"       # 参数配置
    ]

    return APIVersion(
        version=__version__,
        name="CS2CO System API",
        description=__description__.strip(),
        endpoints=endpoints
    )


def success_response(data: Any = None, message: str = "请求成功") -> APIResponse:
    """
    创建成功响应

    Args:
        data: 响应数据
        message: 响应消息

    Returns:
        APIResponse实例
    """
    return APIResponse(
        success=True,
        message=message,
        data=data,
        timestamp=datetime.now()
    )


def error_response(message: str, error_code: str = None, status_code: int = 400) -> Dict:
    """
    创建错误响应

    Args:
        message: 错误消息
        error_code: 错误代码
        status_code: HTTP状态码

    Returns:
        包含错误信息的字典
    """
    response = APIResponse(
        success=False,
        message=message,
        error_code=error_code,
        timestamp=datetime.now()
    )

    return {
        "response": response,
        "status_code": status_code
    }


def validate_api_key(api_key: str, valid_keys: List[str]) -> bool:
    """
    验证API密钥

    Args:
        api_key: 待验证的API密钥
        valid_keys: 有效的API密钥列表

    Returns:
        是否验证通过
    """
    return api_key in valid_keys


# 示例API端点定义
class SDVSCalculationRequest(BaseModel):
    """SDVS计算请求"""
    spare_part_ids: List[str] = Field(..., description="备件ID列表")
    calculation_mode: str = Field(default="realtime", description="计算模式：realtime/batch")
    include_details: bool = Field(default=True, description="是否包含详细信息")


class SDVSReportRequest(BaseModel):
    """SDVS报告请求"""
    report_type: str = Field(default="daily", description="报告类型：daily/weekly/monthly")
    time_window_days: Optional[int] = Field(None, description="时间窗口（天）")
    include_recommendations: bool = Field(default=True, description="是否包含推荐建议")


class OptimizationRequest(BaseModel):
    """优化请求"""
    optimization_type: str = Field(..., description="优化类型：inventory/schedule")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="优化参数")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")


# API路由注册装饰器
def register_routes(app: FastAPI):
    """
    注册API路由

    Args:
        app: FastAPI应用实例
    """
    # 这里可以定义具体的路由注册逻辑
    # 实际项目中应该从各个模块导入并注册路由
    pass


def run_api_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    运行API服务器

    Args:
        app: FastAPI应用实例
        host: 主机地址
        port: 端口号
        reload: 是否热重载
    """
    uvicorn.run(app, host=host, port=port, reload=reload)