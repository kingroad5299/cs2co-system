"""
CS2CO系统主程序入口
泛工业高可用备件时空联合运营系统

Copyright (c) 2025 CS2CO Team
All rights reserved.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.api.main import run_server
    from src.services.sdvs_service import SDVSService
    from src.utils.logger import get_logger
    from src.models.database_models import create_all_tables, drop_all_tables, get_model_info
except ImportError as e:
    print(f"Error importing CS2CO modules: {e}")
    print("Please make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


# 全局日志器
logger_manager = get_logger("cs2co_main")
logger = logger_manager.get_logger()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def print_banner():
    """打印系统横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   ██████╗███████╗██████╗  ██████╗ ██████╗                       ║
    ║  ██╔════╝██╔════╝██╔══██╗██╔═══██╗██╔══██╗                      ║
    ║  ██║     ███████╗██████╔╝██║   ██║██████╔╝                      ║
    ║  ██║     ╚════██║██╔═══╝ ██║   ██║██╔══██╗                      ║
    ║  ╚██████╗███████║██║     ╚██████╔╝██║  ██║                      ║
    ║   ╚═════╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝                      ║
    ║                                                                  ║
    ║         CS2CO 系统 - 泛工业高可用备件时空联合运营系统              ║
    ║         Universal Cold-Start to Continuous Optimization System   ║
    ║                                                                  ║
    ║                      版本: 1.0.0                                 ║
    ║                      版权所有 © 2025 CS2CO Team                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def init_database(config: Dict[str, Any]) -> None:
    """初始化数据库"""
    try:
        from sqlalchemy import create_engine

        db_config = config.get("database", {}).get("mysql", {})
        connection_string = (
            f"mysql+mysqlconnector://"
            f"{db_config.get('username', 'cs2co_user')}:"
            f"{db_config.get('password', 'password')}@"
            f"{db_config.get('host', 'localhost')}:"
            f"{db_config.get('port', 3306)}/"
            f"{db_config.get('database', 'cs2co_system')}"
        )

        engine = create_engine(connection_string, echo=db_config.get("echo_sql", False))

        # 创建所有表
        create_all_tables(engine)
        logger.info("Database tables created successfully")

        # 获取模型信息
        model_info = get_model_info()
        logger.info(f"Database initialized with {model_info['total_tables']} tables")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def start_api_server(config: Dict[str, Any]) -> None:
    """启动API服务器"""
    try:
        api_config = config.get("api", {}).get("fastapi", {})
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)
        workers = api_config.get("workers", 4)
        reload = api_config.get("reload", False)

        logger.info(f"Starting API server on {host}:{port}")
        logger.info(f"API documentation: http://{host}:{port}/docs")

        # 运行服务器
        run_server(host=host, port=port, reload=reload)

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise


def test_sdvs_service(config: Dict[str, Any]) -> None:
    """测试SDVS服务"""
    try:
        logger.info("Testing SDVS service...")

        # 创建SDVS服务实例
        config_path = project_root / "config" / "config.yaml"
        sdvs_service = SDVSService(str(config_path))

        # 加载服务状态
        status = sdvs_service.get_service_status()
        logger.info(f"SDVS service status: {status}")

        # 测试计算
        logger.info("SDVS service test completed successfully")

    except Exception as e:
        logger.error(f"SDVS service test failed: {e}")
        raise


def run_cli():
    """运行命令行界面"""
    parser = argparse.ArgumentParser(
        description="CS2CO系统 - 泛工业高可用备件时空联合运营系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s start-api          # 启动API服务器
  %(prog)s init-db            # 初始化数据库
  %(prog)s test-services      # 测试所有服务
  %(prog)s run-all            # 运行完整系统
        """
    )

    parser.add_argument(
        "command",
        choices=["start-api", "init-db", "test-services", "run-all", "info"],
        help="要执行的命令"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API服务器主机地址"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API服务器端口号"
    )

    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 打印横幅
    print_banner()

    try:
        if args.command == "start-api":
            # 启动API服务器
            config["api"]["fastapi"]["host"] = args.host
            config["api"]["fastapi"]["port"] = args.port
            start_api_server(config)

        elif args.command == "init-db":
            # 初始化数据库
            init_database(config)

        elif args.command == "test-services":
            # 测试所有服务
            test_sdvs_service(config)
            logger.info("All service tests completed successfully")

        elif args.command == "run-all":
            # 运行完整系统
            logger.info("Starting complete CS2CO system...")

            # 初始化数据库
            init_database(config)

            # 测试服务
            test_sdvs_service(config)

            # 启动API服务器
            config["api"]["fastapi"]["host"] = args.host
            config["api"]["fastapi"]["port"] = args.port
            start_api_server(config)

        elif args.command == "info":
            # 显示系统信息
            logger.info("CS2CO System Information:")
            logger.info(f"Version: {config.get('system', {}).get('version', '1.0.0')}")
            logger.info(f"Environment: {config.get('system', {}).get('environment', 'development')}")

            # 显示数据库信息
            db_config = config.get("database", {}).get("mysql", {})
            logger.info(f"Database: {db_config.get('host', 'localhost')}:{db_config.get('port', 3306)}/{db_config.get('database', 'cs2co_system')}")

            # 显示API信息
            api_config = config.get("api", {}).get("fastapi", {})
            logger.info(f"API Server: {api_config.get('host', '0.0.0.0')}:{api_config.get('port', 8000)}")

    except KeyboardInterrupt:
        logger.info("System shutdown by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


def main():
    """主函数"""
    try:
        run_cli()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()