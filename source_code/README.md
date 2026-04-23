# CS2CO系统 - 泛工业高可用备件时空联合运营系统

## 项目概述

CS2CO（Universal Cold-Start to Continuous Optimization System）是一个面向工业设备维修备件管理的智能决策支持系统。系统基于系统动态脆弱性（SDVS）模型、MacDec-POMDP多智能体强化学习框架和安全演化引擎，解决备件管理中的冷启动难题。

## 技术架构

### 核心技术
1. **SDVS模型**：系统动态脆弱性评估
2. **MacDec-POMDP**：宏动作去中心化部分可观测马尔可夫博弈
3. **安全强化学习**：CMDP约束优化与安全护盾
4. **迁移学习**：零数据冷启动支持
5. **在线自适应**：概念漂移检测与适应

### 技术栈
- **后端**：Python 3.9+, PyTorch 2.0+, FastAPI
- **前端**：React.js, Ant Design
- **数据库**：MySQL 8.0, Redis
- **部署**：Docker, Kubernetes
- **算法**：强化学习，图算法，优化算法

## 项目结构

```
cs2co_system/
├── docs/                    # 文档
├── src/                    # 源代码
│   ├── core/              # 核心算法模块
│   ├── models/            # 数据模型
│   ├── services/          # 业务服务
│   ├── api/               # API接口
│   ├── utils/             # 工具函数
│   └── web/               # Web前端
├── tests/                  # 测试代码
├── config/                 # 配置文件
├── data/                   # 数据文件
├── scripts/                # 脚本文件
└── requirements/           # 依赖管理
```

## 安装部署

### 环境要求
- Python 3.9+
- MySQL 8.0+
- Redis 6.0+
- CUDA 11.7+（可选，GPU加速）

### 安装步骤
```bash
# 1. 克隆项目
git clone https://github.com/your-org/cs2co-system.git
cd cs2co-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements/production.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑.env文件配置数据库等参数

# 5. 初始化数据库
python scripts/init_database.py

# 6. 启动服务
python src/main.py
```

## 开发指南

### 代码规范
- 遵循PEP 8 Python代码规范
- 使用类型注解
- 编写单元测试
- 提交前运行代码检查

### 开发流程
1. 创建功能分支
2. 开发实现
3. 编写测试
4. 代码审查
5. 合并到主分支

## 模块说明

### 1. SDVS模块（src/core/sdvs/）
- 设备拓扑网络分析
- 供应链阻力计算
- 设备健康状态评估
- 脆弱性标量生成

### 2. 强化学习模块（src/core/rl/）
- 多智能体环境定义
- MacDec-POMDP算法实现
- 安全约束处理
- 训练与推理

### 3. 行业适配模块（src/core/adapters/）
- 煤炭行业参数映射
- 工程机械行业适配
- 通用接口定义

### 4. Web服务模块（src/api/）
- RESTful API接口
- WebSocket实时通信
- 用户认证授权
- 数据可视化

## 测试

### 单元测试
```bash
python -m pytest tests/unit/ -v
```

### 集成测试
```bash
python -m pytest tests/integration/ -v
```

### 性能测试
```bash
python scripts/performance_test.py
```

## 部署

### Docker部署
```bash
docker build -t cs2co-system .
docker run -p 8000:8000 cs2co-system
```

### Kubernetes部署
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 监控与维护

### 监控指标
- 系统可用性
- 响应时间
- 资源使用率
- 模型性能

### 日志管理
- 应用日志
- 访问日志
- 错误日志
- 审计日志

## 贡献指南

### 提交问题
1. 在GitHub Issues中创建问题
2. 描述问题现象和复现步骤
3. 提供相关日志和截图

### 提交代码
1. Fork项目仓库
2. 创建功能分支
3. 提交Pull Request
4. 通过CI/CD检查

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

- **项目负责人**：王露
- **技术支持**：support@example.com
- **官方网站**：https://cs2co.example.com

## 版本历史

### V1.0 (2025-01-15)
- 首次发布CS2CO系统
- 实现核心算法模块
- 提供完整Web界面
- 支持煤炭行业应用

### V1.1 (计划中)
- 扩展行业适配器
- 优化算法性能
- 增强安全机制
- 改进用户体验