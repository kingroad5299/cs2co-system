"""
安全平滑演化引擎
包含分层迁移学习、CMDP安全护盾、规则兜底护盾和在线域自适应
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import warnings
from collections import deque
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats as stats

from ..sdvs.sdvs_model import SDVSModel


class TrainingPhase(Enum):
    """训练阶段"""
    COLD_START = "cold_start"          # 冷启动阶段
    ONLINE_ADAPTATION = "online_adaptation"  # 在线适应阶段
    SAFETY_EVALUATION = "safety_evaluation"  # 安全评估阶段


class SafetyShieldType(Enum):
    """安全护盾类型"""
    CMDP_CONSTRAINT = "cmdp_constraint"      # CMDP约束护盾
    RULE_BASED = "rule_based"                # 规则兜底护盾
    LOOKAHEAD = "lookahead"                  # 前向推演护盾
    GRADIENT_REVERSAL = "gradient_reversal"  # 梯度反转护盾


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本配置
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99  # 折扣因子
    lambda_: float = 0.95  # GAE lambda

    # 安全约束配置
    safety_threshold: float = 0.95  # 系统可用性阈值
    cost_constraint: float = 1000000.0  # 成本约束
    max_constraint_violation: float = 0.05  # 最大约束违反率

    # 迁移学习配置
    freeze_feature_layers: bool = True
    transfer_middle_layers: bool = True
    randomize_output_layer: bool = True
    source_domain_weight: float = 0.7

    # 域自适应配置
    mmd_threshold: float = 0.3  # MMD漂移检测阈值
    grl_weight: float = 0.1  # 梯度反转层权重
    adaptation_rate: float = 0.01  # 自适应学习率

    # 训练控制
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    checkpoint_frequency: int = 500


class HierarchicalTransferLearning:
    """
    分层迁移学习
    支持冷启动场景下的零数据训练
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_extractor = None
        self.middle_layers = None
        self.output_layer = None
        self.source_policy = None
        self.target_policy = None

    def initialize_from_source(self, source_model_path: str):
        """从源域模型初始化"""
        print("从源域模型初始化分层迁移网络...")

        # 加载源域模型
        source_checkpoint = torch.load(source_model_path)

        # 特征提取层：冻结迁移
        self.feature_extractor = source_checkpoint['feature_extractor']
        if self.config.freeze_feature_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("特征提取层已冻结")

        # 中间层：初始化迁移
        self.middle_layers = source_checkpoint['middle_layers']
        if self.config.transfer_middle_layers:
            print("中间层已迁移初始化")
        else:
            # 重新初始化中间层
            self._initialize_middle_layers_randomly()
            print("中间层已随机初始化")

        # 输出层：随机初始化
        if self.config.randomize_output_layer:
            self._initialize_output_layer_randomly()
            print("输出层已随机初始化")
        else:
            self.output_layer = source_checkpoint['output_layer']
            print("输出层已迁移")

    def _initialize_middle_layers_randomly(self):
        """随机初始化中间层"""
        # 这里应该根据实际网络结构实现
        self.middle_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def _initialize_output_layer_randomly(self):
        """随机初始化输出层"""
        # 这里应该根据实际动作空间实现
        self.output_layer = nn.Linear(64, 10)  # 假设有10个动作

    def warmup_pretraining(self, simulated_data: Dict[str, torch.Tensor]):
        """
        半仿真预热预训练

        Args:
            simulated_data: 仿真生成的数据
        """
        print("开始半仿真预热预训练...")

        # 创建数据集
        states = simulated_data['states']
        actions = simulated_data['actions']
        rewards = simulated_data['rewards']
        next_states = simulated_data['next_states']

        dataset = TensorDataset(states, actions, rewards, next_states)
        dataloader = DataLoader(dataset,
                               batch_size=self.config.batch_size,
                               shuffle=True)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.get_trainable_parameters(),
                              lr=self.config.learning_rate)

        # 训练循环
        num_epochs = 100
        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch_states, batch_actions, batch_rewards, batch_next_states in dataloader:
                # 前向传播
                features = self.feature_extractor(batch_states)
                hidden = self.middle_layers(features)
                predicted_actions = self.output_layer(hidden)

                # 计算损失
                loss = criterion(predicted_actions, batch_actions)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"预热预训练 Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        print("预热预训练完成")

    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []

        if not self.config.freeze_feature_layers:
            trainable_params.extend(self.feature_extractor.parameters())

        if self.config.transfer_middle_layers:
            trainable_params.extend(self.middle_layers.parameters())

        if self.config.randomize_output_layer:
            trainable_params.extend(self.output_layer.parameters())

        return trainable_params

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_extractor(state)
        hidden = self.middle_layers(features)
        action = self.output_layer(hidden)
        return action


class CMDPConstraintShield:
    """
    CMDP（约束马尔可夫决策过程）安全护盾
    基于拉格朗日对偶的硬约束处理
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.lagrange_multiplier = torch.tensor(0.0, requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.lagrange_multiplier],
                                          lr=self.config.learning_rate * 0.1)

        # 约束历史记录
        self.constraint_history = deque(maxlen=100)
        self.violation_history = deque(maxlen=100)

    def compute_lagrangian(self,
                          policy_loss: torch.Tensor,
                          constraint_value: torch.Tensor) -> torch.Tensor:
        """
        计算拉格朗日函数

        L(θ, λ) = J_R(θ) - λ · (J_C(θ) - ε_max)

        Args:
            policy_loss: 策略损失 J_R(θ)
            constraint_value: 约束函数值 J_C(θ)

        Returns:
            拉格朗日函数值
        """
        constraint_violation = constraint_value - self.config.safety_threshold
        lagrangian = policy_loss - self.lagrange_multiplier * constraint_violation
        return lagrangian

    def update_lagrange_multiplier(self, constraint_value: torch.Tensor):
        """
        更新拉格朗日乘子

        λ_{t+1} = max(0, λ_t + η_λ · (J_C(θ) - ε_max))

        Args:
            constraint_value: 约束函数值 J_C(θ)
        """
        constraint_violation = constraint_value - self.config.safety_threshold

        # 记录约束违反情况
        self.constraint_history.append(constraint_value.item())
        self.violation_history.append(max(0, constraint_violation.item()))

        # 计算乘子更新
        lambda_update = self.lagrange_multiplier + \
                       self.config.learning_rate * constraint_violation
        self.lagrange_multiplier = torch.max(torch.tensor(0.0), lambda_update)

        # 更新优化器中的参数
        self.lagrange_multiplier.requires_grad = True
        for param_group in self.lambda_optimizer.param_groups:
            param_group['params'] = [self.lagrange_multiplier]

    def get_safety_penalty(self, constraint_value: torch.Tensor) -> torch.Tensor:
        """
        获取安全惩罚项

        Args:
            constraint_value: 约束函数值

        Returns:
            安全惩罚
        """
        constraint_violation = constraint_value - self.config.safety_threshold
        safety_penalty = self.lagrange_multiplier * constraint_violation
        return safety_penalty

    def check_safety_violation(self, constraint_value: float) -> bool:
        """
        检查安全约束是否违反

        Args:
            constraint_value: 约束函数值

        Returns:
            是否违反安全约束
        """
        return constraint_value < self.config.safety_threshold * (1 - self.config.max_constraint_violation)

    def get_constraint_statistics(self) -> Dict[str, float]:
        """获取约束统计信息"""
        if not self.constraint_history:
            return {
                "mean_constraint": 0.0,
                "max_violation": 0.0,
                "violation_rate": 0.0,
                "current_multiplier": self.lagrange_multiplier.item()
            }

        constraint_array = np.array(self.constraint_history)
        violation_array = np.array(self.violation_history)

        stats_dict = {
            "mean_constraint": float(np.mean(constraint_array)),
            "std_constraint": float(np.std(constraint_array)),
            "max_violation": float(np.max(violation_array)),
            "mean_violation": float(np.mean(violation_array)),
            "violation_rate": float(np.mean(violation_array > 0)),
            "current_multiplier": self.lagrange_multiplier.item()
        }

        return stats_dict


class RuleBasedSafetyShield:
    """
    规则兜底安全护盾
    基于前向推演和启发式规则的安全保障
    """

    def __init__(self, sdvs_model: SDVSModel, config: TrainingConfig):
        self.sdvs_model = sdvs_model
        self.config = config
        self.lookahead_horizon = 7  # 前向推演视野（天）
        self.safety_inventory_level = {}  # 安全库存水平
        self.intervention_history = deque(maxlen=1000)

    def initialize_safety_levels(self, spare_parts: Dict[str, Any]):
        """初始化安全库存水平"""
        for spare_part_id, spare_part in spare_parts.items():
            # 基于SDVS值设置安全库存
            sdvs_value = self.sdvs_model.calculate_sdvs(
                spare_part_id, datetime.now())

            # 高脆弱性备件需要更高的安全库存
            safety_multiplier = 1.0 + sdvs_value * 2.0  # SDVS越高，乘数越大
            safety_level = max(spare_part.safety_stock * safety_multiplier, 1)

            self.safety_inventory_level[spare_part_id] = safety_level

        print(f"初始化安全库存水平完成，共{len(self.safety_inventory_level)}个备件")

    def lookahead_simulation(self,
                           current_inventory: Dict[str, int],
                           proposed_actions: Dict[str, Any],
                           demand_forecast: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        前向推演仿真

        Args:
            current_inventory: 当前库存
            proposed_actions: 建议动作
            demand_forecast: 需求预测

        Returns:
            推演结果
        """
        # 简化的前向推演
        projected_inventory = current_inventory.copy()
        interventions = {}

        for spare_part_id in current_inventory:
            # 获取安全库存水平
            safety_level = self.safety_inventory_level.get(spare_part_id, 1)

            # 模拟未来L天的库存变化
            for day in range(self.lookahead_horizon):
                # 消耗库存（基于需求预测）
                daily_demand = demand_forecast.get(spare_part_id, [0])[0]
                projected_inventory[spare_part_id] -= daily_demand

                # 检查是否需要干预
                if projected_inventory[spare_part_id] < safety_level:
                    # 触发干预
                    intervention_needed = max(
                        safety_level - projected_inventory[spare_part_id],
                        0
                    )

                    if spare_part_id not in interventions:
                        interventions[spare_part_id] = {
                            'day': day,
                            'shortage': intervention_needed,
                            'safety_level': safety_level
                        }

                    # 假设干预立即补充库存
                    projected_inventory[spare_part_id] += intervention_needed

        return {
            'projected_inventory': projected_inventory,
            'interventions': interventions,
            'lookahead_horizon': self.lookahead_horizon
        }

    def apply_safety_shield(self,
                          rl_action: Dict[str, Any],
                          current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用规则安全护盾

        Args:
            rl_action: 强化学习输出的动作
            current_state: 当前状态

        Returns:
            调整后的安全动作
        """
        current_inventory = current_state.get('inventory', {})
        demand_forecast = current_state.get('demand_forecast', {})

        # 执行前向推演
        lookahead_result = self.lookahead_simulation(
            current_inventory, rl_action, demand_forecast)

        interventions = lookahead_result['interventions']

        # 如果没有干预需要，直接返回RL动作
        if not interventions:
            self.intervention_history.append({
                'timestamp': datetime.now(),
                'intervened': False,
                'reason': 'no_safety_violation'
            })
            return rl_action

        # 需要干预，生成安全动作
        safe_action = rl_action.copy()

        for spare_part_id, intervention_info in interventions.items():
            # 记录干预
            self.intervention_history.append({
                'timestamp': datetime.now(),
                'intervened': True,
                'spare_part_id': spare_part_id,
                'shortage': intervention_info['shortage'],
                'safety_level': intervention_info['safety_level'],
                'day': intervention_info['day']
            })

            # 覆盖RL动作，强制执行保底补货
            # 这里简化处理，实际应用中会有更复杂的规则
            safe_action[f'procure_{spare_part_id}'] = intervention_info['shortage']

            print(f"安全护盾干预: 备件{spare_part_id}，"
                  f"短缺{intervention_info['shortage']}，"
                  f"在第{intervention_info['day']}天触发")

        return safe_action

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """获取干预统计信息"""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "intervention_rate": 0.0,
                "recent_interventions": [],
                "most_frequent_part": None
            }

        total_interventions = len(self.intervention_history)
        interventions = list(self.intervention_history)

        # 计算干预率（最近100次）
        recent_interventions = interventions[-100:] if len(interventions) > 100 else interventions
        intervention_rate = sum(1 for i in recent_interventions if i['intervened']) / len(recent_interventions)

        # 找出最常干预的备件
        part_interventions = {}
        for intervention in interventions:
            if intervention['intervened']:
                part_id = intervention.get('spare_part_id')
                if part_id:
                    part_interventions[part_id] = part_interventions.get(part_id, 0) + 1

        most_frequent_part = max(part_interventions.items(),
                                key=lambda x: x[1])[0] if part_interventions else None

        return {
            "total_interventions": total_interventions,
            "intervention_rate": intervention_rate,
            "recent_interventions": recent_interventions[-10:],  # 最近10次干预
            "most_frequent_part": most_frequent_part,
            "part_intervention_counts": part_interventions
        }


class MMDGRLDomainAdaptation:
    """
    基于MMD-GRL的在线域自适应
    最大均值差异检测 + 梯度反转层
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.real_data_buffer = deque(maxlen=1000)
        self.anchor_data_buffer = deque(maxlen=1000)
        self.mmd_history = deque(maxlen=100)
        self.domain_classifier = None
        self.feature_extractor = None

        # MMD检测参数
        self.mmd_threshold = config.mmd_threshold
        self.drift_detected = False
        self.last_drift_time = None

    def initialize_network(self, input_dim: int, feature_dim: int):
        """初始化域自适应网络"""
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # 域分类器
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_val=self.config.grl_weight),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 2个域：源域和目标域
            nn.Softmax(dim=1)
        )

        print(f"初始化域自适应网络: 输入维度={input_dim}, 特征维度={feature_dim}")

    def compute_mmd(self, real_data: np.ndarray, anchor_data: np.ndarray) -> float:
        """
        计算最大均值差异（MMD）

        Args:
            real_data: 实时数据
            anchor_data: 基准锚点数据

        Returns:
            MMD值
        """
        if len(real_data) == 0 or len(anchor_data) == 0:
            return 0.0

        # 使用RBF核计算MMD
        n_real = len(real_data)
        n_anchor = len(anchor_data)

        # 计算核矩阵
        K_real_real = rbf_kernel(real_data, real_data, gamma=0.1)
        K_anchor_anchor = rbf_kernel(anchor_data, anchor_data, gamma=0.1)
        K_real_anchor = rbf_kernel(real_data, anchor_data, gamma=0.1)

        # 计算MMD^2
        mmd_squared = (
            np.sum(K_real_real) / (n_real * n_real) +
            np.sum(K_anchor_anchor) / (n_anchor * n_anchor) -
            2 * np.sum(K_real_anchor) / (n_real * n_anchor)
        )

        mmd = np.sqrt(max(mmd_squared, 0))
        return float(mmd)

    def update_data_buffers(self, real_sample: np.ndarray, is_anchor: bool = False):
        """更新数据缓冲区"""
        if is_anchor:
            self.anchor_data_buffer.append(real_sample)
        else:
            self.real_data_buffer.append(real_sample)

    def check_concept_drift(self, current_mmd: float) -> bool:
        """
        检查概念漂移

        Args:
            current_mmd: 当前MMD值

        Returns:
            是否检测到概念漂移
        """
        # 添加EMA滤波
        self.mmd_history.append(current_mmd)
        if len(self.mmd_history) < 10:
            return False

        ema_mmd = self._compute_ema(self.mmd_history, alpha=0.1)

        # 检查是否超过阈值
        drift_detected = ema_mmd > self.mmd_threshold

        if drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.last_drift_time = datetime.now()
            print(f"检测到概念漂移! MMD_EMA={ema_mmd:.4f}, 阈值={self.mmd_threshold}")

        elif not drift_detected and self.drift_detected:
            self.drift_detected = False
            print("概念漂移已缓解")

        return drift_detected

    def _compute_ema(self, values: deque, alpha: float) -> float:
        """计算指数移动平均"""
        if not values:
            return 0.0

        ema = values[0]
        for value in list(values)[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema

    def perform_domain_adaptation(self, real_batch: torch.Tensor,
                                anchor_batch: torch.Tensor):
        """
        执行域自适应训练

        Args:
            real_batch: 实时数据批次
            anchor_batch: 锚点数据批次
        """
        if not self.drift_detected:
            return

        # 准备训练数据
        real_labels = torch.zeros(len(real_batch), dtype=torch.long)
        anchor_labels = torch.ones(len(anchor_batch), dtype=torch.long)

        domain_data = torch.cat([real_batch, anchor_batch], dim=0)
        domain_labels = torch.cat([real_labels, anchor_labels], dim=0)

        # 训练域分类器
        self._train_domain_classifier(domain_data, domain_labels)

        print(f"执行域自适应训练，批次大小: {len(domain_data)}")

    def _train_domain_classifier(self, domain_data: torch.Tensor,
                               domain_labels: torch.Tensor):
        """训练域分类器"""
        if self.domain_classifier is None or self.feature_extractor is None:
            return

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.domain_classifier.parameters()),
            lr=self.config.adaptation_rate
        )

        # 训练一个epoch
        self.feature_extractor.train()
        self.domain_classifier.train()

        optimizer.zero_grad()

        # 提取特征
        features = self.feature_extractor(domain_data)

        # 域分类
        domain_pred = self.domain_classifier(features)

        # 计算损失
        loss = criterion(domain_pred, domain_labels)

        # 反向传播
        loss.backward()
        optimizer.step()

    def get_drift_statistics(self) -> Dict[str, Any]:
        """获取漂移检测统计信息"""
        if not self.mmd_history:
            return {
                "current_mmd": 0.0,
                "ema_mmd": 0.0,
                "drift_detected": False,
                "last_drift_time": None,
                "mmd_history_length": 0
            }

        current_mmd = self.mmd_history[-1] if self.mmd_history else 0.0
        ema_mmd = self._compute_ema(self.mmd_history, alpha=0.1)

        return {
            "current_mmd": current_mmd,
            "ema_mmd": ema_mmd,
            "drift_detected": self.drift_detected,
            "last_drift_time": self.last_drift_time,
            "mmd_history_length": len(self.mmd_history),
            "mmd_threshold": self.mmd_threshold,
            "real_data_buffer_size": len(self.real_data_buffer),
            "anchor_data_buffer_size": len(self.anchor_data_buffer)
        }


class GradientReversalLayer(nn.Module):
    """梯度反转层"""
    def __init__(self, lambda_val: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -self.lambda_val * grad_output


class SafetySmoothEvolutionEngine:
    """
    安全平滑演化引擎主类
    集成所有安全机制
    """

    def __init__(self, sdvs_model: SDVSModel, config: TrainingConfig):
        self.sdvs_model = sdvs_model
        self.config = config

        # 初始化各安全模块
        self.transfer_learning = HierarchicalTransferLearning(config)
        self.cmdp_shield = CMDPConstraintShield(config)
        self.rule_shield = RuleBasedSafetyShield(sdvs_model, config)
        self.domain_adaptation = MMDGRLDomainAdaptation(config)

        # 训练状态
        self.training_phase = TrainingPhase.COLD_START
        self.current_episode = 0
        self.total_training_steps = 0
        self.best_reward = -float('inf')
        self.safety_violations = 0

        # 性能记录
        self.training_history = {
            'episode_rewards': [],
            'episode_constraints': [],
            'safety_interventions': [],
            'domain_drift_detections': []
        }

    def setup_cold_start(self, source_model_path: str,
                        simulated_data: Dict[str, torch.Tensor]):
        """设置冷启动"""
        print("=== 设置冷启动 ===")

        # 1. 分层迁移初始化
        self.transfer_learning.initialize_from_source(source_model_path)

        # 2. 半仿真预热预训练
        self.transfer_learning.warmup_pretraining(simulated_data)

        # 3. 初始化规则护盾安全水平
        # 这里需要实际的备件数据
        print("冷启动设置完成")
        self.training_phase = TrainingPhase.COLD_START

    def train_episode(self, env, agent) -> Dict[str, float]:
        """训练一个episode"""
        self.current_episode += 1

        # 重置环境
        state = env.reset()
        episode_reward = 0.0
        episode_constraint = 0.0
        episode_steps = 0

        # 记录安全干预
        safety_interventions = []

        done = False
        while not done and episode_steps < self.config.max_steps_per_episode:
            self.total_training_steps += 1

            # 获取RL动作
            rl_action = agent.get_action(state)

            # 应用规则安全护盾
            current_state_info = self._get_current_state_info(env)
            safe_action = self.rule_shield.apply_safety_shield(rl_action, current_state_info)

            # 记录是否发生干预
            if rl_action != safe_action:
                safety_interventions.append({
                    'step': episode_steps,
                    'original_action': rl_action,
                    'safe_action': safe_action
                })

            # 执行动作
            next_state, reward, done, info = env.step(safe_action)

            # 收集数据用于域自适应
            self._collect_domain_adaptation_data(state, info)

            # 计算约束值（系统可用性）
            constraint_value = info.get('system_availability', 0.0)

            # 更新CMDP拉格朗日乘子
            constraint_tensor = torch.tensor(constraint_value, dtype=torch.float32)
            self.cmdp_shield.update_lagrange_multiplier(constraint_tensor)

            # 检查安全约束违反
            if self.cmdp_shield.check_safety_violation(constraint_value):
                self.safety_violations += 1
                print(f"警告: 安全约束违反 at step {episode_steps}")

            # 训练智能体（带CMDP约束）
            agent_loss = self._compute_agent_loss(agent, state, safe_action, reward, next_state, done)
            cmdp_penalty = self.cmdp_shield.get_safety_penalty(constraint_tensor)

            total_loss = agent_loss + cmdp_penalty
            agent.update_parameters(total_loss)

            # 更新状态
            state = next_state
            episode_reward += reward
            episode_constraint += constraint_value
            episode_steps += 1

        # 计算平均约束值
        avg_constraint = episode_constraint / episode_steps if episode_steps > 0 else 0.0

        # 检查概念漂移
        if self._check_and_handle_concept_drift():
            print(f"Episode {self.current_episode}: 检测到概念漂移，执行域自适应")

        # 记录训练历史
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_constraints'].append(avg_constraint)
        self.training_history['safety_interventions'].append(len(safety_interventions))

        # 更新最佳奖励
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

        # 定期评估和检查点
        if self.current_episode % self.config.evaluation_frequency == 0:
            self._perform_evaluation(env, agent)

        if self.current_episode % self.config.checkpoint_frequency == 0:
            self._save_checkpoint(agent)

        return {
            'episode': self.current_episode,
            'reward': episode_reward,
            'constraint': avg_constraint,
            'steps': episode_steps,
            'safety_interventions': len(safety_interventions),
            'safety_violations': self.safety_violations,
            'lambda_multiplier': self.cmdp_shield.lagrange_multiplier.item()
        }

    def _get_current_state_info(self, env) -> Dict[str, Any]:
        """获取当前状态信息"""
        # 这里需要根据实际环境实现
        return {
            'inventory': {},
            'demand_forecast': {},
            'system_availability': 1.0
        }

    def _collect_domain_adaptation_data(self, state: Any, info: Dict):
        """收集域自适应数据"""
        # 提取特征用于域自适应
        state_features = self._extract_state_features(state)

        # 判断是否为锚点数据（基于时间或其他标准）
        is_anchor = self.total_training_steps % 100 == 0

        # 更新数据缓冲区
        self.domain_adaptation.update_data_buffers(state_features, is_anchor)

    def _extract_state_features(self, state: Any) -> np.ndarray:
        """从状态中提取特征"""
        # 这里需要根据实际状态表示实现
        if isinstance(state, np.ndarray):
            return state.flatten()
        elif isinstance(state, dict):
            # 提取数值特征
            features = []
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, np.ndarray):
                    features.extend(value.flatten())
            return np.array(features)
        else:
            return np.array([0.0])

    def _compute_agent_loss(self, agent, state, action, reward, next_state, done) -> torch.Tensor:
        """计算智能体损失"""
        # 这里需要根据实际智能体类型实现
        # 简化版本
        return torch.tensor(0.0, requires_grad=True)

    def _check_and_handle_concept_drift(self) -> bool:
        """检查并处理概念漂移"""
        if len(self.domain_adaptation.real_data_buffer) < 100 or \
           len(self.domain_adaptation.anchor_data_buffer) < 100:
            return False

        # 计算当前MMD
        real_data = np.array(list(self.domain_adaptation.real_data_buffer))
        anchor_data = np.array(list(self.domain_adaptation.anchor_data_buffer))

        current_mmd = self.domain_adaptation.compute_mmd(real_data, anchor_data)

        # 检查漂移
        drift_detected = self.domain_adaptation.check_concept_drift(current_mmd)

        if drift_detected:
            # 执行域自适应
            real_batch = torch.tensor(real_data[-100:], dtype=torch.float32)
            anchor_batch = torch.tensor(anchor_data[-100:], dtype=torch.float32)

            self.domain_adaptation.perform_domain_adaptation(real_batch, anchor_batch)

        return drift_detected

    def _perform_evaluation(self, env, agent):
        """执行评估"""
        print(f"=== Episode {self.current_episode} 评估 ===")

        # 这里应该实现完整的评估逻辑
        eval_stats = {
            'episode': self.current_episode,
            'best_reward': self.best_reward,
            'safety_violations': self.safety_violations,
            'training_phase': self.training_phase.value
        }

        # 获取各模块统计信息
        cmdp_stats = self.cmdp_shield.get_constraint_statistics()
        rule_stats = self.rule_shield.get_intervention_statistics()
        drift_stats = self.domain_adaptation.get_drift_statistics()

        print(f"评估结果: {eval_stats}")
        print(f"CMDP统计: {cmdp_stats}")
        print(f"规则护盾统计: {rule_stats}")
        print(f"域漂移统计: {drift_stats}")

        self.training_history['domain_drift_detections'].append(
            drift_stats.get('drift_detected', False)
        )

    def _save_checkpoint(self, agent, path: str = "checkpoints/"):
        """保存检查点"""
        checkpoint = {
            'episode': self.current_episode,
            'best_reward': self.best_reward,
            'safety_violations': self.safety_violations,
            'training_phase': self.training_phase.value,
            'agent_state_dict': agent.state_dict() if hasattr(agent, 'state_dict') else {},
            'transfer_learning': {
                'feature_extractor': self.transfer_learning.feature_extractor.state_dict()
                if self.transfer_learning.feature_extractor else None,
                'middle_layers': self.transfer_learning.middle_layers.state_dict()
                if self.transfer_learning.middle_layers else None,
                'output_layer': self.transfer_learning.output_layer.state_dict()
                if self.transfer_learning.output_layer else None
            },
            'cmdp_shield': {
                'lagrange_multiplier': self.cmdp_shield.lagrange_multiplier.item(),
                'constraint_history': list(self.cmdp_shield.constraint_history),
                'violation_history': list(self.cmdp_shield.violation_history)
            },
            'training_history': self.training_history
        }

        # 创建目录（如果不存在）
        import os
        os.makedirs(path, exist_ok=True)

        checkpoint_file = f"{path}checkpoint_ep{self.current_episode}.pt"
        torch.save(checkpoint, checkpoint_file)

        print(f"检查点已保存: {checkpoint_file}")

    def get_engine_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = {
            'training_phase': self.training_phase.value,
            'current_episode': self.current_episode,
            'total_training_steps': self.total_training_steps,
            'best_reward': self.best_reward,
            'safety_violations': self.safety_violations,
            'average_reward': np.mean(self.training_history['episode_rewards'][-100:])
            if self.training_history['episode_rewards'] else 0.0,
            'average_constraint': np.mean(self.training_history['episode_constraints'][-100:])
            if self.training_history['episode_constraints'] else 0.0,
            'recent_safety_interventions': sum(self.training_history['safety_interventions'][-10:])
            if self.training_history['safety_interventions'] else 0
        }

        # 添加各模块统计
        stats.update({
            'cmdp_stats': self.cmdp_shield.get_constraint_statistics(),
            'rule_shield_stats': self.rule_shield.get_intervention_statistics(),
            'domain_adaptation_stats': self.domain_adaptation.get_drift_statistics()
        })

        return stats


# 示例使用
if __name__ == "__main__":
    print("=== 安全平滑演化引擎演示 ===")

    # 创建配置
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        safety_threshold=0.95,
        max_episodes=1000
    )

    # 创建SDVS模型（简化）
    sdvs_model = SDVSModel()

    # 创建安全演化引擎
    safety_engine = SafetySmoothEvolutionEngine(sdvs_model, config)

    print("安全平滑演化引擎初始化完成")
    print(f"初始训练阶段: {safety_engine.training_phase.value}")

    # 演示获取统计信息
    stats = safety_engine.get_engine_statistics()
    print("引擎统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("=== 演示结束 ===")