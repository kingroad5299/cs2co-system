"""
MacDec-POMDP（宏动作去中心化部分可观测马尔可夫博弈）框架
用于维修备件时空联合运营的多智能体强化学习环境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings

from ..sdvs.sdvs_model import SDVSModel, SparePart, ComponentNode


class AgentType(Enum):
    """智能体类型"""
    DEMAND_PREDICTION = "demand_prediction"  # 需求预测智能体 (DPA)
    PROCUREMENT = "procurement"              # 采购补给智能体 (PA)
    TRANSSHIPMENT = "transshipment"          # 横向调剂智能体 (TA)
    DYNAMIC_LOCATION = "dynamic_location"    # 动态选址智能体 (DLA)


class ActionType(Enum):
    """动作类型"""
    PROCURE_QUANTITY = "procure_quantity"    # 采购数量
    TRANSSHIP_AMOUNT = "transship_amount"    # 调剂量
    OPEN_WAREHOUSE = "open_warehouse"        # 开设仓库
    CLOSE_WAREHOUSE = "close_warehouse"      # 关闭仓库
    MIGRATE_WAREHOUSE = "migrate_warehouse"  # 迁移仓库
    HOLD = "hold"                            # 保持现状


class WarehouseStatus(Enum):
    """仓库状态"""
    OPERATIONAL = "operational"
    CLOSED = "closed"
    UNDER_CONSTRUCTION = "under_construction"
    MIGRATING = "migrating"


@dataclass
class Warehouse:
    """仓库实体"""
    id: str
    location: Tuple[float, float]  # (x, y)坐标
    capacity: float  # 存储容量（单位）
    fixed_cost: float  # 固定运营成本（元/天）
    variable_cost_rate: float  # 可变成本率（元/单位/天）
    status: WarehouseStatus = WarehouseStatus.OPERATIONAL
    opening_date: Optional[datetime] = None
    closing_date: Optional[datetime] = None
    construction_duration: int = 30  # 建设周期（天）
    migration_duration: int = 15  # 迁移周期（天）
    inventory: Dict[str, int] = field(default_factory=dict)  # 备件ID -> 库存量

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.fixed_cost < 0:
            raise ValueError("fixed_cost must be non-negative")
        if self.variable_cost_rate < 0:
            raise ValueError("variable_cost_rate must be non-negative")


@dataclass
class MacroAction:
    """宏动作定义"""
    action_type: ActionType
    target_warehouse_id: str
    new_location: Optional[Tuple[float, float]] = None
    internal_policy: Optional[Dict] = None
    duration: int = 0  # 执行时长（天）
    start_time: Optional[datetime] = None
    completion_probability: float = 1.0  # 完成概率

    def __post_init__(self):
        if self.duration < 0:
            raise ValueError("duration must be non-negative")
        if not 0 <= self.completion_probability <= 1:
            raise ValueError("completion_probability must be between 0 and 1")


class CS2COEnvironment(gym.Env):
    """
    CS2CO强化学习环境
    基于MacDec-POMDP的多智能体环境
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 sdvs_model: SDVSModel,
                 warehouses: List[Warehouse],
                 spare_parts: List[SparePart],
                 time_horizon: int = 365,
                 max_warehouses: int = 10,
                 procurement_cost_multiplier: float = 1.0,
                 transshipment_cost_rate: float = 0.1,
                 system_availability_target: float = 0.95,
                 budget_constraint: float = 1000000.0,
                 render_mode: Optional[str] = None):
        """
        初始化环境

        Args:
            sdvs_model: SDVS模型实例
            warehouses: 仓库列表
            spare_parts: 备件列表
            time_horizon: 时间范围（天）
            max_warehouses: 最大仓库数量
            procurement_cost_multiplier: 采购成本乘数
            transshipment_cost_rate: 调剂成本率
            system_availability_target: 系统可用性目标
            budget_constraint: 预算约束
            render_mode: 渲染模式
        """
        super().__init__()

        self.sdvs_model = sdvs_model
        self.warehouses = {wh.id: wh for wh in warehouses}
        self.spare_parts = {sp.id: sp for sp in spare_parts}
        self.time_horizon = time_horizon
        self.max_warehouses = max_warehouses
        self.procurement_cost_multiplier = procurement_cost_multiplier
        self.transshipment_cost_rate = transshipment_cost_rate
        self.system_availability_target = system_availability_target
        self.budget_constraint = budget_constraint
        self.render_mode = render_mode

        # 环境状态
        self.current_time = datetime.now()
        self.time_step = 0
        self.total_reward = 0.0
        self.system_availability = 1.0
        self.total_cost = 0.0
        self.stockouts = defaultdict(int)  # 备件ID -> 缺货次数

        # DLA冷却倒计时器
        self.dla_cooling_timers: Dict[str, int] = {}  # 仓库ID -> 冷却剩余天数
        self.min_reconstruction_interval = 90  # 最小重构间隔（天）

        # 智能体定义
        self.agents = {
            AgentType.DEMAND_PREDICTION: self._create_demand_prediction_agent(),
            AgentType.PROCUREMENT: self._create_procurement_agent(),
            AgentType.TRANSSHIPMENT: self._create_transshipment_agent(),
            AgentType.DYNAMIC_LOCATION: self._create_dynamic_location_agent()
        }

        # 定义观测空间和动作空间
        self._define_spaces()

        # 历史记录
        self.history = {
            "rewards": [],
            "costs": [],
            "availabilities": [],
            "actions": [],
            "inventory_levels": defaultdict(list),
            "sdvs_values": []
        }

    def _define_spaces(self):
        """定义观测和动作空间"""
        # 状态空间维度
        n_spare_parts = len(self.spare_parts)
        n_warehouses = len(self.warehouses)

        # 观测空间：库存状态 + SDVS值 + 网络拓扑 + DLA冷却计时器
        obs_dim = (n_spare_parts * n_warehouses +  # 库存矩阵
                  n_spare_parts +                 # SDVS值
                  n_warehouses * n_warehouses +   # 网络距离矩阵
                  n_warehouses)                   # DLA冷却计时器

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 动作空间（分智能体定义）
        self.action_spaces = {
            AgentType.DEMAND_PREDICTION: spaces.Discrete(3),  # 需求预测模式
            AgentType.PROCUREMENT: spaces.Box(
                low=0.0,
                high=100.0,
                shape=(n_spare_parts,),
                dtype=np.float32
            ),
            AgentType.TRANSSHIPMENT: spaces.Box(
                low=0.0,
                high=50.0,
                shape=(n_spare_parts * n_warehouses * n_warehouses,),
                dtype=np.float32
            ),
            AgentType.DYNAMIC_LOCATION: spaces.Discrete(4)  # 宏动作类型
        }

    def _create_demand_prediction_agent(self) -> Dict:
        """创建需求预测智能体"""
        return {
            "type": AgentType.DEMAND_PREDICTION,
            "observation_space": self.observation_space,
            "action_space": self.action_spaces[AgentType.DEMAND_PREDICTION],
            "state": None,
            "policy": None
        }

    def _create_procurement_agent(self) -> Dict:
        """创建采购补给智能体"""
        return {
            "type": AgentType.PROCUREMENT,
            "observation_space": self.observation_space,
            "action_space": self.action_spaces[AgentType.PROCUREMENT],
            "state": None,
            "policy": None
        }

    def _create_transshipment_agent(self) -> Dict:
        """创建横向调剂智能体"""
        return {
            "type": AgentType.TRANSSHIPMENT,
            "observation_space": self.observation_space,
            "action_space": self.action_spaces[AgentType.TRANSSHIPMENT],
            "state": None,
            "policy": None
        }

    def _create_dynamic_location_agent(self) -> Dict:
        """创建动态选址智能体"""
        return {
            "type": AgentType.DYNAMIC_LOCATION,
            "observation_space": self.observation_space,
            "action_space": self.action_spaces[AgentType.DYNAMIC_LOCATION],
            "state": None,
            "policy": None,
            "macro_action_queue": deque(maxlen=10),
            "current_macro_action": None
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境状态"""
        super().reset(seed=seed)

        # 重置时间
        self.current_time = datetime.now()
        self.time_step = 0
        self.total_reward = 0.0
        self.system_availability = 1.0
        self.total_cost = 0.0
        self.stockouts.clear()

        # 重置DLA冷却计时器
        self.dla_cooling_timers.clear()
        for warehouse_id in self.warehouses:
            self.dla_cooling_timers[warehouse_id] = 0

        # 重置仓库库存
        for warehouse in self.warehouses.values():
            warehouse.inventory.clear()
            # 初始化库存
            for spare_part_id in self.spare_parts:
                warehouse.inventory[spare_part_id] = random.randint(5, 20)

        # 重置历史记录
        for key in self.history:
            if isinstance(self.history[key], list):
                self.history[key].clear()
            elif isinstance(self.history[key], defaultdict):
                self.history[key].clear()

        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self) -> np.ndarray:
        """获取当前环境观测"""
        n_spare_parts = len(self.spare_parts)
        n_warehouses = len(self.warehouses)

        # 1. 库存状态矩阵 (n_warehouses × n_spare_parts)
        inventory_matrix = []
        for warehouse_id, warehouse in self.warehouses.items():
            for spare_part_id in self.spare_parts:
                inventory = warehouse.inventory.get(spare_part_id, 0)
                inventory_matrix.append(inventory)

        # 2. SDVS值向量
        sdvs_values = list(self.sdvs_model.batch_calculate_sdvs(self.current_time).values())

        # 3. 网络距离矩阵 (n_warehouses × n_warehouses)
        distance_matrix = []
        warehouse_ids = list(self.warehouses.keys())
        for i, wh1_id in enumerate(warehouse_ids):
            for j, wh2_id in enumerate(warehouse_ids):
                if i == j:
                    distance = 0.0
                else:
                    wh1 = self.warehouses[wh1_id]
                    wh2 = self.warehouses[wh2_id]
                    distance = math.sqrt((wh1.location[0] - wh2.location[0])**2 +
                                       (wh1.location[1] - wh2.location[1])**2)
                distance_matrix.append(distance)

        # 4. DLA冷却计时器向量
        cooling_timers = [self.dla_cooling_timers.get(wh_id, 0)
                         for wh_id in warehouse_ids]

        # 拼接所有观测
        observation = np.concatenate([
            np.array(inventory_matrix, dtype=np.float32),
            np.array(sdvs_values, dtype=np.float32),
            np.array(distance_matrix, dtype=np.float32),
            np.array(cooling_timers, dtype=np.float32)
        ])

        return observation

    def _get_info(self) -> Dict:
        """获取环境信息"""
        # 计算关键指标
        total_inventory = sum(
            sum(wh.inventory.values())
            for wh in self.warehouses.values()
        )

        avg_inventory_level = total_inventory / (len(self.spare_parts) * len(self.warehouses))

        # 计算缺货率
        total_demand = sum(sp.reorder_point for sp in self.spare_parts.values())
        total_stockout = sum(self.stockouts.values())
        stockout_rate = total_stockout / total_demand if total_demand > 0 else 0.0

        info = {
            "time_step": self.time_step,
            "current_time": self.current_time,
            "total_reward": self.total_reward,
            "system_availability": self.system_availability,
            "total_cost": self.total_cost,
            "total_inventory": total_inventory,
            "avg_inventory_level": avg_inventory_level,
            "stockout_rate": stockout_rate,
            "warehouse_count": len(self.warehouses),
            "spare_part_count": len(self.spare_parts),
            "budget_utilization": self.total_cost / self.budget_constraint
        }

        return info

    def step(self, actions: Dict[AgentType, Any]):
        """
        执行一步环境更新

        Args:
            actions: 各智能体的动作字典

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 更新DLA冷却计时器
        self._update_cooling_timers()

        # 执行各智能体动作
        rewards = {}
        costs = {}

        # 1. 需求预测智能体动作
        if AgentType.DEMAND_PREDICTION in actions:
            reward, cost = self._execute_demand_prediction_action(
                actions[AgentType.DEMAND_PREDICTION])
            rewards[AgentType.DEMAND_PREDICTION] = reward
            costs[AgentType.DEMAND_PREDICTION] = cost

        # 2. 采购补给智能体动作
        if AgentType.PROCUREMENT in actions:
            reward, cost = self._execute_procurement_action(
                actions[AgentType.PROCUREMENT])
            rewards[AgentType.PROCUREMENT] = reward
            costs[AgentType.PROCUREMENT] = cost

        # 3. 横向调剂智能体动作
        if AgentType.TRANSSHIPMENT in actions:
            reward, cost = self._execute_transshipment_action(
                actions[AgentType.TRANSSHIPMENT])
            rewards[AgentType.TRANSSHIPMENT] = reward
            costs[AgentType.TRANSSHIPMENT] = cost

        # 4. 动态选址智能体动作（带动作掩膜）
        if AgentType.DYNAMIC_LOCATION in actions:
            valid_action = self._apply_action_mask(
                actions[AgentType.DYNAMIC_LOCATION])
            if valid_action is not None:
                reward, cost = self._execute_dynamic_location_action(valid_action)
                rewards[AgentType.DYNAMIC_LOCATION] = reward
                costs[AgentType.DYNAMIC_LOCATION] = cost

        # 模拟需求产生和库存消耗
        self._simulate_demand_and_consumption()

        # 计算总奖励和成本
        total_reward = sum(rewards.values())
        total_cost = sum(costs.values())

        # 更新系统状态
        self.total_reward += total_reward
        self.total_cost += total_cost
        self.time_step += 1
        self.current_time += timedelta(days=1)

        # 更新系统可用性
        self._update_system_availability()

        # 检查终止条件
        terminated = self.time_step >= self.time_horizon
        truncated = self._check_truncation_conditions()

        # 获取新观测和信息
        observation = self._get_observation()
        info = self._get_info()

        # 添加历史记录
        self.history["rewards"].append(total_reward)
        self.history["costs"].append(total_cost)
        self.history["availabilities"].append(self.system_availability)
        self.history["actions"].append(actions)

        return observation, total_reward, terminated, truncated, info

    def _update_cooling_timers(self):
        """更新DLA冷却计时器"""
        for warehouse_id in self.dla_cooling_timers:
            if self.dla_cooling_timers[warehouse_id] > 0:
                self.dla_cooling_timers[warehouse_id] -= 1

    def _apply_action_mask(self, proposed_action: Any) -> Optional[Any]:
        """
        应用动作掩膜技术
        在冷却期内约束DLA的有效动作空间
        """
        dla_agent = self.agents[AgentType.DYNAMIC_LOCATION]

        # 检查是否有仓库在冷却期内
        warehouses_in_cooldown = [
            wh_id for wh_id, timer in self.dla_cooling_timers.items()
            if timer > 0
        ]

        if warehouses_in_cooldown:
            # 在冷却期内，只能执行HOLD动作
            # 这里我们假设动作空间是离散的，HOLD是动作0
            if isinstance(proposed_action, (int, np.integer)):
                if proposed_action != 0:  # 0表示HOLD
                    warnings.warn(
                        f"DLA action masked: proposed={proposed_action}, "
                        f"forced=HOLD due to cooldown"
                    )
                    return 0  # 强制返回HOLD动作
            else:
                # 对于连续动作空间，返回零动作
                warnings.warn("DLA action masked to zero due to cooldown")
                return np.zeros_like(proposed_action)

        return proposed_action

    def _execute_demand_prediction_action(self, action: int) -> Tuple[float, float]:
        """执行需求预测智能体动作"""
        # 简化的需求预测模型
        # 在实际实现中，这里会调用预测模型
        prediction_accuracy = 0.8 + (action * 0.1)  # 动作影响预测准确度
        prediction_cost = 1000.0 * (1 + action)  # 预测成本

        # 奖励基于预测准确度
        reward = prediction_accuracy * 100.0
        cost = prediction_cost

        return reward, cost

    def _execute_procurement_action(self, action: np.ndarray) -> Tuple[float, float]:
        """执行采购补给智能体动作"""
        total_cost = 0.0
        total_reward = 0.0

        for i, (spare_part_id, spare_part) in enumerate(self.spare_parts.items()):
            procure_quantity = action[i]

            if procure_quantity > 0:
                # 采购成本
                procurement_cost = (
                    spare_part.unit_cost * procure_quantity +
                    spare_part.ordering_cost
                ) * self.procurement_cost_multiplier

                total_cost += procurement_cost

                # 分配到各仓库（简化：均匀分配）
                for warehouse in self.warehouses.values():
                    if warehouse.status == WarehouseStatus.OPERATIONAL:
                        allocated_qty = procure_quantity / len(self.warehouses)
                        warehouse.inventory[spare_part_id] = (
                            warehouse.inventory.get(spare_part_id, 0) +
                            allocated_qty
                        )

                # 奖励：基于SDVS值的加权采购
                sdvs_value = self.sdvs_model.calculate_sdvs(
                    spare_part_id, self.current_time)

                # 高脆弱性备件采购获得更高奖励
                procurement_reward = procure_quantity * (1 + sdvs_value) * 10.0
                total_reward += procurement_reward

        return total_reward, total_cost

    def _execute_transshipment_action(self, action: np.ndarray) -> Tuple[float, float]:
        """执行横向调剂智能体动作"""
        n_spare_parts = len(self.spare_parts)
        n_warehouses = len(self.warehouses)

        # 重塑动作向量为3D张量：备件 × 源仓库 × 目标仓库
        action_3d = action.reshape((n_spare_parts, n_warehouses, n_warehouses))

        total_cost = 0.0
        total_reward = 0.0
        warehouse_ids = list(self.warehouses.keys())

        for spare_part_idx, spare_part_id in enumerate(self.spare_parts.keys()):
            for src_idx, src_wh_id in enumerate(warehouse_ids):
                for dst_idx, dst_wh_id in enumerate(warehouse_ids):
                    if src_idx == dst_idx:
                        continue

                    transship_amount = action_3d[spare_part_idx, src_idx, dst_idx]

                    if transship_amount > 0:
                        src_warehouse = self.warehouses[src_wh_id]
                        dst_warehouse = self.warehouses[dst_wh_id]

                        # 检查源仓库是否有足够库存
                        available_inventory = src_warehouse.inventory.get(spare_part_id, 0)
                        actual_transship = min(transship_amount, available_inventory)

                        if actual_transship > 0:
                            # 更新库存
                            src_warehouse.inventory[spare_part_id] -= actual_transship
                            dst_warehouse.inventory[spare_part_id] = (
                                dst_warehouse.inventory.get(spare_part_id, 0) +
                                actual_transship
                            )

                            # 计算调剂成本（基于距离和数量）
                            src_loc = src_warehouse.location
                            dst_loc = dst_warehouse.location
                            distance = math.sqrt((src_loc[0] - dst_loc[0])**2 +
                                               (src_loc[1] - dst_loc[1])**2)

                            transship_cost = (
                                distance * actual_transship *
                                self.transshipment_cost_rate
                            )
                            total_cost += transship_cost

                            # 奖励：基于SDVS值和库存平衡
                            sdvs_value = self.sdvs_model.calculate_sdvs(
                                spare_part_id, self.current_time)

                            # 库存平衡奖励
                            src_inv_after = src_warehouse.inventory.get(spare_part_id, 0)
                            dst_inv_after = dst_warehouse.inventory.get(spare_part_id, 0)

                            balance_reward = -abs(src_inv_after - dst_inv_after) * 0.1
                            transship_reward = actual_transship * (1 + sdvs_value) * 5.0

                            total_reward += balance_reward + transship_reward

        return total_reward, total_cost

    def _execute_dynamic_location_action(self, action: Union[int, np.ndarray]) -> Tuple[float, float]:
        """执行动态选址智能体动作"""
        total_cost = 0.0
        total_reward = 0.0

        # 简化的DLA动作处理
        # 在实际实现中，这里会处理宏动作队列

        if isinstance(action, (int, np.integer)):
            if action == 0:  # HOLD
                # 保持现状，成本为0，小奖励鼓励稳定
                total_reward = 10.0

            elif action == 1:  # OPEN_WAREHOUSE
                if len(self.warehouses) < self.max_warehouses:
                    # 开设新仓库的成本和奖励
                    total_cost = 50000.0  # 开设成本
                    total_reward = 1000.0  # 网络覆盖奖励

                    # 更新DLA冷却计时器
                    new_warehouse_id = f"wh_{len(self.warehouses) + 1}"
                    self.dla_cooling_timers[new_warehouse_id] = self.min_reconstruction_interval

            elif action == 2:  # CLOSE_WAREHOUSE
                if len(self.warehouses) > 1:
                    # 关闭仓库的成本和奖励
                    total_cost = 20000.0  # 关闭成本
                    total_reward = -500.0  # 关闭惩罚

                    # 更新DLA冷却计时器
                    warehouse_to_close = list(self.warehouses.keys())[-1]
                    self.dla_cooling_timers[warehouse_to_close] = self.min_reconstruction_interval

            elif action == 3:  # MIGRATE_WAREHOUSE
                # 迁移仓库的成本和奖励
                total_cost = 30000.0
                total_reward = 800.0

                # 更新DLA冷却计时器
                warehouse_to_migrate = list(self.warehouses.keys())[0]
                self.dla_cooling_timers[warehouse_to_migrate] = self.min_reconstruction_interval

        return total_reward, total_cost

    def _simulate_demand_and_consumption(self):
        """模拟需求产生和库存消耗"""
        # 简化的需求生成模型
        # 在实际实现中，这里会有更复杂的需求预测模型

        for spare_part_id, spare_part in self.spare_parts.items():
            # 生成随机需求
            base_demand = spare_part.reorder_point * 0.1
            random_factor = random.uniform(0.8, 1.2)
            daily_demand = int(base_demand * random_factor)

            # 尝试从各仓库满足需求
            remaining_demand = daily_demand

            for warehouse in self.warehouses.values():
                if remaining_demand <= 0:
                    break

                inventory = warehouse.inventory.get(spare_part_id, 0)
                fulfill_amount = min(inventory, remaining_demand)

                if fulfill_amount > 0:
                    warehouse.inventory[spare_part_id] -= fulfill_amount
                    remaining_demand -= fulfill_amount

            # 记录缺货
            if remaining_demand > 0:
                self.stockouts[spare_part_id] += remaining_demand

    def _update_system_availability(self):
        """更新系统可用性"""
        # 简化的系统可用性计算
        # 在实际实现中，这里会基于SDVS值和缺货情况计算

        total_stockout = sum(self.stockouts.values())
        total_demand = sum(sp.reorder_point for sp in self.spare_parts.values())

        if total_demand > 0:
            availability = 1.0 - (total_stockout / total_demand)
        else:
            availability = 1.0

        # 平滑更新
        self.system_availability = 0.9 * self.system_availability + 0.1 * availability

    def _check_truncation_conditions(self) -> bool:
        """检查截断条件"""
        # 1. 预算超支
        if self.total_cost > self.budget_constraint:
            return True

        # 2. 系统可用性过低
        if self.system_availability < self.system_availability_target * 0.8:
            return True

        # 3. 连续缺货过多
        recent_stockouts = sum(list(self.stockouts.values())[-7:])  # 最近7天
        if recent_stockouts > 100:  # 阈值
            return True

        return False

    def render(self):
        """渲染环境状态"""
        if self.render_mode == "human":
            print(f"Time Step: {self.time_step}")
            print(f"Current Time: {self.current_time}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"System Availability: {self.system_availability:.4f}")
            print(f"Total Cost: {self.total_cost:.2f}")
            print(f"Budget Utilization: {self.total_cost/self.budget_constraint:.2%}")
            print()

            print("Warehouse Status:")
            for wh_id, warehouse in self.warehouses.items():
                cooldown = self.dla_cooling_timers.get(wh_id, 0)
                print(f"  {wh_id}: {warehouse.status.value}, Cooldown: {cooldown} days")
            print()

            print("Critical Spare Parts (SDVS > 0.7):")
            critical_parts = self.sdvs_model.get_critical_components(
                threshold=0.7, current_time=self.current_time)

            for spare_part_id, sdvs_value in critical_parts[:5]:  # 显示前5个
                stockout = self.stockouts.get(spare_part_id, 0)
                print(f"  {spare_part_id}: SDVS={sdvs_value:.4f}, Stockouts={stockout}")

            if len(critical_parts) > 5:
                print(f"  ... and {len(critical_parts) - 5} more")
            print()

    def close(self):
        """关闭环境"""
        # 清理资源
        pass


# 示例使用
if __name__ == "__main__":
    print("=== CS2CO强化学习环境演示 ===")

    # 创建SDVS模型
    from ..sdvs.sdvs_model import SDVSModel, ComponentNode, SparePart

    sdvs_model = SDVSModel(alpha=0.5, gamma=1.0, mu=0.5)

    # 添加示例组件
    components = [
        ComponentNode(
            id="comp_001",
            name="采煤机截割部",
            component_type="采煤机",
            criticality=0.9,
            downtime_cost=25000.0,
            maintenance_interval=500,
            last_maintenance=datetime.now() - timedelta(days=30),
            health_score=0.85,
            weibull_shape=2.5,
            weibull_scale=8000.0,
            environmental_factor=1.2
        )
    ]

    for comp in components:
        sdvs_model.add_component(comp)

    # 添加示例备件
    spare_parts = [
        SparePart(
            id="sp_001",
            name="截割部齿轮",
            component_id="comp_001",
            unit_cost=5000.0,
            holding_cost_rate=0.15,
            ordering_cost=500.0,
            lead_time_mean=15.0,
            lead_time_std=3.0,
            substitutable_parts=["sp_002"],
            current_inventory=2,
            safety_stock=1,
            reorder_point=3,
            economic_order_quantity=5
        )
    ]

    for sp in spare_parts:
        sdvs_model.add_spare_part(sp)

    # 创建仓库
    warehouses = [
        Warehouse(
            id="wh_001",
            location=(0.0, 0.0),
            capacity=1000.0,
            fixed_cost=1000.0,
            variable_cost_rate=0.05
        ),
        Warehouse(
            id="wh_002",
            location=(10.0, 10.0),
            capacity=800.0,
            fixed_cost=800.0,
            variable_cost_rate=0.04
        )
    ]

    # 创建环境
    env = CS2COEnvironment(
        sdvs_model=sdvs_model,
        warehouses=warehouses,
        spare_parts=spare_parts,
        time_horizon=100,
        max_warehouses=5,
        procurement_cost_multiplier=1.0,
        transshipment_cost_rate=0.1,
        system_availability_target=0.95,
        budget_constraint=500000.0,
        render_mode="human"
    )

    # 重置环境
    obs, info = env.reset()
    print("初始环境信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # 运行几个时间步
    for step in range(5):
        print(f"=== 时间步 {step + 1} ===")

        # 生成随机动作
        actions = {
            AgentType.DEMAND_PREDICTION: random.randint(0, 2),
            AgentType.PROCUREMENT: np.random.uniform(0, 10, size=len(spare_parts)),
            AgentType.TRANSSHIPMENT: np.random.uniform(0, 5, size=len(spare_parts)*len(warehouses)*len(warehouses)),
            AgentType.DYNAMIC_LOCATION: random.randint(0, 3)
        }

        # 执行一步
        obs, reward, terminated, truncated, info = env.step(actions)

        # 渲染环境状态
        env.render()

        if terminated or truncated:
            print("环境已终止!")
            break

    print("=== 演示结束 ===")