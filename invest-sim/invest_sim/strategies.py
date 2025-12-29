from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from .data_models import SimulationConfig


class Strategy(abc.ABC):
    """策略抽象基类。"""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.base_weights = np.array(config.normalized_weights, dtype=float)

    def initialize(self) -> np.ndarray:
        return self.base_weights.copy()

    @abc.abstractmethod
    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回新的权重分配。"""

    @staticmethod
    def _normalize(weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("权重和必须大于 0")
        normalized = weights / total
        return np.clip(normalized, 0.0, 1.0)


class FixedAllocationStrategy(Strategy):
    """固定权重策略，始终回归目标权重。"""

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.base_weights.copy()


class TargetRiskStrategy(Strategy):
    """目标风险策略，若当前风险高于阈值，则向低波动资产倾斜。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.target_volatility = config.strategy.target_volatility or (
            np.sqrt(np.sum((self.base_weights * self._asset_vols) ** 2))
        )
        self._cash_index = int(np.argmin(self._asset_vols))

    @property
    def _asset_vols(self) -> np.ndarray:
        return np.array([asset.volatility for asset in self.config.assets], dtype=float)

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        weights = self.base_weights.copy()
        current_vol = float(np.sqrt(np.sum((weights * self._asset_vols) ** 2)))
        if current_vol <= self.target_volatility:
            return weights

        scale = self.target_volatility / current_vol
        weights *= scale
        residual = 1.0 - weights.sum()
        if residual > 0:
            weights[self._cash_index] += residual
        return self._normalize(weights)


class AdaptiveRebalanceStrategy(Strategy):
    """自适应再平衡策略，只有偏离阈值时才恢复目标权重。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.threshold = config.strategy.rebalance_threshold

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        deviation = np.abs(current_weights - self.base_weights)
        if np.any(deviation > self.threshold):
            return self.base_weights.copy()
        return current_weights


class EqualWeightStrategy(Strategy):
    """等权重策略，所有资产分配相同权重。"""

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(current_weights)
        return np.ones(n) / n


class RiskParityStrategy(Strategy):
    """风险平价策略，根据资产波动率分配权重，使各资产风险贡献相等。"""

    @property
    def _asset_vols(self) -> np.ndarray:
        return np.array([asset.volatility for asset in self.config.assets], dtype=float)

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 风险平价：权重与波动率成反比
        inv_vols = 1.0 / np.maximum(self._asset_vols, 1e-6)  # 避免除零
        weights = inv_vols / inv_vols.sum()
        return self._normalize(weights)


class MinimumVarianceStrategy(Strategy):
    """最小方差策略，基于协方差矩阵优化，最小化组合波动率。"""

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if covariance is None or covariance.size == 0:
            # 如果没有协方差矩阵，回退到等权重
            n = len(current_weights)
            return np.ones(n) / n
        
        # 最小方差组合：w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        try:
            inv_cov = np.linalg.inv(covariance)
            ones = np.ones(len(covariance))
            weights = inv_cov @ ones
            weights = weights / weights.sum()
            return self._normalize(weights)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，回退到等权重
            n = len(current_weights)
            return np.ones(n) / n


class MomentumStrategy(Strategy):
    """动量策略，根据历史收益表现调整权重，表现好的资产权重增加。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        strategy_config = config.strategy
        self.lookback_periods = strategy_config.momentum_lookback if hasattr(strategy_config, 'momentum_lookback') and strategy_config.momentum_lookback is not None else 20
        self.momentum_factor = strategy_config.momentum_factor if hasattr(strategy_config, 'momentum_factor') and strategy_config.momentum_factor is not None else 0.5

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 简化版：基于基础权重和动量调整
        # 实际应用中需要历史收益数据，这里使用基础权重作为代理
        # 可以在这里添加基于历史收益的调整逻辑
        weights = self.base_weights.copy()
        return self._normalize(weights)


class MeanReversionStrategy(Strategy):
    """均值回归策略，当资产偏离均值时反向调整。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        strategy_config = config.strategy
        self.reversion_speed = strategy_config.reversion_speed if hasattr(strategy_config, 'reversion_speed') and strategy_config.reversion_speed is not None else 0.3

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 向目标权重回归
        deviation = self.base_weights - current_weights
        new_weights = current_weights + deviation * self.reversion_speed
        return self._normalize(new_weights)


def build_strategy(config: SimulationConfig) -> Strategy:
    """按名称构建策略实例。"""

    name = config.strategy.name
    if name == "fixed":
        return FixedAllocationStrategy(config)
    if name == "target_risk":
        return TargetRiskStrategy(config)
    if name == "adaptive":
        return AdaptiveRebalanceStrategy(config)
    if name == "equal_weight":
        return EqualWeightStrategy(config)
    if name == "risk_parity":
        return RiskParityStrategy(config)
    if name == "min_variance":
        return MinimumVarianceStrategy(config)
    if name == "momentum":
        return MomentumStrategy(config)
    if name == "mean_reversion":
        return MeanReversionStrategy(config)
    raise ValueError(f"不支持的策略类型：{name}")

