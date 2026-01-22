"""
随机状态管理模块
确保评测系统的可复现性
"""

import random
import numpy as np
from typing import Optional, Any
from loguru import logger


class RandomStateManager:
    """随机状态管理器"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.global_seed: Optional[int] = None
            self.random_states = {}
            self._initialized = True

    def set_global_seed(self, seed: int):
        """
        设置全局随机种子

        Args:
            seed: 随机种子
        """
        self.global_seed = seed

        # 设置Python内置random模块
        random.seed(seed)

        # 设置numpy随机种子
        try:
            np.random.seed(seed)
        except ImportError:
            logger.warning("numpy未安装，跳过numpy随机种子设置")

        # 设置torch随机种子（如果可用）
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        logger.info(f"全局随机种子已设置为：{seed}")

    def get_random_state(self, name: str) -> Any:
        """
        获取指定组件的随机状态

        Args:
            name: 组件名称

        Returns:
            随机状态对象
        """
        if name not in self.random_states:
            # 创建新的随机状态
            rng = random.Random(self.global_seed if self.global_seed is not None else 42)
            self.random_states[name] = rng
            logger.debug(f"为组件 {name} 创建随机状态")

        return self.random_states[name]

    def save_state(self, component_name: str) -> dict:
        """
        保存组件的随机状态

        Args:
            component_name: 组件名称

        Returns:
            状态字典
        """
        if component_name in self.random_states:
            rng = self.random_states[component_name]
            return {
                "state": rng.getstate(),
                "seed": self.global_seed
            }
        return {}

    def restore_state(self, component_name: str, state_dict: dict):
        """
        恢复组件的随机状态

        Args:
            component_name: 组件名称
            state_dict: 状态字典
        """
        if "state" in state_dict:
            rng = random.Random()
            rng.setstate(state_dict["state"])
            self.random_states[component_name] = rng
            logger.debug(f"组件 {component_name} 的随机状态已恢复")

    def reset_all(self):
        """重置所有随机状态"""
        self.random_states.clear()
        if self.global_seed is not None:
            self.set_global_seed(self.global_seed)
        logger.info("所有随机状态已重置")

    def create_deterministic_context(self, seed: Optional[int] = None):
        """
        创建确定性上下文管理器

        Args:
            seed: 随机种子，如果为None则使用全局种子

        Returns:
            上下文管理器
        """
        return DeterministicContext(seed or self.global_seed or 42)


class DeterministicContext:
    """确定性上下文管理器"""

    def __init__(self, seed: int):
        self.seed = seed
        self.original_states = {}

    def __enter__(self):
        """进入上下文，保存当前状态并设置确定性状态"""
        # 保存当前状态
        import random
        self.original_states["random"] = random.getstate()

        try:
            import numpy as np
            self.original_states["numpy"] = np.random.get_state()
        except ImportError:
            pass

        try:
            import torch
            self.original_states["torch"] = torch.get_rng_state()
        except ImportError:
            pass

        # 设置确定性状态
        random.seed(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass

        logger.debug(f"进入确定性上下文，种子：{self.seed}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原始状态"""
        import random
        random.setstate(self.original_states["random"])

        try:
            import numpy as np
            if "numpy" in self.original_states:
                np.random.set_state(self.original_states["numpy"])
        except ImportError:
            pass

        try:
            import torch
            if "torch" in self.original_states:
                torch.set_rng_state(self.original_states["torch"])
        except ImportError:
            pass

        logger.debug("退出确定性上下文，状态已恢复")


# 全局实例
random_state_manager = RandomStateManager()


def set_global_seed(seed: int):
    """设置全局随机种子（便捷函数）"""
    random_state_manager.set_global_seed(seed)


def get_deterministic_context(seed: Optional[int] = None):
    """获取确定性上下文（便捷函数）"""
    return random_state_manager.create_deterministic_context(seed)