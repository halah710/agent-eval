"""
玩家模拟器工厂
"""

from typing import Dict, Any, Type
from loguru import logger

from .base import PlayerSimulator
from .simple_player import SimplePlayer
from .role_player import RolePlayer
from .llm_player import LLMPlayer


class PlayerFactory:
    """玩家模拟器工厂类"""

    _player_classes = {
        "simple": SimplePlayer,
        "role": RolePlayer,
        "llm": LLMPlayer,
    }

    @classmethod
    def register_player(cls, player_type: str, player_class: Type[PlayerSimulator]):
        """注册新的玩家模拟器类型"""
        cls._player_classes[player_type] = player_class
        logger.info(f"注册玩家模拟器类型：{player_type} -> {player_class.__name__}")

    @classmethod
    def create_player(cls, config: Dict[str, Any]) -> PlayerSimulator:
        """
        创建玩家模拟器实例

        Args:
            config: 配置参数，包含type字段指定玩家模拟器类型

        Returns:
            玩家模拟器实例
        """
        player_type = config.get("type", "simple")
        player_config = config.get("config", {})

        if player_type not in cls._player_classes:
            raise ValueError(f"不支持的玩家模拟器类型：{player_type}")

        player_class = cls._player_classes[player_type]

        logger.info(f"创建玩家模拟器：{player_type}")
        return player_class(player_config)

    @classmethod
    def create_player_from_eval_config(cls, eval_config: Dict[str, Any]) -> PlayerSimulator:
        """
        从评测配置创建玩家模拟器

        Args:
            eval_config: 评测配置

        Returns:
            玩家模拟器实例
        """
        player_config = eval_config.get("player_simulator", {})
        return cls.create_player(player_config)