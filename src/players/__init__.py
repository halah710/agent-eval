"""
AI Agent评测系统 - 玩家模拟器模块
提供多样化的玩家模拟器，用于生成真实的玩家对话。
"""

from .base import PlayerSimulator, PlayerProfile, DialogueAction
from .simple_player import SimplePlayer
from .role_player import RolePlayer
from .llm_player import LLMPlayer
from .factory import PlayerFactory

__all__ = [
    "PlayerSimulator",
    "PlayerProfile",
    "DialogueAction",
    "SimplePlayer",
    "RolePlayer",
    "LLMPlayer",
    "PlayerFactory",
]