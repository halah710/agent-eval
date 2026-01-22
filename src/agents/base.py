"""
NPC Agent基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class NPCRole(BaseModel):
    """NPC角色定义"""
    name: str = Field(..., description="角色名称")
    personality: str = Field(..., description="性格描述")
    background: str = Field(..., description="背景故事")
    speaking_style: str = Field(..., description="说话风格")
    values: List[str] = Field(default_factory=list, description="价值观列表")


class DialogueTurn(BaseModel):
    """对话轮次"""
    speaker: str = Field(..., description="说话者：player 或 npc")
    message: str = Field(..., description="对话内容")
    timestamp: float = Field(default_factory=lambda: time.time(), description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DialogueContext(BaseModel):
    """对话上下文"""
    turns: List[DialogueTurn] = Field(default_factory=list, description="对话历史")
    current_topic: Optional[str] = Field(None, description="当前话题")
    player_state: Dict[str, Any] = Field(default_factory=dict, description="玩家状态")
    npc_state: Dict[str, Any] = Field(default_factory=dict, description="NPC状态")

    def add_turn(self, speaker: str, message: str, metadata: Dict[str, Any] = None):
        """添加对话轮次"""
        turn = DialogueTurn(
            speaker=speaker,
            message=message,
            metadata=metadata or {}
        )
        self.turns.append(turn)

    def get_conversation_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """获取对话历史（用于LLM输入）"""
        history = []
        for turn in self.turns[-max_turns:]:
            history.append({
                "role": "user" if turn.speaker == "player" else "assistant",
                "content": turn.message
            })
        return history


class NPCAgent(ABC):
    """NPC Agent基类"""

    def __init__(self, role: NPCRole, config: Dict[str, Any] = None):
        """
        初始化NPC Agent

        Args:
            role: NPC角色定义
            config: 配置参数
        """
        self.role = role
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self):
        """初始化Agent（加载模型、连接API等）"""
        pass

    @abstractmethod
    async def respond(self, player_input: str, context: DialogueContext) -> str:
        """
        生成NPC回复

        Args:
            player_input: 玩家输入
            context: 对话上下文

        Returns:
            NPC回复内容
        """
        pass

    @abstractmethod
    async def close(self):
        """清理资源"""
        pass

    def get_system_prompt(self) -> str:
        """生成系统Prompt"""
        return f"""你是一个游戏NPC，扮演以下角色：

角色名称：{self.role.name}
性格：{self.role.personality}
背景：{self.role.background}
说话风格：{self.role.speaking_style}
价值观：{', '.join(self.role.values)}

请始终保持角色一致性，按照上述设定进行对话。"""

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 导入time模块
import time