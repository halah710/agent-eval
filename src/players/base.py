"""
玩家模拟器基类和核心数据结构
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import random


class PlayerPersonality(str, Enum):
    """玩家性格类型"""
    FRIENDLY = "friendly"  # 友好
    AGGRESSIVE = "aggressive"  # 攻击性
    SHY = "shy"  # 害羞
    CURIOUS = "curious"  # 好奇
    SARCASTIC = "sarcastic"  # 讽刺
    SUPPORTIVE = "supportive"  # 支持性


class DialogueActionType(str, Enum):
    """对话动作类型"""
    CONTINUE = "continue"  # 继续当前话题
    CHANGE_TOPIC = "change_topic"  # 切换话题
    ASK_QUESTION = "ask_question"  # 提问
    EXPRESS_EMOTION = "express_emotion"  # 表达情感
    END_CONVERSATION = "end_conversation"  # 结束对话


class PlayerProfile(BaseModel):
    """玩家角色配置"""
    name: str = Field(..., description="玩家名称")
    personality: PlayerPersonality = Field(..., description="性格类型")
    age: int = Field(..., description="年龄")
    background: str = Field(..., description="背景故事")
    interests: List[str] = Field(default_factory=list, description="兴趣列表")
    speaking_style: str = Field(..., description="说话风格")
    knowledge_level: int = Field(default=5, ge=1, le=10, description="知识水平（1-10）")
    emotional_state: str = Field(default="neutral", description="情感状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def get_personality_traits(self) -> List[str]:
        """获取性格特征"""
        traits = []
        if self.personality == PlayerPersonality.FRIENDLY:
            traits.extend(["友善", "热情", "乐于助人"])
        elif self.personality == PlayerPersonality.AGGRESSIVE:
            traits.extend(["直接", "强势", "挑战性"])
        elif self.personality == PlayerPersonality.SHY:
            traits.extend(["谨慎", "内向", "保守"])
        elif self.personality == PlayerPersonality.CURIOUS:
            traits.extend(["好奇", "探索", "提问"])
        elif self.personality == PlayerPersonality.SARCASTIC:
            traits.extend(["讽刺", "幽默", "挖苦"])
        elif self.personality == PlayerPersonality.SUPPORTIVE:
            traits.extend(["支持", "鼓励", "同理心"])
        return traits


class DialogueAction(BaseModel):
    """对话动作"""
    action_type: DialogueActionType = Field(..., description="动作类型")
    content: str = Field(..., description="动作内容")
    target_topic: Optional[str] = Field(None, description="目标话题（如果切换话题）")
    emotion: Optional[str] = Field(None, description="情感表达")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")


class DialogueContext(BaseModel):
    """对话上下文（简化版，与agents中的区分）"""
    turns: List[Dict[str, Any]] = Field(default_factory=list, description="对话历史")
    current_topic: Optional[str] = Field(None, description="当前话题")
    player_state: Dict[str, Any] = Field(default_factory=dict, description="玩家状态")
    conversation_goals: List[str] = Field(default_factory=list, description="对话目标")


class PlayerSimulator(ABC):
    """玩家模拟器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化玩家模拟器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.profile: Optional[PlayerProfile] = None
        self._initialized = False
        self.random_seed = self.config.get("random_seed", 42)

        # 设置随机种子确保可复现性
        random.seed(self.random_seed)

    @abstractmethod
    async def initialize(self):
        """初始化模拟器"""
        pass

    def set_profile(self, profile_config: Dict[str, Any]):
        """设置玩家角色配置"""
        self.profile = PlayerProfile(**profile_config)

    @abstractmethod
    async def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
        """
        生成玩家回复

        Args:
            npc_reply: NPC的回复
            context: 对话上下文

        Returns:
            玩家回复内容
        """
        pass

    @abstractmethod
    async def decide_next_action(self, context: DialogueContext) -> DialogueAction:
        """
        决定下一步对话动作

        Args:
            context: 对话上下文

        Returns:
            对话动作
        """
        pass

    @abstractmethod
    async def close(self):
        """清理资源"""
        pass

    def _get_personality_based_response(self, base_response: str) -> str:
        """根据性格调整回复"""
        if not self.profile:
            return base_response

        personality = self.profile.personality
        if personality == PlayerPersonality.FRIENDLY:
            return f"{base_response} 很高兴和你聊天！"
        elif personality == PlayerPersonality.AGGRESSIVE:
            return f"{base_response} 不过我觉得你说得不太对。"
        elif personality == PlayerPersonality.SHY:
            return f"嗯... {base_response.lower()}"
        elif personality == PlayerPersonality.CURIOUS:
            return f"{base_response} 你能告诉我更多吗？"
        elif personality == PlayerPersonality.SARCASTIC:
            return f"{base_response} 当然，如果你说的是真的话。"
        elif personality == PlayerPersonality.SUPPORTIVE:
            return f"{base_response} 我理解你的感受。"

        return base_response

    def __str__(self):
        profile_name = self.profile.name if self.profile else "未设置"
        return f"PlayerSimulator(profile={profile_name})"