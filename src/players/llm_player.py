"""
基于LLM的玩家模拟器
使用LLM生成更自然和智能的玩家回复
"""

import os
from typing import List, Dict, Any
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import PlayerSimulator, PlayerProfile, DialogueContext, DialogueAction, DialogueActionType


class LLMPlayer(PlayerSimulator):
    """基于LLM的玩家模拟器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化LLM玩家模拟器

        Args:
            config: 配置参数，包括：
                - api_key: OpenAI API密钥
                - model: 模型名称
                - temperature: 温度参数
                - max_tokens: 最大生成token数
                - system_prompt_template: 系统Prompt模板
        """
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 200))
        self.system_prompt_template = config.get("system_prompt_template", """
你是一个游戏玩家，正在与NPC对话。请根据你的角色设定和当前对话上下文，生成自然、真实的玩家回复。

角色设定：
名称：{name}
性格：{personality}
年龄：{age}
背景：{background}
兴趣：{interests}
说话风格：{speaking_style}
当前情感状态：{emotional_state}

请保持角色一致性，按照上述设定进行对话。你的回复应该自然、真实，符合游戏玩家的特点。
""")
        self.client = None

        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供")

    async def initialize(self):
        """初始化LLM客户端"""
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"LLM玩家模拟器初始化成功，模型：{self.model}")
        except Exception as e:
            logger.error(f"LLM玩家模拟器初始化失败：{e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
        """
        生成LLM玩家回复

        Args:
            npc_reply: NPC的回复
            context: 对话上下文

        Returns:
            LLM玩家回复
        """
        if not self._initialized:
            await self.initialize()

        if not self.profile:
            logger.warning("LLM玩家模拟器未设置角色，使用默认角色")
            self.profile = PlayerProfile(
                name="默认玩家",
                personality="friendly",
                age=25,
                background="普通游戏玩家",
                interests=["游戏", "聊天"],
                speaking_style="友好、自然",
                knowledge_level=5,
                emotional_state="neutral"
            )

        # 构建系统Prompt
        system_prompt = self._build_system_prompt()

        # 构建对话历史
        messages = self._build_messages(npc_reply, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.random_seed  # 固定种子确保可复现性
            )

            player_reply = response.choices[0].message.content.strip()
            logger.debug(f"LLM玩家回复：{player_reply[:100]}...")

            # 记录到上下文
            context.turns.append({
                "speaker": "player",
                "message": player_reply,
                "llm_generated": True
            })

            return player_reply

        except Exception as e:
            logger.error(f"LLM生成玩家回复失败：{e}")
            # 返回降级回复
            return "我明白了。你能再说详细一点吗？"

    def _build_system_prompt(self) -> str:
        """构建系统Prompt"""
        if not self.profile:
            return "你是一个游戏玩家，正在与NPC对话。请生成自然、真实的玩家回复。"

        return self.system_prompt_template.format(
            name=self.profile.name,
            personality=self.profile.personality,
            age=self.profile.age,
            background=self.profile.background,
            interests=", ".join(self.profile.interests),
            speaking_style=self.profile.speaking_style,
            emotional_state=self.profile.emotional_state
        )

    def _build_messages(self, npc_reply: str, context: DialogueContext) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # 添加对话历史
        for turn in context.turns[-6:]:  # 最近6轮对话
            role = "user" if turn.get("speaker") == "player" else "assistant"
            messages.append({
                "role": role,
                "content": turn.get("message", "")
            })

        # 添加当前NPC回复
        messages.append({
            "role": "assistant",
            "content": npc_reply
        })

        return messages

    async def decide_next_action(self, context: DialogueContext) -> DialogueAction:
        """
        决定下一步对话动作（LLM玩家使用LLM决定动作）

        Args:
            context: 对话上下文

        Returns:
            对话动作
        """
        # LLM玩家直接在generate_response中决定回复内容
        # 这里返回一个默认动作
        return DialogueAction(
            action_type=DialogueActionType.CONTINUE,
            content="继续对话",
            confidence=0.8
        )

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info("LLM玩家模拟器已关闭")

    def __str__(self):
        if self.profile:
            return f"LLMPlayer(name={self.profile.name}, model={self.model})"
        return f"LLMPlayer(model={self.model})"