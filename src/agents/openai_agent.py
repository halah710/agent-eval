"""
基于OpenAI API的NPC Agent实现
"""

import os
from typing import Dict, Any, Optional
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import NPCAgent, NPCRole, DialogueContext


class OpenAIAgent(NPCAgent):
    """基于OpenAI API的NPC Agent"""

    def __init__(
        self,
        role: NPCRole,
        config: Dict[str, Any] = None
    ):
        """
        初始化OpenAI Agent

        Args:
            role: NPC角色定义
            config: 配置参数，包括：
                - api_key: OpenAI API密钥（可选，默认从环境变量读取）
                - model: 模型名称（默认：gpt-4o-mini）
                - temperature: 温度参数（默认：0.0，确保确定性）
                - max_tokens: 最大生成token数（默认：500）
                - base_url: API基础URL（可选）
        """
        super().__init__(role, config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 500))
        self.base_url = config.get("base_url") or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请设置OPENAI_API_KEY环境变量或在config中提供api_key")

        self.client = None

    async def initialize(self):
        """初始化OpenAI客户端"""
        try:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._initialized = True
            logger.info(f"OpenAI Agent初始化成功，模型：{self.model}")
        except Exception as e:
            logger.error(f"OpenAI Agent初始化失败：{e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def respond(self, player_input: str, context: DialogueContext) -> str:
        """
        生成NPC回复

        Args:
            player_input: 玩家输入
            context: 对话上下文

        Returns:
            NPC回复内容
        """
        if not self._initialized:
            await self.initialize()

        # 构建对话历史
        messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]

        # 添加对话历史
        history = context.get_conversation_history(max_turns=10)
        messages.extend(history)

        # 添加当前玩家输入
        messages.append({"role": "user", "content": player_input})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=42  # 固定种子确保可复现性
            )

            reply = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI Agent回复：{reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"OpenAI API调用失败：{e}")
            # 返回降级回复
            return "抱歉，我现在无法处理你的请求。请稍后再试。"

    async def close(self):
        """清理资源"""
        if self.client:
            # OpenAI客户端不需要显式关闭
            pass
        self._initialized = False
        logger.info("OpenAI Agent已关闭")

    def __str__(self):
        return f"OpenAIAgent(model={self.model}, role={self.role.name})"