"""
基于Anthropic API的NPC Agent实现
"""

import os
from typing import Dict, Any
import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import NPCAgent, NPCRole, DialogueContext


class AnthropicAgent(NPCAgent):
    """基于Anthropic API的NPC Agent"""

    def __init__(
        self,
        role: NPCRole,
        config: Dict[str, Any] = None
    ):
        """
        初始化Anthropic Agent

        Args:
            role: NPC角色定义
            config: 配置参数，包括：
                - api_key: Anthropic API密钥（可选，默认从环境变量读取）
                - model: 模型名称（默认：claude-3-5-sonnet-20241022）
                - temperature: 温度参数（默认：0.0）
                - max_tokens: 最大生成token数（默认：500）
        """
        super().__init__(role, config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = config.get("model") or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 500))

        if not self.api_key:
            raise ValueError("Anthropic API密钥未提供，请设置ANTHROPIC_API_KEY环境变量或在config中提供api_key")

        self.client = None

    async def initialize(self):
        """初始化Anthropic客户端"""
        try:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self._initialized = True
            logger.info(f"Anthropic Agent初始化成功，模型：{self.model}")
        except Exception as e:
            logger.error(f"Anthropic Agent初始化失败：{e}")
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

        # 构建系统Prompt
        system_prompt = self.get_system_prompt()

        # 构建对话历史（Anthropic使用不同的消息格式）
        messages = []

        # 添加对话历史
        history = context.get_conversation_history(max_turns=10)
        for msg in history:
            # Anthropic使用 'user' 和 'assistant' 角色
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({
                "role": role,
                "content": msg["content"]
            })

        # 添加当前玩家输入
        messages.append({"role": "user", "content": player_input})

        try:
            response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            reply = response.content[0].text.strip()
            logger.debug(f"Anthropic Agent回复：{reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Anthropic API调用失败：{e}")
            # 返回降级回复
            return "抱歉，我现在无法处理你的请求。请稍后再试。"

    async def close(self):
        """清理资源"""
        if self.client:
            # Anthropic客户端不需要显式关闭
            pass
        self._initialized = False
        logger.info("Anthropic Agent已关闭")

    def __str__(self):
        return f"AnthropicAgent(model={self.model}, role={self.role.name})"