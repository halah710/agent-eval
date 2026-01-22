"""
模拟NPC Agent，用于测试和开发
"""

import random
from typing import Dict, Any, List
from loguru import logger

from .base import NPCAgent, NPCRole, DialogueContext


class MockAgent(NPCAgent):
    """模拟NPC Agent，用于测试和开发"""

    def __init__(
        self,
        role: NPCRole,
        config: Dict[str, Any] = None
    ):
        """
        初始化Mock Agent

        Args:
            role: NPC角色定义
            config: 配置参数，包括：
                - response_type: 回复类型，可选：fixed, random, echo
                - fixed_responses: 固定回复列表（当response_type为fixed时使用）
                - response_delay: 回复延迟（秒）
        """
        super().__init__(role, config)
        self.response_type = config.get("response_type", "fixed")
        self.fixed_responses = config.get("fixed_responses", [
            "我明白了。",
            "这很有趣。",
            "请继续。",
            "我不太确定。",
            "你能详细说明一下吗？"
        ])
        self.response_delay = float(config.get("response_delay", 0.0))
        self.response_count = 0

        # 根据角色生成一些回复模板
        self._generate_responses_from_role()

    def _generate_responses_from_role(self):
        """根据角色生成回复模板"""
        role_based_responses = [
            f"作为{self.role.name}，我认为...",
            f"根据我的背景：{self.role.background[:50]}...",
            f"以我的性格{self.role.personality[:30]}...",
            "这让我想起了我的经历。",
            "从我的价值观来看...",
        ]
        self.fixed_responses.extend(role_based_responses)

    async def initialize(self):
        """Mock Agent不需要特殊初始化"""
        self._initialized = True
        logger.info(f"Mock Agent初始化成功，角色：{self.role.name}")

    async def respond(self, player_input: str, context: DialogueContext) -> str:
        """
        生成模拟回复

        Args:
            player_input: 玩家输入
            context: 对话上下文

        Returns:
            模拟回复内容
        """
        if not self._initialized:
            await self.initialize()

        # 模拟延迟
        if self.response_delay > 0:
            import asyncio
            await asyncio.sleep(self.response_delay)

        # 根据回复类型生成回复
        if self.response_type == "fixed":
            # 循环使用固定回复
            response = self.fixed_responses[self.response_count % len(self.fixed_responses)]
            self.response_count += 1
        elif self.response_type == "random":
            # 随机选择回复
            response = random.choice(self.fixed_responses)
        elif self.response_type == "echo":
            # 回声回复
            response = f"你说：{player_input}"
        else:
            # 默认回复
            response = f"作为{self.role.name}，我听到你说：{player_input[:50]}..."

        logger.debug(f"Mock Agent回复：{response}")
        return response

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info("Mock Agent已关闭")

    def __str__(self):
        return f"MockAgent(role={self.role.name}, type={self.response_type})"