"""
Agent工厂，用于创建不同类型的NPC Agent
"""

from typing import Dict, Any
from loguru import logger

from .base import NPCRole
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .mock_agent import MockAgent


class AgentFactory:
    """Agent工厂类"""

    @staticmethod
    def create_agent(
        agent_type: str,
        role_config: Dict[str, Any],
        agent_config: Dict[str, Any] = None
    ):
        """
        创建NPC Agent实例

        Args:
            agent_type: Agent类型，可选：openai, anthropic, mock
            role_config: 角色配置
            agent_config: Agent特定配置

        Returns:
            NPCAgent实例
        """
        agent_config = agent_config or {}

        # 创建角色对象
        role = NPCRole(**role_config)

        # 根据类型创建Agent
        if agent_type == "openai":
            return OpenAIAgent(role, agent_config)
        elif agent_type == "anthropic":
            return AnthropicAgent(role, agent_config)
        elif agent_type == "mock":
            return MockAgent(role, agent_config)
        else:
            raise ValueError(f"不支持的Agent类型：{agent_type}")

    @staticmethod
    def create_agent_from_config(config: Dict[str, Any]):
        """
        从配置字典创建Agent

        Args:
            config: 配置字典，包含：
                - type: Agent类型
                - role: 角色配置
                - config: Agent特定配置

        Returns:
            NPCAgent实例
        """
        agent_type = config.get("type", "mock")
        role_config = config.get("role", {})
        agent_config = config.get("config", {})

        logger.info(f"创建Agent：类型={agent_type}, 角色={role_config.get('name', 'unknown')}")

        return AgentFactory.create_agent(agent_type, role_config, agent_config)