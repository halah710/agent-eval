"""
AI Agent评测系统 - NPC Agent模块
提供不同类型的NPC Agent实现，用于对话评测。
"""

from .base import NPCAgent, DialogueContext, NPCRole
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .mock_agent import MockAgent
from .factory import AgentFactory

__all__ = [
    "NPCAgent",
    "DialogueContext",
    "NPCRole",
    "OpenAIAgent",
    "AnthropicAgent",
    "MockAgent",
    "AgentFactory",
]