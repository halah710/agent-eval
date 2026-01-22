"""
Pytest configuration and fixtures for agent-eval tests
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import asyncio

# Async fixtures support
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Common test fixtures
@pytest.fixture
def mock_npc_config():
    """Fixture providing a mock NPC configuration."""
    return {
        "type": "mock",
        "role": {
            "name": "测试NPC",
            "personality": "友好",
            "background": "测试背景",
            "speaking_style": "测试风格",
            "values": ["测试值1", "测试值2"]
        },
        "config": {
            "response_type": "fixed",
            "fixed_responses": ["你好！", "今天天气不错。", "很高兴和你聊天。"]
        }
    }

@pytest.fixture
def simple_scenario():
    """Fixture providing a simple test scenario."""
    from src.evaluator.base import Scenario

    return Scenario(
        id="test_scenario",
        name="测试场景",
        description="测试描述",
        scenario_type="daily_chat",
        player_profile={
            "name": "测试玩家",
            "personality": "friendly",
            "age": 25,
            "background": "测试玩家背景",
            "interests": ["测试兴趣"],
            "speaking_style": "测试说话风格",
            "knowledge_level": 5,
            "emotional_state": "neutral"
        },
        initial_prompt="你好！",
        max_turns=3
    )

@pytest.fixture
def simple_player_config():
    """Fixture providing a simple player configuration."""
    return {
        "type": "simple",
        "config": {
            "response_templates": ["测试玩家回复1", "测试玩家回复2"]
        }
    }

@pytest.fixture
def simple_grader_config():
    """Fixture providing a simple grader configuration."""
    return {
        "type": "natural_language",
        "config": {
            "assertions": ["NPC保持友好态度", "对话自然流畅"],
            "api_key": "test_key",
            "conversion": {
                "strategy": "simple_ratio",
                "pass_threshold": 0.7
            }
        }
    }