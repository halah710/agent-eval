#!/usr/bin/env python3
"""
基础测试脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_basic_components():
    """测试基础组件"""
    print("[测试] 测试基础组件...")

    # 测试数据模型
    from src.evaluator.base import Scenario, TestSuite
    from src.agents.base import NPCRole
    from src.graders.base import Grader, GradingResult

    # 创建测试数据
    role = NPCRole(
        name="测试NPC",
        personality="友好",
        background="测试背景",
        speaking_style="测试风格",
        values=["测试值1", "测试值2"]
    )

    scenario = Scenario(
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

    test_suite = TestSuite(
        id="test_suite",
        name="测试套件",
        description="测试套件描述",
        suite_type="capability",
        scenarios=[scenario]
    )

    print("[通过] 数据模型测试通过")
    print(f"  - 角色: {role.name}")
    print(f"  - 场景: {scenario.name}")
    print(f"  - 套件: {test_suite.name}")

    # 测试Agent工厂
    from src.agents.factory import AgentFactory

    agent = AgentFactory.create_agent_from_config({
        "type": "mock",
        "role": {
            "name": "工厂测试NPC",
            "personality": "友好",
            "background": "工厂测试背景",
            "speaking_style": "工厂测试风格",
            "values": ["工厂值1"]
        },
        "config": {
            "response_type": "fixed",
            "fixed_responses": ["工厂测试回复1", "工厂测试回复2"]
        }
    })

    print("[通过] Agent工厂测试通过")
    print(f"  - Agent类型: {type(agent).__name__}")

    # 测试评分器工厂
    from src.graders.factory import GraderFactory

    grader = GraderFactory.create_grader("test_grader", {
        "type": "natural_language",
        "config": {
            "assertions": ["测试断言1", "测试断言2"],
            "api_key": "test_key"
        }
    })

    print("[通过] 评分器工厂测试通过")
    print(f"  - 评分器类型: {type(grader).__name__}")

    # 测试玩家模拟器工厂
    from src.players.factory import PlayerFactory

    player = PlayerFactory.create_player({
        "type": "simple",
        "config": {
            "response_templates": ["测试玩家回复"]
        }
    })

    print("[通过] 玩家模拟器工厂测试通过")
    print(f"  - 玩家模拟器类型: {type(player).__name__}")

    print("\n[完成] 所有基础组件测试通过！")


async def test_evaluation_flow():
    """测试评测流程"""
    print("\n[测试] 测试评测流程...")

    from src.agents.factory import AgentFactory
    from src.evaluator.simple_evaluator import SimpleEvaluator
    from src.evaluator.base import Scenario
    from src.graders.factory import GraderFactory
    from src.graders.base import Grader, GradingResult
    from src.players.factory import PlayerFactory

    # 创建Mock Agent
    agent = AgentFactory.create_agent_from_config({
        "type": "mock",
        "role": {
            "name": "流程测试NPC",
            "personality": "友好",
            "background": "流程测试背景",
            "speaking_style": "流程测试风格",
            "values": ["流程值1"]
        },
        "config": {
            "response_type": "fixed",
            "fixed_responses": ["你好！", "今天天气不错。", "很高兴和你聊天。"]
        }
    })

    # 创建Mock评分器
    class MockGrader(Grader):
        def __init__(self, name):
            super().__init__(name, {"type": "mock"})

        async def initialize(self):
            self._initialized = True

        async def grade(self, transcript):
            return GradingResult(
                grader_name=self.name,
                score=0.8,
                passed=True,
                reasoning="测试评分",
                evidence=[]
            )

        async def close(self):
            pass

    # 使用PlayerFactory创建玩家模拟器
    player_simulator = PlayerFactory.create_player({
        "type": "simple",
        "config": {
            "response_templates": ["测试玩家回复1", "测试玩家回复2"]
        }
    })
    await player_simulator.initialize()

    # 创建评测器
    evaluator = SimpleEvaluator(
        agent,
        [MockGrader("test_grader")],
        config={
            "max_turns": 2,
            "player_simulator": player_simulator
        }
    )

    # 创建测试场景
    scenario = Scenario(
        id="flow_test",
        name="流程测试场景",
        description="测试评测流程",
        scenario_type="daily_chat",
        player_profile={
            "name": "流程测试玩家",
            "personality": "friendly",
            "age": 25,
            "background": "测试",
            "interests": ["测试"],
            "speaking_style": "测试",
            "knowledge_level": 5,
            "emotional_state": "neutral"
        },
        initial_prompt="你好！",
        max_turns=2
    )

    # 运行评测
    result = await evaluator.evaluate_scenario(scenario)

    print("[通过] 评测流程测试通过")
    print(f"  - 最终得分: {result.final_score:.2%}")
    print(f"  - 是否通过: {'[通过]' if result.passed else '[失败]'}")
    print(f"  - 对话轮次: {len(result.transcript)}")

    await evaluator.close()
    await player_simulator.close()


async def main():
    """主测试函数"""
    print("[AI] AI Agent评测系统 - 基础测试")
    print("=" * 50)

    try:
        await test_basic_components()
        await test_evaluation_flow()
        print("\n[完成] 所有测试完成！系统基本功能正常。")
    except Exception as e:
        print(f"\n[失败] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())