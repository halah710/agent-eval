#!/usr/bin/env python3
"""
端到端测试 - 使用Mock Agent
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def main():
    print("AI Agent评测系统 - 端到端测试")
    print("=" * 50)

    # 导入必要模块
    from src.agents.factory import AgentFactory
    from src.evaluator.simple_evaluator import SimpleEvaluator
    from src.evaluator.base import Scenario
    from src.graders.factory import GraderFactory
    from src.players.factory import PlayerFactory
    from src.utils.random_state import set_global_seed

    # 设置随机种子
    set_global_seed(42)
    print("[设置] 随机种子: 42")

    # 创建Mock Agent
    print("[创建] Mock Agent...")
    agent = AgentFactory.create_agent_from_config({
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
            "fixed_responses": [
                "你好！",
                "今天天气不错。",
                "很高兴和你聊天。",
                "谢谢你的分享。",
                "再见！"
            ]
        }
    })

    # 创建简单评分器
    print("[创建] 评分器...")
    grader = GraderFactory.create_grader("test_grader", {
        "type": "natural_language",
        "config": {
            "assertions": ["NPC保持友好态度", "对话自然流畅"],
            "api_key": "test_key"
        }
    })

    # 创建玩家模拟器
    print("[创建] 玩家模拟器...")
    player_simulator = PlayerFactory.create_player({
        "type": "simple",
        "config": {
            "response_templates": ["测试玩家回复1", "测试玩家回复2", "继续说吧"]
        }
    })
    await player_simulator.initialize()

    # 创建评测器
    print("[创建] 评测器...")
    evaluator = SimpleEvaluator(
        agent,
        [grader],
        config={
            "max_turns": 3,
            "player_simulator": player_simulator
        }
    )

    # 创建测试场景
    scenario = Scenario(
        id="e2e_test",
        name="端到端测试场景",
        description="测试完整评测流程",
        scenario_type="daily_chat",
        player_profile={
            "name": "测试玩家",
            "personality": "friendly",
            "age": 25,
            "background": "测试背景",
            "interests": ["测试兴趣"],
            "speaking_style": "测试风格",
            "knowledge_level": 5,
            "emotional_state": "neutral"
        },
        initial_prompt="你好！",
        max_turns=3
    )

    # 运行评测
    print("\n[评测] 开始评测...")
    result = await evaluator.evaluate_scenario(scenario)

    # 输出结果
    print(f"\n[结果] 评测完成！")
    print(f"  场景: {result.scenario_name}")
    print(f"  最终得分: {result.final_score:.2%}")
    print(f"  是否通过: {'[通过]' if result.passed else '[失败]'}")
    print(f"  对话轮次: {len(result.transcript)}")
    print(f"  评分器数量: {len(result.grading_results)}")

    for grader_name, grading_result in result.grading_results.items():
        print(f"    - {grader_name}: {grading_result.score:.2%} ({'通过' if grading_result.passed else '失败'})")
        print(f"      理由: {grading_result.reasoning[:60]}...")

    # 清理资源
    await evaluator.close()
    await player_simulator.close()

    print("\n[完成] 端到端测试成功！")

if __name__ == "__main__":
    asyncio.run(main())