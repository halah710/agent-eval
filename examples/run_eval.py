#!/usr/bin/env python3
"""
AI Agent评测系统 - 运行示例
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.factory import AgentFactory
from src.evaluator.simple_evaluator import SimpleEvaluator
from src.evaluator.base import Scenario, TestSuite, EvaluationResult
from src.graders.factory import GraderFactory
from src.players.factory import PlayerFactory
from src.utils.random_state import set_global_seed
from src.reports import generate_both_reports


async def load_scenario(scenario_path: str) -> Scenario:
    """加载场景配置"""
    path = Path(scenario_path)
    if not path.exists():
        # 尝试在configs/scenarios目录下查找
        path = project_root / "configs" / "scenarios" / f"{scenario_path}.yaml"
        if not path.exists():
            path = project_root / "configs" / "scenarios" / f"{scenario_path}"

    if not path.exists():
        raise FileNotFoundError(f"场景文件不存在：{scenario_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Scenario(**data)


async def load_test_suite(suite_name: str, eval_config: Dict[str, Any]) -> TestSuite:
    """加载测试套件"""
    if suite_name not in eval_config.get("test_suites", {}):
        raise ValueError(f"测试套件不存在：{suite_name}")

    suite_config = eval_config["test_suites"][suite_name]
    suite = TestSuite(
        id=suite_name,
        name=suite_config["name"],
        description=suite_config.get("description", ""),
        suite_type=suite_config["suite_type"],
        scenarios=[]
    )

    # 加载所有场景
    for scenario_path in suite_config["scenarios"]:
        scenario = await load_scenario(scenario_path)
        suite.add_scenario(scenario)

    return suite


async def create_agent(eval_config: Dict[str, Any], npc_config: Dict[str, Any]) -> Any:
    """创建NPC Agent"""
    # 这里简化处理，使用mock agent作为示例
    agent_config = npc_config.get("agent_configs", {}).get("mock_agent", {})

    # 替换角色引用
    if isinstance(agent_config.get("role"), str) and agent_config["role"].startswith("{{npc_roles."):
        role_key = agent_config["role"][12:-2]  # 提取角色键名
        if role_key in npc_config.get("npc_roles", {}):
            agent_config["role"] = npc_config["npc_roles"][role_key]

    return AgentFactory.create_agent_from_config(agent_config)


async def create_graders(eval_config: Dict[str, Any]) -> List[Any]:
    """创建评分器列表"""
    graders = []
    for grader_name, grader_config in eval_config.get("graders", {}).items():
        # 只创建几个关键评分器作为示例
        if grader_name in ["character_consistency", "interaction_quality", "basic_rules"]:
            grader = GraderFactory.create_grader(grader_name, grader_config)
            graders.append(grader)

    return graders


async def run_scenario_evaluation(scenario_id: str, eval_config: Dict[str, Any], npc_config: Dict[str, Any]) -> EvaluationResult:
    """运行单个场景评测"""
    print(f"\n[开始] 开始评测场景：{scenario_id}")

    # 加载场景
    scenario = await load_scenario(scenario_id)
    print(f"📋 场景名称：{scenario.name}")
    print(f"📝 场景描述：{scenario.description}")

    # 创建Agent
    print("🤖 创建NPC Agent...")
    agent = await create_agent(eval_config, npc_config)

    # 创建评分器
    print("📊 创建评分器...")
    graders = await create_graders(eval_config)

    # 创建评测器
    evaluator = SimpleEvaluator(
        agent,
        graders,
        config=eval_config.get("evaluator", {}).get("config", {})
    )

    # 运行评测
    print("🔄 开始对话评测...")
    result = await evaluator.evaluate_scenario(scenario)

    # 输出结果
    print(f"\n✅ 评测完成！")
    print(f"🎯 最终得分：{result.final_score:.2%}")
    print(f"📈 是否通过：{'✅' if result.passed else '❌'}")
    print(f"⏱️  耗时：{(result.end_time - result.start_time):.1f}秒")

    # 输出各评分器结果
    print("\n📋 评分器详细结果：")
    for grader_name, grading_result in result.grading_results.items():
        print(f"  {grader_name}: {grading_result.score:.2%} ({'✅' if grading_result.passed else '❌'})")
        print(f"    理由：{grading_result.reasoning[:80]}...")

    # 保存结果
    output_dir = Path(eval_config.get("global", {}).get("output_dir", "./outputs"))
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / f"report_{scenario.id}.json"
    result.save_to_file(str(report_file))
    print(f"\n💾 评测报告已保存：{report_file}")

    # 保存对话记录
    if eval_config.get("global", {}).get("save_transcripts", True):
        transcript_file = output_dir / f"transcript_{scenario.id}.json"
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(result.transcript, f, ensure_ascii=False, indent=2)
        print(f"💾 对话记录已保存：{transcript_file}")

    # 使用新的报告模块生成详细报告
    print("📊 生成详细评测报告...")
    try:
        reports = generate_both_reports(
            results=[result],
            output_dir=str(output_dir),
            test_suite=None
        )
        print(f"📄 JSON详细报告：{reports['json_report']}")
        print(f"🌐 HTML可视化报告：{reports['html_report']}")
    except Exception as e:
        print(f"⚠️ 报告生成失败：{e}")
        import traceback
        traceback.print_exc()

    await evaluator.close()
    return result


async def run_suite_evaluation(suite_name: str, eval_config: Dict[str, Any], npc_config: Dict[str, Any]) -> List[EvaluationResult]:
    """运行测试套件评测"""
    print(f"\n🚀 开始评测套件：{suite_name}")

    # 加载测试套件
    suite = await load_test_suite(suite_name, eval_config)
    print(f"📋 套件名称：{suite.name}")
    print(f"📝 套件描述：{suite.description}")
    print(f"🔢 包含场景数：{len(suite.scenarios)}")

    # 创建Agent（套件内所有场景使用同一个Agent）
    print("🤖 创建NPC Agent...")
    agent = await create_agent(eval_config, npc_config)

    # 创建评分器
    print("📊 创建评分器...")
    graders = await create_graders(eval_config)

    # 创建评测器
    evaluator = SimpleEvaluator(
        agent,
        graders,
        config=eval_config.get("evaluator", {}).get("config", {})
    )

    # 运行所有场景
    results = []
    for i, scenario in enumerate(suite.scenarios, 1):
        print(f"\n🔹 场景 {i}/{len(suite.scenarios)}: {scenario.name}")
        result = await evaluator.evaluate_scenario(scenario)
        results.append(result)

        print(f"   得分：{result.final_score:.2%} ({'✅' if result.passed else '❌'})")

    # 计算套件统计
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.final_score for r in results) / len(results) if results else 0

    print(f"\n📊 套件统计：")
    print(f"  ✅ 通过场景：{passed_count}/{len(suite.scenarios)}")
    print(f"  📈 平均得分：{avg_score:.2%}")
    print(f"  🎯 套件类型：{suite.suite_type}")

    # 保存套件报告
    output_dir = Path(eval_config.get("global", {}).get("output_dir", "./outputs"))
    output_dir.mkdir(exist_ok=True)

    suite_report = {
        "suite_id": suite.id,
        "suite_name": suite.name,
        "suite_type": suite.suite_type,
        "total_scenarios": len(suite.scenarios),
        "passed_scenarios": passed_count,
        "average_score": avg_score,
        "results": [r.to_dict() for r in results]
    }

    report_file = output_dir / f"suite_report_{suite.id}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(suite_report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 套件报告已保存：{report_file}")

    # 使用新的报告模块生成详细报告
    print("📊 生成详细评测报告...")
    try:
        reports = generate_both_reports(
            results=results,
            output_dir=str(output_dir),
            test_suite=suite
        )
        print(f"📄 JSON详细报告：{reports['json_report']}")
        print(f"🌐 HTML可视化报告：{reports['html_report']}")
    except Exception as e:
        print(f"⚠️ 报告生成失败：{e}")
        import traceback
        traceback.print_exc()

    await evaluator.close()
    return results


def load_configs() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """加载配置文件"""
    # 加载NPC配置
    npc_config_path = project_root / "configs" / "npc_config.yaml"
    with open(npc_config_path, "r", encoding="utf-8") as f:
        npc_config = yaml.safe_load(f)

    # 加载评测配置
    eval_config_path = project_root / "configs" / "eval_config.yaml"
    with open(eval_config_path, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    return eval_config, npc_config


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Agent评测系统")
    parser.add_argument("--scenario", help="评测单个场景（场景ID或路径）")
    parser.add_argument("--suite", choices=["capability_suite", "regression_suite"],
                       help="评测整个测试套件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有可用场景")

    args = parser.parse_args()

    # 设置随机种子
    set_global_seed(args.seed)
    print(f"🎲 随机种子：{args.seed}")

    # 加载配置
    eval_config, npc_config = load_configs()

    if args.list_scenarios:
        # 列出所有场景
        scenarios_dir = project_root / "configs" / "scenarios"
        print("\n📁 可用场景：")
        for yaml_file in scenarios_dir.glob("*.yaml"):
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                print(f"  • {data['id']}: {data['name']} ({yaml_file.name})")
        return

    if args.scenario:
        # 运行单个场景评测
        await run_scenario_evaluation(args.scenario, eval_config, npc_config)
    elif args.suite:
        # 运行测试套件评测
        await run_suite_evaluation(args.suite, eval_config, npc_config)
    else:
        # 默认运行示例场景
        print("🤖 AI Agent评测系统")
        print("=" * 50)
        print("未指定场景或套件，运行示例场景...")
        await run_scenario_evaluation("daily_chat", eval_config, npc_config)


if __name__ == "__main__":
    asyncio.run(main())