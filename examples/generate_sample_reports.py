#!/usr/bin/env python3
"""
生成示例评测报告
创建包含5个不同场景的完整评测报告示例
"""
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import random
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluator.base import Scenario, TestSuite, EvaluationResult, ScenarioType
from src.graders.base import GradingResult, CriterionResult
from src.reports import generate_both_reports


def create_mock_grading_result(grader_name: str, scenario_id: str) -> GradingResult:
    """创建模拟评分结果"""
    # 为不同场景设置不同的分数，展示多样性
    scenario_scores = {
        "daily_chat_001": 0.85,
        "emotional_support_001": 0.72,
        "opinion_conflict_001": 0.65,
        "basic_greeting_001": 0.95,
        "simple_qa_001": 0.88
    }

    scenario_reasoning = {
        "daily_chat_001": "NPC在日常聊天中表现良好，友好亲切，对话自然流畅。",
        "emotional_support_001": "NPC在情感支持方面有一定表现，但共情深度有待提升。",
        "opinion_conflict_001": "NPC处理观点冲突时保持礼貌，但未能充分引导对话走向建设性讨论。",
        "basic_greeting_001": "NPC在基础问候场景中表现优秀，回复快速准确。",
        "simple_qa_001": "NPC回答简单问题时准确清晰，但缺乏深度扩展。"
    }

    base_score = scenario_scores.get(scenario_id, 0.75)
    reasoning = scenario_reasoning.get(scenario_id, "评测完成。")

    # 根据评分器名称定义不同的断言和权重（与config/eval_config.yaml保持一致）
    grader_assertions = {
        "character_consistency": {
            "assertions": [
                "NPC的回复符合其角色设定",
                "NPC的说话风格保持一致",
                "NPC的价值观在对话中得到体现",
                "NPC没有出现角色矛盾的行为或言论"
            ],
            "weights": [1.0, 1.0, 1.0, 1.0]
        },
        "interaction_quality": {
            "assertions": [
                "对话自然流畅",
                "NPC的回应与玩家输入相关",
                "NPC能够理解玩家的意图",
                "对话能够持续进行，不会突然中断",
                "NPC的回应有助于推动对话发展"
            ],
            "weights": [1.0, 1.2, 1.5, 0.8, 1.0]
        },
        "basic_rules": {
            "assertions": [
                "不应包含脏话或侮辱性语言",
                "回复长度应在10-500字符之间",
                "对于玩家的提问应给予回应"
            ],
            "weights": [2.0, 1.0, 1.0]
        }
    }

    # 获取当前评分器的断言配置，如果未找到则使用默认
    grader_config = grader_assertions.get(grader_name, {
        "assertions": ["默认断言"],
        "weights": [1.0]
    })

    assertions = grader_config["assertions"]
    weights = grader_config["weights"]

    # 创建模拟的评分准则结果并计算加权总分
    criteria_results = {}
    total_weighted_score = 0.0
    total_weight = 0.0

    for i, (assertion, weight) in enumerate(zip(assertions, weights)):
        # 对于basic_rules的特殊处理：脏话检测总是通过（因为模拟对话中没有脏话）
        if grader_name == "basic_rules" and ("脏话" in assertion or "侮辱" in assertion):
            criterion_passed = True
            criterion_score = 1.0
            criterion_reason = f"准则{i+1}: 满足要求（模拟对话中无脏话）"
        else:
            # 模拟每个准则的通过率与基础分数相关
            criterion_passed = random.random() < base_score  # 通过概率与基础分数正相关
            criterion_score = 1.0 if criterion_passed else 0.0
            criterion_reason = f"准则{i+1}: {'满足要求' if criterion_passed else '有待改进'}"

        criterion_result = CriterionResult(
            criterion_name=f"assertion_{i+1}",
            criterion_description=assertion,
            score=criterion_score,
            passed=criterion_passed,
            reasoning=criterion_reason,
            evidence=[],
            weight=float(weight)
        )
        criteria_results[f"assertion_{i+1}"] = criterion_result

        # 累加加权分数和权重
        total_weighted_score += criterion_score * weight
        total_weight += weight

    # 计算加权平均分作为评分器总分
    calculated_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # 对于高分场景（>=0.9），确保分数不会因随机性过低
    if base_score >= 0.9 and calculated_score < 0.9:
        # 调整分数，使其接近基础分数但保持一致性
        adjustment_factor = 0.95  # 轻微调整
        calculated_score = base_score * adjustment_factor + calculated_score * (1 - adjustment_factor)

    # 通过阈值使用0.7（与config中大多数评分器一致）
    passed = calculated_score >= 0.7

    return GradingResult(
        grader_name=grader_name,
        score=calculated_score,
        passed=passed,
        reasoning=reasoning,
        evidence=[
            {
                "evidence_type": "llm_analysis",
                "content": f"基于对话内容的分析，{reasoning}",
                "relevance": 1.0,
                "source_indices": [0, 1, 2],
                "metadata": {}
            }
        ],
        metadata={
            "evaluation_time": datetime.now().isoformat(),
            "scenario_id": scenario_id,
            "base_score": base_score,  # 保留基础分数用于调试
            "calculated_score": calculated_score
        },
        criteria_results=criteria_results
    )


def create_mock_transcript(scenario_id: str) -> list:
    """创建模拟对话记录"""
    # 不同场景的对话模板
    transcripts = {
        "daily_chat_001": [
            {"speaker": "player", "message": "嗨，你好！今天天气真不错，你经常来这里散步吗？", "timestamp": time.time() - 10},
            {"speaker": "npc", "message": "你好！是的，我几乎每天都会来这个公园散步。今天天气确实很适合户外活动。", "timestamp": time.time() - 9},
            {"speaker": "player", "message": "听起来很棒！我最近刚搬到这附近，有什么推荐的餐馆或咖啡馆吗？", "timestamp": time.time() - 8},
            {"speaker": "npc", "message": "当然！街角那家'阳光咖啡馆'的咖啡很不错，还有'老街餐馆'的传统菜也很受欢迎。", "timestamp": time.time() - 7},
        ],
        "emotional_support_001": [
            {"speaker": "player", "message": "我今天心情很不好，工作压力太大了。", "timestamp": time.time() - 10},
            {"speaker": "npc", "message": "听起来你今天过得很辛苦，工作压力确实会让人感到疲惫。", "timestamp": time.time() - 9},
            {"speaker": "player", "message": "是的，我觉得自己快撑不住了。", "timestamp": time.time() - 8},
            {"speaker": "npc", "message": "每个人都有压力大的时候，重要的是找到适合自己的放松方式。", "timestamp": time.time() - 7},
        ],
        "opinion_conflict_001": [
            {"speaker": "player", "message": "我觉得这个政策完全不合理，应该立即取消。", "timestamp": time.time() - 10},
            {"speaker": "npc", "message": "我理解你的担忧，但也许我们可以从不同角度看待这个问题。", "timestamp": time.time() - 9},
            {"speaker": "player", "message": "有什么不同角度？这个政策明显有缺陷。", "timestamp": time.time() - 8},
            {"speaker": "npc", "message": "政策的制定通常考虑多方面因素，我们可以讨论如何改进而不是完全否定。", "timestamp": time.time() - 7},
        ],
        "basic_greeting_001": [
            {"speaker": "player", "message": "你好！", "timestamp": time.time() - 10},
            {"speaker": "npc", "message": "你好！", "timestamp": time.time() - 9},
            {"speaker": "player", "message": "你好吗？", "timestamp": time.time() - 8},
            {"speaker": "npc", "message": "我很好，谢谢！", "timestamp": time.time() - 7},
        ],
        "simple_qa_001": [
            {"speaker": "player", "message": "请问现在几点了？", "timestamp": time.time() - 10},
            {"speaker": "npc", "message": "现在是下午3点。", "timestamp": time.time() - 9},
            {"speaker": "player", "message": "图书馆在哪里？", "timestamp": time.time() - 8},
            {"speaker": "npc", "message": "图书馆在村庄广场的北侧。", "timestamp": time.time() - 7},
        ]
    }

    return transcripts.get(scenario_id, [
        {"speaker": "player", "message": "测试消息1", "timestamp": time.time() - 10},
        {"speaker": "npc", "message": "测试回复1", "timestamp": time.time() - 9},
        {"speaker": "player", "message": "测试消息2", "timestamp": time.time() - 8},
        {"speaker": "npc", "message": "测试回复2", "timestamp": time.time() - 7},
    ])


def load_scenarios_from_configs() -> list:
    """从配置文件加载场景"""
    scenarios_dir = project_root / "configs" / "scenarios"
    scenarios = []

    # 读取所有场景文件
    scenario_files = list(scenarios_dir.glob("*.yaml"))

    for scenario_file in scenario_files:
        try:
            import yaml
            with open(scenario_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # 创建Scenario对象
            scenario = Scenario(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                scenario_type=ScenarioType(data.get("scenario_type", "daily_chat")),
                player_profile=data.get("player_profile", {}),
                initial_prompt=data.get("initial_prompt", ""),
                max_turns=data.get("max_turns", 5),
                expected_outcomes=data.get("expected_outcomes", []),
                reference_solution=data.get("reference_solution", {}),
                metadata=data.get("metadata", {})
            )
            scenarios.append(scenario)
            print(f"[OK] 加载场景: {scenario.name} ({scenario.id})")

        except Exception as e:
            print(f"[ERROR] 加载场景文件 {scenario_file.name} 失败: {e}")

    return scenarios


def create_mock_evaluation_results(scenarios: list) -> list:
    """创建模拟评测结果"""
    results = []

    # 一轮评测使用同一个Agent和角色设定
    # 使用友好的村民作为默认角色（与npc_config.yaml中的friendly_npc保持一致）
    npc_role = {
        "name": "友好的村民",
        "personality": "热情友好，乐于助人",
        "background": "在这个村庄生活了30年，熟悉村里的每一个人和每一件事",
        "speaking_style": "亲切、温暖、充满关怀",
        "values": ["互助", "诚实", "善良", "社区精神"]
    }

    for scenario in scenarios:
        # 创建评测结果
        result = EvaluationResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            start_time=time.time() - 30,
            end_time=time.time() - 5,
            transcript=create_mock_transcript(scenario.id),
            grading_results={},
            final_score=0.0,
            passed=False,
            errors=[],
            metadata={
                "scenario_type": scenario.scenario_type,
                "random_seed": 42,
                "agent_type": "mock",
                "evaluation_time": datetime.now().isoformat(),
                "npc_role": npc_role  # 所有场景使用相同的NPC角色
            }
        )

        # 添加评分器结果
        graders = ["character_consistency", "interaction_quality", "basic_rules"]
        for grader_name in graders:
            grading_result = create_mock_grading_result(grader_name, scenario.id)
            result.add_grading_result(grader_name, grading_result)

        # 计算最终得分（加权平均）
        weights = {"character_consistency": 0.4, "interaction_quality": 0.4, "basic_rules": 0.2}
        final_score = result.calculate_final_score(weights)
        result.passed = final_score >= 0.7

        results.append(result)
        print(f"[OK] 创建评测结果: {scenario.name} - 得分: {final_score:.2%}")

    return results


def create_test_suite(scenarios: list) -> TestSuite:
    """创建测试套件"""
    return TestSuite(
        id="sample_suite",
        name="示例评测套件",
        description="包含5个不同场景的示例评测套件，展示系统能力",
        suite_type="capability",
        scenarios=scenarios,
        metadata={
            "generated_at": datetime.now().isoformat(),
            "total_scenarios": len(scenarios),
            "purpose": "示例演示"
        }
    )


async def main():
    """主函数"""
    print("[TARGET] AI Agent评测系统 - 示例报告生成器")
    print("=" * 50)

    # 创建输出目录
    output_dir = project_root / "examples" / "sample_reports"
    output_dir.mkdir(exist_ok=True)

    print("\n[FOLDER] 步骤1: 加载测试场景")
    scenarios = load_scenarios_from_configs()

    if not scenarios:
        print("[ERROR] 未找到任何场景配置文件")
        return

    print(f"[CHART] 共加载 {len(scenarios)} 个场景")

    print("\n[FOLDER] 步骤2: 创建模拟评测结果")
    results = create_mock_evaluation_results(scenarios)

    print("\n[FOLDER] 步骤3: 创建测试套件")
    test_suite = create_test_suite(scenarios)

    print("\n[FOLDER] 步骤4: 生成评测报告")

    # 生成JSON详细报告
    json_report_path = output_dir / "sample_detailed_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        report_data = {
            "metadata": {
                "run_id": f"sample_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "generator": "agent-eval-sample-generator",
                "version": "1.0.0",
                "note": "这是一个示例评测报告，包含模拟数据用于演示系统能力"
            },
            "summary": {
                "total_scenarios": len(scenarios),
                "passed_scenarios": sum(1 for r in results if r.passed),
                "failed_scenarios": sum(1 for r in results if not r.passed),
                "average_score": sum(r.final_score for r in results) / len(results) if results else 0.0,
                "min_score": min((r.final_score for r in results), default=0.0),
                "max_score": max((r.final_score for r in results), default=0.0),
                "total_duration": 25.0,
                "suite_type": "capability",
                "run_id": "sample_run",
                "timestamp": datetime.now().isoformat()
            },
            "test_suite": test_suite.model_dump(),
            "detailed_results": [result.to_dict() for result in results],
            "statistics": {
                "scenario_types": {
                    "daily_chat": {
                        "count": 2,
                        "passed": 2,
                        "average_score": 0.865,
                        "min_score": 0.85,
                        "max_score": 0.88,
                        "pass_rate": 1.0,
                        "score_std_dev": 0.021
                    },
                    "emotional_support": {
                        "count": 1,
                        "passed": 1,
                        "average_score": 0.72,
                        "min_score": 0.72,
                        "max_score": 0.72,
                        "pass_rate": 1.0,
                        "score_std_dev": 0.0
                    },
                    "opinion_conflict": {
                        "count": 1,
                        "passed": 0,
                        "average_score": 0.65,
                        "min_score": 0.65,
                        "max_score": 0.65,
                        "pass_rate": 0.0,
                        "score_std_dev": 0.0
                    },
                    "basic_greeting": {
                        "count": 1,
                        "passed": 1,
                        "average_score": 0.95,
                        "min_score": 0.95,
                        "max_score": 0.95,
                        "pass_rate": 1.0,
                        "score_std_dev": 0.0
                    }
                },
                "grader_statistics": {
                    "character_consistency": {
                        "count": 5,
                        "average_score": 0.81,
                        "min_score": 0.65,
                        "max_score": 0.95,
                        "score_std_dev": 0.12
                    },
                    "interaction_quality": {
                        "count": 5,
                        "average_score": 0.79,
                        "min_score": 0.62,
                        "max_score": 0.93,
                        "score_std_dev": 0.11
                    },
                    "basic_rules": {
                        "count": 5,
                        "average_score": 0.90,
                        "min_score": 0.85,
                        "max_score": 0.98,
                        "score_std_dev": 0.05
                    }
                },
                "total_runs": len(scenarios)
            }
        }

        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] JSON详细报告: {json_report_path}")

    # 使用报告生成器生成HTML报告
    try:
        from src.reports import generate_html_report
        html_report_path = generate_html_report(
            results=results,
            output_dir=str(output_dir),
            filename="sample_report.html",
            test_suite=test_suite
        )
        print(f"[OK] HTML可视化报告: {html_report_path}")
    except Exception as e:
        print(f"[WARNING] HTML报告生成失败（可能是依赖问题）: {e}")
        print("  正在创建简单的HTML报告...")

        # 创建简单的HTML报告
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent评测系统 - 示例报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #007acc; color: white; padding: 20px; border-radius: 5px; }}
        .scenario {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .score {{ font-size: 1.2em; font-weight: bold; }}
        .passed .score {{ color: #28a745; }}
        .failed .score {{ color: #dc3545; }}
        .transcript {{ background: #f8f9fa; padding: 10px; border-radius: 3px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI Agent评测系统 - 示例报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>包含 {len(scenarios)} 个测试场景的示例评测结果</p>
    </div>

    <h2>📊 评测摘要</h2>
    <ul>
        <li>测试场景总数: {len(scenarios)}</li>
        <li>通过场景: {sum(1 for r in results if r.passed)}</li>
        <li>失败场景: {sum(1 for r in results if not r.passed)}</li>
        <li>平均得分: {(sum(r.final_score for r in results) / len(results) if results else 0):.2%}</li>
    </ul>

    <h2>📋 详细评测结果</h2>
"""

        for i, result in enumerate(results):
            status_class = "passed" if result.passed else "failed"
            status_text = "通过" if result.passed else "失败"

            html_content += f"""
    <div class="scenario {status_class}">
        <h3>{result.scenario_name} (ID: {result.scenario_id})</h3>
        <p class="score">得分: {result.final_score:.2%} ({status_text})</p>

        <h4>评分结果</h4>
        <ul>
"""

            for grader_name, grading_result in result.grading_results.items():
                html_content += f"""
            <li><strong>{grader_name}</strong>: {grading_result.score:.2%} ({'通过' if grading_result.passed else '失败'})<br>
            <em>{grading_result.reasoning[:100]}...</em></li>
"""

            html_content += f"""
        </ul>

        <h4>对话记录 ({len(result.transcript)} 轮)</h4>
        <div class="transcript">
"""

            for turn in result.transcript:
                html_content += f"""
            <p><strong>{turn.get('speaker', 'unknown')}:</strong> {turn.get('message', '')}</p>
"""

            html_content += f"""
        </div>
    </div>
"""

        html_content += f"""
    <hr>
    <p><em>注：这是一个示例报告，数据为模拟生成，用于展示系统的报告格式和能力。</em></p>
    <p><em>查看JSON详细报告获取完整数据: {json_report_path.name}</em></p>
</body>
</html>
"""

        html_report_path = output_dir / "sample_report_simple.html"
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[OK] 简单HTML报告: {html_report_path}")

    print("\n[CELEBRATE] 示例报告生成完成！")
    print(f"[FOLDER] 报告位置: {output_dir}")
    print(f"[DOCUMENT] JSON详细报告: {json_report_path.name}")
    print(f"[GLOBE] HTML可视化报告: {'sample_report.html' if 'html_report_path' in locals() else 'sample_report_simple.html'}")
    print("\n[BULB] 提示: 这些是示例报告，用于展示系统能力。")
    print("      运行 `python examples/run_eval.py` 可以进行真实评测。")


if __name__ == "__main__":
    asyncio.run(main())