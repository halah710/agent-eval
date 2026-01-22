#!/usr/bin/env python3
"""
测试所有模块导入
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("测试模块导入...")

# 测试agents模块
try:
    from src.agents.base import NPCAgent, NPCRole, DialogueContext
    from src.agents.factory import AgentFactory
    from src.agents.mock_agent import MockAgent
    print("[通过] agents模块导入成功")
except ImportError as e:
    print(f"[失败] agents模块导入失败: {e}")

# 测试evaluator模块
try:
    from src.evaluator.base import Evaluator, Scenario, TestSuite, EvaluationResult
    from src.evaluator.simple_evaluator import SimpleEvaluator
    print("[通过] evaluator模块导入成功")
except ImportError as e:
    print(f"[失败] evaluator模块导入失败: {e}")

# 测试graders模块
try:
    from src.graders.base import Grader, GradingResult, GradingEvidence
    from src.graders.factory import GraderFactory
    from src.graders.natural_language_grader import NaturalLanguageGrader
    from src.graders.rule_based_grader import RuleBasedGrader
    from src.graders.hybrid_grader import HybridGrader
    from src.graders.multi_judge_grader import MultiJudgeGrader
    print("[通过] graders模块导入成功")
except ImportError as e:
    print(f"[失败] graders模块导入失败: {e}")

# 测试players模块
try:
    from src.players.base import PlayerSimulator, PlayerProfile, DialogueAction
    from src.players.factory import PlayerFactory
    from src.players.simple_player import SimplePlayer
    print("[通过] players模块导入成功")
except ImportError as e:
    print(f"[失败] players模块导入失败: {e}")

# 测试utils模块
try:
    from src.utils.random_state import random_state_manager, set_global_seed
    print("[通过] utils模块导入成功")
except ImportError as e:
    print(f"[失败] utils模块导入失败: {e}")

print("\n所有模块导入测试完成！")