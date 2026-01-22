"""
评测器基类和核心数据结构定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import time
import random
import json
from pathlib import Path

from ..agents.base import NPCAgent, DialogueContext
from ..graders.base import Grader, GradingResult


class ScenarioType(str, Enum):
    """测试场景类型"""
    DAILY_CHAT = "daily_chat"  # 日常聊天
    EMOTIONAL_SUPPORT = "emotional_support"  # 情感支持
    OPINION_CONFLICT = "opinion_conflict"  # 观点冲突
    KNOWLEDGE_TEST = "knowledge_test"  # 知识测试
    ROLE_PLAYING = "role_playing"  # 角色扮演


class Scenario(BaseModel):
    """测试场景定义"""
    id: str = Field(..., description="场景ID")
    name: str = Field(..., description="场景名称")
    description: str = Field(..., description="场景描述")
    scenario_type: ScenarioType = Field(..., description="场景类型")
    player_profile: Dict[str, Any] = Field(default_factory=dict, description="玩家角色配置")
    initial_prompt: str = Field(..., description="初始提示（玩家第一句话）")
    max_turns: int = Field(default=5, description="最大对话轮次")
    expected_outcomes: List[str] = Field(default_factory=list, description="预期结果描述")
    reference_solution: Optional[Dict[str, Any]] = Field(None, description="参考解决方案")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """从字典创建"""
        return cls(**data)


class TestSuite(BaseModel):
    """测试套件"""
    id: str = Field(..., description="套件ID")
    name: str = Field(..., description="套件名称")
    description: str = Field(..., description="套件描述")
    suite_type: str = Field(..., description="套件类型：capability 或 regression")
    scenarios: List[Scenario] = Field(default_factory=list, description="包含的场景")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def add_scenario(self, scenario: Scenario):
        """添加场景"""
        self.scenarios.append(scenario)

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """获取指定场景"""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None


class EvaluationResult(BaseModel):
    """评测结果"""
    scenario_id: str = Field(..., description="场景ID")
    scenario_name: str = Field(..., description="场景名称")
    start_time: float = Field(default_factory=time.time, description="开始时间")
    end_time: Optional[float] = Field(None, description="结束时间")
    transcript: List[Dict[str, Any]] = Field(default_factory=list, description="完整对话记录")
    grading_results: Dict[str, GradingResult] = Field(default_factory=dict, description="评分结果")
    final_score: float = Field(default=0.0, description="最终得分（0-1）")
    passed: bool = Field(default=False, description="是否通过")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def add_turn(self, speaker: str, message: str, metadata: Dict[str, Any] = None):
        """添加对话轮次"""
        turn = {
            "speaker": speaker,
            "message": message,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.transcript.append(turn)

    def add_grading_result(self, grader_name: str, result: GradingResult):
        """添加评分结果"""
        self.grading_results[grader_name] = result

    def calculate_final_score(self, weights: Dict[str, float] = None) -> float:
        """计算最终得分"""
        if not self.grading_results:
            return 0.0

        if weights is None:
            # 默认等权重
            weights = {name: 1.0 for name in self.grading_results.keys()}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for grader_name, result in self.grading_results.items():
            weight = weights.get(grader_name, 1.0)
            weighted_sum += result.score * weight

        self.final_score = weighted_sum / total_weight
        return self.final_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = self.model_dump()
        data["duration"] = (self.end_time or time.time()) - self.start_time
        return data

    def save_to_file(self, filepath: str):
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class Evaluator(ABC):
    """评测器基类"""

    def __init__(
        self,
        npc_agent: NPCAgent,
        graders: List[Grader],
        config: Dict[str, Any] = None
    ):
        """
        初始化评测器

        Args:
            npc_agent: NPC Agent实例
            graders: 评分器列表
            config: 配置参数
        """
        self.npc_agent = npc_agent
        self.graders = {grader.name: grader for grader in graders}
        self.config = config or {}
        self.random_seed = self.config.get("random_seed", 42)

        # 设置随机种子确保可复现性
        random.seed(self.random_seed)

    @abstractmethod
    async def evaluate_scenario(self, scenario: Scenario) -> EvaluationResult:
        """
        评测单个场景

        Args:
            scenario: 测试场景

        Returns:
            评测结果
        """
        pass

    async def evaluate_suite(self, test_suite: TestSuite) -> List[EvaluationResult]:
        """
        评测整个测试套件

        Args:
            test_suite: 测试套件

        Returns:
            评测结果列表
        """
        results = []
        for scenario in test_suite.scenarios:
            result = await self.evaluate_scenario(scenario)
            results.append(result)
        return results

    def _create_dialogue_context(self) -> DialogueContext:
        """创建对话上下文"""
        return DialogueContext()

    def _initialize_random_state(self):
        """初始化随机状态（确保可复现性）"""
        random.seed(self.random_seed)
        # 如果有其他随机源也需要设置种子

    def get_grader(self, grader_name: str) -> Optional[Grader]:
        """获取指定评分器"""
        return self.graders.get(grader_name)