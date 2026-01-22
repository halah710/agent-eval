"""
多评判者评分器
基于Anthropic博客中的多评判者投票机制
"""

from typing import List, Dict, Any
from loguru import logger

from .base import Grader, GradingResult


class MultiJudgeGrader(Grader):
    """多评判者评分器，支持多个评分器并行评估和投票"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化多评判者评分器

        Args:
            name: 评分器名称
            config: 配置参数，包括：
                - judges: 评判者列表（评分器名称或配置）
                - voting_strategy: 投票策略（majority, weighted_average, unanimous, consensus）
                - weights: 权重字典（当使用weighted_average时）
                - consensus_threshold: 共识阈值（当使用consensus时）
        """
        super().__init__(name, config)
        self.judges_config = config.get("judges", [])
        self.voting_strategy = config.get("voting_strategy", "majority")
        self.weights = config.get("weights", {})
        self.consensus_threshold = float(config.get("consensus_threshold", 0.8))
        self.judges: List[Grader] = []

    async def initialize(self):
        """初始化多评判者评分器"""
        # 注意：这里不创建子评分器，因为它们应该已经在外部创建
        # 这个评分器主要是协调已存在的评分器
        self._initialized = True
        logger.info(f"多评判者评分器初始化成功：{self.name}，策略：{self.voting_strategy}")

    async def grade(self, transcript: List[Dict[str, Any]]) -> GradingResult:
        """
        对对话记录进行多评判者评分

        Args:
            transcript: 对话记录

        Returns:
            多评判者评分结果
        """
        if not self._initialized:
            await self.initialize()

        # 注意：这里假设judges已经在外部创建并通过graders参数传入
        # 在实际使用中，这个评分器需要访问其他评分器的结果
        # 这里返回一个占位结果，实际实现需要与Evaluator集成

        logger.warning(f"MultiJudgeGrader {self.name} 需要与Evaluator集成才能访问其他评分器结果")

        # 返回默认结果
        return GradingResult(
            grader_name=self.name,
            score=0.5,
            passed=False,
            reasoning="多评判者评分器需要与Evaluator集成才能工作",
            evidence=[],
            confidence=0.5
        )

    def aggregate_results(self, judge_results: Dict[str, GradingResult]) -> GradingResult:
        """
        聚合多个评判者的结果

        Args:
            judge_results: 评判者结果字典

        Returns:
            聚合后的评分结果
        """
        if not judge_results:
            return GradingResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                reasoning="没有可用的评判者结果",
                evidence=[]
            )

        # 根据投票策略聚合结果
        if self.voting_strategy == "majority":
            return self._majority_vote(judge_results)
        elif self.voting_strategy == "weighted_average":
            return self._weighted_average(judge_results)
        elif self.voting_strategy == "unanimous":
            return self._unanimous_vote(judge_results)
        elif self.voting_strategy == "consensus":
            return self._consensus_vote(judge_results)
        else:
            logger.warning(f"未知投票策略：{self.voting_strategy}，使用多数决")
            return self._majority_vote(judge_results)

    def _majority_vote(self, judge_results: Dict[str, GradingResult]) -> GradingResult:
        """多数决"""
        passed_count = sum(1 for r in judge_results.values() if r.passed)
        total_count = len(judge_results)

        final_score = passed_count / total_count if total_count > 0 else 0.0
        passed = final_score >= 0.5  # 简单多数通过

        # 构建理由
        reasoning = f"多数决投票：{passed_count}/{total_count} 个评判者通过。\n"
        for judge_name, result in judge_results.items():
            reasoning += f"- {judge_name}: {result.score:.2f} ({'通过' if result.passed else '未通过'})\n"

        # 收集证据
        evidence = []
        for result in judge_results.values():
            if result.evidence:
                evidence.extend(result.evidence)

        return GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=passed,
            reasoning=reasoning,
            evidence=evidence,
            confidence=0.8
        )

    def _weighted_average(self, judge_results: Dict[str, GradingResult]) -> GradingResult:
        """加权平均"""
        total_weight = 0.0
        weighted_sum = 0.0

        for judge_name, result in judge_results.items():
            weight = self.weights.get(judge_name, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = final_score >= 0.7  # 默认阈值

        # 构建理由
        reasoning = f"加权平均：最终得分 {final_score:.2f}。\n"
        for judge_name, result in judge_results.items():
            weight = self.weights.get(judge_name, 1.0)
            reasoning += f"- {judge_name} (权重{weight:.1f}): {result.score:.2f}\n"

        # 收集证据
        evidence = []
        for result in judge_results.values():
            if result.evidence:
                evidence.extend(result.evidence)

        return GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=passed,
            reasoning=reasoning,
            evidence=evidence,
            confidence=0.85
        )

    def _unanimous_vote(self, judge_results: Dict[str, GradingResult]) -> GradingResult:
        """一致通过"""
        all_passed = all(r.passed for r in judge_results.values())
        final_score = 1.0 if all_passed else 0.0

        reasoning = f"一致通过投票：{'所有评判者一致通过' if all_passed else '未达成一致'}。\n"
        for judge_name, result in judge_results.items():
            reasoning += f"- {judge_name}: {'通过' if result.passed else '未通过'}\n"

        evidence = []
        for result in judge_results.values():
            if result.evidence:
                evidence.extend(result.evidence)

        return GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=all_passed,
            reasoning=reasoning,
            evidence=evidence,
            confidence=0.9 if all_passed else 0.5
        )

    def _consensus_vote(self, judge_results: Dict[str, GradingResult]) -> GradingResult:
        """共识投票（达到阈值比例）"""
        passed_count = sum(1 for r in judge_results.values() if r.passed)
        total_count = len(judge_results)
        consensus_ratio = passed_count / total_count if total_count > 0 else 0.0

        final_score = consensus_ratio
        passed = consensus_ratio >= self.consensus_threshold

        reasoning = f"共识投票：{passed_count}/{total_count} 通过，共识比例 {consensus_ratio:.2f} "
        reasoning += f"(阈值 {self.consensus_threshold})。\n"
        for judge_name, result in judge_results.items():
            reasoning += f"- {judge_name}: {'通过' if result.passed else '未通过'}\n"

        evidence = []
        for result in judge_results.values():
            if result.evidence:
                evidence.extend(result.evidence)

        return GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=passed,
            reasoning=reasoning,
            evidence=evidence,
            confidence=0.8 if passed else 0.6
        )

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info(f"多评判者评分器已关闭：{self.name}")

    def __str__(self):
        return f"MultiJudgeGrader(name={self.name}, strategy={self.voting_strategy}, judges={len(self.judges_config)})"