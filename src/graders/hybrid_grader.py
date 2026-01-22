"""
混合评分器
组合多个评分器的结果
"""

from typing import List, Dict, Any
from loguru import logger

from .base import Grader, GradingResult


class HybridGrader(Grader):
    """混合评分器，组合多个评分器的结果"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化混合评分器

        Args:
            name: 评分器名称
            config: 配置参数，包括：
                - graders: 子评分器配置列表
                - aggregation_strategy: 聚合策略（weighted_average, majority, min, max）
                - weights: 权重字典（当使用weighted_average时）
        """
        super().__init__(name, config)
        self.grader_configs = config.get("graders", [])
        self.aggregation_strategy = config.get("aggregation_strategy", "weighted_average")
        self.weights = config.get("weights", {})
        self.graders: List[Grader] = []

    async def initialize(self):
        """初始化混合评分器及其子评分器"""
        from .factory import GraderFactory

        # 创建子评分器
        for i, grader_config in enumerate(self.grader_configs):
            grader_name = grader_config.get("name", f"{self.name}_sub_{i}")
            grader = GraderFactory.create_grader(grader_name, grader_config)
            await grader.initialize()
            self.graders.append(grader)

        self._initialized = True
        logger.info(f"混合评分器初始化成功：{self.name}，包含 {len(self.graders)} 个子评分器")

    async def grade(self, transcript: List[Dict[str, Any]]) -> GradingResult:
        """
        对对话记录进行混合评分

        Args:
            transcript: 对话记录

        Returns:
            混合评分结果
        """
        if not self._initialized:
            await self.initialize()

        # 运行所有子评分器
        sub_results = []
        for grader in self.graders:
            try:
                result = await grader.grade(transcript)
                sub_results.append((grader.name, result))
                logger.debug(f"子评分器 {grader.name} 完成评分，得分：{result.score:.2f}")
            except Exception as e:
                logger.error(f"子评分器 {grader.name} 评分失败：{e}")
                # 添加默认失败结果
                sub_results.append((grader.name, GradingResult(
                    grader_name=grader.name,
                    score=0.0,
                    passed=False,
                    reasoning=f"评分失败：{str(e)}",
                    evidence=[]
                )))

        # 聚合结果
        final_result = self._aggregate_results(sub_results)

        logger.info(f"混合评分完成：{self.name}，最终得分：{final_result.score:.2f}")
        return final_result

    def _aggregate_results(self, sub_results: List[tuple[str, GradingResult]]) -> GradingResult:
        """聚合子评分器结果"""
        if not sub_results:
            return GradingResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                reasoning="没有可用的评分结果",
                evidence=[]
            )

        # 根据聚合策略计算最终得分
        if self.aggregation_strategy == "weighted_average":
            final_score = self._weighted_average(sub_results)
        elif self.aggregation_strategy == "majority":
            final_score = self._majority_vote(sub_results)
        elif self.aggregation_strategy == "min":
            final_score = min(r.score for _, r in sub_results)
        elif self.aggregation_strategy == "max":
            final_score = max(r.score for _, r in sub_results)
        else:
            logger.warning(f"未知聚合策略：{self.aggregation_strategy}，使用加权平均")
            final_score = self._weighted_average(sub_results)

        # 构建最终结果
        reasoning = f"混合评分（策略：{self.aggregation_strategy}），包含 {len(sub_results)} 个子评分器：\n"
        for grader_name, result in sub_results:
            reasoning += f"- {grader_name}: {result.score:.2f} ({'通过' if result.passed else '未通过'})\n"

        # 收集证据
        evidence = []
        for grader_name, result in sub_results:
            if result.evidence:
                evidence.extend(result.evidence)

        return GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=final_score >= 0.7,  # 默认阈值
            reasoning=reasoning,
            evidence=evidence,
            confidence=0.8  # 混合评分置信度
        )

    def _weighted_average(self, sub_results: List[tuple[str, GradingResult]]) -> float:
        """加权平均"""
        total_weight = 0.0
        weighted_sum = 0.0

        for grader_name, result in sub_results:
            weight = self.weights.get(grader_name, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _majority_vote(self, sub_results: List[tuple[str, GradingResult]]) -> float:
        """多数决"""
        passed_count = sum(1 for _, r in sub_results if r.passed)
        total_count = len(sub_results)

        # 计算通过比例作为分数
        return passed_count / total_count if total_count > 0 else 0.0

    async def close(self):
        """清理资源"""
        for grader in self.graders:
            try:
                await grader.close()
            except Exception as e:
                logger.error(f"关闭子评分器 {grader.name} 时出错：{e}")

        self._initialized = False
        logger.info(f"混合评分器已关闭：{self.name}")

    def __str__(self):
        return f"HybridGrader(name={self.name}, strategy={self.aggregation_strategy}, sub_graders={len(self.graders)})"