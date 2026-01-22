"""
评分转换器 - 将断言的真假判断转换为分数和通过状态
支持多种转换策略：简单比例、加权平均、必选断言等
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from loguru import logger


class ConversionStrategy(str, Enum):
    """转换策略类型"""
    SIMPLE_RATIO = "simple_ratio"  # 简单比例：True的比例作为分数
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均：每个断言有权重
    MUST_PASS = "must_pass"  # 必选断言：某些断言必须通过
    THRESHOLD_BASED = "threshold_based"  # 阈值判定：基于阈值判断通过


class ScoreConverter:
    """评分转换器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化评分转换器

        Args:
            config: 配置参数，包括：
                - strategy: 转换策略（simple_ratio, weighted_average, must_pass, threshold_based）
                - pass_threshold: 通过阈值（默认：0.7）
                - assertion_weights: 断言权重字典（仅weighted_average策略需要）
                - must_pass_assertions: 必须通过的断言索引列表（仅must_pass策略需要）
                - strict_threshold: 严格阈值（仅threshold_based策略需要）
        """
        self.config = config or {}
        self.strategy = self.config.get("strategy", ConversionStrategy.SIMPLE_RATIO)
        self.pass_threshold = float(self.config.get("pass_threshold", 0.7))

        # 断言权重配置
        self.assertion_weights = self.config.get("assertion_weights", {})
        if isinstance(self.assertion_weights, list):
            # 如果是列表，转换为索引字典
            self.assertion_weights = {i: weight for i, weight in enumerate(self.assertion_weights)}

        # 必须通过的断言
        self.must_pass_assertions = self.config.get("must_pass_assertions", [])

        # 严格阈值（用于threshold_based策略）
        self.strict_threshold = float(self.config.get("strict_threshold", 1.0))

    def convert(
        self,
        assertion_results: Dict[int, bool],
        total_assertions: int,
        reasoning: str = ""
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        将断言结果转换为分数和通过状态

        Args:
            assertion_results: 断言结果字典 {断言索引: True/False}
            total_assertions: 总断言数量
            reasoning: 原始评估理由

        Returns:
            Tuple[分数, 是否通过, 转换详情]
        """
        # 确保所有断言都有结果（缺失的视为False）
        complete_results = {}
        for i in range(total_assertions):
            complete_results[i] = assertion_results.get(i, False)

        logger.debug(f"转换断言结果：{complete_results}，策略：{self.strategy}")

        if self.strategy == ConversionStrategy.SIMPLE_RATIO:
            return self._convert_simple_ratio(complete_results, reasoning)
        elif self.strategy == ConversionStrategy.WEIGHTED_AVERAGE:
            return self._convert_weighted_average(complete_results, reasoning)
        elif self.strategy == ConversionStrategy.MUST_PASS:
            return self._convert_must_pass(complete_results, reasoning)
        elif self.strategy == ConversionStrategy.THRESHOLD_BASED:
            return self._convert_threshold_based(complete_results, reasoning)
        else:
            logger.warning(f"未知转换策略：{self.strategy}，使用简单比例策略")
            return self._convert_simple_ratio(complete_results, reasoning)

    def _convert_simple_ratio(
        self,
        assertion_results: Dict[int, bool],
        reasoning: str
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """简单比例转换策略"""
        true_count = sum(1 for v in assertion_results.values() if v)
        total = len(assertion_results)
        score = true_count / total if total > 0 else 0.0
        passed = score >= self.pass_threshold

        details = {
            "strategy": "simple_ratio",
            "true_count": true_count,
            "total_assertions": total,
            "pass_threshold": self.pass_threshold,
            "calculation": f"{true_count}/{total} = {score:.2f}"
        }

        logger.debug(f"简单比例转换：得分={score:.2f}，通过={passed}")
        return score, passed, details

    def _convert_weighted_average(
        self,
        assertion_results: Dict[int, bool],
        reasoning: str
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """加权平均转换策略"""
        total_weight = 0.0
        weighted_sum = 0.0

        for i, is_true in assertion_results.items():
            weight = self.assertion_weights.get(i, 1.0)  # 默认权重为1.0
            total_weight += weight
            if is_true:
                weighted_sum += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = score >= self.pass_threshold

        details = {
            "strategy": "weighted_average",
            "total_weight": total_weight,
            "weighted_sum": weighted_sum,
            "pass_threshold": self.pass_threshold,
            "assertion_weights": self.assertion_weights,
            "calculation": f"{weighted_sum:.2f}/{total_weight:.2f} = {score:.2f}"
        }

        logger.debug(f"加权平均转换：得分={score:.2f}，通过={passed}")
        return score, passed, details

    def _convert_must_pass(
        self,
        assertion_results: Dict[int, bool],
        reasoning: str
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """必选断言转换策略"""
        # 检查必选断言是否都通过
        must_pass_failed = []
        for idx in self.must_pass_assertions:
            if not assertion_results.get(idx, False):
                must_pass_failed.append(idx)

        # 如果必选断言有失败的，直接不通过
        if must_pass_failed:
            score = 0.0
            passed = False
            details = {
                "strategy": "must_pass",
                "must_pass_failed": must_pass_failed,
                "reason": f"必选断言 {must_pass_failed} 未通过"
            }
            logger.debug(f"必选断言转换：必选断言失败，直接不通过")
            return score, passed, details

        # 必选断言都通过，计算其他断言的分数
        optional_results = {i: v for i, v in assertion_results.items()
                          if i not in self.must_pass_assertions}

        true_count = sum(1 for v in optional_results.values() if v)
        total = len(optional_results)
        score = true_count / total if total > 0 else 1.0  # 如果没有可选断言，得分为1.0
        passed = score >= self.pass_threshold

        details = {
            "strategy": "must_pass",
            "must_pass_passed": True,
            "optional_true_count": true_count,
            "optional_total": total,
            "pass_threshold": self.pass_threshold,
            "calculation": f"{true_count}/{total} = {score:.2f}"
        }

        logger.debug(f"必选断言转换：必选断言通过，可选得分={score:.2f}，通过={passed}")
        return score, passed, details

    def _convert_threshold_based(
        self,
        assertion_results: Dict[int, bool],
        reasoning: str
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """阈值判定转换策略"""
        true_count = sum(1 for v in assertion_results.values() if v)
        total = len(assertion_results)
        ratio = true_count / total if total > 0 else 0.0

        # 使用两个阈值：pass_threshold和strict_threshold
        if ratio >= self.strict_threshold:
            # 达到严格阈值，得分为1.0
            score = 1.0
            passed = True
        elif ratio >= self.pass_threshold:
            # 达到通过阈值但未达到严格阈值
            score = ratio  # 使用实际比例作为分数
            passed = True
        else:
            # 未达到通过阈值
            score = ratio
            passed = False

        details = {
            "strategy": "threshold_based",
            "true_count": true_count,
            "total_assertions": total,
            "ratio": ratio,
            "pass_threshold": self.pass_threshold,
            "strict_threshold": self.strict_threshold,
            "calculation": f"{true_count}/{total} = {ratio:.2f}"
        }

        logger.debug(f"阈值判定转换：比例={ratio:.2f}，得分={score:.2f}，通过={passed}")
        return score, passed, details

    def get_assertion_status_text(
        self,
        assertion_results: Dict[int, bool],
        assertions: List[str]
    ) -> str:
        """获取断言状态文本，用于报告输出"""
        lines = []
        for i, assertion in enumerate(assertions):
            status = "[通过]" if assertion_results.get(i, False) else "[失败]"
            weight_info = ""
            if self.strategy == ConversionStrategy.WEIGHTED_AVERAGE:
                weight = self.assertion_weights.get(i, 1.0)
                weight_info = f" (权重: {weight})"
            elif self.strategy == ConversionStrategy.MUST_PASS:
                if i in self.must_pass_assertions:
                    weight_info = " (必选)"

            lines.append(f"{i+1}. {status} {assertion}{weight_info}")

        return "\n".join(lines)


def create_converter_from_config(config: Dict[str, Any]) -> ScoreConverter:
    """从配置创建评分转换器（工厂函数）"""
    return ScoreConverter(config)