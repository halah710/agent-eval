"""
评分器工厂
"""

from typing import Dict, Any, Type
from loguru import logger

from .base import Grader
from .natural_language_grader import NaturalLanguageGrader
from .rule_based_grader import RuleBasedGrader
from .hybrid_grader import HybridGrader
from .multi_judge_grader import MultiJudgeGrader


class GraderFactory:
    """评分器工厂类"""

    _grader_classes = {
        "natural_language": NaturalLanguageGrader,
        "rule_based": RuleBasedGrader,
        "hybrid": HybridGrader,
        "multi_judge": MultiJudgeGrader,
    }

    @classmethod
    def register_grader(cls, grader_type: str, grader_class: Type[Grader]):
        """注册新的评分器类型"""
        cls._grader_classes[grader_type] = grader_class
        logger.info(f"注册评分器类型：{grader_type} -> {grader_class.__name__}")

    @classmethod
    def create_grader(cls, name: str, config: Dict[str, Any]) -> Grader:
        """
        创建评分器实例

        Args:
            name: 评分器名称
            config: 配置参数，包含type字段指定评分器类型

        Returns:
            评分器实例
        """
        grader_type = config.get("type", "natural_language")
        grader_config = config.get("config", {})

        if grader_type not in cls._grader_classes:
            raise ValueError(f"不支持的评分器类型：{grader_type}")

        grader_class = cls._grader_classes[grader_type]

        logger.info(f"创建评分器：{name} (类型：{grader_type})")
        return grader_class(name, grader_config)

    @classmethod
    def create_graders_from_config(cls, graders_config: Dict[str, Dict[str, Any]]) -> Dict[str, Grader]:
        """
        从配置字典创建多个评分器

        Args:
            graders_config: 评分器配置字典

        Returns:
            评分器字典（名称 -> 实例）
        """
        graders = {}
        for name, config in graders_config.items():
            try:
                grader = cls.create_grader(name, config)
                graders[name] = grader
            except Exception as e:
                logger.error(f"创建评分器 {name} 失败：{e}")
                raise

        return graders