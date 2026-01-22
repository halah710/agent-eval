"""
评分器基类和核心数据结构
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class GradingEvidence(BaseModel):
    """评分证据"""
    evidence_type: str = Field(..., description="证据类型：quote, summary, analysis等")
    content: str = Field(..., description="证据内容")
    relevance: float = Field(default=1.0, description="相关性分数（0-1）")
    source_indices: List[int] = Field(default_factory=list, description="来源索引（对话轮次）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class CriterionResult(BaseModel):
    """单个评分准则的结果"""
    criterion_name: str = Field(..., description="准则名称")
    criterion_description: str = Field(..., description="准则描述")
    score: float = Field(..., ge=0.0, le=1.0, description="准则得分（0-1）")
    passed: bool = Field(..., description="是否通过该准则")
    reasoning: str = Field(..., description="该准则的评分理由")
    evidence: List[GradingEvidence] = Field(default_factory=list, description="证据")
    weight: float = Field(default=1.0, description="权重")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


class GradingResult(BaseModel):
    """评分结果"""
    grader_name: str = Field(..., description="评分器名称")
    score: float = Field(..., ge=0.0, le=1.0, description="评分（0-1）")
    passed: bool = Field(..., description="是否通过")
    evidence: List[GradingEvidence] = Field(default_factory=list, description="评分证据")
    reasoning: str = Field(..., description="评分理由")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    criteria_results: Dict[str, CriterionResult] = Field(default_factory=dict, description="各准则详细结果")

    def add_evidence(
        self,
        evidence_type: str,
        content: str,
        relevance: float = 1.0,
        source_indices: List[int] = None
    ):
        """添加证据"""
        evidence = GradingEvidence(
            evidence_type=evidence_type,
            content=content,
            relevance=relevance,
            source_indices=source_indices or []
        )
        self.evidence.append(evidence)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


class Grader(ABC):
    """评分器基类"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化评分器

        Args:
            name: 评分器名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self):
        """初始化评分器"""
        pass

    @abstractmethod
    async def grade(self, transcript: List[Dict[str, Any]]) -> GradingResult:
        """
        对对话记录进行评分

        Args:
            transcript: 对话记录，每轮包含speaker, message, timestamp等字段

        Returns:
            评分结果
        """
        pass

    @abstractmethod
    async def close(self):
        """清理资源"""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class GraderType(str, Enum):
    """评分器类型"""
    NATURAL_LANGUAGE = "natural_language"  # 自然语言断言
    RULE_BASED = "rule_based"  # 基于代码规则
    HYBRID = "hybrid"  # 混合评分器
    MULTI_JUDGE = "multi_judge"  # 多评判者