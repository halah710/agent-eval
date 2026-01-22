"""
基于代码规则的评分器（Code-based Grader）
基于Anthropic博客中提到的代码规则评分器
"""

import re
from typing import List, Dict, Any, Callable
from loguru import logger

from .base import Grader, GradingResult, GradingEvidence


class Rule:
    """评分规则"""

    def __init__(
        self,
        name: str,
        pattern: str,
        description: str,
        weight: float = 1.0,
        required: bool = False,
        case_sensitive: bool = False
    ):
        """
        初始化规则

        Args:
            name: 规则名称
            pattern: 正则表达式模式或关键词
            description: 规则描述
            weight: 权重
            required: 是否必需
            case_sensitive: 是否区分大小写
        """
        self.name = name
        self.pattern = pattern
        self.description = description
        self.weight = weight
        self.required = required
        self.case_sensitive = case_sensitive
        self.regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """检查文本是否匹配规则"""
        return bool(self.regex.search(text))

    def find_matches(self, text: str) -> List[re.Match]:
        """查找所有匹配项"""
        return list(self.regex.finditer(text))

    def __str__(self):
        return f"Rule(name={self.name}, pattern={self.pattern[:20]}...)"


class RuleBasedGrader(Grader):
    """基于代码规则的评分器"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化规则评分器

        Args:
            name: 评分器名称
            config: 配置参数，包括：
                - rules: 规则列表，每个规则包含name, pattern, description, weight, required
                - scoring_logic: 评分逻辑函数（可选）
                - pass_threshold: 通过阈值（默认：0.7）
        """
        super().__init__(name, config)
        self.rules = self._parse_rules(config.get("rules", []))
        self.scoring_logic = config.get("scoring_logic")
        self.pass_threshold = float(config.get("pass_threshold", 0.7))
        self.custom_functions = config.get("custom_functions", {})

    def _parse_rules(self, rules_config: List[Dict[str, Any]]) -> List[Rule]:
        """解析规则配置"""
        rules = []
        for rule_config in rules_config:
            rule = Rule(
                name=rule_config.get("name", f"rule_{len(rules)}"),
                pattern=rule_config.get("pattern", ""),
                description=rule_config.get("description", ""),
                weight=float(rule_config.get("weight", 1.0)),
                required=bool(rule_config.get("required", False)),
                case_sensitive=bool(rule_config.get("case_sensitive", False))
            )
            rules.append(rule)
        return rules

    async def initialize(self):
        """初始化规则评分器"""
        if not self.rules:
            logger.warning(f"规则评分器 {self.name} 没有配置规则")

        # 如果有自定义评分逻辑，验证其可用性
        if self.scoring_logic and callable(self.scoring_logic):
            logger.info(f"规则评分器 {self.name} 使用自定义评分逻辑")
        elif self.scoring_logic and isinstance(self.scoring_logic, str):
            # 尝试从custom_functions中获取
            if self.scoring_logic in self.custom_functions:
                self.scoring_logic = self.custom_functions[self.scoring_logic]
                logger.info(f"规则评分器 {self.name} 使用自定义函数：{self.scoring_logic}")
            else:
                logger.warning(f"未找到自定义函数：{self.scoring_logic}")

        self._initialized = True
        logger.info(f"规则评分器初始化成功：{self.name}，规则数：{len(self.rules)}")

    async def grade(self, transcript: List[Dict[str, Any]]) -> GradingResult:
        """
        对对话记录进行规则评分

        Args:
            transcript: 对话记录

        Returns:
            评分结果
        """
        if not self._initialized:
            await self.initialize()

        # 提取所有NPC回复
        npc_messages = []
        for i, turn in enumerate(transcript):
            if turn.get("speaker") == "npc":
                npc_messages.append((i, turn.get("message", "")))

        # 应用规则评分
        rule_results = []
        total_weight = 0.0
        weighted_score = 0.0

        for rule in self.rules:
            rule_matched = False
            evidence_matches = []

            # 检查每个NPC消息
            for turn_idx, message in npc_messages:
                if rule.matches(message):
                    rule_matched = True
                    matches = rule.find_matches(message)
                    for match in matches:
                        evidence_matches.append({
                            "turn_index": turn_idx,
                            "match_text": match.group(),
                            "start": match.start(),
                            "end": match.end()
                        })

            # 计算规则得分
            if rule.required:
                rule_score = 1.0 if rule_matched else 0.0
            else:
                rule_score = 1.0 if rule_matched else 0.5  # 未匹配但不是必需的，给部分分数

            rule_results.append({
                "rule": rule,
                "matched": rule_matched,
                "score": rule_score,
                "evidence": evidence_matches
            })

            total_weight += rule.weight
            weighted_score += rule_score * rule.weight

        # 计算最终得分
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0

        passed = final_score >= self.pass_threshold

        # 应用自定义评分逻辑（如果有）
        if callable(self.scoring_logic):
            try:
                custom_result = self.scoring_logic(transcript, rule_results)
                if isinstance(custom_result, dict):
                    final_score = custom_result.get("score", final_score)
                    passed = custom_result.get("passed", passed)
            except Exception as e:
                logger.error(f"自定义评分逻辑执行失败：{e}")

        # 构建评分结果
        result = GradingResult(
            grader_name=self.name,
            score=final_score,
            passed=passed,
            reasoning=self._build_reasoning(rule_results, final_score),
            confidence=0.9  # 规则评分置信度较高
        )

        # 添加证据
        for rule_result in rule_results:
            if rule_result["evidence"]:
                evidence_text = f"规则 '{rule_result['rule'].name}' 匹配："
                for evidence in rule_result["evidence"][:3]:  # 最多显示3个匹配
                    evidence_text += f"\n- 第{evidence['turn_index']+1}轮：{evidence['match_text']}"

                result.add_evidence(
                    evidence_type="rule_match",
                    content=evidence_text,
                    relevance=rule_result["rule"].weight,
                    source_indices=[evidence["turn_index"] for evidence in rule_result["evidence"]]
                )

        logger.info(f"规则评分完成：{self.name}，得分：{final_score:.2f}，通过：{passed}")
        return result

    def _build_reasoning(self, rule_results: List[Dict], final_score: float) -> str:
        """构建评分理由"""
        passed_rules = sum(1 for r in rule_results if r["matched"])
        total_rules = len(rule_results)
        required_rules = sum(1 for r in rule_results if r["rule"].required)
        passed_required = sum(1 for r in rule_results if r["rule"].required and r["matched"])

        reasoning = f"评估了 {total_rules} 条规则，匹配了 {passed_rules} 条。"
        if required_rules > 0:
            reasoning += f" 必需规则 {passed_required}/{required_rules} 条通过。"
        reasoning += f" 最终得分：{final_score:.2f}。"

        return reasoning

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info(f"规则评分器已关闭：{self.name}")

    def __str__(self):
        return f"RuleBasedGrader(name={self.name}, rules={len(self.rules)})"