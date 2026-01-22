"""
自然语言断言评分器（Model-based Grader）
基于Anthropic博客中提到的自然语言断言技术
"""

import os
from typing import List, Dict, Any, Optional
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Grader, GradingResult, GradingEvidence, CriterionResult
from .score_converter import ScoreConverter, ConversionStrategy


class NaturalLanguageGrader(Grader):
    """自然语言断言评分器"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化自然语言评分器

        Args:
            name: 评分器名称
            config: 配置参数，包括：
                - assertions: 自然语言断言列表
                - api_key: OpenAI API密钥
                - model: 模型名称
                - temperature: 温度参数
                - scoring_criteria: 评分标准描述
        """
        super().__init__(name, config)
        self.assertions = config.get("assertions", [])
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(config.get("temperature", 0.0))
        self.scoring_criteria = config.get("scoring_criteria", "请根据断言评估对话质量")
        self.client = None

        # 创建评分转换器
        conversion_config = config.get("conversion", {})
        self.score_converter = ScoreConverter(conversion_config)

        if not self.assertions:
            raise ValueError("自然语言断言列表不能为空")

        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供")

    async def initialize(self):
        """初始化OpenAI客户端"""
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"自然语言评分器初始化成功：{self.name}")
        except Exception as e:
            logger.error(f"自然语言评分器初始化失败：{e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def grade(self, transcript: List[Dict[str, Any]]) -> GradingResult:
        """
        对对话记录进行自然语言评分

        Args:
            transcript: 对话记录

        Returns:
            评分结果
        """
        if not self._initialized:
            await self.initialize()

        # 构建对话文本
        dialogue_text = self._format_transcript(transcript)

        # 构建评分Prompt
        prompt = self._build_grading_prompt(dialogue_text)

        try:
            # 调用LLM进行评分
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的对话质量评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                seed=42  # 固定种子确保可复现性
            )

            grading_text = response.choices[0].message.content.strip()

            # 解析评分结果
            result = self._parse_grading_result(grading_text, transcript)
            logger.info(f"自然语言评分完成：{self.name}，得分：{result.score:.2f}")

            return result

        except Exception as e:
            logger.error(f"自然语言评分失败：{e}")
            # 返回默认失败结果
            return GradingResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                reasoning=f"评分失败：{str(e)}",
                evidence=[]
            )

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        """格式化对话记录"""
        lines = []
        for i, turn in enumerate(transcript):
            # 支持字典和DialogueTurn对象
            if isinstance(turn, dict) or hasattr(turn, 'get'):
                # 字典或字典类对象
                speaker = turn.get("speaker", "unknown")
                message = turn.get("message", "")
            else:
                # 对象（如DialogueTurn）
                speaker = getattr(turn, "speaker", "unknown")
                message = getattr(turn, "message", "")
            lines.append(f"{i+1}. {speaker}: {message}")
        return "\n".join(lines)

    def _build_grading_prompt(self, dialogue_text: str) -> str:
        """构建评分Prompt，要求对每个断言进行True/False判断并给出理由"""
        assertions_text = "\n".join([f"{i+1}. {assertion}" for i, assertion in enumerate(self.assertions)])

        prompt = f"""请评估以下对话的质量，基于以下断言：

{assertions_text}

对话内容：
{dialogue_text}

请对每个断言进行独立判断，并为每个断言提供判断结果和简要理由。

请按以下格式回复：
1. 对每个断言的判断（格式：断言序号. True/False - 简要理由）：
{chr(10).join([f'   {i+1}. ' for i in range(len(self.assertions))])}
2. 总体评估（简要说明）：
3. 关键证据（引用对话中的具体内容支持你的判断）：

注意：每个断言必须独立判断，基于对话内容而不是主观感觉。为每个断言提供简短的判断理由。"""
        return prompt

    def _parse_grading_result(self, grading_text: str, transcript: List[Dict[str, Any]]) -> GradingResult:
        """解析评分结果文本，提取每个断言的真假判断并计算分数"""
        lines = grading_text.split("\n")

        # 解析每个断言的真假判断和理由
        assertion_results = {}
        assertion_reasons = {}
        reasoning = ""
        evidence_text = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            if "对每个断言的判断" in line or "断言判断" in line:
                current_section = "assertions"
                continue
            elif "总体评估" in line or "总体说明" in line:
                current_section = "reasoning"
                continue
            elif "关键证据" in line or "证据" in line:
                current_section = "evidence"
                continue

            # 解析断言判断（格式：序号. True/False - 理由）
            if current_section == "assertions" and "." in line:
                parts = line.split(".", 1)
                if len(parts) == 2:
                    try:
                        index = int(parts[0].strip()) - 1  # 转为0-based索引
                        if 0 <= index < len(self.assertions):
                            # 解析True/False和理由
                            rest = parts[1].strip()
                            # 尝试分割理由部分，可能的分隔符：-、–、:等
                            reason = ""
                            is_true = False

                            # 查找True/False关键词
                            rest_lower = rest.lower()
                            if "true" in rest_lower or "是" in rest_lower or "成立" in rest_lower or "✓" in rest_lower:
                                is_true = True
                                assertion_results[index] = True
                            elif "false" in rest_lower or "否" in rest_lower or "不成立" in rest_lower or "✗" in rest_lower:
                                is_true = False
                                assertion_results[index] = False
                            else:
                                # 如果没有明确的关键词，跳过这个断言
                                continue

                            # 提取理由部分（去除True/False关键词后的内容）
                            # 尝试用多种分隔符分割
                            separators = ["-", "–", ":", "—"]
                            reason_text = rest
                            for sep in separators:
                                if sep in rest:
                                    # 分割后，第一部分可能是True/False，第二部分是理由
                                    subparts = rest.split(sep, 1)
                                    if len(subparts) == 2:
                                        reason_text = subparts[1].strip()
                                        break

                            # 如果理由文本仍然包含True/False关键词，尝试进一步清理
                            reason_lower = reason_text.lower()
                            for keyword in ["true", "false", "是", "否", "成立", "不成立"]:
                                if keyword in reason_lower:
                                    # 移除关键词
                                    reason_text = reason_text.replace(keyword, "").strip()

                            # 清理标点符号
                            reason_text = reason_text.strip(":-–— ")
                            assertion_reasons[index] = reason_text if reason_text else f"断言{'成立' if is_true else '不成立'}"

                    except (ValueError, IndexError):
                        pass

            # 解析总体评估
            elif current_section == "reasoning" and not line.startswith("2."):
                if reasoning:
                    reasoning += " " + line
                else:
                    reasoning = line

            # 解析关键证据
            elif current_section == "evidence" and not line.startswith("3."):
                if evidence_text:
                    evidence_text += " " + line
                else:
                    evidence_text = line

        # 使用评分转换器计算分数和通过状态
        if assertion_results:
            # 使用转换器计算分数
            score, passed, conversion_details = self.score_converter.convert(
                assertion_results, len(self.assertions), reasoning
            )

            # 将转换详情添加到理由中
            if reasoning:
                reasoning += f" | 转换策略: {conversion_details.get('strategy', 'unknown')}"
            else:
                reasoning = f"转换策略: {conversion_details.get('strategy', 'unknown')}"

        else:
            # 如果没有解析到断言结果，使用回退逻辑
            score = 0.0
            passed = False
            if not reasoning:
                reasoning = "无法解析断言判断结果"
            conversion_details = {}

        # 创建各准则的详细结果
        criteria_results = {}
        for i in range(len(self.assertions)):
            is_true = assertion_results.get(i, False)
            criterion_name = f"assertion_{i+1}"
            criterion_description = self.assertions[i]
            criterion_reason = assertion_reasons.get(i, f"断言{'成立' if is_true else '不成立'}")
            criterion_score = 1.0 if is_true else 0.0
            criterion_passed = is_true

            # 获取权重
            criterion_weight = 1.0
            if hasattr(self.score_converter, 'assertion_weights'):
                criterion_weight = self.score_converter.assertion_weights.get(i, 1.0)

            criterion_result = CriterionResult(
                criterion_name=criterion_name,
                criterion_description=criterion_description,
                score=criterion_score,
                passed=criterion_passed,
                reasoning=criterion_reason,
                evidence=[],  # 可以为每个断言添加特定证据，但这里留空
                weight=criterion_weight
            )
            criteria_results[criterion_name] = criterion_result

        # 创建评分结果
        result = GradingResult(
            grader_name=self.name,
            score=score,
            passed=passed,
            reasoning=reasoning or "未提供详细评估",
            confidence=0.8,  # 默认置信度
            criteria_results=criteria_results
        )

        # 添加断言判断详情作为证据
        if assertion_results:
            # 使用转换器生成带策略信息的断言状态文本
            assertion_status_text = self.score_converter.get_assertion_status_text(
                assertion_results, self.assertions
            )

            result.add_evidence(
                evidence_type="assertion_judgments",
                content=assertion_status_text,
                relevance=1.0
            )

            # 添加转换详情作为额外证据
            if conversion_details:
                details_text = f"转换策略: {conversion_details.get('strategy', 'unknown')}\n"
                for key, value in conversion_details.items():
                    if key != 'strategy':
                        details_text += f"{key}: {value}\n"

                result.add_evidence(
                    evidence_type="conversion_details",
                    content=details_text.strip(),
                    relevance=0.8
                )

        # 添加关键证据
        if evidence_text:
            result.add_evidence(
                evidence_type="key_evidence",
                content=evidence_text,
                relevance=1.0
            )

        return result

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info(f"自然语言评分器已关闭：{self.name}")

    def __str__(self):
        return f"NaturalLanguageGrader(name={self.name}, assertions={len(self.assertions)})"