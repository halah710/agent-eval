"""
简单评测器实现
"""

import asyncio
import random
from typing import List, Dict, Any, Optional
from loguru import logger

from .base import Evaluator, Scenario, EvaluationResult, DialogueContext
from ..agents.base import NPCAgent
from ..graders.base import Grader
from ..players.base import PlayerSimulator
from ..players.factory import PlayerFactory


class SimpleEvaluator(Evaluator):
    """简单评测器"""

    def __init__(
        self,
        npc_agent: NPCAgent,
        graders: List[Grader],
        config: Dict[str, Any] = None
    ):
        """
        初始化简单评测器

        Args:
            npc_agent: NPC Agent实例
            graders: 评分器列表
            config: 配置参数，包括：
                - max_turns: 最大对话轮次（默认：5）
                - turn_delay: 每轮延迟（秒，默认：0.1）
                - timeout: 超时时间（秒，默认：30）
                - player_simulator: 玩家模拟器配置
        """
        super().__init__(npc_agent, graders, config)
        self.max_turns = int(self.config.get("max_turns", 5))
        self.turn_delay = float(self.config.get("turn_delay", 0.1))
        self.timeout = float(self.config.get("timeout", 30.0))

        # 创建玩家模拟器
        player_config = self.config.get("player_simulator", {})
        if isinstance(player_config, PlayerSimulator):
            # 如果已经提供了玩家模拟器实例，直接使用
            self.player_simulator = player_config
        else:
            # 如果是配置字典，通过PlayerFactory创建
            self.player_simulator = PlayerFactory.create_player(player_config)

        # 初始化Agent和评分器
        self._initialized = False

    async def initialize(self):
        """初始化评测器"""
        if self._initialized:
            return

        try:
            # 初始化NPC Agent
            await self.npc_agent.initialize()

            # 初始化所有评分器
            for grader in self.graders.values():
                await grader.initialize()

            # 初始化玩家模拟器
            await self.player_simulator.initialize()

            self._initialized = True
            logger.info(f"简单评测器初始化成功，NPC：{self.npc_agent}，评分器：{len(self.graders)}个")
        except Exception as e:
            logger.error(f"简单评测器初始化失败：{e}")
            raise

    async def evaluate_scenario(self, scenario: Scenario) -> EvaluationResult:
        """
        评测单个场景

        Args:
            scenario: 测试场景

        Returns:
            评测结果
        """
        if not self._initialized:
            await self.initialize()

        # 创建评测结果
        result = EvaluationResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            metadata={
                "npc_role": self.npc_agent.role.model_dump() if self.npc_agent.role else {},
                "agent_type": self.npc_agent.__class__.__name__,
                "scenario_type": scenario.scenario_type.value if hasattr(scenario.scenario_type, 'value') else str(scenario.scenario_type)
            }
        )

        try:
            # 初始化对话上下文
            context = self._create_dialogue_context()

            # 设置玩家状态
            self.player_simulator.set_profile(scenario.player_profile)

            # 开始对话
            current_turn = 0
            player_input = scenario.initial_prompt

            while current_turn < self.max_turns:
                # 记录玩家输入
                result.add_turn("player", player_input, {"turn": current_turn})
                context.add_turn("player", player_input)

                # NPC回复
                try:
                    npc_reply = await asyncio.wait_for(
                        self.npc_agent.respond(player_input, context),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    npc_reply = "（超时无响应）"
                    result.errors.append(f"第{current_turn}轮NPC响应超时")
                except Exception as e:
                    npc_reply = f"（错误：{str(e)}）"
                    result.errors.append(f"第{current_turn}轮NPC响应错误：{e}")

                # 记录NPC回复
                result.add_turn("npc", npc_reply, {"turn": current_turn})
                context.add_turn("npc", npc_reply)

                # 检查对话是否应该结束
                if self._should_end_dialogue(context, current_turn):
                    logger.debug(f"对话在第{current_turn}轮结束")
                    break

                # 生成下一轮玩家输入
                try:
                    next_input = await asyncio.wait_for(
                        self.player_simulator.generate_response(npc_reply, context),
                        timeout=self.timeout
                    )
                    player_input = next_input
                except asyncio.TimeoutError:
                    player_input = "（玩家模拟器超时）"
                    result.errors.append(f"第{current_turn}轮玩家模拟器超时")
                    break
                except Exception as e:
                    player_input = f"（玩家模拟器错误：{str(e)}）"
                    result.errors.append(f"第{current_turn}轮玩家模拟器错误：{e}")
                    break

                # 轮次延迟（模拟真实对话）
                if self.turn_delay > 0:
                    await asyncio.sleep(self.turn_delay)

                current_turn += 1

            # 对话结束，进行评分
            await self._perform_grading(result)

            # 计算最终得分
            result.calculate_final_score()

            # 设置结束时间
            result.end_time = time.time()

            logger.info(f"场景评测完成：{scenario.name}，得分：{result.final_score:.2f}，通过：{result.passed}")

        except Exception as e:
            logger.error(f"场景评测失败：{scenario.name}，错误：{e}")
            result.errors.append(f"评测过程错误：{e}")
            result.passed = False

        return result

    async def _perform_grading(self, result: EvaluationResult):
        """执行评分"""
        for grader_name, grader in self.graders.items():
            try:
                grading_result = await grader.grade(result.transcript)
                result.add_grading_result(grader_name, grading_result)
                logger.debug(f"评分器 {grader_name} 完成评分，得分：{grading_result.score:.2f}")
            except Exception as e:
                logger.error(f"评分器 {grader_name} 评分失败：{e}")
                # 添加默认失败结果
                result.add_grading_result(grader_name, GradingResult(
                    grader_name=grader_name,
                    score=0.0,
                    passed=False,
                    reasoning=f"评分失败：{str(e)}",
                    evidence=[]
                ))

    def _should_end_dialogue(self, context: DialogueContext, current_turn: int) -> bool:
        """判断对话是否应该结束"""
        # 基于轮次检查
        if current_turn >= self.max_turns - 1:
            return True

        # 检查最后几轮对话是否包含结束信号
        recent_turns = context.turns[-3:] if len(context.turns) >= 3 else context.turns
        end_signals = ["再见", "拜拜", "结束", "完了", "好了", "谢谢", "感谢"]

        for turn in recent_turns:
            # 支持字典和对象类型
            if isinstance(turn, dict) or hasattr(turn, 'get'):
                message = turn.get("message", "")
            else:
                message = getattr(turn, "message", "")
            message = message.lower()
            for signal in end_signals:
                if signal in message:
                    return True

        return False

    async def close(self):
        """清理资源"""
        try:
            # 关闭NPC Agent
            await self.npc_agent.close()

            # 关闭所有评分器
            for grader in self.graders.values():
                await grader.close()

            # 关闭玩家模拟器
            await self.player_simulator.close()

            self._initialized = False
            logger.info("简单评测器已关闭")
        except Exception as e:
            logger.error(f"关闭评测器时出错：{e}")

    def __str__(self):
        return f"SimpleEvaluator(npc={self.npc_agent}, graders={len(self.graders)})"


# 导入time模块
import time