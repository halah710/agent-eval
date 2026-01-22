"""
简单玩家模拟器
"""

import random
from typing import List, Dict, Any
from loguru import logger

from .base import PlayerSimulator, PlayerProfile, DialogueContext, DialogueAction, DialogueActionType


class SimplePlayer(PlayerSimulator):
    """简单玩家模拟器，基于规则的回复生成"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化简单玩家模拟器

        Args:
            config: 配置参数，包括：
                - response_templates: 回复模板列表
                - topic_change_probability: 切换话题概率（默认：0.2）
                - question_probability: 提问概率（默认：0.3）
        """
        super().__init__(config)
        self.response_templates = config.get("response_templates", [
            "我明白了。",
            "这很有趣。",
            "你能详细说说吗？",
            "我不太确定。",
            "原来如此。",
            "这让我想起了...",
            "你怎么看？",
            "继续说吧。"
        ])
        self.topic_change_probability = float(config.get("topic_change_probability", 0.2))
        self.question_probability = float(config.get("question_probability", 0.3))
        self.topics = config.get("topics", [
            "天气", "爱好", "工作", "家庭", "旅行", "美食", "电影", "音乐"
        ])
        self.current_topic_index = 0

    async def initialize(self):
        """初始化简单玩家模拟器"""
        self._initialized = True
        if self.profile:
            logger.info(f"简单玩家模拟器初始化成功，角色：{self.profile.name}")
        else:
            logger.info("简单玩家模拟器初始化成功（未设置角色）")

    async def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
        """
        生成玩家回复

        Args:
            npc_reply: NPC的回复
            context: 对话上下文

        Returns:
            玩家回复内容
        """
        if not self._initialized:
            await self.initialize()

        # 决定下一步动作
        action = await self.decide_next_action(context)

        # 根据动作生成回复
        if action.action_type == DialogueActionType.CHANGE_TOPIC:
            # 切换话题
            response = f"说到{action.target_topic}，{self._get_random_response()}"
        elif action.action_type == DialogueActionType.ASK_QUESTION:
            # 提问
            questions = [
                "你为什么这么想？",
                "你能举个例子吗？",
                "这有什么好处？",
                "还有其他选择吗？",
                "你怎么知道的？"
            ]
            response = f"{self._get_random_response()} {random.choice(questions)}"
        elif action.action_type == DialogueActionType.EXPRESS_EMOTION:
            # 表达情感
            emotions = {
                "happy": ["太好了！", "真让人高兴！", "太棒了！"],
                "sad": ["真遗憾。", "这让人难过。", "唉..."],
                "surprised": ["真的吗？", "没想到！", "太意外了！"],
                "angry": ["这让我生气。", "我不喜欢这样。", "太过分了！"]
            }
            emotion_responses = emotions.get(action.emotion, ["我明白了。"])
            response = f"{self._get_random_response()} {random.choice(emotion_responses)}"
        elif action.action_type == DialogueActionType.END_CONVERSATION:
            # 结束对话
            farewells = ["我得走了，下次再聊。", "谢谢你的时间，再见。", "聊得很开心，再见。"]
            response = random.choice(farewells)
        else:
            # 继续当前话题
            response = self._get_random_response()

        # 根据性格调整回复
        if self.profile:
            response = self._get_personality_based_response(response)

        # 记录到上下文
        context.turns.append({
            "speaker": "player",
            "message": response,
            "action": action.model_dump()
        })

        logger.debug(f"简单玩家回复：{response}")
        return response

    async def decide_next_action(self, context: DialogueContext) -> DialogueAction:
        """
        决定下一步对话动作

        Args:
            context: 对话上下文

        Returns:
            对话动作
        """
        # 计算动作概率
        rand_val = random.random()

        if len(context.turns) >= 5 and rand_val < 0.1:
            # 10%概率在5轮后结束对话
            return DialogueAction(
                action_type=DialogueActionType.END_CONVERSATION,
                content="结束对话"
            )
        elif rand_val < self.topic_change_probability:
            # 切换话题
            self.current_topic_index = (self.current_topic_index + 1) % len(self.topics)
            return DialogueAction(
                action_type=DialogueActionType.CHANGE_TOPIC,
                content="切换话题",
                target_topic=self.topics[self.current_topic_index]
            )
        elif rand_val < self.topic_change_probability + self.question_probability:
            # 提问
            return DialogueAction(
                action_type=DialogueActionType.ASK_QUESTION,
                content="提问"
            )
        elif self.profile and self.profile.emotional_state != "neutral":
            # 表达情感（如果情感状态不是中性）
            return DialogueAction(
                action_type=DialogueActionType.EXPRESS_EMOTION,
                content="表达情感",
                emotion=self.profile.emotional_state
            )
        else:
            # 继续当前话题
            return DialogueAction(
                action_type=DialogueActionType.CONTINUE,
                content="继续对话"
            )

    def _get_random_response(self) -> str:
        """获取随机回复模板"""
        return random.choice(self.response_templates)

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info("简单玩家模拟器已关闭")

    def __str__(self):
        if self.profile:
            return f"SimplePlayer(name={self.profile.name}, personality={self.profile.personality})"
        return "SimplePlayer(未设置角色)"