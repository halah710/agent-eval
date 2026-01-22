"""
角色化玩家模拟器
支持复杂的角色扮演和多样化对话策略
"""

import random
from typing import List, Dict, Any
from loguru import logger

from .base import PlayerSimulator, PlayerProfile, DialogueContext, DialogueAction, DialogueActionType


class RolePlayer(PlayerSimulator):
    """角色化玩家模拟器，支持复杂的角色扮演"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化角色化玩家模拟器

        Args:
            config: 配置参数，包括：
                - role_templates: 角色模板列表
                - conversation_goals: 对话目标列表
                - emotional_range: 情感范围配置
                - topic_knowledge: 话题知识库
        """
        super().__init__(config)
        self.role_templates = config.get("role_templates", [])
        self.conversation_goals = config.get("conversation_goals", [])
        self.emotional_range = config.get("emotional_range", {
            "happy": 0.3,
            "neutral": 0.4,
            "sad": 0.2,
            "angry": 0.1
        })
        self.topic_knowledge = config.get("topic_knowledge", {})
        self.current_goal = None
        self.current_emotion = "neutral"
        self.conversation_history = []

    async def initialize(self):
        """初始化角色化玩家模拟器"""
        self._initialized = True
        if self.profile:
            logger.info(f"角色化玩家模拟器初始化成功，角色：{self.profile.name}")
            # 根据角色性格初始化情感状态
            self._initialize_emotion_state()
        else:
            logger.info("角色化玩家模拟器初始化成功（未设置角色）")

    def _initialize_emotion_state(self):
        """根据角色性格初始化情感状态"""
        if not self.profile:
            return

        personality = self.profile.personality
        if personality == "aggressive":
            self.current_emotion = random.choice(["angry", "neutral"])
        elif personality == "shy":
            self.current_emotion = random.choice(["shy", "neutral"])
        elif personality == "curious":
            self.current_emotion = "happy"
        elif personality == "sarcastic":
            self.current_emotion = "neutral"  # 讽刺可能隐藏真实情感

    async def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
        """
        生成角色化玩家回复

        Args:
            npc_reply: NPC的回复
            context: 对话上下文

        Returns:
            角色化玩家回复
        """
        if not self._initialized:
            await self.initialize()

        # 更新对话历史
        self.conversation_history.append({
            "speaker": "npc",
            "message": npc_reply,
            "turn": len(self.conversation_history)
        })

        # 分析NPC回复并更新情感状态
        self._update_emotion_state(npc_reply)

        # 决定下一步动作
        action = await self.decide_next_action(context)

        # 生成基于角色和情感的回复
        response = self._generate_role_based_response(action, npc_reply, context)

        # 记录到上下文
        context.turns.append({
            "speaker": "player",
            "message": response,
            "action": action.model_dump(),
            "emotion": self.current_emotion
        })

        # 更新对话历史
        self.conversation_history.append({
            "speaker": "player",
            "message": response,
            "turn": len(self.conversation_history)
        })

        logger.debug(f"角色化玩家回复（情感：{self.current_emotion}）：{response}")
        return response

    async def decide_next_action(self, context: DialogueContext) -> DialogueAction:
        """
        决定下一步对话动作

        Args:
            context: 对话上下文

        Returns:
            对话动作
        """
        if not self.profile:
            # 如果没有设置角色，使用简单策略
            return DialogueAction(
                action_type=DialogueActionType.CONTINUE,
                content="继续对话"
            )

        # 根据角色性格和当前情感决定动作
        personality = self.profile.personality
        conversation_length = len(self.conversation_history)

        # 基础概率分布
        if personality == "aggressive":
            # 攻击性角色更可能挑战和切换话题
            if random.random() < 0.3:
                return DialogueAction(
                    action_type=DialogueActionType.CHANGE_TOPIC,
                    content="切换话题",
                    target_topic=random.choice(["观点", "质疑", "挑战"])
                )
            elif random.random() < 0.4:
                return DialogueAction(
                    action_type=DialogueActionType.ASK_QUESTION,
                    content="提问挑战"
                )
        elif personality == "curious":
            # 好奇角色更可能提问
            if random.random() < 0.5:
                return DialogueAction(
                    action_type=DialogueActionType.ASK_QUESTION,
                    content="好奇提问"
                )
        elif personality == "shy":
            # 害羞角色更可能结束对话或简单回应
            if conversation_length >= 3 and random.random() < 0.4:
                return DialogueAction(
                    action_type=DialogueActionType.END_CONVERSATION,
                    content="害羞结束"
                )

        # 根据情感状态决定动作
        if self.current_emotion == "angry" and random.random() < 0.6:
            return DialogueAction(
                action_type=DialogueActionType.EXPRESS_EMOTION,
                content="表达愤怒",
                emotion="angry"
            )
        elif self.current_emotion == "sad" and random.random() < 0.5:
            return DialogueAction(
                action_type=DialogueActionType.EXPRESS_EMOTION,
                content="表达悲伤",
                emotion="sad"
            )

        # 默认继续对话
        return DialogueAction(
            action_type=DialogueActionType.CONTINUE,
            content="继续对话"
        )

    def _generate_role_based_response(self, action: DialogueAction, npc_reply: str, context: DialogueContext) -> str:
        """生成基于角色和情感的回复"""
        if not self.profile:
            return "我明白了。"

        # 基础回复模板
        base_responses = {
            DialogueActionType.CONTINUE: [
                "原来如此。",
                "我明白了。",
                "这很有趣。",
                "请继续说。"
            ],
            DialogueActionType.CHANGE_TOPIC: [
                "说到{target_topic}，我想知道...",
                "换个话题，关于{target_topic}你怎么看？",
                "我其实更关心{target_topic}。"
            ],
            DialogueActionType.ASK_QUESTION: [
                "你能详细解释一下吗？",
                "为什么你会这么想？",
                "这有什么特别的含义吗？",
                "还有其他类似的例子吗？"
            ],
            DialogueActionType.EXPRESS_EMOTION: {
                "happy": ["太好了！", "真让人高兴！", "太棒了！"],
                "sad": ["真让人难过...", "这太遗憾了。", "我心情有点低落。"],
                "angry": ["这让我很生气！", "我不喜欢这样！", "太过分了！"],
                "surprised": ["真的吗？", "没想到会这样！", "太意外了！"]
            },
            DialogueActionType.END_CONVERSATION: [
                "我得走了，下次再聊。",
                "谢谢你的时间，再见。",
                "今天就到这里吧，再见。"
            ]
        }

        # 获取基础回复
        if action.action_type == DialogueActionType.EXPRESS_EMOTION:
            emotion_responses = base_responses[action.action_type].get(action.emotion, ["我明白了。"])
            base_response = random.choice(emotion_responses)
        elif action.action_type == DialogueActionType.CHANGE_TOPIC:
            template = random.choice(base_responses[action.action_type])
            base_response = template.format(target_topic=action.target_topic or "这个话题")
        else:
            base_response = random.choice(base_responses.get(action.action_type, ["我明白了。"]))

        # 根据角色性格调整回复
        response = self._apply_personality_style(base_response)

        # 根据情感状态调整语气
        response = self._apply_emotion_tone(response)

        return response

    def _apply_personality_style(self, response: str) -> str:
        """应用角色性格风格"""
        if not self.profile:
            return response

        personality = self.profile.personality
        if personality == "aggressive":
            return response + " 不过我觉得可以更直接一点。"
        elif personality == "shy":
            return "嗯... " + response.lower()
        elif personality == "curious":
            return response + " 你能告诉我更多吗？"
        elif personality == "sarcastic":
            return response + " 当然，如果你说的是真的话。"
        elif personality == "supportive":
            return response + " 我理解你的想法。"

        return response

    def _apply_emotion_tone(self, response: str) -> str:
        """应用情感语调"""
        if self.current_emotion == "angry":
            return response.upper() + "！"
        elif self.current_emotion == "sad":
            return response + " ..."
        elif self.current_emotion == "happy":
            return response + " 😊"
        elif self.current_emotion == "excited":
            return response + "！"

        return response

    def _update_emotion_state(self, npc_reply: str):
        """根据NPC回复更新情感状态"""
        if not self.profile:
            return

        # 简单的情感分析逻辑
        positive_keywords = ["好", "高兴", "喜欢", "感谢", "帮助", "理解"]
        negative_keywords = ["不好", "生气", "讨厌", "反对", "错误", "问题"]

        reply_lower = npc_reply.lower()

        # 检查关键词
        positive_count = sum(1 for word in positive_keywords if word in reply_lower)
        negative_count = sum(1 for word in negative_keywords if word in reply_lower)

        if positive_count > negative_count:
            # 向积极情感转移
            if self.current_emotion in ["sad", "angry"]:
                self.current_emotion = "neutral"
            elif self.current_emotion == "neutral":
                self.current_emotion = "happy"
        elif negative_count > positive_count:
            # 向消极情感转移
            if self.current_emotion in ["happy", "neutral"]:
                self.current_emotion = "sad"
            elif self.current_emotion == "sad" and random.random() < 0.3:
                self.current_emotion = "angry"

    async def close(self):
        """清理资源"""
        self._initialized = False
        logger.info("角色化玩家模拟器已关闭")

    def __str__(self):
        if self.profile:
            return f"RolePlayer(name={self.profile.name}, personality={self.profile.personality}, emotion={self.current_emotion})"
        return "RolePlayer(未设置角色)"