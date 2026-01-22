# AI Agent评测系统 - 游戏NPC对话评测框架

基于Anthropic博客《Demystifying evals for AI agents》实现的一套对话类AI Agent评测系统，专门用于评估游戏NPC对话系统的表现。

## 📋 项目概述

本项目实现了一个完整的AI Agent评测系统，支持：

- **多轮对话评测**：模拟玩家与NPC进行真实多样的对话
- **自动化评分**：基于自然语言断言、代码规则等多种评分器
- **测试用例管理**：支持多种对话场景定义和管理
- **结果可复现**：全链路随机种子控制，确保评测结果可复现
- **系统解耦**：评测系统与被测NPC完全解耦，支持多种NPC实现
- **Capability vs Regression Eval**：支持能力评测和回归评测双模式

## 🎯 核心特性

### 1. 真实多样的玩家模拟器
- **角色化玩家系统**：支持不同性格、背景、对话风格的玩家
- **多样化对话策略**：日常聊天、情感倾诉、观点冲突等多种对话模式
- **情感状态模拟**：支持不同情感状态的对话交互

### 2. 多类型评分器系统（基于Anthropic博客）
- **自然语言断言评分器**（Model-based Grader）：基于LLM对每个断言进行True/False判断，然后通过代码逻辑转换为分数
- **代码规则评分器**（Code-based Grader）：基于正则表达式和规则的客观评分
- **灵活的评分转换策略**：支持简单比例、加权平均、必选断言、阈值判定等多种转换策略
- **多评判者投票机制**：支持多个评分器并行评估和投票
- **插件化架构**：易于扩展新的评分器类型和转换策略

### 3. 评测结果可复现性
- **全链路随机种子控制**：从对话生成到评分全程确定性
- **完整状态保存**：保存随机种子、配置、中间状态
- **干净环境隔离**：每次评测在独立环境中运行

### 4. 完整对话记录管理
- **结构化记录**：时间戳、说话者、内容、元数据完整保存
- **持久化存储**：JSON格式存储，便于分析和调试
- **可视化展示**：支持对话记录的可视化查看

### 5. 参考解决方案验证
- **任务可解性证明**：为每个测试场景提供可行的参考解决方案
- **明确评分标准**：明确定义评分标准和成功要素
- **任务设计验证**：确保任务定义明确且可解决

### 6. Capability Eval vs Regression Eval
- **双模式评测**：支持能力评测（测试能力边界）和回归评测（防止功能倒退）
- **独立评测套件**：两种评测模式的独立管理和运行
- **差异化阈值**：能力评测通过阈值较低，回归评测通过阈值较高

## 🏗️ 系统架构

```
agent-eval/
├── src/
│   ├── agents/          # NPC Agent实现
│   │   ├── base.py     # NPC Agent基类
│   │   ├── openai_agent.py    # OpenAI Agent
│   │   ├── anthropic_agent.py # Anthropic Agent
│   │   ├── mock_agent.py      # 模拟Agent（测试用）
│   │   └── factory.py         # Agent工厂
│   ├── evaluator/       # 评测器模块
│   │   ├── base.py     # 评测器基类
│   │   ├── simple_evaluator.py    # 简单评测器
│   │   └── multi_evaluator.py     # 多任务评测器
│   ├── graders/         # 评分器模块
│   │   ├── base.py     # 评分器基类
│   │   ├── natural_language_grader.py   # 自然语言断言评分器（True/False判断）
│   │   ├── score_converter.py           # 评分转换器（多种转换策略）
│   │   ├── rule_based_grader.py         # 代码规则评分器
│   │   ├── hybrid_grader.py             # 混合评分器
│   │   ├── multi_judge_grader.py        # 多评判者评分器
│   │   └── factory.py                   # 评分器工厂
│   ├── players/         # 玩家模拟器模块
│   │   ├── base.py     # 玩家模拟器基类
│   │   ├── simple_player.py    # 简单玩家模拟器
│   │   ├── role_player.py      # 角色化玩家模拟器
│   │   ├── llm_player.py       # LLM玩家模拟器
│   │   └── factory.py          # 玩家模拟器工厂
│   └── utils/          # 工具模块
│       └── random_state.py    # 随机状态管理
├── configs/            # 配置文件
│   ├── npc_config.yaml        # NPC配置
│   ├── eval_config.yaml       # 评测系统配置
│   └── scenarios/             # 测试场景定义
│       ├── daily_chat.yaml           # 日常聊天场景
│       ├── emotional_support.yaml    # 情感支持场景
│       ├── opinion_conflict.yaml     # 观点冲突场景
│       ├── basic_greeting.yaml       # 基础问候场景（回归测试）
│       └── simple_qa.yaml            # 简单问答场景（回归测试）
├── examples/           # 示例代码
│   ├── run_eval.py           # 运行评测示例
│   └── sample_report.json    # 示例评测报告
├── tests/              # 单元测试
├── docs/              # 文档
│   ├── DESIGN.md           # 设计决策文档
│   └── API.md             # API文档
├── pyproject.toml     # 项目配置
├── .env.example       # 环境变量示例
└── README.md          # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/halah710/agent-eval-system.git
cd agent-eval

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 2. 配置环境变量

复制环境变量示例文件并配置你的API密钥：

```bash
cp .env.example .env
# 编辑.env文件，设置你的API密钥
```

### 3. 运行示例评测

```bash
# 运行单个场景评测
python examples/run_eval.py --scenario daily_chat

# 运行整个评测套件
python examples/run_eval.py --suite capability

# 查看帮助
python examples/run_eval.py --help
```

### 4. 查看评测结果

评测结果保存在 `outputs/` 目录下，包含：
- `report.json`：结构化评测报告
- `transcript.json`：完整对话记录
- `summary.txt`：评测摘要

## 📖 详细使用指南

### 1. 配置NPC Agent

编辑 `configs/npc_config.yaml` 配置NPC角色和Agent：

```yaml
npc_roles:
  friendly_npc:
    name: "友好的村民"
    personality: "热情友好，乐于助人"
    background: "在这个村庄生活了30年"
    speaking_style: "亲切、温暖"
    values: ["互助", "诚实", "善良"]

agent_configs:
  openai_agent:
    type: "openai"
    role: "{{npc_roles.friendly_npc}}"
    config:
      model: "gpt-4o-mini"
      temperature: 0.0
```

### 2. 配置评测系统

编辑 `configs/eval_config.yaml` 配置评测参数：

```yaml
graders:
  # 角色一致性评分器 - 使用简单比例转换策略
  character_consistency:
    type: "natural_language"
    config:
      assertions:
        - "NPC的回复符合其角色设定"
        - "NPC的说话风格保持一致"
        - "NPC的价值观在对话中得到体现"
        - "NPC没有出现角色矛盾的行为或言论"
      model: "gpt-4o-mini"
      temperature: 0.0
      conversion:
        strategy: "simple_ratio"  # 简单比例转换：True的比例作为分数
        pass_threshold: 0.75      # 通过阈值75%

  # 互动质量评分器 - 使用加权平均转换策略
  interaction_quality:
    type: "natural_language"
    config:
      assertions:
        - "对话自然流畅"
        - "NPC的回应与玩家输入相关"
        - "NPC能够理解玩家的意图"
        - "对话能够持续进行，不会突然中断"
        - "NPC的回应有助于推动对话发展"
      model: "gpt-4o-mini"
      temperature: 0.0
      conversion:
        strategy: "weighted_average"  # 加权平均转换策略
        pass_threshold: 0.7           # 通过阈值70%
        assertion_weights: [1.0, 1.2, 1.5, 0.8, 1.0]  # 每个断言的权重

test_suites:
  capability_suite:
    name: "能力评测套件"
    suite_type: "capability"
    scenarios:
      - "scenarios/daily_chat.yaml"
```

### 3. 创建测试场景

在 `configs/scenarios/` 目录下创建YAML格式的测试场景：

```yaml
id: "my_scenario"
name: "我的测试场景"
description: "场景描述"
scenario_type: "daily_chat"
player_profile:
  name: "测试玩家"
  personality: "friendly"
initial_prompt: "你好！"
max_turns: 5
expected_outcomes:
  - "NPC能够友好回应"
```

### 4. 编程接口使用

```python
import asyncio
from src.agents.factory import AgentFactory
from src.evaluator.simple_evaluator import SimpleEvaluator
from src.graders.factory import GraderFactory

# 创建NPC Agent
agent = AgentFactory.create_agent_from_config({
    "type": "mock",
    "role": {"name": "测试NPC", "personality": "友好"},
    "config": {}
})

# 创建评分器（包含转换策略配置）
graders = [
    GraderFactory.create_grader("character_consistency", {
        "type": "natural_language",
        "assertions": [
            "NPC的回复符合其角色设定",
            "NPC的说话风格保持一致",
            "NPC的价值观在对话中得到体现"
        ],
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "conversion": {
            "strategy": "simple_ratio",  # 简单比例转换策略
            "pass_threshold": 0.75       # 通过阈值75%
        }
    })
]

# 创建评测器
evaluator = SimpleEvaluator(agent, graders)

# 运行评测
async def run_evaluation():
    scenario = Scenario(
        id="test_scenario",
        name="测试场景",
        description="测试描述",
        scenario_type="daily_chat",
        player_profile={},
        initial_prompt="你好！",
        max_turns=3
    )

    result = await evaluator.evaluate_scenario(scenario)
    print(f"评测得分：{result.final_score:.2f}")
    print(f"是否通过：{result.passed}")

asyncio.run(run_evaluation())
```

## 🧪 评测维度

### 1. 角色一致性 (Character Consistency)
- **评估内容**：NPC的回复是否符合人设？口吻、态度、价值观是否自洽？
- **评分器**：自然语言断言评分器 + 代码规则评分器
- **关键指标**：性格一致性、说话风格一致性、价值观体现

### 2. 互动质量 (Interaction Quality)
- **评估内容**：对话是否自然流畅？是否能让玩家想继续聊下去？
- **评分器**：自然语言断言评分器
- **关键指标**：对话流畅性、相关性、理解能力、对话推动力

### 3. 基础规则检查 (Basic Rules)
- **评估内容**：是否违反基本对话规则？如无脏话、回复长度合理等
- **评分器**：代码规则评分器
- **关键指标**：内容安全性、回复质量、基本礼仪

## 🔄 评分转换逻辑

自然语言断言评分器采用了**分层评分架构**，将LLM的主观判断与代码的客观计算相结合：

### 工作流程
1. **LLM进行True/False判断**：LLM对每个自然语言断言进行独立的真伪判断
2. **代码逻辑转换分数**：使用`ScoreConverter`将断言结果转换为0-1的分数
3. **阈值判定通过状态**：根据配置的通过阈值判断是否通过

### 支持的转换策略
- **简单比例** (`simple_ratio`)：True断言的比例作为分数
- **加权平均** (`weighted_average`)：每个断言有权重，计算加权平均分
- **必选断言** (`must_pass`)：某些断言必须通过，否则直接不通过
- **阈值判定** (`threshold_based`)：使用两个阈值（通过阈值和严格阈值）

### 配置示例
```yaml
conversion:
  strategy: "weighted_average"  # 转换策略
  pass_threshold: 0.7           # 通过阈值
  assertion_weights: [1.0, 1.2, 1.5]  # 断言权重（仅weighted_average需要）
  must_pass_assertions: [0, 2]  # 必选断言索引（仅must_pass需要）
  strict_threshold: 0.9         # 严格阈值（仅threshold_based需要）
```

## 🔧 扩展开发

### 1. 添加新的评分器

```python
from src.graders.base import Grader, GradingResult

class CustomGrader(Grader):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    async def grade(self, transcript) -> GradingResult:
        # 实现你的评分逻辑
        score = self._calculate_score(transcript)
        return GradingResult(
            grader_name=self.name,
            score=score,
            passed=score >= 0.7,
            reasoning="自定义评分逻辑",
            evidence=[]
        )
```

### 2. 添加新的玩家模拟器

```python
from src.players.base import PlayerSimulator

class CustomPlayer(PlayerSimulator):
    def __init__(self, config: dict):
        super().__init__(config)

    async def generate_response(self, npc_reply, context) -> str:
        # 实现你的玩家回复生成逻辑
        return "自定义玩家回复"
```

### 3. 添加新的NPC Agent

```python
from src.agents.base import NPCAgent

class CustomAgent(NPCAgent):
    def __init__(self, role, config):
        super().__init__(role, config)

    async def respond(self, player_input, context) -> str:
        # 实现你的NPC回复逻辑
        return "自定义NPC回复"
```

## 📊 评测报告示例

```json
{
  "scenario_id": "daily_chat_001",
  "scenario_name": "日常问候聊天",
  "final_score": 0.85,
  "passed": true,
  "grading_results": {
    "character_consistency": {
      "score": 0.9,
      "passed": true,
      "reasoning": "NPC始终保持友好态度，说话风格一致",
      "evidence": ["第2轮：'你好！是的，我在这里生活了很多年。'"]
    },
    "interaction_quality": {
      "score": 0.8,
      "passed": true,
      "reasoning": "对话自然流畅，NPC能够理解玩家意图",
      "evidence": ["第3轮：主动询问玩家是否是第一次来村"]
    }
  },
  "transcript": [
    {"speaker": "player", "message": "你好！今天天气真不错..."},
    {"speaker": "npc", "message": "你好！是的，我在这里生活了很多年..."}
  ]
}
```

## 🎯 基于Anthropic博客的最佳实践

本项目严格遵循Anthropic博客《Demystifying evals for AI agents》中的最佳实践：

### 1. 明确的任务设计
- 每个任务都有明确的输入和成功标准
- 提供参考解决方案证明任务可解
- 避免模糊或不可解的任务定义

### 2. 稳健的评分器设计
- **分层评分架构**：LLM进行True/False断言判断 + 代码逻辑进行分数转换
- **多种转换策略**：支持简单比例、加权平均、必选断言、阈值判定等多种评分转换方式
- **结合多种评分器类型**（自然语言断言 + 代码规则）
- **支持多评判者投票机制**
- **评分结果有具体证据支持**：包括每个断言的判断结果和转换详情

### 3. 可复现的评测环境
- 全链路随机种子控制
- 确定性LLM参数配置
- 干净环境隔离运行

### 4. 平衡的评测套件
- 区分能力评测和回归评测
- 测试行为"应该发生"和"不应该发生"两种情况
- 避免类别不平衡的评测

### 5. 完整的对话记录
- 保存完整的对话transcript
- 便于问题定位和调试
- 支持结果分析和优化

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于MIT许可证开源。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢Anthropic的博客《Demystifying evals for AI agents》提供了评测AI Agent的核心概念和最佳实践
- 感谢所有开源项目的贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至：2629654745@qq.com

---

**祝您评测愉快！** 🎮🤖