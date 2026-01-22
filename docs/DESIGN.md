# AI Agent评测系统设计文档

## 项目概述

AI Agent评测系统是一个专门用于评估游戏NPC对话系统表现的框架。系统支持多轮对话评测、自动化评分、测试用例管理和结果分析。

### 核心目标
1. 提供一个标准化的NPC对话评测框架
2. 支持真实多样的玩家模拟
3. 实现多种评分器类型（基于自然语言断言、代码规则、多评判者投票等）
4. 确保评测结果的可复现性
5. 提供完整的对话记录管理和结果分析

## 系统架构

### 整体架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  测试用例管理   │    │   评测执行器    │    │   结果分析器    │
│  Test Suite     │───▶│  Evaluator      │───▶│  Analyzer       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  测试场景定义   │    │     NPC代理     │    │  评测报告生成   │
│  Scenarios      │    │     NPC Agent   │    │  Report         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │  LLM API接口层  │
                      │  LLM API Layer  │
                      └─────────────────┘
```

### 核心组件

#### 1. NPC Agent（被测系统）
- **职责**: 模拟游戏NPC对话行为
- **接口**: 统一的对话接口，支持多轮对话
- **实现**: 支持多种LLM API（OpenAI, Anthropic, 本地模型）
- **配置**: 通过YAML配置文件定义角色属性和对话风格

#### 2. 玩家模拟器 (Player Simulator)
- **职责**: 模拟真实玩家与NPC进行对话
- **功能**:
  - 多样化玩家角色（不同性格、背景、对话风格）
  - 多种对话策略（日常聊天、情感倾诉、观点冲突等）
  - 情感状态模拟和话题切换
- **实现**: 基于规则的对话生成，支持随机性和可控性

#### 3. 评测执行器 (Evaluator)
- **职责**: 管理整个评测流程
- **功能**:
  - 加载测试场景和配置
  - 协调NPC Agent和玩家模拟器进行对话
  - 调用评分器进行自动化评估
  - 管理对话流程、错误处理和超时控制

#### 4. 评分器系统 (Grader System)
- **架构**: 插件化评分器系统，支持多种评分策略
- **类型**:
  - **自然语言断言评分器** (Model-based Grader): 基于LLM评估对话是否符合自然语言断言
  - **代码规则评分器** (Code-based Grader): 基于代码规则进行客观评估
  - **混合评分器** (Hybrid Grader): 组合多种评分器结果
  - **多评判者投票机制** (Multi-judge Grader): 多个评分器投票决定最终结果

#### 5. 报告生成器 (Report Generator)
- **职责**: 生成结构化评测报告
- **格式**: 支持JSON（详细数据）和HTML（可视化展示）
- **内容**: 包含摘要统计、详细结果、对话记录、评分证据等

### 数据流设计

```
1. 加载测试场景定义 (YAML格式)
2. 初始化NPC Agent（基于配置选择LLM API）
3. 初始化玩家模拟器（基于场景配置）
4. 对于每个测试场景：
   a. 玩家模拟器发起对话（初始提示）
   b. NPC Agent生成回复
   c. 多轮对话（3-5轮）
   d. 对话结束，调用所有评分器
   e. 收集评分结果和证据
5. 生成评测报告（JSON + HTML）
6. 输出结果和分析
```

## 关键技术设计

### 1. 评分转换逻辑设计（新增）

基于Anthropic博客的最佳实践，系统采用了分层评分架构：

```
LLM True/False判断 → 代码逻辑转换 → 最终分数
```

#### 评分转换策略
系统支持四种转换策略，通过 `ScoreConverter` 模块实现：

1. **简单比例策略** (simple_ratio)
   - 计算断言通过的比例：通过数 / 总数
   - 适用于所有断言同等重要的场景

2. **加权平均策略** (weighted_average)
   - 为每个断言分配权重
   - 计算加权通过率：(∑(权重 × 是否通过)) / ∑权重
   - 适用于不同断言重要性不同的场景

3. **必选断言策略** (must_pass)
   - 指定部分断言为"必选断言"
   - 如果任何必选断言失败，得分为0
   - 可选断言按比例计算得分
   - 适用于有核心要求必须满足的场景

4. **阈值判定策略** (threshold_based)
   - 设置通过阈值和严格阈值
   - 比例 ≥ 严格阈值 → 得分为1.0
   - 比例 ≥ 通过阈值但 < 严格阈值 → 得分为实际比例
   - 比例 < 通过阈值 → 得分为实际比例
   - 适用于需要激励高分表现但允许一定容忍度的场景

#### 配置示例
```yaml
conversion:
  strategy: "simple_ratio"
  pass_threshold: 0.75
  # 可选配置，根据策略不同
  assertion_weights: [1.0, 2.0, 1.5]  # 加权平均策略
  must_pass_assertions: [0, 2]        # 必选断言策略
  strict_threshold: 0.9               # 阈值判定策略
```

### 2. 真实多样的玩家模拟器设计

#### 玩家角色系统
```python
class PlayerPersona:
    def __init__(self, config: dict):
        self.name = config["name"]                # 玩家名称
        self.personality = config["personality"]  # 性格（friendly, shy, aggressive等）
        self.age = config["age"]                  # 年龄
        self.background = config["background"]    # 背景故事
        self.interests = config["interests"]      # 兴趣列表
        self.speaking_style = config["speaking_style"]  # 说话风格
        self.knowledge_level = config["knowledge_level"]  # 知识水平（1-10）
        self.emotional_state = config["emotional_state"]  # 情感状态
```

#### 对话策略引擎
- **日常聊天模式**: 轻松友好的对话，关注日常生活话题
- **情感倾诉模式**: 玩家需要情感支持的场景
- **观点冲突模式**: 测试NPC处理意见分歧的能力
- **知识测试模式**: 验证NPC的知识准确性和深度

### 3. 多类型评分器系统（基于Anthropic博客）

#### 自然语言断言评分器 (Model-based Grader)
- **核心思想**: 使用LLM评估对话是否符合自然语言描述的标准
- **实现**: 提供断言列表，LLM对每个断言进行True/False判断
- **示例断言**:
  - "NPC保持友好的态度"
  - "对话自然流畅，符合角色设定"
  - "NPC提供了有价值的信息"

#### 代码规则评分器 (Code-based Grader)
- **核心思想**: 使用代码规则进行客观、可复现的评估
- **实现**: 关键词检测、模式匹配、响应长度检查等
- **优势**: 完全确定性，不受LLM非确定性的影响

#### 多评判者投票机制
- **核心思想**: 多个独立的评分器投票决定最终结果
- **实现**: 支持多种投票策略（多数决、加权投票、一致性检查）
- **优势**: 减少单个评分器的偏差，提高评估的可靠性

### 4. 评测结果可复现性设计

#### 确定性控制机制
1. **随机种子固定**: 所有随机数生成器使用固定的种子
2. **LLM参数确定性配置**: 设置temperature=0等参数确保LLM输出的确定性
3. **状态完整保存**: 保存评测过程的完整状态（输入、配置、中间结果）
4. **环境隔离**: 每次评测在干净的环境中运行，避免状态污染

#### 状态保存格式
```json
{
  "run_id": "run_20240121_123456",
  "random_seed": 42,
  "config_hash": "abc123...",
  "input_data": "...",
  "intermediate_results": [...],
  "final_results": {...}
}
```

### 5. 完整对话记录管理

#### 对话记录结构
```python
{
  "speaker": "player" | "npc",
  "message": "对话内容",
  "timestamp": 1634567890.123,
  "metadata": {
    "emotional_state": "happy",
    "topic": "weather",
    "turn_number": 3
  }
}
```

#### 记录保存策略
1. **实时记录**: 每轮对话后立即记录
2. **完整保存**: 保存所有对话轮次和元数据
3. **结构化存储**: JSON格式，便于后续分析和可视化

### 6. Capability Eval vs Regression Eval支持

#### 两种评测模式设计
| 维度 | Capability Eval | Regression Eval |
|------|-----------------|-----------------|
| **目标** | 测试NPC的能力边界 | 防止功能倒退 |
| **场景选择** | 挑战性、边缘场景 | 稳定、核心功能场景 |
| **通过率预期** | 可能较低，探索能力上限 | 接近100%，确保稳定性 |
| **重点** | 发现新能力，拓展边界 | 验证现有功能正常 |

#### 评测套件管理
```python
class TestSuite:
    def __init__(self, suite_type: str):  # "capability" 或 "regression"
        self.suite_type = suite_type
        self.scenarios = []

    def add_scenario(self, scenario: Scenario):
        # 根据套件类型筛选合适的场景
        if self._is_suitable_for_suite(scenario):
            self.scenarios.append(scenario)
```

## 接口设计

### 1. NPC接口
```python
class NPCAgent(ABC):
    """NPC Agent抽象基类"""

    @abstractmethod
    async def respond(self, user_input: str, context: DialogueContext) -> str:
        """
        根据用户输入和对话上下文生成回复

        Args:
            user_input: 用户输入文本
            context: 对话上下文，包含历史记录和元数据

        Returns:
            NPC的回复文本
        """
        pass
```

### 2. 评测器接口
```python
class Evaluator(ABC):
    """评测器抽象基类"""

    @abstractmethod
    async def evaluate_scenario(self, scenario: Scenario) -> EvaluationResult:
        """
        评测单个场景

        Args:
            scenario: 测试场景定义

        Returns:
            评测结果，包含分数、对话记录和评分证据
        """
        pass
```

### 3. 评分器接口
```python
class Grader(ABC):
    """评分器抽象基类"""

    @abstractmethod
    async def grade(self, transcript: List[Dict]) -> GradingResult:
        """
        根据对话记录进行评分

        Args:
            transcript: 完整对话记录

        Returns:
            评分结果，包含分数、通过状态、理由和证据
        """
        pass
```

### 4. 玩家模拟器接口
```python
class PlayerSimulator(ABC):
    """玩家模拟器抽象基类"""

    @abstractmethod
    async def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
        """
        根据NPC回复和对话上下文生成玩家回复

        Args:
            npc_reply: NPC的回复文本
            context: 对话上下文

        Returns:
            玩家的回复文本
        """
        pass
```

## 配置文件设计

### NPC配置 (configs/npc_config.yaml)
```yaml
npc_roles:
  friendly_npc:
    name: "友好NPC"
    personality: "友好、热情、乐于助人"
    background: "小镇居民，热爱分享"
    speaking_style: "热情洋溢，喜欢用感叹号"
    values: ["友善", "诚实", "乐于助人"]

agent_configs:
  mock_agent:
    type: "mock"
    role: "{{npc_roles.friendly_npc}}"
    config:
      response_type: "fixed"
      fixed_responses:
        - "你好！很高兴见到你！"
        - "今天天气真不错，你觉得呢？"
        - "我最近在学做菜，你有喜欢的食物吗？"
```

### 评测配置 (configs/eval_config.yaml)
```yaml
global:
  output_dir: "./outputs"
  save_transcripts: true
  max_concurrent_scenarios: 1

evaluator:
  config:
    max_turns: 5
    timeout_seconds: 30

graders:
  character_consistency:
    type: "natural_language"
    config:
      assertions:
        - "NPC的回复符合角色设定：友好、热情、乐于助人"
        - "NPC保持一致的说话风格：热情洋溢，喜欢用感叹号"
        - "NPC的回应体现了其价值观：友善、诚实、乐于助人"
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-haiku-20240307"
      conversion:
        strategy: "simple_ratio"
        pass_threshold: 0.75

  interaction_quality:
    type: "natural_language"
    config:
      assertions:
        - "NPC的回复与玩家输入相关"
        - "对话自然流畅，没有明显的逻辑跳跃"
        - "NPC的回应有助于对话的继续"
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-haiku-20240307"
      conversion:
        strategy: "weighted_average"
        pass_threshold: 0.7
        assertion_weights: [1.0, 1.5, 1.0]

  basic_rules:
    type: "rule_based"
    config:
      rules:
        - type: "length_check"
          min_length: 5
          max_length: 200
        - type: "keyword_check"
          forbidden_keywords: ["仇恨言论", "敏感内容"]
      conversion:
        strategy: "must_pass"
        pass_threshold: 1.0
        must_pass_assertions: [0]  # 长度检查必须通过

test_suites:
  capability_suite:
    name: "能力评测套件"
    description: "测试NPC的能力边界，包含挑战性场景"
    suite_type: "capability"
    scenarios:
      - "emotional_support"
      - "opinion_conflict"
      - "knowledge_challenge"

  regression_suite:
    name: "回归测试套件"
    description: "确保核心功能稳定，防止功能倒退"
    suite_type: "regression"
    scenarios:
      - "daily_chat"
      - "simple_qna"
      - "greeting_response"
```

### 场景配置 (configs/scenarios/daily_chat.yaml)
```yaml
id: "daily_chat"
name: "日常聊天场景"
description: "模拟日常对话，测试NPC的友好度和自然度"
scenario_type: "daily_chat"
player_profile:
  name: "普通玩家"
  personality: "friendly"
  age: 25
  background: "办公室职员，喜欢户外活动"
  interests: ["徒步", "摄影", "美食"]
  speaking_style: "随意、友好"
  knowledge_level: 6
  emotional_state: "neutral"
initial_prompt: "你好！最近过得怎么样？"
max_turns: 5
expected_outcomes:
  - "NPC应该友好回应"
  - "对话应该自然流畅"
  - "NPC应该表现出兴趣并提问"
reference_solution:
  example_dialogue: |
    玩家：你好！最近过得怎么样？
    NPC：我很好！谢谢关心。最近天气不错，你有出去走走吗？
    玩家：是的，我周末去徒步了。
    NPC：真不错！你喜欢哪里的徒步路线？
  key_points:
    - "友好问候"
    - "表现关心"
    - "提问引导对话继续"
```

## 核心算法设计

### 1. 评分转换算法

#### 简单比例策略
```python
def _convert_simple_ratio(self, assertion_results, reasoning):
    true_count = sum(1 for result in assertion_results.values() if result)
    ratio = true_count / self.total_assertions

    passed = ratio >= self.pass_threshold
    details = f"简单比例：{true_count}/{self.total_assertions} ({ratio:.2%})，通过阈值：{self.pass_threshold}"

    return ratio, passed, details
```

#### 加权平均策略
```python
def _convert_weighted_average(self, assertion_results, reasoning):
    weighted_sum = 0.0
    total_weight = 0.0

    for i, (assertion_idx, result) in enumerate(assertion_results.items()):
        weight = self.assertion_weights[i]
        total_weight += weight
        if result:
            weighted_sum += weight

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    passed = score >= self.pass_threshold

    details = f"加权平均：{score:.2%}，权重：{self.assertion_weights}，通过阈值：{self.pass_threshold}"

    return score, passed, details
```

### 2. 玩家模拟算法

#### 对话策略选择
```python
def select_dialogue_strategy(self, context: DialogueContext) -> DialogueStrategy:
    # 基于玩家角色、情感状态和对话历史选择策略
    if context.emotional_state == "sad":
        return DialogueStrategy.EMOTIONAL_SUPPORT
    elif context.current_topic == "controversial":
        return DialogueStrategy.OPINION_CONFLICT
    else:
        return DialogueStrategy.DAILY_CHAT
```

#### 回复生成算法
```python
def generate_response(self, npc_reply: str, context: DialogueContext) -> str:
    # 1. 分析NPC回复的内容和情感
    analysis = self.analyze_npc_reply(npc_reply)

    # 2. 选择对话策略
    strategy = self.select_dialogue_strategy(context)

    # 3. 基于策略生成回复
    if strategy == DialogueStrategy.DAILY_CHAT:
        response = self.generate_daily_chat_response(analysis, context)
    elif strategy == DialogueStrategy.EMOTIONAL_SUPPORT:
        response = self.generate_emotional_response(analysis, context)
    elif strategy == DialogueStrategy.OPINION_CONFLICT:
        response = self.generate_opinion_response(analysis, context)

    # 4. 根据玩家角色调整回复风格
    response = self.adjust_to_persona(response)

    return response
```

## 系统扩展性设计

### 插件化架构

#### 评分器插件系统
```python
class GraderPluginSystem:
    def __init__(self):
        self.plugins = {}

    def register_grader(self, name: str, grader_class: Type[Grader]):
        self.plugins[name] = grader_class

    def create_grader(self, name: str, config: dict) -> Grader:
        return self.plugins[name](config)
```

#### 支持的评分器类型
1. NaturalLanguageAssertionGrader - 自然语言断言评分器
2. RuleBasedGrader - 基于规则的评分器
3. HybridGrader - 混合评分器
4. MultiJudgeGrader - 多评判者投票机制
5. CustomGrader - 用户自定义评分器

### 配置驱动的NPC替换

系统支持通过配置文件切换不同的NPC实现，无需修改代码：
```yaml
agent_configs:
  mock_agent:
    type: "mock"  # 使用Mock Agent
    # ... 配置

  openai_agent:
    type: "openai"  # 使用OpenAI GPT Agent
    config:
      model: "gpt-4"
      temperature: 0.7

  anthropic_agent:
    type: "anthropic"  # 使用Anthropic Claude Agent
    config:
      model: "claude-3-haiku"
      temperature: 0.0
```

## 性能与优化

### 性能优化策略

#### 1. 异步处理
- 所有I/O操作（API调用、文件读写）使用异步处理
- 支持并发运行多个测试场景（配置控制）

#### 2. 缓存机制
- LLM API响应缓存（相同输入产生相同输出）
- 配置解析结果缓存
- 玩家模拟器模板缓存

#### 3. 批量处理
- 批量发送评分请求给LLM API
- 批量保存对话记录和结果

#### 4. 资源管理
- 连接池管理（数据库、API连接）
- 内存使用监控和优化
- 超时控制和错误恢复

### 成本控制

#### 1. API调用优化
- 使用较小模型进行初步评估（如Claude Haiku）
- 只在需要时使用较大模型进行详细评估
- 优化Prompt设计，减少token使用

#### 2. 缓存策略
- 缓存常见的评估结果
- 重用相同输入的API响应

#### 3. 本地模型支持
- 支持使用本地模型减少API成本
- 在离线环境下使用本地模型进行评估

## 安全性设计

### 1. API密钥管理
- 使用环境变量存储API密钥
- 配置文件支持变量替换（${API_KEY}）
- 密钥不在代码或配置文件中硬编码

### 2. 输入验证
- 所有用户输入（测试场景、配置）进行验证
- 防止代码注入和恶意输入
- 配置文件语法和结构验证

### 3. 访问控制
- 输出目录权限控制
- 敏感信息（对话记录、评分结果）访问限制
- 日志信息脱敏处理

## 测试策略

### 1. 单元测试
- 核心算法测试（评分转换、玩家模拟等）
- 接口一致性测试
- 错误处理测试

### 2. 集成测试
- 端到端流程测试
- 配置文件加载和验证测试
- API集成测试

### 3. 性能测试
- 并发处理能力测试
- 内存使用和泄漏测试
- API调用性能测试

### 4. 回归测试
- 使用Regression Test Suite确保核心功能稳定
- 每次代码变更后运行回归测试

## 部署与维护

### 1. 部署要求
- Python 3.8+
- 必要的依赖包（见requirements.txt）
- API密钥（OpenAI/Anthropic等）

### 2. 环境配置
```bash
# 1. 克隆代码
git clone https://github.com/example/agent-eval.git
cd agent-eval

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -e ".[dev]"  # 安装开发依赖

# 4. 配置环境变量
cp .env.example .env
# 编辑.env文件，设置API密钥

# 5. 运行测试
pytest tests/
```

### 3. 监控与日志
- 详细的运行日志记录
- 性能指标收集（API调用次数、耗时等）
- 错误监控和告警

## 未来扩展方向

### 1. 更多评分维度
- 情感支持能力评估
- 知识准确性和深度评估
- 文化适应性和敏感性评估

### 2. 高级NPC功能
- 长期记忆支持
- 情感状态模拟
- 个性化对话生成

### 3. 自动化优化
- 基于评测结果的自动Prompt优化
- 参数调优建议
- 最佳实践推荐

### 4. 可视化工具
- 交互式结果分析界面
- 对话记录可视化查看
- 性能指标仪表板

### 5. 集成测试
- CI/CD流水线集成
- 自动化回归测试
- 版本对比和趋势分析

---

*最后更新: 2026-01-21*