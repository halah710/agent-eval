# AI Agent评测系统 API文档

本文档详细描述了AI Agent评测系统的API接口、使用方法和示例。

## 目录

1. [核心接口概述](#核心接口概述)
2. [NPC Agent接口](#npc-agent接口)
3. [玩家模拟器接口](#玩家模拟器接口)
4. [评测器接口](#评测器接口)
5. [评分器接口](#评分器接口)
6. [报告生成器接口](#报告生成器接口)
7. [配置系统](#配置系统)
8. [工具函数](#工具函数)
9. [使用示例](#使用示例)
10. [错误处理](#错误处理)

## 核心接口概述

AI Agent评测系统采用分层架构，主要接口如下：

```python
# 核心接口导入
from src.agents.base import NPCAgent
from src.players.base import PlayerSimulator
from src.evaluator.base import Evaluator, Scenario, TestSuite, EvaluationResult
from src.graders.base import Grader, GradingResult
from src.reports import ReportGenerator, generate_both_reports
```

## NPC Agent接口

### NPCAgent 抽象基类

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..agents.base import DialogueContext

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

    @abstractmethod
    async def initialize(self) -> None:
        """初始化NPC Agent"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭NPC Agent，释放资源"""
        pass
```

### NPCRole 数据模型

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class NPCRole(BaseModel):
    """NPC角色定义"""
    name: str = Field(..., description="角色名称")
    personality: str = Field(..., description="角色性格")
    background: str = Field(..., description="角色背景")
    speaking_style: str = Field(..., description="说话风格")
    values: List[str] = Field(default_factory=list, description="角色价值观")
    knowledge_domains: List[str] = Field(default_factory=list, description="知识领域")
    emotional_traits: Dict[str, float] = Field(default_factory=dict, description="情感特质")
```

### 工厂模式创建NPC Agent

```python
from src.agents.factory import AgentFactory

# 从配置创建NPC Agent
agent_config = {
    "type": "openai",  # 或 "anthropic", "mock", "local"
    "role": {
        "name": "友好NPC",
        "personality": "友好、热情",
        "background": "小镇居民",
        "speaking_style": "热情洋溢",
        "values": ["友善", "诚实"]
    },
    "config": {
        "model": "gpt-4",
        "temperature": 0.7,
        "api_key": "${OPENAI_API_KEY}"  # 从环境变量读取
    }
}

agent = AgentFactory.create_agent_from_config(agent_config)
await agent.initialize()
```

### 内建的NPC Agent实现

| Agent类型 | 描述 | 适用场景 |
|-----------|------|----------|
| `MockAgent` | 模拟NPC，使用固定回复或随机回复 | 测试、开发 |
| `OpenAIAgent` | 基于OpenAI API的NPC | 生产环境 |
| `AnthropicAgent` | 基于Anthropic Claude API的NPC | 生产环境 |
| `LocalAgent` | 基于本地模型的NPC | 离线环境、成本控制 |

## 玩家模拟器接口

### PlayerSimulator 抽象基类

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..agents.base import DialogueContext

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

    @abstractmethod
    async def initialize(self) -> None:
        """初始化玩家模拟器"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭玩家模拟器，释放资源"""
        pass
```

### PlayerProfile 数据模型

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PlayerProfile(BaseModel):
    """玩家角色配置"""
    name: str = Field(..., description="玩家名称")
    personality: str = Field(..., description="玩家性格")
    age: int = Field(..., description="年龄")
    background: str = Field(..., description="背景故事")
    interests: List[str] = Field(default_factory=list, description="兴趣列表")
    speaking_style: str = Field(..., description="说话风格")
    knowledge_level: int = Field(..., description="知识水平（1-10）")
    emotional_state: str = Field(..., description="情感状态")
```

### 工厂模式创建玩家模拟器

```python
from src.players.factory import PlayerFactory

# 从配置创建玩家模拟器
player_config = {
    "type": "simple",  # 或 "advanced"
    "config": {
        "response_templates": ["你好！", "今天天气不错。", "很高兴和你聊天。"],
        "personality": "friendly",
        "dialogue_strategy": "daily_chat"
    }
}

player = PlayerFactory.create_player(player_config)
await player.initialize()
```

### 内建的玩家模拟器实现

| 模拟器类型 | 描述 | 特点 |
|-----------|------|------|
| `SimplePlayer` | 简单玩家模拟器 | 基于模板的回复，适用于基础测试 |
| `AdvancedPlayer` | 高级玩家模拟器 | 基于规则的对话策略，支持多种对话模式 |

## 评测器接口

### Evaluator 抽象基类

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .base import Scenario, EvaluationResult

class Evaluator(ABC):
    """评测器抽象基类"""

    def __init__(self, npc_agent: NPCAgent, graders: List[Grader], config: Dict[str, Any] = None):
        """
        初始化评测器

        Args:
            npc_agent: NPC Agent实例
            graders: 评分器列表
            config: 配置参数
        """
        self.npc_agent = npc_agent
        self.graders = {grader.name: grader for grader in graders}
        self.config = config or {}

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

    async def evaluate_suite(self, test_suite: TestSuite) -> List[EvaluationResult]:
        """
        评测整个测试套件

        Args:
            test_suite: 测试套件

        Returns:
            评测结果列表
        """
        results = []
        for scenario in test_suite.scenarios:
            result = await self.evaluate_scenario(scenario)
            results.append(result)
        return results
```

### SimpleEvaluator 实现

```python
from src.evaluator.simple_evaluator import SimpleEvaluator

# 创建评测器
evaluator = SimpleEvaluator(
    npc_agent=agent,
    graders=[character_grader, interaction_grader],
    config={
        "max_turns": 5,
        "player_simulator": player,
        "timeout_seconds": 30
    }
)

# 评测单个场景
result = await evaluator.evaluate_scenario(scenario)

# 评测测试套件
results = await evaluator.evaluate_suite(test_suite)
```

### Scenario 数据模型

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class ScenarioType(str, Enum):
    """测试场景类型"""
    DAILY_CHAT = "daily_chat"  # 日常聊天
    EMOTIONAL_SUPPORT = "emotional_support"  # 情感支持
    OPINION_CONFLICT = "opinion_conflict"  # 观点冲突
    KNOWLEDGE_TEST = "knowledge_test"  # 知识测试
    ROLE_PLAYING = "role_playing"  # 角色扮演

class Scenario(BaseModel):
    """测试场景定义"""
    id: str = Field(..., description="场景ID")
    name: str = Field(..., description="场景名称")
    description: str = Field(..., description="场景描述")
    scenario_type: ScenarioType = Field(..., description="场景类型")
    player_profile: Dict[str, Any] = Field(default_factory=dict, description="玩家角色配置")
    initial_prompt: str = Field(..., description="初始提示（玩家第一句话）")
    max_turns: int = Field(default=5, description="最大对话轮次")
    expected_outcomes: List[str] = Field(default_factory=list, description="预期结果描述")
    reference_solution: Optional[Dict[str, Any]] = Field(None, description="参考解决方案")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """从字典创建"""
        return cls(**data)
```

### TestSuite 数据模型

```python
class TestSuite(BaseModel):
    """测试套件"""
    id: str = Field(..., description="套件ID")
    name: str = Field(..., description="套件名称")
    description: str = Field(..., description="套件描述")
    suite_type: str = Field(..., description="套件类型：capability 或 regression")
    scenarios: List[Scenario] = Field(default_factory=list, description="包含的场景")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def add_scenario(self, scenario: Scenario):
        """添加场景"""
        self.scenarios.append(scenario)

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """获取指定场景"""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None
```

### EvaluationResult 数据模型

```python
class EvaluationResult(BaseModel):
    """评测结果"""
    scenario_id: str = Field(..., description="场景ID")
    scenario_name: str = Field(..., description="场景名称")
    start_time: float = Field(default_factory=time.time, description="开始时间")
    end_time: Optional[float] = Field(None, description="结束时间")
    transcript: List[Dict[str, Any]] = Field(default_factory=list, description="完整对话记录")
    grading_results: Dict[str, GradingResult] = Field(default_factory=dict, description="评分结果")
    final_score: float = Field(default=0.0, description="最终得分（0-1）")
    passed: bool = Field(default=False, description="是否通过")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def add_turn(self, speaker: str, message: str, metadata: Dict[str, Any] = None):
        """添加对话轮次"""
        turn = {
            "speaker": speaker,
            "message": message,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.transcript.append(turn)

    def add_grading_result(self, grader_name: str, result: GradingResult):
        """添加评分结果"""
        self.grading_results[grader_name] = result

    def calculate_final_score(self, weights: Dict[str, float] = None) -> float:
        """计算最终得分"""
        if not self.grading_results:
            return 0.0

        if weights is None:
            weights = {name: 1.0 for name in self.grading_results.keys()}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for grader_name, result in self.grading_results.items():
            weight = weights.get(grader_name, 1.0)
            weighted_sum += result.score * weight

        self.final_score = weighted_sum / total_weight
        return self.final_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = self.model_dump()
        data["duration"] = (self.end_time or time.time()) - self.start_time
        return data

    def save_to_file(self, filepath: str):
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
```

## 评分器接口

### Grader 抽象基类

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Grader(ABC):
    """评分器抽象基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化评分器

        Args:
            name: 评分器名称
            config: 评分器配置
        """
        self.name = name
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """初始化评分器"""
        pass

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

    @abstractmethod
    async def close(self) -> None:
        """关闭评分器，释放资源"""
        pass
```

### 评分转换器 ScoreConverter

```python
from src.graders.score_converter import ScoreConverter

# 创建评分转换器
converter = ScoreConverter({
    "strategy": "simple_ratio",  # 或 "weighted_average", "must_pass", "threshold_based"
    "pass_threshold": 0.75,
    # 可选配置，根据策略不同
    "assertion_weights": [1.0, 2.0, 1.5],  # 加权平均策略
    "must_pass_assertions": [0, 2],        # 必选断言策略
    "strict_threshold": 0.9                # 阈值判定策略
})

# 转换断言结果为分数
assertion_results = {0: True, 1: True, 2: False, 3: True}
score, passed, details = converter.convert(assertion_results, 4, "测试用例")
```

### GradingResult 数据模型

```python
class GradingResult(BaseModel):
    """评分结果"""
    grader_name: str = Field(..., description="评分器名称")
    score: float = Field(..., description="得分（0-1）")
    passed: bool = Field(..., description="是否通过")
    reasoning: str = Field(..., description="评分理由")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="评分证据")
    assertion_results: Dict[int, bool] = Field(default_factory=dict, description="断言结果（索引 → True/False）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
```

### 工厂模式创建评分器

```python
from src.graders.factory import GraderFactory

# 创建自然语言断言评分器
nl_grader_config = {
    "type": "natural_language",
    "config": {
        "assertions": [
            "NPC保持友好态度",
            "对话自然流畅",
            "NPC的回复与玩家输入相关"
        ],
        "api_key": "${ANTHROPIC_API_KEY}",
        "model": "claude-3-haiku-20240307",
        "conversion": {
            "strategy": "simple_ratio",
            "pass_threshold": 0.75
        }
    }
}

nl_grader = GraderFactory.create_grader("character_consistency", nl_grader_config)
await nl_grader.initialize()

# 创建规则评分器
rule_grader_config = {
    "type": "rule_based",
    "config": {
        "rules": [
            {
                "type": "length_check",
                "min_length": 5,
                "max_length": 200
            },
            {
                "type": "keyword_check",
                "forbidden_keywords": ["仇恨言论", "敏感内容"]
            }
        ],
        "conversion": {
            "strategy": "must_pass",
            "pass_threshold": 1.0,
            "must_pass_assertions": [0]  # 长度检查必须通过
        }
    }
}

rule_grader = GraderFactory.create_grader("basic_rules", rule_grader_config)
await rule_grader.initialize()
```

### 内建的评分器实现

| 评分器类型 | 描述 | 核心特性 |
|-----------|------|----------|
| `NaturalLanguageGrader` | 自然语言断言评分器 | 基于LLM评估对话是否符合自然语言断言，支持True/False判断和多种转换策略 |
| `RuleBasedGrader` | 基于规则的评分器 | 使用代码规则进行客观评估，完全确定性 |
| `HybridGrader` | 混合评分器 | 组合多个评分器的结果 |
| `MultiJudgeGrader` | 多评判者评分器 | 多个独立评分器投票决定结果 |

## 报告生成器接口

### ReportGenerator 类

```python
from src.reports import ReportGenerator

# 创建报告生成器
generator = ReportGenerator(output_dir="./reports")

# 生成JSON详细报告
json_path = generator.generate_json_report(
    results=results,
    test_suite=test_suite,
    filename="evaluation_report.json"
)

# 生成HTML可视化报告
html_path = generator.generate_html_report(
    results=results,
    test_suite=test_suite,
    filename="evaluation_report.html"
)

# 同时生成两种格式的报告
reports = generator.generate_both_reports(results, test_suite)
print(f"JSON报告: {reports['json_report']}")
print(f"HTML报告: {reports['html_report']}")
```

### 便捷函数

```python
from src.reports import generate_json_report, generate_html_report, generate_both_reports

# 使用便捷函数
json_path = generate_json_report(
    results=results,
    output_dir="./reports",
    test_suite=test_suite
)

html_path = generate_html_report(
    results=results,
    output_dir="./reports",
    test_suite=test_suite
)

both_reports = generate_both_reports(
    results=results,
    output_dir="./reports",
    test_suite=test_suite
)
```

### ReportSummary 数据模型

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ReportSummary:
    """报告摘要"""
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    average_score: float
    min_score: float
    max_score: float
    score_std_dev: float
    total_duration: float
    suite_type: str
    run_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
```

## 配置系统

### 配置文件结构

系统使用YAML格式的配置文件，主要配置文件包括：

1. **NPC配置** (`configs/npc_config.yaml`) - NPC Agent配置
2. **评测配置** (`configs/eval_config.yaml`) - 评测框架配置
3. **场景配置** (`configs/scenarios/*.yaml`) - 测试场景定义

### 配置加载函数

```python
import yaml
from pathlib import Path

def load_npc_config(config_path: str = None) -> Dict[str, Any]:
    """加载NPC配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "npc_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def load_eval_config(config_path: str = None) -> Dict[str, Any]:
    """加载评测配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "eval_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def load_scenario(scenario_id: str) -> Scenario:
    """加载场景配置"""
    scenarios_dir = Path(__file__).parent.parent / "configs" / "scenarios"
    scenario_path = scenarios_dir / f"{scenario_id}.yaml"

    if not scenario_path.exists():
        scenario_path = scenarios_dir / scenario_id

    with open(scenario_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Scenario(**data)
```

### 环境变量支持

配置文件支持环境变量替换：

```yaml
api_key: "${ANTHROPIC_API_KEY}"  # 从环境变量读取
```

## 工具函数

### 随机状态管理

```python
from src.utils.random_state import set_global_seed, random_state_manager

# 设置全局随机种子（确保可复现性）
set_global_seed(42)

# 使用上下文管理器管理随机状态
with random_state_manager(seed=42):
    # 在这个作用域内，所有随机操作都是确定的
    result = await evaluator.evaluate_scenario(scenario)
```

### 对话上下文管理

```python
from src.agents.base import DialogueContext

# 创建对话上下文
context = DialogueContext()

# 添加对话历史
context.add_turn("player", "你好！")
context.add_turn("npc", "你好！很高兴见到你！")

# 获取对话历史
history = context.get_history()

# 清空对话历史
context.clear()
```

### 异步工具函数

```python
from src.utils.async_utils import run_with_timeout, batch_process

# 带超时的异步执行
try:
    result = await run_with_timeout(
        evaluator.evaluate_scenario(scenario),
        timeout=30  # 30秒超时
    )
except asyncio.TimeoutError:
    print("评测超时")

# 批量处理
results = await batch_process(
    items=scenarios,
    process_func=lambda scenario: evaluator.evaluate_scenario(scenario),
    max_concurrency=3  # 最大并发数
)
```

## 使用示例

### 示例1：基础评测流程

```python
import asyncio
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.factory import AgentFactory
from src.players.factory import PlayerFactory
from src.evaluator.simple_evaluator import SimpleEvaluator
from src.evaluator.base import Scenario
from src.graders.factory import GraderFactory
from src.reports import generate_both_reports
from src.utils.random_state import set_global_seed

async def main():
    """基础评测示例"""
    # 设置随机种子
    set_global_seed(42)

    # 创建NPC Agent
    agent = AgentFactory.create_agent_from_config({
        "type": "mock",
        "role": {
            "name": "测试NPC",
            "personality": "友好",
            "background": "测试背景",
            "speaking_style": "测试风格",
            "values": ["测试值1", "测试值2"]
        },
        "config": {
            "response_type": "fixed",
            "fixed_responses": ["你好！", "今天天气不错。", "很高兴和你聊天。"]
        }
    })

    # 创建玩家模拟器
    player = PlayerFactory.create_player({
        "type": "simple",
        "config": {
            "response_templates": ["测试玩家回复1", "测试玩家回复2"]
        }
    })
    await player.initialize()

    # 创建评分器
    character_grader = GraderFactory.create_grader("character_consistency", {
        "type": "natural_language",
        "config": {
            "assertions": ["NPC保持友好态度"],
            "api_key": "test_key",
            "conversion": {
                "strategy": "simple_ratio",
                "pass_threshold": 0.75
            }
        }
    })
    await character_grader.initialize()

    # 创建评测器
    evaluator = SimpleEvaluator(
        agent=agent,
        graders=[character_grader],
        config={
            "max_turns": 3,
            "player_simulator": player
        }
    )

    # 创建测试场景
    scenario = Scenario(
        id="test_scenario",
        name="测试场景",
        description="测试描述",
        scenario_type="daily_chat",
        player_profile={
            "name": "测试玩家",
            "personality": "friendly",
            "age": 25,
            "background": "测试背景",
            "interests": ["测试兴趣"],
            "speaking_style": "测试风格",
            "knowledge_level": 5,
            "emotional_state": "neutral"
        },
        initial_prompt="你好！",
        max_turns=3
    )

    # 运行评测
    result = await evaluator.evaluate_scenario(scenario)

    # 输出结果
    print(f"场景: {result.scenario_name}")
    print(f"得分: {result.final_score:.2%}")
    print(f"通过: {'是' if result.passed else '否'}")

    # 生成报告
    reports = generate_both_reports(
        results=[result],
        output_dir="./reports"
    )
    print(f"报告已生成: {reports}")

    # 清理资源
    await evaluator.close()
    await player.close()
    await character_grader.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例2：测试套件评测

```python
async def run_test_suite():
    """测试套件评测示例"""
    # 设置随机种子
    set_global_seed(42)

    # 加载配置
    eval_config = load_eval_config()
    npc_config = load_npc_config()

    # 创建NPC Agent
    agent = await create_agent(eval_config, npc_config)

    # 创建评分器
    graders = await create_graders(eval_config)

    # 创建评测器
    evaluator = SimpleEvaluator(
        agent=agent,
        graders=graders,
        config=eval_config.get("evaluator", {}).get("config", {})
    )

    # 加载测试套件
    test_suite = await load_test_suite("capability_suite", eval_config)

    # 运行套件评测
    results = await evaluator.evaluate_suite(test_suite)

    # 生成报告
    reports = generate_both_reports(
        results=results,
        output_dir="./reports",
        test_suite=test_suite
    )

    # 输出摘要
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.final_score for r in results) / len(results)

    print(f"测试套件: {test_suite.name}")
    print(f"场景总数: {len(results)}")
    print(f"通过场景: {passed_count}")
    print(f"平均得分: {avg_score:.2%}")
    print(f"报告位置: {reports}")

    # 清理资源
    await evaluator.close()
    for grader in graders:
        await grader.close()

    return results
```

### 示例3：自定义评分器

```python
from src.graders.base import Grader, GradingResult

class CustomGrader(Grader):
    """自定义评分器示例"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.threshold = config.get("threshold", 0.8)

    async def initialize(self):
        self._initialized = True

    async def grade(self, transcript: List[Dict]) -> GradingResult:
        # 自定义评分逻辑
        # 示例：根据对话长度评分
        total_length = sum(len(turn.get("message", "")) for turn in transcript)

        # 计算得分（对话越长得分越高，但有限制）
        max_length = 500
        score = min(total_length / max_length, 1.0)

        passed = score >= self.threshold

        return GradingResult(
            grader_name=self.name,
            score=score,
            passed=passed,
            reasoning=f"对话总长度: {total_length}字符，得分: {score:.2%}",
            evidence=[
                {"type": "length_analysis", "total_length": total_length}
            ]
        )

    async def close(self):
        pass

# 注册自定义评分器
from src.graders.factory import GraderFactory
GraderFactory.register_grader("custom", CustomGrader)

# 使用自定义评分器
custom_grader = GraderFactory.create_grader("custom_length_check", {
    "type": "custom",
    "config": {
        "threshold": 0.6
    }
})
await custom_grader.initialize()
```

### 示例4：并发评测

```python
import asyncio
from src.utils.async_utils import batch_process

async def concurrent_evaluation(scenarios: List[Scenario], max_concurrent: int = 3):
    """并发评测示例"""

    async def evaluate_single(scenario: Scenario):
        """单个场景评测函数"""
        # 为每个场景创建独立的评测器（避免状态共享）
        agent = AgentFactory.create_agent_from_config({...})
        player = PlayerFactory.create_player({...})
        await player.initialize()

        grader = GraderFactory.create_grader(...)
        await grader.initialize()

        evaluator = SimpleEvaluator(
            agent=agent,
            graders=[grader],
            config={"max_turns": 5, "player_simulator": player}
        )

        result = await evaluator.evaluate_scenario(scenario)

        # 清理资源
        await evaluator.close()
        await player.close()
        await grader.close()

        return result

    # 并发运行所有场景评测
    results = await batch_process(
        items=scenarios,
        process_func=evaluate_single,
        max_concurrency=max_concurrent
    )

    return results
```

## 错误处理

### 异常类型

系统定义了以下异常类型：

```python
class AgentEvalError(Exception):
    """评测系统基础异常"""
    pass

class ConfigurationError(AgentEvalError):
    """配置错误"""
    pass

class APIConnectionError(AgentEvalError):
    """API连接错误"""
    pass

class TimeoutError(AgentEvalError):
    """超时错误"""
    pass

class EvaluationError(AgentEvalError):
    """评测过程错误"""
    pass
```

### 错误处理示例

```python
import asyncio
from src.utils.error_handling import retry_on_error

async def safe_evaluation(scenario: Scenario, max_retries: int = 3):
    """带错误重试的安全评测"""

    @retry_on_error(
        max_retries=max_retries,
        retry_on=[APIConnectionError, TimeoutError],
        backoff_factor=2.0
    )
    async def evaluate_with_retry():
        try:
            result = await evaluator.evaluate_scenario(scenario)
            return result
        except APIConnectionError as e:
            print(f"API连接错误: {e}，重试中...")
            raise
        except TimeoutError as e:
            print(f"评测超时: {e}，重试中...")
            raise
        except Exception as e:
            print(f"未知错误: {e}")
            raise EvaluationError(f"评测失败: {e}")

    try:
        return await evaluate_with_retry()
    except Exception as e:
        # 记录错误但继续其他评测
        print(f"场景 {scenario.id} 评测失败: {e}")
        return None
```

### 日志配置

系统使用loguru进行日志记录：

```python
import loguru
from src.utils.logging import setup_logging

# 设置日志
setup_logging(
    level="INFO",
    log_file="./logs/evaluation.log",
    rotation="10 MB",
    retention="1 month"
)

# 使用日志
logger = loguru.logger

async def evaluate_with_logging(scenario: Scenario):
    logger.info(f"开始评测场景: {scenario.name}")

    try:
        result = await evaluator.evaluate_scenario(scenario)
        logger.info(f"场景评测完成: {scenario.name}, 得分: {result.final_score:.2%}")
        return result
    except Exception as e:
        logger.error(f"场景评测失败: {scenario.name}, 错误: {e}")
        raise
```

## 性能调优

### 配置参数调优

```python
# 优化配置示例
optimized_config = {
    "global": {
        "max_concurrent_scenarios": 2,  # 控制并发数，避免API限流
        "save_transcripts": True,       # 保存对话记录用于调试
        "cache_responses": True         # 缓存API响应，提高性能
    },
    "evaluator": {
        "config": {
            "max_turns": 5,             # 优化对话轮次，平衡评估深度和效率
            "timeout_seconds": 30,      # 设置合理超时
            "temperature": 0.0          # 确定性模式，确保可复现性
        }
    },
    "graders": {
        "character_consistency": {
            "config": {
                "model": "claude-3-haiku-20240307",  # 使用较小模型，降低成本
                "max_tokens": 1000,                  # 限制响应长度
                "cache_size": 100                    # 缓存评分结果
            }
        }
    }
}
```

### 批量处理优化

```python
from src.utils.optimization import batch_api_calls, cache_results

# 批量API调用
async def batch_grade_transcripts(transcripts: List[List[Dict]], grader: Grader):
    """批量评分优化"""

    @batch_api_calls(batch_size=10, delay_between_batches=1.0)
    async def grade_batch(batch_transcripts):
        results = []
        for transcript in batch_transcripts:
            result = await grader.grade(transcript)
            results.append(result)
        return results

    return await grade_batch(transcripts)

# 结果缓存
@cache_results(max_size=100, ttl=3600)  # 缓存100个结果，1小时过期
async def cached_evaluation(scenario: Scenario, evaluator: Evaluator):
    """带缓存的评测"""
    return await evaluator.evaluate_scenario(scenario)
```

---

*最后更新: 2026-01-21*