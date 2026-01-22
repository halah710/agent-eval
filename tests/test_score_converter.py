#!/usr/bin/env python3
"""
测试评分转换逻辑
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graders.score_converter import ScoreConverter, ConversionStrategy

def test_simple_ratio():
    """测试简单比例转换策略"""
    print("测试简单比例转换策略...")

    # 配置：简单比例，通过阈值0.7
    config = {
        "strategy": "simple_ratio",
        "pass_threshold": 0.7
    }

    converter = ScoreConverter(config)

    # 测试用例1: 4个断言中3个为True (75%，应该通过)
    assertion_results = {0: True, 1: True, 2: True, 3: False}
    score, passed, details = converter.convert(assertion_results, 4, "测试用例1")
    print(f"  用例1: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert abs(score - 0.75) < 0.01, f"期望得分0.75，实际得分{score}"
    assert passed == True, f"期望通过，实际未通过"

    # 测试用例2: 4个断言中2个为True (50%，不应该通过)
    assertion_results = {0: True, 1: True, 2: False, 3: False}
    score, passed, details = converter.convert(assertion_results, 4, "测试用例2")
    print(f"  用例2: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert abs(score - 0.5) < 0.01, f"期望得分0.5，实际得分{score}"
    assert passed == False, f"期望不通过，实际通过"

    print("  简单比例测试通过！")

def test_weighted_average():
    """测试加权平均转换策略"""
    print("测试加权平均转换策略...")

    # 配置：加权平均，通过阈值0.7
    config = {
        "strategy": "weighted_average",
        "pass_threshold": 0.7,
        "assertion_weights": [1.0, 2.0, 1.5, 1.0]  # 4个断言的权重
    }

    converter = ScoreConverter(config)

    # 测试用例: 4个断言中2个为True，但权重不同
    # True的断言: 索引1(权重2.0)和索引2(权重1.5)
    # False的断言: 索引0(权重1.0)和索引3(权重1.0)
    assertion_results = {0: False, 1: True, 2: True, 3: False}
    score, passed, details = converter.convert(assertion_results, 4, "测试用例")

    # 计算期望值: (2.0 + 1.5) / (1.0 + 2.0 + 1.5 + 1.0) = 3.5 / 5.5 ≈ 0.636
    expected_score = 3.5 / 5.5
    print(f"  用例: 得分={score:.2f}, 通过={passed}, 详情={details}")
    print(f"  期望得分: {expected_score:.3f}")
    assert abs(score - expected_score) < 0.01, f"期望得分{expected_score}，实际得分{score}"
    assert passed == False, f"期望不通过(得分{score:.2f} < 0.7)，实际通过"

    print("  加权平均测试通过！")

def test_must_pass():
    """测试必选断言转换策略"""
    print("测试必选断言转换策略...")

    # 配置：必选断言策略，通过阈值0.7
    config = {
        "strategy": "must_pass",
        "pass_threshold": 0.7,
        "must_pass_assertions": [0, 2]  # 断言0和2必须通过
    }

    converter = ScoreConverter(config)

    # 测试用例1: 必选断言都通过，可选断言部分通过
    # 断言0(必选): True, 断言1(可选): False, 断言2(必选): True, 断言3(可选): True
    assertion_results = {0: True, 1: False, 2: True, 3: True}
    score, passed, details = converter.convert(assertion_results, 4, "测试用例1")
    print(f"  用例1: 得分={score:.2f}, 通过={passed}, 详情={details}")
    # 可选断言：1个True(索引3) / 2个可选 = 0.5
    assert abs(score - 0.5) < 0.01, f"期望得分0.5，实际得分{score}"
    assert passed == False, f"期望不通过(得分{score:.2f} < 0.7)，实际通过"

    # 测试用例2: 必选断言失败，直接0分不通过
    assertion_results = {0: False, 1: True, 2: True, 3: True}  # 断言0(必选)失败
    score, passed, details = converter.convert(assertion_results, 4, "测试用例2")
    print(f"  用例2: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert score == 0.0, f"期望得分0.0，实际得分{score}"
    assert passed == False, f"期望不通过，实际通过"

    print("  必选断言测试通过！")

def test_threshold_based():
    """测试阈值判定转换策略"""
    print("测试阈值判定转换策略...")

    # 配置：阈值判定策略，通过阈值0.7，严格阈值0.9
    config = {
        "strategy": "threshold_based",
        "pass_threshold": 0.7,
        "strict_threshold": 0.9
    }

    converter = ScoreConverter(config)

    # 测试用例1: 比例0.95 ≥ 严格阈值0.9，得分为1.0，通过
    assertion_results = {i: True for i in range(20)}  # 20个断言，19个True (95%)
    assertion_results[19] = False  # 最后一个为False
    score, passed, details = converter.convert(assertion_results, 20, "测试用例1")
    print(f"  用例1: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert score == 1.0, f"期望得分1.0，实际得分{score}"
    assert passed == True, f"期望通过，实际未通过"

    # 测试用例2: 比例0.8 ≥ 通过阈值0.7但<严格阈值0.9，得分为0.8，通过
    assertion_results = {i: True for i in range(16)}  # 20个断言，16个True (80%)
    for i in range(16, 20):
        assertion_results[i] = False
    score, passed, details = converter.convert(assertion_results, 20, "测试用例2")
    print(f"  用例2: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert abs(score - 0.8) < 0.01, f"期望得分0.8，实际得分{score}"
    assert passed == True, f"期望通过，实际未通过"

    # 测试用例3: 比例0.6 < 通过阈值0.7，得分为0.6，不通过
    assertion_results = {i: True for i in range(12)}  # 20个断言，12个True (60%)
    for i in range(12, 20):
        assertion_results[i] = False
    score, passed, details = converter.convert(assertion_results, 20, "测试用例3")
    print(f"  用例3: 得分={score:.2f}, 通过={passed}, 详情={details}")
    assert abs(score - 0.6) < 0.01, f"期望得分0.6，实际得分{score}"
    assert passed == False, f"期望不通过，实际通过"

    print("  阈值判定测试通过！")

def test_assertion_status_text():
    """测试断言状态文本生成"""
    print("测试断言状态文本生成...")

    # 测试不同策略下的文本生成
    assertions = ["断言1", "断言2", "断言3", "断言4"]
    assertion_results = {0: True, 1: False, 2: True, 3: False}

    # 简单比例策略
    config1 = {"strategy": "simple_ratio"}
    converter1 = ScoreConverter(config1)
    text1 = converter1.get_assertion_status_text(assertion_results, assertions)
    print(f"  简单比例策略文本:\n{text1}")

    # 加权平均策略
    config2 = {
        "strategy": "weighted_average",
        "assertion_weights": [1.0, 2.0, 1.5, 1.0]
    }
    converter2 = ScoreConverter(config2)
    text2 = converter2.get_assertion_status_text(assertion_results, assertions)
    print(f"  加权平均策略文本:\n{text2}")

    # 必选断言策略
    config3 = {
        "strategy": "must_pass",
        "must_pass_assertions": [0, 2]
    }
    converter3 = ScoreConverter(config3)
    text3 = converter3.get_assertion_status_text(assertion_results, assertions)
    print(f"  必选断言策略文本:\n{text3}")

    print("  断言状态文本生成测试通过！")

def main():
    """主测试函数"""
    print("评分转换器测试")
    print("=" * 50)

    try:
        test_simple_ratio()
        test_weighted_average()
        test_must_pass()
        test_threshold_based()
        test_assertion_status_text()

        print("\n" + "=" * 50)
        print("所有测试通过！评分转换逻辑正常工作。")

    except AssertionError as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()