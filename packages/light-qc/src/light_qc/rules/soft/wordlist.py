"""
Built-in word list for the BadWordSplit QC rule.

Helps jieba correctly segment Chinese compounds, AI/ML terminology,
and proper names — preventing false-positive word-split detections
and enabling accurate token-boundary checks.

Versioning
----------
Each version adds or removes word entries.  Increment VERSION and add
an entry to CHANGELOG when making changes.  BadWordSplit automatically
picks up the latest version at rule-instantiation time.

Adding words
------------
1. Add to the appropriate set below.
2. Bump VERSION.
3. Add a CHANGELOG entry.
4. Commit.

No migration needed — the old entries remain under earlier version
keys in the changelog.
"""

from __future__ import annotations

VERSION = 1
CHANGELOG: dict[int, str] = {
    1: (
        "Initial wordlist: common Chinese compounds split-prone "
        "by jieba, AI/ML domain terminology, and proper names "
        "from English/mixed languages."
    ),
}

# ── Chinese compounds that jieba's default dictionary
#    may not recognise or may split incorrectly ──

ZH_COMPOUNDS: set[str] = {
    # AI/ML domain terms (multi-character)
    "自监督",
    "联合嵌入",
    "表征学习",
    "表示坍缩",
    "反向传播",
    "深度学习",
    "机器学习",
    "神经网络",
    "生成式模型",
    "生成式",
    "预训练",
    "自回归",
    "编码器",
    "解码器",
    "孪生网络",
    "孪生",
    "互相关",
    "互相关矩阵",
    "损失函数",
    "世界模型",
    "基础模型",
    "语言模型",
    "图像分类",
    "强化学习",
    "监督学习",
    "无监督",
    "语义单元",
    "神经元",
    "训练集",
    "验证集",
    "正则化",
    "归一化",
    # Commonly split 2-char compounds
    "有用",
    "没用",
    "所有",
    "任何",
    "每种",
    "各种",
    "可以",
    "可能",
    "能够",
    "应该",
    "必须",
    "需要",
    "进行",
    "使用",
    "产生",
    "发生",
    "存在",
    "通过",
    "输入",
    "输出",
    "嵌入",
    "重建",
    "相似",
    # 2-char compounds commonly split by newlines in subtitles
    "架构",
    "年代",
    "训练",
    "语言",
    "生成",
    "模型",
    "表征",
    "预测",
    "了解",
    "突破",
    "持续",
    "领域",
    "崛起",
    "主导",
    "称为",
    "转换",
    "注意",
    "阶段",
    "趋势",
    "框架",
    "结构",
    "特征",
    "数据",
    "计算",
    "算法",
    "参数",
    "维度",
    "向量",
    "矩阵",
    # Video / image domain
    "像素",
    "图像",
    "视频",
    "帧率",
    "分辨率",
}

# ── Proper names (English, mixed, or transliterated)
#    Must be kept as single tokens across line/cue breaks ──

PROPER_NAMES: set[str] = {
    # AI models / architectures
    "JEPA",
    "VJEPA",
    "VLJEPA",
    "DINO",
    "Barlow",
    "BarlowTwins",
    "Barlow Twins",
    "VicReg",
    "VICReg",
    # Models
    "GPT",
    "GPT-1",
    "GPT-2",
    "GPT-3",
    "GPT-4",
    "LLM",
    "LLMs",
    "VLA",
    "AlexNet",
    "ImageNet",
    "Siamese",
    "transformer",
    "Transformers",
    # People
    "LeCun",
    "Yann",
    "YannLeCun",
    "Yann LeCun",
    "Sutskever",
    "Radford",
    "Denis",
    "Horace",
    "HoraceBarlow",
    "Horace Barlow",
    # Orgs / misc
    "OpenAI",
    "DeepMind",
    "Hudson",
    "River",
    "Hudson River Trading",
    "WelLabs",
    "Welch",
    "Welch Labs",
    "Meta",
    "Fair",
    "Paris",
    "Atari",
    "RGB",
    "CJK",
}


def get_all_words() -> list[str]:
    """Return all custom words for jieba dictionary loading.

    Sorted for deterministic registration order.
    """
    return sorted(ZH_COMPOUNDS | PROPER_NAMES)


def load_into_jieba() -> None:
    """Add all wordlist entries to jieba's tokenizer.

    Call once at module level before any tokenisation.
    Safe to call multiple times — jieba ignores duplicates.
    """
    import jieba

    for word in get_all_words():
        # Use suggest_freq for compound words to nudge the Viterbi
        # segmentation; add_word for names.
        if len(word) <= 2 or any("\u4e00" <= ch <= "\u9fff" for ch in word):
            jieba.add_word(word, freq=1000)
        else:
            jieba.add_word(word, freq=5000)
