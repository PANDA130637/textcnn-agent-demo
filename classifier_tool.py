"""训练权重 → CPU推理服务 → LangChain结构化工具。"""

from functools import lru_cache
import json
from pathlib import Path

import torch
from langchain_core.tools import tool
from model import LABELS, LABEL_NAMES, ModelConfig, TextCNN, encode, normalize

CHECKPOINT = Path(__file__).resolve().parent / "artifacts" / "textcnn.pt"


class SentimentClassifier:
    """一次加载，多次推理；只从可信的本项目checkpoint加载参数。"""

    def __init__(self, checkpoint: Path = CHECKPOINT):
        if not checkpoint.is_file():
            raise FileNotFoundError("缺少模型权重，请先运行 python train.py")
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or saved.get("labels") != LABELS:
            raise ValueError("不兼容的模型格式或标签映射，请重新训练")
        self.vocab = saved["vocab"]
        self.config = ModelConfig(**saved["config"])
        self.model = TextCNN(self.config).cpu()
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()  # 关闭dropout，推理结果稳定。

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not normalize(text) or len(text) > 2000:
            raise ValueError("文本不能为空，且最多2000字符")
        normalized = normalize(text)
        used = normalized[:self.config.max_length]
        ids = torch.tensor([encode(text, self.vocab, self.config.max_length)], dtype=torch.long)
        probabilities = torch.softmax(self.model(ids), dim=1)[0].tolist()
        index = max(range(3), key=lambda i: probabilities[i])
        unknown_rate = sum(char not in self.vocab for char in used) / len(used)
        return {"label": LABELS[index], "label_zh": LABEL_NAMES[LABELS[index]],
            "score": round(probabilities[index], 6),
            "probabilities": {label: round(p, 6) for label, p in zip(LABELS, probabilities)},
            "truncated": len(normalized) > self.config.max_length,
            "unknown_char_ratio": round(unknown_rate, 4),
            "note": "小型手写数据集训练；softmax分数未校准，不代表业务正确率。"}


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    return SentimentClassifier()


@tool
def classify_sentiment(text: str) -> dict:
    """调用本地PyTorch TextCNN判断中文商品/服务评价情感，输出正面、中性或负面及各类分数。

    当用户要求情感分类或分析评价时使用。普通聊天、知识问答无需调用。
    text为待分析原文，不是整段用户指令；文本最多2000字符，模型使用前64个规范化字符。
    """
    return get_classifier().predict(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="直接调用本地情感分类工具，无需API")
    parser.add_argument("text", nargs="?", default="这个产品很好用，我很满意")
    args = parser.parse_args()
    print(json.dumps(classify_sentiment.invoke({"text": args.text}), ensure_ascii=False, indent=2))
