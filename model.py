"""轻量字符级 TextCNN：Embedding → 多窗口Conv1d → ReLU → MaxPool → 分类。"""

from collections import Counter
from dataclasses import asdict, dataclass
import re
import unicodedata

import torch
from torch import nn

LABELS = ["negative", "neutral", "positive"]
LABEL_NAMES = {"negative": "负面", "neutral": "中性", "positive": "正面"}


def normalize(text: str) -> str:
    """统一全半角与大小写，移除空白；训练和推理共用同一预处理。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    """只传入训练文本，禁止从验证/测试集构建词表。0为PAD，1为UNK。"""
    counts = Counter(char for text in texts for char in normalize(text))
    return {"<PAD>": 0, "<UNK>": 1, **{char: i + 2 for i, char in enumerate(sorted(counts))}}


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(char, 1) for char in normalize(text)[:max_length]]
    return ids + [0] * (max_length - len(ids))


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 32
    num_filters: int = 32
    kernel_sizes: tuple[int, ...] = (2, 3, 4)
    num_classes: int = 3
    dropout: float = 0.2
    max_length: int = 64

    def to_dict(self):
        return asdict(self)


class TextCNN(nn.Module):
    """默认CPU即可训练；多种窗口捕获短语模式，不依赖预训练模型。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.num_filters, kernel_size=k)
            for k in config.kernel_sizes
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_filters * len(config.kernel_sizes), config.num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch, length] → [batch, embedding_dim, length]
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = []
        lengths = input_ids.ne(0).sum(dim=1)
        for conv in self.convs:
            activations = torch.relu(conv(embedded))
            # 去除完全位于padding区域的窗口，避免模型把补零数量作为主要特征。
            positions = torch.arange(activations.size(2), device=input_ids.device)
            valid = positions.unsqueeze(0) < lengths.clamp(min=1).unsqueeze(1)
            activations = activations.masked_fill(~valid.unsqueeze(1), float("-inf"))
            features.append(activations.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(features, dim=1)))  # 原始logits
