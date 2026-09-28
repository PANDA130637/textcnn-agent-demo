"""CPU训练、验证集选模、独立测试评估和checkpoint保存。无需API密钥。"""

import argparse
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import LABELS, ModelConfig, TextCNN, build_vocab, encode, normalize

ROOT = Path(__file__).resolve().parent


def load_data(path: Path):
    """固定划分随数据一起提交；检查重复文本，防止训练/测试泄漏。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        text = normalize(row["text"])
        if not text or text in seen or row["label"] not in LABELS or row["split"] not in splits:
            raise ValueError("数据存在空文本、重复文本、无效标签或无效划分")
        seen.add(text)
        splits[row["split"]].append(row)
    if any(set(row["label"] for row in values) != set(LABELS) for values in splits.values()):
        raise ValueError("每个划分必须包含全部三个类别")
    return splits


def make_loader(rows, vocab, max_length, batch_size, shuffle=False):
    x = torch.tensor([encode(row["text"], vocab, max_length) for row in rows], dtype=torch.long)
    y = torch.tensor([LABELS.index(row["label"]) for row in rows], dtype=torch.long)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    expected, predicted = [], []
    for x, y in loader:
        expected.extend(y.tolist())
        predicted.extend(model(x).argmax(dim=1).tolist())
    return {"accuracy": accuracy_score(expected, predicted),
            "macro_f1": f1_score(expected, predicted, labels=[0, 1, 2], average="macro", zero_division=0),
            "classification_report": classification_report(expected, predicted, labels=[0, 1, 2],
                target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1, 2]).tolist()}


def train(data_path: Path, output_dir: Path, epochs=35, seed=42):
    if not 1 <= epochs <= 500:
        raise ValueError("epochs必须在1到500之间")
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    splits = load_data(data_path)
    vocab = build_vocab([row["text"] for row in splits["train"]])
    config = ModelConfig(vocab_size=len(vocab))
    loaders = {name: make_loader(rows, vocab, config.max_length, 16, name == "train")
               for name, rows in splits.items()}
    model = TextCNN(config).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()  # 接收logits；训练前不要先softmax。
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        validation = evaluate(model, loaders["validation"])
        record = {"epoch": epoch, "train_loss": loss_sum / len(splits["train"]),
                  "validation_macro_f1": validation["macro_f1"]}
        history.append(record)
        print(f"epoch={epoch:02d} loss={record['train_loss']:.4f} val_macro_f1={validation['macro_f1']:.4f}")
        if validation["macro_f1"] > best_score:
            best_score, best_epoch, stale = validation["macro_f1"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 8:
            print("验证指标连续8轮未提升，提前停止。")
            break
    model.load_state_dict(best_state)
    # 测试集仅在选定最终模型之后评估一次，不参与超参数或轮数选择。
    test_metrics = evaluate(model, loaders["test"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format_version": 1, "state_dict": best_state, "config": config.to_dict(),
                  "vocab": vocab, "labels": LABELS, "seed": seed, "best_epoch": best_epoch}
    temporary = output_dir / "textcnn.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / "textcnn.pt")
    metrics = {"dataset": "handwritten_demo_v1", "seed": seed,
        "torch_version": str(torch.__version__), "device": "cpu",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
        "test": test_metrics, "training_history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"独立测试 accuracy={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"已保存：{output_dir / 'textcnn.pt'}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU TextCNN三分类训练")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sentiment.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output_dir, args.epochs, args.seed)
