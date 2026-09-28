"""离线测试：运行真实CPU分类器；只有LLM部分使用明确的脚本替身。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from langchain_core.messages import AIMessage, ToolMessage

from agent_demo import run_agent
from classifier_tool import CHECKPOINT, SentimentClassifier, classify_sentiment
from model import LABELS, ModelConfig, TextCNN, build_vocab, encode
from train import ROOT, load_data


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def bind_tools(self, tools):
        assert [tool.name for tool in tools] == ["classify_sentiment"]
        return self

    def invoke(self, messages):
        self.inputs.append(list(messages))
        return next(self.replies)


def tool_call(text="这个产品很好用", ident="test-call", name="classify_sentiment"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"text": text}, "id": ident}])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.classifier = SentimentClassifier()

    def test_vocab_train_only_and_padding(self):
        vocab = build_vocab(["好用", "不错"])
        self.assertEqual(encode("好新", vocab, 4), [vocab["好"], 1, 0, 0])
        self.assertNotIn("新", vocab)
        self.assertEqual(len(encode("好" * 100, vocab, 64)), 64)

    def test_logits_shape_and_backward(self):
        model = TextCNN(ModelConfig(vocab_size=10))
        x = torch.randint(0, 10, (3, 64))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (3, 3))
        torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2])).backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_deterministic_inference_and_probabilities(self):
        first = self.classifier.predict("这个产品很好用，我很满意")
        self.assertEqual(first, self.classifier.predict("这个产品很好用，我很满意"))
        self.assertIn(first["label"], LABELS)
        self.assertAlmostEqual(sum(first["probabilities"].values()), 1, places=5)

    def test_empty_long_unknown_and_truncation(self):
        for text in ["", "   ", "好" * 2001]:
            with self.subTest(text=text[:10]), self.assertRaises(ValueError):
                self.classifier.predict(text)
        self.assertTrue(self.classifier.predict("好" * 80)["truncated"])
        self.assertEqual(self.classifier.predict("🛸🛸")["unknown_char_ratio"], 1.0)

    def test_roundtrip_weights(self):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = TextCNN(ModelConfig(**saved["config"]))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        ids = torch.tensor([encode("服务很差", saved["vocab"], 64)])
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(ids), self.classifier.model(ids)))

    def test_fixed_dataset_splits(self):
        splits = load_data(ROOT / "data" / "sentiment.jsonl")
        self.assertEqual({name: len(rows) for name, rows in splits.items()},
                         {"train": 90, "validation": 18, "test": 18})
        expected_vocab = build_vocab([row["text"] for row in splits["train"]])
        self.assertEqual(self.classifier.vocab, expected_vocab)

    def test_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(FileNotFoundError, "train.py"):
                SentimentClassifier(Path(path) / "missing.pt")

    def test_langchain_tool_schema_and_invocation(self):
        self.assertIn("text", classify_sentiment.args)
        result = classify_sentiment.invoke({"text": "物流太慢，体验糟糕"})
        self.assertEqual(set(result["probabilities"]), set(LABELS))


class AgentTests(unittest.TestCase):
    def test_no_tool_branch(self):
        model = ScriptedModel([AIMessage(content="你好")])
        answer, messages = run_agent(model, "你好", verbose=False)
        self.assertEqual(answer, "你好")
        self.assertFalse(any(isinstance(m, ToolMessage) for m in messages))

    def test_tool_result_id_and_next_llm_context(self):
        model = ScriptedModel([tool_call(), AIMessage(content="分类完成")])
        _, messages = run_agent(model, "请分类", verbose=False)
        observation = next(m for m in messages if isinstance(m, ToolMessage))
        self.assertEqual(observation.tool_call_id, "test-call")
        self.assertIn("label", json.loads(observation.content))
        self.assertIn(observation, model.inputs[1])

    def test_tool_errors_return_to_model(self):
        model = ScriptedModel([tool_call(name="missing"), tool_call(text=""), AIMessage(content="请补充文本")])
        _, messages = run_agent(model, "测试错误处理", verbose=False)
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(len(observations), 2)
        self.assertTrue(all("error" in json.loads(m.content) for m in observations))

    def test_loop_budget(self):
        model = ScriptedModel([tool_call(ident=str(i)) for i in range(4)])
        with self.assertRaisesRegex(RuntimeError, "预算"):
            run_agent(model, "一直调用", verbose=False)
        self.assertEqual(len(model.inputs), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
