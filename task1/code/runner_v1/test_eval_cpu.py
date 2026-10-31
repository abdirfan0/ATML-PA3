"""CPU-only arithmetic, cache, and real tiny-Qwen/harness integration tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from task1_eval import (
    RequestStore, generation_metadata, local_diagnostic,
    merge_factors, recorded_hflm_class, token_key, make_task,
)
from task1_train import AdapterLinear, seed_all


def toy_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "Q": 3, ":": 4,
             "one": 5, "two": 6, "A": 7, "1": 8, "2": 9, "yes": 10, "no": 11}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    config = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=64,
                         bos_token_id=None, pad_token_id=0, eos_token_id=2,
                         tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config).float().eval()
    model.generation_config.do_sample = False
    return model, tokenizer


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        seed_all(6304)
        torch.set_num_threads(2)

    def test_partial_cache_repair_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            store = RequestStore(path)
            store.put("a", [-2.1, True])
            store.flush()
            with path.open("a") as stream:
                stream.write('{"key":')
            recovered = RequestStore(path)
            self.assertEqual(recovered.values["a"], [-2.1, True])
            recovered.put("b", "answer")
            recovered.flush()
            self.assertEqual(RequestStore(path).values, {"a": [-2.1, True], "b": "answer"})

    def test_truncation_distinguishes_cap_eos_and_text_stop(self):
        class Decoder:
            def decode(self, ids, **kwargs):
                return "".join({1: "x", 2: "<eos>", 3: "Q:"}[i] for i in ids)
        capped = generation_metadata(Decoder(), [8], [1, 1, 1], 3, ["Q:"], {2})
        self.assertTrue(capped["truncated"])
        eos = generation_metadata(Decoder(), [8], [1, 1, 2, 0], 3, [], {2})
        self.assertFalse(eos["truncated"])
        self.assertEqual(eos["output_tokens"], 3)
        stopped = generation_metadata(Decoder(), [8], [1, 1, 3], 3, ["Q:"], {2})
        self.assertFalse(stopped["truncated"])

    def test_local_diagnostic_direction_and_shared_inputs(self):
        w0 = torch.eye(3)
        delta = torch.diag(torch.tensor([0., 2., 0.]))
        u = torch.eye(3)[:, :1]
        z = torch.eye(3)
        rho, eta, r = local_diagnostic(w0, delta, u, z)
        self.assertEqual(rho, 0.)
        self.assertAlmostEqual(eta, 2 / 3**0.5, places=6)
        torch.testing.assert_close(r, torch.tensor([0., 2., 0.]))

    def test_merging_projected_export_matches_unmerged_forward(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.linear = nn.Linear(7, 5)
        wrapped = AdapterLinear(model.block.linear, 2, 123)
        model.block.linear = wrapped
        wrapped.eval()
        with torch.no_grad():
            wrapped.B.normal_(std=0.1)
        x = torch.randn(3, 7)
        before = wrapped(x)
        factors = {"block.linear": {"A": wrapped.A.detach(), "B": wrapped.B.detach(), "scale": 2.}}
        merge_factors(model, factors)
        after = model.block.linear(x)
        self.assertIsInstance(model.block.linear, nn.Linear)
        torch.testing.assert_close(before, after)

    def test_tiny_qwen_generation_cache_and_token_accounting(self):
        from lm_eval.api.instance import Instance
        model, tokenizer = toy_model_and_tokenizer()
        with tempfile.TemporaryDirectory() as folder:
            wrapper = recorded_hflm_class()(
                pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=2,
                max_length=64, add_bos_token=False, enable_thinking=False,
                cache_dir=folder,
            )
            requests = [Instance(request_type="generate_until", doc={},
                                 arguments=(prompt, {"max_gen_toks": 4, "until": ["<eos>"], "do_sample": False}),
                                 idx=0)
                        for prompt in ["Q: one A:", "Q: two A:"]]
            first = wrapper.generate_until(requests, disable_tqdm=True)
            second = wrapper.generate_until(requests, disable_tqdm=True)
            self.assertEqual(first, second)
            for prompt in ["Q: one A:", "Q: two A:"]:
                key = token_key(wrapper.tok_encode(prompt))
                self.assertIn(key, wrapper.raw_generations)
                self.assertLessEqual(wrapper.raw_generations[key]["output_tokens"], 4)

    def test_harness_end_to_end_filters_and_likelihood_cache(self):
        from datasets import Dataset, DatasetDict
        from lm_eval.api.task import ConfigurableTask
        from lm_eval import evaluator
        model, tokenizer = toy_model_and_tokenizer()
        data = DatasetDict({"test": Dataset.from_list([
            {"question": "Q: one A:", "answer": "1", "label": 0},
            {"question": "Q: two A:", "answer": "2", "label": 1},
        ])})
        generation_task = ConfigurableTask(config={
            "task": "toy_generation", "custom_dataset": lambda **kwargs: data,
            "test_split": "test", "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
            "output_type": "generate_until", "num_fewshot": 0,
            "generation_kwargs": {"max_gen_toks": 4, "until": ["<eos>"], "do_sample": False},
            "filter_list": [{"name": "flexible-extract", "filter": [
                {"function": "regex", "regex_pattern": "([0-9]+)", "group_select": -1},
                {"function": "take_first"}]}],
            "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
        })
        likelihood_task = ConfigurableTask(config={
            "task": "toy_likelihood", "custom_dataset": lambda **kwargs: data,
            "test_split": "test", "doc_to_text": "{{question}}", "doc_to_target": "label",
            "doc_to_choice": ["yes", "no"], "output_type": "multiple_choice", "num_fewshot": 0,
            "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
        })
        with tempfile.TemporaryDirectory() as folder:
            wrapper = recorded_hflm_class()(pretrained=model, tokenizer=tokenizer,
                                            backend="causal", batch_size=2, max_length=64,
                                            add_bos_token=False, enable_thinking=False,
                                            cache_dir=folder)
            result = evaluator.simple_evaluate(model=wrapper, tasks=[generation_task, likelihood_task],
                                               num_fewshot=0, apply_chat_template=False,
                                               log_samples=True, bootstrap_iters=0)
            samples = result["samples"]["toy_generation"]
            self.assertEqual(len(samples), 2)
            for sample in samples:
                self.assertEqual(sample["filter"], "flexible-extract")
                self.assertIn("filtered_resps", sample)
                key = token_key(wrapper.tok_encode(sample["arguments"][0][0]))
                self.assertIn(key, wrapper.raw_generations)
            self.assertEqual(len(result["samples"]["toy_likelihood"]), 2)

    def test_stock_task_construction_preserves_prompts_and_filters(self):
        from datasets import Dataset, DatasetDict
        from lm_eval.tasks import TaskManager
        rows = {
            "gsm8k_cot": {"question": "one plus one?", "answer": "two #### 2"},
            "hellaswag": {"ctx": "A context", "ctx_a": "A", "ctx_b": "context",
                          "activity_label": "Activity", "endings": ["yes", "no"], "label": "0"},
            "boolq": {"passage": "Some passage", "question": "is it true", "label": 1},
            "arc_easy": {"question": "Which?", "choices": {"label": ["A", "B"],
                                                             "text": ["yes", "no"]}, "answerKey": "A"},
        }
        manager = TaskManager()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = {"datasets": {name: {"revision": "test_revision"} for name in rows}}
            for name, row in rows.items():
                ds = DatasetDict({split: Dataset.from_list([row]) for split in ["train", "validation", "test"]})
                with patch("lm_eval.api.task.datasets.load_dataset", return_value=ds) as loader:
                    task = make_task(manager, root, name, settings, local_only=False)
                self.assertEqual(loader.call_args.kwargs["revision"], "test_revision")
                self.assertEqual(task.config.num_fewshot, 0)
                text = task.doc_to_text(task.eval_docs[0])
                self.assertIsInstance(text, str)
                if name == "gsm8k_cot":
                    self.assertEqual(text, "Q: one plus one?\nA:")
                    self.assertEqual(task.config.generation_kwargs["max_gen_toks"], 1024)
                    self.assertEqual(task.config.generation_kwargs["until"], ["Q:", "</s>", "<|im_end|>"])
                    self.assertIn("flexible-extract", [entry["name"] for entry in task.config.filter_list])


if __name__ == "__main__":
    unittest.main(verbosity=2)
