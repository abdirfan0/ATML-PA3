"""Small, meaningful CPU checks before an expensive CUDA launch."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from task1_train import (
    AdapterLinear, Recipe, checkpoint_model, collate, cpu_tree,
    latest_checkpoint, parameter_groups, restore_model, save_training_checkpoint,
    seed_all, set_rng, starting_penalty, supervised_sum,
)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        seed_all(6304)
        torch.set_num_threads(2)

    def test_prompt_padding_mask_and_eot(self):
        rows = [
            {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
            {"input_ids": [1, 3, 4], "labels": [-100, 3, 4]},
        ]
        batch, labels = collate(rows, 0, "cpu")
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(labels.tolist(), [[-100, -100, 3, 4], [-100, 3, 4, -100]])
        logits = torch.randn(2, 4, 7, requires_grad=True)
        loss = supervised_sum(logits, labels)
        loss.backward()
        # First prompt-only prediction and padded prediction cannot influence loss.
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros(7)))
        self.assertTrue(torch.equal(logits.grad[1, 2:], torch.zeros(2, 7)))
        # EOT target at index 3 is predicted from index 2, and does influence loss.
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)

    def test_accumulated_token_loss_matches_effective_batch_gradient(self):
        rows = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [1, 2, 5], "labels": [-100, -100, 5]},
            {"input_ids": [1, 3, 4, 5], "labels": [-100, 3, 4, 5]},
        ]
        a = nn.Sequential(nn.Embedding(8, 6), nn.Linear(6, 8))
        b = copy.deepcopy(a)
        batch, labels = collate(rows, 0, "cpu")
        tokens = int((labels[:, 1:] != -100).sum())
        full = supervised_sum(a(batch["input_ids"]), labels) / tokens
        full.backward()
        observed = 0.0
        for row in rows:
            micro, targets = collate([row], 0, "cpu")
            loss = supervised_sum(b(micro["input_ids"]), targets) / tokens
            loss.backward()
            observed += loss.item()
        self.assertAlmostEqual(full.item(), observed, places=6)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=2e-7)

    def test_l2sp_counts_tied_parameters_once_including_biases(self):
        model = nn.Module()
        model.left = nn.Linear(4, 4)
        model.right = nn.Linear(4, 4)
        model.right.weight = model.left.weight
        reference = {name: p.detach().clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        actual = starting_penalty(model, reference)
        unique_count = sum(p.numel() for p in model.parameters())
        self.assertAlmostEqual(actual.item(), 0.5 * unique_count * 0.25**2, places=7)
        actual.backward()
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25))
        state = cpu_tree(model.state_dict())
        self.assertEqual(state["left.weight"].data_ptr(), state["right.weight"].data_ptr())

    def test_weight_decay_excludes_bias_and_one_dimensional_parameters(self):
        model = nn.Sequential(nn.Linear(4, 5), nn.LayerNorm(5))
        groups = parameter_groups(model, Recipe("full"))
        self.assertEqual([p.shape for p in groups[0]["params"]], [torch.Size([5, 4])])
        self.assertEqual(sum(p.numel() for p in groups[1]["params"]), 15)

    def test_projection_constraints_and_matched_initialization(self):
        base = nn.Linear(13, 11, bias=False)
        u, _, vh = torch.linalg.svd(base.weight.detach(), full_matrices=False)
        standard = AdapterLinear(copy.deepcopy(base), 4, 123)
        protected = AdapterLinear(copy.deepcopy(base), 4, 123, (u[:, :3], vh[:3].T))
        torch.testing.assert_close(standard.A, protected.A, rtol=0, atol=0)
        torch.testing.assert_close(standard.B, protected.B, rtol=0, atol=0)
        with torch.no_grad():
            protected.B.normal_()
        a, b = protected.factors()
        delta = protected.scale * b @ a
        self.assertLess((protected.U.T @ delta).norm().item() / delta.norm().item(), 1e-6)
        self.assertLess((delta @ protected.V).norm().item() / delta.norm().item(), 1e-6)
        protected.eval()
        x = torch.randn(2, 5, 13)
        actual = protected(x)
        expected = nn.functional.linear(x, base.weight + delta)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        actual.square().mean().backward()
        self.assertIsNone(protected.base.weight.grad)
        self.assertIsNotNone(protected.A.grad)
        self.assertIsNotNone(protected.B.grad)

    def test_adapter_recovery_contains_only_trainable_factors(self):
        model = AdapterLinear(nn.Linear(7, 5), 2, 42)
        state = checkpoint_model(model, Recipe("lora4"))
        self.assertEqual(set(state), {"A", "B"})
        restored = AdapterLinear(nn.Linear(7, 5), 2, 99)
        restore_model(restored, state, Recipe("lora4"))
        torch.testing.assert_close(restored.A, model.A)

    def test_exact_cpu_resume_restores_optimizer_scheduler_and_dropout_rng(self):
        def new_model():
            return nn.Sequential(nn.Linear(4, 6), nn.Dropout(0.4), nn.Linear(6, 2))
        def optimizer_for(model):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1 / (n + 1))
            return optimizer, scheduler
        def step(model, optimizer, scheduler, x):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
        x = torch.randn(3, 4)
        model = new_model()
        optimizer, scheduler = optimizer_for(model)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        step(model, optimizer, scheduler, x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            save_training_checkpoint(path, model, optimizer, scheduler, scaler,
                                     Recipe("full"), {"test": True}, 1, 3.0, 0)
            step(model, optimizer, scheduler, x)
            expected = copy.deepcopy(model.state_dict())
            restored = new_model()
            opt2, sched2 = optimizer_for(restored)
            saved = torch.load(path, weights_only=False)
            restore_model(restored, saved["model"], Recipe("full"))
            opt2.load_state_dict(saved["optimizer"])
            sched2.load_state_dict(saved["scheduler"])
            set_rng(saved["rng"])
            step(restored, opt2, sched2, x)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), sched2.state_dict())
            self.assertEqual(saved["data_position"], 32)

    def test_incomplete_checkpoint_is_not_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in (312, 625):
                directory = root / f"update_{index:04d}"
                directory.mkdir()
                (directory / "training_state.pt").touch()
            (root / "update_0312/complete.json").write_text(json.dumps({"update": 312}))
            self.assertEqual(latest_checkpoint(root).parent.name, "update_0312")


if __name__ == "__main__":
    unittest.main(verbosity=2)
