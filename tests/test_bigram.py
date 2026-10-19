import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from tiny_issue_lm.bigram import BigramLanguageModel


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class BigramLanguageModelTests(unittest.TestCase):
    def test_shapes(self) -> None:
        model = BigramLanguageModel(vocab_size=7)
        inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])
        logits, loss = model(inputs, inputs)
        self.assertEqual(tuple(logits.shape), (2, 3, 7))
        self.assertIsNotNone(loss)

    def test_can_overfit_one_tiny_batch(self) -> None:
        torch.manual_seed(7)
        model = BigramLanguageModel(vocab_size=5)
        inputs = torch.tensor([[0, 1, 2, 3, 4]])
        targets = torch.tensor([[1, 2, 3, 4, 0]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)

        for _ in range(200):
            _, loss = model(inputs, targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        _, final_loss = model(inputs, targets)
        assert final_loss is not None
        self.assertLess(final_loss.item(), 0.02)


if __name__ == "__main__":
    unittest.main()

