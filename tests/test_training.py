import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from tiny_issue_lm.inference import generate_text
    from tiny_issue_lm.training import TrainConfig, train_bigram


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TrainingIntegrationTests(unittest.TestCase):
    def test_train_save_reload_and_generate(self) -> None:
        text = (
            "abcd abcd abcd abcd.\n\n"
            "bcda bcda bcda bcda.\n\n"
            "cdab cdab cdab cdab."
        )
        config = TrainConfig(
            steps=5,
            batch_size=2,
            context_length=4,
            learning_rate=0.1,
            eval_interval=5,
            eval_batches=2,
            seed=11,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            result = train_bigram(text, checkpoint, config)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(result["device"], "cpu")
            generated = generate_text(
                checkpoint,
                prompt="a",
                max_new_tokens=4,
                top_k=1,
                device_name="cpu",
            )
            self.assertTrue(generated.startswith("a"))


if __name__ == "__main__":
    unittest.main()

