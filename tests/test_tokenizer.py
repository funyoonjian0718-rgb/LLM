import tempfile
import unittest
from pathlib import Path

from tiny_issue_lm.tokenizer import EOT_TOKEN, UNK_TOKEN, CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        text = "Hello, 世界!\n"
        tokenizer = CharacterTokenizer.train([text])
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocabulary_is_deterministic(self) -> None:
        first = CharacterTokenizer.train(["cba"])
        second = CharacterTokenizer.train(["abc"])
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(first.tokens[:2], (UNK_TOKEN, EOT_TOKEN))

    def test_unknown_character(self) -> None:
        tokenizer = CharacterTokenizer.train(["abc"])
        self.assertEqual(tokenizer.encode("z"), [tokenizer.unk_id])
        self.assertEqual(tokenizer.decode([tokenizer.unk_id]), UNK_TOKEN)

    def test_save_and_load(self) -> None:
        tokenizer = CharacterTokenizer.train(["repeat me"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(CharacterTokenizer.load(path), tokenizer)


if __name__ == "__main__":
    unittest.main()

