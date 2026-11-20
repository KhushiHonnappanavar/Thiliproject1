import json
import tempfile
import unittest
from pathlib import Path
from cli import save_result
from sorter import LABELS, rows


class SorterTests(unittest.TestCase):
    def test_splits(self):
        train, test = rows("train"), rows("test")
        for split in [train, test]:
            self.assertEqual({r["label"] for r in split}, set(LABELS))
        self.assertFalse({r["text"] for r in train} & {r["text"] for r in test})

    def test_export_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            records = [{"request": text, "label": "venue inquiry"} for text in ["Hall?", "Terrace?"]]
            for record in records:
                save_result(path, record)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], records)
