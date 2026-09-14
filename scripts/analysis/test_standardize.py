import json
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from standardize import FIELDS, VERSION, canonical, convert, score_value, source_records, validate_record, verify_view

DIGEST = "a" * 64


class StandardizeTests(unittest.TestCase):
    def test_zero_is_not_missing(self):
        self.assertEqual(score_value("0"), (0.0, "numeric"))
        self.assertEqual(score_value(""), (None, "missing"))
        self.assertEqual(score_value(None), (None, "missing"))

    def test_invalid_and_categorical_remain_distinct(self):
        self.assertEqual(score_value("LLM"), (None, "non_numeric"))
        self.assertEqual(score_value("NaN"), (None, "non_finite"))
        self.assertEqual(score_value("Infinity"), (None, "non_finite"))
        self.assertEqual(score_value(True), (None, "non_numeric"))

    def test_original_scale_and_labels(self):
        row = canonical({"completion_id": "c", "axis": "undesired_generalization",
                         "score_name": "bird", "score": "19"}, DIGEST, 1, "judgment")
        self.assertEqual(row["axis"], "unintended_generalization")
        self.assertEqual(row["score"], 19.0)
        self.assertEqual(row["score_raw"], "19")
        self.assertEqual(tuple(row), FIELDS)

    def test_sparse_completions_and_empty_answers(self):
        row = canonical({"completion_id": "c", "completion": ""}, DIGEST, 1, "completion")
        self.assertEqual(row["completion"], "")
        self.assertIsNone(row["axis"])
        self.assertIsNone(row["eval_id"])
        self.assertEqual(row["score_status"], "not_applicable")
        judge = canonical({"completion_id": "c", "completion": "", "score_name": "s", "score": ""}, DIGEST, 1, "judgment")
        self.assertEqual(judge["completion"], "")

    def test_csv_quoting_unicode_and_linebreaks(self):
        data = 'completion_id,completion,score_name,score\r\nc,"é,one\ntwo",coherence,100\r\n'.encode()
        records = list(convert(data, "eval_results.csv", DIGEST))
        self.assertEqual(records[0]["completion"], "é,one\ntwo")
        self.assertEqual(records[0]["score"], 100.0)

    def test_nested_scores_keep_ordinal_and_extra_fields(self):
        source = {"completion_id": "c", "idx": 42, "scores": [
            {"score_name": "s", "score": None}, {"score_name": "coherence", "score": 0}]}
        result = list(convert(json.dumps(source).encode(), "judge_scores.jsonl", DIGEST))
        self.assertEqual([r["source_score_index"] for r in result], [0, 1])
        self.assertEqual(result[0]["extra_fields"], {"idx": 42})
        self.assertEqual(result[1]["score"], 0)

    def test_malformed_csv_is_rejected(self):
        for data in (b"completion_id,score,score\nc,1,2\n",
                     b"completion_id,score_name,score\nc,s\n"):
            with self.assertRaises(ValueError):
                list(source_records(data, "eval_results.csv"))

    def test_empty_score_list_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            list(convert(b'{"completion_id":"c","scores":[]}', "judge_scores.jsonl", DIGEST))

    def test_unknown_axis_and_schema_are_rejected(self):
        with self.assertRaises(ValueError):
            canonical({"completion_id": "c", "completion": "x", "axis": "unexpected"}, DIGEST, 1, "completion")
        row = canonical({"completion_id": "c", "completion": "x"}, DIGEST, 1, "completion")
        row["surprise"] = None
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_persisted_view_is_checked_against_original(self):
        original = b'{"completion_id":"c","completion":"original"}\n'
        class FakeArchive:
            def content(self, digest):
                return original
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = {"source_artifact_id": DIGEST, "filename": "completions.jsonl",
                      "canonical_rows": {"completion": 1}}
            (output / "sources.jsonl").write_text(json.dumps(source) + "\n")
            record = next(convert(original, "completions.jsonl", DIGEST))
            for tamper in (False, True):
                if tamper:
                    record["completion"] = "tampered"
                stored = gzip.compress((json.dumps(record) + "\n").encode())
                (output / "completions.jsonl.gz").write_bytes(stored)
                manifest = {"schema_version": VERSION, "field_order": FIELDS,
                            "records": {"completion": 1}, "files": [{"path": "completions.jsonl.gz",
                            "sha256": hashlib.sha256(stored).hexdigest()}]}
                (output / "manifest.json").write_text(json.dumps(manifest))
                if tamper:
                    with self.assertRaises(ValueError):
                        verify_view(output, FakeArchive())
                else:
                    self.assertTrue(verify_view(output, FakeArchive())["every_record_matches_source"])


if __name__ == "__main__":
    unittest.main()
