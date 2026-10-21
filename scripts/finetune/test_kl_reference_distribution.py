"""Offline tests for the TruthfulQA/MBPP KL-reference ablation.

Run from the repository root with:
    python -m unittest discover -s scripts/finetune -p test_kl_reference_distribution.py

No network, model loading, GPU, publication, or job submission is performed.
Prepared-artifact tests are skipped when the frozen inputs are not present.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import finetune_worker_utility as utility
import kl_reference_distribution as experiment
import kl_reference_distribution_worker as worker


class CharacterTokenizer:
    """Deterministic chat tokenizer exposing the same interfaces as HF."""

    def __init__(self, mismatch=False):
        self.mismatch = mismatch

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = ''.join(
            '<' + message['role'] + '>' + message['content']
            + '</' + message['role'] + '>'
            for message in messages
        )
        if add_generation_prompt:
            text += '<assistant>'
            if self.mismatch:
                text = '!' + text
        return text

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation:
            ids = ids[:max_length]
        return {'input_ids': ids}


def example(user='Question?', answer='Answer.', tokenizer=None):
    tokenizer = tokenizer or CharacterTokenizer()
    messages = [dict(role='user', content=user), dict(role='assistant', content=answer)]
    return dict(id='example', messages=messages, metadata=dict(
        anchor_condition='truthfulqa_original',
        token_profile=experiment.profile(tokenizer, messages),
    ))


def tokenized(records, tokenizer=None):
    return utility.build_tokenized_rows(records, tokenizer or CharacterTokenizer(), dict(
        max_seq_length=2048,
        train_on_responses_only=True,
    ))


class SourceContractTests(unittest.TestCase):
    def test_exact_pinned_sources(self):
        expected = {
            'truthfulqa': (
                'truthfulqa/truthful_qa',
                '741b8276f2d1982aa3d5b832d3ee81ed3b896490',
                'generation/validation-00000-of-00001.parquet',
                817, 'question', 'best_answer',
                'dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5',
            ),
            'mbpp': (
                'google-research-datasets/mbpp',
                '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
                'full/train-00000-of-00001.parquet',
                374, 'text', 'code',
                '09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae',
            ),
        }
        self.assertEqual(set(experiment.SOURCES), set(expected))
        keys = ('repo', 'revision', 'file', 'expected_rows', 'prompt_field', 'answer_field', 'sha256')
        for dataset, values in expected.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(tuple(experiment.SOURCES[dataset][key] for key in keys), values)
        self.assertEqual(experiment.N, 221)
        self.assertEqual(set(experiment.CONDITIONS), worker.ANCHOR_CONDITIONS)
        self.assertEqual(len(experiment.CONDITIONS), 4)

    def test_normalization_is_only_case_and_whitespace(self):
        self.assertEqual(experiment.normalized('  A\tB\nC  '), 'a b c')
        self.assertEqual(experiment.normalized('Q:  2+2?'), 'q: 2+2?')
        self.assertNotEqual(experiment.normalized('Question?'), experiment.normalized('Question'))


class TokenProfileTests(unittest.TestCase):
    def assert_profile_matches_utility(self, record, tokenizer):
        rows = tokenized([record], tokenizer)
        actual = record['metadata']['token_profile']
        self.assertEqual(actual['truncated_tokens'], len(rows[0]['input_ids']))
        self.assertEqual(actual['kl_tokens'], sum(label != -100 for label in rows[0]['labels']))
        self.assertEqual(rows[0]['labels'][:actual['prompt_tokens']], [-100] * actual['prompt_tokens'])
        self.assertEqual(rows[0]['labels'][actual['prompt_tokens']:], rows[0]['input_ids'][actual['prompt_tokens']:])
        self.assertEqual(rows[0]['labels'][0], -100)
        return actual

    def test_untruncated_profile_matches_training(self):
        tokenizer = CharacterTokenizer()
        record = example(tokenizer=tokenizer)
        result = self.assert_profile_matches_utility(record, tokenizer)
        self.assertEqual(result['user_tokens'], len('Question?'))
        self.assertEqual(result['assistant_tokens'], len('Answer.'))
        self.assertEqual(result['full_tokens'], result['truncated_tokens'])
        self.assertGreater(result['kl_tokens'], 0)

    def test_truncated_answer_profile_matches_training(self):
        tokenizer = CharacterTokenizer()
        record = example(answer='a' * 3000, tokenizer=tokenizer)
        result = self.assert_profile_matches_utility(record, tokenizer)
        self.assertGreater(result['full_tokens'], 2048)
        self.assertEqual(result['truncated_tokens'], 2048)
        self.assertEqual(result['kl_tokens'], 2048 - result['prompt_tokens'])

    def test_prompt_prefix_mismatch_rejected_by_both_paths(self):
        tokenizer = CharacterTokenizer(mismatch=True)
        messages = [dict(role='user', content='q'), dict(role='assistant', content='a')]
        with self.assertRaisesRegex(AssertionError, 'prefix mismatch'):
            experiment.profile(tokenizer, messages)
        with self.assertRaisesRegex(ValueError, 'not a prefix'):
            tokenized([dict(messages=messages)], tokenizer)

    def test_prompt_exhausting_sequence_rejected_by_both_paths(self):
        tokenizer = CharacterTokenizer()
        messages = [dict(role='user', content='q' * 2100), dict(role='assistant', content='a')]
        with self.assertRaisesRegex(AssertionError, 'No assistant KL positions'):
            experiment.profile(tokenizer, messages)
        with self.assertRaisesRegex(ValueError, 'no assistant tokens after truncation'):
            tokenized([dict(messages=messages)], tokenizer)

    def test_reference_length_summary_keeps_totals_and_truncations(self):
        records = [example(answer='a'), example(answer='a' * 3000), example(answer='abc')]
        result = experiment.summarize_lengths(records)
        self.assertEqual(result['rows'], 3)
        self.assertFalse(result['lengths_exactly_matched'])
        self.assertEqual(result['truncated_rows'], 1)
        for key in ('user_tokens', 'assistant_tokens', 'prompt_tokens', 'full_tokens', 'kl_tokens'):
            values = sorted(record['metadata']['token_profile'][key] for record in records)
            with self.subTest(field=key):
                self.assertEqual(result[key], dict(
                    min=values[0], max=values[-1], mean=sum(values) / 3,
                    median=values[1], sum=sum(values),
                ))


class WorkerVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.records = [example(answer='a' * (1 + i % 7)) for i in range(221)]
        self.rows = tokenized(self.records, self.tokenizer)

    def test_length_summary_empty_and_nonempty(self):
        self.assertEqual(worker.length_summary([]), dict(count=0, total=0, min=None, max=None, mean=None))
        self.assertEqual(worker.length_summary([1, 3, 5]), dict(count=3, total=9, min=1, max=5, mean=3))

    def test_frozen_profiles_match_training(self):
        result = worker.verify_reference_rows(self.records, self.rows, self.tokenizer, utility)
        for field in ('prompt_tokens', 'full_tokens', 'truncated_tokens', 'kl_tokens'):
            values = [record['metadata']['token_profile'][field] for record in self.records]
            self.assertEqual(result[field], worker.length_summary(values))

    def test_every_verified_profile_field_is_checked(self):
        for field in ('prompt_tokens', 'full_tokens', 'truncated_tokens', 'kl_tokens'):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[13]['metadata']['token_profile'][field] += 1
                with self.assertRaises(AssertionError):
                    worker.verify_reference_rows(records, self.rows, self.tokenizer, utility)

    def test_reference_count_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            worker.verify_reference_rows(self.records[:-1], self.rows[:-1], self.tokenizer, utility)

    def test_explicit_220_rows_allowed_only_for_mbpp_self_distilled(self):
        records=copy.deepcopy(self.records[:-1])
        for r in records:r['metadata']['anchor_condition']='mbpp_self_distilled'
        result=worker.verify_reference_rows(records,self.rows[:-1],self.tokenizer,utility,expected_rows=220)
        self.assertEqual(result['kl_tokens']['count'],220)
        records[0]['metadata']['anchor_condition']='mbpp_original'
        with self.assertRaises(AssertionError):
            worker.verify_reference_rows(records,self.rows[:-1],self.tokenizer,utility,expected_rows=220)

    def test_arbitrary_reference_counts_rejected(self):
        for count in [219,222]:
            with self.assertRaises(AssertionError):
                worker.validate_reference_count(self.records[:count],count)

    def test_unmasked_first_token_rejected_even_with_adjusted_profile(self):
        rows = copy.deepcopy(self.rows)
        records = copy.deepcopy(self.records)
        rows[0]['labels'][0] = rows[0]['input_ids'][0]
        records[0]['metadata']['token_profile']['kl_tokens'] += 1
        with self.assertRaises(AssertionError):
            worker.verify_reference_rows(records, rows, self.tokenizer, utility)


class PreparedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = experiment.BASE
        needed = [base / 'manifest.json', base / 'full/config.json', base / 'full/requests.jsonl']
        needed += [base / 'data' / (key + '_original.jsonl') for key in experiment.SOURCES]
        needed += [base / 'source' / (key + '.parquet') for key in experiment.SOURCES]
        if not all(path.is_file() for path in needed):
            raise unittest.SkipTest('Frozen preparation is not present in this checkout')
        cls.base = base
        cls.manifest = json.loads((base / 'manifest.json').read_text())
        cls.config = json.loads((base / 'full/config.json').read_text())
        cls.requests = experiment.records(base / 'full/requests.jsonl')

    def test_generation_has_only_paired_user_instructions(self):
        self.assertEqual(len(self.requests), 442)
        self.assertEqual(len({row['completion_id'] for row in self.requests}), 442)
        self.assertEqual(len({row['eval_id'] for row in self.requests}), 442)
        originals = {}
        for dataset in experiment.SOURCES:
            rows = experiment.records(self.base / 'data' / (dataset + '_original.jsonl'))
            self.assertEqual(len(rows), 221)
            self.assertEqual(len({row['id'] for row in rows}), 221)
            for row in rows:
                self.assertEqual([message['role'] for message in row['messages']], ['user', 'assistant'])
                self.assertTrue(all(message['content'].strip() for message in row['messages']))
                self.assertEqual(row['metadata']['anchor_condition'], dataset + '_original')
                originals[row['id']] = row
        for i, request in enumerate(self.requests):
            self.assertEqual(request['messages'], originals[request['eval_id']]['messages'][:-1])
            self.assertEqual([message['role'] for message in request['messages']], ['user'])
            self.assertEqual(request['prompt_index'], i)
            self.assertEqual(request['sample_index'], 0)
            self.assertEqual(request['method'], 'vanilla')
            self.assertIsNone(request['training_seed'])

    def test_generation_and_model_contract(self):
        config = self.config
        self.assertEqual(config['request_count'], 442)
        self.assertEqual(config['model'], experiment.MODEL)
        self.assertEqual(config['models'][experiment.GEN_KEY]['revision'], experiment.MODEL_REV)
        self.assertTrue(config['enable_thinking'])
        self.assertEqual(config['sampling']['n'], 1)
        self.assertEqual(config['sampling']['temperature'], 1.0)
        self.assertEqual(config['sampling']['top_p'], 1.0)
        self.assertEqual(config['sampling']['top_k'], 50)
        self.assertEqual(config['sampling']['max_tokens'], 8192)
        self.assertEqual(config['engine']['max_model_len'], 10240)
        raw = (self.base / 'full/requests.jsonl').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(config['requests_sha256'], digest)
        self.assertEqual(self.manifest['generation_requests_sha256'], digest)
        self.assertEqual(self.manifest['generation_config_sha256'], hashlib.sha256(
            (self.base / 'full/config.json').read_bytes()).hexdigest())

    def test_source_snapshots_match_verified_hashes_and_schema(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('Optional pyarrow is not installed')
        for dataset, specification in experiment.SOURCES.items():
            with self.subTest(dataset=dataset):
                path = self.base / 'source' / (dataset + '.parquet')
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), specification['sha256'])
                table = pq.read_table(path)
                self.assertEqual(table.num_rows, specification['expected_rows'])
                self.assertIn(specification['prompt_field'], table.column_names)
                self.assertIn(specification['answer_field'], table.column_names)


class ExclusionPolicyTests(unittest.TestCase):
    def test_waiver_scope_integrity_and_failed_row_are_checked(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            final=[dict(id=f'mbpp_{i}') for i in range(220)]
            experiment.write_records(base/'data/mbpp_self_distilled.jsonl',final)
            diagnostics=[dict(id=r['id'],ready=True) for r in final]
            diagnostics.append(dict(id='mbpp_omitted',ready=False,finish_reason='length',empty_final_answer=True,max_tokens=16384))
            experiment.write_records(base/'generation_export/mbpp_recovery_diagnostics.jsonl',diagnostics)
            policy=dict(user_approved=True,source_rows=221,retained_rows=220,excluded_ids=['mbpp_omitted'],
                reference_sha256=hashlib.sha256((base/'data/mbpp_self_distilled.jsonl').read_bytes()).hexdigest())
            policy_path=base/'generation_export/mbpp_exclusion_20261021.json'
            experiment.shared.save(policy_path,policy)
            with patch.object(experiment,'BASE',base):
                self.assertEqual(experiment.expected_reference_rows('mbpp_self_distilled'),220)
                self.assertEqual(experiment.expected_reference_rows('mbpp_original'),221)
                self.assertEqual(experiment.expected_reference_rows('truthfulqa_self_distilled'),221)
                policy['excluded_ids']=['wrong-row']
                experiment.shared.save(policy_path,policy)
                with self.assertRaises(AssertionError):
                    experiment.expected_reference_rows('mbpp_self_distilled')


class RetryCredentialTests(unittest.TestCase):
    def test_remote_credential_survives_multiple_attempts(self):
        import tempfile
        from types import SimpleNamespace
        messages = []
        meta = dict(purpose='pilot_hf_credential', organization_id='test-org')
        # Deliberately no storage/table clients: startup must never delete remotely.
        ow = SimpleNamespace(organization_id='test-org',
             files=SimpleNamespace(get_by_id=lambda _: meta),
             run=SimpleNamespace(log=messages.append))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'pilot_credential.json'
            for _ in range(2):
                p.write_text(json.dumps(dict(hf_token='dummy-test-value')))
                token = worker.load_retry_safe_credential(ow,dict(credential_file_id='test-file'),p)
                self.assertEqual(token,'dummy-test-value')
                self.assertFalse(p.exists())
        self.assertEqual(len(messages),2)
        self.assertTrue(all(not m['credential_storage_removed'] for m in messages))
        self.assertNotIn('dummy-test-value',json.dumps(messages))

    def test_foreign_credential_rejected(self):
        import tempfile
        from types import SimpleNamespace
        ow=SimpleNamespace(organization_id='test-org',files=SimpleNamespace(
            get_by_id=lambda _:dict(purpose='pilot_hf_credential',organization_id='other-org')))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'pilot_credential.json'
            p.write_text(json.dumps(dict(hf_token='dummy-test-value')))
            with self.assertRaises(AssertionError):
                worker.load_retry_safe_credential(ow,dict(credential_file_id='test-file'),p)
            self.assertTrue(p.exists())


if __name__ == '__main__':
    unittest.main()
