"""Offline completeness and provenance checks for length-only MBPP retries."""
import copy
import unittest
import kl_reference_generation_recovery as recovery
from test_kl_reference_distribution import CharacterTokenizer


class RecoveryTests(unittest.TestCase):
    def test_raw_verification_export_never_overwrites_dataset_readiness(self):
        names=['config.json','requests.jsonl','submission.json','status.json','artifacts.json','verification.json']
        exported=[recovery.recovery_export_name(n) for n in names]
        self.assertEqual(len(exported),len(set(exported)))
        self.assertNotIn('recovery_verification.json',exported)
        self.assertIn('recovery_generation_verification.json',exported)

    def setUp(self):
        self.requests, self.completions, self.details, self.diagnostics = [], [], [], []
        for i in range(221):
            cid = f'mbpp_{i}'
            request = dict(completion_id=cid, eval_id=cid, task_id='mbpp', checkpoint='vanilla',
                condition='no_system', axis='reference_generation', prompt_index=i, sample_index=0,
                inference_seed=i, messages=[dict(role='user', content=f'q{i}')],
                source_metadata=dict(token_profile={}, source_index=i))
            text = 'thinking' if i<17 else '<think>x</think>answer'
            self.requests.append(request)
            self.completions.append(dict(completion_id=cid, eval_id=cid, completion=text))
            self.details.append(dict(request, completion=text, finish_reason='length' if i<17 else 'stop',
                input_token_ids=[i+1], input_tokens=1, output_token_ids=[1,2], output_tokens=2))
            self.diagnostics.append(dict(id=cid, dataset='mbpp', ready=i>=17, finish_reason='length' if i<17 else 'stop'))
        self.ids = [r['completion_id'] for r in self.requests[:17]]
        self.retry_completions = [dict(r, completion='<think>x</think>recovered') for r in self.completions[:17]]
        self.retry_details = [dict(r, completion='<think>x</think>recovered', finish_reason='stop') for r in self.details[:17]]

    def merge(self):
        return recovery.merge_final(self.requests, self.completions, self.details,
            self.retry_completions, self.retry_details, self.ids, CharacterTokenizer())

    def test_only_failed_rows_selected_without_changing_requests(self):
        before = copy.deepcopy(self.requests)
        selected = recovery.select_recovery(self.requests, self.diagnostics)
        self.assertEqual(selected, before[:17])
        self.assertEqual(self.requests, before)
        selected[0]['messages'][0]['content'] = 'changed'
        self.assertEqual(self.requests, before)

    def test_nonlength_failure_rejected(self):
        self.diagnostics[0]['finish_reason'] = 'stop'
        with self.assertRaises(AssertionError):
            recovery.select_recovery(self.requests, self.diagnostics)

    def test_merged_final_preserves_originals_and_mixed_budget_provenance(self):
        originals = copy.deepcopy((self.completions,self.details))
        final, full, diagnostics = self.merge()
        self.assertEqual(len(final),221)
        self.assertEqual(len(full),221)
        self.assertTrue(all(r['ready'] for r in diagnostics))
        self.assertEqual((self.completions,self.details), originals)
        for i,row in enumerate(final):
            self.assertEqual(row['messages'][:-1],self.requests[i]['messages'])
            self.assertEqual(row['messages'][-1]['content'],'recovered' if i<17 else 'answer')
            self.assertEqual(row['metadata']['generation_max_tokens'],16384 if i<17 else 8192)
            self.assertEqual(row['metadata']['length_recovery'],i<17)

    def test_unresolved_recovery_does_not_complete_dataset(self):
        self.retry_details[0]['finish_reason']='length'
        final, _, diagnostics = self.merge()
        self.assertEqual(len(final),220)
        self.assertFalse(diagnostics[0]['ready'])

    def test_prompt_token_or_seed_mismatch_rejected(self):
        for field,value in [('input_token_ids',[999]),('inference_seed',999)]:
            with self.subTest(field=field):
                before=copy.deepcopy(self.retry_details[0])
                self.retry_details[0][field]=value
                with self.assertRaises(AssertionError):
                    self.merge()
                self.retry_details[0]=before

    def test_missing_retry_rejected(self):
        self.retry_completions.pop()
        with self.assertRaises(AssertionError):
            self.merge()

    def test_reference_budget_not_eval_budget(self):
        details=copy.deepcopy(self.retry_details)
        details[0].update(output_token_ids=[1]*8193,output_tokens=8193)
        ids=recovery.validate_generation(self.requests[:17],self.retry_completions,details,16384)
        self.assertEqual(ids,set(self.ids))
        with self.assertRaises(AssertionError):
            recovery.validate_generation(self.requests[:17],self.retry_completions,details,8192)

    def test_reference_validator_rejects_metadata_mismatch(self):
        details=copy.deepcopy(self.retry_details)
        details[0]['inference_seed']+=1
        with self.assertRaises(AssertionError):
            recovery.validate_generation(self.requests[:17],self.retry_completions,details,16384)


if __name__=='__main__':
    unittest.main()
