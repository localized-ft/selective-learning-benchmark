import unittest
import kl_reference_litellm_judge as proxy


class ProxyProtocolTests(unittest.TestCase):
    def test_sdk_payload_preserves_provider_and_reasoning(self):
        body=dict(model='openrouter/deepseek/deepseek-v4-flash',messages=[dict(role='user',content='rubric')],
            max_tokens=256,temperature=1,top_p=1,reasoning=dict(effort='none'),provider=dict(only=['alibaba/fp8/us-east-1'],allow_fallbacks=False))
        args=proxy.sdk_arguments(body)
        reconstructed=dict(args);extra=reconstructed.pop('extra_body')
        reconstructed.update(extra)
        self.assertEqual(reconstructed,body)

    def test_luna_payload_adds_no_sampling_or_cache_fields(self):
        cfg=dict(body=dict(model='openrouter/openai/gpt-6-luna',max_tokens=2000))
        self.assertEqual(proxy.payload(cfg,dict(prompt='p')),dict(model='openrouter/openai/gpt-6-luna',max_tokens=2000,messages=[dict(role='user',content='p')]))

    def test_deepseek_rejects_other_provider_and_reasoning(self):
        self.assertEqual(proxy.parse('deepseek',{},dict(provider='Other'))['error_type'],'UnexpectedProvider')
        data=dict(choices=[dict(message=dict(content='50',reasoning='unexpected'))])
        self.assertEqual(proxy.parse('deepseek',{},data)['error_type'],'UnexpectedReasoning')

    def test_missing_provider_is_not_fabricated(self):
        request=dict(score_name='coherence',task_id='bad_medical_advice',answer_regex=None,score_map=None)
        result=proxy.parse('deepseek',request,dict(choices=[dict(message=dict(content='75'),finish_reason='stop')]))
        self.assertEqual(result['status'],'scored')
        self.assertEqual(result['score'],75)
        self.assertFalse(result['provider_independently_reported'])
        self.assertIsNone(result['provider'])


if __name__=='__main__':unittest.main()
