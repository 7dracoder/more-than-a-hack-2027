import gc,tempfile,unittest
from pathlib import Path
import numpy as np
from possession.inference_cache import cached_block


class InferenceFeatureCacheTests(unittest.TestCase):
    def test_reuse_preserves_every_float32_bit_without_calling_the_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            values=np.random.default_rng(5).normal(size=(71,7)).astype(np.float32);names=[str(i) for i in range(7)]
            a,names_a,first=cached_block(tmp,dict(raw='input',source='producer'),lambda:(values,names),71)
            def unexpected():raise AssertionError('Producer should not run for unchanged inputs')
            b,names_b,second=cached_block(tmp,dict(raw='input',source='producer'),unexpected,71)
            self.assertEqual(a.tobytes(),values.tobytes());self.assertEqual(a.tobytes(),b.tobytes());self.assertEqual(names_a,names_b)
            self.assertFalse(first['reused']);self.assertTrue(second['reused']);self.assertFalse(a.flags.writeable)
            a._mmap.close();b._mmap.close()

    def test_changed_raw_input_or_source_builds_a_separate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts=[]
            for key in (dict(raw='a',source='a'),dict(raw='b',source='a'),dict(raw='a',source='b')):
                x,_,receipt=cached_block(tmp,key,lambda:(np.zeros((2,1),np.float32),['x']),2);receipts.append(receipt);x._mmap.close()
            self.assertEqual(len({r['key_sha256'] for r in receipts}),3)

    def test_modified_cache_or_nonfinite_producer_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            key=dict(raw='a',source='a');x,_,_=cached_block(tmp,key,lambda:(np.zeros((2,1),np.float32),['x']),2)
            path=Path(x.filename);x._mmap.close();del x;gc.collect()
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'provenance changed'):cached_block(tmp,key,lambda:None,2)
            with self.assertRaisesRegex(ValueError,'Invalid generated'):
                cached_block(tmp,dict(raw='other'),lambda:(np.full((2,1),np.nan,np.float32),['x']),2)
