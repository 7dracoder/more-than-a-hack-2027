import copy,gc,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from possession.mapped_features import mapped_member,mapped_normalized
from possession.temporal import normalized,symmetric_normalization


class MappedFeatureTests(unittest.TestCase):
    def fixture(self,root):
        values=np.random.default_rng(91).normal(size=(197,7)).astype(np.float32)*100
        archive=root/'source.npz';np.savez_compressed(archive,x0=values)
        return values,archive

    def test_extracted_raw_and_chunk_normalized_array_bytes_match_the_original_float32_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values,archive=self.fixture(root);before=torch.get_rng_state().clone()
            matrix=mapped_member(archive,'x0.npy',root/'maps')
            self.assertIsInstance(matrix,np.memmap);self.assertFalse(matrix.flags.writeable)
            self.assertEqual(matrix.tobytes(),values.tobytes())
            names=['home_a','away_a','one','two','three','four','five']
            for a,b in zip(symmetric_normalization([values],names),symmetric_normalization([matrix],names)):
                self.assertEqual(a.tobytes(),b.tobytes())
            mean=np.arange(7,dtype=np.float32)/3;scale=np.arange(7,dtype=np.float32)/5+.2
            result=mapped_normalized(matrix,mean,scale,root/'maps',chunk=17)
            self.assertEqual(result.tobytes(),normalized(values,mean,scale).tobytes());self.assertFalse(result.flags.writeable)
            again=mapped_normalized(matrix,mean,scale,root/'maps')
            self.assertEqual(str(result.filename),str(again.filename))
            changed=mapped_normalized(matrix,mean+1,scale,root/'maps')
            self.assertNotEqual(str(result.filename),str(changed.filename))
            torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
            for value in (matrix,result,again,changed):value._mmap.close()

    def test_two_deterministic_optimizer_updates_match_for_memory_and_mapped_inputs(self):
        torch.set_num_threads(2);torch.manual_seed(72)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values,archive=self.fixture(root);mean=np.zeros(7,np.float32);scale=np.ones(7,np.float32)*3
            mapped=mapped_normalized(mapped_member(archive,'x0.npy',root/'maps'),mean,scale,root/'maps',chunk=17)
            ordinary=normalized(values,mean,scale);first=torch.nn.Linear(7,3);second=copy.deepcopy(first)
            target=torch.arange(15,dtype=torch.float32).reshape(5,3)/10
            for model,matrix in ((first,ordinary),(second,mapped)):
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
                for indices in ([3,5,27,88,196],[1,9,55,67,89]):
                    optimizer.zero_grad(set_to_none=True);batch=torch.from_numpy(np.stack([matrix[i] for i in indices]))
                    (model(batch)-target).square().mean().backward();optimizer.step()
            for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            mapped._mmap.close()

    def test_changed_mapped_file_is_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,archive=self.fixture(root);matrix=mapped_member(archive,'x0.npy',root/'maps');path=Path(matrix.filename)
            del matrix;gc.collect()
            with path.open('ab') as output:output.write(b'changed')
            with self.assertRaisesRegex(ValueError,'cache changed'):mapped_member(archive,'x0.npy',root/'maps')

    def test_unverified_writable_inputs_or_invalid_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values,archive=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'read-only'):mapped_normalized(values,np.zeros(7),np.ones(7),root/'maps')
            with self.assertRaises(ValueError):mapped_member(archive,'../x0.npy',root/'maps')
