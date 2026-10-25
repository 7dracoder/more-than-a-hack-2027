import unittest
import numpy as np
import torch
from possession.recurrent import inference_gru


class RecurrentChunkTests(unittest.TestCase):
    def test_multilayer_bidirectional_output_and_hidden_match_whole_sequence(self):
        torch.set_num_threads(2); torch.manual_seed(23)
        for bidirectional in (False,True):
            model=torch.nn.GRU(5,7,num_layers=2,bidirectional=bidirectional,batch_first=True,dropout=.2).double().eval()
            x=torch.randn(2,101,5,dtype=torch.float64)
            pointers=[p.data_ptr() for p in model.parameters()]
            rng=torch.get_rng_state().clone()
            with torch.inference_mode():
                expected,h=model(x)
                actual,ha=inference_gru(model,x,17)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(ha,h,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            self.assertEqual(pointers,[p.data_ptr() for p in model.parameters()])

    def test_time_first_layout_preserves_direction_order(self):
        torch.set_num_threads(2); torch.manual_seed(24)
        model=torch.nn.GRU(3,5,bidirectional=True,batch_first=False).eval()
        x=torch.randn(70,2,3)
        with torch.inference_mode():
            a,ah=model(x); b,bh=inference_gru(model,x,11)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(ah,bh,rtol=1e-6,atol=1e-6)

    def test_carrying_state_is_independent_of_chunk_size(self):
        torch.set_num_threads(2); torch.manual_seed(25)
        model=torch.nn.GRU(3,5,bidirectional=True,batch_first=True).eval()
        x=torch.randn(2,111,3)
        with torch.inference_mode():
            a,_=inference_gru(model,x,13); b,_=inference_gru(model,x,29)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
