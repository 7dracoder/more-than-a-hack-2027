import tempfile
from pathlib import Path
import unittest
import numpy as np
from possession.height import read_ball_height,align_height,height_regression_loss


class HeightTargetTests(unittest.TestCase):
    def test_auxiliary_model_inference_needs_only_coordinate_features(self):
        import torch
        from possession.temporal import make_model
        model=make_model(12,dict(architecture='height_hybrid',hidden_channels=8,dropout=0.)).eval()
        x=torch.randn(2,12,31)
        with torch.no_grad():
            state,boundary,height=model.forward_with_auxiliary(x)
            direct=model(x)
        self.assertEqual(state.shape,(2,3,31))
        self.assertEqual(boundary.shape,(2,31))
        self.assertEqual(height.shape,(2,31))
        torch.testing.assert_close(state,direct,rtol=0.,atol=0.)

    def test_height_loss_updates_shared_representation(self):
        import torch
        from possession.boundary import HeightBoundaryHybrid
        model=HeightBoundaryHybrid(12,8,0.)
        state,boundary,height=model.forward_with_auxiliary(torch.randn(2,12,31))
        height_regression_loss(height,torch.ones_like(height)).backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()),0.)
        self.assertIsNone(model.head[-1].weight.grad)

    def test_reader_ignores_player_z_and_keeps_ball_periods_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'positions.xml'
            path.write_text('<Root><FrameSet TeamId="H" PersonId="1" GameSection="firstHalf"><Frame N="10" Z="99"/></FrameSet><FrameSet TeamId="BALL" GameSection="firstHalf"><Frame N="10" Z="0.1"/><Frame N="12" Z="2"/></FrameSet><FrameSet TeamId="BALL" GameSection="secondHalf"><Frame N="20" Z="4"/></FrameSet></Root>')
            result=read_ball_height(path)
            np.testing.assert_array_equal(result['firstHalf'][0],[10,12])
            np.testing.assert_allclose(result['firstHalf'][1],[.1,2])
            np.testing.assert_allclose(result['secondHalf'][1],[4])

    def test_alignment_does_not_fill_missing_height_or_shift_frames(self):
        z=align_height(np.arange(10,14),np.array([10,12]),np.array([.1,2.],np.float32))
        self.assertTrue(np.isnan(z[[1,3]]).all())
        np.testing.assert_allclose(z[[0,2]],[.1,2])
        with self.assertRaises(ValueError): align_height(np.arange(4),np.array([1,1]),np.array([0.,1.]))

    def test_missing_auxiliary_targets_do_not_create_gradients(self):
        import torch
        prediction=torch.tensor([[0.,1.,3.]],requires_grad=True)
        target=torch.tensor([[0.,float('nan'),2.]])
        loss=height_regression_loss(prediction,target); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(prediction.grad[0,1]),0.)
        self.assertGreater(float(prediction.grad[0,2]),0.)
