import unittest
import numpy as np
from possession.feature_normalization import preserve_unit_masks


class UnitMaskNormalizationTests(unittest.TestCase):
    def trajectory_schema(self):
        from possession.features_v25 import OFFSETS,COMPONENTS
        return ['old'+str(i) for i in range(780)]+[f'{side}_tube{rank}_o{offset}_{component}'
            for side in ('home','away') for rank in (1,2,3) for offset in OFFSETS for component in COMPONENTS]

    def test_physical_trajectory_units_preserve_prefix_and_equal_player_representations(self):
        from possession.feature_normalization import preserve_trajectory_units
        names=self.trajectory_schema();mean=np.arange(960,dtype=np.float32);scale=mean+1
        a,b=preserve_trajectory_units(mean,scale,names)
        np.testing.assert_array_equal(a[:780],mean[:780]);np.testing.assert_array_equal(b[:780],scale[:780])
        raw=np.tile(np.arange(30,dtype=np.float32),6)
        values=((raw-a[-180:])/b[-180:]).reshape(6,30)
        for player in values:np.testing.assert_array_equal(player,values[0])
        np.testing.assert_array_equal(mean,np.arange(960,dtype=np.float32))

    def test_physical_trajectory_units_reject_reordered_or_truncated_schema(self):
        from possession.feature_normalization import preserve_trajectory_units
        names=self.trajectory_schema();names[-1],names[-2]=names[-2],names[-1]
        with self.assertRaises(ValueError):preserve_trajectory_units(np.zeros(960),np.ones(960),names)
        with self.assertRaises(ValueError):preserve_trajectory_units(np.zeros(959),np.ones(959),self.trajectory_schema())

    def test_disabled_policy_preserves_objects_exactly(self):
        mean=np.array([2.,.9],np.float32);scale=np.array([3.,.3],np.float32)
        a,b=preserve_unit_masks(mean,scale,['distance','tube_known'],[])
        self.assertIs(a,mean);self.assertIs(b,scale)

    def test_only_declared_mask_changes_and_binary_values_survive(self):
        mean=np.array([2.,.9,.8],np.float32);scale=np.array([3.,.3,.4],np.float32)
        a,b=preserve_unit_masks(mean,scale,['distance','old_known','tube_known'],['tube_known'])
        np.testing.assert_array_equal(a[:2],mean[:2]);np.testing.assert_array_equal(b[:2],scale[:2])
        np.testing.assert_array_equal((np.array([0.,1.])-a[2])/b[2],[0.,1.])
        self.assertEqual(float(mean[2]),float(np.float32(.8)))

    def test_invalid_mask_names_duplicates_and_schema_are_rejected(self):
        for masks in (['missing_known'],['distance'],['tube_known','tube_known']):
            with self.assertRaises(ValueError):preserve_unit_masks(np.zeros(2),np.ones(2),['distance','tube_known'],masks)
        with self.assertRaises(ValueError):preserve_unit_masks(np.zeros(1),np.ones(2),['distance','tube_known'],['tube_known'])
