import numpy as np
from possession.tiled_ball_detection import crop_windows,translate_boxes,nms,append_nonoverlapping


def test_crop_corners_and_translation_preserve_original_image_clock():
    assert crop_windows(100,50)==[(0,0,60,30),(40,0,100,30),(0,20,60,50),(40,20,100,50)]
    np.testing.assert_array_equal(translate_boxes([[2,3,8,9]],(40,20,100,50),100,50),[[42,23,48,29]])
    assert crop_windows(1,1)==[(0,0,1,1)]


def test_conservative_union_does_not_replace_an_existing_box_with_higher_new_score():
    boxes,scores=append_nonoverlapping([[0,0,10,10]],[.6],[[1,1,11,11],[30,30,35,35]],[.99,.8])
    np.testing.assert_array_equal(boxes,[[0,0,10,10],[30,30,35,35]])
    np.testing.assert_array_equal(scores,[.6,.8])
    selected,confidence=nms([[0,0,10,10],[1,1,11,11],[30,30,35,35]],[.6,.99,.8])
    np.testing.assert_array_equal(selected,[[1,1,11,11],[30,30,35,35]])
