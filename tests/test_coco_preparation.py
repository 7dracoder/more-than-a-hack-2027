import pytest

from scripts.research.video_ball.prepare_coco_calibration import allowed_license, image_url


def test_only_explicit_attribution_or_sharealike_licenses_are_accepted():
    assert allowed_license({'url':'http://creativecommons.org/licenses/by/2.0/'})
    assert allowed_license({'url':'https://creativecommons.org/licenses/by-sa/4.0/'})
    for url in ['https://creativecommons.org/licenses/by-nc/2.0/',
                'https://creativecommons.org/licenses/by-nd/2.0/',
                'https://example.com/licenses/by/4.0/', '']:
        assert not allowed_license({'url':url})


def test_image_source_is_restricted_to_publisher_and_expected_filename():
    record={'file_name':'000000000001.jpg','coco_url':'http://images.cocodataset.org/val2017/000000000001.jpg'}
    assert image_url(record)=='https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000001.jpg'
    with pytest.raises(ValueError):
        image_url(dict(record,file_name='../other.jpg'))
    with pytest.raises(ValueError):
        image_url(dict(record,coco_url='https://example.com/val2017/000000000001.jpg'))
