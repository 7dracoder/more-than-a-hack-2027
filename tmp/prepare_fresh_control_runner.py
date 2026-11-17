from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'scripts/research/video_ball/infer_openimages_detectors.py';target=source.with_name('infer_openimages_fresh_controls.py')
assert not target.exists()
text=source.read_text().replace('Local offline detector comparison on a frozen external image split.','Local frozen detector controls on the disjoint official external test corpus.')
text=text.replace("choices=['development','final_test'],default='development'","choices=['external_test'],default='external_test'")
text=text.replace("args.split=='final_test'","args.split=='external_test'")
text=text.replace('data/external/openimages-football-20261115','data/external/openimages-football-fresh-test-20261115')
text=text.replace('configs/openimages_football_detector.json','configs/openimages_football_fresh_test.json')
text=text.replace("not choice['final_test_accessed'] and choice['selected_model']=='rfdetr_large'","not choice['fresh_test_evaluated'] and choice['selected']['eligible']")
target.write_text(text)
print('Created disjoint-test native-control runner; inference parameters unchanged')
