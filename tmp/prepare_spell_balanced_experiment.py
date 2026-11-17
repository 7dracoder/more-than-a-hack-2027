import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'scripts/research/spatial_memory/train_occurrence_safe_geometry.py'
destination=source.with_name('train_spell_balanced_geometry.py')
assert not destination.exists()
text=source.read_text()
text=text.replace('Retrain expanded external expert with occurrence-preserving raw team exchange.','Test equal-run strong training weights with frozen external selection guards.')
text=text.replace('configs/occurrence_safe_geometry_adaptation.json','configs/spell_balanced_geometry_adaptation.json')
text=text.replace('scripts/research/spatial_memory/train_occurrence_safe_geometry.py','scripts/research/spatial_memory/train_spell_balanced_geometry.py')
text=text.replace("training_phase='occurrence_safe_geometry_strong_HAD_weak_HA'","training_phase='equal_run_strong_HAD_weak_HA'")
needle="    strong,weak,strong_dev,weak_dev,caches = load_data(split,args.weak_study,initial['names'])\n"
assert text.count(needle)==1
text=text.replace(needle,needle+'''    # Reweight only TRAIN strong labels. Development targets and weak masks are unchanged.
    from possession.temporal import training_weights
    inventory=[]
    for h in strong:
        labels=np.array(['H','A','D','X'])[np.where(h['y']>=0,h['y'],3)]
        old=h['w'].copy();h['w']=training_weights(labels,config['strong_run_weight_power'])
        assert np.array_equal(old>0,h['w']>0) and np.isfinite(h['w']).all()
        cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
        masses=[float(h['w'][a:b].sum()) for a,b in zip(cuts[:-1],cuts[1:]) if labels[a] in 'HAD' and b-a>=5]
        assert np.allclose(masses,masses[0],rtol=1e-5)
        inventory.append(dict(match=h['match'],half=h['half'],view=h['view'],known_frames=int((h['w']>0).sum()),
            minimum_positive_weight=float(h['w'][h['w']>0].min()),maximum_weight=float(h['w'].max()),equal_mass_runs=len(masses)))
    dump(args.out/'training_weight_inventory.json',dict(rows=inventory,training_only=True,power=config['strong_run_weight_power']))
''')
destination.write_text(text)
cfg=json.loads((root/'configs/occurrence_safe_geometry_adaptation.json').read_text())
cfg['strong_run_weight_power']=1.0
cfg['policy']='Fixed 30-epoch exploratory external study. Change only strong TRAIN run weighting from inverse square root to inverse duration, with a five-frame floor. Same seed, inputs, initialization and weak loss as occurrence-safe control. No supplied-match tuning or independent improvement claim.'
cfg['augmentation']='Unchanged occurrence-preserving raw team permutation.'
(root/'configs/spell_balanced_geometry_adaptation.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Created frozen equal-run weighting experiment')
