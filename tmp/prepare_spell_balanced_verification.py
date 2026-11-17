from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'scripts/research/spatial_memory/verify_geometry_extension.py'
target=source.with_name('verify_spell_balanced_geometry.py')
assert not target.exists()
text=source.read_text().replace('Recompute external input normalization and reproduce a selected expanded expert.','Verify TRAIN run weighting, normalization, and selected external scores.')
needle="    strong,weak,strong_dev,weak_dev,_ = load_data(split,weak_study,initial['names'])\n"
assert text.count(needle)==1
text=text.replace(needle,needle+'''    assert config['strong_run_weight_power']==1.
    inventory=json.loads((args.run/'training_weight_inventory.json').read_text())
    assert inventory['training_only'] and inventory['power']==1. and len(inventory['rows'])==len(strong)
    from possession.temporal import training_weights
    for h,record in zip(strong,inventory['rows']):
        assert all(record[k]==h[k] for k in ('match','half','view'))
        labels=np.array(['H','A','D','X'])[np.where(h['y']>=0,h['y'],3)]
        actual=training_weights(labels,1.)
        cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
        manual=np.zeros(len(labels),np.float64);masses=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if labels[a] in 'HAD':manual[a:b]=1./max(b-a,5)
        manual/=manual[manual>0].mean()
        np.testing.assert_allclose(actual,manual,rtol=2e-7)
        assert np.array_equal(actual>0,h['w']>0)
        assert int((actual>0).sum())==record['known_frames']
        for a,b in zip(cuts[:-1],cuts[1:]):
            if labels[a] in 'HAD' and b-a>=5:masses.append(float(actual[a:b].sum()))
        assert len(masses)==record['equal_mass_runs'] and np.allclose(masses,masses[0],rtol=1e-5)
        assert np.isclose(actual.max(),record['maximum_weight'])
''')
text=text.replace('selected_scores_reproduced=True,selected_epoch=', 'training_weights_independently_recomputed=True,unknown_masks_unchanged=True,selected_scores_reproduced=True,selected_epoch=')
target.write_text(text)
print('Created independent selected-model and training-weight verifier')
