import numpy as np
from scripts.research.spatial_memory.train_occurrence_safe_geometry import batch as control_batch
from scripts.research.spatial_memory.train_missing_ball_curriculum import batch


def fixture():
    generator = np.random.default_rng(8)
    strong = []
    for view in ['native','masked','mixed']:
        strong.append(dict(view=view,x=generator.normal(size=(100,1088)).astype(np.float32),
            y=np.arange(100)%3,w=np.ones(100,np.float32)))
    weak = dict(x=generator.normal(size=(100,792)).astype(np.float32),
        extra=generator.normal(size=(100,296)).astype(np.float32),
        y=np.where(np.arange(100)%3==0,-100,np.arange(100)%2),
        missing_centers=np.array([50]))
    weak['w'] = (weak['y']>=0).astype(np.float32)
    weak['x'][:,0] = 1.
    weak['x'][50,0] = 0.
    config = dict(batch_size=4,weak_batch_fraction=.5,strong_native_view_probability=.5,
        sequence_length=20,weak_missing_crop_probability=0.,weak_missing_loss_multiplier=1.)
    state = dict(mean=np.zeros(1088),scale=np.ones(1088))
    return config,strong,[weak],state,np.arange(1088)


def test_disabled_curriculum_exactly_reproduces_control_batch_and_random_stream():
    config,strong,weak,state,mapping = fixture()
    a,b = np.random.default_rng(9),np.random.default_rng(9)
    actual = batch(a,np.random.default_rng(10),config,strong,weak,state,mapping)
    expected = control_batch(b,config,strong,weak,state,mapping)
    for x,y in zip(actual,expected):
        np.testing.assert_array_equal(x,y)
    assert a.bit_generator.state == b.bit_generator.state


def test_focused_batch_weights_only_known_missing_targets_and_preserves_strong_batch():
    config,strong,weak,state,mapping = fixture()
    reference = control_batch(np.random.default_rng(9),config,strong,weak,state,mapping)
    config.update(weak_missing_crop_probability=1.,weak_missing_loss_multiplier=4.)
    x,y,w,ns = batch(np.random.default_rng(9),np.random.default_rng(10),config,strong,weak,state,mapping)
    for actual,expected in zip([x[:ns],y[:ns],w[:ns]],[v[:ns] for v in reference[:3]]):
        np.testing.assert_array_equal(actual,expected)
    assert np.all(w[y<0]==0.)
    assert np.all(w[ns:][(y[ns:]>=0)&(x[ns:,0,:]<.5)]==4.)
    assert np.all(w[ns:][(y[ns:]>=0)&(x[ns:,0,:]>.5)]==1.)
    assert np.all((x[ns:,0,:]<.5).sum(1)==1)
