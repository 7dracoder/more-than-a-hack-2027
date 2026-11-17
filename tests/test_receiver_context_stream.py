import numpy as np
from scripts.research.spatial_memory import prepare_receiver_context as prep


def test_stream_truncates_history_and_holds_only_past_snapshots(monkeypatch):
    calls=[]
    def observation(players,teams,ball,weights,pitch):
        tick=int(players[-1,0,0])
        assert ball[-1,0]==tick and len(players)<=51 and len(ball)<=201
        calls.append(tick)
        return np.full(10,tick,np.float32)
    monkeypatch.setattr(prep,'marginal_context',observation)
    players=np.zeros((233,2,2));players[:,:,0]=np.arange(233)[:,None]
    ball=np.zeros((233,2));ball[:,0]=np.arange(233)
    x=prep.stream(players,[0,1],ball,(105,68),{})
    ticks=np.arange(0,233,2)
    np.testing.assert_array_equal(x[:,0],ticks//10*10)
    np.testing.assert_allclose(x[:,-1],ticks%10/10)
    assert calls==list(range(0,233,10))
    players[21:]=999;ball[21:]=999
    y=prep.stream(players[:21],[0,1],ball[:21],(105,68),{})
    np.testing.assert_array_equal(x[:len(y)],y)
