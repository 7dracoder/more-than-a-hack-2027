"""Extract predeclared fresh-test audio representations, including any last partial native bin."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    config_path=ROOT/'configs/audio_whistle_pilot.json';cfg=json.loads(config_path.read_text())
    folder=ROOT/'data/external/volleyball-whistles-5f23e58f6057'
    denied=restrict_external(args.out)
    acquisition=json.loads((folder/'pilot_acquisition.json').read_text());assert acquisition['complete']
    fresh=json.loads((folder/'transfer_acquisition.json').read_text());assert fresh['complete']
    for item in fresh['files']:assert sha256(ROOT/item['path'])==item['sha256']
    for item in acquisition['files']:
        assert sha256(ROOT/item['path'])==item['sha256']
    source=ROOT/acquisition['panns_source'];checkpoint=source/'Cnn14_DecisionLevelMax.pth'
    sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'pytorch'))
    import torch
    import soundfile as sf
    import librosa,torchlibrosa
    from scipy.signal import resample_poly,stft
    from scipy.ndimage import uniform_filter1d
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=importlib.util.spec_from_file_location('external_panns_models',source/'pytorch/models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.Cnn14_DecisionLevelMax(sample_rate=32000,classes_num=527,**cfg['frontend'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True);model.to(device).eval()
    dependencies=dict(torch=torch.__version__,numpy=np.__version__,librosa=librosa.__version__,torchlibrosa=torchlibrosa.__version__,soundfile=sf.__version__)
    paths=[config_path,checkpoint,source/'pytorch/models.py',source/'pytorch/pytorch_utils.py',source/'metadata/class_labels_indices.csv',folder/'pilot_acquisition.json',ROOT/'requirements-audio.txt',folder/'transfer_acquisition.json',ROOT/'configs/audio_whistle_transfer.json']
    plan=dict(config=cfg,input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},source_sha256=sha256(Path(__file__)),dependencies=dependencies,device=str(device),provided_match_used=False)
    if (args.out/'plan.json').exists():assert json.loads((args.out/'plan.json').read_text())==plan
    dump(args.out/'plan.json',plan)
    with torch.no_grad():
        probe=model(torch.zeros((1,round((cfg['chunk_seconds']+2*cfg['context_seconds'])*32000)),device=device))['framewise_output']
    assert probe.shape[2]==527 and torch.isfinite(probe).all()
    if args.smoke:
        dump(args.out/'gpu_preflight.json',dict(passed=True,shape=list(probe.shape),finite=True,provided_match_used=False));print('External audio GPU preflight passed.',flush=True);return
    captured=[]
    hook=model.fc_audioset.register_forward_pre_hook(lambda module,values: captured.append(values[0].detach()))
    records={}
    transfer=json.loads((ROOT/'configs/audio_whistle_transfer.json').read_text())
    for match in [transfer['fresh_test']]:
        path=args.out/(match+'_embeddings.npz');audio=folder/(match+'.wav')
        if path.exists() and path.with_suffix('.json').exists():
            old=json.loads(path.with_suffix('.json').read_text());assert old['plan']==plan and old['sha256']==sha256(path) and old['audio_sha256']==sha256(audio)
            records[match]=old;continue
        started=time.monotonic()
        with sf.SoundFile(audio) as recording:
            rate=recording.samplerate;total=len(recording);duration=total/rate
            frames=math.ceil(duration*100);native_frames=(frames-1+8)//32+1;embeddings=np.empty((native_frames,2048),np.float32)
            chunk=cfg['chunk_frames'];context=cfg['context_seconds'];chunks=math.ceil(frames/chunk)
            for first in range(0,chunks,cfg['batch_size']):
                waves=[];indices=[];dsp=[]
                for number in range(first,min(first+cfg['batch_size'],chunks)):
                    start=number*chunk/100;left=round((start-context)*rate);length=round((cfg['chunk_seconds']+2*context)*rate)
                    actual=max(0,left);stop=min(total,left+length);recording.seek(actual)
                    wave=np.zeros(length,np.float32);samples=recording.read(max(0,stop-actual),dtype='float32',always_2d=True).mean(1)
                    wave[actual-left:actual-left+len(samples)]=samples
                    divisor=math.gcd(rate,32000);wave=resample_poly(wave,32000//divisor,rate//divisor).astype(np.float32)
                    assert len(wave)==round((cfg['chunk_seconds']+2*context)*32000)
                    waves.append(wave);indices.append(number)
                    frequencies,_,z=stft(wave,fs=32000,nperseg=1024,noverlap=704,boundary='zeros',padded=False)
                    power=np.abs(z)**2;tonal=uniform_filter1d(power,size=cfg['dsp_peak_bins'],axis=0,mode='constant')*cfg['dsp_peak_bins']
                    band=(frequencies>=cfg['dsp_band_hz'][0])&(frequencies<=cfg['dsp_band_hz'][1])
                    score=np.divide(tonal[band].max(0),power.sum(0),out=np.zeros(power.shape[1]),where=power.sum(0)>1e-12)
                    dsp.append(np.clip(score,0,1))
                captured.clear()
                with torch.no_grad():model(torch.as_tensor(np.stack(waves),device=device))
                hidden=captured[-1].cpu().numpy()
                for row,number in enumerate(indices):
                    lo=number*32;count=min(33 if number==chunks-1 else 32,native_frames-lo)
                    if count>0:embeddings[lo:lo+count]=hidden[row,6:6+count]
                if first%40==0:print(f'{match}: {min(first+len(indices),chunks)}/{chunks} chunks',flush=True)
        assert np.isfinite(embeddings).all()
        np.savez_compressed(path,embedding=embeddings,time=(np.arange(len(embeddings))*32+8)/100,duration=np.array(duration),output_frames=np.array(frames))
        record=dict(plan=plan,sha256=sha256(path),audio_sha256=sha256(audio),frames=frames,duration_seconds=duration,seconds=time.monotonic()-started)
        dump(path.with_suffix('.json'),record);records[match]=record;print('Frozen scores '+match,flush=True)
    for name,digest in plan['input_sha256'].items():assert sha256(ROOT/name)==digest
    assert sha256(Path(__file__))==plan['source_sha256']
    dump(args.out/'embedding_freeze.json',dict(complete=True,records=records,plan_sha256=sha256(args.out/'plan.json'),labels_used_for_inference=False,provided_match_used=False,unexpected_competition_access=denied))


if __name__=='__main__':main()
