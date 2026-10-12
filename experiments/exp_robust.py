import sys, json, time, numpy as np, cv2
sys.path.insert(0,'/home/claude/patch/backend')
exec(open('ablate.py').read().split("N=100")[0].split("sys.path.insert")[0])      # imports + rng line
import importlib
src=open('ablate.py').read(); head=src.split("N=100")[0]
head=head.replace("sys.path.insert(0,'/home/claude/repo/backend')","sys.path.insert(0,'/home/claude/patch/backend')")
exec(head)
from detection_methods.custom import CustomDefectDetector
from temporal import TemporalVoter
rng=np.random.default_rng(5); NS=100; W_=5
def det(norm):
    d=CustomDefectDetector(0,illum_norm=norm); d._save_intermediates=False; d._roi_threshold=8; d._roi_min_area=150; return d
def stress_params(rng):
    return dict(gain=float(rng.uniform(-0.05,0.05)), sh=(int(rng.integers(-2,3)),int(rng.integers(-2,3))), shadow=float(rng.uniform(0.02,0.06)))
def render(sc,p,rng):
    xx=np.linspace(0,1,W,dtype=np.float32)[None,:,None]
    return noisy(sc*(1-p['shadow']*xx),rng,gain=p['gain'],shift=p['sh'])
V={'released':(False,False),'+illum-norm':(True,False),'+vote 3/5':(False,True),'+both':(True,True)}
acc={v:dict(det_nom=[],det_str=[],fa_flick=[],fa_pers=[]) for v in V}
t_scene=0
for i in range(NS):
    sc=base_scene(rng); ref=noisy(sc,rng)
    dets={n:det(n) for n in (False,True)}
    for d in dets.values(): d.initialize(ref)
    def seq_pred(norm,frames):
        return [int(dets[norm]._detect(f).prediction) for f in frames]
    # nominal detection (single frame, no stress)
    for fn in (add_wire,add_sticker,add_stain,add_texture):
        cur=noisy(fn(sc,rng),rng)
        for v,(norm,vote) in V.items(): acc[v]['det_nom'].append(dets[norm]._detect(cur).prediction)
    # stress detection: defect persistent, stress parameters i.i.d. per frame
    for fn in (add_wire,add_sticker,add_stain,add_texture):
        dsc=fn(sc,rng); frames=[render(dsc,stress_params(rng),rng) for _ in range(W_)]
        for v,(norm,vote) in V.items():
            pr=seq_pred(norm,frames); acc[v]['det_str'].append(int(sum(pr)>=3) if vote else float(np.mean(pr)))
    # clean flicker: i.i.d. stress per frame
    frames=[render(sc,stress_params(rng),rng) for _ in range(W_)]
    for v,(norm,vote) in V.items():
        pr=seq_pred(norm,frames); acc[v]['fa_flick'].append(int(sum(pr)>=3) if vote else float(np.mean(pr)))
    # clean persistent: same stress for all frames
    p=stress_params(rng); frames=[render(sc,p,rng) for _ in range(W_)]
    for v,(norm,vote) in V.items():
        pr=seq_pred(norm,frames); acc[v]['fa_pers'].append(int(sum(pr)>=3) if vote else float(np.mean(pr)))
res={v:{k:round(100*float(np.mean(x)),1) for k,x in d.items()} for v,d in acc.items()}
print(json.dumps(res,indent=1))
# timing overhead on 640x480
f0=np.random.default_rng(1).integers(90,170,(480,640,3),dtype=np.uint8); f1=np.clip(f0.astype(np.int16)+3,0,255).astype(np.uint8)
tm={}
for norm in (False,True):
    d=det(norm); d.initialize(f0); d._detect(f1); t=time.perf_counter()
    for _ in range(40): d._detect(f1)
    tm[str(norm)]=round((time.perf_counter()-t)/40*1000,2)
print('ms per frame 640x480',tm); res['timing_ms']=tm
json.dump(res,open('robust.json','w'))
