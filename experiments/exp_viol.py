import sys, json, numpy as np, cv2
src=open('ablate.py').read().split("N=100")[0].replace("sys.path.insert(0,'/home/claude/repo/backend')","sys.path.insert(0,'/home/claude/patch/backend')")
exec(src)
from detection_methods.custom import CustomDefectDetector
rng=np.random.default_rng(9)
def det(n):
    d=CustomDefectDetector(0,illum_norm=n); d._save_intermediates=False; d._roi_threshold=8; d._roi_min_area=150; return d
def cct(img,da,db):                                   # colour-temperature shift in LAB (a,b offsets)
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32); lab[...,1]+=da; lab[...,2]+=db
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
def blob(img,amp,r):
    cx,cy=int(rng.integers(60,W-60)),int(rng.integers(50,H-50)); yy,xx=np.mgrid[0:H,0:W]
    a=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*(r/2)**2))[...,None]
    return np.clip(img.astype(np.float32)*(1+amp*a),0,255).astype(np.uint8)
conds={'colour-temp shift (a,b ±4)':lambda im:cct(im,float(rng.choice([-4,4])),float(rng.choice([-4,4]))),
       'local highlight (+25%, r=40)':lambda im:blob(im,0.25,40),
       'local highlight (+25%, r=15)':lambda im:blob(im,0.25,15)}
out={}
for name,fn in conds.items():
    c={False:0,True:0}
    for i in range(100):
        sc=base_scene(rng); ref=noisy(sc,rng); cur=noisy(fn(sc.astype(np.uint8)),rng,sigma=2.5)
        for n in (False,True):
            d=det(n); d.initialize(ref); c[n]+=int(d._detect(cur).prediction)
    out[name]={'released':c[False],'illum_norm':c[True]}; print(name,out[name])
json.dump(out,open('viol.json','w'))
