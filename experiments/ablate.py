import sys, json, numpy as np, cv2
sys.path.insert(0,'/home/claude/repo/backend')
from skimage.metrics import structural_similarity as ssim
from detection_methods.custom import CustomDefectDetector
rng=np.random.default_rng(7)
H,W=240,320
def base_scene(rng):
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    lum=150+10*np.sin(xx/55.0)+8*np.cos(yy/40.0)+0.03*(xx-W/2)
    # brushed-metal micro texture
    tex=cv2.GaussianBlur(rng.normal(0,6,(H,W)).astype(np.float32),(0,0),1.2)
    img=np.dstack([lum+tex*0.8,lum+tex*0.8+2,lum+tex*0.8+5])  # slightly warm-cool gray BGR
    return np.clip(img,0,255)
def noisy(img,rng,sigma=2.5,gain=0.0,shift=(0,0)):
    out=img*(1+gain)
    if shift!=(0,0):
        M=np.float32([[1,0,shift[0]],[0,1,shift[1]]]); out=cv2.warpAffine(out,M,(W,H),borderMode=cv2.BORDER_REFLECT)
    out=out+rng.normal(0,sigma,out.shape)
    return np.clip(out,0,255).astype(np.uint8)
def add_wire(img,rng):
    img=img.copy(); n=rng.integers(0,3)
    x0=int(rng.integers(40,120)); y0=int(rng.integers(40,200)); x1=x0+int(rng.integers(100,160)); y1=y0+int(rng.integers(-40,40))
    mid=((x0+x1)//2,(y0+y1)//2+int(rng.integers(-20,20)))
    pts=np.array([[x0,y0],mid,[x1,y1]],np.int32)
    dark=float(rng.uniform(0.25,0.5))
    ov=img.copy(); cv2.polylines(ov,[pts],False,(0,0,0),int(rng.integers(4,7)),cv2.LINE_AA)
    m=(ov.sum(2)<1)  # not used
    mask=np.zeros((H,W),np.uint8); cv2.polylines(mask,[pts],False,255,int(rng.integers(4,7)),cv2.LINE_AA)
    a=(mask/255.0)[...,None]
    return (img*(1-a)+img*dark*a).astype(np.float32)
def add_sticker(img,rng):
    img=img.copy(); s=int(rng.integers(22,40)); x=int(rng.integers(30,W-30-s)); y=int(rng.integers(30,H-30-s))
    lab=cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
    da=float(rng.choice([-1,1]))*float(rng.uniform(22,45)); db=float(rng.choice([-1,1]))*float(rng.uniform(22,45))
    lab[y:y+s,x:x+s,1]+=da; lab[y:y+s,x:x+s,2]+=db   # L untouched -> purely chromatic
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)
def add_stain(img,rng):
    img=img.copy(); r=int(rng.integers(12,22)); cx=int(rng.integers(40,W-40)); cy=int(rng.integers(40,H-40))
    yy,xx=np.mgrid[0:H,0:W]; d=np.sqrt((xx-cx)**2+(yy-cy)**2)
    a=np.clip(1-d/r,0,1)**0.6*float(rng.uniform(0.15,0.30))
    return (img*(1-a[...,None])).astype(np.float32)
def add_texture(img,rng):
    img=img.copy(); s=int(rng.integers(36,56)); x=int(rng.integers(30,W-30-s)); y=int(rng.integers(30,H-30-s))
    yy,xx=np.mgrid[0:s,0:s]; chk=((xx+yy)%2*2-1).astype(np.float32)*float(rng.uniform(22,35))   # zero-mean, 1-px checker
    img[y:y+s,x:x+s,:]+=chk[...,None]
    return np.clip(img,0,255).astype(np.float32)
def variant(ref,cur,thr=8,min_area=150,ab=True,l=True,ssim_mode=None,sthr=30):
    lr=cv2.GaussianBlur(cv2.cvtColor(ref,cv2.COLOR_BGR2LAB),(7,7),0); lc=cv2.GaussianBlur(cv2.cvtColor(cur,cv2.COLOR_BGR2LAB),(7,7),0)
    dab=cv2.absdiff(lr[...,1:].astype('float32'),lc[...,1:].astype('float32')); mag=np.sqrt(dab[...,0]**2+dab[...,1]**2)
    dl=cv2.absdiff(lr[...,0].astype('float32'),lc[...,0].astype('float32'))*1.5
    comb=np.zeros_like(mag)
    if ab: comb=np.maximum(comb,mag)
    if l: comb=np.maximum(comb,dl)
    _,t=cv2.threshold(np.clip(comb,0,255).astype('uint8'),thr,255,cv2.THRESH_BINARY)
    if ssim_mode:
        g0=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY); g1=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
        _,sd=ssim(g0,g1,full=True,data_range=255); sm=((1-sd)*127.5).clip(0,255).astype('uint8'); _,st=cv2.threshold(sm,sthr,255,cv2.THRESH_BINARY)
        t=cv2.bitwise_or(t,st) if ssim_mode=='or' else cv2.bitwise_and(t,st)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)); m=cv2.dilate(cv2.morphologyEx(t,cv2.MORPH_OPEN,k,iterations=2),k,iterations=2)
    cn,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c)>min_area for c in cn), m
VARS={
 'A-B only (colour)':dict(ab=True,l=False,ssim_mode=None),
 'L only':dict(ab=False,l=True,ssim_mode=None),
 'A-B + SSIM-AND (v1)':dict(ab=True,l=False,ssim_mode='and'),
 'A-B + L (max)':dict(ab=True,l=True,ssim_mode=None),
 'A-B + L + SSIM-OR (released)':dict(ab=True,l=True,ssim_mode='or'),
}
N=100
res={v:{'wire':0,'sticker':0,'stain':0,'texture':0,'clean':0,'stress':0} for v in VARS}
# sanity: compare released variant to the real class
cls_match=0;cls_n=0
for i in range(N):
    sc=base_scene(rng); ref=noisy(sc,rng)
    for name,fn in [('wire',add_wire),('sticker',add_sticker),('stain',add_stain),('texture',add_texture)]:
        cur=noisy(fn(sc,rng),rng)
        for v,kw in VARS.items():
            d,_=variant(ref,cur,**kw); res[v][name]+=int(d)
        if i<20:
            det=CustomDefectDetector(0,use_ssim=True); det._save_intermediates=False; det._roi_threshold=8; det._roi_min_area=150
            det.initialize(ref); r=det._detect(cur); d,_=variant(ref,cur,**VARS['A-B + L + SSIM-OR (released)'])
            cls_match+=int(bool(r.prediction)==d); cls_n+=1
    # clean frames: new noise, illumination drift +-1.5%, 50% with 1px jitter
    gain=float(rng.uniform(-0.015,0.015)); sh=(int(rng.integers(-1,2)),int(rng.integers(-1,2))) if i%2 else (0,0)
    cur=noisy(sc,rng,gain=gain,shift=sh)
    for v,kw in VARS.items():
        d,_=variant(ref,cur,**kw); res[v]['clean']+=int(d)
    # stress: +-5% gain, soft shadow gradient, 2px camera vibration
    g2=float(rng.uniform(-0.05,0.05)); sh2=(int(rng.integers(-2,3)),int(rng.integers(-2,3)))
    xx=np.linspace(0,1,W,dtype=np.float32)[None,:,None]; shadow=1-float(rng.uniform(0.02,0.06))*xx
    cur2=noisy(sc*shadow,rng,gain=g2,shift=sh2)
    for v,kw in VARS.items():
        d,_=variant(ref,cur2,**kw); res[v]['stress']+=int(d)
# placeholder
print('class-vs-reimpl agreement',cls_match,'/',cls_n)
for v in res: print(v,res[v])
json.dump(dict(N=N,res=res,agree=[cls_match,cls_n]),open('ablate.json','w'))
