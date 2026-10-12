import sys, json, numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('ablate.py').read().split("N=100")[0])
rng=np.random.default_rng(21)
K=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
def decide(m,min_area=150):
    t=cv2.dilate(cv2.morphologyEx(m,cv2.MORPH_OPEN,K,iterations=2),K,iterations=2)
    cn,_=cv2.findContours(t,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); return any(cv2.contourArea(c)>min_area for c in cn)
def maps(ref,cur):
    lr=cv2.GaussianBlur(cv2.cvtColor(ref,cv2.COLOR_BGR2LAB),(7,7),0); lc=cv2.GaussianBlur(cv2.cvtColor(cur,cv2.COLOR_BGR2LAB),(7,7),0)
    dab=cv2.absdiff(lr[...,1:].astype('float32'),lc[...,1:].astype('float32')); mag=np.sqrt(dab[...,0]**2+dab[...,1]**2)
    dl=cv2.absdiff(lr[...,0].astype('float32'),lc[...,0].astype('float32'))*1.5
    g0=cv2.GaussianBlur(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY),(7,7),0); g1=cv2.GaussianBlur(cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY),(7,7),0)
    fd=cv2.absdiff(g0,g1).astype('float32')
    g0=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY); g1=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    _,sd=ssim(g0,g1,full=True,data_range=255); sm=((1-sd)*127.5).clip(0,255).astype('uint8'); _,st=cv2.threshold(sm,30,255,cv2.THRESH_BINARY)
    return dict(fd=fd,ab=mag,abl=np.maximum(mag,dl),st=st)
TAUS=list(range(2,41,2)); NT=100
pos=[];neg=[]
for i in range(NT):
    sc=base_scene(rng); ref=noisy(sc,rng)
    for fn in (add_wire,add_sticker,add_stain,add_texture): pos.append(maps(ref,noisy(fn(sc,rng),rng)))
    gain=float(rng.uniform(-0.015,0.015)); sh=(int(rng.integers(-1,2)),int(rng.integers(-1,2))) if i%2 else (0,0)
    neg.append(maps(ref,noisy(sc,rng,gain=gain,shift=sh)))
    g2=float(rng.uniform(-0.05,0.05)); sh2=(int(rng.integers(-2,3)),int(rng.integers(-2,3)))
    xx=np.linspace(0,1,W,dtype=np.float32)[None,:,None]; shadow=1-float(rng.uniform(0.02,0.06))*xx
    neg.append(maps(ref,noisy(sc*shadow,rng,gain=g2,shift=sh2)))
def rate(ms,var,tau):
    c=0
    for m in ms:
        if var=='fd': b=(m['fd']>tau)
        elif var=='ab': b=(m['ab']>tau)
        elif var=='abl': b=(m['abl']>tau)
        else: b=(m['abl']>tau)
        b=(b*255).astype('uint8')
        if var=='full': b=cv2.bitwise_or(b,m['st'])
        c+=decide(b)
    return c/len(ms)
curves={}
for var in ['fd','ab','abl','full']:
    curves[var]=[(rate(neg,var,t),rate(pos,var,t)) for t in TAUS]
json.dump({k:v for k,v in curves.items()},open('sweep.json','w'))
def auc_part(c,fmax=0.5):
    pts=sorted(set([(0.0,0.0)]+c+[(1.0,1.0)])); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return np.trapezoid(ys,xs)
for k,c in curves.items():
    print(k,'AUC(sweep)=%.3f'%auc_part(c),'TPR@FPR<=0.05=%.2f'%max([y for x,y in c if x<=0.05]+[0]),'TPR@FPR<=0.30=%.2f'%max([y for x,y in c if x<=0.30]+[0]), 'tau=8:',curves[k][3],'tau=24:',curves[k][11])
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':7})
fig,ax=plt.subplots(figsize=(3.4,2.05))
sty={'fd':('#4C78A8','Gray FD','s'),'ab':('#59A14F','A–B only','^'),'abl':('#F28E2B','A–B + L','o'),'full':('#D62728','A–B + L + SSIM (OR)','D')}
for k,(c,l,mk) in sty.items():
    cc=sorted(curves[k]); ax.plot([0]+[p[0] for p in cc]+[1],[0]+[p[1] for p in cc]+[1],'-'+mk,color=c,ms=2.6,lw=1.0,label=l)
ax.plot([0,1],[0,1],'--',color='gray',lw=.5); ax.set_xlabel('False-alarm rate (clean + stress frames)'); ax.set_ylabel('Detection rate (4 defect types)')
ax.legend(fontsize=5.8,frameon=False,loc='lower right'); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.3,lw=.4); ax.set_xlim(-0.01,1.01); ax.set_ylim(0,1.03)
plt.tight_layout(pad=0.3); plt.savefig('fig/f6_sweep.png',dpi=300); plt.close()
