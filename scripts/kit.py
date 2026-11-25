import os, glob, math, json
from PIL import Image
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (round((c-r)/d,3),round((c+r)/d,3))
def gt_boxes(lp,W,H):
    out=[]
    for l in open(lp).read().splitlines():
        t=l.split()
        if len(t)<5: continue
        c=int(t[0]); cx,cy,w,h=map(float,t[1:5]); out.append((c,(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H))
    return out
def predict(model,files,conf=0.25,imgsz=640):
    import torch, gc
    out={}
    for i in range(0,len(files),16):
        ch=files[i:i+16]
        with torch.no_grad(): res=model.predict(ch,conf=conf,imgsz=imgsz,verbose=False)
        for f,r in zip(ch,res): out[f]=[(int(b.cls),float(b.conf),*b.xyxy[0].tolist()) for b in r.boxes]
        del res
        if i%160==0: gc.collect(); torch.cuda.empty_cache()
    return out
def prf(model,split_dir,conf=0.25,imgsz=640):
    files=sorted(glob.glob(split_dir+'/images/*')); preds=predict(model,files,conf,imgsz); st={c:dict(tp=0,fp=0,fn=0) for c in (0,1)}
    for f in files:
        W,H=Image.open(f).size; g=gt_boxes(f.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt',W,H); used=set()
        for c,cf,*bb in sorted(preds[f],key=lambda x:-x[1]):
            best=-1;bi=0
            for k,(gc,*gb) in enumerate(g):
                if gc==c and k not in used:
                    v=iou(bb,gb)
                    if v>=0.5 and v>bi: bi=v;best=k
            if best>=0: used.add(best); st[c]['tp']+=1
            else: st[c]['fp']+=1
        for k,(gc,*_) in enumerate(g):
            if k not in used: st[gc]['fn']+=1
    out={}
    for c,n in ((0,'helmet'),(1,'no_helmet')):
        tp,fp,fn=st[c]['tp'],st[c]['fp'],st[c]['fn']
        out[n]=dict(n_gt=tp+fn,recall=round(tp/(tp+fn),3) if tp+fn else None,recall_ci=wilson(tp,tp+fn),precision=round(tp/(tp+fp),3) if tp+fp else None,tp=tp,fp=fp,fn=fn)
    return out
def ref_dets(model,files,conf=0.5):
    p=predict(model,files,conf); return {f:[d for d in v] for f,v in p.items()}
def retention(ref,model,conf=0.25):
    files=list(ref); cand=predict(model,files,conf); tot=0;hit=0;sub={'small<50px':[0,0],'helmet':[0,0],'no_helmet':[0,0]}
    for f in files:
        for c,cf,*bb in ref[f]:
            m=any(cc==c and iou(bb,cb)>=0.5 for cc,_,*cb in cand[f]); tot+=1; hit+=m
            keys=['helmet' if c==0 else 'no_helmet']+(['small<50px'] if (bb[2]-bb[0])<50 else [])
            for k in keys: sub[k][1]+=1; sub[k][0]+=m
    return dict(n_ref=tot,retained=hit,retention=round(hit/tot,3) if tot else None,ci=wilson(hit,tot),subsets={k:dict(hit=v[0],n=v[1],rate=round(v[0]/v[1],3) if v[1] else None) for k,v in sub.items()})
def full_report(model,start_model,ds='/content/ds',user='/content/user3',tag=''):
    rep={}
    for sp in ('dev','test'): rep[sp+'_prf@0.25']=prf(model,f'{ds}/{sp}')
    af=sorted(glob.glob(ds+'/anchor/images/*')); uf=sorted(glob.glob(user+'/*'))
    rep['anchor_retention']=retention(ref_dets(start_model,af),model)
    rep['user3_retention']=retention(ref_dets(start_model,uf,0.25),model)
    return rep

def leak_check(ds='/content/ds',thr=4):
    import numpy as np
    from PIL import Image as I
    RES=getattr(I,'Resampling',I).BILINEAR
    def dh(p):
        im=I.open(p); im.draft('L',(64,64)); a=np.asarray(im.convert('L').resize((9,8),RES),dtype=np.int16); v=0
        for b in (a[:,1:]>a[:,:-1]).flatten(): v=(v<<1)|int(b)
        return v
    H={sp:[dh(p) for p in sorted(glob.glob(f'{ds}/{sp}/images/*'))] for sp in ('train','dev','test')}
    out={}
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        out[a+'|'+b]=sum(1 for x in H[a] for y in H[b] if bin(x^y).count('1')<=thr)
    return out
def names_ok(model):
    return {int(k):v for k,v in model.names.items()}=={0:'helmet',1:'no_helmet'}
