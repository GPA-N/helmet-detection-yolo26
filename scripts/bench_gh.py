# Many-input WSP: DeBoniz CLIP-mined caps -> stability-selected cls-branch channels (depthwise kernels) -> dampen. Forward-only.
import sys, glob, os, json, random, cv2, numpy as np, torch
sys.path.insert(0,'/content')
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
import kit
dev='cuda'; BASE='/content/runs/r2_plumbing/weights/last.pt'; R='/content/data/deboniz'; random.seed(0); torch.manual_seed(0)
# ---- 1. mine caps with CLIP from class-1 boxes ----
clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval(); proc=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
CATS=[['a photo of a person wearing a baseball cap','a photo of a person wearing a hat','a photo of a person wearing a cap on their head'],['a photo of a woman wearing a hijab','a photo of a person wearing a headscarf']]
T_=[]
with torch.no_grad():
    for c in CATS:
        e=clip.text_projection(clip.text_model(**proc(text=c,return_tensors='pt',padding=True).to(dev)).pooler_output); e=e/e.norm(dim=-1,keepdim=True); T_.append(e.mean(0))
    e=clip.text_projection(clip.text_model(**proc(text=['a photo of a person with a bare head and visible hair','a photo of a person with no hat or helmet','a photo of a motorcycle helmet','a photo of a turban'],return_tensors='pt',padding=True).to(dev)).pooler_output); e=e/e.norm(dim=-1,keepdim=True); T_.append(e.mean(0))
T_=torch.stack(T_); T_=T_/T_.norm(dim=-1,keepdim=True)
items=[]
for sp in ('train','val','test'):
    for lp in glob.glob(f'{R}/**/{sp}/labels/*.txt',recursive=True):
        for ln in open(lp):
            a=ln.split()
            if len(a)>=5 and int(a[0])==1: items.append((sp,lp,[float(v) for v in a[1:5]]))
crops=[];keep=[];P=[]
def fl():
    global crops
    if crops:
        with torch.no_grad():
            e=clip.visual_projection(clip.vision_model(**proc(images=crops,return_tensors='pt').to(dev)).pooler_output); e=e/e.norm(dim=-1,keepdim=True); P.append(torch.softmax(100*e@T_.T,dim=-1).cpu())
        crops=[]
for sp,lp,(cx,cy,bw,bh) in items:
    ip=None
    for ext in ('.jpg','.jpeg','.png'):
        c_=lp.replace('/labels/','/images/').rsplit('.',1)[0]+ext
        if os.path.exists(c_): ip=c_; break
    if not ip: continue
    im=cv2.imread(ip); h,w=im.shape[:2]; x1,y1,x2,y2=(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h
    if (x2-x1)<32 or (y2-y1)<32: continue
    pad=0.15*max(x2-x1,y2-y1); cr=im[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    if cr.size==0: continue
    crops.append(cv2.cvtColor(cr,cv2.COLOR_BGR2RGB)); keep.append((sp,ip,[x1,y1,x2,y2]))
    if len(crops)==128: fl()
fl(); P=torch.cat(P); pc=(P[:,0]+P[:,1]).numpy(); del clip; torch.cuda.empty_cache()
sel={sp:[i for i in range(len(keep)) if keep[i][0]==sp and pc[i]>=0.7] for sp in ('train','val','test')}
print('DEB cap picks',{k:len(v) for k,v in sel.items()},flush=True)
# one box per image for train positives (avoid double counting), cap 500
seen=set(); tr=[]
for i in sorted(sel['train'],key=lambda i:-pc[i]):
    if keep[i][1] not in seen: seen.add(keep[i][1]); tr.append(i)
tr=tr[:500]; ho=[]; seen=set()
for i in sel['val']+sel['test']:
    if keep[i][1] not in seen: seen.add(keep[i][1]); ho.append(i)
print('DEB train positives',len(tr),'held-out cap images',len(ho),flush=True)

import shutil, yaml
for nm,sp in (('train_only','train'),('dev_only','dev'),('test_only','test')): open(f'/content/ds/{nm}.yaml','w').write(f"path: /content/ds\ntrain: train/images\nval: {sp}/images\nnames:\n  0: helmet\n  1: no_helmet\n")
MODELS={'start':'/content/m/start.pt','r2_finetune':'/content/m/r2.pt','r3_full_caps':'/content/m_r3.pt','e4_lora_caps':'/content/m_e4.pt'}
def L(p): mm=YOLO(p); mm.model.names={0:'helmet',1:'no_helmet'}; return mm
hs=sorted({keep[i][1] for i in ho}); gt={}
for i in ho: gt.setdefault(keep[i][1],[]).append(keep[i][2])
def cap_rates(mm):
    pr=kit.predict(mm,hs,0.25); n=nn_=nh=0
    for f in hs:
        for b in gt[f]:
            n+=1; best=None
            for dd_ in pr[f]:
                io=kit.iou(b,dd_[2:])
                if io>0.3 and (best is None or dd_[1]>best[1]): best=dd_
            if best is not None: nn_+=best[0]==1; nh+=best[0]==0
    return dict(n=n,called_no_helmet=round(nn_/n,3),called_helmet=round(nh/n,3))
anc=sorted(glob.glob('/content/ds/anchor/images/*'))[1::2]; r2m=L(MODELS['r2_finetune']); ref=kit.predict(r2m,anc,0.25)
def retention(mm):
    new=kit.predict(mm,anc,0.25); n=k=0
    for f in anc:
        for r in ref[f]:
            if r[1]<0.5: continue
            n+=1
            if any(d[0]==r[0] and kit.iou(r[2:],d[2:])>=0.5 for d in new[f]): k+=1
    return round(k/max(n,1),3)
OUT={}
for nm,p in MODELS.items():
    mm=L(p); OUT[nm]={}
    for sp in ('train','dev','test'):
        r=mm.val(data=f'/content/ds/{sp}_only.yaml',split='val',imgsz=640,batch=16,conf=0.001,verbose=False,plots=False)
        OUT[nm][sp]=dict(mAP50=round(float(r.box.map50),3),mAP50_95=round(float(r.box.map),3),per_class={c:dict(P=round(float(r.box.p[i]),3),R=round(float(r.box.r[i]),3),AP50=round(float(r.box.ap50[i]),3),AP50_95=round(float(r.box.ap[i]),3)) for i,c in enumerate(('helmet','no_helmet'))})
    OUT[nm]['heldout_caps']=cap_rates(mm); OUT[nm]['retention_vs_r2']=retention(mm); print('BENCH',nm,json.dumps(OUT[nm]),flush=True)
json.dump(OUT,open('/content/bench.json','w')); print('BENCH_END',flush=True)
