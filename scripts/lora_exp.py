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
sel['train']=[i for i in range(len(keep)) if keep[i][0]=='train' and pc[i]>=float(os.environ.get('PTH','0.7'))]
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
from ultralytics.cfg import DEFAULT_CFG_DICT
def dh(im): g=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(9,8)); return (g[:,1:]>g[:,:-1]).flatten()
hoh=np.stack([dh(cv2.imread(keep[i][1])) for i in ho])
ours=np.stack([dh(cv2.imread(p)) for p in glob.glob('/content/ds/dev/images/*')+glob.glob('/content/ds/test/images/*')])
bad=np.concatenate([hoh,ours])
# hard-negative images: CLIP cap picks in DeBoniz TRAIN whose label file has ONLY class-1 boxes, not near-dup of held-out/dev/test
HN='/content/ds/hardneg'; shutil.rmtree(HN,ignore_errors=True); os.makedirs(HN+'/images'); os.makedirs(HN+'/labels'); cand=[]; seen=set()
for i in sorted(sel['train'],key=lambda i:-pc[i]):
    ip=keep[i][1]
    if ip in seen: continue
    seen.add(ip); L=[l.split() for l in open(ip.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt') if len(l.split())>=5]
    if not L or any(int(a[0])!=1 for a in L): continue
    if (bad!=dh(cv2.imread(ip))).sum(1).min()<=6: continue
    cand.append(ip)
    if len(cand)>=int(os.environ.get('NHN','400')): break
for k,ip in enumerate(cand):
    shutil.copy(ip,f'{HN}/images/deb{k}.jpg'); shutil.copy(ip.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt',f'{HN}/labels/deb{k}.txt')
print('FT hard-negative images',len(cand),'our train imgs',len(glob.glob('/content/ds/train/images/*')),flush=True)

print('EXP hardneg',len(cand),flush=True)
import math, yaml, torch.nn as nn
from ultralytics.cfg import DEFAULT_CFG_DICT
class LoRAConv2d(nn.Module):
    def __init__(s, base, r=int(os.environ.get('LR','8')), alpha=int(os.environ.get('LA','8'))):
        super().__init__(); s.base=base; s.scale=alpha/r
        for a in ('in_channels','out_channels','kernel_size','stride','padding','dilation','groups'): setattr(s,a,getattr(base,a))
        s.lora_A=nn.Conv2d(base.in_channels,r,base.kernel_size,base.stride,base.padding,base.dilation,bias=False)
        s.lora_B=nn.Conv2d(r,base.out_channels,1,bias=False)
        nn.init.kaiming_uniform_(s.lora_A.weight,a=math.sqrt(5)); nn.init.zeros_(s.lora_B.weight)
    def forward(s,x): return s.base(x)+s.lora_B(s.lora_A(x))*s.scale
    @property
    def weight(s): return s.merged()
    @property
    def bias(s): return s.base.bias
    @torch.no_grad()
    def merged(s): return s.base.weight+torch.einsum('or,rikl->oikl',s.lora_B.weight[:,:,0,0],s.lora_A.weight)*s.scale
class FrozenBN(nn.BatchNorm2d):
    def train(s,mode=True): return super().train(False)
def wrap(model):
    n=0
    for _,mod in list(model.named_modules()):
        for cn,ch in list(mod.named_children()):
            if type(ch) is nn.Conv2d and ch.groups==1: setattr(mod,cn,LoRAConv2d(ch)); n+=1
    for mod in model.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.__class__=FrozenBN
    return n
from ultralytics.models.yolo.detect import DetectionTrainer
class LoRATrainer(DetectionTrainer):
  def setup_model(self):
    ck=super().setup_model(); cb(self); return ck
def cb(trainer):
    n=wrap(trainer.model); fr=[]
    for k,v in trainer.model.named_parameters():
        if 'lora_' not in k: v.requires_grad=False; fr.append(k.rsplit('.',1)[0][len('model.'):] if k.startswith('model.') else None)
    trainer.args.freeze=[x for x in fr if x]
    tp=sum(p.numel() for k,p in trainer.model.named_parameters() if 'lora_' in k); tot=sum(p.numel() for k,p in trainer.model.named_parameters())
    print('LORA wrapped convs',n,'trainable adapter params',tp,'of',tot,'=',round(100*tp/tot,2),'%',flush=True)

NAME=os.environ.get('NAME','r5'); EP=int(os.environ.get('EP','15'))

REPLAY=os.environ.get('REPLAY')=='1'; anc_all=sorted(glob.glob('/content/ds/anchor/images/*'))
if REPLAY:
    _m0=YOLO(BASE); _m0.model.names={0:'helmet',1:'no_helmet'}; rp=anc_all[::2]; shutil.rmtree('/content/ds/replay',ignore_errors=True); os.makedirs('/content/ds/replay/images'); os.makedirs('/content/ds/replay/labels'); _pr=kit.predict(_m0,rp,0.25); nb=0
    for k,f in enumerate(rp):
        im_=cv2.imread(f); h_,w_=im_.shape[:2]; shutil.copy(f,f'/content/ds/replay/images/rp{k}.jpg'); L=[]
        for c_,cf_,x1,y1,x2,y2 in _pr[f]:
            if cf_>=0.5: L.append(f"{int(c_)} {(x1+x2)/2/w_:.6f} {(y1+y2)/2/h_:.6f} {(x2-x1)/w_:.6f} {(y2-y1)/h_:.6f}"); nb+=1
        open(f'/content/ds/replay/labels/rp{k}.txt','w').write('\n'.join(L))
    print('EXP replay images',len(rp),'pseudo boxes',nb,flush=True); del _m0
open('/content/ds/ft.yaml','w').write("path: /content/ds\ntrain:\n  - train/images\n  - hardneg/images\n"+("  - replay/images\n" if REPLAY else "")+"val: dev/images\nnames:\n  0: helmet\n  1: no_helmet\n")
a=yaml.safe_load(open('/content/runs/r2_plumbing/args.yaml')) if os.path.exists('/content/runs/r2_plumbing/args.yaml') else {}
deny={'model','data','epochs','time','name','project','save_dir','resume','pretrained','device','exist_ok','val','plots','save','split','task','mode','source','patience','freeze'}
kw={k:v for k,v in a.items() if k in DEFAULT_CFG_DICT and k not in deny}
m=YOLO(BASE); m.model.names={0:'helmet',1:'no_helmet'}
LP=f'/content/runs/{NAME}/weights/last.pt'
if not os.path.exists(LP):
    try: m.train(trainer=LoRATrainer,data='/content/ds/ft.yaml',epochs=EP,project='/content/runs',name=NAME,exist_ok=True,plots=False,**kw)
    except Exception as e: print('EXP train ended with',type(e).__name__,str(e)[:100],flush=True)
tr=YOLO(LP); plain=YOLO(BASE); plain.model.names={0:'helmet',1:'no_helmet'}; nch=0; tot=0; dmax=0.0
for name,mod in tr.model.named_modules():
    if isinstance(mod,LoRAConv2d):
        pc_=plain.model.get_submodule(name); w=mod.merged().float().cpu(); d=(w-pc_.weight.data.cpu()).abs(); dmax=max(dmax,float(d.max())); nch+=int((d>1e-6).sum()); tot+=w.numel(); pc_.weight.data.copy_(w)
plain.save(f'/content/runs/{NAME}/merged.pt'); print('EXP merged: changed',nch,'of',tot,'max delta',round(dmax,4),flush=True)
hs=sorted({keep[i][1] for i in ho}); gt={}
for i in ho: gt.setdefault(keep[i][1],[]).append(keep[i][2])
def cap_rates(mm,fl):
    pr=kit.predict(mm,fl,0.25); n=nn_=nh=0
    for f in fl:
        for b in gt[f]:
            n+=1; best=None
            for dd_ in pr[f]:
                io=kit.iou(b,dd_[2:])
                if io>0.3 and (best is None or dd_[1]>best[1]): best=dd_
            if best is not None: nn_+=best[0]==1; nh+=best[0]==0
    return dict(n=n,N=round(nn_/n,3),H=round(nh/n,3))
def devs(mm):
    dv=kit.prf(mm,'/content/ds/dev'); return dict(Hr=round(dv['helmet']['recall'],3),Hp=round(dv['helmet']['precision'],3),Nr=round(dv['no_helmet']['recall'],3),Np=round(dv['no_helmet']['precision'],3))
anc=anc_all[1::2] if REPLAY else anc_all; mb=YOLO(BASE); mb.model.names={0:'helmet',1:'no_helmet'}; ref=kit.predict(mb,anc,0.25)
def retention(mm):
    new=kit.predict(mm,anc,0.25); n=k=0
    for f in anc:
        for r in ref[f]:
            if r[1]<0.5: continue
            n+=1
            if any(d[0]==r[0] and kit.iou(r[2:],d[2:])>=0.5 for d in new[f]): k+=1
    return round(k/max(n,1),3)
ml=YOLO(f'/content/runs/{NAME}/merged.pt'); ml.model.names={0:'helmet',1:'no_helmet'}
R={}
for nm,mo in (('base',mb),('new',ml)):
    R[nm]=dict(cap_sel=cap_rates(mo,hs[::2]),cap_fin=cap_rates(mo,hs[1::2]),dev=devs(mo),retention=retention(mo)); print('EXP_EVAL',NAME,nm,json.dumps(R[nm]),flush=True)
for _n in ('r4_lora','e1','e3'):
    _p=f'/content/runs/{_n}/merged.pt'
    if os.path.exists(_p) and _n!=NAME:
        _m=YOLO(_p); _m.model.names={0:'helmet',1:'no_helmet'}; print('EXP_RETENTION_EVALHALF',_n,retention(_m),flush=True)
N=R['new'];B=R['base']
g=dict(cap_sel=N['cap_sel']['N']>=0.95,dev=all(N['dev'][k]>=B['dev'][k]-0.03 for k in N['dev']),retention=N['retention']>=0.98)
print('EXP_GATE',NAME,json.dumps(dict(g,all=all(g.values()))),flush=True)
if os.environ.get('TEST')=='1':
    tm={}
    for nm,mo in (('base',mb),('new',ml)):
        r=mo.val(data='/content/ds/test_only.yaml',split='val',imgsz=640,batch=16,conf=0.001,verbose=False,plots=False); tm[nm]=dict(mAP50=round(float(r.box.map50),3),mAP50_95=round(float(r.box.map),3))
    print('EXP_TEST',NAME,json.dumps(tm),flush=True)
for f in [p for p in ('/content/user3/img3.png','/content/user4.png','/content/user5.png','/content/user6.png') if os.path.exists(p)]:
    print('EXP_USER',NAME,os.path.basename(f),[f"{['H','N'][c]}{cf:.2f}" for c,cf,*b in kit.predict(ml,[f],0.25)[f]],flush=True)
print('EXP_END',NAME,flush=True)
