# Build cleaned head-level helmet dataset + anchor set. Logs every drop; never edits sources.
import os, hashlib, random, shutil, json, collections
import numpy as np
from PIL import Image
D='/content/data'; OUT='/content/ds'
SRC={'rider-with-helmet-without-helmet-number-plate':{0:0,1:1},   # 0 helmet, 1 no_helmet; drop rider, number plate
     'indian-helmet-detection-dataset':{1:1,2:0,3:0}}             # decoded from class montages; drop 0 plate, 4 rider
MIN_PX=6; IMG=('.jpg','.jpeg','.png')
RES=getattr(Image,'Resampling',Image).BILINEAR
def dhash(p):
    im=Image.open(p); im.draft('L',(64,64)); a=np.asarray(im.convert('L').resize((9,8),RES),dtype=np.int16)
    v=0
    for b in (a[:,1:]>a[:,:-1]).flatten(): v=(v<<1)|int(b)
    return v
def lab(img):
    for a in ('/images/','\\images\\'):
        if a in img:
            p=os.path.splitext(img.replace(a,a.replace('images','labels')))[0]+'.txt'
            if os.path.exists(p): return p
    return None
drops=collections.Counter(); items=[]
for s,cmap in SRC.items():
    for dp,dn,fn in os.walk(f'{D}/{s}'):
        dn.sort()
        for f in sorted(fn):
            if not f.lower().endswith(IMG): continue
            p=os.path.join(dp,f); lp=lab(p)
            if lp is None: drops[s+':no_label']+=1; continue
            try:
                with Image.open(p) as im: W,H=im.size
            except Exception: drops[s+':unreadable']+=1; continue
            boxes=[]
            for l in open(lp,errors='ignore').read().splitlines():
                t=l.split()
                if len(t)<5: continue
                try: c=int(float(t[0])); cx,cy,w,h=map(float,t[1:5])
                except Exception: drops[s+':bad_line']+=1; continue
                if c not in cmap: drops[s+f':ignored_class_{c}']+=1; continue
                if not(0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1): drops[s+':bad_box']+=1; continue
                if min(w*W,h*H)<MIN_PX: drops[s+':tiny_box']+=1; continue
                boxes.append((cmap[c],cx,cy,w,h))
            if not boxes: drops[s+':no_kept_boxes']+=1; continue
            items.append(dict(src=s,path=p,boxes=boxes,md5=hashlib.md5(open(p,'rb').read()).hexdigest(),dh=dhash(p)))
print('kept images',len(items),dict(collections.Counter(i['src'] for i in items)))
# clusters: exact md5 or dHash hamming<=4, across sources (union-find)
N=len(items); par=list(range(N))
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
H=np.array([i['dh'] for i in items],dtype=np.uint64)
pc=np.array([bin(i).count('1') for i in range(65536)],dtype=np.uint8)
def pop(x): return sum(pc[((x>>np.uint64(16*k))&np.uint64(0xFFFF)).astype(np.intp)] for k in range(4))
for i in range(0,N,256):
    d=pop(H[i:i+256,None]^H[None,:]); ii,jj=np.nonzero(d<=4)
    for a,b in zip(ii+i,jj):
        if a<b: par[find(a)]=find(b)
md=collections.defaultdict(list)
for k,it in enumerate(items): md[it['md5']].append(k)
for L in md.values():
    for k in L[1:]: par[find(k)]=find(L[0])
cl=collections.defaultdict(list)
for k in range(N): cl[find(k)].append(k)
print('clusters',len(cl),'largest',max(len(v) for v in cl.values()))
# split at cluster level (seeded). test = half of the aneesarom clusters (held-out source portion); dev = 15% of other clusters
rng=random.Random(0); split={}
for cid in sorted(cl): 
    srcs={items[k]['src'] for k in cl[cid]}
    if 'rider-with-helmet-without-helmet-number-plate' in srcs: sp='test' if rng.random()<0.5 else 'train'
    else: sp='dev' if rng.random()<0.15 else 'train'
    for k in cl[cid]: split[k]=sp
for sp in ('train','dev','test'):
    os.makedirs(f'{OUT}/{sp}/images',exist_ok=True); os.makedirs(f'{OUT}/{sp}/labels',exist_ok=True)
stats=collections.defaultdict(lambda: collections.Counter())
for k,it in enumerate(items):
    sp=split[k]; name=f"{it['src'][:6]}_{k:05d}"
    shutil.copy(it['path'],f"{OUT}/{sp}/images/{name}.jpg" if it['path'].lower().endswith('.jpg') else f"{OUT}/{sp}/images/{name}"+os.path.splitext(it['path'])[1])
    open(f'{OUT}/{sp}/labels/{name}.txt','w').write(''.join(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n' for c,cx,cy,w,h in it['boxes']))
    stats[sp]['images']+=1
    for c,*_ in it['boxes']: stats[sp][['helmet','no_helmet'][c]]+=1
    stats[sp]['src:'+it['src'][:10]]+=1
json.dump({f"{items[k]['src']}/{os.path.relpath(items[k]['path'],D+'/'+items[k]['src'])}":split[k] for k in range(N)},open(f'{OUT}/split_manifest.json','w'),indent=0)
open(f'{OUT}/data.yaml','w').write(f"path: {OUT}\ntrain: train/images\nval: dev/images\ntest: test/images\nnames:\n  0: helmet\n  1: no_helmet\n")
# anchor set: unlabeled frames from the person-level sources, not near-dup of any train/dev/test image
random.seed(0); cand=[]
for s in ('trafffic-violations-triple-riding-no-helmet-plate','helmet'):
    ps=[os.path.join(dp,f) for dp,_,fn in os.walk(f'{D}/{s}') for f in fn if f.lower().endswith(IMG)]
    random.shuffle(ps); cand+=ps[:900]
os.makedirs(f'{OUT}/anchor/images',exist_ok=True); kept=0; seen=[]; ref=H
for p in cand:
    if kept>=400: break
    try: h=np.uint64(dhash(p))
    except Exception: continue
    if (pop(ref^h)<=4).any() or any(bin(int(h)^int(x)).count('1')<=4 for x in seen): continue
    seen.append(h); shutil.copy(p,f'{OUT}/anchor/images/a_{kept:04d}'+os.path.splitext(p)[1]); kept+=1
stats['anchor']['images']=kept
res={'drops':dict(drops),'splits':{k:dict(v) for k,v in stats.items()},'clusters':len(cl)}
json.dump(res,open(f'{OUT}/build_report.json','w'),indent=1); print(json.dumps(res,indent=1))
