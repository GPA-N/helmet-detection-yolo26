import sys, os, glob, cv2, numpy as np
sys.path.insert(0,'/content')
from ultralytics import YOLO
def L(p): mm=YOLO(p); mm.model.names={0:'helmet',1:'no_helmet'}; return mm
M=[('Start model',L('/content/m/start.pt')),('Fine-tuned',L('/content/m/r2.pt')),('+ caps (LoRA)',L('/content/m_e4.pt'))]
test=sorted(glob.glob('/content/ds/test/images/*')); dev=sorted(glob.glob('/content/ds/dev/images/*'))
deb=sorted(glob.glob('/content/data/deboniz/**/test/images/*',recursive=True))
import json
sel=[('deboniz',deb[i]) for i in range(0,len(deb),50)][:13]
os.makedirs('/content/ex',exist_ok=True)
for k,(src,f) in enumerate(sel):
    panels=[]
    for nm,m in M:
        im=cv2.imread(f); h,w=im.shape[:2]; s=480/max(h,w); im=cv2.resize(im,(int(w*s),int(h*s)))
        r=m.predict(f,conf=0.25,verbose=False)[0]
        for b,c,cf in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2=[int(v*s) for v in b]; col=(0,170,0) if int(c)==0 else (0,0,230)
            cv2.rectangle(im,(x1,y1),(x2,y2),col,2); cv2.putText(im,f"{['helmet','no_helmet'][int(c)]} {cf:.2f}",(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
        cv2.rectangle(im,(0,0),(160,20),(0,0,0),-1); cv2.putText(im,nm,(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); panels.append(im)
    hh=max(p.shape[0] for p in panels); panels=[cv2.copyMakeBorder(p,0,hh-p.shape[0],0,6,cv2.BORDER_CONSTANT,value=(255,255,255)) for p in panels]
    cv2.imwrite(f'/content/ex/example_{k+1:02d}_{src}.jpg',np.concatenate(panels,1),[cv2.IMWRITE_JPEG_QUALITY,85])
os.system('cd /content/ex && zip -q -r ../ex.zip .'); print('RENDER_OK',len(sel),flush=True)
