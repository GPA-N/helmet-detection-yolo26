import json, os, sys
sys.path.insert(0,'/content')
from ultralytics import YOLO
M=json.loads(os.environ['MODELS']); out={}
for nm,p in M.items():
    m=YOLO(p); m.model.names={0:'helmet',1:'no_helmet'}
    for sp,y in (('deb_test','deb_only'),('our_test','test_only'),('our_dev','dev_only')):
        r=m.val(data=f'/content/ds/{y}.yaml',conf=0.001,verbose=False,plots=False,workers=2)
        b=r.box; out[f'{nm}|{sp}']=dict(map50=round(float(b.map50),3),map=round(float(b.map),3),P=[round(float(x),3) for x in b.p],R=[round(float(x),3) for x in b.r],ap50=[round(float(x),3) for x in b.ap50])
        print('EV',nm,sp,json.dumps(out[f'{nm}|{sp}']),flush=True)
json.dump(out,open('/content/eval_stock.json','w')); print('EV_END',flush=True)
