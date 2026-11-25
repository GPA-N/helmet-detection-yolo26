# Stock Ultralytics fine-tune of the start model. Library defaults; only data, model, time budget set.
import glob, os, sys
sys.path.insert(0,'/content')
R='/content/data/deboniz'
tr=[os.path.dirname(p) for p in glob.glob(f'{R}/**/train/images',recursive=True)]
va=[os.path.dirname(p) for p in glob.glob(f'{R}/**/val/images',recursive=True)]
print('STOCK dirs',tr,va,flush=True)
open('/content/stock.yaml','w').write("path: /\ntrain:\n  - "+tr[0].lstrip('/')+"/../train/images\n  - content/ds/train/images\nval: "+va[0].lstrip('/')+"/../val/images\nnames:\n  0: helmet\n  1: no_helmet\n")
from ultralytics import YOLO
m=YOLO('/content/m/start.pt')
m.train(data='/content/stock.yaml', time=float(os.environ.get('HOURS','0.5')), project='/content/runs', name='stock', exist_ok=True)
print('STOCK_END',flush=True)
