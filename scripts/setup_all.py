import os, subprocess, sys, time, shutil, threading
os.makedirs('/root/.kaggle',exist_ok=True); shutil.copy('/content/kaggle.json','/root/.kaggle/kaggle.json'); os.chmod('/root/.kaggle/kaggle.json',0o600)
t0=time.time()
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True)
sh('pip -q install kaggle ultralytics transformers 2>&1 | tail -1')
print(f'[{time.time()-t0:4.0f}s] pip done',flush=True)
def kg(s):
    d='/content/data/'+s.split('/')[1]; os.makedirs(d,exist_ok=True); r=subprocess.run(['kaggle','datasets','download','-d',s,'-p',d,'--unzip','-q'],capture_output=True,text=True); print(f'[{time.time()-t0:4.0f}s] {s} rc={r.returncode} {r.stderr[-150:]}',flush=True)
def deb():
    d='/content/data/deboniz'; os.makedirs(d,exist_ok=True); sh(f'cd {d} && curl -sL -o d.zip https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset/resolve/main/helmet-detection-dataset.zip && unzip -q d.zip && rm d.zip'); print(f'[{time.time()-t0:4.0f}s] deboniz done',flush=True)
th=[threading.Thread(target=kg,args=(s,)) for s in ["aneesarom/rider-with-helmet-without-helmet-number-plate","aryanvaid13/indian-helmet-detection-dataset","devgurucodes/trafffic-violations-triple-riding-no-helmet-plate","pkdarabi/helmet"]]+[threading.Thread(target=deb)]
for t in th: t.start()
for t in th: t.join()
os.makedirs('/content/runs/r2_plumbing/weights',exist_ok=True); shutil.copy('/content/r2_last.pt','/content/runs/r2_plumbing/weights/last.pt')
r=sh('cd /content && python build_dataset.py 2>&1 | tail -25'); print(r.stdout[-1800:],flush=True)
print(f'[{time.time()-t0:4.0f}s] SETUP DONE',flush=True)
if os.path.isdir('/content/ds/train') and os.path.isdir('/content/ds/anchor'):
    subprocess.Popen("cd /content && LR=8 LA=8 EP=15 NHN=900 PTH=0.5 REPLAY=1 NAME=e4 nohup python -u lora_exp.py > exp_e4.log 2>&1 &",shell=True); print('E4 LAUNCHED',flush=True)
else: print('DATASET MISSING',flush=True)
