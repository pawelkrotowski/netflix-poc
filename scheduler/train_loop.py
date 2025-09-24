# scheduler/train_loop.py
import os, time, traceback
from trainer import run_once

INTERVAL = int(os.getenv("TRAIN_INTERVAL_SEC", str(24*60*60)))  # default: 24h

if __name__ == "__main__":
    while True:
        try:
            out = run_once()
            print("Train OK:", out, flush=True)
        except Exception as e:
            print("Train ERROR:", e, flush=True)
            traceback.print_exc()
        time.sleep(INTERVAL)
