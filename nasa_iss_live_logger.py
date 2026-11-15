"""
Log LIVE ISS cabin telemetry from NASA's public ISS Live (Lightstreamer) feed to a CSV.

The feed only pushes a value when it CHANGES, so slow signals (like cabin pressure) update rarely.
This logger therefore keeps the latest value of each item and writes one row per item every --every seconds,
giving a regular time series (rows repeat the last known value until NASA sends a new one).

Setup:  pip install lightstreamer-client-lib
Usage:  python nasa_iss_live_logger.py --minutes 30 --every 10 --out data/nasa/iss_cabin.csv
        python nasa_iss_live_logger.py --summary data/nasa/iss_cabin.csv

ITEMS: name -> ISS Live telemetry item ID. USLAB000058 read ~757.7 on 2026-11-15, which matches cabin
pressure in mmHg. Look up more IDs (cabin temperature, etc.) in ISS_Public_Telemetry.xlsx in the
ISS-Mimic/Mimic GitHub repo (Telemetry folder), check the units there, and add them below.
"""
import argparse, csv, os, time, threading

SERVER = "https://push.lightstreamer.com"
ADAPTER = "ISSLIVE"
ITEMS = {"cabin_pressure_mmhg": "USLAB000058"}      # add e.g. "cabin_temp_c": "<ID from the Mimic sheet>"

def summary(path):
    import pandas as pd
    d = pd.read_csv(path)
    print(d.groupby("name").value.describe().round(2).to_string())

def log(minutes, every, out):
    from lightstreamer.client import LightstreamerClient, Subscription
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ids = {v: k for k, v in ITEMS.items()}
    latest, lock = {}, threading.Lock()

    class Listener:
        def onItemUpdate(self, update):
            v = update.getValue("Value")
            if v is None: return
            with lock: latest[ids.get(update.getItemName(), update.getItemName())] = v
        def onSubscription(self): print("subscribed")
        def onSubscriptionError(self, code, message): print("subscription error", code, message)

    client = LightstreamerClient(SERVER, ADAPTER)
    client.connect()
    sub = Subscription("MERGE", list(ITEMS.values()), ["Value"])
    sub.addListener(Listener())
    client.subscribe(sub)
    print(f"logging {list(ITEMS)} every {every}s for {minutes} min -> {out}")
    new = not os.path.exists(out)
    end = time.time() + minutes * 60
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["utc_time", "name", "item", "value"])
        try:
            while time.time() < end:
                time.sleep(every)
                with lock: snap = dict(latest)
                t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                for name, val in snap.items():
                    w.writerow([t, name, ITEMS.get(name, ""), val])
                f.flush()
        finally:
            client.disconnect()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--out", default="data/nasa/iss_cabin.csv")
    ap.add_argument("--summary")
    a = ap.parse_args()
    summary(a.summary) if a.summary else log(a.minutes, a.every, a.out)
