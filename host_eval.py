"""
host_eval.py -- stream test segments to the board, collect its predictions and timings, and compare
them with what Python predicts for the same exported model (device_reference.npz).

Needs:  pip install pyserial
Usage (next to train_hdc.py, board flashed and connected):
  python host_eval.py --port COM5                 # all 5,625 test segments (Windows port name)
  python host_eval.py --port COM5 --n 200         # a quick check first
  python host_eval.py --port /dev/ttyACM0         # Linux / macOS
"""
import argparse
import json
import os
import struct
import time

import numpy as np

import data as D
import metrics as M

REPLY = struct.Struct("<2sBhhII15B")          # 30 bytes
REPLY_PROFILE = struct.Struct("<2sBhhII15B3I")  # 42 bytes, firmware built with HDC_PROFILE = 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--ref", default="./device_export/device_reference.npz")
    p.add_argument("--n", type=int, default=None, help="number of test segments (default: all)")
    p.add_argument("--out", default="./device_export/device_eval")
    p.add_argument("--profile", action="store_true",
                   help="firmware was built with HDC_PROFILE = 1: also read the per-stage cycle counts")
    a = p.parse_args()
    import serial

    splits = D.load_splits(a.cache)
    x = np.asarray(splits["test"]["x"][..., 0], np.float32)
    ref = np.load(a.ref, allow_pickle=True)
    n = len(x) if a.n is None else min(a.n, len(x))

    pred, h0, h1, cyc = (np.zeros(n, np.int64) for _ in range(4))
    lev = np.zeros((n, 15), np.int64)
    stage = np.zeros((n, 3), np.int64)             # cycles: features, encoding, classification
    reply = REPLY_PROFILE if a.profile else REPLY
    clock = None
    with serial.Serial(a.port, a.baud, timeout=10) as ser:
        time.sleep(0.5)
        ser.reset_input_buffer()
        t_start = time.time()
        for i in range(n):
            ser.write(b"\xA5\x5A" + x[i].astype("<f4").tobytes())
            raw = ser.read(reply.size)
            if len(raw) != reply.size:
                raise RuntimeError(f"segment {i}: timeout, got {len(raw)} of {reply.size} bytes "
                                   f"(check that --profile matches HDC_PROFILE in the firmware)")
            fields = reply.unpack(raw)
            if fields[0] != b"\x5A\xA5":
                raise RuntimeError(f"segment {i}: bad sync {fields[0]!r}")
            pred[i], h0[i], h1[i], cyc[i], clock = fields[1], fields[2], fields[3], fields[4], fields[5]
            lev[i] = fields[6:21]
            if a.profile:
                stage[i] = fields[21:24]
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{n} segments, {time.time() - t_start:.0f}s")

    ms = cyc / clock * 1000.0
    agree_pred = float(np.mean(pred == ref["pred"][:n]))
    agree_lev = float(np.mean(np.all(lev == ref["levels"][:n], axis=1)))
    agree_ham = float(np.mean((h0 == ref["h0"][:n]) & (h1 == ref["h1"][:n])))
    res = dict(n=n, core_clock_hz=int(clock),
               latency_ms=dict(mean=float(ms.mean()), median=float(np.median(ms)),
                               p99=float(np.percentile(ms, 99)), max=float(ms.max())),
               agreement=dict(prediction=agree_pred, all_15_levels=agree_lev, hamming=agree_ham))
    if a.profile:
        sm = stage.mean(0) / clock * 1000.0
        res["latency_breakdown_ms"] = dict(features=float(sm[0]), encoding=float(sm[1]), classification=float(sm[2]))
    if n == len(x):
        res["device_test"] = M.primary_report(ref["y"], pred, ref["subject"], beta=2.0)
        res["python_test"] = M.primary_report(ref["y"], ref["pred"], ref["subject"], beta=2.0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez(a.out + ".npz", pred=pred, h0=h0, h1=h1, cycles=cyc, levels=lev)

    print(f"\nsegments {n}, core clock {clock / 1e6:.0f} MHz")
    print(f"latency per segment: mean {ms.mean():.2f} ms, median {np.median(ms):.2f}, "
          f"p99 {np.percentile(ms, 99):.2f}, max {ms.max():.2f}")
    if a.profile:
        tot = sm.sum()
        print(f"breakdown (mean): features {sm[0]:.2f} ms ({100 * sm[0] / tot:.1f}%), "
              f"encoding {sm[1]:.2f} ms ({100 * sm[1] / tot:.1f}%), "
              f"classification {sm[2]:.3f} ms ({100 * sm[2] / tot:.1f}%)")
    print(f"agreement with Python: predictions {100 * agree_pred:.2f}%, all 15 levels {100 * agree_lev:.2f}%, "
          f"Hamming distances {100 * agree_ham:.2f}%")
    if n == len(x):
        d, py = res["device_test"], res["python_test"]
        print(f"ON-DEVICE test F-beta pooled {d['fbeta_pooled']:.4f} (Python {py['fbeta_pooled']:.4f}), "
              f"sens {100 * d['sensitivity']:.1f}%, spec {100 * d['specificity']:.1f}%")
    print("wrote", a.out + ".json")


if __name__ == "__main__":
    main()
