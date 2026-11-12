"""Score the 12-room DeepStream run against the expert boxes and build data for the video -> out/center_data.json.
A polyp counts as caught if an AI box (conf >= 0.45) overlaps its expert box (IoU >= 0.2) on >= 3 frames.
An alarm episode = >= 3 of the last 5 detector passes (interval=2 -> a 15-frame window) with a box >= 0.45;
it is false if none of its boxes is on a polyp."""
import collections, csv, json
from pathlib import Path

H = Path.home(); EC = H / "endo-center"; GT = H / "colon-db/gt"
rooms = [l.split() for l in (EC / "rooms.txt").read_text().split("\n") if l.strip()]
N = 1800


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


ai = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(EC / "out/center_ai_wall.csv")):
    if float(r["conf"]) >= 0.45:
        ai[int(r["room"])][int(r["frame"])].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])
out = {"rooms": [], "stats": json.loads((EC / "out/center_stats_wall.json").read_text()),
       "bench": json.loads((EC / "out/center_stats_bench.json").read_text())}
tot_gt = tot_caught = tot_alarms = tot_false = 0
for k, (v, start) in enumerate(rooms):
    start = int(start)
    gt = collections.defaultdict(list)
    for r in csv.DictReader(open(GT / f"{v}.csv")):
        f = int(r["frame"]) - start
        if 0 <= f < N:
            gt[f].append((r["lesion_id"], [int(r[x]) for x in ("x1", "y1", "x2", "y2")]))
    lesions = sorted({lid for f in gt for lid, _ in gt[f]})
    hits = collections.Counter(); on_polyp = [False] * N; has = [False] * N
    for f in range(N):
        for b in ai[k].get(f, []):
            has[f] = True
            for lid, g in gt.get(f, []):
                if iou(b, g) >= 0.2:
                    hits[lid] += 1; on_polyp[f] = True
    caught = sorted(l for l in lesions if hits[l] >= 3)
    alarms, f = [], 0
    while f < N:
        if sum(has[f:f + 15]) >= 3:
            e = f
            while e < N and sum(has[e:e + 15]) >= 1:
                e += 1
            alarms.append({"t0": round(f / 30, 2), "t1": round(e / 30, 2), "true": any(on_polyp[f:e])}); f = e
        else:
            f += 1
    nf = sum(not a["true"] for a in alarms)
    tot_gt += len(lesions); tot_caught += len(caught); tot_alarms += len(alarms); tot_false += nf
    first = {l: round(min(f for f in gt if any(x == l for x, _ in gt[f])) / 30, 2) for l in lesions}
    out["rooms"].append({"room": k + 1, "video": v, "polyps": lesions, "caught": caught, "first_seen": first,
                         "alarms": alarms, "false_alarms": nf})
    print(v, "polyps", len(lesions), "caught", len(caught), "alarms", len(alarms), "false", nf)
out["totals"] = {"polyps": tot_gt, "caught": tot_caught, "alarms": tot_alarms, "false_alarms": tot_false, "minutes": 12}
json.dump(out, open(EC / "out/center_data.json", "w"), indent=1)
print("TOTAL", out["totals"])
