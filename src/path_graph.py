from __future__ import annotations
from pathlib import Path
import csv
import matplotlib.pyplot as plt

CSV_PATH = Path("outputs/bar_path.csv")

xs, ys = [], []
with open(CSV_PATH, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))

plt.figure(figsize=(6, 6))
plt.plot(xs, ys)
plt.gca().invert_yaxis()  # image y increases downward
plt.title("Bar path (pixel coordinates)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()