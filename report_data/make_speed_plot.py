"""Plot radar speed against top-down measured speed range, per delivery, for one ball."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

FOLDER = Path("report_data/")

RADAR_FILE = FOLDER / "radar_speeds.yaml"
ACTUAL_FILE = FOLDER / "actual_top_down_data.csv"
BALL = 1
OUTPUT = FOLDER / f"speeds_ball_{BALL}.png"

BOX_WIDTH = 0.6  # width of each min–max rectangle, in delivery units

# Rows look like: 1,1,[135, 118, 120, 118],-27
# The speed list is unquoted, so it can't be read as a normal CSV.
ROW = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*\[([^\]]*)\]\s*,\s*(-?[\d.]+)\s*$")


def load_radar(path, ball):
    """Return {delivery: speed_kmh} for one ball."""
    data = yaml.safe_load(path.read_text())
    deliveries = data.get(f"ball_{ball}", {})
    return {int(key.split("_")[1]): float(speed) for key, speed in deliveries.items()}


def load_actual(path, ball):
    """Return {delivery: [speed, ...]} for one ball."""
    speeds = {}
    for line in path.read_text().splitlines()[1:]:  # skip header
        if not line.strip():
            continue
        match = ROW.match(line)
        if not match:
            print(f"Skipping unreadable row: {line!r}")
            continue
        row_ball, delivery, speed_list, _seam = match.groups()
        if int(row_ball) != ball:
            continue
        values = [float(s) for s in speed_list.split(",") if s.strip()]
        if values:
            speeds[int(delivery)] = values
    return speeds


def main():
    radar = load_radar(RADAR_FILE, BALL)
    actual = load_actual(ACTUAL_FILE, BALL)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Top-down speeds: floating rectangle from min to max, with a bar at the mean
    for delivery, values in sorted(actual.items()):
        low, high = min(values), max(values)
        mean = sum(values) / len(values)
        left = delivery - BOX_WIDTH / 2
        ax.add_patch(Rectangle((left, low), BOX_WIDTH, high - low,
                               facecolor="tab:orange", alpha=0.35,
                               edgecolor="tab:orange", linewidth=1.2))
        ax.hlines(mean, left, left + BOX_WIDTH, colors="tab:orange", linewidth=2.5)

    # Radar speeds: one dot per delivery
    deliveries = sorted(radar)
    ax.scatter(deliveries, [radar[d] for d in deliveries],
               color="tab:blue", s=30, zorder=3)

    # Legend (patches added with add_patch don't register automatically)
    ax.scatter([], [], color="tab:blue", s=30, label="Radar speed")
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="tab:orange", alpha=0.35,
                           edgecolor="tab:orange", label="Top-down range (min–max)"))
    ax.plot([], [], color="tab:orange", linewidth=2.5, label="Top-down mean")

    all_delivs = sorted(set(radar) | set(actual))
    all_speeds = list(radar.values()) + [v for vals in actual.values() for v in vals]
    ax.set_xlim(min(all_delivs) - 1, max(all_delivs) + 1)
    ax.set_ylim(min(all_speeds) - 5, max(all_speeds) + 5)
    ax.set_xticks(all_delivs)
    ax.tick_params(axis="x", labelsize=8)

    ax.set_xlabel("Delivery")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Ball {BALL}: radar vs top-down speed")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()