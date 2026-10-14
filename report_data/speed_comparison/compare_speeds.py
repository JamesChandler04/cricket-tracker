"""Compares each delivery's radar speed with the per-frame speeds the top-down camera gave
for it, in a chart, a table and a printed summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if __package__ in (None, ""):
    # Started as a plain script (python report_data/speed_comparison/<name>.py, or the
    # Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.speed_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from report_data.speed_comparison.speed_data import (FOLDER, RADAR_PATH, SPEEDS_PATH, load_calculated_speeds,
                                                     load_radar_speeds)

CHART_PATH = FOLDER / "speed_comparison.png"
"""PNG file of the chart."""
TABLE_PATH = FOLDER / "speed_comparison.csv"
"""CSV of each delivery's radar speed, its lowest, mean and highest top-down speed, and
how far the mean is from the radar speed.
"""
WITHIN_PERCENT = 5.0
"""Default largest difference from the radar speed counted as agreeing, in percent. The
radar speeds' error bars on the chart are drawn this far either side.
"""

BAR_WIDTH = 0.6
"""Widest bar, in deliveries along the x axis."""
MAX_BAR_INCHES = 0.22
"""Widest bar on the saved chart, in inches, so that bars stay thin when there are few deliveries."""
BALL_GAP = 1.0
"""Extra space between the deliveries of one ball and the next, in deliveries."""
INCHES_PER_DELIVERY = 0.3
"""Chart width given to each delivery, in inches."""
CHART_WIDTH_RANGE = (8.0, 40.0)
"""Narrowest and widest chart, in inches."""

# Chart colours, the same as the other report charts.
SURFACE = "#fcfcfb"
"""Background of the chart."""
INK = "#0b0b0b"
"""Colour of the title."""
INK_2 = "#52514e"
"""Colour of the axis labels, legend and ball names."""
MUTED = "#898781"
"""Colour of the tick labels and notes."""
GRID = "#e1e0d9"
"""Colour of the grid lines."""
AXIS = "#c3c2b7"
"""Colour of the axes and the lines between balls."""
RADAR_COLOUR = "#2a78d6"
"""Blue for the radar speeds."""
CALCULATED_COLOUR = "#eb6834"
"""Orange for the top-down speeds."""

TABLE_COLUMNS = ["ball", "delivery", "radar_km_h", "frames", "lowest_km_h", "mean_km_h", "highest_km_h",
                 "mean_minus_radar_km_h", "mean_minus_radar_percent", "radar_within_range"]
"""Column headings of speed_comparison.csv."""


@dataclass
class Delivery:
    """One delivery's radar speed and the per-frame speeds the top-down camera gave for it."""

    ball: int
    number: int
    radar: float | None      # km/h, None if there is no radar reading
    frame_speeds: list[float]

    @property
    def lowest(self) -> float:
        """Return the lowest top-down speed, in km/h."""
        return min(self.frame_speeds)

    @property
    def highest(self) -> float:
        """Return the highest top-down speed, in km/h."""
        return max(self.frame_speeds)

    @property
    def mean(self) -> float:
        """Return the mean top-down speed, in km/h."""
        return sum(self.frame_speeds) / len(self.frame_speeds)

    @property
    def difference(self) -> float | None:
        """Return the mean top-down speed minus the radar speed in km/h, or None without a radar speed."""
        return None if self.radar is None else self.mean - self.radar

    @property
    def percent(self) -> float | None:
        """Return the mean top-down speed minus the radar speed as a percentage of the radar
        speed, or None without a radar speed.
        """
        difference = self.difference
        return None if difference is None or not self.radar else 100 * difference / self.radar

    @property
    def radar_in_range(self) -> bool | None:
        """Return whether the radar speed lies between the lowest and highest top-down speed,
        or None without a radar speed.
        """
        return None if self.radar is None else self.lowest <= self.radar <= self.highest


def load_deliveries(balls: list[int] | None = None) -> list[Delivery]:
    """Return every delivery with top-down speeds, with its radar speed if there is one,
    sorted by ball and then delivery. Only the given balls if balls is not None.
    """
    radar = load_radar_speeds(RADAR_PATH)
    deliveries = []
    for (ball, number), speeds in sorted(load_calculated_speeds(SPEEDS_PATH).items()):
        if speeds and (balls is None or ball in balls):
            deliveries.append(Delivery(ball, number, radar.get((ball, number)), speeds))
    return deliveries


def save_table(deliveries: list[Delivery], path: Path) -> None:
    """Write speed_comparison.csv: one row per delivery."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_COLUMNS)
        for delivery in deliveries:
            writer.writerow([delivery.ball, delivery.number, _format(delivery.radar, 1), len(delivery.frame_speeds),
                             f"{delivery.lowest:.1f}", f"{delivery.mean:.2f}", f"{delivery.highest:.1f}",
                             _format(delivery.difference, 2), _format(delivery.percent, 2),
                             "" if delivery.radar_in_range is None else ("yes" if delivery.radar_in_range else "no")])


def summarise(deliveries: list[Delivery], within: float) -> list[str]:
    """Return printable lines on how the mean top-down speeds compare with the radar speeds,
    overall and for each ball.
    """
    compared = [delivery for delivery in deliveries if delivery.radar is not None]
    lines = [f"Top-down against radar speed: {len(deliveries)} deliveries, {len(compared)} with a radar speed"]
    groups = [("All balls", compared)] + [(f"Ball {ball}", [d for d in compared if d.ball == ball])
                                          for ball in sorted({d.ball for d in compared})]
    for label, group in groups:
        lines.append(_describe(label, group, within))
    missing = [delivery for delivery in deliveries if delivery.radar is None]
    if missing:
        lines.append("No radar speed for: " + ", ".join(f"ball {d.ball} delivery {d.number}" for d in missing))
    return lines


def _describe(label: str, group: list[Delivery], within: float) -> str:
    """Return one summary line for a group of deliveries with radar speeds."""
    if not group:
        return f"  {label}: none to compare"
    differences = np.array([d.difference for d in group if d.difference is not None], dtype=np.float64)
    agreeing = sum(1 for d in group if d.percent is not None and abs(d.percent) <= within)
    inside = sum(1 for d in group if d.radar_in_range)
    sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    rms = float(np.sqrt(np.mean(differences ** 2)))
    return (f"  {label} (n={len(group)}): mean minus radar {differences.mean():+.1f} km/h, sd {sd:.1f}, "
            f"rms {rms:.1f}, largest {float(np.abs(differences).max()):.1f} km/h; "
            f"{agreeing} of {len(group)} ({agreeing / len(group):.0%}) within {within:g}%; "
            f"radar inside the top-down range for {inside}")


def plot_speeds(deliveries: list[Delivery], within: float, path: Path) -> Figure:
    """Draw and save the chart: each delivery's top-down speeds as a bar from lowest to highest
    with a line at the mean, and its radar speed as a dot with error bars of plus or minus
    within percent, grouped by ball along the x axis.
    """
    positions = _positions(deliveries)
    span = positions[-1] + 1
    width = min(max(2.0 + INCHES_PER_DELIVERY * span, CHART_WIDTH_RANGE[0]), CHART_WIDTH_RANGE[1])
    figure, axes = plt.subplots(figsize=(width, 5.4), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(True, axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)

    # The plot area is about 1.4 inches narrower than the chart.
    bar = min(BAR_WIDTH, MAX_BAR_INCHES * (span + 0.6) / (width - 1.4))
    for x, delivery in zip(positions, deliveries):
        axes.bar(x, delivery.highest - delivery.lowest, bottom=delivery.lowest, width=bar,
                 color=CALCULATED_COLOUR, alpha=0.3, linewidth=0, zorder=2)
        axes.hlines(delivery.mean, x - bar / 2, x + bar / 2, color=CALCULATED_COLOUR, linewidth=2, zorder=3)
    with_radar = [(x, d.radar) for x, d in zip(positions, deliveries) if d.radar is not None]
    radar_handle: ErrorbarContainer | Line2D = Line2D([], [], marker="o", linestyle="", markersize=7,
                                                      markerfacecolor=RADAR_COLOUR, markeredgecolor=SURFACE)
    if with_radar:
        radar_speeds = [speed for _, speed in with_radar]
        radar_handle = axes.errorbar([x for x, _ in with_radar], radar_speeds,
                                     yerr=[speed * within / 100 for speed in radar_speeds], fmt="o",
                                     markersize=6, markerfacecolor=RADAR_COLOUR, markeredgecolor=SURFACE,
                                     markeredgewidth=1.2, ecolor=RADAR_COLOUR, elinewidth=1.0, capsize=3.5,
                                     capthick=1.0, zorder=4)
        # The error bars go under the top-down mean lines, so that a mean stays readable
        # where it crosses one; the dots stay on top.
        _, caps, bars = radar_handle.lines
        for error_bar in (*caps, *bars):
            error_bar.set_zorder(2.5)

    speeds = [s for d in deliveries for s in (d.lowest, d.highest)]
    speeds += [s * (1 + side * within / 100) for _, s in with_radar for side in (-1, 1)]
    low, high = min(speeds), max(speeds)
    pad = max(2.0, 0.08 * (high - low))
    axes.set_ylim(min(low - pad, 100), max(high + pad, 140))
    axes.set_xlim(-0.8, span - 0.2)

    # Delivery numbers on the axis, the ball numbers under them, and a line between balls.
    axes.set_xticks(positions)
    axes.set_xticklabels([str(d.number) for d in deliveries], rotation=90 if len(deliveries) > 30 else 0)
    axes.tick_params(axis="x", colors=MUTED, labelsize=8, length=3)
    axes.tick_params(axis="y", colors=MUTED, labelsize=10)
    balls = sorted({d.ball for d in deliveries})
    centres = [float(np.mean([x for x, d in zip(positions, deliveries) if d.ball == ball])) for ball in balls]
    ball_axis = axes.secondary_xaxis("bottom")
    ball_axis.set_xticks(centres, labels=[f"Ball {ball}" for ball in balls])
    ball_axis.tick_params(length=0, pad=34 if len(deliveries) > 30 else 20, labelsize=10, labelcolor=INK_2)
    ball_axis.spines["bottom"].set_visible(False)
    ball_axis.set_xlabel("Delivery number", color=INK_2, fontsize=11)
    for index in range(1, len(deliveries)):
        if deliveries[index].ball != deliveries[index - 1].ball:
            axes.axvline((positions[index - 1] + positions[index]) / 2, color=AXIS, linewidth=0.8, zorder=1)

    axes.set_ylabel("Speed (km/h)", color=INK_2, fontsize=11)
    axes.set_title("Top-down speed against radar speed", color=INK, fontsize=12, loc="left")
    # legend() draws an ErrorbarContainer with its error bar, though matplotlib's type
    # stubs only allow Artists as handles.
    handles = [cast(Artist, radar_handle),
               Patch(facecolor=CALCULATED_COLOUR, alpha=0.3, linewidth=0),
               Line2D([], [], color=CALCULATED_COLOUR, linewidth=2)]
    labels = [f"Radar speed ± {within:g}%" if with_radar else "Radar speed", "Top-down speed range",
              "Top-down mean"]
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, labelcolor=INK_2,
                  bbox_to_anchor=(0.5, 0.995))
    figure.text(0.01, 0.01, _footnote(deliveries, within), color=MUTED, fontsize=9, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    return figure


def _positions(deliveries: list[Delivery]) -> list[float]:
    """Return each delivery's place along the x axis, leaving a gap between balls."""
    positions = []
    x = 0.0
    for index, delivery in enumerate(deliveries):
        if index and delivery.ball != deliveries[index - 1].ball:
            x += BALL_GAP
        positions.append(x)
        x += 1
    return positions


def _footnote(deliveries: list[Delivery], within: float) -> str:
    """Return the note under the chart: the overall difference, and any delivery without a radar speed."""
    compared = [d for d in deliveries if d.difference is not None]
    parts = []
    if compared:
        mean = float(np.mean([d.difference for d in compared if d.difference is not None]))
        agreeing = sum(1 for d in compared if d.percent is not None and abs(d.percent) <= within)
        parts.append(f"Mean top-down speed minus radar speed: {mean:+.1f} km/h on average. "
                     f"{agreeing} of {len(compared)} deliveries lie within {within:g}% of the radar speed.")
    missing = [d for d in deliveries if d.radar is None]
    if missing:
        parts.append("No radar speed: " + ", ".join(f"ball {d.ball} delivery {d.number}" for d in missing) + ".")
    return "  ".join(parts)


def _format(value: float | None, decimals: int) -> str:
    """Format a number to decimals places, or leave it blank when there is none."""
    return "" if value is None else f"{value:.{decimals}f}"


def main(argv: list[str] | None = None) -> None:
    """Compare the top-down speeds saved by the GUI with the radar speeds, and save the chart
    and table.
    """
    parser = argparse.ArgumentParser(description="Compare the top-down speeds with the radar speeds.")
    parser.add_argument("--balls", nargs="+", type=int, metavar="BALL", help="only these balls (default: all)")
    parser.add_argument("--within", type=float, default=WITHIN_PERCENT,
                        help=f"difference from the radar speed counted as agreeing, and the size of the "
                             f"radar error bars, in percent (default {WITHIN_PERCENT:g})")
    parser.add_argument("--no-show", action="store_true", help="save the chart without opening it")
    args = parser.parse_args(argv)

    if not RADAR_PATH.exists():
        print(f"There are no radar speeds: put radar_speeds.yaml in {FOLDER}.")
        raise SystemExit(1)
    deliveries = load_deliveries(args.balls)
    if not deliveries:
        print(f"There are no top-down speeds in {SPEEDS_PATH} yet"
              + ("" if args.balls is None else " for those balls")
              + ". Run the top-down stage in the GUI on videos named like Jordan_B1_Birdseye_003.mp4.")
        raise SystemExit(1)

    save_table(deliveries, TABLE_PATH)
    print("\n".join(summarise(deliveries, args.within)))
    figure = plot_speeds(deliveries, args.within, CHART_PATH)
    print(f"\nSaved {CHART_PATH.name} and {TABLE_PATH.name} to {FOLDER}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
