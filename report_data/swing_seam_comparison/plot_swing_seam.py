"""Plots the swing and seam comparison from swing_seam_comparison.csv in two charts: the swing
against seam angle for every delivery, and the same zoomed in on the seam angles counted as
none. The swing is the one at the common distance calculate_swing gives every delivery (8 m
by default), or with --last-point the one at each delivery's last tracked point. Each
delivery has an error bar over the speed error, and its marker and error bar are coloured by
its speed, from green for the slowest to red for the fastest. A summary is printed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if __package__ in (None, ""):
    # Started as a plain script (python report_data/swing_seam_comparison/<name>.py, or
    # the Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.swing_seam_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.automated.automatic_seam_angle import MAX_SPREAD_DEG
from report_data.swing_seam_comparison.swing_seam_data import (FOLDER, SEAM_METHODS, TABLE_PATH, SwingValues,
                                                               load_swing_values)

FloatArray = npt.NDArray[np.float64]
"""Type of the float64 numpy arrays used in this module's type annotations."""

SEAM_CHART_PATH = FOLDER / "swing_against_seam_angle.png"
"""PNG file of the chart of swing against seam angle for every delivery."""
NO_SEAM_CHART_PATH = FOLDER / "swing_no_seam_angle.png"
"""PNG file of the same chart zoomed in on the seam angles counted as none."""
ZERO_SEAM_DEG = 2.5
"""Largest seam angle either way that counts as no seam angle, in degrees."""
ZOOM_REACH_DEG = 3.0
"""How far the zoomed chart's seam angle axis reaches either side of 0, in degrees."""
BOOTSTRAP_SAMPLES = 10_000
"""Resamples of the deliveries used for the printed 95% confidence intervals."""
BOOTSTRAP_SEED = 0
"""Seed of the resampling, so that the confidence intervals are the same on every run."""
LEGEND_COLUMN_INCHES = 2.6
"""Width given to each column of a legend, in inches."""
CAP_SIZE_PT = 4.0
"""Half the width of the caps on the error bars, in points."""
LEGEND_ROW_INCHES = 0.25
"""Height given to each row of a legend, in inches."""
FOOTNOTE_LINE_FRACTION = 0.0235
"""Height left under a chart for each line of the note under it, as a fraction of the figure's height."""
SWING_MARGIN = 0.08
"""Room left above and below the error bars on the swing axis, as a fraction of their span."""
SPEED_PAD_KM_H = 1.0
"""How far the speed colour scale reaches either side of the speed when every delivery has the same one, in km/h."""

# Chart colours, the same as the other report charts.
SURFACE = "#fcfcfb"
"""Background of the charts."""
INK = "#0b0b0b"
"""Colour of the titles."""
INK_2 = "#52514e"
"""Colour of the axis labels and legends."""
MUTED = "#898781"
"""Colour of the tick labels and the notes."""
GRID = "#e1e0d9"
"""Colour of the grid lines and the band of seam angles counted as none."""
AXIS = "#c3c2b7"
"""Colour of the axes and the zero lines."""
SPEED_COLOURS = ("#339e44", "#818400", "#a66500", "#b64600", "#b52a2a")
"""Colour scale for the deliveries' speeds, from green for the slowest to red for the
fastest. The colours get steadily darker towards red, so the order still reads in
greyscale and for colour-blind readers, and each stands out at least 3:1 from the
background.
"""
LEGEND_COLOUR = INK_2
"""Grey for the delivery in the legend, since a delivery's colour shows its speed."""
FIT_COLOUR = INK
"""Near-black for the fitted line, so that it stands apart from the speed colours."""


@dataclass
class Delivery:
    """One delivery on a chart: its swing values and the seam angle it is plotted at."""

    values: SwingValues
    seam_deg: float


@dataclass
class LineFit:
    """Straight-line fit of swing against seam angle, with bootstrap 95% confidence intervals."""

    intercept_cm: float           # swing the line gives at no seam angle
    slope_cm_per_deg: float
    intercept_ci: tuple[float, float]
    slope_ci: tuple[float, float]
    pearson_r: float
    spearman_rho: float


def at_common_distance(values: list[SwingValues]) -> tuple[list[SwingValues], list[SwingValues]]:
    """Return copies of the rows with their swings at the common distance in place of those
    at the last tracked point, and the rows with no swing at the common distance. A slower
    or faster swing that is missing there is replaced by the swing at the speed used.
    """
    moved, left_out = [], []
    for row in values:
        swing = row.common_swing_cm
        if swing is None:
            left_out.append(row)
            continue
        slower, faster = row.common_swing_slower_cm, row.common_swing_faster_cm
        moved.append(replace(row, swing_cm=swing, swing_slower_cm=swing if slower is None else slower,
                             swing_faster_cm=swing if faster is None else faster))
    return moved, left_out


def seam_deliveries(values: list[SwingValues], method: str) -> tuple[list[Delivery], list[SwingValues]]:
    """Return the deliveries with a seam angle measured by method, sorted by seam angle, and
    those without one.
    """
    deliveries = [Delivery(row, angle) for row in values if (angle := row.seam_angle(method)) is not None]
    missing = [row for row in values if row.seam_angle(method) is None]
    return sorted(deliveries, key=lambda delivery: delivery.seam_deg), missing


def fit_line(angles_deg: FloatArray, swings_cm: FloatArray) -> LineFit | None:
    """Fit swing = a + b × seam angle by least squares, with 95% confidence intervals for a
    and b from BOOTSTRAP_SAMPLES resamples of the deliveries. Returns None with fewer than
    three deliveries, or if they all have the same seam angle.
    """
    count = len(angles_deg)
    if count < 3 or np.ptp(angles_deg) == 0:
        return None
    slope, intercept = np.polyfit(angles_deg, swings_cm, 1)

    picks = np.random.default_rng(BOOTSTRAP_SEED).integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
    x, y = angles_deg[picks], swings_cm[picks]
    x_mean, y_mean = x.mean(axis=1), y.mean(axis=1)
    spread = ((x - x_mean[:, None]) ** 2).sum(axis=1)
    usable = spread > 0  # a resample with one seam angle repeated has no slope
    slopes = ((x - x_mean[:, None]) * (y - y_mean[:, None])).sum(axis=1)[usable] / spread[usable]
    intercepts = y_mean[usable] - slopes * x_mean[usable]

    return LineFit(
        intercept_cm=float(intercept),
        slope_cm_per_deg=float(slope),
        intercept_ci=_interval(intercepts),
        slope_ci=_interval(slopes),
        pearson_r=float(np.corrcoef(angles_deg, swings_cm)[0, 1]),
        spearman_rho=float(np.corrcoef(_ranks(angles_deg), _ranks(swings_cm))[0, 1]),
    )


def mean_interval(swings_cm: FloatArray) -> tuple[float, float] | None:
    """Return the 95% confidence interval of the mean swing from BOOTSTRAP_SAMPLES resamples
    of the deliveries, or None with fewer than two.
    """
    count = len(swings_cm)
    if count < 2:
        return None
    picks = np.random.default_rng(BOOTSTRAP_SEED).integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
    return _interval(swings_cm[picks].mean(axis=1))


def summarise(deliveries: list[Delivery], missing: list[SwingValues], method: str, fit: LineFit | None,
              distance_m: float | None = None) -> list[str]:
    """Return printable lines on how the swing follows the seam angle, and on the swing of
    the deliveries with no seam angle. distance_m is the common distance the swings are at,
    or None for swings at the last tracked point.
    """
    values = [delivery.values for delivery in deliveries]
    distances = np.array([row.last_point_distance_m for row in values], dtype=np.float64)
    lines = [f"Swing {_place(distance_m)} against the {method} seam angle. {_speed_note(values)} "
             f"Error bars cover ±{values[0].speed_error_percent:g}% speed.",
             f"Tracked {distances.min():.1f} to {distances.max():.1f} m from the first tracked point "
             f"(mean {distances.mean():.1f} m)."]
    if distance_m is not None:
        lines.append(_reach_note(values, distance_m))
    if missing:
        lines.append(f"No {method} seam angle: " + ", ".join(f"ball {row.ball} delivery {row.delivery}"
                                                            for row in missing))

    lines.append(f"\nAll {len(deliveries)} deliveries:")
    if fit is None:
        lines.append("  Too few deliveries, or seam angles, for a line fit.")
    else:
        sign = "-" if fit.slope_cm_per_deg < 0 else "+"
        lines.append(f"  Fit: swing = {fit.intercept_cm:.2f} cm {sign} {abs(fit.slope_cm_per_deg):.3f} cm per degree "
                     f"of seam angle")
        low, high = fit.slope_ci
        trend = ("above 0, so the swing follows the seam angle" if low > 0 else
                 "below 0, so the swing goes against the seam angle" if high < 0 else "includes 0, so no clear trend")
        lines.append(f"  Slope: 95% CI {low:+.3f} to {high:+.3f} cm per degree, {trend}")
        low, high = fit.intercept_ci
        lines.append(f"  Swing the line gives at no seam angle: 95% CI {low:+.2f} to {high:+.2f} cm, which "
                     + ("includes 0" if low <= 0 <= high else "does not include 0"))
        lines.append(f"  Pearson r {fit.pearson_r:+.2f}, Spearman rho {fit.spearman_rho:+.2f}")

    no_seam = _no_seam(deliveries)
    lines.append(f"\nNo seam angle (within ±{ZERO_SEAM_DEG:g} deg): {len(no_seam)} deliveries")
    if no_seam:
        swings = np.array([delivery.values.swing_cm for delivery in no_seam], dtype=np.float64)
        interval = mean_interval(swings)
        ci = "" if interval is None else f" (95% CI {interval[0]:+.2f} to {interval[1]:+.2f} cm)"
        sd = f", sd {float(swings.std(ddof=1)):.2f} cm" if len(swings) > 1 else ""
        lines.append(f"  Mean swing {float(swings.mean()):+.2f} cm{ci}{sd}, largest "
                     f"{float(np.abs(swings).max()):.2f} cm")
        lines.append(f"  {_bars_through_zero(no_seam)} of {len(no_seam)} error bars include 0 cm")
    return lines


def plot_swing(deliveries: list[Delivery], method: str, fit: LineFit | None, path: Path, zoomed: bool,
               distance_m: float | None = None) -> Figure:
    """Draw and save a chart of the swing against seam angle, with an error bar over the
    speed error for each delivery, coloured by its speed on a scale beside the chart, and
    the line fitted to every delivery. The swing is at the common distance distance_m, or
    at the last tracked point when it is None. The zoomed chart shows only the seam angles
    counted as none, coloured on the same speed scale as the full chart.
    """
    shown = _no_seam(deliveries) if zoomed else deliveries
    values = [delivery.values for delivery in shown]
    angles = np.array([delivery.seam_deg for delivery in shown], dtype=np.float64)
    left, right = (-ZOOM_REACH_DEG, ZOOM_REACH_DEG) if zoomed else _angle_limits(angles)

    figure, axes = plt.subplots(figsize=(10.0, 6.0), facecolor=SURFACE)
    _style(axes, "both")
    if not zoomed:
        axes.axvspan(-ZERO_SEAM_DEG, ZERO_SEAM_DEG, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    axes.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
    axes.axvline(0, color=AXIS, linewidth=0.9, zorder=1)
    if fit is not None:
        ends = np.array([left, right])
        axes.plot(ends, fit.intercept_cm + fit.slope_cm_per_deg * ends, color=FIT_COLOUR, linewidth=2, zorder=3)

    scale = _speed_scale([delivery.values for delivery in deliveries])
    warned = [method == "automatic" and row.automatic_spread_deg is not None
              and row.automatic_spread_deg > MAX_SPREAD_DEG for row in values]
    for angle, row, hollow in zip(angles, values, warned):
        _error_bar(axes, float(angle), row, _speed_colour(scale, row.speed_km_h), hollow)
    axes.set_xlim(left, right)
    _set_swing_limits(axes, values)
    axes.set_xlabel("Seam angle from the direction of travel (°)", color=INK_2, fontsize=11)
    place = _place(distance_m)
    axes.set_ylabel(f"Swing {place} (cm)", color=INK_2, fontsize=11)
    axes.set_title(f"Swing {place}, seam angles within ±{ZERO_SEAM_DEG:g}°" if zoomed
                   else f"Swing {place} against seam angle", color=INK, fontsize=12, loc="left")

    bar = figure.colorbar(scale, ax=axes, fraction=0.035, pad=0.015, aspect=30)
    bar.set_label("Delivery speed (km/h)", color=INK_2, fontsize=10)
    bar.ax.tick_params(colors=MUTED, labelsize=9)
    bar.outline.set_visible(False)

    # legend() draws an ErrorbarContainer with its error bar, though matplotlib's type
    # stubs only allow Artists as handles.
    error = values[0].speed_error_percent
    handles: list[Artist] = []
    labels = []
    if not all(warned):
        handles.append(cast(Artist, _legend_bar(axes, hollow=False)))
        labels.append(f"Delivery, ±{error:g}% speed range")
    if any(warned):
        handles.append(cast(Artist, _legend_bar(axes, hollow=True)))
        labels.append(f"Seam spread over {MAX_SPREAD_DEG:g}°")
    if fit is not None:
        handles.append(Line2D([], [], color=FIT_COLOUR, linewidth=2))
        labels.append("Line of best fit")
    if not zoomed:
        handles.append(Patch(facecolor=GRID, alpha=0.6))
        labels.append(f"No seam angle (±{ZERO_SEAM_DEG:g}°)")
    top = _legend(figure, handles, labels)
    footnote = _footnote(deliveries, fit, distance_m)
    figure.text(0.01, 0.01, footnote, color=MUTED, fontsize=8.5, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.01 + FOOTNOTE_LINE_FRACTION * (footnote.count("\n") + 1), 1, top))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    return figure


def _no_seam(deliveries: list[Delivery]) -> list[Delivery]:
    """Return the deliveries whose seam angle counts as none."""
    return [delivery for delivery in deliveries if abs(delivery.seam_deg) <= ZERO_SEAM_DEG]


def _style(axes: Axes, grid: Literal["both", "x", "y"]) -> None:
    """Give a chart's axes the report style, with grid lines across the axis or axes named by grid."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, axis=grid, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelsize=10)


def _legend(figure: Figure, handles: list[Artist], labels: list[str]) -> float:
    """Put the legend above the chart, so that it covers no data, in as many columns as fit
    the figure's width. Returns the fraction of the figure's height left for the chart.
    """
    width, height = figure.get_size_inches()
    columns = max(1, min(len(handles), int(width // LEGEND_COLUMN_INCHES)))
    figure.legend(handles, labels, loc="upper center", ncol=columns, frameon=False, fontsize=9.5, labelcolor=INK_2,
                  bbox_to_anchor=(0.5, 0.995))
    rows = -(-len(handles) // columns)
    return 0.985 - rows * LEGEND_ROW_INCHES / float(height)


def _speed_scale(values: list[SwingValues]) -> ScalarMappable:
    """Return the colour scale of the deliveries' speeds, from SPEED_COLOURS, running from the
    slowest delivery to the fastest.
    """
    speeds = [row.speed_km_h for row in values]
    low, high = min(speeds), max(speeds)
    if high - low < SPEED_PAD_KM_H / 10:  # one speed for every delivery
        low, high = low - SPEED_PAD_KM_H, high + SPEED_PAD_KM_H
    return ScalarMappable(norm=Normalize(vmin=low, vmax=high),
                          cmap=LinearSegmentedColormap.from_list("delivery_speed", SPEED_COLOURS))


def _speed_colour(scale: ScalarMappable, speed_km_h: float) -> str:
    """Return the colour of a speed on the speed colour scale."""
    low, high = scale.get_clim()
    return to_hex(scale.get_cmap()(min(max((speed_km_h - low) / (high - low), 0.0), 1.0)))


def _error_bar(axes: Axes, angle: float, row: SwingValues, colour: str, hollow: bool) -> None:
    """Draw a delivery's swing at its seam angle in colour, with an error bar from its lowest to
    its highest swing over the speed error. A hollow marker is white inside.
    """
    low, high = row.swing_range_cm
    axes.errorbar([angle], [row.swing_cm], yerr=[[row.swing_cm - low], [high - row.swing_cm]], fmt="o", markersize=6,
                  markerfacecolor=SURFACE if hollow else colour, markeredgecolor=colour if hollow else SURFACE,
                  markeredgewidth=1.4 if hollow else 1.0, ecolor=colour, elinewidth=1.2, capsize=CAP_SIZE_PT,
                  capthick=1.2, zorder=4)


def _legend_bar(axes: Axes, hollow: bool) -> ErrorbarContainer:
    """Return an error bar in LEGEND_COLOUR that draws nothing on the chart, to stand for the
    deliveries in the legend. A hollow marker is white inside.
    """
    return axes.errorbar([np.nan], [np.nan], yerr=[[np.nan], [np.nan]], fmt="o", markersize=6,
                         markerfacecolor=SURFACE if hollow else LEGEND_COLOUR,
                         markeredgecolor=LEGEND_COLOUR if hollow else SURFACE,
                         markeredgewidth=1.4 if hollow else 1.0, ecolor=LEGEND_COLOUR, elinewidth=1.2,
                         capsize=CAP_SIZE_PT, capthick=1.2)


def _set_swing_limits(axes: Axes, values: list[SwingValues]) -> None:
    """Set the swing axis to cover 0 and every error bar, with a margin of SWING_MARGIN."""
    shown = [0.0, *(swing for row in values for swing in row.swing_range_cm)]
    low, high = min(shown), max(shown)
    span = max(high - low, 1.0)
    axes.set_ylim(low - SWING_MARGIN * span, high + SWING_MARGIN * span)


def _bars_through_zero(deliveries: list[Delivery]) -> int:
    """Return how many of the deliveries' error bars include no swing."""
    return sum(1 for delivery in deliveries
               if delivery.values.swing_range_cm[0] <= 0 <= delivery.values.swing_range_cm[1])


def _footnote(deliveries: list[Delivery], fit: LineFit | None, distance_m: float | None = None) -> str:
    """Return the note under a chart: the fit, where the swing is taken, what the colours
    show, and the speeds.
    """
    values = [delivery.values for delivery in deliveries]
    lines = ["Colour: the delivery's speed, from green for the slowest to red for the fastest.",
             f"Error bars: swing difference at speeds {values[0].speed_error_percent:g}% below and above.",
             f"{_speed_note(values)} Negative swing is to the bowler's left."]
    if distance_m is not None:
        lines.insert(0, f"Swing at {distance_m:g} m from the first tracked point: deliveries tracked further are cut "
                        f"there, and shorter ones were extrapolated out.")
    if fit is not None:
        sign = "-" if fit.slope_cm_per_deg < 0 else "+"
        lines.insert(0, f"Fit to all {len(deliveries)} deliveries: swing = {fit.intercept_cm:.2f} {sign} "
                        f"{abs(fit.slope_cm_per_deg):.3f} x seam angle (cm).")
    return "\n".join(lines)


def _place(distance_m: float | None) -> str:
    """Return where the swing is taken, for titles and notes: at the common distance
    distance_m, or at the last tracked point when it is None.
    """
    return "at the last tracked point" if distance_m is None else f"at {distance_m:g} m"


def _reach_note(values: list[SwingValues], distance_m: float) -> str:
    """Return the sentence saying how the deliveries' swings were taken to the common distance distance_m."""
    carried = [row.extrapolated_m or 0.0 for row in values if not row.points_past_common_distance]
    cut = [row for row in values if row.points_past_common_distance]
    parts = []
    if carried:
        parts.append(f"{len(carried)} carried on to it from shorter tracks, by {min(carried):.1f} to "
                     f"{max(carried):.1f} m, assuming a constant side force")
    if cut:
        parts.append(f"{len(cut)} tracked past it and cut there")
    return f"Every swing is at {distance_m:g} m from the first tracked point: " + "; ".join(parts) + "."


def _speed_note(values: list[SwingValues]) -> str:
    """Return the sentence saying what speeds the deliveries were reconstructed at."""
    sources = {row.speed_source for row in values}
    if sources == {"set"}:
        return f"Every delivery reconstructed at {values[0].speed_km_h:g} km/h."
    if "radar" in sources:
        return ("Reconstructed using each delivery's radar speed"
                + (", or its mean top-down speed without one." if "top-down (no radar)" in sources else "."))
    return "Reconstructed at each delivery's mean top-down speed."


def _angle_limits(angles: FloatArray) -> tuple[float, float]:
    """Return the full chart's seam angle axis limits: every delivery and the band of seam
    angles counted as none, with a margin, in degrees.
    """
    low = min(float(angles.min()), -ZERO_SEAM_DEG)
    high = max(float(angles.max()), ZERO_SEAM_DEG)
    margin = max(1.0, 0.05 * (high - low))
    return low - margin, high + margin


def _interval(values: FloatArray) -> tuple[float, float]:
    """Return the middle 95% of values, as (low, high)."""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _ranks(values: FloatArray) -> FloatArray:
    """Return the ranks of values, from 1, with tied values given their average rank."""
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[np.argsort(values, kind="mergesort")] = np.arange(1, len(values) + 1)
    _, groups, sizes = np.unique(values, return_inverse=True, return_counts=True)
    return np.bincount(groups, weights=ranks)[groups] / sizes[groups]


def main(argv: list[str] | None = None) -> None:
    """Plot the saved swing values in the two charts, print the summary and show the charts."""
    parser = argparse.ArgumentParser(description="Plot the swing values saved by calculate_swing.py against the "
                                                 "seam angles.")
    parser.add_argument("--seam", choices=SEAM_METHODS, default="manual",
                        help="seam angles to plot the swing against (default manual)")
    parser.add_argument("--last-point", action="store_true",
                        help="plot the swing at each delivery's last tracked point instead of at the common distance")
    parser.add_argument("--no-show", action="store_true", help="save the charts without opening them")
    args = parser.parse_args(argv)

    values = load_swing_values(TABLE_PATH) if TABLE_PATH.exists() else []
    if not values:
        print(f"There are no swing values in {TABLE_PATH} yet. Make them with calculate_swing.py first.")
        raise SystemExit(1)
    distance_m = None
    if not args.last_point:
        known = [row.common_distance_m for row in values if row.common_distance_m is not None]
        if len(known) < len(values):
            print(f"{TABLE_PATH.name} has no swing at a common distance. Run calculate_swing.py again to add it, "
                  f"or plot the swing at the last tracked point with --last-point.")
            raise SystemExit(1)
        distance_m = known[0]
        values, left_out = at_common_distance(values)
        for row in left_out:
            print(f"Ball {row.ball} delivery {row.delivery}: left out, too few points before {distance_m:g} m")
        if not values:
            print(f"No delivery has a swing at {distance_m:g} m.")
            raise SystemExit(1)
    deliveries, missing = seam_deliveries(values, args.seam)
    if not deliveries:
        print(f"No delivery in {TABLE_PATH.name} has a {args.seam} seam angle"
              + ("." if args.seam == "automatic" else "; try --seam automatic."))
        raise SystemExit(1)

    fit = fit_line(np.array([delivery.seam_deg for delivery in deliveries], dtype=np.float64),
                   np.array([delivery.values.swing_cm for delivery in deliveries], dtype=np.float64))
    print("\n".join(summarise(deliveries, missing, args.seam, fit, distance_m)))

    figures = [plot_swing(deliveries, args.seam, fit, SEAM_CHART_PATH, zoomed=False, distance_m=distance_m)]
    saved = [SEAM_CHART_PATH.name]
    if _no_seam(deliveries):
        figures.append(plot_swing(deliveries, args.seam, fit, NO_SEAM_CHART_PATH, zoomed=True, distance_m=distance_m))
        saved.append(NO_SEAM_CHART_PATH.name)
    else:
        NO_SEAM_CHART_PATH.unlink(missing_ok=True)  # an earlier run's chart would no longer match the data
    print(f"\nSaved {' and '.join(saved)} to {FOLDER}")
    if not args.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
