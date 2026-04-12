from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath


OUTPUT_FILE = Path("docs/plots/diagrams/workflow_diagram_1.png")


def add_glow_line(ax, points, base_color, glow_color, width=6, glow_width=12):
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=glow_color,
        linewidth=glow_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        alpha=0.35,
        zorder=1,
    )
    ax.plot(
        xs,
        ys,
        color=base_color,
        linewidth=width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )


def add_arrow_head(ax, start, end, base_color, glow_color, width=6, glow_scale=34, scale=26):
    glow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=glow_scale,
        linewidth=0,
        color=glow_color,
        alpha=0.35,
        zorder=1,
    )
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=0,
        color=base_color,
        zorder=3,
    )
    ax.add_patch(glow)
    ax.add_patch(arrow)


def add_operation_label(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#222222",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.18",
            "facecolor": "white",
            "edgecolor": "#d8d8d8",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )


def draw_document(ax, x, y, label):
    width = 9.0
    height = 12.5
    fold = 2.5
    vertices = [
        (x, y),
        (x + width - fold, y),
        (x + width, y + fold),
        (x + width, y + height),
        (x, y + height),
        (x, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    page = PathPatch(
        MplPath(vertices, codes),
        facecolor="#edf7ff",
        edgecolor="#3d7ba6",
        linewidth=1.4,
        joinstyle="round",
        zorder=5,
    )
    page.set_path_effects([pe.withSimplePatchShadow(offset=(1, -1), alpha=0.15)])
    ax.add_patch(page)

    fold_patch = Polygon(
        [(x + width - fold, y), (x + width - fold, y + fold), (x + width, y + fold)],
        closed=True,
        facecolor="#d8ebfb",
        edgecolor="#3d7ba6",
        linewidth=1.2,
        zorder=6,
    )
    ax.add_patch(fold_patch)

    for idx in range(5):
        y_line = y + 2.3 + idx * 1.45
        ax.plot([x + 1.5, x + 5.8], [y_line, y_line], color="#5f85a3", linewidth=1.0, zorder=6)

    gear_center = (x + 6.8, y + 8.8)
    gear = Circle(gear_center, 2.1, facecolor="#d6e9fb", edgecolor="#648aad", linewidth=1.2, zorder=6)
    ax.add_patch(gear)
    hub = Circle(gear_center, 0.75, facecolor="white", edgecolor="#648aad", linewidth=1.0, zorder=7)
    ax.add_patch(hub)
    for dx, dy in [(0, -2.7), (1.9, -1.9), (2.7, 0), (1.9, 1.9), (0, 2.7), (-1.9, 1.9), (-2.7, 0), (-1.9, -1.9)]:
        tooth = Circle((gear_center[0] + dx, gear_center[1] + dy), 0.33, facecolor="#648aad", edgecolor="none", zorder=7)
        ax.add_patch(tooth)

    ax.text(x + width / 2, y + height + 3.2, label, ha="center", va="bottom", fontsize=10, color="#222222")
    ax.text(x + width / 2, y + height + 6.0, "+ item definitions", ha="center", va="bottom", fontsize=9, color="#222222")


def draw_summary(ax, x, y, title, body_text):
    width = 28
    height = 13.5
    outer = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor="#e6f7ec",
        edgecolor="#69b78c",
        linewidth=1.2,
        zorder=5,
    )
    outer.set_path_effects([pe.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.18), pe.withStroke(linewidth=7, foreground="#b7f0ca", alpha=0.25)])
    ax.add_patch(outer)

    ax.text(x + width / 2, y + 1.5, title, ha="center", va="center", fontsize=10.5, color="#222222", zorder=7)

    inner = FancyBboxPatch(
        (x + 1.0, y + 4.3),
        width - 2.0,
        5.8,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        facecolor="#ecfbf1",
        edgecolor="#67a883",
        linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(inner)

    ax.plot([x + 2.3, x + 5.2], [y + 5.9, y + 5.9], color="#488467", linewidth=1.2, zorder=7)
    ax.plot([x + 2.3, x + 5.2], [y + 6.8, y + 6.8], color="#488467", linewidth=1.2, zorder=7)
    ax.plot([x + 2.3, x + 5.2], [y + 7.7, y + 7.7], color="#488467", linewidth=1.2, zorder=7)
    ax.plot([x + 2.3, x + 2.9], [y + 5.9, y + 5.9], marker="s", markersize=2.6, color="#488467", zorder=7)
    ax.plot([x + 2.3, x + 2.9], [y + 6.8, y + 6.8], marker="s", markersize=2.6, color="#488467", zorder=7)
    ax.plot([x + 2.3, x + 2.9], [y + 7.7, y + 7.7], marker="s", markersize=2.6, color="#488467", zorder=7)

    ax.text(x + 4.5, y + 6.8, body_text, ha="left", va="center", fontsize=8.4, color="#2c4636", zorder=7)


def build_diagram():
    fig, ax = plt.subplots(figsize=(13.6, 7.68), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 130)
    ax.set_ylim(100, 0)
    ax.axis("off")

    docs = {
        "Doc 1": (7, 6),
        "Doc 2": (46.0, 6),
        "Doc 3": (85.0, 6),
    }
    summaries = {
        "Summary 1": (7, 42),
        "Summary 2": (49, 42),
        "Summary 3": (91, 42),
    }

    draw_document(ax, *docs["Doc 1"], "Doc 1")
    draw_document(ax, *docs["Doc 2"], "Doc 2")
    draw_document(ax, *docs["Doc 3"], "Doc 3")

    draw_summary(ax, *summaries["Summary 1"], "Summary 1", "{item: span} -> Summary 1")
    draw_summary(ax, *summaries["Summary 2"], "Summary 2", "{item: span} -> Summary 2")
    draw_summary(ax, *summaries["Summary 3"], "Summary 3", "{item: span} -> Summary 3")

    blue = "#0f5e86"
    blue_glow = "#5fd0f0"
    green = "#20895e"
    green_glow = "#6ee7a0"

    extract_points = [(16.0, 13.8), (19.5, 13.8), (19.5, 42.0)]
    update_2_points = [(55.0, 13.8), (61.5, 13.8), (61.5, 42.0)]
    update_3_points = [(94.0, 13.8), (103.5, 13.8), (103.5, 42.0)]
    feed_1_points = [(32.0, 49.0), (49.0, 49.0)]
    feed_2_points = [(74.0, 49.0), (91.0, 49.0)]

    for points in (extract_points, update_2_points, update_3_points):
        add_glow_line(ax, points, blue, blue_glow, width=6, glow_width=11)
        add_arrow_head(ax, points[-2], points[-1], blue, blue_glow)

    for points in (feed_1_points, feed_2_points):
        add_glow_line(ax, points, green, green_glow, width=6, glow_width=11)
        add_arrow_head(ax, points[-2], points[-1], green, green_glow)

    add_operation_label(ax, extract_points[1][0] + 4.0, (extract_points[1][1] + extract_points[2][1]) / 2, "extract")
    add_operation_label(ax, update_2_points[1][0] + 4.0, (update_2_points[1][1] + update_2_points[2][1]) / 2, "update")
    add_operation_label(ax, update_3_points[1][0] + 4.0, (update_3_points[1][1] + update_3_points[2][1]) / 2, "update")
    add_operation_label(ax, (feed_1_points[0][0] + feed_1_points[1][0]) / 2, feed_1_points[0][1] - 5.2, "feed")
    add_operation_label(ax, (feed_2_points[0][0] + feed_2_points[1][0]) / 2, feed_2_points[0][1] - 5.2, "feed")

    return fig


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig = build_diagram()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved diagram to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
