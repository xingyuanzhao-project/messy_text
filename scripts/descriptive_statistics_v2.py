"""
Generate descriptive statistics plots for the selected three labels.

This script is self-contained on purpose: it carries the original and merged
category definitions needed for plotting so one script run always regenerates
the current descriptive plot set without depending on the current reduced
taxonomy file.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "df_text_by_report.csv"
OUTPUT_DIR = ROOT / "plots" / "descriptive_statistics"
OUTPUT_SUFFIX = "_v2"

LABEL_CONFIGS = {
    "desenlace": {
        "original_options": [
            "Still disappeared",
            "Liberated by captors",
            "Liberated by authorities",
            "Found dead",
            "Escaped or was liberated through their own means",
            "Found alive",
            "Found, but does not specify if dead or alive",
            "No information",
        ],
        "merged_options": ["Still disappeared", "Found", "No information"],
        "category_merging": {
            "Still disappeared": "Still disappeared",
            "Liberated by captors": "Found",
            "Liberated by authorities": "Found",
            "Found dead": "Found",
            "Escaped or was liberated through their own means": "Found",
            "Found alive": "Found",
            "Found, but does not specify if dead or alive": "Found",
            "No information": "No information",
        },
        "original_short_labels": {
            "Still disappeared": "Still disappeared",
            "Liberated by captors": "Lib. by captors",
            "Liberated by authorities": "Lib. by authorities",
            "Found dead": "Found dead",
            "Escaped or was liberated through their own means": "Escaped / self-liberated",
            "Found alive": "Found alive",
            "Found, but does not specify if dead or alive": "Found, unspecified",
            "No information": "No information",
        },
    },
    "vic_grupo_social": {
        "original_options": [
            "Professionals (Entrepreneur, Engineer, Professor, Journalist, etc)",
            "People that work in service industries (taxi driver, salesman, etc)",
            "Civil servants (Police, mayor, public worker, etc)",
            "Belonging to some sexual identity group (LGBTQ)",
            "People associated with politics",
            "Activists (political activist, human rights, etc)",
            "Organized crime",
            "Students",
            "Land Worker",
            "Other",
            "No information",
        ],
        "merged_options": [
            "Professionals",
            "Civil servants",
            "LGBTQ",
            "Activists",
            "Organized crime",
            "Students",
            "Other",
            "No information",
        ],
        "category_merging": {
            "Professionals (Entrepreneur, Engineer, Professor, Journalist, etc)": "Professionals",
            "People that work in service industries (taxi driver, salesman, etc)": "Professionals",
            "Civil servants (Police, mayor, public worker, etc)": "Civil servants",
            "Belonging to some sexual identity group (LGBTQ)": "Other",
            "People associated with politics": "Civil servants",
            "Activists (political activist, human rights, etc)": "Other",
            "Organized crime": "Organized crime",
            "Students": "Students",
            "Land Worker": "Professionals",
            "Other": "Other",
            "No information": "No information",
        },
        "original_short_labels": {
            "Professionals (Entrepreneur, Engineer, Professor, Journalist, etc)": "Professionals",
            "People that work in service industries (taxi driver, salesman, etc)": "Service workers",
            "Civil servants (Police, mayor, public worker, etc)": "Civil servants",
            "Belonging to some sexual identity group (LGBTQ)": "LGBTQ",
            "People associated with politics": "Politics",
            "Activists (political activist, human rights, etc)": "Activists",
            "Organized crime": "Organized crime",
            "Students": "Students",
            "Land Worker": "Land workers",
            "Other": "Other",
            "No information": "No information",
        },
    },
    "captura_tipo": {
        "original_options": [
            "Places related to the victim (house, workplace, private property)",
            "Economic, social, industrial, agricultural and service centers",
            "Authorities (government offices, military facilities)",
            "Educational and medical facilities",
            "Places for free expression, association and gatherings",
            "Unoccupied or barren public spaces",
            "Means and routes of transport and places of connection",
            "International and protected spaces",
            "Special centers and barracks for detention",
            "No information",
        ],
        "merged_options": [
            "Places related to the victim",
            "Public and institutional spaces",
            "No information",
        ],
        "category_merging": {
            "Places related to the victim (house, workplace, private property)": "Places related to the victim",
            "Economic, social, industrial, agricultural and service centers": "Public and institutional spaces",
            "Authorities (government offices, military facilities)": "Public and institutional spaces",
            "Educational and medical facilities": "Public and institutional spaces",
            "Places for free expression, association and gatherings": "Public and institutional spaces",
            "Unoccupied or barren public spaces": "Public and institutional spaces",
            "Means and routes of transport and places of connection": "Public and institutional spaces",
            "International and protected spaces": "Public and institutional spaces",
            "Special centers and barracks for detention": "Public and institutional spaces",
            "No information": "No information",
        },
        "original_short_labels": {
            "Places related to the victim (house, workplace, private property)": "Victim-related places",
            "Economic, social, industrial, agricultural and service centers": "Economic / social centers",
            "Authorities (government offices, military facilities)": "Authorities",
            "Educational and medical facilities": "Educational / medical",
            "Places for free expression, association and gatherings": "Expression / gatherings",
            "Unoccupied or barren public spaces": "Open public spaces",
            "Means and routes of transport and places of connection": "Transport / connection",
            "International and protected spaces": "International / protected",
            "Special centers and barracks for detention": "Detention centers",
            "No information": "No information",
        },
    },
}

PLOT_LABELS = list(LABEL_CONFIGS.keys())

VICTIM_COLOR = "#5470c6"
REPORT_COLOR = "#91cc75"
BEFORE_COLOR = "#5470c6"
AFTER_COLOR = "#91cc75"
STATS_BOX = dict(boxstyle="round", facecolor="white", alpha=0.8)

def normalize_categorical_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return value


def consensus(series: pd.Series) -> Any:
    values = series.map(normalize_categorical_value).dropna()
    return values.mode().iloc[0] if len(values) and len(values.mode()) else None


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def numeric_stats_text(series: pd.Series) -> str:
    data = series.dropna()
    distinct_count = int(data.nunique())
    return (
        f"n = {len(data):,}\n"
        f"Missing: {series.isna().sum():,} ({percentage(series.isna().sum(), len(series)):.0f}%)\n"
        f"Distinct: {distinct_count:,} ({percentage(distinct_count, len(data)):.0f}%)\n"
        f"Min: {data.min():,.0f}\n"
        f"Max: {data.max():,.0f}"
    )


def categorical_stats_text(series: pd.Series) -> str:
    data = series.dropna()
    distinct_count = int(data.nunique())
    if data.empty:
        top_line = "Top: n/a"
    else:
        top_counts = data.value_counts()
        top_line = f"Top: {top_counts.index[0]} ({top_counts.iloc[0]:,})"

    return (
        f"n = {len(data):,}\n"
        f"Missing: {series.isna().sum():,} ({percentage(series.isna().sum(), len(series)):.0f}%)\n"
        f"Distinct: {distinct_count:,} ({percentage(distinct_count, len(data)):.0f}%)\n"
        f"{top_line}"
    )


def make_histogram(
    series: pd.Series,
    output_path: Path,
    title: str,
    x_label: str,
    color: str,
    discrete: bool = False,
) -> None:
    data = series.dropna()
    if data.empty:
        raise ValueError(f"No data available for plot: {output_path.name}")

    if discrete:
        bins = range(int(data.min()), int(data.max()) + 2)
    else:
        bins = 30

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.8)

    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:,.0f}",
    )
    ax.axvline(
        median_val,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"Median: {median_val:,.0f}",
    )

    ax.set_ylabel("Count")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.text(
        0.97,
        0.97,
        numeric_stats_text(series),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=STATS_BOX,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def wrap_tick_labels(labels: list[str], width: int = 28) -> list[str]:
    return [fill(str(label), width=width) for label in labels]


def get_short_labels(
    label: str,
    categories: list[str],
    label_configs: dict[str, dict[str, Any]],
) -> list[str]:
    short_labels = label_configs[label].get("original_short_labels", {})
    return [short_labels.get(category, category) for category in categories]


def ordered_categories(
    series: pd.Series,
    preferred_order: list[str] | None = None,
) -> list[str]:
    counts = series.value_counts()
    if preferred_order is None:
        return list(counts.index)

    ordered = [category for category in preferred_order if category in counts.index]
    extras = [category for category in counts.index if category not in ordered]
    return ordered + extras


def make_category_distribution(
    series: pd.Series,
    output_path: Path,
    title: str,
    x_label: str,
    color: str,
    preferred_order: list[str] | None = None,
) -> None:
    data = series.dropna()
    if data.empty:
        raise ValueError(f"No data available for plot: {output_path.name}")

    order = ordered_categories(data, preferred_order)
    counts = data.value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        range(len(counts)),
        counts.values,
        color=color,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(wrap_tick_labels(list(counts.index)), rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel(x_label)
    ax.set_title(title)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.text(
        0.97,
        0.97,
        categorical_stats_text(series),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=STATS_BOX,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_before_category_distribution_subplots(
    victim_labels: pd.DataFrame,
    label_configs: dict[str, dict[str, Any]],
    output_path: Path,
    color: str,
) -> None:
    labels = list(label_configs.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(18, 5.5))

    for index, (ax, label) in enumerate(zip(axes, labels)):
        series = victim_labels[label]
        data = series.dropna()
        if data.empty:
            raise ValueError(f"No data available for plot: {output_path.name} ({label})")

        order = ordered_categories(data, make_before_order(label, label_configs))
        counts = data.value_counts().reindex(order)
        tick_labels = wrap_tick_labels(
            get_short_labels(label, list(counts.index), label_configs),
            width=16,
        )

        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color=color,
            edgecolor="black",
            alpha=0.8,
        )

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_title(label)
        if index == 0:
            ax.set_ylabel("Count")

        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.text(
            0.97,
            0.97,
            categorical_stats_text(series),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=STATS_BOX,
        )

    fig.suptitle("Distribution of selected labels (Victim-level, Before merging)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_inputs(
    df: pd.DataFrame,
    label_configs: dict[str, dict[str, Any]],
) -> None:
    required_columns = {"victim", "text_len", *label_configs.keys()}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Input DataFrame is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    errors: list[str] = []

    for label, config in label_configs.items():
        original_options = set(config["original_options"])
        merged_options = set(config["merged_options"])
        mapping = config["category_merging"]

        invalid_targets = sorted(set(mapping.values()) - merged_options)
        if invalid_targets:
            errors.append(
                f"{label}: mapping targets not present in merged_options: {invalid_targets}"
            )

        uncovered_originals = sorted(
            value
            for value in original_options
            if value not in merged_options and value not in mapping
        )
        if uncovered_originals:
            errors.append(
                f"{label}: original options missing from merged options and mapping: "
                f"{uncovered_originals}"
            )

        observed_values = {
            value
            for value in df[label].map(normalize_categorical_value).dropna().unique()
        }
        uncovered_values = sorted(
            value
            for value in observed_values
            if value not in merged_options and value not in mapping
        )
        if uncovered_values:
            errors.append(
                f"{label}: observed values missing from reduced options and mapping: "
                f"{uncovered_values}"
            )

    if errors:
        raise ValueError("Taxonomy validation failed:\n- " + "\n- ".join(errors))


def merge_category(
    label: str,
    value: Any,
    label_configs: dict[str, dict[str, Any]],
) -> Any:
    value = normalize_categorical_value(value)
    if value is None:
        return None

    merged_options = set(label_configs[label]["merged_options"])
    mapping = label_configs[label]["category_merging"]

    if value in merged_options:
        return value
    if value in mapping:
        return mapping[value]

    raise ValueError(f"Unmapped value for {label}: {value!r}")


def build_victim_level_labels(
    df: pd.DataFrame,
    label_configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    return df.groupby("victim", as_index=False)[list(label_configs.keys())].agg(consensus)


def make_before_order(
    label: str,
    label_configs: dict[str, dict[str, Any]],
) -> list[str]:
    ordered = list(label_configs[label]["original_options"])
    for option in label_configs[label]["merged_options"]:
        if option not in ordered:
            ordered.append(option)
    return ordered


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    label_configs = LABEL_CONFIGS

    validate_inputs(df, label_configs)

    victim_labels = build_victim_level_labels(df, label_configs)

    make_before_category_distribution_subplots(
        victim_labels=victim_labels,
        label_configs=label_configs,
        output_path=OUTPUT_DIR
        / f"selected_labels_distribution_by_victim_before_merging{OUTPUT_SUFFIX}.png",
        color=BEFORE_COLOR,
    )

    print(f"Saved descriptive statistics plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
