"""
Crop Yield Chart Maker - short and simple.
Put this file in the same folder as your CSV, then run:  python chart_studio.py
Pick a chart -> (optional) filter -> pick columns -> chart is shown and saved in 'graphs'.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
GRAPHS = BASE / "graphs"
GRAPHS.mkdir(exist_ok=True)
saved = []  # charts made this session


# ------------------------------------------------------------ helpers
def choose(title, options, fmt=str, skip=False):
    """Show a numbered list and return the chosen item (None if skipped)."""
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {fmt(o)}")
    if skip:
        print("  (press Enter to skip)")
    while True:
        raw = input("Number: ").strip()
        if raw == "" and skip:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Please type a number from the list.")


def numbers(df):
    return list(df.select_dtypes("number").columns)


def groups(df):
    return [c for c in df.columns if c not in numbers(df)]


def pick_num(df, title, exclude=()):
    cols = [c for c in numbers(df) if c not in exclude]
    return choose(title, cols, fmt=lambda c: f"{c}  (min {df[c].min():.4g}, max {df[c].max():.4g})")


def pick_group(df, title):
    return choose(title, groups(df), fmt=lambda c: f"{c}  ({df[c].nunique()} groups)")


def wide(s):
    """True if a column spans a huge range (a few giant values squash everything else)."""
    s = s[s > 0]
    return len(s) > 0 and s.max() / s.median() > 1000


def save_and_show(fig, name):
    fig.tight_layout()
    path = GRAPHS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    saved.append(path)
    print(f"\n✅ Saved: {path}")
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------ load data
def load():
    files = list(BASE.glob("*.csv")) or list(Path.cwd().glob("*.csv"))
    if not files:
        raise SystemExit("No CSV file found. Put this script in the same folder as your CSV.")
    cleaned = [p for p in files if "clean" in p.name.lower()]
    if len(cleaned) == 1:  # crop_yield_cleaned.csv wins over any messy original
        path = cleaned[0]
    else:
        path = files[0] if len(files) == 1 else choose("Which CSV file?", files, fmt=lambda p: p.name)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in groups(df):
        df[c] = df[c].astype(str).str.strip()
    print(f"\nLoaded {path.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def ask_filter(df):
    """One optional question: keep only one Crop / State / Season / etc."""
    cols = [c for c in groups(df) if df[c].nunique() <= 60]
    col = choose("Filter the data first? (optional)", cols, fmt=lambda c: f"{c}  ({df[c].nunique()} values)", skip=True)
    if col is None:
        return df
    value = choose(f"Which {col}?", sorted(df[col].dropna().unique()))
    print(f"  Using only {col} = {value}")
    return df[df[col] == value]


# ------------------------------------------------------------ charts
def bar(df):
    x = pick_group(df, "Which category goes on the X axis?")
    how = choose("What should each bar show?", ["Average", "Total", "Count"],
                 fmt=lambda h: {"Average": "Average  - typical value", "Total": "Total  - add everything up",
                                "Count": "Count  - number of records"}[h])
    if how == "Count":
        s, label = df.groupby(x).size(), "Number of records"
    else:
        y = pick_num(df, "Which number to measure?")
        s = df.groupby(x)[y].mean() if how == "Average" else df.groupby(x)[y].sum()
        label = f"{how} {y}"
    s = s.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    s.plot(kind="bar", ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=8, padding=2)
    plt.xticks(rotation=45, ha="right")
    ax.set(title=f"{label} by {x}", xlabel=x, ylabel=label)
    save_and_show(fig, "Bar")


def line(df):
    x = "Crop_Year" if "Crop_Year" in df.columns else pick_num(df, "Which column goes on the X axis?")
    y = pick_num(df, "Which number do you want to track?", exclude=[x])
    s = df.groupby(x)[y].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s.index, s.values, marker="o")
    ax.grid(True, alpha=0.3)
    ax.set(title=f"Average {y} by {x}", xlabel=x, ylabel=f"Average {y}")
    save_and_show(fig, "Line")


def scatter(df):
    x = pick_num(df, "Which number goes on the X axis?")
    y = pick_num(df, "Which number goes on the Y axis?", exclude=[x])
    fig, ax = plt.subplots(figsize=(9, 6))
    d, xl, yl = df, x, y
    if wide(df[x]):
        d, xl = d[d[x] > 0], f"{x} (log scale)"
        ax.set_xscale("log")
    if wide(df[y]):
        d, yl = d[d[y] > 0], f"{y} (log scale)"
        ax.set_yscale("log")
    if (xl, yl) != (x, y):
        print("  ℹ Values range hugely, so a log scale is used to keep the chart readable.")
    ax.scatter(d[x], d[y], alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set(title=f"{y} vs {x}", xlabel=xl, ylabel=yl)
    print(f"  Correlation: {df[x].corr(df[y]):.3f}")
    save_and_show(fig, "Scatter")


def histogram(df):
    x = pick_num(df, "Which number do you want to see the spread of?")
    fig, ax = plt.subplots(figsize=(9, 6))
    s, label = df[x].dropna(), x
    if wide(s):  # a few huge values would squash the chart, so use a log scale
        s = s[s > 0]
        ax.hist(s, bins=np.logspace(np.log10(s.min()), np.log10(s.max()), 30), edgecolor="black")
        ax.set_xscale("log")
        label = f"{x} (log scale)"
        print("  ℹ Values range hugely, so a log scale is used to keep the chart readable.")
    else:
        ax.hist(s, bins=30, edgecolor="black")
    ax.set(title=f"Distribution of {x}", xlabel=label, ylabel="Frequency")
    save_and_show(fig, "Histogram")


def box(df):
    g = pick_group(df, "Compare across which category?")
    y = pick_num(df, "Which number do you want to compare?")
    top = df.groupby(g)[y].mean().sort_values(ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([df.loc[df[g] == t, y].dropna() for t in top])
    ax.set_xticks(range(1, len(top) + 1), [str(t) for t in top], rotation=45, ha="right")
    ax.set(title=f"{y} by {g}", xlabel=g, ylabel=y)
    save_and_show(fig, "Box")


def heatmap(df):
    cols = numbers(df)
    if len(cols) < 2:
        return print("Need at least 2 numeric columns.")
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Heatmap")
    save_and_show(fig, "Heatmap")


MENU = [
    ("Bar chart", "compare groups (average / total / count)", bar),
    ("Line graph", "how a value changes over the years", line),
    ("Scatter plot", "relationship between two numbers", scatter),
    ("Histogram", "how the values of one number are spread", histogram),
    ("Box plot", "spread and outliers for each group", box),
    ("Heatmap", "which numbers move together", heatmap),
    ("Dataset info", "columns, types, missing values", None),
    ("Exit", "", None),
]


# ------------------------------------------------------------ main
def goodbye():
    print("\n" + "=" * 50)
    if saved:
        print(f"  🌾 {len(saved)} chart{'s' if len(saved) != 1 else ''} created and saved in:")
        print(f"     {GRAPHS}")
    else:
        print("  🌾 No charts this time - come back anytime!")
    print("  Happy analysing - may your yields always be high! 📈")
    print("=" * 50)


def main():
    df = load()
    while True:
        item = choose("=== WHAT DO YOU WANT TO MAKE? ===", MENU,
                      fmt=lambda m: f"{m[0]}" + (f"  - {m[1]}" if m[1] else ""))
        name, _, fn = item
        if name == "Exit":
            return goodbye()
        if name == "Dataset info":
            print(f"\n{df.shape[0]} rows x {df.shape[1]} columns\n")
            print(pd.DataFrame({"type": df.dtypes.astype(str), "missing": df.isnull().sum(),
                                "unique": df.nunique()}).to_string())
            print("\n", df.describe().round(2).T.to_string())
            continue
        data = ask_filter(df)
        if data.empty:
            print("No records match that filter.")
            continue
        try:
            fn(data)
        except Exception as e:
            print(f"  Could not make that chart: {e}")
            plt.close("all")


if __name__ == "__main__":
    main()
