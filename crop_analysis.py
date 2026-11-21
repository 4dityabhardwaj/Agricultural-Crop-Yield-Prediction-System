"""
crop_analysis.py - pulls the key facts, tables, graphs and a baseline model out of
the cleaned crop yield CSV, exactly as described in the SRS.

Put it in the same folder as crop_yield_clean.csv and run:
    python -m pip install pandas numpy matplotlib scikit-learn
    python crop_analysis.py

Everything is written to a folder called 'analysis_output' next to this script:
    analysis_output/tables/   -> CSV files with the numbers
    analysis_output/figures/  -> PNG graphs
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save graphs to files (no pop-up windows)
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG, TAB = BASE / "analysis_output" / "figures", BASE / "analysis_output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

EXPECTED = ["Crop", "Crop_Year", "Season", "State", "Area", "Production",
            "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
CATEGORICAL = ["Crop", "Season", "State"]
NUMERIC = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
GREEN, AMBER, BLUE, GREY = "#1B5E20", "#E08E0B", "#1F6F8B", "#9CA3AF"

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure saved: {name}.png")


# ---------------------------------------------------------------- 1. load
def load():
    path = BASE / "crop_yield_clean.csv"
    if not path.exists():
        found = list(BASE.glob("*clean*.csv"))
        if not found:
            raise SystemExit("crop_yield_clean.csv not found. Put it next to this script.")
        path = found[0]
    raw = pd.read_csv(path)
    df = raw.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    print(f"Loaded {path.name}: {len(df):,} rows x {df.shape[1]} columns")
    return raw, df


# ---------------------------------------------------------------- 2. validation report
def validate(raw, df):
    last = df["Crop_Year"].max()
    pos = df[(df["Area"] > 0) & (df["Production"] > 0)]
    rel = (pos["Yield"] - pos["Production"] / pos["Area"]).abs() / (pos["Production"] / pos["Area"])
    rows = [
        ("Expected columns all present", set(EXPECTED) <= set(df.columns)),
        ("Rows", len(df)), ("Columns", df.shape[1]),
        ("Missing cells", int(df.isna().sum().sum())),
        ("Exact duplicate rows", int(df.duplicated().sum())),
        ("Duplicate Crop+Year+Season+State keys", int(df.duplicated(["Crop", "Crop_Year", "Season", "State"]).sum())),
        ("Crops", df["Crop"].nunique()), ("Seasons", df["Season"].nunique()), ("States", df["State"].nunique()),
        ("First year", int(df["Crop_Year"].min())), ("Last year", int(last)),
        (f"Records in {last}", int((df["Crop_Year"] == last).sum())),
        (f"Records in {last - 1}", int((df["Crop_Year"] == last - 1).sum())),
        ("Uttar Pradesh records", int((df["State"] == "Uttar Pradesh").sum())),
        ("Zero Production rows", int((df["Production"] == 0).sum())),
        ("Zero Yield rows", int((df["Yield"] == 0).sum())),
        ("Rows with both Production and Yield zero", int(((df["Production"] == 0) & (df["Yield"] == 0)).sum())),
        ("Category values with extra spaces (before trimming)", int(sum((raw[c].astype(str) != raw[c].astype(str).str.strip()).sum() for c in CATEGORICAL))),
        ("Yield vs Production/Area: correlation", round(float(np.corrcoef(pos["Yield"], pos["Production"] / pos["Area"])[0, 1]), 4)),
        ("Yield vs Production/Area: median relative difference", round(float(rel.median()), 4)),
        ("Highest Yield (crop, state, year)", "{} / {} / {}".format(*df.loc[df["Yield"].idxmax(), ["Crop", "State", "Crop_Year"]])),
    ]
    rep = pd.DataFrame(rows, columns=["Item", "Value"])
    rep.to_csv(TAB / "validation_report.csv", index=False)
    print("\n=== VALIDATION REPORT ===")
    print(rep.to_string(index=False))


# ---------------------------------------------------------------- 3. summary tables
def tables(df):
    df[NUMERIC].describe().T.round(2).to_csv(TAB / "numeric_summary.csv")
    df["Crop_Year"].value_counts().sort_index().rename("records").to_csv(TAB / "records_by_year.csv")
    (df.groupby("Crop")["Yield"].agg(records="count", median="median", mean="mean", maximum="max")
       .round(2).sort_values("records", ascending=False).to_csv(TAB / "crop_summary.csv"))
    (df.groupby("State")["Yield"].agg(records="count", median="median", mean="mean")
       .round(2).sort_values("records", ascending=False).to_csv(TAB / "state_summary.csv"))
    df.groupby("Season")["Yield"].agg(records="count", median="median", mean="mean").round(2).to_csv(TAB / "season_summary.csv")
    df[NUMERIC + ["Crop_Year"]].corr(method="spearman").round(3).to_csv(TAB / "correlation_spearman.csv")
    print("\nTables saved in:", TAB)


# ---------------------------------------------------------------- 4. graphs
def barh(series, title, xlabel, name, color=GREEN, log=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    series.iloc[::-1].plot(kind="barh", ax=ax, color=color)
    if log:
        ax.set_xscale("log")
    ax.bar_label(ax.containers[0], fmt="%.4g", padding=3, fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel="")
    save(fig, name)


def figures(df):
    print("\nDrawing figures...")
    # 1 records per year (the sparse last year is highlighted)
    y = df["Crop_Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(y.index, y.values, color=[AMBER if i == y.index.max() else GREEN for i in y.index])
    ax.set(title="Records per year (sparse final year highlighted)", xlabel="Crop year", ylabel="Records")
    save(fig, "fig01_records_per_year")
    # 2-4 category counts
    barh(df["Crop"].value_counts().head(15), "15 most frequent crops", "Records", "fig02_top_crops")
    s = df["Season"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index, s.values, color=BLUE)
    ax.bar_label(ax.containers[0], padding=2, fontsize=9)
    ax.set(title="Records by season", ylabel="Records")
    save(fig, "fig03_records_by_season")
    barh(df["State"].value_counts().head(15), "15 states with the most records", "Records", "fig04_top_states", BLUE)
    # 5 Yield distribution
    pos = df.loc[df["Yield"] > 0, "Yield"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pos, bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 40), color=GREEN, edgecolor="white")
    ax.set_xscale("log")
    ax.set(title="Distribution of Yield (log scale, zero values excluded)", xlabel="Yield (unit unspecified)", ylabel="Records")
    save(fig, "fig05_yield_distribution")
    # 6 median yield by crop
    med = df.groupby("Crop")["Yield"].median().sort_values(ascending=False).head(15)
    barh(med.round(2), "Median Yield by crop - 15 highest (log scale)", "Median Yield (unit unspecified)",
         "fig06_median_yield_by_crop", AMBER, log=True)
    # 7 correlation heatmap
    cols = ["Crop_Year"] + NUMERIC
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Rank correlation between numeric fields")
    save(fig, "fig07_correlation_heatmap")
    # 8 yield trend for the 4 most frequent crops (final sparse year left out)
    top4 = df["Crop"].value_counts().head(4).index
    d = df[(df["Crop"].isin(top4)) & (df["Crop_Year"] < df["Crop_Year"].max())]
    t = d.groupby(["Crop_Year", "Crop"])["Yield"].median().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t.plot(ax=ax, marker="o", markersize=3)
    ax.set(title="Median Yield per year - 4 most frequent crops", xlabel="Crop year", ylabel="Median Yield")
    save(fig, "fig08_yield_trend")
    # 9 rainfall vs yield for the most frequent crop
    crop = df["Crop"].value_counts().index[0]
    r = df[df["Crop"] == crop]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(r["Annual_Rainfall"], r["Yield"], s=10, alpha=0.4, color=BLUE)
    m, b = np.polyfit(r["Annual_Rainfall"], r["Yield"], 1)
    xs = np.linspace(r["Annual_Rainfall"].min(), r["Annual_Rainfall"].max(), 50)
    ax.plot(xs, m * xs + b, "--", color="black", label=f"Trend (r = {r['Annual_Rainfall'].corr(r['Yield']):.2f})")
    ax.legend()
    ax.set(title=f"Annual rainfall vs Yield - {crop}", xlabel="Annual rainfall (unit unspecified)", ylabel="Yield")
    save(fig, "fig09_rainfall_vs_yield")
    # 10 evidence that Yield is tied to Production / Area (leakage check)
    p = df[(df["Area"] > 0) & (df["Production"] > 0) & (df["Yield"] > 0)]
    ratio = p["Production"] / p["Area"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(ratio, p["Yield"], s=6, alpha=0.3, color=GREEN)
    lim = [min(ratio.min(), p["Yield"].min()), max(ratio.max(), p["Yield"].max())]
    ax.plot(lim, lim, "--", color=AMBER, label="Yield = Production / Area")
    ax.set(xscale="log", yscale="log", title="Yield compared with Production / Area",
           xlabel="Production / Area", ylabel="Recorded Yield")
    ax.legend()
    save(fig, "fig10_yield_vs_production_over_area")


# ---------------------------------------------------------------- 5. baseline model
def baseline(df):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        print("\nscikit-learn is not installed - skipping the model. Run: python -m pip install scikit-learn")
        return
    # Production is NOT a predictor (leakage control, SRS FR-05); Area is held out until reviewed.
    cat, num = ["Crop", "Season", "State"], ["Crop_Year", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    X, y = df[cat + num], df["Yield"]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])

    def run(name, Xtr, Xte, ytr, yte):
        out = {}
        for label, est in (("Mean baseline", DummyRegressor()), ("Linear regression", LinearRegression())):
            model = Pipeline([("prep", prep), ("model", est)]).fit(Xtr, ytr)
            pred = model.predict(Xte)
            out[label] = pred
            rows.append({"Split": name, "Model": label, "Train rows": len(Xtr), "Test rows": len(Xte),
                         "MAE": round(mean_absolute_error(yte, pred), 2),
                         "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 2),
                         "R2": round(r2_score(yte, pred), 3)})
        return out

    rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = run("Random 80/20 (seed 42)", Xtr, Xte, ytr, yte)
    cut = 2015  # chronological split: train up to 2015, test 2016 onwards
    tr = df["Crop_Year"] <= cut
    run(f"Chronological (train <= {cut})", X[tr], X[~tr], y[tr], y[~tr])
    a, b = Xtr["Crop"] != "Coconut", Xte["Crop"] != "Coconut"  # Coconut's huge scale dominates pooled scores
    preds_ex = run("Random 80/20, Coconut excluded", Xtr[a], Xte[b], ytr[a], yte[b])
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "model_metrics.csv", index=False)
    print("\n=== BASELINE MODEL RESULTS (illustrative, units unspecified) ===")
    print(res.to_string(index=False))

    # per-crop MAE on the random split (pooled model vs model trained without Coconut vs mean baseline)
    t = Xte.assign(actual=yte.values, lin=preds["Linear regression"], base=preds["Mean baseline"], lin_ex=np.nan)
    t.loc[b.values, "lin_ex"] = preds_ex["Linear regression"]
    g = t.groupby("Crop").apply(lambda d: pd.Series({
        "Test rows": len(d), "MAE linear (pooled)": (d.actual - d.lin).abs().mean(),
        "MAE linear (no Coconut)": (d.actual - d.lin_ex).abs().mean(),
        "MAE mean baseline": (d.actual - d.base).abs().mean()}), include_groups=False)
    g = g[g["Test rows"] >= 30].sort_values("Test rows", ascending=False).round(2)
    g.to_csv(TAB / "model_mae_by_crop.csv")
    top = g.head(10)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top.drop(columns="Test rows").iloc[::-1].plot(kind="barh", ax=ax, color=[GREEN, BLUE, GREY], width=0.8)
    ax.set(title="Held-out MAE by crop - 10 crops with most test rows", xlabel="MAE (Yield units)", ylabel="")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    save(fig, "fig12_model_mae_by_crop")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        pv = res.pivot(index="Split", columns="Model", values=metric).loc[res["Split"].unique()]
        pv.index = [textwrap.fill(i, 18) for i in pv.index]
        pv[["Mean baseline", "Linear regression"]].plot(kind="bar", ax=ax, color=[GREY, GREEN], width=0.75, rot=0)
        ax.set(title=f"{metric} on held-out records", xlabel="", ylabel=f"{metric} (Yield units)")
        ax.legend(fontsize=8)
    save(fig, "fig11_model_metrics_comparison")


if __name__ == "__main__":
    raw, df = load()
    validate(raw, df)
    tables(df)
    figures(df)
    baseline(df)
    print("\nDone. Open the 'analysis_output' folder to see the tables and figures.")
