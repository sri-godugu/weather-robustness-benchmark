import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metric_vs_severity(df: pd.DataFrame, metric: str, outpath: str):
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    for corruption in sorted(df["corruption"].unique()):
        sub = df[df["corruption"] == corruption].groupby("severity")[metric].mean().reset_index()
        plt.figure()
        plt.plot(sub["severity"], sub[metric], marker="o")
        plt.xlabel("Severity (1-5)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Severity — {corruption}")
        plt.xticks([1,2,3,4,5])
        plt.grid(True)
        plt.savefig(out.parent / f"{metric}_vs_severity_{corruption}.png", bbox_inches="tight")
        plt.close()
