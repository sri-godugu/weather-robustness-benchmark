import argparse
import pandas as pd
from plotting import plot_metric_vs_severity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/robustness.csv")
    ap.add_argument("--outdir", type=str, default="results/plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plot_metric_vs_severity(df, "accuracy", f"{args.outdir}/accuracy.png")
    plot_metric_vs_severity(df, "mean_confidence", f"{args.outdir}/confidence.png")
    plot_metric_vs_severity(df, "mean_entropy", f"{args.outdir}/entropy.png")

if __name__ == "__main__":
    main()
