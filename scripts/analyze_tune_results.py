
import os, argparse, matplotlib.pyplot as plt, pandas as pd
from ray import tune

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True,
                   help="Path to runs/detect/tune-YYYY-MM-DD-HH-MM")
    return p.parse_args()

def main():
    args = cli()
    print("Loading results from", args.exp)
    tuner   = tune.Tuner.restore(args.exp)
    results = tuner.get_results()

    if results.errors:
        print("⚠️  One or more trials had errors:")
        for r in results:
            if r.error:
                print(" •", r.error)
    else:
        print("✅ All trials finished successfully.")

    best = results.get_best_result(metric="metrics/mAP50-95(B)", mode="max")
    print("\nBest trial config:\n", best.config)
    print("Best trial fitness:",
          best.metrics["metrics/mAP50-95(B)"])

    ax = None
    for r in results:
        df: pd.DataFrame = r.metrics_dataframe
        label = f"trial {r.path.split(os.sep)[-1]}"
        if ax is None:
            ax = df.plot("training_iteration",
                         "metrics/mAP50-95(B)", label=label)
        else:
            df.plot("training_iteration",
                    "metrics/mAP50-95(B)", ax=ax, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Ray Tune trials")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
