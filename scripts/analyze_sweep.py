"""Analyze C4 sweep results: generate comparison tables, charts, and identify top configs.

Usage:
    python scripts/analyze_sweep.py [--results output/sweep/sweep_results.jsonl]
    python scripts/analyze_sweep.py --top 10   # Show top 10 only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_path: Path) -> list[dict]:
    """Load sweep results from JSONL file."""
    results = []
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "completed":
                results.append(r)
    return results


def print_leaderboard(results: list[dict], top_n: int = 50):
    """Print ranked leaderboard."""
    # Sort by major accuracy
    results.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)

    print(f"\n{'='*100}")
    print(f"{'RANK':<5} {'CONFIG':<55} {'MAJOR':>7} {'BROAD':>7} {'F1':>7} {'TOP3':>7} {'TOP5':>7} {'TIME':>6}")
    print(f"{'='*100}")

    for i, r in enumerate(results[:top_n]):
        cfg = r["config"]
        st = r["synthetic_test"]
        name = cfg["name"][:55]
        print(f"{i+1:<5} {name:<55} {st['major_accuracy']:>7.4f} {st['broad_accuracy']:>7.4f} "
              f"{st['macro_f1']:>7.4f} {st['top3_accuracy']:>7.4f} {st['top5_accuracy']:>7.4f} "
              f"{r['train_time_seconds']:>5.0f}s")

    print(f"\nTotal completed: {len(results)} / configs")


def print_grouped_analysis(results: list[dict]):
    """Analyze results grouped by hyperparameter."""
    results.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)

    # Group by model
    print(f"\n{'='*70}")
    print("BY MODEL (best per model)")
    print(f"{'='*70}")
    by_model = {}
    for r in results:
        model = r["config"]["model_name"]
        if model not in by_model:
            by_model[model] = r
    for model, r in by_model.items():
        st = r["synthetic_test"]
        print(f"  {model:<45} major={st['major_accuracy']:.4f} broad={st['broad_accuracy']:.4f} f1={st['macro_f1']:.4f}")

    # Group by learning rate (SciBERT only)
    print(f"\n{'='*70}")
    print("BY LEARNING RATE (SciBERT, averaged)")
    print(f"{'='*70}")
    by_lr = {}
    for r in results:
        if "scibert" in r["config"]["model_name"]:
            lr = r["config"]["lr"]
            by_lr.setdefault(lr, []).append(r["synthetic_test"]["major_accuracy"])
    for lr in sorted(by_lr.keys()):
        accs = by_lr[lr]
        print(f"  lr={lr:.0e}: mean={sum(accs)/len(accs):.4f}, "
              f"max={max(accs):.4f}, min={min(accs):.4f} (n={len(accs)})")

    # Group by epochs (SciBERT only)
    print(f"\n{'='*70}")
    print("BY EPOCHS (SciBERT, averaged)")
    print(f"{'='*70}")
    by_ep = {}
    for r in results:
        if "scibert" in r["config"]["model_name"]:
            ep = r["config"]["epochs"]
            by_ep.setdefault(ep, []).append(r["synthetic_test"]["major_accuracy"])
    for ep in sorted(by_ep.keys()):
        accs = by_ep[ep]
        print(f"  epochs={ep}: mean={sum(accs)/len(accs):.4f}, "
              f"max={max(accs):.4f}, min={min(accs):.4f} (n={len(accs)})")

    # Label smoothing effect
    print(f"\n{'='*70}")
    print("LABEL SMOOTHING EFFECT (SciBERT)")
    print(f"{'='*70}")
    by_ls = {}
    for r in results:
        if "scibert" in r["config"]["model_name"]:
            ls = r["config"]["label_smoothing"]
            by_ls.setdefault(ls, []).append(r["synthetic_test"]["major_accuracy"])
    for ls in sorted(by_ls.keys()):
        accs = by_ls[ls]
        print(f"  label_smoothing={ls:.2f}: mean={sum(accs)/len(accs):.4f}, "
              f"max={max(accs):.4f} (n={len(accs)})")


def generate_charts(results: list[dict], output_dir: Path):
    """Generate visualization charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping charts")
        return

    results.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall leaderboard bar chart (top 20)
    fig, ax = plt.subplots(figsize=(14, 8))
    top20 = results[:20]
    names = [r["config"]["name"][:40] for r in top20]
    majors = [r["synthetic_test"]["major_accuracy"] for r in top20]
    broads = [r["synthetic_test"]["broad_accuracy"] for r in top20]
    f1s = [r["synthetic_test"]["macro_f1"] for r in top20]

    x = range(len(names))
    width = 0.28
    ax.barh([i - width for i in x], majors, width, label="Major Acc", color="#2196F3")
    ax.barh([i for i in x], broads, width, label="Broad Acc", color="#4CAF50")
    ax.barh([i + width for i in x], f1s, width, label="Macro F1", color="#FF9800")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("C4 Sweep: Top 20 Configurations")
    ax.legend()
    ax.set_xlim(0.7, 1.0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(charts_dir / "leaderboard_top20.png", dpi=150)
    plt.close()
    print(f"  Saved: {charts_dir / 'leaderboard_top20.png'}")

    # 2. LR vs Epochs heatmap (SciBERT only)
    scibert = [r for r in results if "scibert" in r["config"]["model_name"]
               and r["config"]["label_smoothing"] == 0.0
               and r["config"]["scheduler_type"] == "linear"
               and r["config"]["gradient_accumulation_steps"] == 1
               and r["config"]["freeze_layers"] == 0]
    if scibert:
        lrs = sorted(set(r["config"]["lr"] for r in scibert))
        epochs = sorted(set(r["config"]["epochs"] for r in scibert))

        heatmap = [[None] * len(epochs) for _ in range(len(lrs))]
        for r in scibert:
            li = lrs.index(r["config"]["lr"])
            ei = epochs.index(r["config"]["epochs"])
            heatmap[li][ei] = r["synthetic_test"]["major_accuracy"]

        fig, ax = plt.subplots(figsize=(8, 6))
        import numpy as np
        data = np.array([[v if v is not None else 0 for v in row] for row in heatmap])
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.88, vmax=0.95)
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_title("SciBERT Major Accuracy: LR × Epochs")
        # Annotate cells
        for i in range(len(lrs)):
            for j in range(len(epochs)):
                if heatmap[i][j] is not None:
                    ax.text(j, i, f"{heatmap[i][j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(charts_dir / "lr_epochs_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: {charts_dir / 'lr_epochs_heatmap.png'}")

    # 3. Model comparison box plot
    by_model = {}
    for r in results:
        model = r["config"]["model_name"].split("/")[-1]
        by_model.setdefault(model, []).append(r["synthetic_test"]["major_accuracy"])

    if len(by_model) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(by_model.keys())
        data = [by_model[m] for m in labels]
        ax.boxplot(data, tick_labels=labels, vert=True)
        ax.set_ylabel("Major Accuracy")
        ax.set_title("Model Comparison (all configs)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(charts_dir / "model_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved: {charts_dir / 'model_comparison.png'}")

    # 4. Training time vs accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        model = r["config"]["model_name"].split("/")[-1]
        color = {"scibert_scivocab_uncased": "#2196F3",
                 "deberta-v3-base": "#4CAF50",
                 "deberta-v3-large": "#F44336",
                 "BiomedNLP-BiomedBERT-base-uncased-abstract": "#9C27B0"}.get(model, "#666666")
        ax.scatter(r["train_time_seconds"] / 60, r["synthetic_test"]["major_accuracy"],
                   c=color, s=50, alpha=0.7, label=model)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Major Accuracy")
    ax.set_title("Accuracy vs Training Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "time_vs_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {charts_dir / 'time_vs_accuracy.png'}")

    print(f"\n  All charts saved to: {charts_dir}/")


def save_summary_json(results: list[dict], output_dir: Path):
    """Save a structured summary for easy comparison."""
    results.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)

    summary = {
        "total_configs": len(results),
        "best_config": results[0]["config"] if results else None,
        "best_metrics": results[0]["synthetic_test"] if results else None,
        "top_10": [
            {
                "rank": i + 1,
                "name": r["config"]["name"],
                "model": r["config"]["model_name"],
                "lr": r["config"]["lr"],
                "epochs": r["config"]["epochs"],
                "label_smoothing": r["config"]["label_smoothing"],
                "scheduler": r["config"]["scheduler_type"],
                "major_accuracy": r["synthetic_test"]["major_accuracy"],
                "broad_accuracy": r["synthetic_test"]["broad_accuracy"],
                "macro_f1": r["synthetic_test"]["macro_f1"],
                "train_time_s": r["train_time_seconds"],
            }
            for i, r in enumerate(results[:10])
        ],
    }

    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze C4 sweep results")
    parser.add_argument("--results", type=str, default="output/sweep/sweep_results.jsonl",
                        help="Path to sweep results JSONL")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to show")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_path = project_root / args.results
    output_dir = results_path.parent

    if not results_path.exists():
        print(f"No results file found at {results_path}")
        print("Run the sweep first: python scripts/sweep_finetune.py")
        return

    results = load_results(results_path)
    if not results:
        print("No completed results found.")
        return

    print(f"Loaded {len(results)} completed results from {results_path}")

    print_leaderboard(results, top_n=args.top)
    print_grouped_analysis(results)
    save_summary_json(results, output_dir)

    if not args.no_charts:
        print("\nGenerating charts...")
        generate_charts(results, output_dir)


if __name__ == "__main__":
    main()
