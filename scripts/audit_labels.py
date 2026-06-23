"""C2: Label Quality Audit — analyze model predictions vs. database labels.

Loads prediction sets from B4 (SetFit) and B5 (SciBERT) on real TACC data,
computes agreement statistics, and generates diagnostic charts to understand
label noise in the database.

Usage:
    python scripts/audit_labels.py [--output-dir output/audit]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_predictions(path: Path):
    """Load a PredictionSet from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data


def find_prediction_files(predictions_dir: Path) -> dict[str, Path]:
    """Find the latest B4 and B5 prediction files for real TACC."""
    files = {}
    for p in sorted(predictions_dir.glob("predictions_*real_tacc.json")):
        name = p.stem
        if "setfit" in name:
            files["setfit"] = p
        elif "finetune" in name:
            files["finetune"] = p
    return files


def compute_agreement_stats(preds_b4: list[dict], preds_b5: list[dict]) -> dict:
    """Compute agreement statistics between two prediction sets."""
    assert len(preds_b4) == len(preds_b5), (
        f"Prediction sets have different sizes: {len(preds_b4)} vs {len(preds_b5)}"
    )

    n = len(preds_b4)
    agree_major = 0
    agree_broad = 0
    agree_and_match_db = 0
    agree_and_disagree_db = 0
    disagree_models = 0

    # Per-field analysis
    field_stats = defaultdict(lambda: {
        "n": 0, "b4_matches_db": 0, "b5_matches_db": 0,
        "models_agree": 0, "agree_match_db": 0, "agree_disagree_db": 0,
        "b4_predictions": Counter(), "b5_predictions": Counter(),
        "confidences_agree_db": [], "confidences_disagree_db": [],
    })

    agreement_by_confidence = []

    for p4, p5 in zip(preds_b4, preds_b5):
        db_major = p4["true_major_field"]
        pred4 = p4["predicted_major_field"]
        pred5 = p5["predicted_major_field"]
        conf4 = p4.get("confidence", 0) or 0
        conf5 = p5.get("confidence", 0) or 0
        max_conf = max(conf4, conf5)

        models_agree = pred4 == pred5

        if models_agree:
            agree_major += 1
            if pred4 == db_major:
                agree_and_match_db += 1
            else:
                agree_and_disagree_db += 1
        else:
            disagree_models += 1

        # Broad-level agreement
        broad4 = p4["predicted_broad_field"]
        broad5 = p5["predicted_broad_field"]
        if broad4 == broad5:
            agree_broad += 1

        # Per-field stats
        fs = field_stats[db_major]
        fs["n"] += 1
        fs["b4_predictions"][pred4] += 1
        fs["b5_predictions"][pred5] += 1
        if pred4 == db_major:
            fs["b4_matches_db"] += 1
        if pred5 == db_major:
            fs["b5_matches_db"] += 1
        if models_agree:
            fs["models_agree"] += 1
            if pred4 == db_major:
                fs["agree_match_db"] += 1
                fs["confidences_agree_db"].append(max_conf)
            else:
                fs["agree_disagree_db"] += 1
                fs["confidences_disagree_db"].append(max_conf)

        agreement_by_confidence.append({
            "db_label": db_major,
            "pred_b4": pred4,
            "pred_b5": pred5,
            "conf_b4": conf4,
            "conf_b5": conf5,
            "max_conf": max_conf,
            "models_agree": models_agree,
            "matches_db": pred4 == db_major if models_agree else None,
        })

    return {
        "n": n,
        "agree_major": agree_major,
        "agree_major_pct": agree_major / n,
        "agree_broad": agree_broad,
        "agree_broad_pct": agree_broad / n,
        "agree_and_match_db": agree_and_match_db,
        "agree_and_disagree_db": agree_and_disagree_db,
        "disagree_models": disagree_models,
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "agreement_by_confidence": agreement_by_confidence,
    }


def plot_confidence_vs_agreement(stats: dict, output_dir: Path):
    """Plot: at each confidence threshold, what fraction of agreed predictions match DB?"""
    import matplotlib.pyplot as plt

    records = [r for r in stats["agreement_by_confidence"] if r["models_agree"]]
    if not records:
        print("No agreed predictions to plot.")
        return

    thresholds = np.arange(0.0, 1.01, 0.05)
    n_above = []
    pct_match_db = []

    for thresh in thresholds:
        above = [r for r in records if r["max_conf"] >= thresh]
        n_above.append(len(above))
        if above:
            matches = sum(1 for r in above if r["matches_db"])
            pct_match_db.append(matches / len(above))
        else:
            pct_match_db.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.plot(thresholds, pct_match_db, color=color1, marker="o", markersize=4, label="Agrees with DB label")
    ax1.set_xlabel("Confidence Threshold (min of max(B4, B5) confidence)")
    ax1.set_ylabel("Fraction agreeing with DB label", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    ax2.plot(thresholds, n_above, color=color2, marker="s", markersize=4, label="N predictions")
    ax2.set_ylabel("Number of predictions above threshold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Model Agreement vs. DB Label Match by Confidence Threshold\n"
                  "(When B4 and B5 agree, how often does their prediction match the DB label?)")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_vs_db_agreement.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'confidence_vs_db_agreement.png'}")


def plot_agreement_rate_by_field(stats: dict, output_dir: Path):
    """Bar chart: model agreement rate per DB-labeled field, sorted."""
    import matplotlib.pyplot as plt

    field_stats = stats["field_stats"]
    fields = []
    agree_rates = []
    supports = []

    for field, fs in sorted(field_stats.items(), key=lambda x: x[1]["n"], reverse=True):
        if fs["n"] < 10:
            continue
        fields.append(field)
        agree_rates.append(fs["models_agree"] / fs["n"])
        supports.append(fs["n"])

    # Sort by agreement rate
    order = np.argsort(agree_rates)
    fields = [fields[i] for i in order]
    agree_rates = [agree_rates[i] for i in order]
    supports = [supports[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(8, len(fields) * 0.35)))
    bars = ax.barh(range(len(fields)), agree_rates, color="steelblue")

    # Annotate with support
    for i, (rate, n) in enumerate(zip(agree_rates, supports)):
        ax.text(rate + 0.01, i, f"n={n}", va="center", fontsize=8)

    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=8)
    ax.set_xlabel("B4/B5 Agreement Rate")
    ax.set_title("Inter-Model Agreement Rate by Field (DB Label)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.set_xlim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "agreement_rate_by_field.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'agreement_rate_by_field.png'}")


def plot_problem_field_predictions(stats: dict, output_dir: Path, min_support: int = 100):
    """For fields with 0% F1 or low agreement, show what models predict instead."""
    import matplotlib.pyplot as plt

    field_stats = stats["field_stats"]

    # Identify problem fields: models agree with each other but disagree with DB > 50% of the time
    problem_fields = []
    for field, fs in field_stats.items():
        if fs["n"] < min_support:
            continue
        if fs["models_agree"] > 0:
            agree_disagree_rate = fs["agree_disagree_db"] / fs["models_agree"]
            if agree_disagree_rate > 0.5:
                problem_fields.append((field, fs))

    if not problem_fields:
        print("  No problem fields found (all have >50% agreement with DB when models agree)")
        return

    # Sort by support descending
    problem_fields.sort(key=lambda x: x[1]["n"], reverse=True)

    for field, fs in problem_fields[:6]:  # Top 6 problem fields
        # Combine B4 and B5 predictions
        combined = Counter()
        for pred, count in fs["b4_predictions"].items():
            combined[pred] += count
        for pred, count in fs["b5_predictions"].items():
            combined[pred] += count

        # Top predictions (excluding the DB label itself for clarity)
        top = combined.most_common(8)
        labels = [t[0][:30] for t in top]
        values = [t[1] for t in top]
        colors = ["green" if t[0] == field else "salmon" for t in top]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Total predictions (B4 + B5)")
        ax.set_title(f"What models predict for DB label: '{field}' (n={fs['n']})\n"
                     f"Green = matches DB label")
        fig.tight_layout()

        safe_name = field.replace("/", "_").replace(" ", "_")[:40]
        fig.savefig(output_dir / f"problem_field_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: problem_field_{safe_name}.png")


def plot_confidence_distribution_by_agreement(stats: dict, output_dir: Path):
    """Histogram of confidence scores split by agrees/disagrees with DB label."""
    import matplotlib.pyplot as plt

    records = stats["agreement_by_confidence"]
    agreed = [r for r in records if r["models_agree"]]

    conf_match = [r["max_conf"] for r in agreed if r["matches_db"]]
    conf_disagree = [r["max_conf"] for r in agreed if not r["matches_db"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 1.05, 0.05)

    if conf_match:
        ax.hist(conf_match, bins=bins, alpha=0.6, label=f"Agrees with DB (n={len(conf_match)})", color="green")
    if conf_disagree:
        ax.hist(conf_disagree, bins=bins, alpha=0.6, label=f"Disagrees with DB (n={len(conf_disagree)})", color="red")

    ax.set_xlabel("Max Confidence (max of B4, B5)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: When Models Agree, Do They Match DB?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_distribution_agreement.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'confidence_distribution_agreement.png'}")


def generate_audit_report(stats: dict, output_dir: Path):
    """Generate a text summary of the audit findings."""
    lines = []
    lines.append("# Label Quality Audit Report\n")
    lines.append(f"Total predictions analyzed: {stats['n']}")
    lines.append(f"B4/B5 agree on major field: {stats['agree_major']} ({stats['agree_major_pct']:.1%})")
    lines.append(f"B4/B5 agree on broad field: {stats['agree_broad']} ({stats['agree_broad_pct']:.1%})")
    lines.append(f"When models agree, matches DB: {stats['agree_and_match_db']} "
                 f"({stats['agree_and_match_db'] / max(stats['agree_major'], 1):.1%})")
    lines.append(f"When models agree, disagrees with DB: {stats['agree_and_disagree_db']} "
                 f"({stats['agree_and_disagree_db'] / max(stats['agree_major'], 1):.1%})")
    lines.append("")

    # Problem fields table
    lines.append("## Problem Fields (models agree but disagree with DB >50% of the time)\n")
    lines.append("| Field | N | Models Agree | Agree+Match DB | Agree+Disagree DB | Likely Noise Rate |")
    lines.append("|-------|---|-------------|----------------|-------------------|-------------------|")

    field_stats = stats["field_stats"]
    problem_rows = []
    for field, fs in field_stats.items():
        if fs["n"] < 50:
            continue
        if fs["models_agree"] > 0:
            noise_rate = fs["agree_disagree_db"] / fs["models_agree"]
            if noise_rate > 0.3:
                problem_rows.append((field, fs, noise_rate))

    problem_rows.sort(key=lambda x: x[2], reverse=True)
    for field, fs, noise_rate in problem_rows:
        lines.append(f"| {field} | {fs['n']} | {fs['models_agree']} | "
                     f"{fs['agree_match_db']} | {fs['agree_disagree_db']} | {noise_rate:.1%} |")

    lines.append("")

    # Silver label potential
    agreed_records = [r for r in stats["agreement_by_confidence"] if r["models_agree"]]
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        above = [r for r in agreed_records if r["max_conf"] >= thresh]
        lines.append(f"Confidence ≥ {thresh}: {len(above)} records available for silver labels")

    report = "\n".join(lines)
    report_path = output_dir / "audit_report.md"
    report_path.write_text(report)
    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Label Quality Audit")
    parser.add_argument("--predictions-dir", type=Path, default=Path("output/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/audit"))
    parser.add_argument("--archive-dir", type=Path, default=Path("archive/output/predictions"),
                        help="Also search archive for prediction files")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find prediction files (check both active and archive)
    pred_files = find_prediction_files(args.predictions_dir)
    if len(pred_files) < 2 and args.archive_dir.exists():
        # Check archive too
        archive_files = find_prediction_files(args.archive_dir)
        for k, v in archive_files.items():
            if k not in pred_files:
                pred_files[k] = v

    if "setfit" not in pred_files:
        print("ERROR: No SetFit prediction file found for real TACC.")
        print(f"  Searched: {args.predictions_dir} and {args.archive_dir}")
        return
    if "finetune" not in pred_files:
        print("ERROR: No SciBERT finetune prediction file found for real TACC.")
        print(f"  Searched: {args.predictions_dir} and {args.archive_dir}")
        return

    print(f"B4 (SetFit): {pred_files['setfit']}")
    print(f"B5 (SciBERT): {pred_files['finetune']}")

    # Load predictions
    data_b4 = load_predictions(pred_files["setfit"])
    data_b5 = load_predictions(pred_files["finetune"])

    preds_b4 = data_b4["predictions"]
    preds_b5 = data_b5["predictions"]

    # Filter to records that have a true_major_field (exclude empty/unassigned)
    valid_indices = []
    for i in range(len(preds_b4)):
        label = preds_b4[i].get("true_major_field", "")
        if label and label != "UNASSIGNED":
            valid_indices.append(i)

    preds_b4_filtered = [preds_b4[i] for i in valid_indices]
    preds_b5_filtered = [preds_b5[i] for i in valid_indices]
    print(f"Filtered to {len(preds_b4_filtered)} records (excluded empty/UNASSIGNED)")

    # Compute agreement stats
    print("\nComputing agreement statistics...")
    stats = compute_agreement_stats(preds_b4_filtered, preds_b5_filtered)

    # Save raw stats (minus the large agreement_by_confidence list)
    stats_summary = {k: v for k, v in stats.items() if k != "agreement_by_confidence" and k != "field_stats"}
    stats_summary["n_fields"] = len(stats["field_stats"])
    with open(output_dir / "agreement_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    # Generate charts
    print("\nGenerating diagnostic charts...")
    plot_confidence_vs_agreement(stats, output_dir)
    plot_agreement_rate_by_field(stats, output_dir)
    plot_problem_field_predictions(stats, output_dir)
    plot_confidence_distribution_by_agreement(stats, output_dir)

    # Generate report
    print("\nGenerating audit report...")
    generate_audit_report(stats, output_dir)

    print(f"\nAudit complete. Results in: {output_dir}/")


if __name__ == "__main__":
    main()
