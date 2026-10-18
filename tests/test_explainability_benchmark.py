from argparse import Namespace

from scripts.benchmark_explainability import _summarize, _write_report


def test_report_includes_seed_specific_engine_correlations(tmp_path):
    rows = []
    for model_seed in (42, 43):
        for engine_id in range(4):
            row = {
                "model_seed": model_seed,
                "engine_id": engine_id,
                "eri": float(engine_id + model_seed % 2),
                "agreement_ig__shap_rank_agreement_top_k": float(engine_id / 4),
                "agreement_ig__shap_sign_agreement_top_k": 1.0,
                "agreement_ig__shap_rank_correlation_spearman": float(engine_id / 4),
            }
            for explainer in ("ig", "shap", "lime"):
                row[f"{explainer}_prediction_gap_top_50"] = float(engine_id + 1)
                row[f"{explainer}_deletion_gap_auc"] = float(engine_id / 4)
                row[f"{explainer}_insertion_auc"] = float(engine_id / 4)
            rows.append(row)

    summary = _summarize(rows)
    report_path = tmp_path / "report.md"
    _write_report(
        report_path,
        rows,
        summary,
        Namespace(
            subset="FD001",
            model="gru",
            target="rul",
            use_default_checkpoint=False,
            seeds=[42, 43],
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "engine-mean-seed-42" in report
    assert "engine-mean-seed-43" in report
