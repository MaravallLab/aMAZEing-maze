"""CLI orchestrating the ordered validation pipeline. Refuses past a failed gate.

Order (Phase 1, day 1 only):
  cohort filter (log exclusions, report N) -> sanity -> model-free design
  analysis (incl. group) -> build features (+ collinearity) -> recovery
  (+ confusion). Then the day-2 secondary section (model-free + block
  time-course) into a clearly separated part. Phase 2 (PyMC LOO / posterior
  predictive / individual diffs) requires `pymc`+`arviz` and is not run here.

Usage:
    python -m model_validation.run_validation --results_dir <grammar/> --out_dir <out>
    python run_validation.py --results_dir <grammar/> --out_dir <out> --day both
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

# make the package importable whether run as a script or with -m
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from model_validation import config as cfgmod          # noqa: E402
from model_validation import data_loading as dl         # noqa: E402
from model_validation import latent_regressors as lr    # noqa: E402
from model_validation import sanity_checks as sc        # noqa: E402
from model_validation import design_analysis as da       # noqa: E402
from model_validation import recovery as rec             # noqa: E402
from model_validation import reporting as rp             # noqa: E402


def _verdict(results: dict) -> str:
    parts = []
    des = results.get("design", {})
    rec_r = results.get("recovery", {})
    # 1. model-free grammar (EE-SC) effect
    egs = des.get("environment_effect_by_group", [])
    if egs:
        signs = {r.get("group"): r.get("env_effect_EE_minus_SC") for r in egs}
        parts.append(f"Model-free EE−SC effect by group: {signs}; "
                     f"group-consistent sign = {des.get('group_consistent')}.")
    # 2. high-complexity simple effect
    for r in des.get("simple_effects_by_tier", []):
        if r.get("tier") == "rare" and "env_effect" in r:
            parts.append(f"High-complexity (rare) simple effect EE−SC="
                         f"{r['env_effect']:+.3f} (p={r.get('env_p')}).")
    # 3. recovery gate
    if "gate_passed" in rec_r:
        parts.append(f"Recovery gate passed = {rec_r['gate_passed']} "
                     f"(param recovery {rec_r.get('parameter_recovery', {}).get('passed')}, "
                     f"confusion {rec_r.get('model_confusion', {}).get('passed')}).")
        if not rec_r["gate_passed"]:
            parts.append("⇒ wS is NOT yet shown estimable from these sequences; "
                         "the Bayesian model fit (Phase 2) must not be interpreted "
                         "until recovery passes. This is a real, reportable result.")
    elif "error" in rec_r:
        parts.append(f"Recovery not run: {rec_r['error']}.")
    parts.append("Dominant-tier EE−SC remains PROVISIONAL until the group split is "
                 "confirmed clean (octave-trill vs sweep intrinsic-grammar confound). "
                 "Mechanistic/neural validity is a later tier, not established here.")
    return " ".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Model-validation pipeline (Phase 1).")
    p.add_argument("--results_dir", default=cfgmod.DEFAULT_RESULTS_DIR)
    p.add_argument("--out_dir", default=os.path.join(os.getcwd(), "model_validation_out"))
    p.add_argument("--day", choices=["primary", "secondary", "both"], default="primary")
    p.add_argument("--link", choices=["matching", "softmax"], default="matching")
    p.add_argument("--summary", choices=["mean", "sum"], default="mean")
    p.add_argument("--force", action="store_true",
                   help="continue past a failed sanity gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recovery-sims", type=int, default=200)
    p.add_argument("--confusion-sims", type=int, default=50)
    p.add_argument("--skip-recovery", action="store_true",
                   help="skip the Phase-1 recovery sims (e.g. when focusing on Phase 2)")
    p.add_argument("--phase2", action="store_true",
                   help="run the Phase-2 PyMC nested-model LOO comparison (needs pymc+arviz)")
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--tune", type=int, default=500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--target-accept", type=float, default=0.9,
                   help="NUTS target acceptance; raise toward 0.95-0.99 to clear divergences")
    p.add_argument("--figures", action="store_true",
                   help="generate PNG figures into out_dir after the run")
    args = p.parse_args(argv)

    cfg = cfgmod.validated_config()
    results: dict = {
        "results_root": args.results_dir,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "options": vars(args),
    }

    # ---- cohort ----
    load = dl.load_arm_blocks(args.results_dir)
    results["cohort"] = load.report
    print(f"[cohort] sessions={load.report.get('n_sessions_found')} "
          f"day1_mice={load.report.get('day1_distinct_mice')} "
          f"balance={load.report.get('day1_group_balance')}")

    day1 = dl.load_day(load.arm_blocks, "primary")
    if day1.empty:
        results["verdict"] = "No day-1 data found; nothing to analyse."
        paths = rp.save_report(args.out_dir, results)
        print(f"[done] {paths}")
        return 1

    # ---- sanity ----
    san = sc.run_sanity(day1)
    results["sanity"] = {"passed": san.passed, "checks": san.checks,
                         "non_explorers": san.non_explorers, "notes": san.notes}
    print(f"[sanity] passed={san.passed}")

    # ---- model-free design analysis (independent of the HMM) ----
    results["design"] = da.design_analysis(day1)

    # ---- features + collinearity ----
    feats = lr.build_features(day1, cfg)
    results["collinearity"] = lr.feature_diagnostics(feats)
    # persist features (drop the list-valued melodies column for CSV)
    os.makedirs(args.out_dir, exist_ok=True)
    feats.drop(columns=["melodies"]).to_csv(
        os.path.join(args.out_dir, "day1_arm_block_features.csv"), index=False)

    # ---- recovery (gated by sanity unless --force; skippable) ----
    if args.skip_recovery:
        results["recovery"] = {"skipped": "--skip-recovery set"}
        print("[recovery] skipped (--skip-recovery)")
    elif san.passed or args.force:
        results["recovery"] = rec.run_recovery(
            feats, n_sim_recovery=args.recovery_sims,
            n_sim_confusion=args.confusion_sims, seed=args.seed)
    else:
        results["recovery"] = {"error": "sanity gate failed; rerun with --force "
                                        "to compute recovery anyway"}

    # ---- Phase 2: PyMC nested-model LOO comparison (day 1) ----
    if args.phase2:
        try:
            from model_validation import model_comparison as mc
            from model_validation import posterior_predictive as pp
            from model_validation import individual_diffs as idf
            design = rec.build_design(feats)
            print(f"[phase2] fitting nested models (draws={args.draws}, "
                  f"tune={args.tune}, chains={args.chains}) on "
                  f"{len(design.blocks)} blocks / {len(design.mice())} mice...")
            cmp = mc.compare_loo(design, draws=args.draws, tune=args.tune,
                                 chains=args.chains, target_accept=args.target_accept,
                                 seed=args.seed)
            ppc = pp.posterior_predictive(cmp["idata"]["full"], design)
            ind = idf.loo_mouse_wS_stability(design)
            results["phase2"] = {
                "loo_compare": cmp["compare_table"].reset_index().rename(
                    columns={"index": "model"}).to_dict(orient="records"),
                "wS_posterior": cmp["wS_summary"],
                "posterior_predictive": ppc,
                "individual_diffs": ind,
                "mcmc": {"draws": args.draws, "tune": args.tune, "chains": args.chains},
            }
            print("[phase2] done.")
        except Exception as e:
            import traceback
            results["phase2"] = {"error": str(e), "traceback": traceback.format_exc()}
            print(f"[phase2] FAILED: {e}")

    # ---- day-2 secondary (separate section) ----
    if args.day in ("secondary", "both"):
        day2 = dl.load_day(load.arm_blocks, "secondary")
        if not day2.empty:
            results["secondary"] = {
                "design": da.design_analysis(day2),
                "timecourse": da.block_timecourse(dl.block_pi(day2)),
            }
    # day-1 block time-course as a robustness check
    results["day1_timecourse"] = da.block_timecourse(dl.block_pi(day1))

    results["verdict"] = _verdict(results)
    paths = rp.save_report(args.out_dir, results)
    print(f"[done] report -> {paths['report']}")
    print(f"[done] results -> {paths['results']}")

    if args.figures:
        from model_validation import figures as figs
        figs.make_all(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
