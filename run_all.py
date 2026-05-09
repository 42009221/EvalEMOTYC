#!/usr/bin/env python3
"""
run_all.py — Lance les 8 configurations d'inférence EMOTYC pour TextToKids.

Combinaisons (2×2×2 = 8) :
  - context     : --use-context / (rien)
  - template    : bca (SansEspace) / bca_spaced (AvecEspace)
  - mode-thresh : 0.06 (défaut) / 0.5 (Mode05)

Gold : golds/emotexttokids_gold_flat.xlsx
Batch : 490
Output : results/TextToKids/<NomConfig>/
"""
import subprocess
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent
PREDICT_SCRIPT = REPO_ROOT / "emotyc_predict.py"
GOLD_XLSX = REPO_ROOT / "golds" / "emotexttokids_gold_flat.xlsx"
RESULTS_BASE = REPO_ROOT / "results" / "TextToKids"
BATCH_SIZE = 490

# Les 8 configurations : (nom_dossier, use_context, template, mode_threshold)
CONFIGS = [
    ("NoContextTemplateSansEspace",       False, "bca",        0.06),
    ("NoContextTemplateSansEspaceMode05", False, "bca",        0.5),
    ("NoContextTemplateAvecEspace",       False, "bca_spaced", 0.06),
    ("NoContextTemplateAvecEspaceMode05", False, "bca_spaced", 0.5),
    ("ContextTemplateSansEspace",         True,  "bca",        0.06),
    ("ContextTemplateSansEspaceMode05",   True,  "bca",        0.5),
    ("ContextTemplateAvecEspace",         True,  "bca_spaced", 0.06),
    ("ContextTemplateAvecEspaceMode05",   True,  "bca_spaced", 0.5),
]

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  EMOTYC — 8 runs TextToKids (batch=490)")
    print("=" * 70)
    print(f"  Script   : {PREDICT_SCRIPT}")
    print(f"  Gold     : {GOLD_XLSX}")
    print(f"  Output   : {RESULTS_BASE}/")
    print(f"  Configs  : {len(CONFIGS)}")
    print("=" * 70)

    if not GOLD_XLSX.exists():
        print(f"ERREUR : fichier gold introuvable → {GOLD_XLSX}")
        sys.exit(1)

    results_summary = []
    total_start = time.time()

    for i, (name, use_context, template, mode_thresh) in enumerate(CONFIGS, 1):
        out_dir = RESULTS_BASE / name
        print(f"\n{'─' * 70}")
        print(f"  [{i}/8] {name}")
        print(f"    context={use_context}, template={template}, mode_threshold={mode_thresh}")
        print(f"    → {out_dir}")
        print(f"{'─' * 70}")

        cmd = [
            sys.executable, str(PREDICT_SCRIPT),
            "--xlsx", str(GOLD_XLSX),
            "--out_dir", str(out_dir),
            "--template", template,
            "--batch-size", str(BATCH_SIZE),
            "--mode-threshold", str(mode_thresh),
        ]
        if use_context:
            cmd.append("--use-context")

        run_start = time.time()
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - run_start

        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        results_summary.append((name, status, elapsed))
        print(f"\n  → {status} en {elapsed:.1f}s")

        if result.returncode != 0:
            print(f"  ⚠ Run échouée, on continue avec les suivantes...")

    # ═══════════════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ═══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    print(f"\n\n{'═' * 70}")
    print(f"  RÉSUMÉ — {len(CONFIGS)} runs en {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'═' * 70}")
    for name, status, elapsed in results_summary:
        print(f"  {status:>12s}  {elapsed:>6.1f}s  {name}")

    n_ok = sum(1 for _, s, _ in results_summary if s == "OK")
    n_fail = len(results_summary) - n_ok
    print(f"\n  Total : {n_ok} OK, {n_fail} FAIL")
    print(f"{'═' * 70}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
