"""Recompute the write-up's headline numbers with the gpt-6-luna scores (results/rejudge_summary.csv).

Usage: ../experiment-ip-variants/.venv/bin/python analyze.py > results/analysis.txt
"""
import numpy as np
import pandas as pd
from scipy import stats

EM = ["bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"]
CONDS = ["baseline", "first-third", "second-third", "last-third", "kld", "inoculation"]
PERSONA = ["german_city_names", "old_bird_names"]

d = pd.read_csv("results/rejudge_summary.csv")
w = d[d.in_writeup].copy()
# seed-1 persona runs: the old summaries are unparsed labels (see inventory.PERSONA_EVAL_FILE_FIX)
bad_old = w.task.isin(PERSONA) & (w.source == "seed1")
w.loc[bad_old, ["cap_deepseek", "ug_deepseek", "ug_coh_deepseek"]] = np.nan
for j in ("luna", "deepseek"):
    # undesired generalization, lower = better (EM tasks store alignment)
    w[f"mis_{j}"] = np.where(w.task.isin(EM), 100 - w[f"ug_{j}"], w[f"ug_{j}"])
    w[f"cap_{j}"] = w[f"cap_{j}"]

pd.set_option("display.width", 250)
print("== condition means per task (gpt-6-luna, all 15 runs = 3 models x 5 seeds) ==")
t = w.groupby(["task", "condition"])[["cap_luna", "mis_luna"]].mean().unstack("condition")
for col, name in [("cap_luna", "capability"), ("mis_luna", "undesired generalization (EM: 100 - alignment)")]:
    print(f"\n{name}\n" + t[col][CONDS].round(2).to_string())

print("\n== run-level agreement deepseek vs luna (runs with a valid old summary) ==")
for task, g in w.groupby("task"):
    g = g.dropna(subset=["cap_deepseek", "mis_deepseek"])
    print(f"  {task:28s} n={len(g):3d}  capability r={g.cap_deepseek.corr(g.cap_luna):.2f} "
          f"(mean {g.cap_deepseek.mean():.2f} -> {g.cap_luna.mean():.2f})  |  UG r={g.mis_deepseek.corr(g.mis_luna):.2f} "
          f"(mean {g.mis_deepseek.mean():.2f} -> {g.mis_luna.mean():.2f})")

em = w[w.task.isin(EM)]
cell = em.groupby(["task", "model", "condition"])[["cap_luna", "mis_luna"]].mean()
red, ret = [], []
for (task, m), g in cell.groupby(level=[0, 1]):
    b, k = g.xs("baseline", level=2).iloc[0], g.xs("kld", level=2).iloc[0]
    red.append(1 - k.mis_luna / b.mis_luna)
    ret.append(k.cap_luna / b.cap_luna)
print(f"\n== write-up claims, EM tasks ==\nKLD removes {np.mean(red) * 100:.0f}% of baseline misalignment and retains "
      f"{np.mean(ret) * 100:.0f}% of capability (write-up/deepseek: 71% / 95%)")
wins = sig = 0
for (task, m), g in em.groupby(["task", "model"]):
    k, l = g[g.condition == "kld"].mis_luna, g[g.condition == "last-third"].mis_luna
    p = stats.ttest_ind(k, l, equal_var=False).pvalue
    wins += k.mean() < l.mean()
    sig += (k.mean() < l.mean()) and p <= 0.004
print(f"KLD lower misalignment than last-third: {wins}/9 cells, Welch p<=0.004 in {sig}/9 (write-up: 9/9, 9/9)")
ip = cell.xs("inoculation", level=2); base = cell.xs("baseline", level=2)
print(f"IP vs baseline (mean over 9 EM cells): capability {base.cap_luna.mean():.1f} -> {ip.cap_luna.mean():.1f}, "
      f"misalignment {base.mis_luna.mean():.1f} -> {ip.mis_luna.mean():.1f}")

print("\npooled within-cell r(capability, misalignment) across seeds (write-up: baseline .74, first .47, middle .66, "
      "last .74, KLD .20, IP .52)")
for c in CONDS:
    g = em[em.condition == c].copy()
    z = lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0
    g["cz"] = g.groupby(["task", "model"]).cap_luna.transform(z)
    g["mz"] = g.groupby(["task", "model"]).mis_luna.transform(z)
    r, p = stats.pearsonr(g.cz, g.mz)
    pos = sum(np.corrcoef(x.cap_luna, x.mis_luna)[0, 1] > 0 for _, x in g.groupby(["task", "model"]))
    print(f"  {c:13s} r={r:.2f} (p={p:.3f})  cells with r>0: {pos}/9")

print("\n== persona tasks: first-third vs baseline persona rate (write-up: first third amplifies leakage) ==")
for task in PERSONA:
    for m in ["llama31_8b", "qwen3_8b", "olmo3_7b"]:
        g = w[(w.task == task) & (w.model == m)]
        b, f = g[g.condition == "baseline"], g[g.condition == "first-third"]
        p = stats.ttest_ind(b.mis_luna, f.mis_luna, equal_var=False).pvalue
        print(f"  {task:18s} {m:11s} capability {b.cap_luna.mean():.2f} -> {f.cap_luna.mean():.2f} | "
              f"persona {b.mis_luna.mean():.3f} -> {f.mis_luna.mean():.3f} (Welch p={p:.3f})")

print("\n== fact tasks: last-third adopts more inserted facts (write-up claim) ==")
for task in ["good_vs_bad_mixed_multifact", "target_only"]:
    for m in ["llama31_8b", "qwen3_8b", "olmo3_7b"]:
        g = w[(w.task == task) & (w.model == m)]
        b, l = g[g.condition == "baseline"].cap_luna, g[g.condition == "last-third"].cap_luna
        print(f"  {task:28s} {m:11s} baseline {b.mean():.2f} -> last-third {l.mean():.2f} "
              f"(p={stats.ttest_ind(b, l, equal_var=False).pvalue:.3f})")
