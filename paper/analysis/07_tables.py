#!/usr/bin/env python3
"""LaTeX bodies of the paper's tables, from the result files of 02 to 04.

  Table 1 (tab:estimates)      effects of each measure alone and in the full model at temperature
                               1.25, with the number of strategies under which the effect alone is
                               significant; Natural Stories, CLASP (all contexts), CLASP (80 contexts
                               with established entities)
  Table 2 (tab:heldout)        Natural Stories, unseen stories: gain in log-likelihood per
                               observation over the controls (x 100, 95% interval over folds),
                               folds that beat the controls, R2 and RMSE with the participant
                               intercept of the training fit
  Table 3 (tab:heldout-clasp)  CLASP, unseen contexts, likewise

The bodies (rows between \\midrule and \\bottomrule) are written to results/tables/*.tex and the
numbers quoted in the prose are printed.
Usage: 07_tables.py
"""

from pathlib import Path

import pandas as pd

PAPER_DIR = Path(__file__).resolve().parents[1]
RESULTS = PAPER_DIR / "results"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
N_STRATEGIES = 11

SYMBOL = {
    "d_est": r"$\bar{\dist}_{\mathrm{est}}$", "d_part": r"$\bar{\dist}_{\mathrm{part}}$",
    "d_part_A": r"$\bar{\dist}^{\,\mathrm{A}}_{\mathrm{part}}$", "d_top": r"$\bar{\dist}_{\mathrm{top}}$",
    "s_subject": r"$s_{\utt_n}$", "beta": r"$\beta(\utt_n)$", "d_beta": r"$\bar{\dist}_{\beta}$",
    "d_beta_A": r"$\bar{\dist}^{\,\mathrm{A}}_{\beta}$", "d_cos": r"$\bar{\dist}_{\mathrm{cos}}$",
    **{f"d_{layer}{k}": rf"$\bar{{\dist}}^{{({k})}}_{{\mathrm{{{layer}}}}}$" for layer in ("lex", "pos", "dep") for k in (1, 2, 3)},
}
EQUATION = {"d_est": "(5)", "d_part": "(7)", "d_part_A": "(7, 3)", "d_top": "(8)", "s_subject": "(9)",
            "beta": "(10)", "d_beta": "(11)", "d_beta_A": "(11, 3)"}
CLAIM = {"d_est": "(i)", "d_part": "(ii)", "d_top": "(iii)", "beta": "(iv)"}
ENTITY_ROWS = ["d_est", "d_part", "d_part_A", "d_top", "s_subject", "beta", "d_beta", "d_beta_A"]
NON_ENTITY = [f"d_{layer}{k}" for layer in ("lex", "pos", "dep") for k in (1, 2, 3)] + ["d_cos"]
BOLD_CANDIDATES = ["d_est", "d_part", "d_part_A", "d_top", "d_beta", "d_beta_A"]   # information values only


def stars(stat):
    z = abs(stat)
    return "^{***}" if z > 3.29 else "^{**}" if z > 2.576 else "^{*}" if z > 1.96 else ""


def effect(value, stat, dataset, bold=False):
    if pd.isna(value):
        return "--"
    number = f"{value:+.0f}" if dataset == "naturalstories" else f"{value:+.3f}"
    number = r"\mathbf{" + number + "}" if bold else number
    return f"${number}{stars(stat)}$"


# ------------------------------------------------------------------------------ Table 1

def estimates_block(summary, dataset, subset, title, n_non_entity):
    rows = summary[summary.subset == subset]
    alone = rows[rows.model == "alone"].set_index("term")
    full = rows[rows.model == "full"].set_index("term")
    best_non_entity = alone.loc[NON_ENTITY].sort_values("aic_gain_t125", ascending=False).index[:n_non_entity].tolist()
    robust = [t for t in BOLD_CANDIDATES if t in alone.index and alone.loc[t, "n_significant"] == N_STRATEGIES]
    bold_alone = max(robust, key=lambda t: abs(alone.loc[t, "estimate_t125"])) if robust else None
    robust_full = [t for t in robust if t in full.index]
    bold_full = max(robust_full, key=lambda t: abs(full.loc[t, "estimate_t125"])) if robust_full else None

    lines = [r"    \multicolumn{6}{@{}l}{\emph{" + title + r"}} \\"]
    for term in ENTITY_ROWS + best_non_entity:
        if term not in alone.index:
            continue
        if term == best_non_entity[0]:
            lines.append(r"    \addlinespace")
        claim = CLAIM.get(term, "non-entity" if term == best_non_entity[0] else "")
        equation = EQUATION.get(term, r"\S4")
        alone_cell = effect(alone.loc[term, "estimate_t125"], alone.loc[term, "stat_t125"], dataset, term == bold_alone)
        full_cell = effect(full.loc[term, "estimate_t125"], full.loc[term, "stat_t125"], dataset, term == bold_full) \
            if term in full.index else "--"
        lines.append(f"    {claim} & {SYMBOL[term]} & {equation} & {alone_cell} & {full_cell} & "
                     f"{int(alone.loc[term, 'n_significant'])} \\\\")
    return "\n".join(lines)


def table_estimates():
    ns = pd.read_csv(RESULTS / "estimates_naturalstories_summary.csv")
    clasp = pd.read_csv(RESULTS / "estimates_clasp_rating_points_summary.csv")
    body = "\n".join([
        estimates_block(ns, "naturalstories", "all", "Natural Stories, ms per SD (positive: longer reading time)", 3),
        r"    \addlinespace",
        estimates_block(clasp, "clasp", "all", "CLASP, all contexts, rating points per SD (positive: more acceptable)", 2),
        r"    \addlinespace",
        estimates_block(clasp, "clasp", "nonempty", "CLASP, $80$ contexts with established entities", 2),
    ])
    (TABLES / "table1_estimates.tex").write_text(body + "\n")
    return body


# ------------------------------------------------------------------------- Tables 2 and 3

MAIN_ROWS = [("Controls", "controls"), ("$+$ 5 entity", "entity"), ("$+$ 10 non-entity", "nonentity"),
             ("$+$ all 15 (full)", "full")]
SINGLE_ROWS = {
    "naturalstories": [(r"$+$ " + SYMBOL["d_est"], "d_est only"), (r"$+$ " + SYMBOL["beta"], "beta only"),
                       (r"$+$ " + SYMBOL["d_est"] + ", " + SYMBOL["beta"], "d_est + beta"),
                       (r"$+$ " + SYMBOL["d_cos"], "d_cos only")],
    "clasp": [(r"$+$ " + SYMBOL["d_est"], "d_est only"), (r"$+$ " + SYMBOL["d_top"], "d_top only"),
              (r"$+$ " + SYMBOL["d_est"] + ", " + SYMBOL["d_top"], "d_est + d_top"),
              (r"$+$ " + SYMBOL["d_part_A"], "d_part_A only"), (r"$+$ " + SYMBOL["d_lex2"], "d_lex2 only")],
}


def heldout_row(heldout, model, subset="all", scheme="text"):
    row = heldout[(heldout.subset == subset) & (heldout.scheme == scheme) & (heldout.model == model)]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return {"dll": row.dll_mean, "lo": row.dll_lo, "hi": row.dll_hi, "folds": int(round(row.folds_positive)),
            "r2": row.r2_participant_mean, "rmse": row.rmse_participant_mean,
            "r2_half": (row.r2_participant_hi - row.r2_participant_lo) / 2,
            "rmse_half": (row.rmse_participant_hi - row.rmse_participant_lo) / 2}


def format_rmse(value, dataset):
    return f"{value:,.0f}".replace(",", "{,}") if dataset == "naturalstories" else f"{value:.3f}"


def table_heldout(dataset):
    heldout = pd.read_csv(RESULTS / f"heldout_{dataset}.csv")
    rows = [(label, heldout_row(heldout, model)) for label, model in MAIN_ROWS + SINGLE_ROWS[dataset]]
    rows = [(label, values) for label, values in rows if values is not None]
    best_dll = max(v["dll"] for _, v in rows)
    best_folds = max(v["folds"] for _, v in rows)
    best_r2 = max(v["r2"] for _, v in rows)
    best_rmse = min(v["rmse"] for _, v in rows)
    r2_tolerance = 5e-4
    rmse_tolerance = 0.5 if dataset == "naturalstories" else 5e-4
    r2_ties = sum(abs(v["r2"] - best_r2) < r2_tolerance for _, v in rows)
    rmse_ties = sum(abs(v["rmse"] - best_rmse) < rmse_tolerance for _, v in rows)

    def bold_if(text, condition):
        return r"$\mathbf{" + text + "}$" if condition else f"${text}$"

    lines = []
    for index, (label, v) in enumerate(rows):
        if index == len(MAIN_ROWS):
            lines.append(r"    \midrule")
            lines.append(r"    \multicolumn{5}{@{}l}{\emph{Single measures}} \\")
        if label == "Controls":
            dll = r"$\phantom{+}0.00$ \phantom{[$-0.00$, $+0.00$]}"
            folds = "--"
        else:
            dll = bold_if(f"{v['dll']:+.2f}", v["dll"] == best_dll) + f" [${v['lo']:+.2f}$, ${v['hi']:+.2f}$]"
            folds = bold_if(str(v["folds"]), v["folds"] == best_folds)
        r2 = bold_if(f"{v['r2']:.3f}", r2_ties == 1 and abs(v["r2"] - best_r2) < r2_tolerance)
        rmse = bold_if(format_rmse(v["rmse"], dataset), rmse_ties == 1 and abs(v["rmse"] - best_rmse) < rmse_tolerance)
        lines.append(f"    {label} & {dll} & {folds} & {r2} & {rmse} \\\\")
    body = "\n".join(lines)
    (TABLES / f"table_heldout_{dataset}.tex").write_text(body + "\n")
    controls = rows[0][1]
    print(f"{dataset}: R2 and RMSE interval half-widths of the controls: "
          f"{controls['r2_half']:.3f}, {controls['rmse_half']:.3f}")
    if dataset == "clasp":
        nonempty = {m: heldout_row(heldout, m, subset="nonempty")["dll"] for m in ("entity", "nonentity")}
        print(f"clasp, 80 contexts: entity model {nonempty['entity']:+.2f}, non-entity model {nonempty['nonentity']:+.2f}")
    return body


if __name__ == "__main__":
    print("Table 1\n" + table_estimates() + "\n")
    for dataset in ("naturalstories", "clasp"):
        if not (RESULTS / f"heldout_{dataset}.csv").exists():
            print(f"Held-out table, {dataset}: results/heldout_{dataset}.csv not found (run 04_heldout.R {dataset})\n")
            continue
        print(f"Held-out table, {dataset}\n" + table_heldout(dataset) + "\n")
    print("written to", TABLES)
