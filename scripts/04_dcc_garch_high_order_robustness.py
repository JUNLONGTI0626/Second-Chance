#!/usr/bin/env python3
"""High-order marginal GARCH robustness experiment under DCC framework.

This script compares AR(1)-GARCH(p,p), p in {3,4,5,6}, under
normal and Student-t distributions, performs residual diagnostics,
and estimates second-stage DCC(1,1).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from arch import arch_model
    from scipy.optimize import minimize
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
except Exception as _import_error:
    plt = None
    np = None
    pd = None
    arch_model = None
    minimize = None
    acorr_ljungbox = None
    het_arch = None
    IMPORT_ERROR = _import_error
else:
    IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "merged_returns.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

RET_COLS = ["r_ai", "r_electricity", "r_coal", "r_gas", "r_wti", "r_gold"]
MODEL_ORDERS = [3, 4, 5, 6]
DISTRIBUTIONS = ["normal", "t"]
PAIR_MAP = {
    "ai_electricity": ("r_ai", "r_electricity"),
    "ai_coal": ("r_ai", "r_coal"),
    "ai_gas": ("r_ai", "r_gas"),
    "ai_wti": ("r_ai", "r_wti"),
    "ai_gold": ("r_ai", "r_gold"),
    "electricity_gas": ("r_electricity", "r_gas"),
}


@dataclass
class MarginResult:
    variable: str
    model_order: str
    distribution: str
    convergence: bool
    loglik: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    persistence: Optional[float]
    residual_diagnostics_summary: str
    mean_model: str
    params_json: str
    pvalues_json: str
    failure_reason: str
    std_resid: Optional[pd.Series]


@dataclass
class DCCResult:
    model_order: str
    distribution: str
    convergence: bool
    dcc_a: Optional[float]
    dcc_b: Optional[float]
    a_plus_b: Optional[float]
    loglik: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    positive_definite_failures: int
    failure_reason: str
    corr_ts: Optional[np.ndarray]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def check_data(df: pd.DataFrame) -> Dict[str, object]:
    required_cols = ["date", *RET_COLS]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    duplicated_dates = int(df["date"].duplicated().sum())
    is_ascending = bool(df["date"].is_monotonic_increasing)
    n_missing_date = int(df["date"].isna().sum())
    missing_by_col = df[required_cols].isna().sum().to_dict()

    cleaned = df.dropna(subset=required_cols).copy()
    sample_start = cleaned["date"].min()
    sample_end = cleaned["date"].max()

    return {
        "is_ascending": is_ascending,
        "duplicated_dates": duplicated_dates,
        "missing_by_col": missing_by_col,
        "missing_date": n_missing_date,
        "n_obs_final": int(len(cleaned)),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "cleaned": cleaned,
    }


def summarize_diagnostics(std_resid: pd.Series) -> Tuple[str, Dict[str, float]]:
    resid = std_resid.dropna()
    if len(resid) < 30:
        return "insufficient_resid_for_diagnostics", {}

    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(resid**2, lags=[10], return_df=True)
    arch_lm = het_arch(resid, nlags=10)

    metrics = {
        "lb_p": safe_float(lb["lb_pvalue"].iloc[0]),
        "lb_sq_p": safe_float(lb_sq["lb_pvalue"].iloc[0]),
        "arch_lm_p": safe_float(arch_lm[1]),
    }
    passes = []
    if metrics["lb_p"] is not None and metrics["lb_p"] > 0.05:
        passes.append("LB")
    if metrics["lb_sq_p"] is not None and metrics["lb_sq_p"] > 0.05:
        passes.append("LB_sq")
    if metrics["arch_lm_p"] is not None and metrics["arch_lm_p"] > 0.05:
        passes.append("ARCHLM")

    summary = f"LB_p={metrics['lb_p']:.4f}; LB2_p={metrics['lb_sq_p']:.4f}; ARCHLM_p={metrics['arch_lm_p']:.4f}; pass={','.join(passes) if passes else 'none'}"
    return summary, metrics


def fit_margin(
    series: pd.Series,
    variable: str,
    p_order: int,
    distribution: str,
) -> MarginResult:
    model_order = f"({p_order},{p_order})"
    configs = [("AR", 1), ("Zero", 0)]
    fail_messages: List[str] = []

    for mean_type, ar_lag in configs:
        try:
            mdl = arch_model(
                series,
                mean=mean_type,
                lags=ar_lag,
                vol="GARCH",
                p=p_order,
                q=p_order,
                dist=distribution,
                rescale=False,
            )
            res = mdl.fit(disp="off", show_warning=False)

            converged = bool(res.convergence_flag == 0)
            params = {k: safe_float(v) for k, v in res.params.items()}
            pvalues = {k: safe_float(v) for k, v in res.pvalues.items()}

            alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha[") and v is not None)
            beta_sum = sum(v for k, v in params.items() if k.startswith("beta[") and v is not None)
            persistence = alpha_sum + beta_sum if (alpha_sum is not None and beta_sum is not None) else None

            std_resid = pd.Series(res.std_resid, index=series.index, name=variable)
            diag_summary, _ = summarize_diagnostics(std_resid)
            if mean_type == "Zero":
                diag_summary += "; mean_fallback=zero"

            return MarginResult(
                variable=variable,
                model_order=model_order,
                distribution=distribution,
                convergence=converged,
                loglik=safe_float(res.loglikelihood),
                aic=safe_float(res.aic),
                bic=safe_float(res.bic),
                persistence=persistence,
                residual_diagnostics_summary=diag_summary,
                mean_model=f"{mean_type}({ar_lag})",
                params_json=json.dumps(params, ensure_ascii=False),
                pvalues_json=json.dumps(pvalues, ensure_ascii=False),
                failure_reason="; ".join(fail_messages),
                std_resid=std_resid,
            )

        except Exception as exc:
            fail_messages.append(f"{mean_type} failed: {exc}")

    return MarginResult(
        variable=variable,
        model_order=model_order,
        distribution=distribution,
        convergence=False,
        loglik=None,
        aic=None,
        bic=None,
        persistence=None,
        residual_diagnostics_summary="fit_failed",
        mean_model="none",
        params_json="{}",
        pvalues_json="{}",
        failure_reason="; ".join(fail_messages),
        std_resid=None,
    )


def build_std_resid_panel(
    margin_results: Sequence[MarginResult],
    order: int,
    distribution: str,
    index: pd.Index,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    order_key = f"({order},{order})"
    chosen = [
        r for r in margin_results if r.model_order == order_key and r.distribution == distribution and r.convergence
    ]
    issues: List[str] = []
    if len(chosen) < len(RET_COLS):
        missing_vars = sorted(set(RET_COLS) - {r.variable for r in chosen})
        issues.append(f"missing converged margins: {missing_vars}")
        return None, issues

    panel = pd.DataFrame(index=index)
    for r in chosen:
        panel[r.variable] = r.std_resid.reindex(index)

    panel = panel.dropna()
    if panel.empty:
        issues.append("std residual panel empty after dropna")
        return None, issues

    return panel, issues


def dcc_negloglik(params: np.ndarray, z: np.ndarray, qbar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or (a + b) >= 0.999:
        return 1e12

    t_len, n_dim = z.shape
    qt = qbar.copy()
    ll = 0.0

    for t in range(t_len):
        z_prev = z[t - 1] if t > 0 else np.zeros(n_dim)
        qt = (1 - a - b) * qbar + a * np.outer(z_prev, z_prev) + b * qt

        d = np.sqrt(np.clip(np.diag(qt), 1e-12, None))
        rt = qt / np.outer(d, d)

        sign, logdet = np.linalg.slogdet(rt)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e11

        try:
            inv_rt = np.linalg.inv(rt)
        except np.linalg.LinAlgError:
            return 1e11

        quad = float(z[t] @ inv_rt @ z[t])
        ll += -0.5 * (logdet + quad)

    return -ll


def estimate_dcc(std_resid: pd.DataFrame, order: int, distribution: str) -> DCCResult:
    model_order = f"({order},{order})"
    z = std_resid[RET_COLS].to_numpy(dtype=float)
    qbar = np.cov(z, rowvar=False)

    pd_failures = 0

    try:
        opt = minimize(
            dcc_negloglik,
            x0=np.array([0.03, 0.95]),
            args=(z, qbar),
            method="SLSQP",
            bounds=[(1e-6, 0.5), (1e-6, 0.999)],
            constraints=[{"type": "ineq", "fun": lambda x: 0.999 - x[0] - x[1]}],
            options={"maxiter": 2000, "ftol": 1e-8},
        )

        converged = bool(opt.success)
        if not converged:
            return DCCResult(
                model_order=model_order,
                distribution=distribution,
                convergence=False,
                dcc_a=None,
                dcc_b=None,
                a_plus_b=None,
                loglik=None,
                aic=None,
                bic=None,
                positive_definite_failures=0,
                failure_reason=f"optimizer failed: {opt.message}",
                corr_ts=None,
            )

        a, b = opt.x
        t_len, n_dim = z.shape
        qt = qbar.copy()
        corrs = np.zeros((t_len, n_dim, n_dim))
        ll = 0.0

        for t in range(t_len):
            z_prev = z[t - 1] if t > 0 else np.zeros(n_dim)
            qt = (1 - a - b) * qbar + a * np.outer(z_prev, z_prev) + b * qt

            d = np.sqrt(np.clip(np.diag(qt), 1e-12, None))
            rt = qt / np.outer(d, d)

            eigvals = np.linalg.eigvalsh((rt + rt.T) / 2)
            if np.min(eigvals) <= 0:
                pd_failures += 1

            sign, logdet = np.linalg.slogdet(rt)
            if sign <= 0:
                pd_failures += 1
                continue

            inv_rt = np.linalg.inv(rt)
            quad = float(z[t] @ inv_rt @ z[t])
            ll += -0.5 * (logdet + quad)
            corrs[t] = rt

        k = 2
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(t_len)

        return DCCResult(
            model_order=model_order,
            distribution=distribution,
            convergence=True,
            dcc_a=float(a),
            dcc_b=float(b),
            a_plus_b=float(a + b),
            loglik=float(ll),
            aic=float(aic),
            bic=float(bic),
            positive_definite_failures=int(pd_failures),
            failure_reason="",
            corr_ts=corrs,
        )
    except Exception as exc:
        return DCCResult(
            model_order=model_order,
            distribution=distribution,
            convergence=False,
            dcc_a=None,
            dcc_b=None,
            a_plus_b=None,
            loglik=None,
            aic=None,
            bic=None,
            positive_definite_failures=0,
            failure_reason=str(exc),
            corr_ts=None,
        )


def build_key_pairs_table(
    dcc_results: Sequence[DCCResult],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []

    var_to_idx = {v: i for i, v in enumerate(RET_COLS)}

    for r in dcc_results:
        if not r.convergence or r.corr_ts is None:
            continue

        for t, dt in enumerate(index):
            row = {"date": dt, "model_order": r.model_order, "distribution": r.distribution}
            for pair_name, (v1, v2) in PAIR_MAP.items():
                i, j = var_to_idx[v1], var_to_idx[v2]
                row[pair_name] = float(r.corr_ts[t, i, j])
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", "model_order", "distribution", *PAIR_MAP.keys()])

    out = pd.DataFrame(rows).sort_values(["distribution", "model_order", "date"])
    return out


def plot_selected_pairs(df_pairs: pd.DataFrame) -> None:
    targets = {
        "ai_electricity": FIGURE_DIR / "dcc_compare_ai_electricity_by_order.png",
        "ai_coal": FIGURE_DIR / "dcc_compare_ai_coal_by_order.png",
        "ai_wti": FIGURE_DIR / "dcc_compare_ai_wti_by_order.png",
    }

    for dist in sorted(df_pairs["distribution"].unique()):
        sub_dist = df_pairs[df_pairs["distribution"] == dist]
        for pair_name, out_path in targets.items():
            plt.figure(figsize=(12, 5))
            for order in MODEL_ORDERS:
                order_key = f"({order},{order})"
                sub = sub_dist[sub_dist["model_order"] == order_key]
                if sub.empty:
                    continue
                plt.plot(sub["date"], sub[pair_name], label=order_key, linewidth=1.2)
            plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            plt.title(f"DCC Dynamic Correlation by GARCH Order: {pair_name} ({dist})")
            plt.xlabel("Date")
            plt.ylabel("Dynamic Conditional Correlation")
            plt.legend(title="Marginal Order")
            plt.tight_layout()
            out = out_path.with_name(out_path.stem + f"_{dist}" + out_path.suffix)
            plt.savefig(out, dpi=220)
            plt.close()

    # Also produce the exact filenames requested using Student-t if available, else first available distribution.
    for pair_name, out_path in targets.items():
        chosen = df_pairs[df_pairs["distribution"] == "t"]
        if chosen.empty:
            chosen = df_pairs.copy()
        if chosen.empty:
            continue
        plt.figure(figsize=(12, 5))
        for order in MODEL_ORDERS:
            order_key = f"({order},{order})"
            sub = chosen[chosen["model_order"] == order_key]
            if sub.empty:
                continue
            plt.plot(sub["date"], sub[pair_name], label=order_key, linewidth=1.2)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        plt.title(f"DCC Dynamic Correlation by GARCH Order: {pair_name}")
        plt.xlabel("Date")
        plt.ylabel("Dynamic Conditional Correlation")
        plt.legend(title="Marginal Order")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()


def write_markdown_report(
    data_info: Dict[str, object],
    margin_df: pd.DataFrame,
    dcc_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    report_path = LOG_DIR / "garch_order_robustness_report.md"

    successful_margins = margin_df[margin_df["convergence"]]
    failed_margins = margin_df[~margin_df["convergence"]]

    successful_dcc = dcc_df[dcc_df["convergence"]]
    failed_dcc = dcc_df[~dcc_df["convergence"]]

    best_aic = margin_df.dropna(subset=["AIC"]).sort_values("AIC").head(10)
    best_bic = margin_df.dropna(subset=["BIC"]).sort_values("BIC").head(10)

    def residual_pass_count(row: pd.Series) -> int:
        txt = str(row["residual_diagnostics_summary"])
        return sum(flag in txt for flag in ["LB", "LB_sq", "ARCHLM"])

    margin_df2 = margin_df.copy()
    margin_df2["diag_pass_count"] = margin_df2.apply(residual_pass_count, axis=1)
    diag_rank = (
        margin_df2[margin_df2["convergence"]]
        .groupby(["model_order", "distribution"], as_index=False)["diag_pass_count"]
        .mean()
        .sort_values("diag_pass_count", ascending=False)
    )

    dcc_stability = (
        successful_dcc.assign(stable=lambda x: (x["a_plus_b"] < 1.0) & (x["positive_definite_failures"] == 0))
        .sort_values(["stable", "AIC"], ascending=[False, True])
    )

    lines = [
        "# GARCH高阶边际稳健性实验报告",
        "",
        "## 1) 数据与样本信息",
        f"- 输入文件: `{INPUT_PATH.as_posix()}`",
        f"- 日期升序: `{data_info['is_ascending']}`",
        f"- 重复日期数: `{data_info['duplicated_dates']}`",
        f"- 缺失值统计: `{data_info['missing_by_col']}`",
        f"- 最终样本期: `{data_info['sample_start'].date()} ~ {data_info['sample_end'].date()}`",
        f"- 最终观测值数量: `{data_info['n_obs_final']}`",
        "",
        "## 2) 高阶边际模型收敛情况",
        f"- 成功收敛数量: `{len(successful_margins)}`",
        f"- 失败数量: `{len(failed_margins)}`",
    ]

    if not failed_margins.empty:
        lines.extend(["", "### 边际模型失败明细", failed_margins[["variable", "model_order", "distribution", "failure_reason"]].to_markdown(index=False)])

    lines.extend([
        "",
        "## 3) AIC/BIC 相对优势（边际层面）",
        "### AIC 最优前10",
        best_aic[["variable", "model_order", "distribution", "AIC", "BIC"]].to_markdown(index=False) if not best_aic.empty else "- 无",
        "",
        "### BIC 最优前10",
        best_bic[["variable", "model_order", "distribution", "AIC", "BIC"]].to_markdown(index=False) if not best_bic.empty else "- 无",
        "",
        "## 4) 残差诊断表现（均值通过指标数）",
        diag_rank.to_markdown(index=False) if not diag_rank.empty else "- 无",
        "",
        "## 5) DCC 稳定性",
        f"- DCC 收敛数量: `{len(successful_dcc)}`",
        f"- DCC 失败数量: `{len(failed_dcc)}`",
    ])

    if not successful_dcc.empty:
        lines.extend(["", "### DCC 稳定性排序（优先 stable=True 且 AIC 更小）", dcc_stability[["model_order", "distribution", "dcc_a", "dcc_b", "a_plus_b", "positive_definite_failures", "AIC", "BIC", "stable"]].to_markdown(index=False)])
    if not failed_dcc.empty:
        lines.extend(["", "### DCC 失败明细", failed_dcc[["model_order", "distribution", "failure_reason"]].to_markdown(index=False)])

    lines.extend([
        "",
        "## 6) 关键变量对结论变化",
        "- 请结合 `outputs/tables/dcc_key_pairs_by_garch_order.csv` 与图形比较不同阶数下路径差异。",
        f"- 关键变量对记录行数: `{len(pair_df)}`",
        "",
        "## 7) 论文写法建议（自动草案）",
        "- 建议将 `a+b<1` 且 `positive_definite_failures=0`、并在 AIC/BIC 与残差诊断上较优的阶数作为稳健性对照。",
        "- 若某阶数频繁出现收敛失败或正定性问题，可放附录并说明其数值不稳定，不建议进入主文核心结果。",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if IMPORT_ERROR is not None:
        missing_report = LOG_DIR / "garch_order_robustness_report.md"
        missing_report.write_text(
            "# GARCH高阶边际稳健性实验报告\n\n"
            "运行环境缺少必要Python依赖，无法执行估计。\n\n"
            f"- 依赖错误: `{IMPORT_ERROR}`\n"
            "- 请先安装: pandas, numpy, matplotlib, statsmodels, arch, scipy。\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Missing dependencies: {IMPORT_ERROR}")

    df = pd.read_csv(INPUT_PATH)
    data_info = check_data(df)
    data = data_info["cleaned"]
    idx = data["date"]

    margin_results: List[MarginResult] = []
    for order in MODEL_ORDERS:
        for dist in DISTRIBUTIONS:
            for col in RET_COLS:
                res = fit_margin(data[col], col, order, dist)
                margin_results.append(res)

    margin_table = pd.DataFrame(
        [
            {
                "variable": r.variable,
                "model_order": r.model_order,
                "distribution": r.distribution,
                "convergence": r.convergence,
                "loglik": r.loglik,
                "AIC": r.aic,
                "BIC": r.bic,
                "persistence": r.persistence,
                "mean_model": r.mean_model,
                "params_json": r.params_json,
                "pvalues_json": r.pvalues_json,
                "failure_reason": r.failure_reason,
                "residual_diagnostics_summary": r.residual_diagnostics_summary,
            }
            for r in margin_results
        ]
    )

    dcc_results: List[DCCResult] = []
    pair_tables: List[pd.DataFrame] = []

    for order in MODEL_ORDERS:
        for dist in DISTRIBUTIONS:
            panel, issues = build_std_resid_panel(margin_results, order, dist, data.index)
            if panel is None:
                dcc_results.append(
                    DCCResult(
                        model_order=f"({order},{order})",
                        distribution=dist,
                        convergence=False,
                        dcc_a=None,
                        dcc_b=None,
                        a_plus_b=None,
                        loglik=None,
                        aic=None,
                        bic=None,
                        positive_definite_failures=0,
                        failure_reason="; ".join(issues),
                        corr_ts=None,
                    )
                )
                continue

            dcc_res = estimate_dcc(panel, order, dist)
            dcc_results.append(dcc_res)

            if dcc_res.convergence and dcc_res.corr_ts is not None:
                pair_df = build_key_pairs_table([dcc_res], index=data.loc[panel.index, "date"])
                if not pair_df.empty:
                    pair_tables.append(pair_df)

    dcc_table = pd.DataFrame(
        [
            {
                "model_order": r.model_order,
                "distribution": r.distribution,
                "convergence": r.convergence,
                "dcc_a": r.dcc_a,
                "dcc_b": r.dcc_b,
                "a_plus_b": r.a_plus_b,
                "loglik": r.loglik,
                "AIC": r.aic,
                "BIC": r.bic,
                "positive_definite_failures": r.positive_definite_failures,
                "failure_reason": r.failure_reason,
            }
            for r in dcc_results
        ]
    )

    key_pairs = (
        pd.concat(pair_tables, ignore_index=True)
        if pair_tables
        else pd.DataFrame(columns=["date", "model_order", "distribution", *PAIR_MAP.keys()])
    )

    if not key_pairs.empty:
        key_pairs["date"] = pd.to_datetime(key_pairs["date"])
        key_pairs = key_pairs.sort_values(["distribution", "model_order", "date"])
        plot_selected_pairs(key_pairs)

    margin_table.to_csv(TABLE_DIR / "garch_order_comparison_margins.csv", index=False)
    dcc_table.to_csv(TABLE_DIR / "garch_order_comparison_dcc.csv", index=False)

    export_key_pairs = key_pairs[["date", "model_order", "ai_electricity", "ai_coal", "ai_gas", "ai_wti", "ai_gold", "electricity_gas"]].copy() if not key_pairs.empty else pd.DataFrame(columns=["date", "model_order", "ai_electricity", "ai_coal", "ai_gas", "ai_wti", "ai_gold", "electricity_gas"])
    export_key_pairs.to_csv(TABLE_DIR / "dcc_key_pairs_by_garch_order.csv", index=False)

    write_markdown_report(data_info, margin_table, dcc_table, export_key_pairs)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_path = LOG_DIR / "garch_order_robustness_report.md"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            "# GARCH高阶边际稳健性实验报告\n\n"
            "脚本运行失败，请检查依赖与输入数据。\n\n"
            "```\n"
            f"{traceback.format_exc()}"
            "\n```\n",
            encoding="utf-8",
        )
        raise
