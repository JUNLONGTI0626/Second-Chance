import importlib
import subprocess
import sys
from pathlib import Path


def ensure_packages():
    required = ["pandas", "numpy", "scipy", "statsmodels"]
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_packages()

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller


# 用户可替换为 GitHub 原始数据链接，例如：
# https://raw.githubusercontent.com/username/repository/main/data/raw/
base_url = "在这里替换为你的GitHub原始数据链接/"

FILE_MAP = {
    "AI index.csv": "AI",
    "Electricity index.csv": "Electricity",
    "Coal index.csv": "Coal",
    "GAS price.csv": "Gas",
    "WTI price.csv": "WTI",
    "Gold spot price.csv": "Gold",
}

PRICE_CANDIDATES = ["Close", "Price", "Adj Close", "close", "price", "adj close", "AdjClose"]
DATE_CANDIDATES = ["Date", "date", "DATE"]


def _read_csv_flexible(filename: str, local_data_dir: Path) -> pd.DataFrame:
    remote_path = f"{base_url.rstrip('/')}/{filename}" if base_url.strip() else filename
    # 若 base_url 尚未替换，则优先使用本地 data/raw 以保证可执行
    use_local = "在这里替换" in base_url or "raw.githubusercontent.com/username" in base_url

    if use_local:
        path = local_data_dir / filename
        return pd.read_csv(path)

    try:
        return pd.read_csv(remote_path)
    except Exception:
        # 远程读取失败时回退本地文件
        path = local_data_dir / filename
        return pd.read_csv(path)


def identify_date_col(df: pd.DataFrame) -> str:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    raise ValueError("无法识别日期列")


def identify_price_col(df: pd.DataFrame) -> str:
    for c in PRICE_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower().replace("_", " ").strip()
        if lc in {"close", "price", "adj close", "adjclose"}:
            return c
    # 最后尝试：第一个可数值化列
    for c in df.columns:
        if "date" in str(c).lower():
            continue
        sample = pd.to_numeric(df[c], errors="coerce")
        if sample.notna().sum() > 0:
            return c
    raise ValueError("无法识别价格列")


def to_booktabs_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    row_end = r"\\"

    def fmt(val):
        if isinstance(val, (float, np.floating)):
            return f"{val:.4f}"
        return str(val)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(cols) + f" {row_end}",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(row[c]) for c in cols) + f" {row_end}")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parents[1]
    local_data_dir = root / "data" / "raw"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    merged = None
    transformation_log = []

    for filename, var in FILE_MAP.items():
        raw = _read_csv_flexible(filename, local_data_dir)
        date_col = identify_date_col(raw)
        price_col = identify_price_col(raw)

        df = raw[[date_col, price_col]].copy()
        df.columns = ["Date", "Price"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Price"] = (
            df["Price"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        )
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        missing_before = int(df.isna().any(axis=1).sum())
        df = df.dropna().drop_duplicates(subset=["Date"]).sort_values("Date")
        df = df.rename(columns={"Price": var})

        transformation_log.append(
            {
                "Variable": var,
                "Date column": date_col,
                "Price column": price_col,
                "Rows (raw)": len(raw),
                "Rows dropped (missing/invalid)": missing_before,
                "Rows (clean)": len(df),
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)

    returns = merged.copy()
    for v in FILE_MAP.values():
        returns[v] = np.log(returns[v]) - np.log(returns[v].shift(1))
    returns = returns.dropna().reset_index(drop=True)

    returns_out = returns.copy()
    returns_out["Date"] = returns_out["Date"].dt.strftime("%Y-%m-%d")
    returns_out.to_csv(results_dir / "returns.csv", index=False)

    # Table 2
    records = []
    for v in FILE_MAP.values():
        s = returns[v].dropna()
        jb_stat, jb_p = jarque_bera(s)
        records.append(
            {
                "Variable": v,
                "Mean": s.mean(),
                "Std. Dev.": s.std(ddof=1),
                "Skewness": s.skew(),
                "Kurtosis": s.kurtosis(),
                "Jarque-Bera Statistic": jb_stat,
                "Jarque-Bera p-value": jb_p,
            }
        )

    table2 = pd.DataFrame(records)
    table2.to_csv(results_dir / "Table2_Descriptive_Statistics.csv", index=False, float_format="%.4f")
    (results_dir / "Table2_Descriptive_Statistics.tex").write_text(
        to_booktabs_latex(
            table2,
            caption="Descriptive statistics of daily returns",
            label="tab:descriptive_statistics",
        ),
        encoding="utf-8",
    )

    # Table 3
    test_records = []
    for v in FILE_MAP.values():
        s = returns[v].dropna()
        adf_res = adfuller(s, regression="c", autolag="AIC")
        arch_res = het_arch(s - s.mean(), nlags=10)
        test_records.append(
            {
                "Variable": v,
                "ADF Statistic": adf_res[0],
                "ADF p-value": adf_res[1],
                "ARCH-LM Statistic": arch_res[0],
                "ARCH-LM p-value": arch_res[1],
            }
        )

    table3 = pd.DataFrame(test_records)
    table3.to_csv(results_dir / "Table3_ADF_ARCH.csv", index=False, float_format="%.4f")
    (results_dir / "Table3_ADF_ARCH.tex").write_text(
        to_booktabs_latex(
            table3,
            caption="ADF and ARCH-LM test results",
            label="tab:adf_arch",
        ),
        encoding="utf-8",
    )

    # 终端打印
    print("=" * 80)
    print("Transformation log:")
    print(pd.DataFrame(transformation_log).to_string(index=False))
    print("=" * 80)
    print(
        f"Merged sample period: {returns['Date'].min().date()} to {returns['Date'].max().date()} | "
        f"Observations: {len(returns)}"
    )
    print("=" * 80)
    print("Table 2: Descriptive statistics")
    print(table2.round(4).to_string(index=False))
    print("=" * 80)
    print("Table 3: ADF and ARCH-LM")
    print(table3.round(4).to_string(index=False))
    print("=" * 80)
    print("Generated files:")
    for f in [
        "returns.csv",
        "Table2_Descriptive_Statistics.csv",
        "Table2_Descriptive_Statistics.tex",
        "Table3_ADF_ARCH.csv",
        "Table3_ADF_ARCH.tex",
    ]:
        p = results_dir / f
        print(f" - {p} | exists={p.exists()}")


if __name__ == "__main__":
    main()
