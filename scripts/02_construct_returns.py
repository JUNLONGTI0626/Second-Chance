import csv
import math
from datetime import datetime
from pathlib import Path

INPUT_PATH = Path('data/processed/merged_levels.csv')
OUTPUT_PATH = Path('data/processed/merged_returns.csv')
REPORT_PATH = Path('outputs/logs/returns_construction_report.md')

REQUIRED_COLUMNS = ['date', 'ai', 'electricity', 'coal', 'gas', 'wti', 'gold']
PRICE_COLUMNS = ['ai', 'electricity', 'coal', 'gas', 'wti', 'gold']
RETURN_COLUMNS = ['r_ai', 'r_electricity', 'r_coal', 'r_gas', 'r_wti', 'r_gold']

DATE_FORMATS = ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']


def parse_date(value: str) -> str:
    text = (value or '').strip().replace('\ufeff', '')
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError(f'Unsupported date format: {value!r}')


def parse_float(value: str):
    text = (value or '').strip().replace(',', '')
    if text == '':
        return None
    return float(text)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Input file not found: {INPUT_PATH}')

    rows = []
    anomalies = []

    with INPUT_PATH.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')

        for idx, row in enumerate(reader, start=2):
            date_text = row.get('date', '')
            try:
                date_iso = parse_date(date_text)
            except ValueError as exc:
                anomalies.append(f'Line {idx}: invalid date `{date_text}` ({exc})')
                continue

            parsed = {'date': date_iso}
            for col in PRICE_COLUMNS:
                try:
                    parsed[col] = parse_float(row.get(col, ''))
                except ValueError:
                    anomalies.append(f'Line {idx}: non-numeric value in `{col}` -> `{row.get(col, "")}`')
                    parsed[col] = None
            rows.append(parsed)

    if not rows:
        raise ValueError('No valid rows available after parsing.')

    rows.sort(key=lambda x: x['date'])

    date_counts = {}
    for r in rows:
        date_counts[r['date']] = date_counts.get(r['date'], 0) + 1
    duplicate_dates = sorted([d for d, c in date_counts.items() if c > 1])

    # Keep the last occurrence for duplicate dates (if any)
    deduped_map = {}
    for r in rows:
        deduped_map[r['date']] = r
    levels_rows = [deduped_map[d] for d in sorted(deduped_map)]

    level_missing_counts = {
        col: sum(1 for r in levels_rows if r[col] is None)
        for col in PRICE_COLUMNS
    }

    # Construct log returns: 100 * ln(P_t / P_(t-1))
    return_rows = []
    invalid_return_points = []
    prev = None
    for curr in levels_rows:
        out = {'date': curr['date']}
        for p_col, r_col in zip(PRICE_COLUMNS, RETURN_COLUMNS):
            ret = None
            if prev is not None:
                p_t = curr[p_col]
                p_tm1 = prev[p_col]
                if p_t is None or p_tm1 is None:
                    ret = None
                elif p_t <= 0 or p_tm1 <= 0:
                    invalid_return_points.append(
                        f"{curr['date']} `{p_col}` has non-positive price (P_t={p_t}, P_t-1={p_tm1})"
                    )
                else:
                    ret = 100.0 * math.log(p_t / p_tm1)
            out[r_col] = ret
        return_rows.append(out)
        prev = curr

    # Remove first row (mechanical NA from lag operation)
    if return_rows:
        return_rows = return_rows[1:]

    # Remove rows with any missing return to ensure complete model input
    complete_return_rows = [
        r for r in return_rows
        if all(r[c] is not None for c in RETURN_COLUMNS)
    ]

    return_missing_counts_before_drop = {
        col: sum(1 for r in return_rows if r[col] is None)
        for col in RETURN_COLUMNS
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date'] + RETURN_COLUMNS)
        writer.writeheader()
        for r in complete_return_rows:
            row_out = {'date': r['date']}
            for col in RETURN_COLUMNS:
                row_out[col] = f"{r[col]:.10f}" if r[col] is not None else ''
            writer.writerow(row_out)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open('w', encoding='utf-8') as f:
        f.write('# Returns Construction Report\n\n')
        f.write('## 1) Input checks\n')
        f.write(f'- Input file: `{INPUT_PATH.as_posix()}`\n')
        f.write(f'- Required columns present: **yes** ({", ".join(REQUIRED_COLUMNS)})\n')
        f.write(f'- Duplicate dates found in input: **{"yes" if duplicate_dates else "no"}**\n')
        f.write(f'- Number of duplicate dates: **{len(duplicate_dates)}**\n')
        if duplicate_dates:
            shown = ', '.join(duplicate_dates[:10])
            more = '' if len(duplicate_dates) <= 10 else f' ... (+{len(duplicate_dates)-10} more)'
            f.write(f'- Duplicate date examples: {shown}{more}\n')
        f.write('\n')

        f.write('## 2) Sample periods\n')
        f.write(f'- `merged_levels.csv` sample period: **{levels_rows[0]["date"]} to {levels_rows[-1]["date"]}**\n')
        if complete_return_rows:
            f.write(
                f'- `merged_returns.csv` final sample period: **{complete_return_rows[0]["date"]} to {complete_return_rows[-1]["date"]}**\n'
            )
        else:
            f.write('- `merged_returns.csv` final sample period: **N/A**\n')
        f.write('\n')

        f.write('## 3) Missing-value diagnostics\n')
        f.write('- Missing values in level data (`merged_levels.csv`):\n')
        for col in PRICE_COLUMNS:
            f.write(f'  - `{col}`: {level_missing_counts[col]}\n')
        f.write('- Missing values in return data before final completeness filter:\n')
        for col in RETURN_COLUMNS:
            f.write(f'  - `{col}`: {return_missing_counts_before_drop[col]}\n')
        f.write('\n')

        f.write('## 4) Return construction\n')
        f.write('- Formula used for all variables:\n')
        f.write('  - `return_t = 100 * ln(P_t / P_(t-1))`\n')
        f.write('- Return columns: `r_ai, r_electricity, r_coal, r_gas, r_wti, r_gold`\n')
        f.write('- First row after lag operation is removed as required.\n')
        f.write('- Rows with any missing return are removed to keep complete model input.\n\n')

        f.write('## 5) Final output summary\n')
        f.write(f'- Output file: `{OUTPUT_PATH.as_posix()}`\n')
        f.write(f'- Final complete observations: **{len(complete_return_rows)}**\n')
        f.write(f'- Parsed/format anomalies: **{len(anomalies) + len(invalid_return_points)}**\n')

        if anomalies or invalid_return_points:
            f.write('\n## 6) Anomalies and format issues\n')
            if anomalies:
                f.write('- Parsing issues:\n')
                for item in anomalies[:30]:
                    f.write(f'  - {item}\n')
                if len(anomalies) > 30:
                    f.write(f'  - ... (+{len(anomalies)-30} more)\n')
            if invalid_return_points:
                f.write('- Non-positive prices detected during return computation:\n')
                for item in invalid_return_points[:30]:
                    f.write(f'  - {item}\n')
                if len(invalid_return_points) > 30:
                    f.write(f'  - ... (+{len(invalid_return_points)-30} more)\n')
        else:
            f.write('\n## 6) Anomalies and format issues\n')
            f.write('- No additional anomalies detected.\n')


if __name__ == '__main__':
    main()
