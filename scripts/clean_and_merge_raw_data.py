import csv
from datetime import datetime
from pathlib import Path
from collections import Counter

BASE = Path('.')
RAW_DIR = BASE / 'data' / 'raw'
PROCESSED_PATH = BASE / 'data' / 'processed' / 'merged_levels.csv'
REPORT_PATH = BASE / 'outputs' / 'logs' / 'data_cleaning_report.md'

FILES = {
    'ai': RAW_DIR / 'AI index.csv',
    'electricity': RAW_DIR / 'Electricity index.csv',
    'coal': RAW_DIR / 'Coal index.csv',
    'gas': RAW_DIR / 'GAS price.csv',
    'wti': RAW_DIR / 'WTI price.csv',
    'gold': RAW_DIR / 'Gold spot price.csv',
}

DATE_FORMATS = [
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%Y/%m/%d',
    '%d/%m/%Y',
]


def parse_date(value: str) -> str:
    s = (value or '').strip().replace('\ufeff', '')
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError(f'Unsupported date format: {value!r}')


def parse_number(value: str):
    s = (value or '').strip().replace(',', '')
    if s == '':
        return None
    return float(s)


def load_series(path: Path, var_name: str):
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if var_name == 'gold':
            # Keep only Date and Price; drop blank/useless columns.
            for r in reader:
                rows.append((parse_date(r.get('Date', '')), parse_number(r.get('Price', ''))))
        else:
            val_col = [c for c in reader.fieldnames if c != 'observation_date'][0]
            for r in reader:
                rows.append((parse_date(r.get('observation_date', '')), parse_number(r.get(val_col, ''))))

    date_counts = Counter(d for d, _ in rows)
    dup_dates = sorted([d for d, c in date_counts.items() if c > 1])

    deduped = {}
    for d, v in rows:
        deduped[d] = v

    missing_vals = sum(1 for _, v in rows if v is None)

    sorted_rows = sorted(deduped.items(), key=lambda x: x[0])
    return {
        'var': var_name,
        'rows_raw': len(rows),
        'rows_dedup': len(sorted_rows),
        'dup_date_count': len(dup_dates),
        'dup_dates': dup_dates,
        'missing_values': missing_vals,
        'data': dict(sorted_rows),
        'min_date': sorted_rows[0][0] if sorted_rows else None,
        'max_date': sorted_rows[-1][0] if sorted_rows else None,
    }


def main():
    series_info = {k: load_series(p, k) for k, p in FILES.items()}

    common_dates = set(series_info['ai']['data'].keys())
    for k in ['electricity', 'coal', 'gas', 'wti', 'gold']:
        common_dates &= set(series_info[k]['data'].keys())
    common_dates = sorted(common_dates)

    columns = ['date', 'ai', 'electricity', 'coal', 'gas', 'wti', 'gold']
    merged_rows = []
    for d in common_dates:
        merged_rows.append({
            'date': d,
            'ai': series_info['ai']['data'][d],
            'electricity': series_info['electricity']['data'][d],
            'coal': series_info['coal']['data'][d],
            'gas': series_info['gas']['data'][d],
            'wti': series_info['wti']['data'][d],
            'gold': series_info['gold']['data'][d],
        })

    merged_dup_dates = len(merged_rows) - len({r['date'] for r in merged_rows})
    merged_missing_by_col = {
        c: sum(1 for r in merged_rows if r[c] is None)
        for c in columns if c != 'date'
    }
    complete_obs = sum(1 for r in merged_rows if all(r[c] is not None for c in columns if c != 'date'))

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_PATH.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in merged_rows:
            out = {'date': r['date']}
            for c in columns[1:]:
                out[c] = '' if r[c] is None else f"{r[c]:.6f}".rstrip('0').rstrip('.')
            writer.writerow(out)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open('w', encoding='utf-8') as f:
        f.write('# Data Cleaning Report\n\n')
        f.write('## 1) Source files and cleaning actions\n')
        f.write('- Standardized column names to: `date, ai, electricity, coal, gas, wti, gold`.\n')
        f.write('- Removed blank/useless columns from the gold source file and kept only `Date` + `Price`.\n')
        f.write('- Converted comma-formatted price strings (e.g., `"1,517.30"`) to numeric values.\n')
        f.write('- Parsed all date fields into ISO format `YYYY-MM-DD`.\n')
        f.write('- Sorted all series by date in ascending order before merging.\n\n')

        f.write('## 2) Per-file diagnostics (before/after de-duplication)\n')
        f.write('| variable | raw rows | rows after date de-dup | duplicate dates | missing values | min date | max date |\n')
        f.write('|---|---:|---:|---:|---:|---|---|\n')
        for k in ['ai', 'electricity', 'coal', 'gas', 'wti', 'gold']:
            info = series_info[k]
            f.write(
                f"| {k} | {info['rows_raw']} | {info['rows_dedup']} | {info['dup_date_count']} | {info['missing_values']} | {info['min_date']} | {info['max_date']} |\n"
            )
        f.write('\n')

        f.write('## 3) Merged dataset checks\n')
        f.write(f'- Merge rule: **inner join on date** across all six variables.\n')
        f.write(f'- Output file: `{PROCESSED_PATH.as_posix()}`.\n')
        f.write(f'- Total merged rows: **{len(merged_rows)}**.\n')
        f.write(f'- Duplicate dates in merged data: **{merged_dup_dates}**.\n')
        f.write('- Missing values by variable in merged data:\n')
        for c in ['ai', 'electricity', 'coal', 'gas', 'wti', 'gold']:
            f.write(f'  - `{c}`: {merged_missing_by_col[c]}\n')
        f.write('\n')

        if merged_rows:
            f.write('## 4) Final common sample\n')
            f.write(f'- Common sample period: **{merged_rows[0]["date"]} to {merged_rows[-1]["date"]}**.\n')
        else:
            f.write('## 4) Final common sample\n')
            f.write('- Common sample period: **N/A (no overlapping dates)**.\n')
        f.write(f'- Number of complete usable observations: **{complete_obs}**.\n')


if __name__ == '__main__':
    main()
