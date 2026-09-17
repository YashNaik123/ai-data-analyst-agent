import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> dict:
    report = []
    cleaned = df.copy()

    original_cols = cleaned.columns.tolist()
    cleaned.columns = [c.strip() for c in cleaned.columns]
    if original_cols != cleaned.columns.tolist():
        report.append("Trimmed whitespace from column names")

    for col in cleaned.select_dtypes(include="object").columns:
        before = cleaned[col].copy()
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        if not before.equals(cleaned[col]):
            report.append(f"Trimmed whitespace / normalized blanks in '{col}'")

    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates()
        report.append(f"Removed {dup_count} exact duplicate row(s)")

    for col in cleaned.select_dtypes(include="object").columns:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        if converted.notna().sum() >= 0.9 * cleaned[col].notna().sum() and converted.notna().sum() > 0:
            cleaned[col] = converted
            report.append(f"Converted '{col}' from text to numeric")

    for col in cleaned.select_dtypes(include="object").columns:
        try:
            converted = pd.to_datetime(cleaned[col], errors="coerce")
            if converted.notna().sum() >= 0.9 * cleaned[col].notna().sum() and converted.notna().sum() > 0:
                cleaned[col] = converted
                report.append(f"Converted '{col}' from text to datetime")
        except Exception:
            pass

    missing_summary = cleaned.isna().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = round(count / len(cleaned) * 100, 1)
            report.append(f"'{col}' has {count} missing value(s) ({pct}%) — left as-is; each analysis method handles this appropriately")

    outlier_summary = {}
    for col in cleaned.select_dtypes(include="number").columns:
        q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
        iqr = q3 - q1
        outliers = cleaned[(cleaned[col] < q1 - 1.5 * iqr) | (cleaned[col] > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            outlier_summary[col] = len(outliers)
            report.append(f"'{col}' has {len(outliers)} statistical outlier(s) — flagged, not removed")

    return {
        "cleaned_df": cleaned,
        "report": report,
        "rows_before": len(df),
        "rows_after": len(cleaned),
        "outlier_summary": outlier_summary,
    }