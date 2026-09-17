import json
from llm.llm_client import LLMClient

METHOD_PARAM_SPEC = {
    "descriptive_stats": {},
    "correlation_analysis": {
        "cols": "list of at least 2 numeric column names to check correlation between"
    },
    "attribute_ranking_regression": {
        "target": "single numeric column name that is the outcome to explain",
        "features": "list of column names (numeric or categorical) that might explain the target"
    },
    "segmentation": {
        "features": "list of numeric column names to cluster records on"
    },
    "anomaly_detection": {
        "col": "single numeric column name to check for outliers"
    },
    "hypothesis_testing": {
        "group_col": "single categorical column name whose groups are being compared",
        "value_col": "single numeric column name being measured/compared across groups"
    },
}

ALL_COLUMNS_PHRASES = ["all column", "all attribute", "all feature", "every column", "every attribute"]

def _detect_explicit_target(question: str, numeric_cols: list) -> str:
    q_lower = question.lower()
    for col in numeric_cols:
        if col.lower() in q_lower:
            return col
    return None

def _fallback_value(param: str, numeric_cols: list, categorical_cols: list, exclude=None):
    exclude = exclude or []
    available_numeric = [c for c in numeric_cols if c not in exclude]
    available_categorical = [c for c in categorical_cols if c not in exclude]

    if param == "cols":
        return available_numeric[:3] if len(available_numeric) >= 2 else None
    if param == "target":
        return available_numeric[0] if available_numeric else None
    if param == "features":
        pool = available_numeric + available_categorical
        return pool[:3] if pool else None
    if param == "col":
        return available_numeric[0] if available_numeric else None
    if param == "group_col":
        return available_categorical[0] if available_categorical else None
    if param == "value_col":
        return available_numeric[0] if available_numeric else None
    return None

def select_columns(method: str, question: str, df, llm: LLMClient) -> dict:
    spec = METHOD_PARAM_SPEC.get(method, {})
    if not spec:
        return {}

    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    all_cols = set(df.columns)

    if method == "attribute_ranking_regression":
        q_lower = question.lower()
        explicit_target = _detect_explicit_target(question, numeric_cols)
        wants_all_columns = any(phrase in q_lower for phrase in ALL_COLUMNS_PHRASES)

        usable_cols = [c for c in df.columns if df[c].nunique() < 0.9 * len(df)]

        if explicit_target and wants_all_columns:
            features = [c for c in usable_cols if c != explicit_target]
            return {"target": explicit_target, "features": features}
        elif explicit_target:
            features = [c for c in usable_cols if c != explicit_target][:5]
            return {"target": explicit_target, "features": features}

    prompt = f"""Business question: "{question}"

Available columns:
- Numeric columns: {numeric_cols}
- Categorical/text columns: {categorical_cols}

For the analysis method "{method}", provide values for EXACTLY these parameter names (use these exact keys, nothing else):
{json.dumps(spec, indent=2)}

Respond ONLY with a JSON object using the exact parameter names shown above as keys. Use exact column names from the lists above as values. No other text, no markdown formatting."""

    raw = llm.generate(
        prompt=prompt,
        system="You are a data analyst assistant. Use the exact parameter names given. Only use column names that exist in the provided lists, spelled exactly as given."
    )
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        selected = json.loads(cleaned)
        if not isinstance(selected, dict):
            selected = {}
    except json.JSONDecodeError:
        print(f"[column_selector] JSON parse failed for method '{method}'. Raw output:\n{raw}")
        selected = {}

    final = {}
    used_cols = []
    for param in spec.keys():
        value = selected.get(param)

        if isinstance(value, list):
            valid_list = [v for v in value if v in all_cols]
            if valid_list:
                final[param] = valid_list
                used_cols.extend(valid_list)
                continue
        elif isinstance(value, str) and value in all_cols:
            final[param] = value
            used_cols.append(value)
            continue

        fallback = _fallback_value(param, numeric_cols, categorical_cols, exclude=used_cols)
        if fallback is not None:
            print(f"[column_selector] Falling back for '{param}' in method '{method}': using {fallback}")
            final[param] = fallback
            if isinstance(fallback, list):
                used_cols.extend(fallback)
            else:
                used_cols.append(fallback)

    return final