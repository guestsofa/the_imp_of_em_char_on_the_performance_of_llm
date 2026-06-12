#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Унифицированный скрипт регрессионного анализа для задачи: Шахматы.

Модель сохраняет логику предыдущего анализа:
- основная независимая переменная: prompt_emotion_type;
- дополнительная независимая переменная: job_polarity_clear;
- контроль: фиксированные эффекты задачи C(question_id);
- робастные стандартные ошибки HC3.

Дополнительное правило очистки:
- из регрессии полностью исключаются задачи/question_id, в которых все значения
  зависимой переменной равны максимальному возможному значению.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

TASK_NAME = 'Шахматы'
DEPENDENT_VAR = 'judge_solution_score_0_10'
MODEL_KIND = 'ols'  # "logit" или "ols"
DEFAULT_STEM = 'chess'
DEFAULT_SHEET = None
MIN_ROWS_FOR_REGRESSION = 30

# Максимум зависимой переменной. Если внутри question_id все значения равны этому максимуму,
# вся задача исключается из регрессии.
MAX_DEPENDENT_VALUE = 10
EXCLUDE_QUESTIONS_WITH_ALL_MAX_SCORES = True

# Столбцы, которые именно для этой таблицы обязательно должны быть в файле.
REQUIRED_COLUMNS = ['judge_solution_score_0_10']

# Если соответствующий столбец есть в таблице, строки фильтруются по этим значениям.
# None означает, что фильтр для этого статуса не применяется.
ACCEPTED_STATUS_VALUES = ['success']
ACCEPTED_JUDGE_STATUS_VALUES = ['success', 'None']

# Для шахмат question_id восстанавливается из condition_id / condition_image_path.
# Для остальных задач используется явный question_id.
ALLOW_DERIVE_QUESTION_ID = True

PROMPT_LEVELS = [
    "None",
    "happiness+obvious", "happiness+hidden",
    "wonder+obvious", "wonder+hidden",
    "fear+obvious", "fear+hidden",
    "anger+obvious", "anger+hidden",
    "disgust+obvious", "disgust+hidden",
]

JOB_LEVELS = ["None", "positive", "negative"]


def load_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Читает CSV/XLSX. Для CSV есть мягкий fallback на случай многострочного текста."""
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        if sheet:
            return pd.read_excel(path, sheet_name=sheet)
        excel = pd.ExcelFile(path)
        return pd.read_excel(path, sheet_name=excel.sheet_names[0])

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception as first_error:
            try:
                return pd.read_csv(
                    path,
                    engine="python",
                    on_bad_lines="skip",
                    quoting=csv.QUOTE_MINIMAL,
                    keep_default_na=True,
                )
            except Exception as second_error:
                raise ValueError(
                    "Не удалось прочитать CSV. Возможно, в файле нарушены кавычки "
                    "или есть проблемный многострочный текст. Лучше пересохранить файл как UTF-8 CSV или XLSX.\n"
                    f"Первая ошибка: {first_error}\n"
                    f"Fallback-ошибка: {second_error}"
                ) from second_error

    raise ValueError(f"Неподдерживаемый тип файла: {path.suffix}")


def normalize_string_col(series: pd.Series, none_label: str = "None") -> pd.Series:
    """Приводит строковую переменную к единому виду и заменяет пропуски на None."""
    s = series.copy().astype("object")
    s = s.where(~pd.isna(s), none_label)
    s = s.astype(str).str.strip()
    s = s.replace({
        "": none_label,
        "nan": none_label,
        "NaN": none_label,
        "NoneType": none_label,
        "<NA>": none_label,
    })
    return s


def first_available_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_prompt_emotion_type(df: pd.DataFrame) -> pd.Series:
    """
    Берет готовый prompt_emotion_type, если он есть.
    Иначе строит его из emotion + emotion_visibility.
    """
    if "prompt_emotion_type" in df.columns:
        return normalize_string_col(df["prompt_emotion_type"])

    missing = [c for c in ["emotion", "emotion_visibility"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Нельзя построить prompt_emotion_type: нет готового столбца prompt_emotion_type "
            f"и не хватает столбцов {missing}."
        )

    emotion = normalize_string_col(df["emotion"])
    visibility = normalize_string_col(df["emotion_visibility"])
    return pd.Series(
        np.where(emotion.eq("None"), "None", emotion + "+" + visibility),
        index=df.index,
        dtype="object",
    )


def build_job_polarity_clear(df: pd.DataFrame) -> pd.Series:
    """Берет job_polarity_clear, если есть; иначе строит из job_polarity."""
    col = first_available_column(df, ["job_polarity_clear", "job_polarity"])
    if col is None:
        raise ValueError("В таблице нет ни job_polarity_clear, ни job_polarity.")
    return normalize_string_col(df[col])


def derive_question_id(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает question_id.
    Для обычных таблиц используется явный question_id.
    Для шахмат разрешен fallback: condition_id -> condition_image_path.
    """
    if "question_id" in df.columns:
        q = pd.to_numeric(df["question_id"], errors="coerce")
        if q.notna().any():
            return q
        if not ALLOW_DERIVE_QUESTION_ID:
            return q

    if not ALLOW_DERIVE_QUESTION_ID:
        raise ValueError("В таблице нет usable question_id, а восстановление question_id для этого типа задач отключено.")

    if "condition_id" in df.columns:
        q = pd.to_numeric(df["condition_id"], errors="coerce")
        if q.notna().any():
            return q

    if "condition_image_path" in df.columns:
        names = (
            df["condition_image_path"]
            .astype("object")
            .where(~pd.isna(df["condition_image_path"]), "None")
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.split("/")
            .str[-1]
            .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        )
        codes, _ = pd.factorize(names, sort=True)
        q = pd.Series(codes + 1, index=df.index, dtype="float")
        q = q.where(names.ne("None"), np.nan)
        return q

    return pd.Series(np.nan, index=df.index, dtype="float")


def ordered_categories(values: pd.Series, preferred_order: list[str]) -> list[str]:
    """Сохраняет базовый порядок категорий и добавляет неожиданные уровни в конец."""
    observed = list(pd.Series(values).dropna().astype(str).unique())
    present_preferred = [x for x in preferred_order if x in observed]
    remaining = sorted([x for x in observed if x not in present_preferred])
    return present_preferred + remaining


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В таблице для {TASK_NAME} не хватает обязательных столбцов: {missing}")


def apply_optional_status_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует status/judge_status только если соответствующий столбец есть и фильтр задан."""
    out = df.copy()

    if ACCEPTED_STATUS_VALUES is not None and "status" in out.columns:
        out["status"] = normalize_string_col(out["status"])
        out = out[out["status"].isin(ACCEPTED_STATUS_VALUES)].copy()

    if ACCEPTED_JUDGE_STATUS_VALUES is not None and "judge_status" in out.columns:
        out["judge_status"] = normalize_string_col(out["judge_status"])
        out = out[out["judge_status"].isin(ACCEPTED_JUDGE_STATUS_VALUES)].copy()

    return out


def find_questions_with_all_max_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит задачи, в которых все значения зависимой переменной равны максимуму.
    Возвращает таблицу question_id, n, min, max, mean.
    """
    if df.empty:
        return pd.DataFrame(columns=["question_id", "n", "min", "max", "mean"])

    summary = (
        df.groupby("question_id", observed=False)[DEPENDENT_VAR]
        .agg(n="size", min="min", max="max", mean="mean")
        .reset_index()
    )
    excluded = summary[(summary["min"].eq(MAX_DEPENDENT_VALUE)) & (summary["max"].eq(MAX_DEPENDENT_VALUE))].copy()
    return excluded.sort_values("question_id")


def exclude_questions_with_all_max_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Исключает из данных целые question_id, где все оценки максимальные."""
    excluded = find_questions_with_all_max_scores(df)
    if excluded.empty:
        return df.copy(), excluded

    excluded_ids = set(excluded["question_id"].tolist())
    filtered = df[~df["question_id"].isin(excluded_ids)].copy()
    return filtered, excluded


def question_id_source(df: pd.DataFrame) -> str:
    if "question_id" in df.columns and pd.to_numeric(df["question_id"], errors="coerce").notna().any():
        return "question_id"
    if ALLOW_DERIVE_QUESTION_ID and "condition_id" in df.columns and pd.to_numeric(df["condition_id"], errors="coerce").notna().any():
        return "condition_id"
    if ALLOW_DERIVE_QUESTION_ID and "condition_image_path" in df.columns:
        return "condition_image_path"
    return "missing"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    check_required_columns(df)

    out = df.copy()
    rows_original = len(out)

    out["prompt_emotion_type"] = build_prompt_emotion_type(out)
    out["job_polarity_clear"] = build_job_polarity_clear(out)
    out["question_id"] = derive_question_id(out)
    out[DEPENDENT_VAR] = pd.to_numeric(out[DEPENDENT_VAR], errors="coerce")

    out = apply_optional_status_filters(out)
    rows_after_status_filters = len(out)

    out = out[out[DEPENDENT_VAR].notna()].copy()
    out = out[out["question_id"].notna()].copy()

    if MODEL_KIND == "logit":
        out = out[out[DEPENDENT_VAR].isin([0, 1])].copy()
        out[DEPENDENT_VAR] = out[DEPENDENT_VAR].astype(int)

    out["question_id"] = pd.to_numeric(out["question_id"], errors="coerce").astype(int)
    rows_before_all_max_exclusion = len(out)

    if EXCLUDE_QUESTIONS_WITH_ALL_MAX_SCORES:
        out, excluded_all_max = exclude_questions_with_all_max_scores(out)
    else:
        excluded_all_max = pd.DataFrame(columns=["question_id", "n", "min", "max", "mean"])

    rows_removed_all_max = rows_before_all_max_exclusion - len(out)

    out["question_id"] = out["question_id"].astype("category")

    prompt_categories = ordered_categories(out["prompt_emotion_type"], PROMPT_LEVELS)
    job_categories = ordered_categories(out["job_polarity_clear"], JOB_LEVELS)

    out["prompt_emotion_type"] = pd.Categorical(out["prompt_emotion_type"], categories=prompt_categories)
    out["job_polarity_clear"] = pd.Categorical(out["job_polarity_clear"], categories=job_categories)

    metadata = {
        "rows_original": rows_original,
        "rows_after_status_filters": rows_after_status_filters,
        "rows_before_all_max_exclusion": rows_before_all_max_exclusion,
        "rows_removed_all_max": rows_removed_all_max,
        "rows_prepared": len(out),
        "n_questions_excluded_all_max": len(excluded_all_max),
        "excluded_all_max_question_ids": [str(x) for x in excluded_all_max.get("question_id", pd.Series(dtype="object")).tolist()],
        "excluded_all_max_summary": excluded_all_max,
        "prompt_categories": prompt_categories,
        "job_categories": job_categories,
        "question_id_source": question_id_source(df),
    }
    return out, metadata


def precheck(df: pd.DataFrame) -> list[str]:
    messages: list[str] = []

    if len(df) < MIN_ROWS_FOR_REGRESSION:
        messages.append(
            f"После очистки осталось только {len(df)} строк. "
            "Для устойчивой регрессии этого мало."
        )

    if df[DEPENDENT_VAR].nunique(dropna=True) < 2:
        messages.append(f"{DEPENDENT_VAR} имеет меньше двух уникальных значений после фильтрации.")

    if df["question_id"].nunique(dropna=True) < 2:
        messages.append("После фильтрации остался только один question_id; fixed effects по задачам не оценить.")

    if df["prompt_emotion_type"].nunique(dropna=True) < 2:
        messages.append("После фильтрации остался только один уровень prompt_emotion_type.")

    if df["job_polarity_clear"].nunique(dropna=True) < 2:
        messages.append("После фильтрации остался только один уровень job_polarity_clear.")

    if MODEL_KIND == "logit" and not set(df[DEPENDENT_VAR].dropna().unique()).issubset({0, 1}):
        messages.append(f"Для логистической регрессии {DEPENDENT_VAR} должна принимать только 0/1.")

    return messages


def regression_formula() -> str:
    return (
        f"{DEPENDENT_VAR} ~ "
        "C(prompt_emotion_type, Treatment(reference='None')) + "
        "C(job_polarity_clear, Treatment(reference='None')) + "
        "C(question_id)"
    )


def fit_model(df: pd.DataFrame):
    formula = regression_formula()

    if MODEL_KIND == "logit":
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC3")
    elif MODEL_KIND == "ols":
        model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    else:
        raise ValueError(f"Неизвестный MODEL_KIND: {MODEL_KIND}")

    return formula, model


def to_scalar(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def save_cleaned_data(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    priority_cols = [
        "timestamp", "model", "run_block",
        "emotion", "emotion_visibility", "job_polarity", "job_polarity_clear", "prompt_emotion_type",
        "condition", "condition_id", "question_id", "complexity",
        "condition_image_path", "answer_image_path",
        "right_answer", "answer", "answer_match",
        DEPENDENT_VAR,
        "judge_condition_recognition_0_1", "judge_error_type", "judge_error_comment",
        "judge_penalty_reason_categories", "judge_penalty_reason_categories_joined", "judge_penalty_reason_short",
        "status", "judge_status", "error_message", "judge_error_message",
        "response_text",
    ]
    keep_cols = []
    for col in priority_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    # Добавляем остальные столбцы в конец, чтобы ничего не терялось.
    for col in df.columns:
        if col not in keep_cols:
            keep_cols.append(col)

    df[keep_cols].to_csv(outdir / f"{stem}_cleaned_for_regression.csv", index=False, encoding="utf-8-sig")


def save_excluded_all_max_questions(metadata: dict[str, Any], outdir: Path, stem: str) -> None:
    excluded = metadata.get("excluded_all_max_summary")
    if isinstance(excluded, pd.DataFrame) and not excluded.empty:
        excluded.to_csv(outdir / f"{stem}_excluded_all_max_questions.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["question_id", "n", "min", "max", "mean"]).to_csv(
            outdir / f"{stem}_excluded_all_max_questions.csv", index=False, encoding="utf-8-sig"
        )


def save_descriptives(df: pd.DataFrame, metadata: dict[str, Any], outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    save_cleaned_data(df, outdir, stem)
    save_excluded_all_max_questions(metadata, outdir, stem)

    (df["prompt_emotion_type"].value_counts(dropna=False)
        .rename_axis("prompt_emotion_type")
        .reset_index(name="n")
        .to_csv(outdir / f"{stem}_prompt_emotion_counts.csv", index=False, encoding="utf-8-sig"))

    (df["job_polarity_clear"].value_counts(dropna=False)
        .rename_axis("job_polarity_clear")
        .reset_index(name="n")
        .to_csv(outdir / f"{stem}_job_polarity_counts.csv", index=False, encoding="utf-8-sig"))

    summary = (
        df.groupby("question_id", observed=False)[DEPENDENT_VAR]
        .agg(n="size", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values("question_id")
    )
    if MODEL_KIND == "logit":
        summary = summary.rename(columns={"mean": "mean_accuracy", "n": "n"})
        summary["correct"] = df.groupby("question_id", observed=False)[DEPENDENT_VAR].sum().values
    else:
        summary = summary.rename(columns={"mean": "mean_score"})
    summary.to_csv(outdir / f"{stem}_question_summary.csv", index=False, encoding="utf-8-sig")


def save_precheck(
    original_path: Path,
    prepared: pd.DataFrame,
    metadata: dict[str, Any],
    messages: list[str],
    outdir: Path,
    stem: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{stem}_precheck.txt", "w", encoding="utf-8") as f:
        f.write(f"Task: {TASK_NAME}\n")
        f.write(f"Input file: {original_path}\n")
        f.write(f"Model kind: {MODEL_KIND}\n")
        f.write(f"Dependent variable: {DEPENDENT_VAR}\n")
        f.write(f"Maximum dependent value used for all-max exclusion: {MAX_DEPENDENT_VALUE}\n")
        f.write(f"Rows in original table: {metadata['rows_original']}\n")
        f.write(f"Rows after optional status filters: {metadata['rows_after_status_filters']}\n")
        f.write(f"Rows before all-max question exclusion: {metadata['rows_before_all_max_exclusion']}\n")
        f.write(f"Questions excluded because all scores were maximum: {metadata['n_questions_excluded_all_max']}\n")
        f.write(f"Excluded all-max question_id values: {', '.join(metadata['excluded_all_max_question_ids']) if metadata['excluded_all_max_question_ids'] else 'none'}\n")
        f.write(f"Rows removed by all-max question exclusion: {metadata['rows_removed_all_max']}\n")
        f.write(f"Rows prepared for regression: {metadata['rows_prepared']}\n")
        f.write(f"Unique question_id after filtering: {prepared['question_id'].nunique(dropna=True)}\n")
        f.write(f"question_id source used: {metadata['question_id_source']}\n")
        f.write(f"Unique {DEPENDENT_VAR} values after filtering: {prepared[DEPENDENT_VAR].nunique(dropna=True)}\n")
        f.write(f"prompt_emotion_type levels: {', '.join(map(str, metadata['prompt_categories']))}\n")
        f.write(f"job_polarity_clear levels: {', '.join(map(str, metadata['job_categories']))}\n\n")

        if messages:
            f.write("Regression was not estimated because:\n")
            for msg in messages:
                f.write(f"- {msg}\n")
        else:
            f.write("Precheck passed. Regression can be estimated.\n")


def save_regression_results(formula: str, model, outdir: Path, stem: str) -> None:
    conf = model.conf_int()

    stat_col = "z_value" if MODEL_KIND == "logit" else "t_value"
    coef_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err_hc3": model.bse.values,
        stat_col: model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_lower": conf[0].values,
        "ci_upper": conf[1].values,
    })

    if MODEL_KIND == "logit":
        coef_df["odds_ratio"] = np.exp(coef_df["coef"])
        coef_df["or_ci_lower"] = np.exp(coef_df["ci_lower"])
        coef_df["or_ci_upper"] = np.exp(coef_df["ci_upper"])

    coef_df.to_csv(outdir / f"{stem}_coefficients.csv", index=False, encoding="utf-8-sig")

    wt = model.wald_test_terms(skip_single=False, scalar=True)
    rows = []
    for term, res in wt.table.iterrows():
        row = {
            "term": term,
            "wald_statistic": to_scalar(res.get("statistic", np.nan)),
            "p_value": to_scalar(res.get("pvalue", np.nan)),
            "df_constraint": to_scalar(res.get("df_constraint", np.nan)),
        }
        if "df_denom" in res.index:
            row["df_denom"] = to_scalar(res.get("df_denom", np.nan))
        rows.append(row)
    wald_df = pd.DataFrame(rows)
    wald_df.to_csv(outdir / f"{stem}_wald_tests.csv", index=False, encoding="utf-8-sig")

    with open(outdir / f"{stem}_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Task: {TASK_NAME}\n")
        f.write(f"Model kind: {MODEL_KIND}\n")
        f.write(f"Dependent variable: {DEPENDENT_VAR}\n")
        f.write("Formula model:\n")
        f.write(formula + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\nWald tests by term:\n")
        f.write(wald_df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Унифицированный регрессионный анализ: {TASK_NAME}")
    parser.add_argument("input_file", help="Путь к CSV/XLSX-файлу с данными")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Имя листа Excel; для CSV игнорируется")
    parser.add_argument("--outdir", default=None, help="Папка для результатов; по умолчанию рядом с input_file/regression_output")
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Префикс имен выходных файлов")
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent / "regression_output"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path, sheet=args.sheet)
    prepared, metadata = prepare_data(df)

    save_descriptives(prepared, metadata, outdir, args.stem)
    messages = precheck(prepared)
    save_precheck(input_path, prepared, metadata, messages, outdir, args.stem)

    if messages:
        print("Regression was not estimated.")
        for msg in messages:
            print("-", msg)
        print(f"Descriptive files were saved to: {outdir}")
        return 0

    formula, model = fit_model(prepared)
    save_regression_results(formula, model, outdir, args.stem)

    print(f"Task: {TASK_NAME}")
    print(f"Input file: {input_path}")
    print(f"Rows in original table: {metadata['rows_original']}")
    print(f"Rows before all-max question exclusion: {metadata['rows_before_all_max_exclusion']}")
    print(f"Questions excluded because all scores were maximum: {metadata['n_questions_excluded_all_max']}")
    print(f"Excluded all-max question_id values: {', '.join(metadata['excluded_all_max_question_ids']) if metadata['excluded_all_max_question_ids'] else 'none'}")
    print(f"Rows removed by all-max question exclusion: {metadata['rows_removed_all_max']}")
    print(f"Rows used in regression: {len(prepared)}")
    print(f"Unique question_id: {prepared['question_id'].nunique(dropna=True)}")
    print(f"Output directory: {outdir}")
    if MODEL_KIND == "ols":
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    else:
        print(f"Pseudo R-squared (CS): {model.pseudo_rsquared(kind='cs'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
