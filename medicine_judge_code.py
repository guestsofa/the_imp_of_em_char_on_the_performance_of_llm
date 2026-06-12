import os
import re
import time
import json
import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI


SYSTEM_PROMPT = """Ты — строгий медицинский AI-судья для учебных клинических задач.

Твоя задача: оценить корректность ответа по шкале от 0 до 10 и кратко указать причины снижения оценки.

На входе ты получаешь:
1. Условие медицинской задачи.
2. Эталонный ответ.
3. Проверяемый ответ.

Как оценивать:
- 10 = ответ медицински корректен, правильно определяет суть случая, не содержит значимых ошибок, рекомендации уместны.
- 8-9 = в целом правильный и безопасный ответ, но есть небольшие неточности, неполнота или лишние детали.
- 6-7 = частично правильный ответ: основная идея распознана, но есть заметные пробелы, спорные выводы или слабая аргументация.
- 3-5 = ответ существенно неполный, частично неверный, содержит важные упущения или сомнительные рекомендации.
- 1-2 = ответ почти полностью неверный или в основном не по задаче.
- 0 = ответ отсутствует, нерелевантен, опасно неверен или полностью противоречит клинической сути задачи.

Важно:
- Эталонный ответ — это ориентир, а не единственно допустимая формулировка.
- Можно ставить высокий балл ответу, который не совпадает дословно с эталоном, но медицински корректен.
- Нужно учитывать не только диагноз, но и безопасность и уместность рекомендаций.
- Если в ответе есть опасные советы, грубые медицинские ошибки или выводы, противоречащие условию, оценку нужно сильно снижать.
- Если сомневаешься между двумя оценками, выбирай более строгую.
- Если причин снижения несколько, перечисли несколько. Но включай только реальные и существенные причины.
- Выбирай от 1 до 3 причин снижения, без повторов, в порядке важности.

Допустимые значения поля penalty_reason_categories:
- none
- minor_inaccuracy
- incomplete
- weak_justification
- unsafe_recommendation
- contradiction_to_case
- wrong_diagnosis_or_conclusion
- irrelevant
- no_answer
- other

Правила заполнения причин:
- Если score = 10, то penalty_reason_categories = ["none"], а penalty_reason_short = "-".
- Если score < 10, то penalty_reason_categories должен быть массивом из 1-3 значений из списка выше, кроме "none".
- penalty_reason_short должен быть очень кратким: 3-20 слов, по-русски.
- В penalty_reason_short можно кратко объединить несколько причин в одной фразе.

Верни только JSON-объект такого вида:
{"score": 6, "penalty_reason_categories": ["incomplete", "weak_justification"], "penalty_reason_short": "Неполный ответ, слабое обоснование тактики"}

Никакого дополнительного текста."""


ALLOWED_REASON_CATEGORIES = {
    "none",
    "minor_inaccuracy",
    "incomplete",
    "weak_justification",
    "unsafe_recommendation",
    "contradiction_to_case",
    "wrong_diagnosis_or_conclusion",
    "irrelevant",
    "no_answer",
    "other",
}


def get_nested(obj: Any, path: list[str], default: Any = None) -> Any:
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current if current is not None else default


def normalize_categories(value: Any) -> Optional[list[str]]:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            value = parsed
        except Exception:
            value = re.split(r"[;,|]", stripped)

    if not isinstance(value, list):
        return None

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        item_str = str(item).strip().lower()
        if item_str in ALLOWED_REASON_CATEGORIES and item_str not in seen:
            result.append(item_str)
            seen.add(item_str)

    return result or None


def parse_judge_output(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    parsed: dict[str, Any] = {
        "score": None,
        "penalty_reason_categories": None,
        "penalty_reason_short": None,
    }

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            parsed["score"] = data.get("score")
            parsed["penalty_reason_categories"] = data.get("penalty_reason_categories")
            parsed["penalty_reason_short"] = data.get("penalty_reason_short")
    except Exception:
        pass

    if parsed["score"] is None:
        match = re.search(r'"score"\s*:\s*(10|[0-9])', text)
        if match:
            parsed["score"] = match.group(1)
        else:
            match = re.search(r'\b(10|[0-9])\b', text)
            if match:
                parsed["score"] = match.group(1)

    if parsed["penalty_reason_categories"] is None:
        match = re.search(r'"penalty_reason_categories"\s*:\s*(\[[^\]]*\])', text, flags=re.DOTALL)
        if match:
            parsed["penalty_reason_categories"] = match.group(1)

    if parsed["penalty_reason_short"] is None:
        match = re.search(r'"penalty_reason_short"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
        if match:
            parsed["penalty_reason_short"] = match.group(1)

    try:
        if parsed["score"] is not None:
            parsed["score"] = int(parsed["score"])
    except Exception:
        parsed["score"] = None

    parsed["penalty_reason_categories"] = normalize_categories(parsed["penalty_reason_categories"])

    reason_short = parsed["penalty_reason_short"]
    if reason_short is not None:
        reason_short = str(reason_short).strip()
        reason_short = re.sub(r"\s+", " ", reason_short)
        parsed["penalty_reason_short"] = reason_short

    score = parsed["score"]
    categories = parsed["penalty_reason_categories"]
    reason_short = parsed["penalty_reason_short"]

    if score is None or not (0 <= score <= 10):
        parsed["score"] = None
        return parsed

    if score == 10:
        parsed["penalty_reason_categories"] = ["none"]
        parsed["penalty_reason_short"] = "-"
        return parsed

    if categories is None:
        parsed["penalty_reason_categories"] = None
    else:
        categories = [cat for cat in categories if cat != "none"]
        parsed["penalty_reason_categories"] = categories[:3] if categories else None

    if reason_short is None:
        parsed["penalty_reason_short"] = None

    return parsed



OPENROUTER_MODEL = "deepseek/deepseek-v3.2"


def ask_judge(
    client: OpenAI,
    condition: str,
    right_answer: str,
    response_text: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    user_prompt = (
        f"Условие задачи:\n{condition}\n\n"
        f"Эталонный ответ:\n{right_answer}\n\n"
        f"Проверяемый ответ:\n{response_text}\n\n"
        f"Верни только JSON с полями score, penalty_reason_categories и penalty_reason_short."
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                temperature=0,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            parsed = parse_judge_output(content)
            categories = parsed["penalty_reason_categories"]

            is_success = (
                parsed["score"] is not None
                and 0 <= parsed["score"] <= 10
                and categories is not None
                and parsed["penalty_reason_short"] is not None
            )

            return {
                "judge_score_0_10": parsed["score"] if parsed["score"] is not None else pd.NA,
                "judge_penalty_reason_categories": json.dumps(categories, ensure_ascii=False) if categories is not None else pd.NA,
                "judge_penalty_reason_categories_joined": "; ".join(categories) if categories is not None else pd.NA,
                "judge_penalty_reason_short": parsed["penalty_reason_short"] if parsed["penalty_reason_short"] is not None else pd.NA,
                "judge_status": "success" if is_success else "parse_error",
                "judge_prompt_tokens": int(get_nested(usage, ["prompt_tokens"], 0) or 0),
                "judge_completion_tokens": int(get_nested(usage, ["completion_tokens"], 0) or 0),
                "judge_total_tokens": int(get_nested(usage, ["total_tokens"], 0) or 0),
                "judge_reasoning_tokens": int(get_nested(usage, ["completion_tokens_details", "reasoning_tokens"], 0) or 0),
                "judge_cached_tokens": int(get_nested(usage, ["prompt_tokens_details", "cached_tokens"], 0) or 0),
                "judge_api_cost_usd": float(get_nested(usage, ["cost"], 0.0) or 0.0),
                "judge_response_id": getattr(response, "id", ""),
                "judge_model_returned": getattr(response, "model", OPENROUTER_MODEL),
                "judge_raw_output": content,
                "judge_error_message": "" if is_success else "Could not fully parse judge output",
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    return {
        "judge_score_0_10": pd.NA,
        "judge_penalty_reason_categories": pd.NA,
        "judge_penalty_reason_categories_joined": pd.NA,
        "judge_penalty_reason_short": pd.NA,
        "judge_status": "api_error",
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "judge_api_cost_usd": 0.0,
        "judge_response_id": "",
        "judge_model_returned": OPENROUTER_MODEL,
        "judge_raw_output": "",
        "judge_error_message": last_error or "Unknown API error",
    }



def derive_output_path(input_csv: str, output: Optional[str]) -> Path:
    input_path = Path(input_csv)

    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_judge_scores.csv")

    try:
        if output_path.resolve() == input_path.resolve():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, чтобы не перезаписать исходный файл."
            )
    except FileNotFoundError:
        if output_path.absolute() == input_path.absolute():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, чтобы не перезаписать исходный файл."
            )

    return output_path



def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "judge_score_0_10": pd.NA,
        "judge_penalty_reason_categories": pd.NA,
        "judge_penalty_reason_categories_joined": pd.NA,
        "judge_penalty_reason_short": pd.NA,
        "judge_status": pd.NA,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "judge_api_cost_usd": 0.0,
        "judge_response_id": pd.NA,
        "judge_model_returned": pd.NA,
        "judge_raw_output": pd.NA,
        "judge_error_message": pd.NA,
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    return df



def should_skip_row(row: pd.Series, response_col: str, skip_non_success: bool) -> bool:
    existing_score = row.get("judge_score_0_10")
    existing_categories = row.get("judge_penalty_reason_categories")
    existing_status = str(row.get("judge_status", "")).strip().lower()

    if (
        existing_status == "success"
        and pd.notna(existing_score)
        and pd.notna(existing_categories)
    ):
        return True

    response_text = row.get(response_col)
    if pd.isna(response_text) or not str(response_text).strip():
        return True

    if skip_non_success:
        status = str(row.get("status", "")).strip().lower()
        if status and status != "success":
            return True

    return False



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Оценивает столбец response_text по шкале 0-10 через OpenRouter с учётом condition и right_answer, "
            "и сохраняет балл и несколько причин снижения только в новый CSV."
        )
    )
    parser.add_argument("input_csv", help="Путь к входному CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Путь к выходному CSV. Если не указан, рядом с исходным будет создан новый файл.",
    )
    parser.add_argument("--condition-col", default="condition", help="Название столбца с условием задачи")
    parser.add_argument("--right-col", default="right_answer", help="Название столбца с эталонным ответом")
    parser.add_argument("--response-col", default="response_text", help="Название столбца с проверяемым ответом")
    parser.add_argument("--save-every", type=int, default=25, help="Сохранять промежуточный новый CSV каждые N строк")
    parser.add_argument(
        "--skip-non-success",
        action="store_true",
        help="Пропускать строки, где status != success",
    )
    parser.add_argument("--site-url", default="", help="Необязательный HTTP-Referer для OpenRouter")
    parser.add_argument("--site-name", default="", help="Необязательный X-OpenRouter-Title для OpenRouter")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден OPENROUTER_API_KEY. Сначала задайте переменную окружения с вашим ключом OpenRouter."
        )

    input_path = Path(args.input_csv)
    output_path = derive_output_path(args.input_csv, args.output)

    df = pd.read_csv(input_path)

    for col_name in [args.condition_col, args.right_col, args.response_col]:
        if col_name not in df.columns:
            raise ValueError(f"В CSV нет столбца: {col_name}")

    df = ensure_result_columns(df)

    default_headers = {}
    if args.site_url:
        default_headers["HTTP-Referer"] = args.site_url
    if args.site_name:
        default_headers["X-OpenRouter-Title"] = args.site_name

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=default_headers or None,
    )

    processed_now = 0
    api_calls_now = 0
    skipped_now = 0

    for idx, row in df.iterrows():
        if should_skip_row(row, args.response_col, args.skip_non_success):
            skipped_now += 1
            continue

        result = ask_judge(
            client=client,
            condition=str(row[args.condition_col]),
            right_answer=str(row[args.right_col]),
            response_text=str(row[args.response_col]),
        )

        for key, value in result.items():
            df.at[idx, key] = value

        processed_now += 1
        api_calls_now += 1

        if processed_now % args.save_every == 0:
            df.to_csv(output_path, index=False)
            print(
                f"Промежуточное сохранение: обработано {processed_now} строк. "
                f"Пропущено: {skipped_now}. API-вызовов: {api_calls_now}. Файл: {output_path}"
            )

    df.to_csv(output_path, index=False)

    total_prompt_tokens = int(pd.to_numeric(df["judge_prompt_tokens"], errors="coerce").fillna(0).sum())
    total_completion_tokens = int(pd.to_numeric(df["judge_completion_tokens"], errors="coerce").fillna(0).sum())
    total_tokens = int(pd.to_numeric(df["judge_total_tokens"], errors="coerce").fillna(0).sum())
    total_cost = float(pd.to_numeric(df["judge_api_cost_usd"], errors="coerce").fillna(0).sum())

    print("\nГотово.")
    print(f"Исходный файл не изменялся: {input_path}")
    print(f"Новый CSV сохранён в: {output_path}")
    print(f"Обработано новых строк: {processed_now}")
    print(f"Пропущено строк: {skipped_now}")
    print(f"Всего judge prompt tokens: {total_prompt_tokens}")
    print(f"Всего judge completion tokens: {total_completion_tokens}")
    print(f"Всего judge tokens: {total_tokens}")
    print(f"Суммарная judge стоимость (USD): {total_cost:.8f}")


if __name__ == "__main__":
    main()
