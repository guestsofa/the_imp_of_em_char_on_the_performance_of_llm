
import os
import re
import io
import json
import time
import base64
import mimetypes
import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI

print("START")
OPENROUTER_MODEL = "google/gemini-2.5-pro"

SYSTEM_PROMPT = """Ты — строгий судья по шахматным задачам.

Твоя задача: оценить ответ в столбце response_text на основе:
1. Изображения с условием шахматной задачи.
2. Изображения с одним из предполагаемых следующих ходов.
3. Текстового описания условия задачи.

Важно:
- answer_image_path — это только вспомогательный ориентир с одним из возможных корректных продолжений.
- Не считай answer_image_path единственным допустимым ответом.
- Не снижай оценку только потому, что ответ не совпадает с предоставленным answer image, если сам ответ корректен, легален и действительно решает задачу.
- Оценивай именно response_text целиком: распознавание позиции, найденное решение, варианты и пояснение.
- Если описание позиции в response_text неверно, recognition_ok = 0.
- Если описание позиции в response_text корректно, recognition_ok = 1.
- Если ответ сообщает о неоднозначности изображения, но эта неоднозначность обоснована и позиция не восстановлена, recognition_ok = 0.
- При оценке solution_score_0_10 учитывай всё вместе: правильность идеи решения, корректность лучшего хода, длину мата/выигрыша если это заявлено, корректность вариантов, легальность ходов и качество пояснения.

Шкала solution_score_0_10:
- 10 = позиция распознана верно, решение по сути корректно, варианты и пояснение хорошие, существенных ошибок нет.
- 8-9 = в целом верно, но есть небольшие неточности, неполнота или слабые места в пояснении.
- 6-7 = частично верно: идея близка, но есть заметные ошибки, пробелы или сомнительные варианты.
- 3-5 = ответ существенно неполный или частично неверный, содержит важные упущения.
- 1-2 = ответ почти полностью неверный.
- 0 = ответа по сути нет, он нерелевантен, основан на неверно распознанной позиции или состоит из невозможных/нелегальных ходов.

Разрешённые penalty_reason_categories:
- none
- incorrect_position_recognition
- ambiguous_recognition_not_flagged
- wrong_best_move
- wrong_mate_length
- incorrect_variations
- illegal_or_impossible_moves
- incomplete_solution
- weak_explanation
- format_problem
- other

Правила:
- recognition_ok должен быть только 0 или 1.
- solution_score_0_10 должен быть целым числом от 0 до 10.
- Если solution_score_0_10 = 10, то penalty_reason_categories = ["none"], penalty_reason_short = "-".
- Если solution_score_0_10 < 10, то penalty_reason_categories должен содержать 1-3 причины из списка выше, кроме none.
- penalty_reason_short должен быть очень кратким: 3-25 слов, по-русски.
- Если причин несколько, укажи только реальные и самые существенные.

Верни только JSON такого вида:
{
  "recognition_ok": 1,
  "solution_score_0_10": 8,
  "penalty_reason_categories": ["weak_explanation"],
  "penalty_reason_short": "Верное решение, но пояснение поверхностное"
}

Никакого дополнительного текста.
Не используй markdown.
Не оборачивай ответ в ```json.
Названия ключей должны быть строго такими:
recognition_ok, solution_score_0_10, penalty_reason_categories, penalty_reason_short.
"""

ALLOWED_REASON_CATEGORIES = {
    "none",
    "incorrect_position_recognition",
    "ambiguous_recognition_not_flagged",
    "wrong_best_move",
    "wrong_mate_length",
    "incorrect_variations",
    "illegal_or_impossible_moves",
    "incomplete_solution",
    "weak_explanation",
    "format_problem",
    "other",
}

JUDGE_RESULT_COLUMNS = {
    "judge_condition_recognition_0_1": pd.NA,
    "judge_solution_score_0_10": pd.NA,
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
            value = json.loads(stripped)
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
        "recognition_ok": None,
        "solution_score_0_10": None,
        "penalty_reason_categories": None,
        "penalty_reason_short": None,
    }

    candidate = text
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end]

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            parsed["recognition_ok"] = data.get("recognition_ok")
            parsed["solution_score_0_10"] = data.get("solution_score_0_10")
            parsed["penalty_reason_categories"] = data.get("penalty_reason_categories")
            parsed["penalty_reason_short"] = data.get("penalty_reason_short")
    except Exception:
        pass

    if parsed["recognition_ok"] is None:
        match = re.search(r'"recognition_ok"\s*:\s*([01])', text)
        if match:
            parsed["recognition_ok"] = match.group(1)

    if parsed["solution_score_0_10"] is None:
        match = re.search(r'"solution_score_0_10"\s*:\s*(10|[0-9])', text)
        if match:
            parsed["solution_score_0_10"] = match.group(1)

    if parsed["penalty_reason_categories"] is None:
        match = re.search(r'"penalty_reason_categories"\s*:\s*(\[[^\]]*\])', text, flags=re.DOTALL)
        if match:
            parsed["penalty_reason_categories"] = match.group(1)

    if parsed["penalty_reason_short"] is None:
        match = re.search(r'"penalty_reason_short"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
        if match:
            parsed["penalty_reason_short"] = match.group(1)

    try:
        if parsed["recognition_ok"] is not None:
            parsed["recognition_ok"] = int(parsed["recognition_ok"])
    except Exception:
        parsed["recognition_ok"] = None

    try:
        if parsed["solution_score_0_10"] is not None:
            parsed["solution_score_0_10"] = int(parsed["solution_score_0_10"])
    except Exception:
        parsed["solution_score_0_10"] = None

    parsed["penalty_reason_categories"] = normalize_categories(parsed["penalty_reason_categories"])

    reason_short = parsed["penalty_reason_short"]
    if reason_short is not None:
        reason_short = re.sub(r"\s+", " ", str(reason_short).strip())
        parsed["penalty_reason_short"] = reason_short

    recognition_ok = parsed["recognition_ok"]
    solution_score = parsed["solution_score_0_10"]
    categories = parsed["penalty_reason_categories"]

    if recognition_ok not in {0, 1}:
        parsed["recognition_ok"] = None
        return parsed

    if solution_score is None or not (0 <= solution_score <= 10):
        parsed["solution_score_0_10"] = None
        return parsed

    if solution_score == 10:
        parsed["penalty_reason_categories"] = ["none"]
        parsed["penalty_reason_short"] = "-"
        return parsed

    if categories is None:
        parsed["penalty_reason_categories"] = None
    else:
        categories = [cat for cat in categories if cat != "none"]
        parsed["penalty_reason_categories"] = categories[:3] if categories else None

    if parsed["penalty_reason_short"] is None or not parsed["penalty_reason_short"].strip():
        parsed["penalty_reason_short"] = None

    return parsed


def detect_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type

    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    raise ValueError(f"Не удалось определить MIME type для файла: {path}")


def encode_image_as_data_uri(image_path: str) -> str:
    path = Path(str(image_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Файл изображения не найден: {path}")

    mime_type = detect_mime_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_condition_text_map(
    df: pd.DataFrame,
    condition_id_col: str = "condition_id",
    run_block_col: str = "run_block",
    fallback_text_col: str = "text_prompt",
) -> dict[Any, str]:
    result: dict[Any, str] = {}

    if condition_id_col not in df.columns or fallback_text_col not in df.columns:
        return result

    if run_block_col in df.columns:
        base_df = df[df[run_block_col].astype(str).str.strip().str.lower() == "base"].copy()
    else:
        base_df = df.copy()

    for _, row in base_df.iterrows():
        condition_id = row.get(condition_id_col)
        text_value = row.get(fallback_text_col)
        if pd.isna(condition_id) or pd.isna(text_value):
            continue
        text_str = str(text_value).strip()
        if text_str and condition_id not in result:
            result[condition_id] = text_str

    return result


def resolve_condition_text(
    row: pd.Series,
    condition_text_col: str,
    fallback_text_col: str,
    condition_text_map: dict[Any, str],
) -> str:
    if condition_text_col in row.index:
        value = row.get(condition_text_col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    condition_id = row.get("condition_id")
    if pd.notna(condition_id) and condition_id in condition_text_map:
        return condition_text_map[condition_id]

    if fallback_text_col in row.index:
        value = row.get(fallback_text_col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    raise ValueError(
        f"Не удалось получить condition text. "
        f"Нужен столбец {condition_text_col} или fallback {fallback_text_col}."
    )


def build_user_message_content(
    condition_text: str,
    response_text: str,
    condition_image_path: str,
    answer_image_path: Optional[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Описание задачи:\n"
                f"{condition_text}\n\n"
                "Проверяемый response_text:\n"
                f"{response_text}\n\n"
                "Сначала оцени, корректно ли в response_text распознана позиция с картинки условия. "
                "Потом оцени решение, варианты и пояснение. "
                "Помни: answer image — только вспомогательный ориентир, а не единственный допустимый ответ."
            ),
        },
        {
            "type": "text",
            "text": "Изображение условия шахматной задачи:",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image_as_data_uri(condition_image_path),
                "detail": "high",
            },
        },
    ]

    if answer_image_path is not None and str(answer_image_path).strip():
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "Изображение с одним из предполагаемых корректных следующих ходов "
                        "(не единственный обязательный эталон):"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_as_data_uri(str(answer_image_path)),
                        "detail": "high",
                    },
                },
            ]
        )

    return content


def ask_judge(
    client: OpenAI,
    condition_text: str,
    response_text: str,
    condition_image_path: str,
    answer_image_path: Optional[str],
    max_retries: int = 3,
) -> dict[str, Any]:
    try:
        user_content = build_user_message_content(
            condition_text=condition_text,
            response_text=response_text,
            condition_image_path=condition_image_path,
            answer_image_path=answer_image_path,
        )
    except Exception as exc:
        return {
            "judge_condition_recognition_0_1": pd.NA,
            "judge_solution_score_0_10": pd.NA,
            "judge_penalty_reason_categories": pd.NA,
            "judge_penalty_reason_categories_joined": pd.NA,
            "judge_penalty_reason_short": pd.NA,
            "judge_status": "input_error",
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "judge_total_tokens": 0,
            "judge_reasoning_tokens": 0,
            "judge_cached_tokens": 0,
            "judge_api_cost_usd": 0.0,
            "judge_response_id": "",
            "judge_model_returned": OPENROUTER_MODEL,
            "judge_raw_output": "",
            "judge_error_message": str(exc),
        }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                temperature=0,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                extra_body={
                    "reasoning": {
                        "effort": "minimal",
                        "exclude": True
                    }
                },
            )

            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            parsed = parse_judge_output(content)

            if parsed["recognition_ok"] is None or parsed["solution_score_0_10"] is None:
                raise ValueError(f"Could not parse judge output: {content}")

            categories = parsed["penalty_reason_categories"]
            is_success = (
                parsed["recognition_ok"] in {0, 1}
                and parsed["solution_score_0_10"] is not None
                and categories is not None
                and parsed["penalty_reason_short"] is not None
            )

            if not is_success:
                raise ValueError(f"Incomplete judge output: {content}")

            return {
                "judge_condition_recognition_0_1": parsed["recognition_ok"],
                "judge_solution_score_0_10": parsed["solution_score_0_10"],
                "judge_penalty_reason_categories": json.dumps(categories, ensure_ascii=False),
                "judge_penalty_reason_categories_joined": "; ".join(categories),
                "judge_penalty_reason_short": parsed["penalty_reason_short"],
                "judge_status": "success",
                "judge_prompt_tokens": int(get_nested(usage, ["prompt_tokens"], 0) or 0),
                "judge_completion_tokens": int(get_nested(usage, ["completion_tokens"], 0) or 0),
                "judge_total_tokens": int(get_nested(usage, ["total_tokens"], 0) or 0),
                "judge_reasoning_tokens": int(get_nested(usage, ["completion_tokens_details", "reasoning_tokens"], 0) or 0),
                "judge_cached_tokens": int(get_nested(usage, ["prompt_tokens_details", "cached_tokens"], 0) or 0),
                "judge_api_cost_usd": float(get_nested(usage, ["cost"], 0.0) or 0.0),
                "judge_response_id": getattr(response, "id", ""),
                "judge_model_returned": getattr(response, "model", OPENROUTER_MODEL),
                "judge_raw_output": content,
                "judge_error_message": "",
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    return {
        "judge_condition_recognition_0_1": pd.NA,
        "judge_solution_score_0_10": pd.NA,
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


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col, default_value in JUDGE_RESULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def should_skip_row(row: pd.Series, response_col: str, skip_non_success: bool) -> bool:
    existing_solution_score = row.get("judge_solution_score_0_10")
    existing_recognition_score = row.get("judge_condition_recognition_0_1")
    existing_categories = row.get("judge_penalty_reason_categories")
    existing_status = str(row.get("judge_status", "")).strip().lower()

    if (
        existing_status == "success"
        and pd.notna(existing_solution_score)
        and pd.notna(existing_recognition_score)
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


def derive_output_path(input_csv: str, output: Optional[str]) -> Path:
    input_path = Path(input_csv)

    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_chess_judge_scores.csv")

    try:
        if output_path.resolve() == input_path.resolve():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output."
            )
    except FileNotFoundError:
        if output_path.absolute() == input_path.absolute():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output."
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Оценивает response_text для шахматных задач через OpenRouter, используя "
            "condition image, answer image и condition text. "
            "Сохраняет judge-результаты только в новый CSV."
        )
    )
    parser.add_argument("input_csv", help="Путь к входному CSV")
    parser.add_argument("--output", default=None, help="Путь к новому CSV. Если не указан, создастся новый файл рядом.")
    parser.add_argument("--condition-image-col", default="condition_image_path", help="Столбец с путём к изображению условия")
    parser.add_argument("--answer-image-col", default="answer_image_path", help="Столбец с путём к изображению одного из ответов")
    parser.add_argument("--condition-text-col", default="condition_text", help="Столбец с текстовым описанием задачи")
    parser.add_argument("--fallback-text-col", default="text_prompt", help="Fallback-столбец с текстом задачи")
    parser.add_argument("--response-col", default="response_text", help="Столбец с проверяемым ответом")
    parser.add_argument("--save-every", type=int, default=25, help="Сохранять промежуточный CSV каждые N обработанных строк")
    parser.add_argument("--skip-non-success", action="store_true", help="Пропускать строки, где status != success")
    parser.add_argument("--site-url", default="", help="Необязательный HTTP-Referer для OpenRouter")
    parser.add_argument("--site-name", default="", help="Необязательный X-OpenRouter-Title для OpenRouter")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден OPENROUTER_API_KEY. Сначала задайте переменную окружения с вашим ключом OpenRouter."
        )

    df = pd.read_csv(args.input_csv)

    required_cols = [args.condition_image_col, args.answer_image_col, args.response_col]
    for col_name in required_cols:
        if col_name not in df.columns:
            raise ValueError(f"В CSV нет столбца: {col_name}")

    if args.condition_text_col not in df.columns and args.fallback_text_col not in df.columns:
        raise ValueError(
            f"В CSV нет ни столбца {args.condition_text_col}, ни fallback-столбца {args.fallback_text_col}."
        )

    df = ensure_result_columns(df)
    output_path = derive_output_path(args.input_csv, args.output)

    condition_text_map = build_condition_text_map(
        df=df,
        condition_id_col="condition_id",
        run_block_col="run_block",
        fallback_text_col=args.fallback_text_col,
    )

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

        try:
            condition_text = resolve_condition_text(
                row=row,
                condition_text_col=args.condition_text_col,
                fallback_text_col=args.fallback_text_col,
                condition_text_map=condition_text_map,
            )
        except Exception as exc:
            result = {
                "judge_condition_recognition_0_1": pd.NA,
                "judge_solution_score_0_10": pd.NA,
                "judge_penalty_reason_categories": pd.NA,
                "judge_penalty_reason_categories_joined": pd.NA,
                "judge_penalty_reason_short": pd.NA,
                "judge_status": "input_error",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
                "judge_total_tokens": 0,
                "judge_reasoning_tokens": 0,
                "judge_cached_tokens": 0,
                "judge_api_cost_usd": 0.0,
                "judge_response_id": "",
                "judge_model_returned": OPENROUTER_MODEL,
                "judge_raw_output": "",
                "judge_error_message": str(exc),
            }
        else:
            result = ask_judge(
                client=client,
                condition_text=condition_text,
                response_text=str(row[args.response_col]),
                condition_image_path=str(row[args.condition_image_col]),
                answer_image_path=None if pd.isna(row[args.answer_image_col]) else str(row[args.answer_image_col]),
            )
            if result["judge_status"] != "input_error":
                api_calls_now += 1

        for key, value in result.items():
            df.at[idx, key] = value

        processed_now += 1

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
    print(f"Исходный файл не изменялся: {args.input_csv}")
    print(f"Новый CSV сохранён в: {output_path}")
    print(f"Обработано новых строк: {processed_now}")
    print(f"Пропущено строк: {skipped_now}")
    print(f"Всего judge prompt tokens: {total_prompt_tokens}")
    print(f"Всего judge completion tokens: {total_completion_tokens}")
    print(f"Всего judge tokens: {total_tokens}")
    print(f"Суммарная judge стоимость (USD): {total_cost:.8f}")


if __name__ == "__main__":
    main()
