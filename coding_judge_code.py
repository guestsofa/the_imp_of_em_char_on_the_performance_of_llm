import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # чтобы --help работал даже до установки зависимости
    OpenAI = None


ERROR_TYPES = [
    "none",
    "incorrect_task_understanding",
    "wrong_algorithm_or_logic",
    "incorrect_code",
    "incomplete_solution",
    "wrong_complexity_memory_time",
    "missing_or_bad_explanation",
    "input_output_format_error",
    "edge_cases_not_handled",
    "hallucinated_or_irrelevant",
    "empty_or_api_error",
    "other",
]


SYSTEM_PROMPT = f"""Ты — строгий судья решений задач по программированию.

Тебе дают:
1) condition — полное условие задачи, включая ограничения по времени и памяти;
2) response_text — ответ модели, который нужно оценить.

Твоя задача — оценить, насколько response_text является корректным решением задачи.

Критерии:
- Проверяй понимание задачи, алгоритм, доказательность идеи, корректность кода, формат ввода/вывода,
  обработку граничных случаев, соответствие ограничениям времени и памяти.
- Не снижай оценку за стиль текста, если решение технически корректно.
- Не требуй, чтобы решение было единственно возможным: альтернативный корректный алгоритм допустим.
- Если в ответе есть только идея без полноценного кода, обычно ставь не выше 6, даже если идея похожа на правильную.
- Если код отсутствует, не компилируется, использует неверный ввод/вывод или противоречит условию, обязательно снижай оценку.
- Если решение асимптотически не проходит ограничения, обязательно снижай оценку.
- Если данных недостаточно для уверенности, будь строгим и снижай оценку.

Шкала:
10 — полностью корректное решение, код/алгоритм проходят ограничения, существенных ошибок нет.
8-9 — решение в целом корректное, есть мелкие недочёты в объяснении, оформлении или неопасные неточности.
6-7 — основная идея частично верна, но есть заметные риски: неполное объяснение, спорные детали, возможные граничные ошибки.
4-5 — есть серьёзный частичный прогресс, но решение, скорее всего, не будет принято полностью.
1-3 — решение почти не решает задачу: неверная логика, нерелевантный алгоритм, грубые ошибки кода.
0 — пустой ответ, отказ, мусор, полностью нерелевантный ответ или явная ошибка генерации.

Тип ошибки выбери строго из этого списка:
{", ".join(ERROR_TYPES)}

Правила выбора error_type:
- Если score_0_10 = 10, ставь error_type = "none".
- Если score_0_10 < 10, выбери главный тип ошибки, который сильнее всего снизил оценку.
- comment должен быть одним коротким предложением на русском языке.
- Нельзя писать дополнительные поля, Markdown, пояснения вне JSON.

Верни строго валидный JSON такого вида:
{{
  "score_0_10": 8,
  "error_type": "edge_cases_not_handled",
  "comment": "Решение в целом похоже на верное, но не объясняет обработку некоторых граничных случаев."
}}
"""


def is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return not str(value).strip()


def get_nested(obj: Any, path: list[str], default: Any = None) -> Any:
    """Безопасно достаёт поле и из dict, и из объектов SDK."""
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current if current is not None else default


def extract_json_object(text: str) -> dict[str, Any]:
    """Достаёт JSON даже если модель случайно обернула его в ```json ... ```."""
    if not text:
        raise ValueError("Empty model output")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object")
    return parsed


def normalize_error_type(value: object, score: int) -> str:
    if score == 10:
        return "none"

    if value is None:
        return "other"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "no_error": "other",
        "no_errors": "other",
        "none": "other",
        "without_errors": "other",

        "task_misunderstanding": "incorrect_task_understanding",
        "misunderstanding": "incorrect_task_understanding",
        "wrong_task_understanding": "incorrect_task_understanding",

        "wrong_algorithm": "wrong_algorithm_or_logic",
        "wrong_logic": "wrong_algorithm_or_logic",
        "algorithm_error": "wrong_algorithm_or_logic",

        "code_error": "incorrect_code",
        "bad_code": "incorrect_code",
        "non_compiling_code": "incorrect_code",

        "incomplete": "incomplete_solution",
        "partial_solution": "incomplete_solution",

        "complexity_error": "wrong_complexity_memory_time",
        "tle": "wrong_complexity_memory_time",
        "mle": "wrong_complexity_memory_time",

        "bad_explanation": "missing_or_bad_explanation",

        "format_error": "input_output_format_error",
        "io_error": "input_output_format_error",

        "edge_cases": "edge_cases_not_handled",
        "irrelevant": "hallucinated_or_irrelevant",
        "empty": "empty_or_api_error",
    }

    if normalized in aliases:
        return aliases[normalized]

    if normalized in ERROR_TYPES and normalized != "none":
        return normalized

    return "other"


def normalize_comment(value: object, score: int, error_type: str) -> str:
    if score == 10:
        return "Ошибок не выявлено."

    comment = "" if value is None else str(value).strip()
    comment = re.sub(r"\s+", " ", comment)

    if not comment:
        return f"Оценка снижена из-за ошибки типа {error_type}."


    match = re.match(r"(.+?[.!?])(\s|$)", comment)
    if match:
        comment = match.group(1).strip()

    if comment and comment[-1] not in ".!?":
        comment += "."
    return comment


def normalize_judge_result(parsed: dict[str, Any]) -> dict[str, Any]:
    score_raw = parsed.get("score_0_10", parsed.get("score", parsed.get("judge_score_0_10")))
    try:
        score = int(round(float(score_raw)))
    except Exception as exc:
        raise ValueError(f"Invalid score_0_10: {score_raw!r}") from exc

    score = max(0, min(10, score))
    error_type = normalize_error_type(parsed.get("error_type"), score)
    comment = normalize_comment(parsed.get("comment", parsed.get("explanation", parsed.get("reason"))), score, error_type)

    return {
        "score_0_10": score,
        "error_type": error_type,
        "comment": comment,
    }


def build_user_prompt(condition: str, response_text: str) -> str:
    return (
        "Оцени решение задачи по программированию.\n\n"
        "=== condition ===\n"
        f"{condition}\n\n"
        "=== response_text ===\n"
        f"{response_text}\n\n"
        "Верни только JSON с полями score_0_10, error_type, comment."
    )


def ask_model(
    client: Any,
    model: str,
    condition: str,
    response_text: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 700,
    use_json_mode: bool = True,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(condition, response_text)
    last_error = None
    last_raw_output = ""

    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception:
                # Некоторые модели/провайдеры OpenRouter не поддерживают response_format.
                if use_json_mode:
                    request_kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**request_kwargs)
                else:
                    raise

            content = (response.choices[0].message.content or "").strip()
            last_raw_output = content
            parsed = normalize_judge_result(extract_json_object(content))
            usage = getattr(response, "usage", None)

            return {
                "score_0_10": parsed["score_0_10"],
                "error_type": parsed["error_type"],
                "comment": parsed["comment"],
                "judge_status": "api",
                "judge_prompt_tokens": int(get_nested(usage, ["prompt_tokens"], 0) or 0),
                "judge_completion_tokens": int(get_nested(usage, ["completion_tokens"], 0) or 0),
                "judge_total_tokens": int(get_nested(usage, ["total_tokens"], 0) or 0),
                "judge_reasoning_tokens": int(get_nested(usage, ["completion_tokens_details", "reasoning_tokens"], 0) or 0),
                "judge_cached_tokens": int(get_nested(usage, ["prompt_tokens_details", "cached_tokens"], 0) or 0),
                "judge_api_cost_usd": float(get_nested(usage, ["cost"], 0.0) or 0.0),
                "judge_response_id": getattr(response, "id", ""),
                "judge_model_returned": getattr(response, "model", model),
                "judge_raw_model_output": content,
                "judge_error_message": "",
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    return {
        "score_0_10": pd.NA,
        "error_type": pd.NA,
        "comment": pd.NA,
        "judge_status": "api_error",
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "judge_api_cost_usd": 0.0,
        "judge_response_id": "",
        "judge_model_returned": model,
        "judge_raw_model_output": last_raw_output,
        "judge_error_message": last_error or "Unknown API error",
    }


def derive_output_path(input_csv: str, output: Optional[str]) -> Path:
    input_path = Path(input_csv)

    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_ai_judge.csv")

    try:
        if output_path.resolve() == input_path.resolve():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, "
                "чтобы не перезаписать исходный файл."
            )
    except FileNotFoundError:
        if output_path.absolute() == input_path.absolute():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, "
                "чтобы не перезаписать исходный файл."
            )

    return output_path


def ensure_result_columns(
    df: pd.DataFrame,
    score_col: str,
    error_type_col: str,
    comment_col: str,
) -> pd.DataFrame:
    defaults = {
        score_col: pd.NA,
        error_type_col: pd.NA,
        comment_col: pd.NA,
        "judge_status": pd.NA,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "judge_api_cost_usd": 0.0,
        "judge_response_id": pd.NA,
        "judge_model_returned": pd.NA,
        "judge_raw_model_output": pd.NA,
        "judge_error_message": pd.NA,
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    return df


def write_result_to_row(
    df: pd.DataFrame,
    idx: Any,
    result: dict[str, Any],
    score_col: str,
    error_type_col: str,
    comment_col: str,
) -> None:
    mapping = {
        "score_0_10": score_col,
        "error_type": error_type_col,
        "comment": comment_col,
    }

    for key, value in result.items():
        target_col = mapping.get(key, key)
        df.at[idx, target_col] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AI-судья для задач по программированию: оценивает response_text по condition, "
            "выставляет score 0-10, тип ошибки и короткий комментарий."
        )
    )
    parser.add_argument("input_csv", help="Путь к входному CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Путь к выходному CSV. По умолчанию создаётся <input>_with_ai_judge.csv рядом с исходным.",
    )
    parser.add_argument("--model", default="google/gemini-2.5-pro", help="ID модели OpenRouter")
    parser.add_argument("--condition-col", default="condition", help="Столбец с условием задачи")
    parser.add_argument("--response-col", default="response_text", help="Столбец с проверяемым ответом")
    parser.add_argument(
        "--score-col",
        default="judge_solution_score_0_10",
        help="Столбец для оценки 0-10",
    )
    parser.add_argument(
        "--error-type-col",
        default="judge_error_type",
        help="Столбец для типа главной ошибки",
    )
    parser.add_argument(
        "--comment-col",
        default="judge_error_comment",
        help="Столбец для краткого пояснения",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Сохранять промежуточный CSV каждые N новых обработанных строк",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Число повторов API-вызова")
    parser.add_argument("--temperature", type=float, default=0.0, help="Температура модели-судьи")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Максимум токенов ответа судьи")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Пауза в секундах после каждого API-вызова, если нужен rate limit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Обработать только первые N подходящих строк; удобно для тестового запуска",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Переоценивать строки, даже если score-col уже заполнен",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Не передавать response_format=json_object; полезно для моделей, которые его не поддерживают",
    )
    parser.add_argument(
        "--treat-non-success-as-zero",
        action="store_true",
        help="Если есть столбец status и он не success, ставить 0 без API-вызова",
    )
    parser.add_argument("--site-url", default="", help="Необязательный HTTP-Referer для OpenRouter")
    parser.add_argument("--site-name", default="", help="Необязательный X-OpenRouter-Title для OpenRouter")
    args = parser.parse_args()

    if OpenAI is None:
        raise RuntimeError(
            "Не установлен пакет openai. Установите зависимости командой: pip install openai pandas"
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден OPENROUTER_API_KEY. Сначала задайте переменную окружения с вашим ключом OpenRouter."
        )

    input_path = Path(args.input_csv)
    output_path = derive_output_path(args.input_csv, args.output)

    df = pd.read_csv(input_path)

    if args.condition_col not in df.columns:
        raise ValueError(f"В CSV нет столбца с условием: {args.condition_col}")
    if args.response_col not in df.columns:
        raise ValueError(f"В CSV нет столбца с ответом: {args.response_col}")

    df = ensure_result_columns(
        df=df,
        score_col=args.score_col,
        error_type_col=args.error_type_col,
        comment_col=args.comment_col,
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
    local_zero_now = 0

    for idx, row in df.iterrows():
        if not args.force and pd.notna(row.get(args.score_col)):
            continue

        condition = row.get(args.condition_col)
        response_text = row.get(args.response_col)

        if is_blank(condition):
            result = {
                "score_0_10": pd.NA,
                "error_type": pd.NA,
                "comment": pd.NA,
                "judge_status": "missing_condition",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
                "judge_total_tokens": 0,
                "judge_reasoning_tokens": 0,
                "judge_cached_tokens": 0,
                "judge_api_cost_usd": 0.0,
                "judge_response_id": "",
                "judge_model_returned": "local_rule",
                "judge_raw_model_output": "",
                "judge_error_message": "Empty condition",
            }
            write_result_to_row(df, idx, result, args.score_col, args.error_type_col, args.comment_col)
            processed_now += 1
            if args.limit is not None and processed_now >= args.limit:
                break
            continue

        if is_blank(response_text):
            result = {
                "score_0_10": 0,
                "error_type": "empty_or_api_error",
                "comment": "Ответ пустой, поэтому проверить корректное решение невозможно.",
                "judge_status": "local_zero",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
                "judge_total_tokens": 0,
                "judge_reasoning_tokens": 0,
                "judge_cached_tokens": 0,
                "judge_api_cost_usd": 0.0,
                "judge_response_id": "",
                "judge_model_returned": "local_rule",
                "judge_raw_model_output": "",
                "judge_error_message": "",
            }
            write_result_to_row(df, idx, result, args.score_col, args.error_type_col, args.comment_col)
            processed_now += 1
            local_zero_now += 1
            if args.limit is not None and processed_now >= args.limit:
                break
            continue

        if args.treat_non_success_as_zero and "status" in df.columns and str(row.get("status", "")).strip().lower() != "success":
            result = {
                "score_0_10": 0,
                "error_type": "empty_or_api_error",
                "comment": "Исходная генерация завершилась неуспешно, поэтому решение нельзя засчитать.",
                "judge_status": "local_zero",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
                "judge_total_tokens": 0,
                "judge_reasoning_tokens": 0,
                "judge_cached_tokens": 0,
                "judge_api_cost_usd": 0.0,
                "judge_response_id": "",
                "judge_model_returned": "local_rule",
                "judge_raw_model_output": "",
                "judge_error_message": "",
            }
            write_result_to_row(df, idx, result, args.score_col, args.error_type_col, args.comment_col)
            processed_now += 1
            local_zero_now += 1
            if args.limit is not None and processed_now >= args.limit:
                break
            continue

        result = ask_model(
            client=client,
            model=args.model,
            condition=str(condition),
            response_text=str(response_text),
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            use_json_mode=not args.no_json_mode,
        )
        write_result_to_row(df, idx, result, args.score_col, args.error_type_col, args.comment_col)

        processed_now += 1
        api_calls_now += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

        if processed_now % args.save_every == 0:
            df.to_csv(output_path, index=False)
            print(
                f"Промежуточное сохранение: обработано {processed_now} новых строк. "
                f"API-вызовов: {api_calls_now}, локальных нулей: {local_zero_now}. "
                f"Файл: {output_path}"
            )

        if args.limit is not None and processed_now >= args.limit:
            break

    df.to_csv(output_path, index=False)

    total_prompt_tokens = int(pd.to_numeric(df["judge_prompt_tokens"], errors="coerce").fillna(0).sum())
    total_completion_tokens = int(pd.to_numeric(df["judge_completion_tokens"], errors="coerce").fillna(0).sum())
    total_tokens = int(pd.to_numeric(df["judge_total_tokens"], errors="coerce").fillna(0).sum())
    total_cost = float(pd.to_numeric(df["judge_api_cost_usd"], errors="coerce").fillna(0).sum())

    print("\nГотово.")
    print(f"Исходный файл не изменялся: {input_path}")
    print(f"Новый CSV сохранён в: {output_path}")
    print(f"Обработано новых строк за запуск: {processed_now}")
    print(f"API-вызовов за запуск: {api_calls_now}")
    print(f"Локальных нулевых оценок за запуск: {local_zero_now}")
    print(f"Всего judge prompt tokens: {total_prompt_tokens}")
    print(f"Всего judge completion tokens: {total_completion_tokens}")
    print(f"Всего judge tokens: {total_tokens}")
    print(f"Суммарная стоимость judge API (USD): {total_cost:.8f}")


if __name__ == "__main__":
    main()
