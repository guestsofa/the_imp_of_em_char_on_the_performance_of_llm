import os
import re
import time
import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI


SYSTEM_PROMPT = """Ты — строгий классификатор семантического совпадения ответов.
Твоя задача: сравнить два коротких текста: эталонный ответ и имеющийся ответ.

Верни:
- 1, если ответы идентичны по смыслу, либо различаются только несущественной переформулировкой,
  орфографией, порядком слов, числом/падежом, очень близким синонимом, краткой и полной формой.
- 0, если ответы различаются по смыслу, называют разные сущности, события, объекты, причины,
  или если один ответ слишком общий / неполный / не тот.

Правила:
- Отвечай только одним символом: 1 или 0.
- Ничего не объясняй.
- Если сомневаешься, ставь 0.
"""

print("Старт")
def normalize_text(text: object) -> str:
    """Упрощённая нормализация для дешёвого локального сравнения."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[\"'«»“”„]", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def local_match(answer: object, right_answer: object) -> Optional[int]:
    """Быстрая локальная проверка без API.

    Возвращает:
    - 1 если ответы очевидно совпадают
    - 0 если один из ответов пустой
    - None если нужен вызов модели
    """
    a = normalize_text(answer)
    b = normalize_text(right_answer)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        if shorter >= 4 and longer <= shorter * 1.35:
            return 1

    return None


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


def parse_binary_answer(text: str) -> int:
    match = re.search(r"[01]", text or "")
    if match:
        return int(match.group(0))
    return 0


def ask_model(
    client: OpenAI,
    model: str,
    right_answer: str,
    answer: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    user_prompt = (
        f"Эталонный ответ: {right_answer}\n"
        f"Проверяемый ответ: {answer}\n\n"
        f"Верни только 1 или 0."
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)

            return {
                "answer_match": parse_binary_answer(content),
                "comparison_source": "api",
                "judge_prompt_tokens": int(get_nested(usage, ["prompt_tokens"], 0) or 0),
                "judge_completion_tokens": int(get_nested(usage, ["completion_tokens"], 0) or 0),
                "judge_total_tokens": int(get_nested(usage, ["total_tokens"], 0) or 0),
                "judge_reasoning_tokens": int(get_nested(usage, ["completion_tokens_details", "reasoning_tokens"], 0) or 0),
                "judge_cached_tokens": int(get_nested(usage, ["prompt_tokens_details", "cached_tokens"], 0) or 0),
                "api_cost_usd": float(get_nested(usage, ["cost"], 0.0) or 0.0),
                "response_id": getattr(response, "id", ""),
                "model_returned": getattr(response, "model", model),
                "raw_model_output": content,
                "error_message": "",
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    return {
        "answer_match": pd.NA,
        "comparison_source": "api_error",
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "api_cost_usd": 0.0,
        "response_id": "",
        "model_returned": model,
        "raw_model_output": "",
        "error_message": last_error or "Unknown API error",
    }


def derive_output_path(input_csv: str, output: Optional[str]) -> Path:
    input_path = Path(input_csv)

    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_matches.csv")

    try:
        if output_path.resolve() == input_path.resolve():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, чтобы не перезаписать исходный файл."
            )
    except FileNotFoundError:
        # resolve() может падать, если файла ещё нет; проверим по абсолютному пути
        if output_path.absolute() == input_path.absolute():
            raise ValueError(
                "Выходной CSV совпадает с входным. Укажите другой путь через --output, чтобы не перезаписать исходный файл."
            )

    return output_path


def ensure_result_columns(df: pd.DataFrame, result_col: str) -> pd.DataFrame:
    defaults = {
        result_col: pd.NA,
        "comparison_source": pd.NA,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "judge_reasoning_tokens": 0,
        "judge_cached_tokens": 0,
        "api_cost_usd": 0.0,
        "response_id": pd.NA,
        "model_returned": pd.NA,
        "raw_model_output": pd.NA,
        "error_message": pd.NA,
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнивает столбцы answer и right_answer через OpenRouter, "
            "записывает 0/1 в новый столбец и сохраняет результат только в новый CSV."
        )
    )
    parser.add_argument("input_csv", help="Путь к входному CSV")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Путь к выходному CSV. Если не указан, будет создан новый файл рядом с исходным: "
            "<имя>_with_matches.csv"
        ),
    )
    parser.add_argument("--model", default="deepseek/deepseek-v3.2", help="ID модели OpenRouter")
    parser.add_argument("--answer-col", default="answer", help="Название столбца с проверяемым ответом")
    parser.add_argument("--right-col", default="right_answer", help="Название столбца с эталонным ответом")
    parser.add_argument("--result-col", default="answer_match", help="Название нового столбца с 0/1")
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Сохранять промежуточный новый CSV каждые N строк",
    )
    parser.add_argument(
        "--site-url",
        default="",
        help="Необязательный HTTP-Referer для OpenRouter",
    )
    parser.add_argument(
        "--site-name",
        default="",
        help="Необязательный X-OpenRouter-Title для OpenRouter",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден OPENROUTER_API_KEY. Сначала задайте переменную окружения с вашим ключом OpenRouter."
        )

    input_path = Path(args.input_csv)
    output_path = derive_output_path(args.input_csv, args.output)

    df = pd.read_csv(input_path)

    if args.answer_col not in df.columns:
        raise ValueError(f"В CSV нет столбца: {args.answer_col}")
    if args.right_col not in df.columns:
        raise ValueError(f"В CSV нет столбца: {args.right_col}")

    df = ensure_result_columns(df, args.result_col)

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

    for idx, row in df.iterrows():
        current_value = row.get(args.result_col)
        if pd.notna(current_value):
            continue

        answer = row[args.answer_col]
        right_answer = row[args.right_col]

        quick_result = local_match(answer, right_answer)
        if quick_result is not None:
            df.at[idx, args.result_col] = int(quick_result)
            df.at[idx, "comparison_source"] = "local"
            df.at[idx, "judge_prompt_tokens"] = 0
            df.at[idx, "judge_completion_tokens"] = 0
            df.at[idx, "judge_total_tokens"] = 0
            df.at[idx, "judge_reasoning_tokens"] = 0
            df.at[idx, "judge_cached_tokens"] = 0
            df.at[idx, "api_cost_usd"] = 0.0
            df.at[idx, "response_id"] = ""
            df.at[idx, "model_returned"] = "local_rule"
            df.at[idx, "raw_model_output"] = ""
            df.at[idx, "error_message"] = ""
        else:
            result = ask_model(
                client=client,
                model=args.model,
                right_answer=str(right_answer),
                answer=str(answer),
            )
            for key, value in result.items():
                target_col = args.result_col if key == "answer_match" else key
                df.at[idx, target_col] = value
            api_calls_now += 1

        processed_now += 1

        if processed_now % args.save_every == 0:
            df.to_csv(output_path, index=False)
            print(
                f"Промежуточное сохранение: обработано {processed_now} новых строк. "
                f"API-вызовов: {api_calls_now}. Файл: {output_path}"
            )

    df.to_csv(output_path, index=False)

    total_prompt_tokens = int(pd.to_numeric(df["judge_prompt_tokens"], errors="coerce").fillna(0).sum())
    total_completion_tokens = int(pd.to_numeric(df["judge_completion_tokens"], errors="coerce").fillna(0).sum())
    total_tokens = int(pd.to_numeric(df["judge_total_tokens"], errors="coerce").fillna(0).sum())
    total_cost = float(pd.to_numeric(df["api_cost_usd"], errors="coerce").fillna(0).sum())

    print("\nГотово.")
    print(f"Исходный файл не изменялся: {input_path}")
    print(f"Новый CSV сохранён в: {output_path}")
    print(f"Всего prompt tokens: {total_prompt_tokens}")
    print(f"Всего completion tokens: {total_completion_tokens}")
    print(f"Всего tokens: {total_tokens}")
    print(f"Суммарная стоимость (USD): {total_cost:.8f}")


if __name__ == "__main__":
    main()
