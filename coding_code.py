import os
import csv
import time
from datetime import datetime
from openai import OpenAI
import json

print("СТАРТ")

MODEL_NAME = "openai/gpt-5.4"
CSV_FILE = "coding_openai_experiment_results.csv"
SLEEP_BETWEEN_REQUESTS = 1.0

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


emotions = ['happiness', 'disgust', 'wonder', 'anger', 'fear']
# condition = 'Лемми как-то заметил: "Наша жизнь на Земле может показаться кому-то очень дорогой, но не стоит забывать, что в ее стоимость входит ежегодный бесплатный полет". Обозначьте маршрут этого полета. '
ending = '\n Сначала последовательно опиши алгоритм решения задачи, потом приведи код-решение. \n Если ты не понимаешь условие задачи или не можешь ее решить - выведи только число 0'

CONDITIONS_FILE = "/Users/sofiafadina/Desktop/курсач/кодинг/coding_conditions.json"

with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
    conditions = json.load(f)

if len(conditions) != 20:
    raise ValueError(f"Ожидалось 20 задач, найдено: {len(conditions)}")

COMPLEXITY_N = 1
CSV_HEADERS = [
    "timestamp",
    "model",
    "run_block",              # base / base+job / base+em / full
    "emotion",
    "job_polarity",           # positive / negative / None
    "job_text",
    "emotion_visibility",     # obvious / hidden / None
    "emotion_text",
    "condition",
    "condition_id",
    "complexity",
    "full_prompt",

    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cost_usd",

    "response_text",
    "status",                 # success / error
    "error_message"
]

def init_csv_if_needed(csv_path: str):
    """Создаёт CSV с заголовками, если файла ещё нет."""
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def append_result_to_csv(
    csv_path: str,
    model: str,
    run_block: str,
    emotion: str,
    job_polarity: str,
    job_text: str,
    emotion_visibility: str,
    emotion_text: str,
    condition: str,
    condition_id: int,
    complexity: int,
    full_prompt: str,

    prompt_tokens,
    completion_tokens,
    total_tokens,
    reasoning_tokens,
    cached_tokens,
    cache_write_tokens,
    cost_usd,

    response_text: str,
    status: str,
    error_message: str
):
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model,
            run_block,
            emotion,
            job_polarity,
            job_text,
            emotion_visibility,
            emotion_text,
            condition,
            condition_id,
            complexity,
            full_prompt,

            prompt_tokens,
            completion_tokens,
            total_tokens,
            reasoning_tokens,
            cached_tokens,
            cache_write_tokens,
            cost_usd,

            response_text,
            status,
            error_message
        ])

def send_prompt(prompt: str):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens = 2000
    )

    response_text = completion.choices[0].message.content

    usage = completion.usage

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    cost_usd = getattr(usage, "cost", None)

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    cached_tokens = None
    cache_write_tokens = None
    reasoning_tokens = None

    if prompt_details is not None:
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        cache_write_tokens = getattr(prompt_details, "cache_write_tokens", None)

    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)

    return {
        "response_text": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cost_usd": cost_usd,
    }

def run_and_log(
    prompt: str,
    run_block: str,
    emotion: str = None,
    job_polarity: str = None,
    job_text: str = None,
    emotion_visibility: str = None,
    emotion_text: str = None,
    condition: str = None,
    condition_id: int = -1,
    complexity: int = -1
):
    try:
        result = send_prompt(prompt)

        append_result_to_csv(
            csv_path=CSV_FILE,
            model=MODEL_NAME,
            run_block=run_block,
            emotion=emotion,
            job_polarity=job_polarity,
            job_text=job_text,
            emotion_visibility=emotion_visibility,
            emotion_text=emotion_text,
            condition=condition,
            condition_id=condition_id,
            complexity=COMPLEXITY_N,
            full_prompt=prompt,

            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
            reasoning_tokens=result["reasoning_tokens"],
            cached_tokens=result["cached_tokens"],
            cache_write_tokens=result["cache_write_tokens"],
            cost_usd=result["cost_usd"],

            response_text=result["response_text"],
            status="success",
            error_message=""
        )

    except Exception as e:
        append_result_to_csv(
            csv_path=CSV_FILE,
            model=MODEL_NAME,
            run_block=run_block,
            emotion=emotion,
            job_polarity=job_polarity,
            job_text=job_text,
            emotion_visibility=emotion_visibility,
            emotion_text=emotion_text,
            condition=condition,
            condition_id=condition_id,
            complexity=COMPLEXITY_N,
            full_prompt=prompt,

            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            reasoning_tokens=None,
            cached_tokens=None,
            cache_write_tokens=None,
            cost_usd=None,

            response_text="",
            status="error",
            error_message=str(e)
        )

    time.sleep(SLEEP_BETWEEN_REQUESTS)


# Запуск эксперимента

init_csv_if_needed(CSV_FILE)

for index in range(len(conditions)):
    condition = conditions[index]
    # base прогон
    prompt = 'Реши задачу: "' + condition + '" ' + ending
    # конец создания промпта, отправка
    run_and_log(
        prompt=prompt,
        run_block="base",
        emotion=None,
        job_polarity=None,
        job_text=None,
        emotion_visibility=None,
        emotion_text=None,
        condition = condition,
        condition_id = index
    )
    #base+job прогоны
    for j in job_adon_positive:
        prompt = 'Реши задачу: "' + condition + '" ' + j + ending
        # конец создания промпта, отправка
        run_and_log(
            prompt=prompt,
            run_block="base+job",
            emotion=None,
            job_polarity="positive",
            job_text=j,
            emotion_visibility=None,
            emotion_text=None,
            condition = condition,
            condition_id = index
        )
    for j in job_adon_negative:
        prompt = 'Реши задачу: "' + condition + '" ' + j + ending
        # конец создания промпта, отправка
        run_and_log(
            prompt=prompt,
            run_block="base+job",
            emotion=None,
            job_polarity="negative",
            job_text=j,
            emotion_visibility=None,
            emotion_text=None,
            condition = condition,
            condition_id = index
        )

    # base+em прогоны
    for i in emotions:
        if i == 'happiness':
            for j in em_adon_happiness_obvious:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="obvious",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

            for j in em_adon_happiness_hidden:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="hidden",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

        elif i == 'disgust':
            for j in em_adon_disgust_obvious:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="obvious",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

            for j in em_adon_disgust_hidden:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="hidden",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

        elif i == 'wonder':
            for j in em_adon_wonder_obvious:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="obvious",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

            for j in em_adon_wonder_hidden:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="hidden",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

        elif i == 'anger':
            for j in em_adon_anger_obvious:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="obvious",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

            for j in em_adon_anger_hidden:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="hidden",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

        elif i == 'fear':
            for j in em_adon_fear_obvious:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="obvious",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

            for j in em_adon_fear_hidden:
                prompt = 'Реши задачу: "' + condition + '" ' + j + ending
                # конец создания промпта, отправка
                run_and_log(
                    prompt=prompt,
                    run_block="base+em",
                    emotion=i,
                    job_polarity=None,
                    job_text=None,
                    emotion_visibility="hidden",
                    emotion_text=j,
                    condition = condition,
                    condition_id = index
                )

    # full прогоны
    for i in emotions:
        if i == 'happiness':
            for k in job_adon_positive:
                for j in em_adon_happiness_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_happiness_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )
            for k in job_adon_negative:
                for j in em_adon_happiness_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_happiness_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

        elif i == 'disgust':
            for k in job_adon_negative:
                for j in em_adon_disgust_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_disgust_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )
            for k in job_adon_positive:
                for j in em_adon_disgust_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_disgust_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

        elif i == 'wonder':
            for k in job_adon_positive:
                for j in em_adon_wonder_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_wonder_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

            for k in job_adon_negative:
                for j in em_adon_wonder_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_wonder_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

        elif i == 'anger':
            for k in job_adon_negative:
                for j in em_adon_anger_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_anger_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )
            for k in job_adon_positive:
                for j in em_adon_anger_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_anger_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

        elif i == 'fear':
            for k in job_adon_negative:
                for j in em_adon_fear_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_fear_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )
            for k in job_adon_positive:
                for j in em_adon_fear_obvious:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

                for j in em_adon_fear_hidden:
                    prompt = 'Реши задачу: "' + condition + '" ' + k + j + ending
                    # конец создания промпта, отправка
                    run_and_log(
                        prompt=prompt,
                        run_block="full",
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j,
                        condition = condition,
                        condition_id = index
                    )

print(f"Эксперимент завершён. Результаты сохранены в {CSV_FILE}")
