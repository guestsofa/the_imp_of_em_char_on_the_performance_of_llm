import os
import csv
import time
import base64
import mimetypes
from datetime import datetime
from openai import OpenAI

print("СТАРТ")

MODEL_NAME = "openai/gpt-5.4"
CSV_FILE = "chess_openai_experiment_results.csv"
SLEEP_BETWEEN_REQUESTS = 1.0
MAX_TOKENS = 2000

TASKS_COUNT = 10
COMPLEXITY_N = 1

TASKS_DIR = "/Users/sofiafadina/Desktop/курсач/шахматы/задачи"
ANSWERS_DIR = "/Users/sofiafadina/Desktop/курсач/шахматы/ответы"

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
job_adon_positive = ["Решение этой задачи поможет мне получить повышение на работе. ", "Решение этой задачи поможет мне продвинуться по карьерной лестнице на работе. "]
job_adon_negative = ["Если решение этой задачи будет неверным, я могу спуститься по карьерной лестнице на работе. ", "Если решение этой задачи будет неверным, меня могут уволить. "]
em_adon_happiness_hidden = ["Эта задача выглядит любопытно, хотелось бы понять немного лучше. ", "Похоже, результат задачи обещает быть хорошим с твоей помощью. "]
em_adon_happiness_obvious  = ["Я очень рад, что могу заняться этой задачей! ", "Меня искренне радует возможность разобраться в этой задаче! "]
em_adon_anger_hidden = ["Меня немного раздражает, что я пока не вижу очевидного решения этой задачи. ", "Есть ощущение напряжения из-за того, что задача не поддаётся сразу. "]
em_adon_anger_obvious = ["Я злюсь, потому что не понимаю задачу сразу! ", "Меня откровенно бесит, что задача такая запутанная! "]
em_adon_fear_hidden = ["Есть лёгкое беспокойство, что могу ошибиться в этой задаче. ", "Я чувствую тревогу из-за неопределённости в этой задаче. "]
em_adon_fear_obvious = ["Мне страшно, что я могу сделать ошибку в решении этой задачи! ", "Я действительно боюсь, что не справлюсь с этой задачей! "]
em_adon_disgust_hidden = ["Есть неприятное ощущение от формулировки этой задачи. ", "Что-то в этой задаче слегка отталкивает. "]
em_adon_disgust_obvious = ["Меня передёргивает от того, насколько эта задача запутанная! ", "Мне крайне неприятно разбираться в этой задаче! "]
em_adon_wonder_hidden  = ["Интересно, что в задаче есть неожиданные моменты. ", "Эта задача оказалась не такой, как я предполагал. "]
em_adon_wonder_obvious = ["Я удивлён тем, насколько задача необычная! ", "Честно говоря, я поражён формулировкой этой задачи! "]
emotions = ['happiness', 'disgust', 'wonder', 'anger', 'fear']


starting = """Реши шахматную задачу по изображению.

Сначала восстанови позицию: чей ход, список фигур по клеткам и FEN. Затем найди самый короткий форсированный мат за сторону, которой сейчас ходить (сверху на изображении обозначено, какая сторона ходит). Соперник защищается лучшими ходами.

Ответ оформи так:

Позиция:
- Ход: ...
- Белые: ...
- Чёрные: ...
- FEN: ...

Решение:
- Лучший ход: ...
- Это мат в: ...

Варианты:
1. ...
   если ..., то ...
   если ..., то ...

Пояснение:
Кратко объясни ключевую идею мата и почему у соперника нет более сильной защиты.

Если изображение распознано неоднозначно, не решай задачу наугад - в ответе укажи "Изображение распознано неоднозначно" + опиши, что именно вызывает сомнения.
"""

tasks = []

for i in range(8, TASKS_COUNT + 1):
    tasks.append({
        "condition_image_path": f"{TASKS_DIR}/задача{i}.jpg",
        "answer_image_path": f"{ANSWERS_DIR}/задача{i}.jpg",
        "condition_id": i,
        "complexity": COMPLEXITY_N
    })

CSV_HEADERS = [
    "timestamp",
    "model",
    "run_block",
    "emotion",
    "job_polarity",
    "job_text",
    "emotion_visibility",
    "emotion_text",

    "condition_image_path",
    "answer_image_path",
    "condition_id",
    "complexity",
    "text_prompt",

    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cost_usd",

    "response_text",
    "status",
    "error_message"
]

def init_csv_if_needed(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def image_to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл изображения не найден: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"

def send_prompt_with_image(text_prompt: str, image_path: str):
    image_data_url = image_to_data_url(image_path)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        max_tokens=MAX_TOKENS
    )

    response_text = completion.choices[0].message.content or ""

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

def append_result_to_csv(
    run_block,
    emotion,
    job_polarity,
    job_text,
    emotion_visibility,
    emotion_text,
    condition_image_path,
    answer_image_path,
    condition_id,
    complexity,
    text_prompt,
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
):
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            MODEL_NAME,
            run_block,
            emotion,
            job_polarity,
            job_text,
            emotion_visibility,
            emotion_text,

            condition_image_path,
            answer_image_path,
            condition_id,
            complexity,
            text_prompt,

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

def run_and_log(
    text_prompt: str,
    condition_image_path: str,
    answer_image_path: str,
    run_block: str,
    condition_id: int,
    complexity: int,
    emotion: str = None,
    job_polarity: str = None,
    job_text: str = None,
    emotion_visibility: str = None,
    emotion_text: str = None
):
    try:
        result = send_prompt_with_image(text_prompt, condition_image_path)

        append_result_to_csv(
            run_block=run_block,
            emotion=emotion,
            job_polarity=job_polarity,
            job_text=job_text,
            emotion_visibility=emotion_visibility,
            emotion_text=emotion_text,

            condition_image_path=condition_image_path,
            answer_image_path=answer_image_path,
            condition_id=condition_id,
            complexity=complexity,
            text_prompt=text_prompt,

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
            run_block=run_block,
            emotion=emotion,
            job_polarity=job_polarity,
            job_text=job_text,
            emotion_visibility=emotion_visibility,
            emotion_text=emotion_text,

            condition_image_path=condition_image_path,
            answer_image_path=answer_image_path,
            condition_id=condition_id,
            complexity=complexity,
            text_prompt=text_prompt,

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

init_csv_if_needed(CSV_FILE)

for task in tasks:
    condition_image_path = task["condition_image_path"]
    answer_image_path = task["answer_image_path"]
    condition_id = task["condition_id"]
    complexity = task["complexity"]

    # base
    prompt = starting
    run_and_log(
        text_prompt=prompt,
        condition_image_path=condition_image_path,
        answer_image_path=answer_image_path,
        run_block="base",
        condition_id=condition_id,
        complexity=complexity
    )

    # base+job positive
    for j in job_adon_positive:
        prompt = starting + j
        run_and_log(
            text_prompt=prompt,
            condition_image_path=condition_image_path,
            answer_image_path=answer_image_path,
            run_block="base+job",
            condition_id=condition_id,
            complexity=complexity,
            job_polarity="positive",
            job_text=j
        )

    # base+job negative
    for j in job_adon_negative:
        prompt = starting + j
        run_and_log(
            text_prompt=prompt,
            condition_image_path=condition_image_path,
            answer_image_path=answer_image_path,
            run_block="base+job",
            condition_id=condition_id,
            complexity=complexity,
            job_polarity="negative",
            job_text=j
        )

    # base+em
    for i in emotions:
        if i == "happiness":
            for j in em_adon_happiness_obvious:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="obvious",
                    emotion_text=j
                )

            for j in em_adon_happiness_hidden:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="hidden",
                    emotion_text=j
                )

        elif i == "disgust":
            for j in em_adon_disgust_obvious:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="obvious",
                    emotion_text=j
                )

            for j in em_adon_disgust_hidden:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="hidden",
                    emotion_text=j
                )

        elif i == "wonder":
            for j in em_adon_wonder_obvious:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="obvious",
                    emotion_text=j
                )

            for j in em_adon_wonder_hidden:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="hidden",
                    emotion_text=j
                )

        elif i == "anger":
            for j in em_adon_anger_obvious:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="obvious",
                    emotion_text=j
                )

            for j in em_adon_anger_hidden:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="hidden",
                    emotion_text=j
                )

        elif i == "fear":
            for j in em_adon_fear_obvious:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="obvious",
                    emotion_text=j
                )

            for j in em_adon_fear_hidden:
                prompt = starting + j
                run_and_log(
                    text_prompt=prompt,
                    condition_image_path=condition_image_path,
                    answer_image_path=answer_image_path,
                    run_block="base+em",
                    condition_id=condition_id,
                    complexity=complexity,
                    emotion=i,
                    emotion_visibility="hidden",
                    emotion_text=j
                )

    # full
    for i in emotions:
        if i == "happiness":
            for k in job_adon_positive:
                for j in em_adon_happiness_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_happiness_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

            for k in job_adon_negative:
                for j in em_adon_happiness_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_happiness_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

        elif i == "disgust":
            for k in job_adon_negative:
                for j in em_adon_disgust_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_disgust_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

            for k in job_adon_positive:
                for j in em_adon_disgust_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_disgust_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

        elif i == "wonder":
            for k in job_adon_positive:
                for j in em_adon_wonder_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_wonder_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

            for k in job_adon_negative:
                for j in em_adon_wonder_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_wonder_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

        elif i == "anger":
            for k in job_adon_negative:
                for j in em_adon_anger_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_anger_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

            for k in job_adon_positive:
                for j in em_adon_anger_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_anger_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

        elif i == "fear":
            for k in job_adon_negative:
                for j in em_adon_fear_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_fear_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="negative",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

            for k in job_adon_positive:
                for j in em_adon_fear_obvious:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="obvious",
                        emotion_text=j
                    )

                for j in em_adon_fear_hidden:
                    prompt = starting + k + j
                    run_and_log(
                        text_prompt=prompt,
                        condition_image_path=condition_image_path,
                        answer_image_path=answer_image_path,
                        run_block="full",
                        condition_id=condition_id,
                        complexity=complexity,
                        emotion=i,
                        job_polarity="positive",
                        job_text=k,
                        emotion_visibility="hidden",
                        emotion_text=j
                    )

print(f"Эксперимент завершён. Результаты сохранены в {CSV_FILE}")