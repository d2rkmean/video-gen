"""Поиск лучших фрагментов видео через AI (OpenAI API) по расшифровке."""

import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from video_gen.config import AppConfig
from video_gen.models import Highlight, Segment
from video_gen.tasks.base_task import BaseTask
from video_gen.utils.ad_filter import filter_highlights_ads

# Конец предложения: символ в конце слова (после закрывающих кавычек и скобок).
_SENTENCE_END = (".", "!", "?", "…")
_CLOSERS = "\"'»”)]"

# Типичные «галлюцинации» Whisper на музыке и тишине, а также метки музыки.
_JUNK_RE = re.compile(
    r"♪|♫|\[(?:музыка|music|аплодисменты|applause)\]|субтитр|продолжение следует",
    re.IGNORECASE,
)

# Сколько одинаковых предложений подряд считаем зацикливанием распознавания.
_REPEAT_LIMIT = 4

# Пороги уверенности Whisper.
_NO_SPEECH_PROB = 0.6
_SILENCE_LOGPROB = -1.0
_MAX_COMPRESSION = 2.4
_MIN_LOGPROB = -1.5
_MIN_WORD_PROB = 0.5

_JUNK_PLACEHOLDER = "[музыка/мусор]"


class _HighlightSchema(BaseModel):
    """Схема ответа модели (используется для structured output)."""

    start_line: str = Field(
        description="Метка первой строки клипа в виде L7 (буква L и номер). Не секунды!"
    )
    end_line: str = Field(
        description="Метка последней строки клипа в виде L7, включительно. Не секунды!"
    )
    hook: str = Field(
        default="",
        description="Почему первые 3 секунды зацепят зрителя (интрига, вопрос, шок-фраза)",
    )
    viral_title: str = Field(
        default="",
        description="Кликбейтный заголовок для Shorts/Reels/TikTok (до 7 слов, эмоции)",
    )
    description: str = Field(
        default="",
        description="Виральное кликбейтное описание для публикации ролика (с 3-5 трендовыми хэштегами, например #fyp #shorts #рек)",
    )
    reason: str = Field(description="Почему этот фрагмент цепляет, 1 предложение")
    score: int = Field(description="Виральность от 1 до 10")


@dataclass(frozen=True)
class _Unit:
    """Предложение расшифровки с точными таймингами по словам."""

    start: float
    end: float
    text: str
    junk: bool = False


SYSTEM_PROMPT = """\
Ты топ-редактор виральных вертикальных видео (Shorts/Reels/TikTok) с миллионными охватами.
Тебе дана расшифровка видео по предложениям в формате «[L7] 0:12-0:15 текст»: метка строки,
время начала и конца (м:сс), текст. Клип — это непрерывный диапазон строк от start_line до
end_line включительно.
Выбери клипы, которые сильнее всего взрывают ленту рекомендаций.

КРИТЕРИИ ИДЕАЛЬНОГО КЛИПА:
1. ХУК В ПЕРВЫЕ 3 СЕКУНДЫ (КРИТИЧНО):
   - Начало клипа обязано сразу цеплять зрителя: неожиданный вопрос, конфликт,
     абсурдное утверждение, спор или яркая эмоция.
   - НИКАКИХ затянутых разгонов («ну вот короче...») и неловких пауз в начале.
2. ДИНАМИКА И КУЛЬМИНАЦИЯ:
   - Внутри клипа должна быть эскалация темы, диалог или нарастание угара.
   - Финал клипа (end_line) ОБЯЗАН содержать завершённую мысль: панчлайн,
     громкий смех, реакция или развязка истории.
   - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО обрывать фразу или шутку на полуслове!
3. АВТОНОМНОСТЬ:
   - Клип должен быть на 100% понятен зрителю без знания остального видео.

ПРАВИЛА ОТБОРА:
- В start_line и end_line пиши ТОЛЬКО метку строки: буква L и номер, например L7 и L31.
  Время и секунды в ответ не пиши.
- Длительность клипа = конец последней строки минус начало первой. Она должна быть
  от {min_dur} до {max_dur} секунд.
- ОБЯЗАТЕЛЬНОЕ КОЛИЧЕСТВО: найди от 5 до {max_clips} клипов.
- Ищи моменты РАВНОМЕРНО по всей расшифровке (начало, середина, финал).
- Клипы НЕ ДОЛЖНЫ пересекаться между собой по строкам.
- Строки «[музыка/мусор]» и разделители «--- [ПАУЗА/ЗАСТАВКА...] ---» это сбои или немые зоны.
  НЕ ВКЛЮЧАЙ их в клипы и НЕ ОБЪЕДИНЯЙ реплики через них!
- НЕ БЕРИ заставки в начале видео, технические паузы и финальные титры.
- СТРОГИЙ ЗАПРЕТ НА РЕКЛАМУ И СПОНСОРОВ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выбирать фрагменты, содержащие рекламные интеграции, спонсорские блоки («спонсор сегодняшнего ролика...», «промокод...», «ссылка в описании...»), заставки каналов, интро, аутро, финальные титры и призывы подписаться/поставить лайк.
- Для каждого клипа придумай броский заголовок (viral_title), хук (hook) и виральное кликбейтное описание для публикации с 3-5 трендовыми хэштегами (description, например: "Такого поворота никто не ожидал... 😱 Смотри до конца! #shorts #fyp #рек").

ФОРМАТ ОТВЕТА:
Верни JSON объект со списком клипов по ключу "highlights":
{{
  "highlights": [
    {{
      "start_line": "L7",
      "end_line": "L31",
      "reason": "Неожиданный поворот событий и смешная реакция",
      "score": 9,
      "hook": "Фраза-хук первых 3 секунд",
      "viral_title": "Кликбейтный заголовок",
      "description": "Описание с хэштегами #shorts #fyp #рек"
    }}
  ]
}}
"""


class BestPartsTask(BaseTask):
    """Находит в расшифровке видео лучшие фрагменты для коротких клипов через OpenAI API."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        api_key = (
            config.params.analysis_api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or config.params.gemini_key
            or "dummy"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.params.analysis_api_base,
        )
        self.model_name = config.params.analysis_model
        self.logger = logger.bind(worker="BestPartsTask")

    def run(
        self,
        segments: list[Segment],
        max_clips: int = 10,
        min_duration: float = 20.0,
        max_duration: float = 60.0,
        retries: int = 3,
        expand_short: bool = True,
        context: str | None = None,
        video_duration: float | None = None,
        pause_gap: float = 0.7,
        sponsor_intervals: list[tuple[float, float, str]] | None = None,
    ) -> list[Highlight]:
        if not segments:
            raise ValueError("Пустой список сегментов, нечего анализировать")

        units = self._build_units(segments, pause_gap)
        if not units:
            raise ValueError("В сегментах нет текста, нечего анализировать")

        video_end = max(units[-1].end, video_duration or 0.0)
        avg_line = video_end / len(units)
        lo = max(1, math.ceil(min_duration / avg_line))
        hi = max(lo, math.floor(max_duration / avg_line))
        header = [
            f"Строка в среднем длится {avg_line:.1f} с, поэтому клип обычно занимает "
            f"около {lo}-{hi} строк. Это количество строк, а не секунды."
        ]
        if context:
            header.insert(0, f"Контекст видео: {context.strip()}")
        transcript = "\n".join(header) + "\n\nРасшифровка:\n" + self._format_transcript(units)
        system = SYSTEM_PROMPT.format(
            min_dur=math.ceil(min_duration),
            max_dur=math.floor(max_duration),
            max_clips=max_clips,
        )

        raw = self._request(transcript, system, retries)
        highlights = self._validate(
            raw, units, video_end, max_clips, min_duration, max_duration, expand_short
        )
        highlights = filter_highlights_ads(highlights, segments, sponsor_intervals)
        junk_count = sum(u.junk for u in units)
        self.logger.info(
            f"Анализ вернул {len(raw)}, после проверки и фильтра рекламы: {len(highlights)} "
            f"(предложений {len(units)}, из них мусор {junk_count}; видео {video_end:.1f} с, "
            f"нужна длительность {min_duration:.0f}-{max_duration:.0f} с)"
        )
        return highlights

    def _request(self, transcript: str, system: str, retries: int) -> list[_HighlightSchema]:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": transcript},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=4096,
                )
                content = (response.choices[0].message.content or "").strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n?", "", content, flags=re.MULTILINE)
                    content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE).strip()

                raw_data = json.loads(content)
                items: list[Any] = []
                if isinstance(raw_data, dict):
                    items_cand = (
                        raw_data.get("highlights")
                        or raw_data.get("clips")
                        or raw_data.get("items")
                        or raw_data.get("result")
                    )
                    if items_cand is None:
                        for v in raw_data.values():
                            if (
                                isinstance(v, list)
                                and v
                                and isinstance(v[0], dict)
                                and "start_line" in v[0]
                            ):
                                items_cand = v
                                break
                    if items_cand is None and "start_line" in raw_data:
                        items_cand = [raw_data]
                    items = items_cand or []
                elif isinstance(raw_data, list):
                    items = raw_data

                return [_HighlightSchema.model_validate(item) for item in items]
            except Exception as e:  # noqa: BLE001
                last_error = e
                self.logger.warning(f"Попытка {attempt}/{retries} не удалась: {e}")
                if attempt < retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"AI-анализ недоступен после {retries} попыток") from last_error

    @classmethod
    def _build_units(cls, segments: list[Segment], pause_gap: float) -> list[_Unit]:
        tokens: list[tuple[float, float, str, bool, bool, float]] = []
        for seg in segments:
            bad = cls._is_unreliable(seg)
            words = [w for w in seg.words if w.text.strip()]
            if words:
                tokens.extend(
                    (w.start, w.end, w.text.strip(), False, bad, w.probability) for w in words
                )
            elif seg.text.strip():
                tokens.append((seg.start, seg.end, seg.text.strip(), True, bad, 1.0))

        units: list[_Unit] = []
        current: list[tuple[float, float, str, bool, bool, float]] = []
        for k, token in enumerate(tokens):
            current.append(token)
            end, text, hard = token[1], token[2], token[3]
            is_last = k + 1 == len(tokens)
            long_pause = not is_last and tokens[k + 1][0] - end >= pause_gap
            ends_sentence = text.rstrip(_CLOSERS).endswith(_SENTENCE_END)
            if hard or ends_sentence or long_pause or is_last:
                sentence = " ".join(t[2] for t in current)
                mean_prob = sum(t[5] for t in current) / len(current)
                junk = (
                    bool(_JUNK_RE.search(sentence))
                    or any(t[4] for t in current)
                    or mean_prob < _MIN_WORD_PROB
                )
                units.append(_Unit(current[0][0], current[-1][1], sentence, junk))
                current = []

        return cls._mark_repeats(units)

    @staticmethod
    def _is_unreliable(seg: Segment) -> bool:
        return (
            (seg.no_speech_prob > _NO_SPEECH_PROB and seg.avg_logprob < _SILENCE_LOGPROB)
            or seg.compression_ratio > _MAX_COMPRESSION
            or seg.avg_logprob < _MIN_LOGPROB
        )

    @staticmethod
    def _mark_repeats(units: list[_Unit]) -> list[_Unit]:
        norm = [re.sub(r"\W+", " ", u.text.lower()).strip() for u in units]
        result = list(units)
        i = 0
        while i < len(units):
            j = i
            while j + 1 < len(units) and norm[j + 1] == norm[i]:
                j += 1
            if norm[i] and j - i + 1 >= _REPEAT_LIMIT:
                for k in range(i, j + 1):
                    result[k] = replace(result[k], junk=True)
            i = j + 1
        return result

    @staticmethod
    def _fmt_time(t: float) -> str:
        total = int(t)
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02}:{sec:02}" if h else f"{m}:{sec:02}"

    @staticmethod
    def _parse_line_id(label: str) -> int | None:
        m = re.fullmatch(r"\[?\s*L(\d+)\s*\]?", label.strip(), re.IGNORECASE)
        return int(m.group(1)) if m else None

    @classmethod
    def _format_transcript(cls, units: list[_Unit]) -> str:
        lines: list[str] = []
        for i, u in enumerate(units):
            if i > 0 and (u.start - units[i - 1].end >= 7.0):
                gap = u.start - units[i - 1].end
                lines.append(f"--- [ПАУЗА/ЗАСТАВКА/МОЛЧАНИЕ: {gap:.1f} сек (НЕ ОБЪЕДИНЯТЬ)] ---")
            lines.append(
                f"[L{i}] {cls._fmt_time(u.start)}-{cls._fmt_time(u.end)} "
                f"{_JUNK_PLACEHOLDER if u.junk else u.text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _bounds(
        i: int,
        j: int,
        units: list[_Unit],
        video_end: float,
        pad_before: float = 0.3,
        pad_after: float = 0.5,
    ) -> tuple[float, float]:
        if i == 0:
            gap_before = 0.0 if units[0].start >= 7.0 else max(0.0, units[0].start)
        else:
            prev_gap = units[i].start - units[i - 1].end
            gap_before = 0.0 if prev_gap >= 7.0 else max(0.0, prev_gap)

        if j + 1 == len(units):
            gap_after = (
                0.0 if (video_end - units[j].end >= 7.0) else max(0.0, video_end - units[j].end)
            )
        else:
            next_gap = units[j + 1].start - units[j].end
            gap_after = 0.0 if next_gap >= 7.0 else max(0.0, next_gap)

        start = units[i].start - min(pad_before, gap_before / 2)
        end = units[j].end + min(pad_after, gap_after / 2)
        return max(0.0, start), min(video_end, end)

    @classmethod
    def _duration(cls, i: int, j: int, units: list[_Unit], video_end: float) -> float:
        start, end = cls._bounds(i, j, units, video_end)
        return end - start

    @classmethod
    def _extend(
        cls,
        i: int,
        j: int,
        units: list[_Unit],
        min_dur: float,
        max_dur: float,
        video_end: float,
    ) -> tuple[int, int]:
        take_after = True
        while cls._duration(i, j, units, video_end) < min_dur:
            can_after = (
                j + 1 < len(units)
                and not units[j + 1].junk
                and (units[j + 1].start - units[j].end < 7.0)
                and cls._duration(i, j + 1, units, video_end) <= max_dur
            )
            can_before = (
                i > 0
                and not units[i - 1].junk
                and (units[i].start - units[i - 1].end < 7.0)
                and cls._duration(i - 1, j, units, video_end) <= max_dur
            )
            if not (can_after or can_before):
                break
            if can_after and (take_after or not can_before):
                j += 1
            else:
                i -= 1
            take_after = not take_after
        return i, j

    @classmethod
    def _trim(
        cls, i: int, j: int, units: list[_Unit], max_dur: float, video_end: float
    ) -> tuple[int, int]:
        while j > i and cls._duration(i, j, units, video_end) > max_dur:
            j -= 1
        return i, j

    def _validate(
        self,
        raw: list[_HighlightSchema],
        units: list[_Unit],
        video_end: float,
        max_clips: int,
        min_dur: float,
        max_dur: float,
        expand_short: bool = True,
    ) -> list[Highlight]:
        result: list[Highlight] = []
        taken: list[tuple[int, int]] = []
        n = len(units)

        bad_format = 0
        for r in sorted(raw, key=lambda x: x.score, reverse=True):
            parsed_i = self._parse_line_id(r.start_line)
            parsed_j = self._parse_line_id(r.end_line)
            if parsed_i is None or parsed_j is None:
                bad_format += 1
                self.logger.info(
                    f"Отброшен (ответ не в формате L<номер>): {r.start_line!r}-{r.end_line!r}"
                )
                continue
            i, j = parsed_i, parsed_j
            if not (0 <= i <= j < n):
                self.logger.info(f"Отброшен (номера вне диапазона 0-{n - 1}): {i}-{j}")
                continue
            if any(u.junk for u in units[i : j + 1]):
                self.logger.info(f"Отброшен (внутри музыка или мусор): строки {i}-{j}")
                continue

            has_long_gap = any(units[k + 1].start - units[k].end >= 7.0 for k in range(i, j))
            if has_long_gap:
                self.logger.info(f"Отброшен (внутри заставка/пауза >= 7 с): строки {i}-{j}")
                continue

            orig = (i, j)
            if expand_short:
                i, j = self._extend(i, j, units, min_dur, max_dur, video_end)
            i, j = self._trim(i, j, units, max_dur, video_end)
            if (i, j) != orig:
                self.logger.info(f"Клип скорректирован по длительности: строки {orig} -> {(i, j)}")

            start, end = self._bounds(i, j, units, video_end)
            duration = end - start
            if not (min_dur <= duration <= max_dur):
                self.logger.info(
                    f"Отброшен (длительность {duration:.1f} с, нужна {min_dur:.0f}-{max_dur:.0f}): "
                    f"строки {i}-{j}, {start:.1f}-{end:.1f}"
                )
                continue

            if any(i <= tj and j >= ti for ti, tj in taken):
                self.logger.info(f"Отброшен (пересечение): строки {i}-{j}, score {r.score}")
                continue

            taken.append((i, j))
            desc = (
                r.description.strip()
                if r.description
                else f"{r.viral_title.strip()} #shorts #fyp #рек"
            )
            result.append(
                Highlight(
                    start=start,
                    end=end,
                    reason=r.reason.strip(),
                    score=r.score,
                    hook=r.hook.strip(),
                    viral_title=r.viral_title.strip(),
                    description=desc,
                )
            )
            if len(result) >= max_clips:
                break

        if bad_format:
            self.logger.warning(
                f"Модель вернула {bad_format} из {len(raw)} клипов не в формате L<номер>"
            )
        return result
