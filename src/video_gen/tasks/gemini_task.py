"""Поиск лучших фрагментов видео через Gemini по расшифровке Whisper."""

import time
from typing import cast

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field

from video_gen.config import AppConfig
from video_gen.models import Highlight, Segment
from video_gen.tasks.base_task import BaseTask


class _HighlightSchema(BaseModel):
    """Схема ответа Gemini (используется только для structured output)."""

    start: float = Field(description="Начало фрагмента в секундах")
    end: float = Field(description="Конец фрагмента в секундах")
    reason: str = Field(description="Почему этот фрагмент цепляет, 1 предложение")
    score: int = Field(description="Виральность от 1 до 10")


SYSTEM_PROMPT = """\
Ты редактор коротких вертикальных видео (Shorts/Reels/TikTok).
Тебе дана расшифровка видео в формате «[начало -> конец] текст», время в секундах.
Выбери фрагменты, которые лучше всего работают как самостоятельные клипы.

Критерии хорошего клипа:
- цепляющее начало в первые 3 секунды (вопрос, сильное утверждение, интрига);
- законченная мысль: клип понятен без остального видео;
- эмоция, неожиданный поворот, польза или яркая история;
- начинается и заканчивается на границе фраз, не посреди предложения.

Правила:
- Используй только тайминги, которые есть в расшифровке. Не выдумывай.
- start бери из начала первой нужной фразы, end из конца последней.
- Длительность каждого клипа от {min_dur} до {max_dur} секунд.
- Клипы не должны пересекаться.
- Верни не более {max_clips} клипов, лучшие первыми.
"""


class GeminiTask(BaseTask):
    """Находит в расшифровке видео лучшие фрагменты для коротких клипов.

    Принимает сегменты Whisper, отправляет их в Gemini как простой текст
    с таймингами и получает структурированный JSON. Ответ валидируется:
    тайминги ограничиваются длительностью видео, границы ставятся в паузы
    между фразами, клипы фильтруются по длительности и пересечениям.

    Экземпляр рассчитан на многократное использование: состояние
    конкретного видео в ``self`` не хранится.

    Attributes:
        model_name: Название модели Gemini из конфига.
        client: Клиент Gemini API.
        logger: Логгер с привязкой ``worker="GeminiTask"``.
    """

    def __init__(self, config: AppConfig) -> None:
        """Создаёт клиент Gemini.

        Args:
            config: Конфигурация приложения. Используются
                ``config.params.gemini_model`` и ``config.secrets.gemini_api_key``
                (поправь под свою структуру конфига).
        """
        super().__init__(config)
        self.model_name = config.params.gemini_model
        self.client = genai.Client(api_key=config.params.gemini_key)
        self.logger = logger.bind(worker="GeminiTask")

    def run(
        self,
        segments: list[Segment],
        max_clips: int = 5,
        min_duration: float = 20.0,
        max_duration: float = 60.0,
        retries: int = 3,
    ) -> list[Highlight]:
        """Выбирает лучшие фрагменты видео.

        Args:
            segments: Сегменты Whisper с абсолютными таймингами.
            max_clips: Максимум клипов в результате.
            min_duration: Минимальная длительность клипа, секунды.
            max_duration: Максимальная длительность клипа, секунды.
            retries: Число попыток при ошибке API или пустом ответе.

        Returns:
            Валидные фрагменты, отсортированные по убыванию ``score``.
            Пустой список, если подходящих фрагментов не нашлось.

        Raises:
            ValueError: Если ``segments`` пуст.
            RuntimeError: Если все попытки запроса провалились.
        """
        if not segments:
            raise ValueError("Пустой список сегментов, нечего анализировать")

        video_end = segments[-1].end
        transcript = self._format_transcript(segments)
        system = SYSTEM_PROMPT.format(
            min_dur=int(min_duration),
            max_dur=int(max_duration),
            max_clips=max_clips,
        )

        raw = self._request(transcript, system, retries)
        highlights = self._validate(raw, segments, video_end, max_clips, min_duration, max_duration)
        self.logger.info(f"Gemini вернул {len(raw)}, после проверки: {len(highlights)}")
        return highlights

    def _request(self, transcript: str, system: str, retries: int) -> list[_HighlightSchema]:
        """Запрашивает фрагменты у Gemini с повторами и экспоненциальной паузой.

        Args:
            transcript: Расшифровка в формате ``[начало -> конец] текст``.
            system: Системный промпт.
            retries: Число попыток.

        Returns:
            Список фрагментов, как их вернула модель (без проверки).

        Raises:
            RuntimeError: Если все попытки провалились.
        """
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=transcript,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        response_schema=list[_HighlightSchema],
                        temperature=0.3,
                    ),
                )
                if response.parsed:
                    return cast(list[_HighlightSchema], response.parsed)
                raise ValueError("Пустой или нераспарсенный ответ")
            except Exception as e:  # noqa: BLE001
                last_error = e
                self.logger.warning(f"Попытка {attempt}/{retries} не удалась: {e}")
                if attempt < retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Gemini недоступен после {retries} попыток") from last_error

    @staticmethod
    def _format_transcript(segments: list[Segment]) -> str:
        """Превращает сегменты в текст ``[12.4 -> 15.1] фраза``.

        Args:
            segments: Сегменты Whisper.

        Returns:
            Многострочный текст, по строке на сегмент.
        """
        return "\n".join(f"[{s.start:.1f} -> {s.end:.1f}] {s.text}" for s in segments)

    def _validate(
        self,
        raw: list[_HighlightSchema],
        segments: list[Segment],
        video_end: float,
        max_clips: int,
        min_dur: float,
        max_dur: float,
    ) -> list[Highlight]:
        """Чистит ответ модели: границы, пауза по краям, длительность, пересечения.

        Фрагменты обрабатываются по убыванию ``score``, поэтому при
        пересечении остаётся более высоко оценённый.

        Args:
            raw: Ответ модели.
            segments: Сегменты Whisper.
            video_end: Конец последнего сегмента, секунды.
            max_clips: Максимум клипов в результате.
            min_dur: Минимальная длительность клипа.
            max_dur: Максимальная длительность клипа.

        Returns:
            Валидные фрагменты. Плохие отбрасываются без исключений.
        """
        result: list[Highlight] = []

        for r in sorted(raw, key=lambda x: x.score, reverse=True):
            start = max(0.0, r.start)
            end = min(video_end, r.end)
            if end <= start:
                self.logger.debug(f"Отброшен (end <= start): {r}")
                continue

            start, end = self._snap(start, end, segments, video_end)

            duration = end - start
            if not (min_dur <= duration <= max_dur):
                self.logger.debug(f"Отброшен (длительность {duration:.1f} с): {r}")
                continue

            if any(start < h.end and end > h.start for h in result):
                self.logger.debug(f"Отброшен (пересечение): {r}")
                continue

            result.append(Highlight(start, end, r.reason.strip(), r.score))
            if len(result) >= max_clips:
                break

        return result

    @staticmethod
    def _snap(
        start: float,
        end: float,
        segments: list[Segment],
        video_end: float,
        pad_before: float = 0.3,
        pad_after: float = 0.5,
    ) -> tuple[float, float]:
        """Ставит границы клипа в паузы вокруг ближайших фраз.

        Берёт первую и последнюю фразу внутри диапазона и отступает от них
        в тишину: не больше ``pad_before``/``pad_after`` секунд и не дальше
        середины паузы до соседней фразы, чтобы не зацепить чужую речь.

        Args:
            start: Начало диапазона от модели, секунды.
            end: Конец диапазона от модели, секунды.
            segments: Сегменты Whisper.
            video_end: Конец видео, секунды.
            pad_before: Максимальный отступ в тишину перед первой фразой.
            pad_after: Максимальный отступ в тишину после последней фразы.

        Returns:
            Кортеж ``(start, end)`` со скорректированными границами.
            Если в диапазон не попал ни один сегмент, границы не меняются.
        """
        idx = [i for i, s in enumerate(segments) if s.end > start and s.start < end]
        if not idx:
            return start, end
        i, j = idx[0], idx[-1]

        prev_end = segments[i - 1].end if i > 0 else 0.0
        next_start = segments[j + 1].start if j + 1 < len(segments) else video_end

        gap_before = max(0.0, segments[i].start - prev_end)
        gap_after = max(0.0, next_start - segments[j].end)

        new_start = segments[i].start - min(pad_before, gap_before / 2)
        new_end = segments[j].end + min(pad_after, gap_after / 2)
        return max(0.0, new_start), min(video_end, new_end)
