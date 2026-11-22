import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_gen.config import AppConfig


@dataclass
class Word:
    """Слово с таймингами.

    Attributes:
        start: Начало слова, секунды.
        end: Конец слова, секунды.
        text: Текст слова (вместе со знаками препинания).
        probability: Уверенность Whisper в слове, 0..1.
    """

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    """Сегмент расшифровки.

    Attributes:
        start: Начало сегмента, секунды.
        end: Конец сегмента, секунды.
        text: Текст сегмента.
        words: Слова с таймингами.
        avg_logprob: Средний логарифм вероятности токенов. Ниже -1 обычно плохо.
        no_speech_prob: Вероятность, что в сегменте нет речи, 0..1.
        compression_ratio: Степень сжатия текста. Выше 2.4 признак зацикливания.
    """

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0


@dataclass
class Highlight:
    """Лучший фрагмент видео.

    Attributes:
        start: Начало клипа в секундах от начала исходного видео
            (уже сдвинуто в паузу перед первой фразой).
        end: Конец клипа в секундах от начала исходного видео
            (уже сдвинут в паузу после последней фразы).
        reason: Почему фрагмент выбран.
        score: Оценка виральности от 1 до 10.
        hook: Хук первых 3 секунд.
        viral_title: Кликбейтный заголовок.
        description: Виральное описание с хэштегами для публикации.
    """

    start: float
    end: float
    reason: str
    score: int
    hook: str = ""
    viral_title: str = ""
    description: str = ""
    code: str | None = None
    clip_idx: int | None = None


@dataclass
class VideoJob:
    """Видеофайл с привязанным эффективным конфигом источника.

    Attributes:
        path: Путь к исходному видеофайлу.
        config: Эффективный конфиг для данного видео (локальные настройки + глобальные).
        set_idx: Номер набора источников в top_videos (1-indexed).
        part_idx: Номер части исходника в данном наборе (1-indexed).
        source_idx: Номер источника (синоним set_idx для обратной совместимости).
        metadata: Дополнительные метаданные (например, SponsorBlock сегменты или YouTube инфо).
        caption_name: Название источника для подписи (например, "Южный Парк").
        caption: Кастомный шаблон или текст подписи.
        source_type: Тип источника ("video", "youtube", "pygame").
        source_number: Порядковый номер/ID каталога источника.
    """

    path: Path
    config: AppConfig
    set_idx: int = 1
    part_idx: int = 1
    source_idx: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    caption_name: str | None = None
    caption: str | None = None
    source_type: str = "video"
    source_number: int | None = None
    codename: str | None = None
    is_film: bool = False
    season: int = 1
    episode: int | None = None
    base_code: str | None = None
    video_code: str | None = None

    def __post_init__(self) -> None:
        if self.source_idx != 1 and self.set_idx == 1:
            self.set_idx = self.source_idx
        elif self.set_idx != 1 and self.source_idx == 1:
            self.source_idx = self.set_idx

        if self.base_code is None:
            if self.source_number is not None:
                self.base_code = f"{self.source_number:05d}"
            else:
                self.base_code = f"{secrets.randbelow(100000):05d}"

        if self.episode is None and not self.is_film:
            self.episode = self.part_idx

        if self.video_code is None and self.base_code:
            from video_gen.tasks.film_worker import FilmCode

            self.video_code = FilmCode.build_video_code(
                self.base_code,
                is_film=self.is_film,
                episode=self.episode,
            )

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def __iter__(self):
        yield self.path
        yield self.config
