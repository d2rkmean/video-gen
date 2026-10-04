from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


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
    """

    start: float
    end: float
    reason: str
    score: int
