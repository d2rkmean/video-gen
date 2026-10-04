"""Генерация .ass-субтитров для клипа с подсветкой слов."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from video_gen.config import AppConfig
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.base_task import BaseTask


@dataclass
class SubtitleStyle:
    """Оформление субтитров. Цвета в формате ASS: ``&HAABBGGRR``.

    Attributes:
        font: Название семейства шрифта, как его видит fontconfig
            (проверка: ``fc-list | grep -i <имя>``).
        font_size: Размер шрифта в пикселях кадра 1080x1920.
        spoken_color: Цвет уже произнесённых слов (подсветка).
        unspoken_color: Цвет ещё не произнесённых слов.
        outline_color: Цвет обводки.
        outline: Толщина обводки.
        margin_v: Отступ от нижнего края. Большое значение уводит текст
            из-под интерфейса платформы.
        margin_h: Боковые отступы.
        uppercase: Выводить текст заглавными буквами.
        res_x: Ширина кадра (PlayResX).
        res_y: Высота кадра (PlayResY).
    """

    font: str = "Montserrat"
    font_size: int = 84
    spoken_color: str = "&H0000FFFF"
    unspoken_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline: int = 6
    margin_v: int = 420
    margin_h: int = 60
    uppercase: bool = True
    res_x: int = 1080
    res_y: int = 1920


class SubtitleTask(BaseTask):
    """Строит .ass-субтитры для клипа из сегментов Whisper.

    Берёт слова, попавшие в диапазон ``Highlight``, сдвигает тайминги так,
    чтобы клип начинался с нуля, группирует слова в короткие строки и
    расставляет karaoke-теги ``\\kf`` для плавной подсветки.

    Состояние конкретного клипа в ``self`` не хранится.

    Attributes:
        style: Оформление субтитров.
        max_words: Максимум слов в одной строке на экране.
        max_chars: Максимум символов в одной строке на экране.
        max_gap: Пауза между словами (сек), после которой строка рвётся.
        hold: Сколько секунд строка остаётся на экране после последнего слова.
        logger: Логгер с привязкой ``worker="SubtitleTask"``.
    """

    SENTENCE_END = (".", "!", "?", "…")

    def __init__(
        self,
        config: AppConfig,
        style: SubtitleStyle | None = None,
        max_words: int = 3,
        max_chars: int = 24,
        max_gap: float = 0.6,
        hold: float = 0.2,
    ) -> None:
        """Инициализирует таск.

        Args:
            config: Конфигурация приложения.
            style: Оформление. По умолчанию ``SubtitleStyle()``.
            max_words: Максимум слов в строке.
            max_chars: Максимум символов в строке.
            max_gap: Пауза, после которой начинается новая строка.
            hold: Задержка исчезновения строки после последнего слова.
        """
        super().__init__(config)
        self.style = style or SubtitleStyle()
        self.max_words = max_words
        self.max_chars = max_chars
        self.max_gap = max_gap
        self.hold = hold
        self.logger = logger.bind(worker="SubtitleTask")

    def run(self, segments: list[Segment], highlight: Highlight, out_path: Path) -> Path:
        """Создаёт .ass для одного клипа.

        Args:
            segments: Сегменты Whisper с абсолютными таймингами.
            highlight: Клип, для которого нужны субтитры.
            out_path: Куда записать .ass.

        Returns:
            Путь к созданному файлу (``out_path``). Тайминги внутри
            относительные: ноль соответствует ``highlight.start``.
        """
        duration = highlight.end - highlight.start
        words = self._collect_words(segments, highlight)
        chunks = self._chunk(words)

        lines = [
            self._dialogue(chunk, chunks[i + 1] if i + 1 < len(chunks) else None, duration)
            for i, chunk in enumerate(chunks)
        ]
        if not lines:
            self.logger.warning(f"В диапазоне {highlight.start:.1f}-{highlight.end:.1f} нет речи")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self._header() + "\n".join(lines) + "\n", encoding="utf-8")
        self.logger.info(f"Субтитры: {out_path} (строк: {len(lines)})")
        return out_path

    def _collect_words(self, segments: list[Segment], h: Highlight) -> list[Word]:
        """Собирает слова клипа и переводит время в относительное.

        Слова, пересекающие границы клипа, обрезаются. Если у сегмента нет
        пословных таймингов, слова синтезируются из текста пропорционально
        длине.
        """
        duration = h.end - h.start
        result: list[Word] = []
        for seg in segments:
            if seg.end <= h.start or seg.start >= h.end:
                continue
            for w in seg.words or self._fake_words(seg):
                if w.end <= h.start or w.start >= h.end:
                    continue
                start = min(max(w.start - h.start, 0.0), duration)
                end = min(max(w.end - h.start, 0.0), duration)
                text = w.text.upper() if self.style.uppercase else w.text
                if text.strip():
                    result.append(Word(start, max(end, start), text.strip()))
        return result

    @staticmethod
    def _fake_words(seg: Segment) -> list[Word]:
        """Делит текст сегмента на слова, раскладывая время по длине слов."""
        parts = seg.text.split()
        total = sum(len(p) for p in parts)
        if not parts or total == 0:
            return []
        words, t = [], seg.start
        span = seg.end - seg.start
        for p in parts:
            d = span * len(p) / total
            words.append(Word(t, t + d, p))
            t += d
        return words

    def _chunk(self, words: list[Word]) -> list[list[Word]]:
        """Группирует слова в короткие экранные строки."""
        chunks: list[list[Word]] = []
        cur: list[Word] = []
        for w in words:
            if cur:
                too_many = len(cur) >= self.max_words
                too_long = len(" ".join(x.text for x in cur)) + 1 + len(w.text) > self.max_chars
                long_pause = w.start - cur[-1].end > self.max_gap
                sentence_end = cur[-1].text.endswith(self.SENTENCE_END)
                if too_many or too_long or long_pause or sentence_end:
                    chunks.append(cur)
                    cur = []
            cur.append(w)
        if cur:
            chunks.append(cur)
        return chunks

    def _dialogue(self, chunk: list[Word], nxt: list[Word] | None, duration: float) -> str:
        """Формирует строку Dialogue с karaoke-тегами для одной строки."""
        start = chunk[0].start
        end = chunk[-1].end + self.hold
        if nxt:
            end = min(end, nxt[0].start)
        end = min(max(end, start + 0.05), duration)

        bounds = [round((w.start - start) * 100) for w in chunk]
        bounds.append(round((chunk[-1].end - start) * 100))

        parts = []
        for i, w in enumerate(chunk):
            cs = max(bounds[i + 1] - bounds[i], 1)
            parts.append(f"{{\\kf{cs}}}{self._escape(w.text)}")
        text = " ".join(parts)

        return f"Dialogue: 0,{self._ts(start)},{self._ts(end)},Default,,0,0,0,,{text}"

    def _header(self) -> str:
        """Заголовок .ass: размеры кадра и стиль Default."""
        s = self.style
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {s.res_x}\n"
            f"PlayResY: {s.res_y}\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{s.font},{s.font_size},{s.spoken_color},{s.unspoken_color},"
            f"{s.outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{s.outline},0,"
            f"2,{s.margin_h},{s.margin_h},{s.margin_v},1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    @staticmethod
    def _ts(t: float) -> str:
        """Секунды в формат ASS ``H:MM:SS.cc``."""
        cs = round(t * 100)
        h, cs = divmod(cs, 360000)
        m, cs = divmod(cs, 6000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02}:{s:02}.{cs:02}"

    @staticmethod
    def _escape(text: str) -> str:
        """Убирает символы, которые ASS трактует как разметку."""
        return text.replace("\\", "").replace("{", "(").replace("}", ")")
