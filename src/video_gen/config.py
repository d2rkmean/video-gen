"""Конфигурация фермы: загрузка и валидация config.json."""

import contextlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ASS_COLOR = re.compile(r"^&H[0-9A-Fa-f]{8}$")


class Strict(BaseModel):
    """База: опечатка в ключе даёт ошибку, а не молча игнорируется."""

    model_config = ConfigDict(extra="forbid")


def _resolve_duration_fields(data: Any) -> Any:
    """Вспомогательный резолвер синонимов длительности (duration, min_duration, max_duration)."""
    if isinstance(data, dict):
        dur = data.pop("duration", None)
        if dur is not None:
            if isinstance(dur, (list, tuple)) and len(dur) == 2:
                if "min_part_duration" not in data and "min_duration" not in data:
                    data["min_part_duration"] = int(dur[0])
                if "part_duration" not in data and "max_duration" not in data:
                    data["part_duration"] = int(dur[1])
            elif isinstance(dur, (int, float)) and (
                "part_duration" not in data and "max_duration" not in data
            ):
                data["part_duration"] = int(dur)
        min_d = data.pop("min_duration", None)
        if min_d is not None and "min_part_duration" not in data:
            data["min_part_duration"] = int(min_d)
        max_d = data.pop("max_duration", None)
        if max_d is not None and "part_duration" not in data:
            data["part_duration"] = int(max_d)
    return data


class PipelineParams(Strict):
    output_dir: Path = Field(
        default=Path("output/"),
        description="Директория для сохранения готовых нарезок и метаданных.",
    )

    # Transcription backend
    transcribe_backend: Literal["local", "groq", "openai"] = Field(
        default="local",
        description="Бэкенд транскрипции. local=faster-whisper GPU, groq/openai=облачный API.",
    )
    transcribe_api_key: str = Field(
        default="",
        description="API ключ для groq/openai транскрипции.",
    )
    transcribe_api_base: str = Field(
        default="",
        description="Кастомный base_url для API транскрипции.",
    )
    transcribe_model: str = Field(
        default="whisper-large-v3",
        description="Модель для groq/openai транскрипции.",
    )

    # Analysis backend
    analysis_backend: Literal["openai", "groq", "custom"] = Field(
        default="openai",
        description="Бэкенд для AI-анализа лучших частей видео: openai, groq или custom.",
    )
    analysis_api_key: str = Field(
        default="",
        description="API ключ для AI-анализа (OpenAI/Groq). Обязателен, если не задан через env.",
    )
    analysis_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Базовый URL эндпоинта для AI-анализа.",
    )
    analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Модель для AI-анализа. Groq: llama-3.3-70b-versatile. OpenAI: gpt-4o-mini.",
    )

    # Pipeline tuning
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Максимальное количество повторных попыток при запросах к API.",
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="Задержка между повторными попытками в секундах.",
    )
    concurrent_renders: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Количество одновременных процессов рендеринга FFmpeg.",
    )

    # Gemini (для обратной совместимости)
    gemini_key: str = Field(
        default="",
        description="Ключ Gemini API (для обратной совместимости).",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Модель Gemini (для обратной совместимости).",
    )

    parts_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Максимальное количество клипов, нарезаемых из одного видео.",
    )
    min_part_duration: int = Field(
        default=20,
        ge=5,
        le=170,
        description="Минимальная длительность клипа в секундах.",
    )
    part_duration: int = Field(
        default=60,
        ge=10,
        le=180,
        description="Максимальная длительность клипа в секундах.",
    )
    silence_gap_threshold: float = Field(
        default=7.0,
        ge=1.0,
        le=60.0,
        description="Порог паузы/тишины между репликами в секундах для разбивки.",
    )

    # Whisper (локальный)
    whisper_model: str = Field(
        default="large-v3",
        description="Модель локального faster-whisper (tiny, base, small, medium, large-v3).",
    )
    whisper_device: Literal["cuda", "cpu", "auto"] = Field(
        default="cuda",
        description="Устройство выполнения для локального Whisper (cuda, cpu, auto).",
    )
    whisper_compute_type: Literal["float16", "int8_float16", "int8", "float32"] = Field(
        default="float16",
        description="Тип квантования/вычислений для локального Whisper.",
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Ширина луча (beam size) для декодирования Whisper.",
    )
    language: str = Field(
        default="ru",
        min_length=2,
        max_length=5,
        description="Код языка для распознавания речи (ru, en и др.).",
    )

    # Режим нарезки
    mode: Literal["highlights", "speech_split"] = Field(
        default="highlights",
        description="Режим нарезки: highlights (AI отбор лучших моментов) или speech_split (нарезка по паузам).",
    )
    sponsorblock_enabled: bool = Field(
        default=True,
        description="Включение вырезания спонсорских интеграций через SponsorBlock API.",
    )
    sponsorblock_categories: list[str] = Field(
        default_factory=lambda: ["sponsor", "selfpromo", "interaction", "intro", "outro"],
        description="Категории сегментов SponsorBlock для фильтрации.",
    )

    # Кэширование
    use_whisper_cache: bool = Field(
        default=True,
        description="Флаг использования дискового кэша распознавания речи.",
    )
    use_gemini_cache: bool = Field(
        default=True,
        description="Флаг использования кэша AI-анализа фрагментов.",
    )
    use_render_cache: bool = Field(
        default=True,
        description="Флаг пропуска уже отрендеренных клипов при совпадении хэша.",
    )

    @model_validator(mode="before")
    @classmethod
    def key_from_env(cls, data: Any) -> Any:
        data = _resolve_duration_fields(data)
        if isinstance(data, dict):
            if not data.get("analysis_api_key"):
                env_ai = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if env_ai:
                    data = {**data, "analysis_api_key": env_ai}
            if not data.get("transcribe_api_key"):
                env_tr = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if env_tr:
                    data = {**data, "transcribe_api_key": env_tr}
            if not data.get("gemini_key"):
                env = os.environ.get("GEMINI_API_KEY")
                if env:
                    data = {**data, "gemini_key": env}
        return data

    @model_validator(mode="after")
    def durations_consistent(self) -> Self:
        if self.min_part_duration >= self.part_duration:
            raise ValueError("min_part_duration должен быть меньше part_duration")
        return self

    @field_validator("output_dir", mode="after")
    @classmethod
    def ensure_output_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


class AssetsConfig(Strict):
    banner: Path | None = Field(
        default=None,
        description="Путь к видеофайлу баннера (нет ключа = без баннера).",
    )
    banners_dir: Path | None = Field(
        default=None,
        description="Директория с баннерами; если задана, баннер выбирается случайно.",
    )
    music_dir: Path | None = Field(
        default=None,
        description="Директория с фоновой музыкой; нет ключа = без музыки.",
    )
    fonts_dir: Path = Field(
        default=Path("fonts/"),
        description="Директория со шрифтами для генерации субтитров ASS.",
    )

    @field_validator("banner")
    @classmethod
    def banner_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_file():
            raise ValueError(f"баннер не найден: {p}")
        return p

    @field_validator("banners_dir")
    @classmethod
    def banners_dir_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_dir():
            raise ValueError(f"папка с баннерами не найдена: {p}")
        return p

    @field_validator("music_dir")
    @classmethod
    def music_dir_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_dir():
            raise ValueError(f"папка с музыкой не найдена: {p}")
        return p


class RenderConfig(Strict):
    """Поля совпадают с RenderSettings: RenderSettings(**cfg.model_dump())."""

    width: int = Field(
        default=1080,
        ge=360,
        le=4096,
        description="Ширина выходного видео. Стандарт Reels/Shorts: 1080.",
    )
    height: int = Field(
        default=1920,
        ge=640,
        le=4096,
        description="Высота выходного видео. Стандарт Reels/Shorts: 1920.",
    )
    fps: int = Field(
        default=30,
        ge=15,
        le=60,
        description="Частота кадров выходного видео (FPS).",
    )
    top_parts: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Доля верхнего видео в общей вертикальной раскладке.",
    )
    bottom_parts: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Доля нижнего видео в общей вертикальной раскладке.",
    )
    mirror: bool = Field(
        default=True,
        description="Зеркальное отражение верхнего видео по горизонтали (hflip).",
    )
    top_zoom: float = Field(
        default=1.0,
        ge=1.0,
        le=3.0,
        description="Множитель приближения (зум) верхнего видео.",
    )
    top_x_offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Смещение верхнего видео по горизонтали X (-1.0 до 1.0).",
    )
    top_y_offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Смещение верхнего видео по вертикали Y (-1.0 до 1.0).",
    )
    contrast: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Контраст верхнего видео.",
    )
    saturation: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Насыщенность верхнего видео.",
    )
    brightness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Яркость верхнего видео (-1.0 до 1.0).",
    )
    gamma: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Гамма-коррекция верхнего видео.",
    )
    bottom_contrast: float = Field(
        default=1.10,
        ge=0.5,
        le=2.0,
        description="Контраст нижнего видео.",
    )
    bottom_saturation: float = Field(
        default=1.25,
        ge=0.0,
        le=3.0,
        description="Насыщенность нижнего видео.",
    )
    bottom_brightness: float = Field(
        default=0.01,
        ge=-0.3,
        le=0.3,
        description="Яркость нижнего видео.",
    )
    bottom_gamma: float = Field(
        default=1.02,
        ge=0.5,
        le=2.0,
        description="Гамма-коррекция нижнего видео.",
    )
    noise_top: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Интенсивность шума на верхнем видео.",
    )
    noise_bottom: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Интенсивность шума на нижнем видео.",
    )
    bottom_mode: Literal["random", "sequential"] = Field(
        default="random",
        description="Режим выбора фрагментов нижнего видео: random или sequential.",
    )
    blur_sigma: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Радиус размытия кадра на время стоп-кадра/баннера.",
    )
    blur_in: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Длительность нарастания размытия в секундах.",
    )
    blur_out: float = Field(
        default=0.3,
        ge=0.0,
        le=5.0,
        description="Длительность затухания размытия в секундах.",
    )
    banner_width: int = Field(
        default=800,
        ge=100,
        le=4096,
        description="Ширина баннера в пикселях.",
    )
    banner_chromakey: bool = Field(
        default=False,
        description="Включение хромакея для удаления фона баннера.",
    )
    banner_chroma_color: str = Field(
        default="green",
        description="Цвет фона хромакея баннера (например green, 0x0100f2).",
    )
    banner_chroma_similarity: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="Допуск хромакея по оттенку.",
    )
    banner_chroma_blend: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Сглаживание границ хромакея.",
    )
    banner_autocrop: bool = Field(
        default=False,
        description="Автоматическая обрезка прозрачных краев баннера.",
    )
    banner_autocrop_threshold: int = Field(
        default=10,
        ge=0,
        le=255,
        description="Порог непрозрачности для автообрезки баннера.",
    )
    banner_mode: Literal["center", "overlay", "end", "interval_minutes", "interval_percent"] = (
        Field(
            default="center",
            description="Режим показа баннера: center (стоп-кадр по центру), overlay (поверх), end (в конце), interval_minutes (каждые N минут), interval_percent (каждые N% ролика).",
        )
    )
    banner_interval_minutes: float = Field(
        default=1.0,
        gt=0.0,
        le=180.0,
        description="Интервал показа баннера в минутах (для режима interval_minutes).",
    )
    banner_interval_percent: float = Field(
        default=25.0,
        gt=0.0,
        le=100.0,
        description="Интервал показа баннера в процентах длительности ролика (для режима interval_percent).",
    )
    banner_freeze: bool = Field(
        default=True,
        description="Останавливать ли основное видео (стоп-кадр с размытием) на время показа баннера.",
    )
    banner_anchor: Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"] = (
        Field(
            default="center",
            description="Точка привязки баннера при оверлее.",
        )
    )
    banner_x: int | None = Field(
        default=None,
        description="Пользовательское смещение баннера по X.",
    )
    banner_y: int | None = Field(
        default=None,
        description="Пользовательское смещение баннера по Y.",
    )
    banner_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Прозрачность баннера (0.0..1.0).",
    )
    banner_scale: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Масштаб баннера.",
    )
    duck_volume: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Громкость музыки при автоприглушении (ducking).",
    )
    music_volume: float = Field(
        default=0.07,
        ge=0.0,
        le=1.0,
        description="Громкость фоновой музыки.",
    )
    speech_volume: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Громкость речи/основного звука.",
    )
    audio_fade_in: float = Field(
        default=0.3,
        ge=0.0,
        le=3.0,
        description="Плавное нарастание звука в начале клипа (секунды).",
    )
    audio_fade_out: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="Плавное затухание звука в конце клипа (секунды).",
    )
    cq: int = Field(
        default=28,
        ge=1,
        le=51,
        description="Коэффициент постоянного качества кодирования (CQ / CRF).",
    )
    maxrate: str = Field(
        default="6M",
        description="Максимальный битрейт видеопотока.",
    )
    bufsize: str = Field(
        default="12M",
        description="Размер буфера видеопотока.",
    )
    strip_metadata: bool = Field(
        default=True,
        description="Удаление всех метаданных и скрытых тегов из файла.",
    )
    enable_antidetect: bool = Field(
        default=True,
        description="Включение микро-рандомизации параметров для обхода детекта повторного контента.",
    )
    speed_jitter: bool = Field(
        default=True,
        description="Микро-колебание скорости видеоряда.",
    )
    speed_jitter_min: float = Field(
        default=0.985,
        ge=0.9,
        le=1.0,
        description="Минимальный коэффициент скорости при джиттере.",
    )
    speed_jitter_max: float = Field(
        default=1.015,
        ge=1.0,
        le=1.1,
        description="Максимальный коэффициент скорости при джиттере.",
    )
    face_tracking: bool = Field(
        default=True,
        description="Центрирование верхнего видео по лицу говорящего.",
    )
    sharpen: bool = Field(
        default=True,
        description="Фильтр повышения резкости (unsharp).",
    )
    unsharp_luma: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Сила резкости канала яркости.",
    )
    unsharp_chroma: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Сила резкости канала цвета.",
    )
    video_codec: Literal["h264_nvenc", "libx264", "h265_nvenc", "libx265"] = Field(
        default="h264_nvenc",
        description="Видеокодек для рендеринга (аппаратный NVENC или процессорный).",
    )
    audio_codec: Literal["aac", "mp3", "copy"] = Field(
        default="aac",
        description="Аудиокодек выходного ролика.",
    )
    audio_bitrate: str = Field(
        default="192k",
        description="Битрейт аудиодорожки.",
    )
    preset: str = Field(
        default="p4",
        description="Пресет кодирования видео (p1..p7 для NVENC или veryfast..slow для libx264).",
    )

    # Кастомная подложка, размытие и отступы
    enable_canvas_blur: bool = Field(
        default=False,
        description="Включение размытой подложки канваса под видео.",
    )
    boxblur_radius: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description="Радиус размытия boxblur подложки.",
    )
    boxblur_power: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Количество итераций размытия boxblur.",
    )
    canvas_bg_color: str = Field(
        default="black",
        description="Цвет фона подложки.",
    )
    top_offset_y: int = Field(
        default=0,
        description="Смещение позиции верхнего видео по вертикали Y.",
    )
    bottom_offset_y: int = Field(
        default=0,
        description="Смещение позиции нижнего видео по вертикали Y.",
    )
    top_padding: int = Field(
        default=0,
        description="Внутренний вертикальный отступ верхнего видео.",
    )
    bottom_padding: int = Field(
        default=0,
        description="Внутренний вертикальный отступ нижнего видео.",
    )

    @model_validator(mode="after")
    def banner_fits(self) -> Self:
        self.banner_width = min(self.banner_width, self.width)
        return self


class SubtitlesConfig(Strict):
    """Стиль (-> SubtitleStyle) и правила разбивки на строки (-> SubtitleTask)."""

    enabled: bool = Field(
        default=True,
        description="Включение рендеринга и наложения субтитров на ролик.",
    )
    font: str = Field(
        default="Unbounded",
        description="Семейство шрифта для субтитров.",
    )
    font_size: int = Field(
        default=78,
        ge=20,
        le=300,
        description="Размер шрифта субтитров в пикселях.",
    )
    bold: bool = Field(
        default=False,
        description="Жирный шрифт субтитров.",
    )
    italic: bool = Field(
        default=False,
        description="Курсивный шрифт субтитров.",
    )
    fill_color: str = Field(
        default="&H00FFFFFF",
        description="Основной цвет заливки шрифта в формате ASS (&HAABBGGRR).",
    )
    spoken_color: str = Field(
        default="&H0000FFFF",
        description="Цвет активного произносимого слова (&HAABBGGRR).",
    )
    unspoken_color: str = Field(
        default="&H00FFFFFF",
        description="Цвет ещё не произнесённых слов (&HAABBGGRR).",
    )
    outline_color: str = Field(
        default="&H00000000",
        description="Цвет обводки субтитров (&HAABBGGRR).",
    )
    outline: int = Field(
        default=6,
        ge=0,
        le=20,
        description="Толщина обводки текста.",
    )
    shadow: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Глубина тени текста.",
    )
    pop_scale: float = Field(
        default=1.15,
        ge=1.0,
        le=2.0,
        description="Масштаб всплывания слова при pop-анимации.",
    )
    pop_duration: float = Field(
        default=0.08,
        ge=0.01,
        le=0.5,
        description="Длительность эффекта всплывания слова в секундах.",
    )
    fade_in: bool = Field(
        default=False,
        description="Плавное появление строк субтитров.",
    )
    fade_out: bool = Field(
        default=False,
        description="Плавное затухание строк субтитров.",
    )
    word_spacing: int = Field(
        default=0,
        ge=-20,
        le=50,
        description="Дополнительный интервал между словами.",
    )
    line_spacing: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Множитель межстрочного интервала.",
    )
    alignment: int = Field(
        default=2,
        ge=1,
        le=9,
        description="Выравнивание текста по сетке ASS (2=снизу по центру, 5=центр).",
    )
    position: Literal["bottom", "top", "middle"] = Field(
        default="bottom",
        description="Вертикальная позиция субтитров (bottom, top, middle).",
    )
    margin_v: int = Field(
        default=0,
        ge=0,
        description="Вертикальный отступ текста субтитров.",
    )
    margin_h: int = Field(
        default=60,
        ge=0,
        description="Боковые отступы текста субтитров.",
    )
    uppercase: bool = Field(
        default=True,
        description="Преобразование всего текста субтитров в верхний регистр.",
    )
    animation: Literal["pop", "bounce", "none"] = Field(
        default="pop",
        description="Тип анимации караоке-подсветки слов (pop, bounce, none).",
    )
    pop_animation: bool = Field(
        default=True,
        description="Включение анимации увеличения произносимого слова.",
    )
    seam_offset: int = Field(
        default=0,
        description="Смещение субтитров относительно линии стыка верх/низ.",
    )
    max_words: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Максимальное количество слов в одной экранной строке.",
    )
    max_chars: int = Field(
        default=24,
        ge=8,
        le=60,
        description="Максимальное количество символов в одной экранной строке.",
    )
    max_gap: float = Field(
        default=0.6,
        ge=0.1,
        le=3.0,
        description="Максимальная пауза между словами для объединения в одну строку.",
    )
    hold: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Время удержания строки на экране после последнего слова.",
    )

    @field_validator("spoken_color", "unspoken_color", "outline_color", "fill_color")
    @classmethod
    def valid_ass_color(cls, v: str) -> str:
        if not ASS_COLOR.match(v):
            raise ValueError(f"цвет должен быть в формате &HAABBGGRR, получено: {v}")
        return v


class PartialPipelineParams(Strict):
    output_dir: Path | None = None
    transcribe_backend: Literal["local", "groq", "openai"] | None = None
    transcribe_api_key: str | None = None
    transcribe_api_base: str | None = None
    transcribe_model: str | None = None
    analysis_backend: Literal["openai", "groq", "custom"] | None = None
    analysis_api_key: str | None = None
    analysis_api_base: str | None = None
    analysis_model: str | None = None
    max_retries: int | None = Field(default=None, ge=1, le=10)
    retry_delay: float | None = Field(default=None, ge=0.5, le=30.0)
    concurrent_renders: int | None = Field(default=None, ge=1, le=4)
    gemini_key: str | None = None
    gemini_model: str | None = None
    parts_count: int | None = Field(default=None, ge=1, le=20)
    min_part_duration: int | None = Field(default=None, ge=5, le=170)
    part_duration: int | None = Field(default=None, ge=10, le=180)
    silence_gap_threshold: float | None = Field(default=None, ge=1.0, le=60.0)
    whisper_model: str | None = None
    whisper_device: Literal["cuda", "cpu", "auto"] | None = None
    whisper_compute_type: Literal["float16", "int8_float16", "int8", "float32"] | None = None
    beam_size: int | None = Field(default=None, ge=1, le=10)
    language: str | None = Field(default=None, min_length=2, max_length=5)
    mode: Literal["highlights", "speech_split"] | None = None
    sponsorblock_enabled: bool | None = None
    sponsorblock_categories: list[str] | None = None
    use_whisper_cache: bool | None = None
    use_gemini_cache: bool | None = None
    use_render_cache: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_durations(cls, data: Any) -> Any:
        return _resolve_duration_fields(data)

    @model_validator(mode="after")
    def durations_consistent(self) -> Self:
        if (
            self.min_part_duration is not None
            and self.part_duration is not None
            and self.min_part_duration >= self.part_duration
        ):
            raise ValueError(
                f"min_part_duration ({self.min_part_duration}) должен быть меньше part_duration ({self.part_duration})"
            )
        return self


class PartialAssetsConfig(Strict):
    banner: Path | None = None
    banners_dir: Path | None = None
    music_dir: Path | None = None
    fonts_dir: Path | None = None

    @field_validator("banner")
    @classmethod
    def banner_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_file():
            raise ValueError(f"баннер не найден: {p}")
        return p

    @field_validator("banners_dir")
    @classmethod
    def banners_dir_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_dir():
            raise ValueError(f"папка с баннерами не найдена: {p}")
        return p

    @field_validator("music_dir")
    @classmethod
    def music_dir_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_dir():
            raise ValueError(f"папка с музыкой не найдена: {p}")
        return p


class PartialRenderConfig(Strict):
    width: int | None = Field(default=None, ge=360, le=4096)
    height: int | None = Field(default=None, ge=640, le=4096)
    fps: int | None = Field(default=None, ge=15, le=60)
    top_parts: int | None = Field(default=None, ge=1, le=10)
    bottom_parts: int | None = Field(default=None, ge=1, le=10)
    mirror: bool | None = None
    top_zoom: float | None = Field(default=None, ge=1.0, le=3.0)
    top_x_offset: float | None = Field(default=None, ge=-1.0, le=1.0)
    top_y_offset: float | None = Field(default=None, ge=-1.0, le=1.0)
    contrast: float | None = Field(default=None, ge=0.5, le=2.0)
    saturation: float | None = Field(default=None, ge=0.0, le=3.0)
    brightness: float | None = Field(default=None, ge=-1.0, le=1.0)
    gamma: float | None = Field(default=None, ge=0.1, le=3.0)
    bottom_contrast: float | None = Field(default=None, ge=0.5, le=2.0)
    bottom_saturation: float | None = Field(default=None, ge=0.0, le=3.0)
    bottom_brightness: float | None = Field(default=None, ge=-0.3, le=0.3)
    bottom_gamma: float | None = Field(default=None, ge=0.5, le=2.0)
    noise_top: int | None = Field(default=None, ge=0, le=100)
    noise_bottom: int | None = Field(default=None, ge=0, le=100)
    bottom_mode: Literal["random", "sequential"] | None = None
    blur_sigma: float | None = Field(default=None, ge=0.0, le=100.0)
    blur_in: float | None = Field(default=None, ge=0.0, le=5.0)
    blur_out: float | None = Field(default=None, ge=0.0, le=5.0)
    banner_width: int | None = Field(default=None, ge=100, le=4096)
    banner_chromakey: bool | None = None
    banner_chroma_color: str | None = None
    banner_chroma_similarity: float | None = Field(default=None, ge=0.01, le=1.0)
    banner_chroma_blend: float | None = Field(default=None, ge=0.0, le=1.0)
    banner_autocrop: bool | None = None
    banner_autocrop_threshold: int | None = Field(default=None, ge=0, le=255)
    banner_mode: Literal["center", "overlay", "end", "interval_minutes", "interval_percent"] | None = None
    banner_interval_minutes: float | None = Field(default=None, gt=0.0, le=180.0)
    banner_interval_percent: float | None = Field(default=None, gt=0.0, le=100.0)
    banner_freeze: bool | None = None
    banner_anchor: (
        Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"] | None
    ) = None
    banner_x: int | None = None
    banner_y: int | None = None
    banner_opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    banner_scale: float | None = Field(default=None, ge=0.1, le=5.0)
    duck_volume: float | None = Field(default=None, ge=0.0, le=1.0)
    music_volume: float | None = Field(default=None, ge=0.0, le=1.0)
    speech_volume: float | None = Field(default=None, ge=0.1, le=3.0)
    audio_fade_in: float | None = Field(default=None, ge=0.0, le=3.0)
    audio_fade_out: float | None = Field(default=None, ge=0.0, le=3.0)
    cq: int | None = Field(default=None, ge=1, le=51)
    maxrate: str | None = None
    bufsize: str | None = None
    strip_metadata: bool | None = None
    enable_antidetect: bool | None = None
    speed_jitter: bool | None = None
    speed_jitter_min: float | None = Field(default=None, ge=0.9, le=1.0)
    speed_jitter_max: float | None = Field(default=None, ge=1.0, le=1.1)
    face_tracking: bool | None = None
    sharpen: bool | None = None
    unsharp_luma: float | None = Field(default=None, ge=0.0, le=2.0)
    unsharp_chroma: float | None = Field(default=None, ge=0.0, le=2.0)
    video_codec: Literal["h264_nvenc", "libx264", "h265_nvenc", "libx265"] | None = None
    audio_codec: Literal["aac", "mp3", "copy"] | None = None
    audio_bitrate: str | None = None
    preset: str | None = None
    enable_canvas_blur: bool | None = None
    boxblur_radius: float | None = Field(default=None, ge=1.0, le=100.0)
    boxblur_power: int | None = Field(default=None, ge=1, le=5)
    canvas_bg_color: str | None = None
    top_offset_y: int | None = None
    bottom_offset_y: int | None = None
    top_padding: int | None = None
    bottom_padding: int | None = None


class PartialSubtitlesConfig(Strict):
    enabled: bool | None = None
    font: str | None = None
    font_size: int | None = Field(default=None, ge=20, le=300)
    bold: bool | None = None
    italic: bool | None = None
    fill_color: str | None = None
    spoken_color: str | None = None
    unspoken_color: str | None = None
    outline_color: str | None = None
    outline: int | None = Field(default=None, ge=0, le=20)
    shadow: int | None = Field(default=None, ge=0, le=20)
    pop_scale: float | None = Field(default=None, ge=1.0, le=2.0)
    pop_duration: float | None = Field(default=None, ge=0.01, le=0.5)
    fade_in: bool | None = None
    fade_out: bool | None = None
    word_spacing: int | None = Field(default=None, ge=-20, le=50)
    line_spacing: float | None = Field(default=None, ge=0.5, le=3.0)
    alignment: int | None = Field(default=None, ge=1, le=9)
    position: Literal["bottom", "top", "middle"] | None = None
    margin_v: int | None = Field(default=None, ge=0)
    margin_h: int | None = Field(default=None, ge=0)
    uppercase: bool | None = None
    animation: Literal["pop", "bounce", "none"] | None = None
    pop_animation: bool | None = None
    seam_offset: int | None = None
    max_words: int | None = Field(default=None, ge=1, le=8)
    max_chars: int | None = Field(default=None, ge=8, le=60)
    max_gap: float | None = Field(default=None, ge=0.1, le=3.0)
    hold: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("spoken_color", "unspoken_color", "outline_color", "fill_color")
    @classmethod
    def valid_ass_color(cls, v: str | None) -> str | None:
        if v is not None and not ASS_COLOR.match(v):
            raise ValueError(f"цвет должен быть в формате &HAABBGGRR, получено: {v}")
        return v


class DatabaseConfig(Strict):
    """Настройки подключения к базе данных PostgreSQL."""

    enabled: bool = Field(
        default=False,
        description="Включение интеграции с базой данных PostgreSQL.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Хост сервера PostgreSQL.",
    )
    port: int = Field(
        default=5432,
        description="Порт сервера PostgreSQL.",
    )
    user: str = Field(
        default="videogen",
        description="Пользователь базы данных PostgreSQL.",
    )
    password: str = Field(
        default="",
        description="Пароль пользователя PostgreSQL.",
    )
    database: str = Field(
        default="videogen_db",
        description="Имя базы данных PostgreSQL.",
    )
    min_connections: int = Field(
        default=1,
        description="Минимальный размер пула соединений asyncpg.",
    )
    max_connections: int = Field(
        default=10,
        description="Максимальный размер пула соединений asyncpg.",
    )
    timeout: float = Field(
        default=15.0,
        description="Таймаут операций с БД в секундах.",
    )

    @model_validator(mode="before")
    @classmethod
    def env_credentials(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("host") and os.environ.get("POSTGRES_HOST"):
                data["host"] = os.environ["POSTGRES_HOST"]
            if not data.get("port") and os.environ.get("POSTGRES_PORT"):
                with contextlib.suppress(ValueError):
                    data["port"] = int(os.environ["POSTGRES_PORT"])
            if not data.get("user") and os.environ.get("POSTGRES_USER"):
                data["user"] = os.environ["POSTGRES_USER"]
            if not data.get("password") and os.environ.get("POSTGRES_PASSWORD"):
                data["password"] = os.environ["POSTGRES_PASSWORD"]
            if not data.get("database") and os.environ.get("POSTGRES_DB"):
                data["database"] = os.environ["POSTGRES_DB"]
        return data


class VideoSource(Strict):
    type: Literal["video", "pygame", "youtube"] = Field(
        description="Тип видеоисточника: video, pygame или youtube.",
    )
    source: str = Field(
        description="Путь к файлу, папке или ссылка на YouTube.",
    )
    caption: str | None = Field(
        default=None,
        description="Пользовательский шаблон или текст подписи.",
    )
    caption_name: str | None = Field(
        default=None,
        description="Отображаемое название источника для подписей.",
    )
    codename: str | None = Field(
        default=None,
        description="Кодовое название источника.",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Индивидуальная папка для готовых клипов данного источника.",
    )
    source_number: int | None = Field(
        default=None,
        description="Порядковый номер источника в каталоге.",
    )
    mode: Literal["highlights", "speech_split"] | None = Field(
        default=None,
        description="Индивидуальный режим нарезки видео.",
    )
    is_film: bool | None = Field(
        default=None,
        description="Флаг фильма (True) или сериала/видео (False).",
    )
    base_code: str | None = Field(
        default=None,
        description="Базовый иерархический код контента.",
    )
    season: int | None = Field(
        default=None,
        description="Номер сезона.",
    )
    episode: int | None = Field(
        default=None,
        description="Номер серии.",
    )

    # Настройки длительности и количества клипов конкретного источника:
    part_duration: int | None = Field(
        default=None,
        ge=10,
        le=180,
        description="Индивидуальная максимальная длительность клипа.",
    )
    min_part_duration: int | None = Field(
        default=None,
        ge=5,
        le=170,
        description="Индивидуальная минимальная длительность клипа.",
    )
    parts_count: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Индивидуальное количество клипов.",
    )

    params: PartialPipelineParams | None = Field(
        default=None,
        description="Локальные переопределения параметров конвейера.",
    )
    render: PartialRenderConfig | None = Field(
        default=None,
        description="Локальные переопределения настроек рендеринга.",
    )
    subtitles: PartialSubtitlesConfig | None = Field(
        default=None,
        description="Локальные переопределения настроек субтитров.",
    )
    assets: PartialAssetsConfig | None = Field(
        default=None,
        description="Локальные переопределения путей к ассетам.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_durations(cls, data: Any) -> Any:
        return _resolve_duration_fields(data)

    @model_validator(mode="after")
    def durations_consistent(self) -> Self:
        if (
            self.min_part_duration is not None
            and self.part_duration is not None
            and self.min_part_duration >= self.part_duration
        ):
            raise ValueError(
                f"min_part_duration ({self.min_part_duration}) должен быть меньше part_duration ({self.part_duration})"
            )
        return self


class VideosConfig(Strict):
    top_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="video", source="input/")],
        description="Список верхних видео (основной нарезаемый контент).",
    )
    bottom_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="video", source="bottom/")],
        validation_alias=AliasChoices("bottom_videos", "buttom_videos"),
        description="Список нижних видео (фоновый геймплей или заливка).",
    )

    @property
    def buttom_videos(self) -> list[VideoSource]:
        """Для обратной совместимости с опечаткой buttom_videos."""
        return self.bottom_videos

    @field_validator("top_videos")
    @classmethod
    def top_only_video(cls, v: list[VideoSource]) -> list[VideoSource]:
        if not v:
            raise ValueError("top_videos не может быть пустым")
        if any(s.type not in ("video", "youtube") for s in v):
            raise ValueError('в top_videos поддерживаются только type = "video" или "youtube"')
        return v

    @field_validator("bottom_videos")
    @classmethod
    def bottom_not_empty(cls, v: list[VideoSource]) -> list[VideoSource]:
        if not v:
            raise ValueError("bottom_videos не может быть пустым")
        return v


class AppConfig(Strict):
    videos: VideosConfig = Field(
        description="Конфигурация видеоисточников (top_videos и bottom_videos).",
    )
    params: PipelineParams = Field(
        description="Параметры выполнения конвейера (транскрипция, анализ, нарезка).",
    )
    assets: AssetsConfig = Field(
        default_factory=AssetsConfig,
        description="Ассеты (баннеры, музыка, шрифты).",
    )
    render: RenderConfig = Field(
        default_factory=RenderConfig,
        description="Параметры видеомонтажа и кодирования FFmpeg.",
    )
    subtitles: SubtitlesConfig = Field(
        default_factory=SubtitlesConfig,
        description="Стилизация и параметры генерации караоке-субтитров.",
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Опциональная конфигурация каталогизации в PostgreSQL.",
    )

    def get_effective_config(self, source: VideoSource) -> "AppConfig":
        """Создаёт эффективный конфиг для источника."""
        return get_effective_config(self, source)

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if p.suffix.lower() == ".toml":
            return cls.from_toml(p)
        return cls.model_validate(json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def from_toml(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".toml":
            data = tomllib.loads(raw)
            data.pop("telegram", None)
            return cls.model_validate(data)
        return cls.model_validate(json.loads(raw))


def _merge_section[T: BaseModel](base: T, override: BaseModel | None) -> T:
    if override is None:
        return base
    base_data = base.model_dump()
    for field_name in override.model_fields_set:
        val = getattr(override, field_name)
        if val is not None:
            base_data[field_name] = val
    if isinstance(base, PipelineParams) and isinstance(
        override, (PipelineParams, PartialPipelineParams)
    ):
        has_min = (
            "min_part_duration" in override.model_fields_set
            and override.min_part_duration is not None
        )
        has_part = (
            "part_duration" in override.model_fields_set and override.part_duration is not None
        )
        if has_part and not has_min:
            if base_data["min_part_duration"] >= base_data["part_duration"]:
                base_data["min_part_duration"] = max(5, int(base_data["part_duration"] * 0.5))
        elif (
            has_min
            and not has_part
            and base_data["min_part_duration"] >= base_data["part_duration"]
        ):
            base_data["part_duration"] = min(180, int(base_data["min_part_duration"] * 1.5))
    return type(base).model_validate(base_data)


def get_effective_config(global_config: AppConfig, source: VideoSource) -> AppConfig:
    """Создаёт эффективный AppConfig для источника, накладывая локальные настройки поверх глобальных."""
    effective_params = _merge_section(global_config.params, source.params)

    param_updates: dict[str, Any] = {}
    if source.output_dir is not None:
        param_updates["output_dir"] = source.output_dir
    if source.mode is not None:
        param_updates["mode"] = source.mode
    elif source.type == "youtube":
        param_updates["mode"] = "speech_split"
    if source.parts_count is not None:
        param_updates["parts_count"] = source.parts_count
    if source.part_duration is not None:
        param_updates["part_duration"] = source.part_duration
    if source.min_part_duration is not None:
        param_updates["min_part_duration"] = source.min_part_duration

    if param_updates:
        target_min = param_updates.get("min_part_duration", effective_params.min_part_duration)
        target_part = param_updates.get("part_duration", effective_params.part_duration)
        if target_min >= target_part:
            if "min_part_duration" not in param_updates:
                param_updates["min_part_duration"] = max(5, int(target_part * 0.5))
            elif "part_duration" not in param_updates:
                param_updates["part_duration"] = min(180, int(target_min * 1.5))
            else:
                raise ValueError(
                    f"min_part_duration ({target_min}) должен быть меньше part_duration ({target_part})"
                )
        dump = effective_params.model_dump()
        dump.update(param_updates)
        effective_params = PipelineParams.model_validate(dump)

    effective_render = _merge_section(global_config.render, source.render)
    effective_subtitles = _merge_section(global_config.subtitles, source.subtitles)
    effective_assets = _merge_section(global_config.assets, source.assets)
    return AppConfig(
        videos=global_config.videos,
        params=effective_params,
        render=effective_render,
        subtitles=effective_subtitles,
        assets=effective_assets,
        database=global_config.database,
    )


__all__ = (
    "AppConfig",
    "AssetsConfig",
    "DatabaseConfig",
    "PartialAssetsConfig",
    "PartialPipelineParams",
    "PartialRenderConfig",
    "PartialSubtitlesConfig",
    "PipelineParams",
    "RenderConfig",
    "Strict",
    "SubtitlesConfig",
    "VideoSource",
    "VideosConfig",
    "get_effective_config",
)
