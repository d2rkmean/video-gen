"""Сборка финального вертикального ролика: верх/низ, баннер, музыка (FFmpeg + NVENC)."""

import json
import random
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

from video_gen.config import AppConfig
from video_gen.gameplay import generate_pygame_video
from video_gen.models import Highlight
from video_gen.tasks.base_task import BaseTask
from video_gen.utils.audio_utils import detect_preferred_audio_stream
from video_gen.utils.face_tracker import detect_speaker_center_x

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
BAD_PATH_CHARS = set("\\':,[];")


@dataclass
class RenderSettings:
    """Параметры рендера.

    Attributes:
        width: Ширина кадра.
        height: Высота кадра. Делится на верх и низ в пропорции
            ``top_parts : bottom_parts`` (по умолчанию 1:2).
        top_parts: Доля верхнего видео в раскладке.
        bottom_parts: Доля нижнего видео в раскладке.
        fps: Частота кадров.
        mirror: Зеркально отразить верхнее видео (hflip).
        top_zoom: Во сколько раз приблизить верхнее видео кропом по центру
            (1.0 отключает).
        contrast: Контраст верхнего видео.
        saturation: Насыщенность верхнего видео.
        brightness: Яркость верхнего видео (около 0).
        gamma: Гамма верхнего видео.
        noise_top: Сила шума на верхнем видео.
        noise_bottom: Сила шума на нижнем видео.
        bottom_mode: ``"random"`` (случайное следующее) или
            ``"sequential"`` (по порядку файлов).
        blur_sigma: Сила размытия в середине ролика (на плато).
        blur_in: Секунды плавного появления блюра на стоп-кадре.
        blur_out: Секунды плавного исчезновения блюра перед концом баннера
            (0 даёт резкий выход).
        banner_width: Ширина баннера на экране.
        banner_chromakey: Вырезать ли фон баннера (хромакей). ``False``
            оставляет баннер как есть.
        banner_chroma_color: Цвет вырезаемого фона: ``0xRRGGBB``,
            ``#RRGGBB`` или имя цвета ffmpeg (``green``, ``blue``...).
        banner_chroma_similarity: Допуск по цвету (0.01-1.0). Больше значит
            вырезается более широкий диапазон оттенков.
        banner_chroma_blend: Мягкость края (0.0-1.0). 0 даёт жёсткий край.
        banner_autocrop: Автоматически обрезать баннер по рамке видимого
            содержимого (без вырезанного хромакея) до масштабирования.
            Тогда ``banner_width`` задаёт ширину именно содержимого, а не
            всего кадра с фоном. Работает только при ``banner_chromakey``.
        banner_autocrop_threshold: Порог альфы (0-255) для автообрезки.
            Пиксели с альфой не выше порога считаются фоном. Поднимите,
            если остатки фона (шум сжатия) не дают обрезать баннер плотно.
        music_volume: Громкость фоновой музыки (0.05-0.10).
        cq: Качество NVENC (меньше значит лучше и тяжелее файл).
        strip_metadata: Убрать из итогового файла метаданные: теги исходников
            (title, encoder, creation_time, геотеги и т.п.), главы,
            субтитровые и data-дорожки, а также следы FFmpeg
            (тег ``encoder``, версию библиотек).
    """

    width: int = 1080
    height: int = 1920
    fps: int = 30
    top_parts: int = 1
    bottom_parts: int = 2
    mirror: bool = True
    top_zoom: float = 1.12
    top_x_offset: float = 0.0
    top_y_offset: float = 0.0
    contrast: float = 1.08
    saturation: float = 1.16
    brightness: float = 0.01
    gamma: float = 1.02
    bottom_contrast: float = 1.10
    bottom_saturation: float = 1.25
    bottom_brightness: float = 0.01
    bottom_gamma: float = 1.02
    noise_top: int = 6
    noise_bottom: int = 5
    bottom_mode: str = "random"
    blur_sigma: float = 40.0
    blur_in: float = 0.4
    blur_out: float = 0.4
    banner_width: int = 1080
    banner_chromakey: bool = True
    banner_chroma_color: str = "0x0100f2"
    banner_chroma_similarity: float = 0.15
    banner_chroma_blend: float = 0.05
    banner_autocrop: bool = True
    banner_autocrop_threshold: int = 16
    banner_mode: str = "center"
    banner_interval_minutes: float = 1.0
    banner_interval_percent: float = 25.0
    banner_freeze: bool = True
    banner_anchor: str = "center"
    banner_x: int | None = None
    banner_y: int | None = None
    banner_opacity: float = 1.0
    banner_scale: float = 1.0
    duck_volume: float = 0.10
    music_volume: float = 0.07
    speech_volume: float = 1.0
    audio_fade_in: float = 0.3
    audio_fade_out: float = 0.5
    cq: int = 23
    maxrate: str = "6M"
    bufsize: str = "12M"
    strip_metadata: bool = True
    enable_antidetect: bool = True
    speed_jitter: bool = True
    speed_jitter_min: float = 0.985
    speed_jitter_max: float = 1.015
    face_tracking: bool = True
    sharpen: bool = True
    unsharp_luma: float = 0.8
    unsharp_chroma: float = 0.4
    enable_canvas_blur: bool = False
    boxblur_radius: float = 20.0
    boxblur_power: int = 2
    canvas_bg_color: str = "black"
    top_offset_y: int = 0
    bottom_offset_y: int = 0
    top_padding: int = 0
    bottom_padding: int = 0
    video_codec: str = "h264_nvenc"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "p4"


@lru_cache(maxsize=256)
def _probe(path: str) -> tuple[float, bool]:
    """Возвращает (длительность в секундах, есть ли аудиодорожка)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    duration = float(data["format"].get("duration", 0.0))
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return duration, has_audio


def _check_path(path: Path) -> str:
    """Проверяет, что путь безопасно подставлять в фильтр ffmpeg."""
    p = str(path.resolve())
    if BAD_PATH_CHARS & set(p):
        raise ValueError(f"В пути есть спецсимволы фильтра ffmpeg: {p}")
    return p


_COLOR_RE = re.compile(r"^(0x|#)?[0-9a-fA-F]{6}$|^[a-zA-Z]+$")


def _check_color(color: str) -> str:
    """Проверяет цвет для фильтра ffmpeg (hex или имя), чтобы не сломать граф."""
    if not _COLOR_RE.match(color):
        raise ValueError(f"Некорректный цвет хромакея: {color!r}")
    return color


def _anchor_expr(
    anchor: str, x_off: int | None = None, y_off: int | None = None
) -> tuple[str, str]:
    """Возвращает (x, y) выражения для фильтра overlay в ffmpeg."""
    anchors = {
        "top-left": ("0", "0"),
        "top-right": ("W-w", "0"),
        "bottom-left": ("0", "H-h"),
        "bottom-right": ("W-w", "H-h"),
        "center": ("(W-w)/2", "(H-h)/2"),
    }
    bx, by = anchors.get(anchor, ("(W-w)/2", "(H-h)/2"))
    if x_off is not None and x_off != 0:
        bx = f"({bx})+{x_off}" if x_off > 0 else f"({bx}){x_off}"
    if y_off is not None and y_off != 0:
        by = f"({by})+{y_off}" if y_off > 0 else f"({by}){y_off}"
    return bx, by


@lru_cache(maxsize=64)
def _probe_size(path: str) -> tuple[int, int]:
    """Возвращает (ширина, высота) первой видеодорожки."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = proc.stdout.strip().split(",")[:2]
    return int(w), int(h)


_BBOX_RE = re.compile(r"lavfi\.bbox\.(x1|y1|x2|y2)=(-?\d+)")


def _detect_banner_crop(
    path: Path,
    color: str,
    similarity: float,
    blend: float,
    threshold: int,
    duration: float,
) -> tuple[int, int, int, int] | None:
    """Ищет рамку содержимого баннера без хромакея.

    Прогоняет баннер через тот же ``chromakey``, что и при рендере,
    берёт альфа-канал и считает ``bbox`` на каждом кадре. Итоговая рамка
    есть объединение рамок всех кадров, поэтому анимированный баннер не
    обрежется в моменты, когда он больше, чем на первом кадре.

    Args:
        path: Видео баннера.
        color: Цвет хромакея.
        similarity: Допуск хромакея.
        blend: Мягкость края хромакея.
        threshold: Порог альфы (0-255), выше которого пиксель считается
            содержимым.
        duration: Сколько секунд баннера анализировать (столько же, сколько
            будет показано).

    Returns:
        ``(w, h, x, y)`` для фильтра ``crop`` или ``None``, если обрезать
        нечего (содержимое занимает весь кадр) либо анализ не удался.
    """
    vf = (
        f"chromakey=color={_check_color(color)}:similarity={similarity}:blend={blend},"
        f"format=yuva420p,alphaextract,"
        f"bbox=min_val={int(threshold)},metadata=mode=print:file=-"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(path),
        "-an",
        "-vf",
        vf,
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.warning(f"Автообрезка баннера: анализ не удался: {proc.stderr[-500:]}")
        return None

    x1 = y1 = 10**9
    x2 = y2 = -1
    for key, val in _BBOX_RE.findall(proc.stdout):
        v = int(val)
        if key == "x1":
            x1 = min(x1, v)
        elif key == "y1":
            y1 = min(y1, v)
        elif key == "x2":
            x2 = max(x2, v)
        else:
            y2 = max(y2, v)
    if x2 < 0 or y2 < 0:
        logger.warning("Автообрезка баннера: видимого содержимого не найдено, пропускаю")
        return None

    fw, fh = _probe_size(str(path))
    # Границы расширяем до чётных (требование субдискретизации цвета), чтобы
    # crop не сдвигал рамку сам и не срезал крайний пиксель.
    x1 = max(x1 & ~1, 0)
    y1 = max(y1 & ~1, 0)
    x2 = min(x2 | 1, fw - 1)
    y2 = min(y2 | 1, fh - 1)
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if (w, h) == (fw, fh):
        return None
    return w, h, x1, y1


def _smooth(x: float) -> float:
    """Smoothstep: плавный переход 0..1 с нулевой скоростью на краях."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ramps(s: RenderSettings, bd: float) -> tuple[float, float]:
    """Длительности плавного входа и выхода (сек), вписанные в баннер ``bd``."""
    r_in = max(0.0, min(s.blur_in, bd / 2))
    r_out = max(0.0, min(s.blur_out, bd - r_in))
    return r_in, r_out


@dataclass(frozen=True)
class _FreezePlan:
    """План стоп-кадра на время баннера.

    Видео замирает на кадре в исходном времени ``ta`` на ``bd`` секунд и
    затем продолжается с того же места. Ролик становится длиннее на
    ``bd``, содержимое не теряется.

    Attributes:
        ta: Момент остановки (исходное время; в выходном времени
            остановка начинается в тот же момент).
        bd: Длительность остановки (она же длительность баннера).
    """

    ta: float
    bd: float

    @property
    def extra(self) -> float:
        """На сколько секунд ролик становится длиннее."""
        return self.bd


@dataclass(frozen=True)
class _BannerPlan:
    """Расписание показов баннера, стоп-кадров и параметры автообрезки."""

    windows: list[tuple[float, float]]
    freezes: list[_FreezePlan]
    d_out: float
    banner_crop: tuple[int, int, int, int] | None
    b_audio: bool
    bd: float


def _compute_banner_timestamps(
    d: float,
    mode: str,
    interval_minutes: float,
    interval_percent: float,
) -> list[float]:
    """Вычисляет моменты показа баннера (в исходном времени клипа, сек)."""
    tas: list[float] = []
    if d <= 0:
        return tas
    if mode == "center":
        tas = [d / 2.0]
    elif mode == "end":
        tas = [d]
    elif mode == "interval_minutes" and interval_minutes > 0:
        step = interval_minutes * 60.0
        k = 1
        while k * step <= d + 1e-6:
            tas.append(min(k * step, d))
            k += 1
    elif mode == "interval_percent" and interval_percent > 0:
        if interval_percent >= 100.0:
            tas = [d]
        else:
            k = 1
            while k * interval_percent < 100.0 - 1e-6:
                tas.append(d * (k * interval_percent / 100.0))
                k += 1
    return tas


def _non_freeze_windows(tas: list[float], d: float, bd: float) -> list[tuple[float, float]]:
    """Строит непересекающиеся окна показа баннера без стоп-кадра."""
    windows: list[tuple[float, float]] = []
    for ta in tas:
        w0 = min(ta, max(0.0, d - bd)) if ta >= d - 1e-3 else ta
        w1 = min(d, w0 + bd)
        if w1 - w0 >= 0.2 and (not windows or w0 >= windows[-1][1]):
            windows.append((w0, w1))
    return windows


def _resolve_windows_and_freezes(info: dict) -> tuple[list[tuple[float, float]], list[_FreezePlan]]:
    """Извлекает окна показа и планы стоп-кадров с обратной совместимостью по ``t0``/``t1``."""
    windows: list[tuple[float, float]] | None = info.get("windows")
    if windows is None:
        t0 = float(info.get("t0", 0.0))
        t1 = float(info.get("t1", 0.0))
        windows = [(t0, t1)] if t1 > t0 else []

    freezes: list[_FreezePlan] | None = info.get("freezes")
    if freezes is None:
        single_freeze: _FreezePlan | None = info.get("freeze")
        freezes = [single_freeze] if single_freeze is not None else []
    return windows, freezes


def _plan_freeze(d: float, bd: float) -> _FreezePlan | None:
    """Ставит остановку в середину клипа длиной ``d``.

    Возвращает ``None``, если баннера нет (``bd <= 0``).
    """
    if bd <= 0.0:
        return None
    return _FreezePlan(ta=d / 2.0, bd=bd)


def _freeze_expr(p: _FreezePlan) -> str:
    """Выражение для ``setpts``: после ``ta`` все кадры сдвигаются на ``bd``.

    В потоке образуется дыра в ``bd`` секунд, а следующий за ним
    ``fps`` заполняет её повтором последнего кадра. Получается стоп-кадр.
    """
    t = "(PTS*TB)"
    return f"(if(lt({t},{p.ta:.6f}),{t},{t}+{p.bd:.6f}))/TB"


def _blur_script(
    sigma: float, t0: float, t1: float, r_in: float, r_out: float, fps: int
) -> tuple[float, str]:
    """Сценарий размытия окна баннера для ``sendcmd``.

    Sigma нарастает по smoothstep за ``r_in`` секунд после ``t0`` и спадает
    за ``r_out`` секунд до ``t1``. Команды ставятся по одной на кадр со
    смещением на полкадра, чтобы срабатывать ровно на своём кадре.

    Returns:
        Кортеж ``(стартовая sigma, строка команд)``. Строка пустая, если
        переходов нет.
    """
    floor = 0.1  # почти нулевое размытие, но без вырожденного фильтра

    def at(p: float) -> float:
        return max(sigma * p, floor)

    n_in = round(r_in * fps)
    n_out = round(r_out * fps)
    start = at(0.0) if n_in > 0 else sigma

    events: list[tuple[float, float]] = []
    for j in range(1, n_in + 1):
        events.append((t0 + (j - 0.5) / fps, at(_smooth(j / n_in))))
    ts = t1 - r_out
    for j in range(1, n_out + 1):
        events.append((ts + (j - 0.5) / fps, at(_smooth(1.0 - j / n_out))))

    script = ";".join(
        f"{tm:.4f} gblur@bl sigma {v:.3f}, gblur@bl sigmaV {v:.3f}" for tm, v in events
    )
    return start, script


def _blur_script_windows(
    sigma: float, windows: list[tuple[float, float]], s: RenderSettings
) -> tuple[float, str]:
    """Сценарий размытия для одного или нескольких окон баннера (``sendcmd``)."""
    if len(windows) == 1:
        w0, w1 = windows[0]
        r_in, r_out = _ramps(s, w1 - w0)
        return _blur_script(sigma, w0, w1, r_in, r_out, s.fps)

    floor = 0.1
    fps = s.fps

    def at(p: float) -> float:
        return max(sigma * p, floor)

    events: list[tuple[float, float]] = []
    start: float | None = None
    for w0, w1 in windows:
        r_in, r_out = _ramps(s, w1 - w0)
        n_in = round(r_in * fps)
        n_out = round(r_out * fps)
        if start is None:
            start = at(0.0) if n_in > 0 else sigma
        if n_in > 0:
            for j in range(1, n_in + 1):
                events.append((w0 + (j - 0.5) / fps, at(_smooth(j / n_in))))
        elif events:
            events.append((w0 + 0.5 / fps, sigma))
        ts = w1 - r_out
        for j in range(1, n_out + 1):
            events.append((ts + (j - 0.5) / fps, at(_smooth(1.0 - j / n_out))))

    script = ";".join(
        f"{tm:.4f} gblur@bl sigma {v:.3f}, gblur@bl sigmaV {v:.3f}" for tm, v in events
    )
    return (start if start is not None else sigma), script


_NVENC_TO_X264_PRESET = {
    "p1": "ultrafast",
    "p2": "superfast",
    "p3": "veryfast",
    "p4": "fast",
    "p5": "medium",
    "p6": "slow",
    "p7": "slower",
}


def _resolve_codec_preset(video_codec: str, preset: str) -> str:
    if "nvenc" not in video_codec and preset.lower() in _NVENC_TO_X264_PRESET:
        return _NVENC_TO_X264_PRESET[preset.lower()]
    return preset


def _video_profile(video_codec: str) -> str:
    return "main" if "265" in video_codec else "high"


class FFmpegTask(BaseTask):
    """Собирает готовый вертикальный ролик из клипа, фонового видео и баннера.

    Раскладка кадра::

        ┌──────────────┐
        │  top (клип)  │  1 часть: зеркало, кроп, цвет, шум
        ├──────────────┤
        │              │
        │ bottom (фон) │  2 части: сборная нарезка, шум
        │              │
        └──────────────┘

    В середине ролика видео полностью замирает (стоп-кадр), кадр плавно
    (за ``blur_in`` секунд) размывается, по центру экрана показывается
    баннер (видео и звук), звук основной дорожки на это время выключается.
    К концу баннера блюр так же плавно уходит, и видео продолжается с
    того же места. Остановка растягивает время, поэтому итоговый ролик
    длиннее исходного клипа на длительность баннера.
    Субтитры лежат под блюром и баннером, баннер всегда в приоритете.
    Поверх всего тихо играет случайный трек.

    Состояние конкретного ролика в ``self`` не хранится.

    Attributes:
        s: Параметры рендера.
        bottom_dir: Папка с фоновыми видео.
        banner: Видео баннера или ``None``.
        music_dir: Папка с музыкой или ``None``.
        fonts_dir: Папка со шрифтами для .ass.
        logger: Логгер с привязкой ``worker="FFmpegTask"``.
    """

    def __init__(self, config: AppConfig, settings: RenderSettings | None = None) -> None:
        super().__init__(config)
        self.s = settings or RenderSettings(**config.render.model_dump())
        self.bottom_sources = config.videos.buttom_videos
        self.banner = config.assets.banner
        self.banners_dir = config.assets.banners_dir
        self.music_dir = config.assets.music_dir
        self.fonts_dir = config.assets.fonts_dir
        self.logger = logger.bind(worker="FFmpegTask")

    def run(
        self,
        top_video: Path,
        highlight: Highlight,
        subtitles: Path | None,
        out_path: Path,
        progress_cb: Callable[[float], None] | None = None,
    ) -> Path:
        """Рендерит один ролик."""
        d = highlight.end - highlight.start
        out_path.parent.mkdir(parents=True, exist_ok=True)

        inputs_list, inputs_info = self._build_inputs(top_video, highlight, d)

        d_out = inputs_info.get("d_out", d)  # со стоп-кадром ролик длиннее клипа
        speed_factor = inputs_info.get("speed_factor", 1.0)
        d_final = d_out / speed_factor

        filter_str, has_audio = self._build_filter_graph(
            self.s, d_final, inputs_info, subtitles, self.fonts_dir
        )

        # При очистке метаданных рендерим во временный файл, затем
        # ремуксим его в итоговый (см. ``_strip_metadata``).
        strip = self.s.strip_metadata
        render_path = (
            out_path.with_name(f"{out_path.stem}.raw{out_path.suffix}") if strip else out_path
        )

        cmd = self._build_command(inputs_list, filter_str, has_audio, d_final, render_path)

        self.logger.info(
            f"Рендер {out_path.name}: клип {d:.1f} с -> ролик {d_final:.1f} с "
            f"(скорость {speed_factor:.3f}x), фон-сегментов: {len(inputs_info['bottom_idx'])}"
        )
        self._execute(cmd, render_path, total_duration=d_final, progress_cb=progress_cb)
        if strip:
            try:
                self._strip_metadata(render_path, out_path)
            finally:
                render_path.unlink(missing_ok=True)
        return out_path

    def _get_active_banner(self) -> Path | None:
        """Выбирает баннер: случайный из banners_dir, либо одиночный banner."""
        if self.banners_dir and self.banners_dir.is_dir():
            supported_exts = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".webm"}
            candidates = [
                p
                for p in self.banners_dir.iterdir()
                if p.is_file() and p.suffix.lower() in supported_exts
            ]
            if candidates:
                chosen = random.choice(candidates)
                self.logger.info(f"Выбран случайный баннер из {self.banners_dir}: {chosen.name}")
                return chosen
        if self.banner and self.banner.is_file():
            return self.banner
        return None

    def _maybe_autocrop(
        self, banner: Path, is_image: bool, probe_dur: float
    ) -> tuple[int, int, int, int] | None:
        if is_image or not (self.s.banner_chromakey and self.s.banner_autocrop):
            return None
        banner_crop = _detect_banner_crop(
            banner,
            self.s.banner_chroma_color,
            self.s.banner_chroma_similarity,
            self.s.banner_chroma_blend,
            self.s.banner_autocrop_threshold,
            probe_dur,
        )
        if banner_crop:
            cw, ch, cx, cy = banner_crop
            self.logger.info(f"Автообрезка баннера: crop={cw}:{ch}:{cx}:{cy}")
        return banner_crop

    def _setup_banner(
        self, d: float, banner_path: Path | None = None
    ) -> _BannerPlan:
        """Рассчитывает расписание показов баннера, стоп-кадры и автокроп."""
        banner = banner_path if banner_path is not None else self.banner
        if not banner or not banner.exists() or d <= 0:
            return _BannerPlan(
                windows=[], freezes=[], d_out=d, banner_crop=None, b_audio=False, bd=0.0
            )

        is_image = banner.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        if self.s.banner_mode == "overlay":
            banner_crop = self._maybe_autocrop(banner, is_image, min(5.0, d))
            return _BannerPlan(
                windows=[(0.0, d)],
                freezes=[],
                d_out=d,
                banner_crop=banner_crop,
                b_audio=False,
                bd=d,
            )

        tas = _compute_banner_timestamps(
            d,
            self.s.banner_mode,
            self.s.banner_interval_minutes,
            self.s.banner_interval_percent,
        )
        if not tas:
            return _BannerPlan(
                windows=[], freezes=[], d_out=d, banner_crop=None, b_audio=False, bd=0.0
            )

        if is_image:
            b_dur, b_audio = 3.8, False
        else:
            b_dur, b_audio = _probe(str(banner))

        if self.s.banner_mode == "end":
            bd = min(b_dur, max(1.0, d))
        elif self.s.banner_mode == "center":
            bd = min(b_dur, d * 0.6)
        else:
            bd = min(b_dur, max(0.5, d * 0.8))

        if bd <= 0.0:
            return _BannerPlan(
                windows=[], freezes=[], d_out=d, banner_crop=None, b_audio=False, bd=0.0
            )

        if self.s.banner_freeze:
            freezes = [_FreezePlan(ta=ta, bd=bd) for ta in tas]
            windows = [(ta + idx * bd, ta + (idx + 1) * bd) for idx, ta in enumerate(tas)]
            d_out = d + len(freezes) * bd
        else:
            freezes = []
            d_out = d
            windows = _non_freeze_windows(tas, d, bd)
            if not windows:
                return _BannerPlan(
                    windows=[], freezes=[], d_out=d, banner_crop=None, b_audio=False, bd=0.0
                )

        probe_dur = min(5.0, bd) if self.s.banner_mode == "end" else bd
        banner_crop = self._maybe_autocrop(banner, is_image, probe_dur)
        return _BannerPlan(
            windows=windows,
            freezes=freezes,
            d_out=d_out,
            banner_crop=banner_crop,
            b_audio=b_audio,
            bd=bd,
        )

    def _setup_antidetect(self) -> dict[str, float | int]:
        """Генерирует уникальные микро-вариации для обхода повторного контента."""
        speed_factor = 1.0
        contrast = self.s.contrast
        saturation = self.s.saturation
        brightness = self.s.brightness
        gamma = self.s.gamma
        bottom_contrast = self.s.bottom_contrast
        bottom_saturation = self.s.bottom_saturation
        bottom_brightness = self.s.bottom_brightness
        bottom_gamma = self.s.bottom_gamma
        noise_top = self.s.noise_top
        noise_bottom = self.s.noise_bottom

        if self.s.enable_antidetect:
            if self.s.speed_jitter:
                speed_factor = random.uniform(self.s.speed_jitter_min, self.s.speed_jitter_max)
            contrast *= random.uniform(0.99, 1.01)
            saturation *= random.uniform(0.99, 1.01)
            brightness += random.uniform(-0.003, 0.003)
            gamma *= random.uniform(0.99, 1.01)
            bottom_contrast *= random.uniform(0.99, 1.01)
            bottom_saturation *= random.uniform(0.99, 1.01)
            bottom_brightness += random.uniform(-0.003, 0.003)
            bottom_gamma *= random.uniform(0.99, 1.01)
            noise_top = max(0, noise_top + random.randint(-1, 1))
            noise_bottom = max(0, noise_bottom + random.randint(-1, 1))

        return {
            "speed_factor": speed_factor,
            "contrast": contrast,
            "saturation": saturation,
            "brightness": brightness,
            "gamma": gamma,
            "bottom_contrast": bottom_contrast,
            "bottom_saturation": bottom_saturation,
            "bottom_brightness": bottom_brightness,
            "bottom_gamma": bottom_gamma,
            "noise_top": noise_top,
            "noise_bottom": noise_bottom,
        }

    def _build_inputs(
        self, top_video: Path, highlight: Highlight, d: float
    ) -> tuple[list[str], dict]:
        inputs: list[str] = []
        count = 0

        def add(*args: str) -> int:
            nonlocal count
            inputs.extend(args)
            count += 1
            return count - 1

        active_banner = self._get_active_banner()
        plan = self._setup_banner(d, active_banner)

        i_top = add(
            "-sn", "-ss", f"{highlight.start:.3f}", "-t", f"{d:.3f}", "-i", str(top_video)
        )
        top_has_audio = _probe(str(top_video))[1]

        bottom_idx = [
            add("-ss", f"{off:.3f}", "-t", f"{take:.3f}", "-i", str(path))
            for path, off, take in self._plan_bottom(d)
        ]
        i_banner = None
        if active_banner and plan.windows:
            if self.s.banner_mode == "overlay":
                if active_banner.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    i_banner = add("-loop", "1", "-t", f"{d:.3f}", "-i", str(active_banner))
                else:
                    i_banner = add("-stream_loop", "-1", "-t", f"{d:.3f}", "-i", str(active_banner))
            elif active_banner.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                i_banner = add("-loop", "1", "-t", f"{plan.bd:.3f}", "-i", str(active_banner))
            else:
                i_banner = add("-t", f"{plan.bd:.3f}", "-i", str(active_banner))

        i_music = None
        music = self._pick_music()
        if music:
            i_music = add("-stream_loop", "-1", "-i", str(music))

        speaker_x = (
            detect_speaker_center_x(top_video, highlight.start, d) if self.s.face_tracking else 0.5
        )
        audio_idx = detect_preferred_audio_stream(top_video)
        anti = self._setup_antidetect()

        t0, t1 = plan.windows[0] if plan.windows else (0.0, 0.0)
        info = {
            "i_top": i_top,
            "top_has_audio": top_has_audio,
            "audio_idx": audio_idx,
            "bottom_idx": bottom_idx,
            "i_banner": i_banner,
            "active_banner": active_banner,
            "banner_crop": plan.banner_crop,
            "b_audio": plan.b_audio,
            "banner_mode": self.s.banner_mode,
            "banner_freeze": self.s.banner_freeze,
            "t0": t0,
            "t1": t1,
            "windows": plan.windows,
            "freeze": plan.freezes[0] if plan.freezes else None,
            "freezes": plan.freezes,
            "d_src": d,
            "d_out": plan.d_out,
            "i_music": i_music,
            "speaker_x": speaker_x,
            **anti,
        }
        return inputs, info

    @staticmethod
    def _split_heights(s: RenderSettings) -> tuple[int, int]:
        """Высоты верха и низа по пропорции ``top_parts : bottom_parts``."""
        total = s.top_parts + s.bottom_parts
        top_h = (s.height * s.top_parts // total) // 2 * 2
        return top_h, s.height - top_h

    @staticmethod
    def _build_timed_banner_chain(
        base_bn: str,
        cur: str,
        bx: str,
        by: str,
        windows: list[tuple[float, float]],
    ) -> list[str]:
        """Строит цепочку оверлеев баннера для одного или нескольких окон показа."""
        if len(windows) == 1:
            w0, w1 = windows[0]
            en = f"enable='between(t,{w0:.3f},{w1:.3f})'"
            return [
                f"{base_bn},setpts=PTS-STARTPTS+{w0:.3f}/TB[bn]",
                f"[{cur}][bn]overlay={bx}:{by}:eof_action=pass:{en}[ov]",
            ]
        f: list[str] = []
        split_labels = "".join(f"[bns{idx}]" for idx in range(len(windows)))
        f.append(f"{base_bn},split={len(windows)}{split_labels}")
        for idx, (w0, w1) in enumerate(windows):
            f.append(f"[bns{idx}]setpts=PTS-STARTPTS+{w0:.3f}/TB[bn{idx}]")
            out_lbl = "ov" if idx == len(windows) - 1 else f"ov{idx}"
            f.append(
                f"[{cur}][bn{idx}]overlay={bx}:{by}:eof_action=pass:"
                f"enable='between(t,{w0:.3f},{w1:.3f})'[{out_lbl}]"
            )
            cur = out_lbl
        return f

    @staticmethod
    def _build_banner_overlay(
        s: RenderSettings,
        cur: str,
        info: dict,
    ) -> tuple[list[str], str]:
        """Формирует оверлей баннера, стоп-кадр и размытие фона."""
        i_banner = info.get("i_banner")
        if i_banner is None:
            return [], cur

        banner_mode = info.get("banner_mode", s.banner_mode)
        active_banner: Path | None = info.get("active_banner")
        is_png = active_banner is not None and active_banner.suffix.lower() == ".png"
        chroma = (
            f"chromakey=color={_check_color(s.banner_chroma_color)}:"
            f"similarity={s.banner_chroma_similarity}:blend={s.banner_chroma_blend},"
            if s.banner_chromakey and not is_png
            else ""
        )
        crop = (
            f"crop={info['banner_crop'][0]}:{info['banner_crop'][1]}:"
            f"{info['banner_crop'][2]}:{info['banner_crop'][3]},"
            if info.get("banner_crop")
            else ""
        )

        scaled_w = max(16, (int(s.banner_width * s.banner_scale) // 2) * 2)
        opacity_filter = (
            f",colorchannelmixer=aa={max(0.0, min(1.0, s.banner_opacity)):.3f}"
            if s.banner_opacity < 1.0
            else ""
        )
        bx, by = _anchor_expr(s.banner_anchor, s.banner_x, s.banner_y)
        base_bn = (
            f"[{i_banner}:v]{crop}scale={scaled_w}:-2,{chroma}"
            f"format=yuva420p{opacity_filter},fps={s.fps}"
        )

        if banner_mode == "overlay":
            return [
                f"{base_bn},setpts=PTS-STARTPTS[bn]",
                f"[{cur}][bn]overlay={bx}:{by}:eof_action=repeat[ov]",
            ], "ov"

        windows, freezes = _resolve_windows_and_freezes(info)
        if not windows:
            return [], cur

        f: list[str] = []
        if freezes:
            d_src = info.get("d_src")
            if d_src is not None and freezes[-1].ta >= float(d_src) - max(
                1e-3, 1.5 / max(s.fps, 1)
            ):
                f.append(f"[{cur}]tpad=stop_mode=clone:stop_duration={freezes[-1].bd:.3f}[padded]")
                cur = "padded"
            freeze_expr = (
                _freeze_expr(freezes[0])
                if len(freezes) == 1
                else f"((PTS*TB)+{'+'.join(f'{p.bd:.6f}*gte((PTS*TB),{p.ta:.6f})' for p in freezes)})/TB"
            )
            f.append(f"[{cur}]setpts='{freeze_expr}',fps={s.fps}[frozen]")
            cur = "frozen"

        if info.get("banner_freeze", s.banner_freeze) and s.blur_sigma > 0:
            en = f"enable='{'+'.join(f'between(t,{w0:.3f},{w1:.3f})' for w0, w1 in windows)}'"
            sigma0, script = _blur_script_windows(s.blur_sigma, windows, s)
            blur = f"gblur@bl=sigma={sigma0:.3f}:sigmaV={sigma0:.3f}:{en}"
            if script:
                blur = f"sendcmd=c='{script}',{blur}"
            f.append(f"[{cur}]{blur}[blurred]")
            cur = "blurred"

        f.extend(FFmpegTask._build_timed_banner_chain(base_bn, cur, bx, by, windows))
        return f, "ov"

    @staticmethod
    def _build_video_graph(  # noqa: PLR0915
        s: RenderSettings,
        info: dict,
        subtitles: Path | None,
        fonts_dir: Path,
    ) -> list[str]:
        """Строит видеотракт: кроп верха с трекингом лица, фон, субтитры, баннер/блюр."""
        top_h, bottom_h = FFmpegTask._split_heights(s)
        f: list[str] = []
        mirror = "hflip," if s.mirror else ""

        i_top = info["i_top"]
        speaker_x = info.get("speaker_x", 0.5)
        contrast = info.get("contrast", s.contrast)
        saturation = info.get("saturation", s.saturation)
        brightness = info.get("brightness", s.brightness)
        gamma = info.get("gamma", s.gamma)
        bottom_contrast = info.get("bottom_contrast", s.bottom_contrast)
        bottom_saturation = info.get("bottom_saturation", s.bottom_saturation)
        bottom_brightness = info.get("bottom_brightness", s.bottom_brightness)
        bottom_gamma = info.get("bottom_gamma", s.bottom_gamma)
        noise_top = info.get("noise_top", s.noise_top)
        noise_bottom = info.get("noise_bottom", s.noise_bottom)

        has_custom_canvas = (
            s.enable_canvas_blur
            or s.top_padding > 0
            or s.bottom_padding > 0
            or s.top_offset_y != 0
            or s.bottom_offset_y != 0
        )

        crop_x_expr = f"min(max(0,iw*{speaker_x:.3f}-ow/2),iw-ow)"
        crop_top = f"crop=w=iw/{s.top_zoom}:h=ih/{s.top_zoom}:x='{crop_x_expr}':y='(ih-oh)/2'"

        if has_custom_canvas:
            top_w = max(16, (s.width - 2 * s.top_padding) // 2 * 2)
            top_actual_h = max(16, (top_h - 2 * s.top_padding) // 2 * 2)
            top_y = s.top_padding + s.top_offset_y

            bot_w = max(16, (s.width - 2 * s.bottom_padding) // 2 * 2)
            bot_actual_h = max(16, (bottom_h - 2 * s.bottom_padding) // 2 * 2)
            bot_y = top_h + s.bottom_padding + s.bottom_offset_y

            f.append(f"[{i_top}:v]split[top_raw][bg_raw]")
            if s.enable_canvas_blur:
                f.append(
                    f"[bg_raw]scale={s.width}:{s.height}:force_original_aspect_ratio=increase,"
                    f"crop={s.width}:{s.height},"
                    f"boxblur={s.boxblur_radius}:{s.boxblur_power},"
                    f"fps={s.fps},setsar=1,format=yuv420p[bg_canvas]"
                )
            else:
                f.append(
                    f"[bg_raw]scale={s.width}:{s.height},"
                    f"drawbox=t=fill:c={_check_color(s.canvas_bg_color)},"
                    f"fps={s.fps},setsar=1,format=yuv420p[bg_canvas]"
                )

            f.append(
                f"[top_raw]{mirror}{crop_top},"
                f"scale={top_w}:{top_actual_h}:force_original_aspect_ratio=increase,"
                f"crop={top_w}:{top_actual_h},"
                f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:"
                f"brightness={brightness:.3f}:gamma={gamma:.3f},"
                f"noise=alls={noise_top}:allf=t+u,fps={s.fps},setsar=1,format=yuv420p[top]"
            )

            bottom_idx = info["bottom_idx"]
            for k, i in enumerate(bottom_idx):
                f.append(
                    f"[{i}:v]scale={bot_w}:{bot_actual_h}:force_original_aspect_ratio=increase,"
                    f"crop={bot_w}:{bot_actual_h},fps={s.fps},setsar=1,format=yuv420p[b{k}]"
                )
            joined = "".join(f"[b{k}]" for k in range(len(bottom_idx)))
            f.append(
                f"{joined}concat=n={len(bottom_idx)}:v=1:a=0,"
                f"eq=contrast={bottom_contrast:.3f}:saturation={bottom_saturation:.3f}:"
                f"brightness={bottom_brightness:.3f}:gamma={bottom_gamma:.3f},"
                f"noise=alls={noise_bottom}:allf=t+u[bot]"
            )

            f.append(f"[bg_canvas][top]overlay=(W-w)/2:{top_y}[bg_top]")
            f.append(f"[bg_top][bot]overlay=(W-w)/2:{bot_y}[stack]")
            cur = "stack"
        else:
            f.append(
                f"[{i_top}:v]{mirror}{crop_top},"
                f"scale={s.width}:{top_h}:force_original_aspect_ratio=increase,"
                f"crop={s.width}:{top_h},"
                f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:"
                f"brightness={brightness:.3f}:gamma={gamma:.3f},"
                f"noise=alls={noise_top}:allf=t+u,fps={s.fps},setsar=1,format=yuv420p[top]"
            )

            bottom_idx = info["bottom_idx"]
            for k, i in enumerate(bottom_idx):
                f.append(
                    f"[{i}:v]scale={s.width}:{bottom_h}:force_original_aspect_ratio=increase,"
                    f"crop={s.width}:{bottom_h},fps={s.fps},setsar=1,format=yuv420p[b{k}]"
                )
            joined = "".join(f"[b{k}]" for k in range(len(bottom_idx)))
            f.append(
                f"{joined}concat=n={len(bottom_idx)}:v=1:a=0,"
                f"eq=contrast={bottom_contrast:.3f}:saturation={bottom_saturation:.3f}:"
                f"brightness={bottom_brightness:.3f}:gamma={bottom_gamma:.3f},"
                f"noise=alls={noise_bottom}:allf=t+u[bot]"
            )

            f.append("[top][bot]vstack=inputs=2[stack]")
            cur = "stack"

        if s.sharpen:
            f.append(
                f"[{cur}]unsharp=lx=5:ly=5:la={s.unsharp_luma:.2f}:"
                f"cx=3:cy=3:ca={s.unsharp_chroma:.2f}[sharp]"
            )
            cur = "sharp"

        # Субтитры идут до блюра и баннера: баннер всегда поверх них.
        if subtitles:
            ass = f"ass={_check_path(subtitles)}"
            if fonts_dir.exists():
                ass += f":fontsdir={_check_path(fonts_dir)}"
            f.append(f"[{cur}]{ass}[sub]")
            cur = "sub"

        banner_filters, cur = FFmpegTask._build_banner_overlay(s, cur, info)
        f.extend(banner_filters)

        speed_factor = info.get("speed_factor", 1.0)
        if abs(speed_factor - 1.0) > 0.001:
            f.append(f"[{cur}]setpts=PTS/{speed_factor:.4f}[vspd]")
            cur = "vspd"

        f.append(f"[{cur}]format=yuv420p[vout]")
        return f

    @staticmethod
    def _build_frozen_speech_audio(
        a_in: str,
        fmt: str,
        d: float,
        d_src_val: float | None,
        freezes: list[_FreezePlan],
    ) -> list[str]:
        """Разрезает речь по стоп-кадрам и вставляет тишину на время баннеров."""
        f: list[str] = []
        fade = 0.05
        prev_ta = 0.0
        timeline: list[tuple[str, float, float | None]] = []
        for p in freezes:
            if p.ta - prev_ta > 0.005:
                timeline.append(("seg", prev_ta, p.ta))
            timeline.append(("gap", p.bd, None))
            prev_ta = p.ta
        if d_src_val is None or d_src_val - prev_ta > 0.005:
            timeline.append(("seg", prev_ta, None))

        n_segs = sum(1 for item in timeline if item[0] == "seg")
        if n_segs > 1:
            split_lbls = "".join(f"[tpa{k}]" for k in range(n_segs))
            f.append(f"{a_in}{fmt},asplit={n_segs}{split_lbls}")
        elif n_segs == 1:
            f.append(f"{a_in}{fmt}[tpa0]")

        concat_parts: list[str] = []
        seg_counter = 0
        gap_counter = 0
        for kind, v1, v2 in timeline:
            if kind == "seg":
                in_lbl = f"[tpa{seg_counter}]"
                out_lbl = f"tseg{seg_counter}"
                seg_counter += 1
                if v2 is not None:
                    seg_dur = v2 - v1
                    seg_fade = min(fade, seg_dur / 3.0)
                    trim_expr = (
                        f"atrim=end={v2:.3f}" if v1 == 0.0 else f"atrim=start={v1:.3f}:end={v2:.3f}"
                    )
                    fades = (
                        f"afade=t=in:d={seg_fade:.3f},"
                        f"afade=t=out:st={max(seg_dur - seg_fade, 0):.3f}:d={seg_fade:.3f}"
                    )
                else:
                    trim_expr = f"atrim=start={v1:.3f}"
                    fades = f"afade=t=in:d={fade:.3f}"
                f.append(f"{in_lbl}{trim_expr},asetpts=PTS-STARTPTS,{fades}[{out_lbl}]")
                concat_parts.append(f"[{out_lbl}]")
            else:
                out_lbl = f"tgap{gap_counter}"
                gap_counter += 1
                f.append(f"anullsrc=r=44100:cl=stereo:d={v1:.3f}[{out_lbl}]")
                concat_parts.append(f"[{out_lbl}]")
        f.append(
            f"{''.join(concat_parts)}concat=n={len(concat_parts)}:v=0:a=1,"
            f"afade=t=out:st={max(d - 0.1, 0):.3f}:d=0.1[am]"
        )
        return f

    @staticmethod
    def _build_banner_audio_filters(
        i_banner: int, fmt: str, windows: list[tuple[float, float]]
    ) -> tuple[list[str], list[str]]:
        """Строит аудио-фильтры баннера для одного или нескольких окон показа."""
        if len(windows) == 1:
            w0 = windows[0][0]
            return [f"[{i_banner}:a]{fmt},adelay={int(w0 * 1000)}:all=1[ab]"], ["[ab]"]
        asplit_labels = "".join(f"[abs{idx}]" for idx in range(len(windows)))
        f = [f"[{i_banner}:a]{fmt},asplit={len(windows)}{asplit_labels}"]
        labels: list[str] = []
        for idx, (w0, _w1) in enumerate(windows):
            f.append(f"[abs{idx}]adelay={int(w0 * 1000)}:all=1[ab{idx}]")
            labels.append(f"[ab{idx}]")
        return f, labels

    @staticmethod
    def _build_audio_graph(
        s: RenderSettings,
        d: float,
        info: dict,
    ) -> tuple[list[str], bool]:
        """Строит звуковой тракт: основной звук, звук баннера, музыка, темпо-джиттер."""
        fmt = "aformat=sample_rates=44100:channel_layouts=stereo:sample_fmts=fltp"
        f: list[str] = []
        labels: list[str] = []
        i_top = info["i_top"]
        speed_factor = info.get("speed_factor", 1.0)
        windows, freezes = _resolve_windows_and_freezes(info)

        if info["top_has_audio"]:
            audio_idx = info.get("audio_idx", 0)
            a_in = f"[{i_top}:a:{audio_idx}]"
            if freezes:
                f.extend(
                    FFmpegTask._build_frozen_speech_audio(
                        a_in, fmt, d, info.get("d_src"), freezes
                    )
                )
            else:
                duck_filter = ""
                banner_mode = info.get("banner_mode", s.banner_mode)
                if windows and banner_mode != "overlay" and s.duck_volume < 1.0:
                    duck_en = "+".join(f"between(t,{w0:.3f},{w1:.3f})" for w0, w1 in windows)
                    duck_filter = f",volume='{s.duck_volume}':enable='{duck_en}'"
                f.append(
                    f"{a_in}{fmt},afade=t=in:d=0.05,"
                    f"afade=t=out:st={max(d - 0.1, 0):.3f}:d=0.1{duck_filter}[am]"
                )
            labels.append("[am]")

        i_banner = info.get("i_banner")
        if i_banner is not None and info.get("b_audio") and windows:
            b_filters, b_labels = FFmpegTask._build_banner_audio_filters(i_banner, fmt, windows)
            f.extend(b_filters)
            labels.extend(b_labels)

        i_music = info.get("i_music")
        if i_music is not None:
            f.append(
                f"[{i_music}:a]{fmt},volume={s.music_volume},"
                f"afade=t=out:st={max(d - 1, 0):.3f}:d=1[amus]"
            )
            labels.append("[amus]")

        if len(labels) > 1:
            f.append(
                f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[amixed]"
            )
            cur_a = "[amixed]"
        elif labels:
            f.append(f"{labels[0]}anull[amixed]")
            cur_a = "[amixed]"
        else:
            return f, False

        if abs(speed_factor - 1.0) > 0.001:
            f.append(f"{cur_a}atempo={speed_factor:.4f}[aout]")
        else:
            f.append(f"{cur_a}anull[aout]")

        return f, True

    @staticmethod
    def _build_filter_graph(
        s: RenderSettings,
        d: float,
        info: dict,
        subtitles: Path | None,
        fonts_dir: Path,
    ) -> tuple[str, bool]:
        """Собирает полный ``filter_complex``."""
        vf = FFmpegTask._build_video_graph(s, info, subtitles, fonts_dir)
        af, has_audio = FFmpegTask._build_audio_graph(s, d, info)
        return ";".join(vf + af), has_audio

    def _build_command(
        self,
        inputs: list[str],
        filter_str: str,
        has_audio: bool,
        d: float,
        out_path: Path,
    ) -> list[str]:
        s = self.s
        preset = _resolve_codec_preset(s.video_codec, s.preset)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            *inputs,
            "-filter_complex",
            filter_str,
            "-map",
            "[vout]",
            *(["-map", "[aout]"] if has_audio else ["-an"]),
            "-t",
            f"{d:.3f}",
            "-c:v",
            s.video_codec,
            "-preset",
            preset,
            *(
                ["-rc", "vbr", "-cq", str(s.cq), "-b:v", "0"]
                if "nvenc" in s.video_codec
                else ["-crf", str(s.cq)]
            ),
            "-maxrate",
            str(s.maxrate),
            "-bufsize",
            str(s.bufsize),
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            _video_profile(s.video_codec),
            "-r",
            str(s.fps),
            "-c:a",
            s.audio_codec,
            *(["-b:a", str(s.audio_bitrate)] if s.audio_codec != "copy" else []),
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            *self._metadata_args(),
            str(out_path),
        ]
        self.logger.debug(" ".join(cmd))
        return cmd

    def _metadata_args(self) -> list[str]:
        if not self.s.strip_metadata:
            return []
        return [
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-sn",
            "-dn",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-flags:a",
            "+bitexact",
        ]

    def _strip_metadata(self, src: Path, dst: Path) -> None:
        """Ремукс без перекодирования для удаления остаточных метаданных и SEI-юнитов."""
        sei_filter = (
            "filter_units=remove_types=39|40"
            if "265" in self.s.video_codec
            else "filter_units=remove_types=6"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-i",
            str(src),
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-sn",
            "-dn",
            "-fflags",
            "+bitexact",
            "-bsf:v",
            sei_filter,
            "-movflags",
            "+faststart",
            str(dst),
        ]
        self.logger.debug(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg (очистка метаданных) упал:\n{proc.stderr[-2000:]}")

    def _execute(
        self,
        cmd: list[str],
        out_path: Path,
        total_duration: float = 0.0,
        progress_cb: Any = None,
    ) -> None:
        cmd_with_prog = list(cmd)
        if progress_cb and "-progress" not in cmd:
            cmd_with_prog = [*cmd[:-1], "-progress", "pipe:1", "-nostats", cmd[-1]]

        proc = subprocess.Popen(
            cmd_with_prog,
            stdout=subprocess.PIPE if progress_cb else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stderr_chunks: list[str] = []

        def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            stderr_chunks.append(proc.stderr.read())

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            if progress_cb and proc.stdout:
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k == "out_time_ms" and total_duration > 0:
                            try:
                                cur_time = int(v) / 1_000_000.0
                                pct = min(100.0, max(0.0, (cur_time / total_duration) * 100.0))
                                progress_cb(pct)
                            except ValueError:
                                pass
                        elif k == "progress" and v == "end":
                            progress_cb(100.0)
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
            stderr_thread.join(timeout=5.0)

        stderr = "".join(stderr_chunks)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg упал:\n{stderr[-2000:]}")
        self.logger.info(f"Готово: {out_path}")

    def _bottom_files(self, need: float = 60.0) -> list[Path]:
        files: list[Path] = []
        for src in self.bottom_sources:
            if src.type == "pygame":
                _, bottom_h = self._split_heights(self.s)
                py_duration = (
                    float(src.part_duration) if src.part_duration else max(60.0, need + 10.0)
                )
                py_path = generate_pygame_video(
                    preset=src.source or "runner",
                    width=self.s.width,
                    height=bottom_h,
                    fps=self.s.fps,
                    duration=max(py_duration, need + 10.0),
                )
                files.append(py_path)
            elif src.type == "video":
                p = Path(src.source)
                if p.is_dir():
                    files += sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT)
                elif p.is_file():
                    files.append(p)
            else:
                self.logger.warning(f"Пропускаю источник type='{src.type}': неизвестный тип")
        return files

    def _plan_bottom(self, need: float) -> list[tuple[Path, float, float]]:
        """Набирает фоновые видео на ``need`` секунд.

        Видео берётся до конца, затем следующее (по порядку или случайное,
        но не то же самое подряд). Если видео длиннее остатка, стартовая
        точка выбирается случайно.

        Returns:
            Список ``(путь, смещение, длительность)``.

        Raises:
            FileNotFoundError: Если в папке нет видео.
        """
        files = self._bottom_files(need)
        if not files:
            raise FileNotFoundError("Нет фоновых видео в источниках")

        idx = random.randrange(len(files))
        plan: list[tuple[Path, float, float]] = []
        total = 0.0
        last: Path | None = None

        for _ in range(50):  # защита от бесконечного цикла на битых файлах
            if total >= need - 0.01:
                break
            if self.s.bottom_mode == "sequential":
                path = files[idx % len(files)]
                idx += 1
            else:
                path = random.choice([x for x in files if x != last] or files)
            last = path

            dur, _ = _probe(str(path))
            if dur <= 0.5:
                continue
            remaining = need - total
            offset = random.uniform(0, dur - remaining) if dur > remaining else 0.0
            take = min(dur - offset, remaining)
            plan.append((path, offset, take))
            total += take

        if total < need - 0.5:
            raise RuntimeError("Не удалось набрать фоновое видео нужной длины")
        return plan

    def _pick_music(self) -> Path | None:
        """Случайный трек из ``music_dir`` или ``None``, если музыки нет."""
        if not self.music_dir or not self.music_dir.exists():
            return None
        tracks = [p for p in self.music_dir.iterdir() if p.suffix.lower() in AUDIO_EXT]
        return random.choice(tracks) if tracks else None
