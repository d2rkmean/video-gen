"""Сборка финального вертикального ролика: верх/низ, баннер, музыка (FFmpeg + NVENC)."""

import json
import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from video_gen.config import AppConfig
from video_gen.models import Highlight
from video_gen.tasks.base_task import BaseTask

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
BAD_PATH_CHARS = set("\\':,[];")


@dataclass
class RenderSettings:
    """Параметры рендера.

    Attributes:
        width: Ширина кадра.
        height: Высота кадра. Верхняя и нижняя половины по ``height // 2``.
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
        blur_sigma: Сила размытия в середине ролика.
        banner_width: Ширина баннера на экране.
        duck_volume: Громкость основной дорожки, пока идёт баннер.
        music_volume: Громкость фоновой музыки (0.05-0.10).
        cq: Качество NVENC (меньше значит лучше и тяжелее файл).
    """

    width: int = 1080
    height: int = 1920
    fps: int = 30
    mirror: bool = True
    top_zoom: float = 1.12
    contrast: float = 1.08
    saturation: float = 1.15
    brightness: float = 0.02
    gamma: float = 1.03
    noise_top: int = 6
    noise_bottom: int = 5
    bottom_mode: str = "random"
    blur_sigma: float = 40.0
    banner_width: int = 900
    duck_volume: float = 0.10
    music_volume: float = 0.07
    cq: int = 23


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


class FFmpegTask(BaseTask):
    """Собирает готовый вертикальный ролик из клипа, фонового видео и баннера.

    Раскладка кадра::

        ┌──────────────┐
        │  top (клип)  │  зеркало, кроп, цвет, шум
        ├──────────────┤  <- субтитры по центру экрана
        │ bottom (фон) │  сборная нарезка, шум
        └──────────────┘

    В середине ролика кадр размывается, по центру экрана показывается
    баннер (видео и звук), основная дорожка на это время приглушается.
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
        self.music_dir = config.assets.music_dir
        self.fonts_dir = config.assets.fonts_dir
        self.logger = logger.bind(worker="FFmpegTask")

    def run(
        self,
        top_video: Path,
        highlight: Highlight,
        subtitles: Path | None,
        out_path: Path,
    ) -> Path:
        """Рендерит один ролик."""
        d = highlight.end - highlight.start
        out_path.parent.mkdir(parents=True, exist_ok=True)

        inputs_list, inputs_info = self._build_inputs(top_video, highlight, d)

        filter_str, has_audio = self._build_filter_graph(
            self.s, d, inputs_info, subtitles, self.fonts_dir
        )

        cmd = self._build_command(inputs_list, filter_str, has_audio, d, out_path)

        self.logger.info(
            f"Рендер {out_path.name}: {d:.1f} с, фон-сегментов: {len(inputs_info['bottom_idx'])}"
        )
        self._execute(cmd, out_path)
        return out_path

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

        i_top = add("-ss", f"{highlight.start:.3f}", "-t", f"{d:.3f}", "-i", str(top_video))
        top_has_audio = _probe(str(top_video))[1]

        bottom_idx = [
            add("-ss", f"{off:.3f}", "-t", f"{take:.3f}", "-i", str(path))
            for path, off, take in self._plan_bottom(d)
        ]

        i_banner = b_audio = None
        t0 = t1 = 0.0
        if self.banner and self.banner.exists():
            b_dur, b_audio = _probe(str(self.banner))
            bd = min(b_dur, d * 0.6)
            t0 = (d - bd) / 2
            t1 = t0 + bd
            i_banner = add("-t", f"{bd:.3f}", "-i", str(self.banner))

        i_music = None
        music = self._pick_music()
        if music:
            i_music = add("-stream_loop", "-1", "-i", str(music))

        info = {
            "i_top": i_top,
            "top_has_audio": top_has_audio,
            "bottom_idx": bottom_idx,
            "i_banner": i_banner,
            "b_audio": b_audio,
            "t0": t0,
            "t1": t1,
            "i_music": i_music,
        }
        return inputs, info

    @staticmethod
    def _build_filter_graph(
        s: RenderSettings,
        d: float,
        info: dict,
        subtitles: Path | None,
        fonts_dir: Path,
    ) -> tuple[str, bool]:
        half = s.height // 2
        f: list[str] = []
        mirror = "hflip," if s.mirror else ""

        i_top = info["i_top"]
        f.append(
            f"[{i_top}:v]{mirror}crop=iw/{s.top_zoom}:ih/{s.top_zoom},"
            f"scale={s.width}:{half}:force_original_aspect_ratio=increase,"
            f"crop={s.width}:{half},"
            f"eq=contrast={s.contrast}:saturation={s.saturation}:"
            f"brightness={s.brightness}:gamma={s.gamma},"
            f"noise=alls={s.noise_top}:allf=t+u,fps={s.fps},setsar=1,format=yuv420p[top]"
        )

        bottom_idx = info["bottom_idx"]
        for k, i in enumerate(bottom_idx):
            f.append(
                f"[{i}:v]scale={s.width}:{half}:force_original_aspect_ratio=increase,"
                f"crop={s.width}:{half},fps={s.fps},setsar=1,format=yuv420p[b{k}]"
            )
        joined = "".join(f"[b{k}]" for k in range(len(bottom_idx)))
        f.append(
            f"{joined}concat=n={len(bottom_idx)}:v=1:a=0,noise=alls={s.noise_bottom}:allf=t+u[bot]"
        )

        f.append("[top][bot]vstack=inputs=2[stack]")
        cur = "stack"

        if subtitles:
            ass = f"ass={_check_path(subtitles)}"
            if fonts_dir.exists():
                ass += f":fontsdir={_check_path(fonts_dir)}"
            f.append(f"[{cur}]{ass}[sub]")
            cur = "sub"

        i_banner = info["i_banner"]
        t0 = info["t0"]
        t1 = info["t1"]
        if i_banner is not None:
            en = f"enable='between(t,{t0:.3f},{t1:.3f})'"
            f.append(f"[{cur}]split[main][tmp]")
            f.append(
                f"[tmp]trim=start={t0:.3f}:end={t1:.3f},"
                f"setpts=PTS-STARTPTS+{t0:.3f}/TB,gblur=sigma={s.blur_sigma}[blur]"
            )
            f.append(f"[main][blur]overlay=0:0:eof_action=pass:{en}[blurred]")
            f.append(
                f"[{i_banner}:v]scale={s.banner_width}:-2,fps={s.fps},"
                f"setpts=PTS-STARTPTS+{t0:.3f}/TB[bn]"
            )
            f.append(f"[blurred][bn]overlay=(W-w)/2:(H-h)/2:eof_action=pass:{en}[ov]")
            cur = "ov"

        f.append(f"[{cur}]format=yuv420p[vout]")

        fmt = "aformat=sample_rates=44100:channel_layouts=stereo"
        labels: list[str] = []

        if info["top_has_audio"]:
            duck = (
                f",volume='if(between(t,{t0:.3f},{t1:.3f}),{s.duck_volume},1)':eval=frame"
                if i_banner is not None
                else ""
            )
            f.append(
                f"[{i_top}:a]{fmt},afade=t=in:d=0.05,"
                f"afade=t=out:st={max(d - 0.1, 0):.3f}:d=0.1{duck}[am]"
            )
            labels.append("[am]")

        if i_banner is not None and info["b_audio"]:
            f.append(f"[{i_banner}:a]{fmt},adelay={int(t0 * 1000)}:all=1[ab]")
            labels.append("[ab]")

        i_music = info["i_music"]
        if i_music is not None:
            f.append(
                f"[{i_music}:a]{fmt},volume={s.music_volume},"
                f"afade=t=out:st={max(d - 1, 0):.3f}:d=1[amus]"
            )
            labels.append("[amus]")

        if len(labels) > 1:
            f.append(
                f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[aout]"
            )
        elif labels:
            f.append(f"{labels[0]}anull[aout]")

        return ";".join(f), bool(labels)

    def _build_command(
        self,
        inputs: list[str],
        filter_str: str,
        has_audio: bool,
        d: float,
        out_path: Path,
    ) -> list[str]:
        s = self.s
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
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(s.cq),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-r",
            str(s.fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        self.logger.debug(" ".join(cmd))
        return cmd

    def _execute(self, cmd: list[str], out_path: Path) -> None:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg упал:\n{proc.stderr[-2000:]}")
        self.logger.info(f"Готово: {out_path}")

    def _bottom_files(self) -> list[Path]:
        files: list[Path] = []
        for src in self.bottom_sources:
            if src.type != "video":
                self.logger.warning(f"Пропускаю источник type='{src.type}': пока не реализовано")
                continue
            p = Path(src.source)
            if p.is_dir():
                files += sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT)
            elif p.is_file():
                files.append(p)
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
        files = self._bottom_files()
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
