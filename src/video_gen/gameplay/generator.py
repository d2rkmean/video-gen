"""Процедурный генератор фоновых геймплейных видео на PyGame.

Работает полностью headless (через виртуальный драйвер SDL 'dummy') без необходимости
в X11/Wayland дисплее. Рендерит кадры напрямую в поток FFmpeg в формате MP4.
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Any

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from loguru import logger

CACHE_DIR = Path("cache/pygame")


class RunnerScene:
    """Процедурный неоновый бесконечный раннер (геймплей с препятствиями и частицами)."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.w = width
        self.h = height
        self.fps = fps
        self.player_x: float = float(width // 2)
        self.player_y = int(height * 0.78)
        self.player_vx = 0.0
        self.target_x: float = self.player_x
        self.obstacles: list[dict[str, float]] = []
        self.particles: list[dict[str, Any]] = []
        self.score = 0
        self.speed = 14.0
        self.grid_offset = 0.0
        self.font = pygame.font.Font(None, max(24, int(width * 0.045)))

    def update(self, frame: int) -> None:
        self.score += int(self.speed)
        self.grid_offset = (self.grid_offset + self.speed) % 80

        if frame % int(self.fps * 0.8) == 0:
            lanes = [self.w * 0.25, self.w * 0.5, self.w * 0.75]
            self.target_x = int(random.choice(lanes))

        self.player_x += (self.target_x - self.player_x) * 0.15

        if frame % int(self.fps * 0.6) == 0:
            lane = random.choice([self.w * 0.25, self.w * 0.5, self.w * 0.75])
            self.obstacles.append(
                {
                    "x": lane,
                    "y": 0.0,
                    "w": self.w * 0.18,
                    "h": self.w * 0.12,
                    "color_hue": random.uniform(0, 360),
                }
            )

        for ob in self.obstacles:
            ob["y"] += self.speed * 1.2
            if random.random() < 0.3:
                self.particles.append(
                    {
                        "x": ob["x"] + random.uniform(-ob["w"] / 2, ob["w"] / 2),
                        "y": ob["y"],
                        "vx": random.uniform(-1, 1),
                        "vy": -random.uniform(2, 5),
                        "life": 20.0,
                        "max_life": 20.0,
                        "color": (255, 60, 120),
                    }
                )

        self.obstacles = [ob for ob in self.obstacles if ob["y"] < self.h + 100]

        self.particles.append(
            {
                "x": self.player_x + random.uniform(-15, 15),
                "y": self.player_y + 20,
                "vx": random.uniform(-2, 2),
                "vy": random.uniform(3, 7),
                "life": 25.0,
                "max_life": 25.0,
                "color": (0, 240, 255),
            }
        )

        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1.0

        self.particles = [p for p in self.particles if p["life"] > 0]

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill((15, 10, 28))

        horizon = int(self.h * 0.15)
        grid_lines = 16
        for i in range(grid_lines):
            x_bottom = (self.w // grid_lines) * i * 2 - (self.w // 2)
            pygame.draw.line(surf, (45, 25, 65), (self.w // 2, horizon), (x_bottom, self.h), 2)

        for y_raw in range(int(self.grid_offset), self.h - horizon, 60):
            step_ratio = y_raw / (self.h - horizon)
            y = int(horizon + (step_ratio**1.8) * (self.h - horizon))
            alpha = min(255, int(step_ratio * 200))
            color = (max(0, alpha // 4), max(0, alpha // 2), alpha)
            pygame.draw.line(surf, color, (0, y), (self.w, y), max(1, int(step_ratio * 3)))

        for ob in self.obstacles:
            ox = int(ob["x"] - ob["w"] / 2)
            oy = int(ob["y"])
            ow = int(ob["w"])
            oh = int(ob["h"])
            pygame.draw.rect(surf, (255, 30, 90), (ox, oy, ow, oh), border_radius=8)
            pygame.draw.rect(
                surf, (255, 180, 220), (ox + 4, oy + 4, ow - 8, oh - 8), 2, border_radius=6
            )

        for part in self.particles:
            alpha = part["life"] / part["max_life"]
            size = max(1, int(6 * alpha))
            px, py = int(part["x"]), int(part["y"])
            pygame.draw.circle(surf, part["color"], (px, py), size)

        pw, ph = int(self.w * 0.12), int(self.w * 0.12)
        px = int(self.player_x - pw / 2)
        py = int(self.player_y)

        pygame.draw.rect(surf, (0, 220, 255), (px, py, pw, ph), border_radius=10)
        pygame.draw.rect(
            surf, (255, 255, 255), (px + 6, py + 6, pw - 12, ph - 12), 3, border_radius=6
        )

        score_txt = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        surf.blit(score_txt, (30, 30))
        speed_txt = self.font.render(f"SPEED: {int(self.speed * 10)} KM/H", True, (0, 240, 255))
        surf.blit(speed_txt, (30, 70))


class ParticleMatrixScene:
    """Процедурный каскад неоновых частиц и матричных цифровых потоков."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.w = width
        self.h = height
        self.fps = fps
        self.streams = [
            {
                "x": int(x),
                "y": random.uniform(-height, 0),
                "speed": random.uniform(8, 20),
                "len": random.randint(10, 25),
            }
            for x in range(20, width, max(24, width // 25))
        ]

    def update(self, _frame: int) -> None:
        for s in self.streams:
            s["y"] += s["speed"]
            if s["y"] > self.h + 200:
                s["y"] = random.uniform(-300, -50)
                s["speed"] = random.uniform(8, 20)

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill((8, 12, 18))
        char_h = 24
        for s in self.streams:
            sx = int(s["x"])
            sy = int(s["y"])
            for j in range(int(s["len"])):
                cy = sy - j * char_h
                if 0 <= cy < self.h:
                    alpha = max(0.1, 1.0 - (j / s["len"]))
                    color = (255, 255, 255) if j == 0 else (0, int(255 * alpha), int(120 * alpha))
                    pygame.draw.circle(surf, color, (sx, cy), max(2, int(5 * alpha)))


def generate_pygame_video(
    preset: str = "runner",
    width: int = 1080,
    height: int = 1280,
    fps: int = 30,
    duration: float = 60.0,
    output_path: Path | None = None,
) -> Path:
    """Генерирует процедурное фоновое видео через PyGame и FFmpeg с кэшированием."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_preset = preset.lower().strip() or "runner"
    target = output_path or (
        CACHE_DIR / f"{clean_preset}_{width}x{height}_{fps}fps_{int(duration)}s.mp4"
    )

    if target.is_file() and target.stat().st_size > 100 * 1024:
        logger.info(f"[CACHE HIT] [PYGAME] Найдено готовое видео: {target.name}")
        return target

    logger.info(
        f"[PYGAME] Генерация видео '{clean_preset}': {width}x{height} @ {fps}fps, {duration:.1f}s -> {target.name}"
    )

    pygame.init()
    surf = pygame.Surface((width, height))

    scene: RunnerScene | ParticleMatrixScene
    if clean_preset in ("matrix", "particles"):
        scene = ParticleMatrixScene(width, height, fps)
    else:
        scene = RunnerScene(width, height, fps)

    total_frames = int(fps * duration)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for f in range(total_frames):
            scene.update(f)
            scene.draw(surf)
            raw_rgb = pygame.image.tobytes(surf, "RGB")
            if proc.stdin:
                proc.stdin.write(raw_rgb)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        pygame.quit()

    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        raise RuntimeError(f"PyGame FFmpeg рендер упал: {err[-1000:]}")

    logger.info(f"[PYGAME] Успешно создано фоновое видео: {target}")
    return target
