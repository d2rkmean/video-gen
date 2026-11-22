import math
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

WIDTH, HEIGHT = 1440, 600
FPS = 60
DURATION = 3.8
TOTAL_FRAMES = int(FPS * DURATION)
CHROMA_BG = (39, 60, 1)  # #273c01

BANNER_TITLE = os.environ.get("BANNER_TITLE", "VIDEOGEN")
BOT_LINK = os.environ.get("BANNER_BOT_LINK", "@videogen_bot")
CLIPS_LINK = os.environ.get("BANNER_CLIPS_LINK", "@videogen_clips")

ROOT_DIR = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT_DIR / "fonts"
FONT_TITLE = ImageFont.truetype(str(FONTS_DIR / "Unbounded-Black.ttf"), 58)
FONT_LINK = ImageFont.truetype(str(FONTS_DIR / "GoogleSans-Medium.ttf"), 31)

BRANDING_DIR = ROOT_DIR / "input" / "branding"
AVATAR_IMG = Image.open(BRANDING_DIR / "logo_avatar_v2.jpg").convert("RGBA")
CARTMAN_IMG = Image.open(BRANDING_DIR / "cartman_boosted.png").convert("RGBA")
POPCORN_IMG = Image.open(BRANDING_DIR / "popcorn_boosted.png").convert("RGBA")
CLAPPER_IMG = Image.open(BRANDING_DIR / "clapper_boosted.png").convert("RGBA")

# Карточка баннера: 860x260
CARD_W, CARD_H = 860, 260
CARD_TARGET_X = (WIDTH - CARD_W) // 2  # 290
CARD_TARGET_Y = (HEIGHT - CARD_H) // 2  # 170
CARD_RADIUS = 36


def spring(t: float, freq: float = 7.2, damping: float = 0.65) -> float:
    """Затухающая гармоническая пружина C2-гладкости (Apple Spring Physics)."""
    if t <= 0.0:
        return 0.0
    omega = freq
    decay = math.exp(-damping * omega * t)
    d_sqrt = math.sqrt(max(0.001, 1.0 - damping**2))
    return 1.0 - decay * (math.cos(omega * t) + (damping / d_sqrt) * math.sin(omega * t))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smoothstep с нулевой первой производной на обоих концах."""
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def render():
    c_ratio = 355 / CARTMAN_IMG.height
    cartman_scaled = CARTMAN_IMG.resize(
        (int(CARTMAN_IMG.width * c_ratio), 355), Image.Resampling.LANCZOS
    )

    p_ratio = 245 / POPCORN_IMG.height
    popcorn_scaled = POPCORN_IMG.resize(
        (int(POPCORN_IMG.width * p_ratio), 245), Image.Resampling.LANCZOS
    )

    cl_ratio = 205 / CLAPPER_IMG.height
    clapper_scaled = CLAPPER_IMG.resize(
        (int(CLAPPER_IMG.width * cl_ratio), 205), Image.Resampling.LANCZOS
    )

    AV_SIZE = 152
    avatar_res = AVATAR_IMG.resize((AV_SIZE, AV_SIZE), Image.Resampling.LANCZOS)
    av_mask = Image.new("L", (AV_SIZE, AV_SIZE), 0)
    ImageDraw.Draw(av_mask).ellipse([0, 0, AV_SIZE, AV_SIZE], fill=255)
    avatar_round = Image.new("RGBA", (AV_SIZE, AV_SIZE), (0, 0, 0, 0))
    avatar_round.paste(avatar_res, (0, 0), av_mask)

    raw_video_path = str(ROOT_DIR / "input" / "banner_temp_v8.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        raw_video_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    print(f"Рендерим {TOTAL_FRAMES} кадров баннера с разлетом элементов в конце...")

    T_START = 0.20  # Задержка под размытие фона
    SPRING_DUR = 0.60
    OUTRO_START = 3.20  # Старт кинематографичного вылета за экран (за 0.6с до конца)
    OUTRO_DUR = 0.50   # За 0.5с все элементы полностью улетают за пределы холста

    BTN_W_BASE, BTN_H_BASE = 280, 66
    y_idx, x_idx = np.indices((BTN_H_BASE, BTN_W_BASE))
    glare_coords = x_idx - y_idx * 0.35

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / FPS
        frame = Image.new("RGB", (WIDTH, HEIGHT), CHROMA_BG)

        # До T_START гарантированно отдаем чистый хромакей
        if t < T_START:
            proc.stdin.write(frame.tobytes())
            continue

        # ── 1. ВРЕМЕННЫЕ ШКАЛЫ И КИНЕМАТОГРАФИЧНЫЙ РАЗЛЕТ (OUTRO) ──
        tau_spring = max(0.0, (t - T_START) / SPRING_DUR)

        # Вылет за экран: начинается мягко с 0 скорости, затем стремительно ускоряется (кубическая прогрессия)
        if t >= OUTRO_START:
            raw_out = min(1.0, (t - OUTRO_START) / OUTRO_DUR)
            # Разгон с ускорением наружу
            fly_prog = math.pow(raw_out, 2.3)
            # Затухание idle-колебаний
            idle_fade = max(0.0, 1.0 - raw_out * 1.5)
        else:
            raw_out = 0.0
            fly_prog = 0.0
            idle_fade = 1.0

        # Если анимация вылета полностью завершена (последние кадры), отдаем чистый хромакей
        if raw_out >= 1.0:
            proc.stdin.write(frame.tobytes())
            continue

        IDLE_ORIGIN = T_START + 0.35  # 0.55с
        dt_idle = max(0.0, t - IDLE_ORIGIN)
        idle_blend = smoothstep(IDLE_ORIGIN, IDLE_ORIGIN + 0.40, t) * idle_fade

        # Органическое гармоническое дыхание всей сцены
        scene_float_y = math.sin(dt_idle * 2.1) * 6.5 * idle_blend
        scene_float_x = math.cos(dt_idle * 1.7) * 3.5 * idle_blend

        # Карточка: пружинный вход сверху
        card_scale = spring(tau_spring, freq=7.2, damping=0.68)
        card_y_intro = (1.0 - smoothstep(0.0, 1.0, min(1.0, tau_spring * 1.1))) * 110

        # ВЫЛЕТ КАРТОЧКИ: стремительно улетает вниз за нижнюю границу экрана
        card_fly_y = int(fly_prog * 700)
        card_fly_tilt = -fly_prog * 8.0

        card_w_cur = int(CARD_W * card_scale)
        card_h_cur = int(CARD_H * card_scale)
        card_cur_x = int(CARD_TARGET_X + (CARD_W - card_w_cur) // 2 + scene_float_x)
        card_cur_y = int(CARD_TARGET_Y + (CARD_H - card_h_cur) // 2 + card_y_intro + scene_float_y + card_fly_y)

        # ── 2. КАРТОЧКА БАННЕРА ──
        if card_scale > 0.08 and card_w_cur > 20 and card_h_cur > 20 and card_cur_y < HEIGHT:
            card_canvas = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))

            # Градиент темного стекла
            for y in range(CARD_H):
                rat = y / CARD_H
                r = int(14 + rat * 10)
                g = int(16 + rat * 8)
                b = int(28 + rat * 16)
                card_canvas.paste(Image.new("RGBA", (CARD_W, 1), (r, g, b, 255)), (0, y))

            # Деликатный зеркальный блик
            spec_h = int(CARD_H * 0.42)
            spec_layer = Image.new("RGBA", (CARD_W, spec_h), (0, 0, 0, 0))
            for y in range(spec_h):
                alpha = int((1.0 - y / spec_h) * 32)
                spec_layer.paste(Image.new("RGBA", (CARD_W, 1), (255, 255, 255, alpha)), (0, y))
            card_canvas.paste(spec_layer, (0, 0), spec_layer)

            # Маска скругления
            c_mask = Image.new("L", (CARD_W, CARD_H), 0)
            ImageDraw.Draw(c_mask).rounded_rectangle(
                [0, 0, CARD_W, CARD_H], radius=CARD_RADIUS, fill=255
            )

            card_bg = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
            card_bg.paste(card_canvas, (0, 0), c_mask)
            draw = ImageDraw.Draw(card_bg)

            # Перелив неонового градиента рамки
            border_mask = Image.new("L", (CARD_W, CARD_H), 0)
            ImageDraw.Draw(border_mask).rounded_rectangle(
                [2, 2, CARD_W - 3, CARD_H - 3], radius=CARD_RADIUS, outline=255, width=4
            )

            b_arr = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
            shift = (t * 0.40) % 1.0
            x_coords = (np.linspace(0, 1, CARD_W) + shift) % 1.0
            for x_idx, x_val in enumerate(x_coords):
                factor = 0.5 + 0.5 * math.cos(x_val * 2 * math.pi)
                r_c = int(255 * factor + 0 * (1 - factor))
                g_c = int(45 * factor + 225 * (1 - factor))
                b_c = int(160 * factor + 255 * (1 - factor))
                b_arr[:, x_idx, 0] = r_c
                b_arr[:, x_idx, 1] = g_c
                b_arr[:, x_idx, 2] = b_c
                b_arr[:, x_idx, 3] = 255

            border_grad = Image.fromarray(b_arr, mode="RGBA")
            card_bg.paste(border_grad, (0, 0), border_mask)

            # Внутренний тонкий светлый контур (1px)
            draw.rounded_rectangle(
                [5, 5, CARD_W - 6, CARD_H - 6],
                radius=CARD_RADIUS - 3,
                outline=(255, 255, 255, 50),
                width=1,
            )

            # ── 3. АВАТАРКА С ПЛАВНО ПАРИРУЮЩИМ НЕОНОВЫМ КОЛЬЦОМ ──
            av_x = 40
            av_y = (CARD_H - AV_SIZE) // 2
            card_bg.paste(avatar_round, (av_x, av_y), avatar_round)

            ring_alpha = int(180 + 40 * math.sin(dt_idle * 2.8) * idle_blend)
            draw.ellipse(
                [av_x - 3, av_y - 3, av_x + AV_SIZE + 3, av_y + AV_SIZE + 3],
                outline=(0, 220, 255, ring_alpha),
                width=2,
            )
            rot_angle = int((t * 80.0) % 360)
            draw.arc(
                [av_x - 4, av_y - 4, av_x + AV_SIZE + 4, av_y + AV_SIZE + 4],
                start=rot_angle,
                end=rot_angle + 90,
                fill=(0, 255, 255, 220),
                width=3,
            )

            # ── 4. ЗАГОЛОВОК БАННЕРА ──
            text_x = av_x + AV_SIZE + 35
            draw.text((text_x + 2, 48), BANNER_TITLE, font=FONT_TITLE, fill=(0, 0, 0, 180))
            draw.text((text_x, 46), BANNER_TITLE, font=FONT_TITLE, fill=(255, 255, 255, 255))

            # ── 5. КНОПКИ С МЯГКИМ СВЕТОВЫМ БЛИКОМ ──
            btn_y = 142
            btn_h = 66
            b1_w = 280
            b2_w = 295

            b1_breathe = 1.0 + 0.018 * math.sin(dt_idle * 2.6) * idle_blend
            b1_w_cur = int(b1_w * b1_breathe)
            b1_h_cur = int(btn_h * b1_breathe)
            b1_x0 = text_x - (b1_w_cur - b1_w) // 2
            b1_y0 = btn_y - (b1_h_cur - btn_h) // 2

            b1_surf = Image.new("RGBA", (b1_w, btn_h), (0, 0, 0, 0))
            for by in range(btn_h):
                bfactor = by / btn_h
                b1_surf.paste(Image.new("RGBA", (b1_w, 1), (0, int(145 - bfactor * 25), int(250 - bfactor * 20), 255)), (0, by))

            glare_cycle = (t % 2.2) / 2.2
            if glare_cycle < 0.45:
                glare_prog = glare_cycle / 0.45
                glare_pos = -40 + glare_prog * (b1_w + 80)
                dist = np.abs(glare_coords - glare_pos)
                glare_mask = np.exp(-0.5 * (dist / 16.0)**2) * 85.0
                glare_arr = np.zeros((btn_h, b1_w, 4), dtype=np.uint8)
                glare_arr[:, :, :3] = 255
                glare_arr[:, :, 3] = glare_mask.astype(np.uint8)
                glare_im = Image.fromarray(glare_arr, mode="RGBA")
                b1_surf = Image.alpha_composite(b1_surf, glare_im)

            b1_mask = Image.new("L", (b1_w, btn_h), 0)
            ImageDraw.Draw(b1_mask).rounded_rectangle([0, 0, b1_w, btn_h], radius=16, fill=255)
            b1_final = Image.new("RGBA", (b1_w, btn_h), (0, 0, 0, 0))
            b1_final.paste(b1_surf, (0, 0), b1_mask)

            if abs(b1_breathe - 1.0) > 0.003:
                b1_final = b1_final.resize((b1_w_cur, b1_h_cur), Image.Resampling.BILINEAR)

            card_bg.paste(b1_final, (b1_x0, b1_y0), b1_final)

            t1_bbox = FONT_LINK.getbbox(BOT_LINK)
            t1_w = t1_bbox[2] - t1_bbox[0]
            t1_x = b1_x0 + (b1_w_cur - t1_w) // 2
            draw.text((t1_x, b1_y0 + 17), BOT_LINK, font=FONT_LINK, fill=(0, 45, 90, 200))
            draw.text((t1_x, b1_y0 + 15), BOT_LINK, font=FONT_LINK, fill=(255, 255, 255, 255))

            # Кнопка 2: канал
            b2_x0 = text_x + b1_w + 16
            b2_y0 = btn_y
            b2_x1, b2_y1 = b2_x0 + b2_w, b2_y0 + btn_h

            b2_glow = int(180 + 40 * math.sin(dt_idle * 2.3 + 1.2) * idle_blend)
            draw.rounded_rectangle(
                [b2_x0, b2_y0, b2_x1, b2_y1],
                radius=16,
                fill=(22, 30, 50, 245),
                outline=(0, 205, 255, b2_glow),
                width=2,
            )
            t2_bbox = FONT_LINK.getbbox(CLIPS_LINK)
            t2_w = t2_bbox[2] - t2_bbox[0]
            t2_x = b2_x0 + (b2_w - t2_w) // 2
            draw.text(
                (t2_x, b2_y0 + 15), CLIPS_LINK, font=FONT_LINK, fill=(225, 245, 255, 255)
            )

            # Масштаб и легкий наклон при вылете
            if abs(card_scale - 1.0) > 0.005 or abs(card_fly_tilt) > 0.1:
                card_final = card_bg.resize((card_w_cur, card_h_cur), Image.Resampling.BILINEAR)
                if abs(card_fly_tilt) > 0.1:
                    card_final = card_final.rotate(card_fly_tilt, resample=Image.Resampling.BICUBIC, expand=True)
                frame.paste(card_final, (card_cur_x, card_cur_y), card_final)
            else:
                frame.paste(card_bg, (card_cur_x, card_cur_y), card_bg)

        # ── 6. КАХОТНЫЙ ЖИВОЙ КАРТМАН (Вылет влево за экран) ──
        cart_target_x = card_cur_x - 170
        cart_start_x = -500

        c_tau = max(0.0, (t - T_START) / 0.52)
        c_spring = spring(c_tau, freq=6.8, damping=0.64)

        # ВЫЛЕТ КАРТМАНА ВЛЕВО: мощно улетает влево за пределы кадра (X < -500)
        cart_fly_x = int(fly_prog * 800)
        cart_fly_tilt = -fly_prog * 26.0

        c_x_pos = int(cart_start_x + (cart_target_x - cart_start_x) * c_spring) - cart_fly_x
        intro_tilt = (1.0 - min(1.0, c_tau)) * 12.0
        c_bob = math.sin(dt_idle * 2.4) * 6.0 * idle_blend
        c_tilt = (math.sin(dt_idle * 2.0) * 3.0 + intro_tilt) * idle_blend + cart_fly_tilt
        c_lean = int(math.cos(dt_idle * 2.0) * 4.0 * idle_blend)

        c_rot = cartman_scaled.rotate(c_tilt, resample=Image.Resampling.BICUBIC, expand=True)
        cy = int(HEIGHT - c_rot.height - 45 + c_bob + int(fly_prog * 120))
        final_cart_x = c_x_pos + c_lean
        if final_cart_x + c_rot.width > 0 and final_cart_x < WIDTH and cy < HEIGHT:
            frame.paste(c_rot, (final_cart_x, cy), c_rot)

        # ── 7. ПОПКОРН (Вылет вверх-вправо в небо за экран) ──
        p_target_y = card_cur_y - 120
        p_start_y = -450

        p_tau = max(0.0, (t - T_START - 0.05) / 0.54)
        p_spring = spring(p_tau, freq=6.6, damping=0.62)

        # ВЫЛЕТ ПОПКОРНА: взлетает вверх и вправо (Y < -300, X > 1500)
        pop_fly_x = int(fly_prog * 500)
        pop_fly_y = int(fly_prog * 650)
        pop_fly_tilt = fly_prog * 42.0

        p_y_pos = int(p_start_y + (p_target_y - p_start_y) * p_spring) - pop_fly_y
        p_bob = math.sin(dt_idle * 2.2 + 1.2) * 6.5 * idle_blend
        p_tilt = (math.cos(dt_idle * 1.8 + 0.6) * 3.8) * idle_blend + pop_fly_tilt
        p_drift = int(math.sin(dt_idle * 1.5) * 3.0 * idle_blend)

        p_rot = popcorn_scaled.rotate(p_tilt, resample=Image.Resampling.BICUBIC, expand=True)
        px = card_cur_x + card_w_cur - 90 + p_drift + pop_fly_x
        py = int(p_y_pos + p_bob)
        if py + p_rot.height > 0 and py < HEIGHT and px < WIDTH:
            frame.paste(p_rot, (px, py), p_rot)

        # ── 8. ХЛОПУШКА И VIP-БИЛЕТ (Вылет вниз-вправо за экран) ──
        cl_target_y = card_cur_y + card_h_cur - 120
        cl_start_y = HEIGHT + 200

        cl_tau = max(0.0, (t - T_START - 0.10) / 0.55)
        cl_spring = spring(cl_tau, freq=6.5, damping=0.63)

        # ВЫЛЕТ ХЛОПУШКИ: улетает вниз и вправо (Y > 700, X > 1500)
        clap_fly_x = int(fly_prog * 500)
        clap_fly_y = int(fly_prog * 650)
        clap_fly_tilt = -fly_prog * 36.0

        cl_y_pos = int(cl_start_y + (cl_target_y - cl_start_y) * cl_spring) + clap_fly_y
        cl_bob = math.cos(dt_idle * 2.0 + 2.0) * 6.0 * idle_blend
        cl_tilt = (math.sin(dt_idle * 1.7 + 1.4) * 3.2) * idle_blend + clap_fly_tilt
        cl_drift = int(math.cos(dt_idle * 1.4) * 3.0 * idle_blend)

        cl_rot = clapper_scaled.rotate(cl_tilt, resample=Image.Resampling.BICUBIC, expand=True)
        clx = card_cur_x + card_w_cur - 30 + cl_drift + clap_fly_x
        cly = int(cl_y_pos + cl_bob)
        if cly < HEIGHT and cly + cl_rot.height > 0 and clx < WIDTH:
            frame.paste(cl_rot, (clx, cly), cl_rot)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print("Видеопоток с динамичным разлетом элементов отрендерен.")

    # Объединение с озвучкой
    audio_file = os.environ.get("BANNER_AUDIO_FILE", str(ROOT_DIR / "input" / "banner_audio.mp3"))
    final_banner_path = str(ROOT_DIR / "input" / "banner.mp4")

    merge_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        raw_video_path,
        "-i",
        audio_file,
        "-filter_complex",
        "[1:a]adelay=300|300,apad=whole_dur=3.8[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        "3.8",
        final_banner_path,
    ]
    subprocess.run(merge_cmd, check=True)
    subprocess.run(["rm", "-f", raw_video_path])
    print("Финал успешно сохранен в input/banner.mp4!")


if __name__ == "__main__":
    render()
