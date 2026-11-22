"""Детекция лиц спикера для умного кроп-центрирования верхнего видео."""

from pathlib import Path

import cv2
from loguru import logger


def detect_speaker_center_x(
    video_path: Path,
    start: float,
    duration: float,
    samples: int = 8,
) -> float:
    """Определяет среднюю горизонтальную координату X лица спикера (0.0..1.0).

    Сэмплирует несколько кадров из диапазона клипа, находит лица через
    легковесный Haar Cascade и возвращает усреднённый центр лица.
    Если лица не найдены или произошла ошибка, возвращает 0.5 (центр кадра).

    Args:
        video_path: Путь к исходному видео.
        start: Начало клипа в секундах.
        duration: Длительность клипа в секундах.
        samples: Количество сэмплируемых кадров.

    Returns:
        Нормализованная координата X центра лица от 0.0 до 1.0 (по умолчанию 0.5).
    """
    if not video_path.exists() or duration <= 0:
        return 0.5

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)  # type: ignore[attr-defined]
        if face_cascade.empty():
            return 0.5

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.5

        step = max(duration / (samples + 1), 0.5)

        detected_centers: list[float] = []

        for k in range(1, samples + 1):
            t = start + k * step
            if t >= start + duration:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            scale = 480.0 / max(w, 1)
            small = cv2.resize(frame, (480, int(h * scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(30, 30),
            )

            if len(faces) > 0:
                # Берём самое крупное лицо (скорее всего активный спикер)
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, _, fw_box, _ = largest
                center_norm = (fx + fw_box / 2.0) / 480.0
                detected_centers.append(center_norm)

        cap.release()

        if detected_centers:
            avg_x = sum(detected_centers) / len(detected_centers)
            logger.debug(
                f"FaceTracker: обнаружено лиц на {len(detected_centers)}/{samples} кадрах, X={avg_x:.3f}"
            )
            return max(0.1, min(0.9, avg_x))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"FaceTracker: ошибка детекции лиц ({e}), центрирую по умолчанию")

    return 0.5
