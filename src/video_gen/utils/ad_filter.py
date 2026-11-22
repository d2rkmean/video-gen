"""Фильтрация рекламных интеграций, спонсорских вставок, заставок, интро и титров.

Поддерживает:
1. Запрос к базе SponsorBlock API (для роликов из YouTube).
2. Семантический и эвристический анализ расшифровки (по ключевым паттернам рекламы).
3. Пометку и исключение фрагментов из выдачи Gemini.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from loguru import logger

from video_gen.models import Highlight, Segment

# Регулярные выражения для поиска рекламных интеграций и призывов к действию
_AD_PATTERNS = [
    re.compile(r"промокод\w*|скидк[аеиу]|скидочк\w*", re.IGNORECASE),
    re.compile(r"спонсор\w*|партнёр\w* этого выпуск\w*", re.IGNORECASE),
    re.compile(r"ссылк[аеиу] в описани\w*|по ссылке в описани\w*|в закреплённом\w*", re.IGNORECASE),
    re.compile(r"в шапке профил\w*|переходи по ссылк\w*", re.IGNORECASE),
    re.compile(r"boosty|бусти|patreon|патреон|донат\w*", re.IGNORECASE),
    re.compile(r"телеграм[- ]канал\w*|тг[- ]канал\w*|подписывайтесь на телеграм", re.IGNORECASE),
    re.compile(r"1xbet|фонбет|винлайн|букмекер\w*|казино\w*|фрибет\w*", re.IGNORECASE),
    re.compile(r"ставь\w* лайк\w*|подпишись на канал|жмите колокольчик", re.IGNORECASE),
    re.compile(r"титры|ролики озвучивал\w*|перевод и озвучка", re.IGNORECASE),
]


def is_ad_text(text: str) -> bool:
    """Проверяет, содержит ли текст реплики признаки рекламы или спонсорства."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _AD_PATTERNS)


def fetch_sponsorblock_segments(
    video_id: str,
    categories: list[str] | None = None,
    timeout: float = 4.0,
    retries: int = 3,
) -> list[tuple[float, float, str]]:
    """Запрашивает известные спонсорские сегменты из открытой базы SponsorBlock API с повторами."""
    if not video_id:
        return []

    cats = categories or ["sponsor", "intro", "outro", "selfpromo", "preview", "interaction"]
    params = urllib.parse.urlencode({"videoID": video_id, "categories": json.dumps(cats)})
    url = f"https://sponsor.ajay.app/api/skipSegments?{params}"

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "video-gen/1.0 (https://github.com)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    results: list[tuple[float, float, str]] = []
                    for item in data:
                        seg = item.get("segment")
                        category = item.get("category", "sponsor")
                        if seg and len(seg) == 2:
                            results.append((float(seg[0]), float(seg[1]), str(category)))
                    logger.info(
                        f"[SPONSOR] Найдено {len(results)} рекламных/интро интервалов в SponsorBlock для {video_id}"
                    )
                    return results
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"[SPONSOR] Сегменты в SponsorBlock для {video_id} не найдены (404)")
                return []
            logger.warning(
                f"[SPONSOR] Ошибка HTTP {e.code} при запросе к SponsorBlock (попытка {attempt}/{retries}): {e}"
            )
        except Exception as e:
            logger.warning(
                f"[SPONSOR] Ошибка при запросе к SponsorBlock (попытка {attempt}/{retries}): {e}"
            )

        if attempt < retries:
            time.sleep(1.0 * (2 ** (attempt - 1)))

    return []


def highlight_overlaps_sponsor(
    start: float,
    end: float,
    sponsor_intervals: list[tuple[float, float, str]],
    tolerance: float = 1.0,
) -> bool:
    """Проверяет, пересекается ли интервал клипа со спонсорским интервалом."""
    for s_start, s_end, _ in sponsor_intervals:
        # Пересечение отрезков [start, end] и [s_start, s_end]
        if max(start, s_start) < min(end, s_end) - tolerance:
            return True
    return False


def filter_highlights_ads(
    highlights: list[Highlight],
    segments: list[Segment],
    sponsor_intervals: list[tuple[float, float, str]] | None = None,
) -> list[Highlight]:
    """Отфильтровывает кандидаты клипов, если они содержат рекламу или попадают в SponsorBlock."""
    intervals = sponsor_intervals or []
    cleaned: list[Highlight] = []

    for h in highlights:
        if intervals and highlight_overlaps_sponsor(h.start, h.end, intervals):
            logger.warning(
                f"[SPONSOR] Отклонён клип {h.start:.1f}-{h.end:.1f}: пересекается со спонсорским интервалом SponsorBlock"
            )
            continue

        clip_segments = [s for s in segments if s.start >= h.start - 0.5 and s.end <= h.end + 0.5]
        if any(is_ad_text(s.text) for s in clip_segments):
            logger.warning(
                f"[SPONSOR] Отклонён клип {h.start:.1f}-{h.end:.1f}: содержит рекламный текст или промокод"
            )
            continue

        cleaned.append(h)

    return cleaned
