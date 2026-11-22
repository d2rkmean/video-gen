"""Нарезка видео на последовательные части с привязкой к паузам в речи (speech_split).

Использует пословные таймстемпы Whisper, чтобы:
1. Строго избегать обрыва на полуслове или середине фразы.
2. Исключать спонсорские интеграции, заставки и аутро (по SponsorBlock).
3. Формировать последовательные клипы (Часть 1, Часть 2, ...) с виральными описаниями.
"""

from __future__ import annotations

from loguru import logger

from video_gen.models import Highlight, Segment, Word

_SENTENCE_END = (".", "!", "?", "…")
_CLAUSE_END = (",", ";", ":", "—", "-")
_CLOSERS = "\"'»”)]"


def _is_in_sponsor(
    start: float,
    end: float,
    sponsor_intervals: list[tuple[float, float, str]],
    tolerance: float = 0.5,
) -> bool:
    """Проверяет, попадает ли интервал слова/фразы в спонсорский блок."""
    for s_start, s_end, _ in sponsor_intervals:
        if max(start, s_start) < min(end, s_end) - tolerance:
            return True
    return False


def _get_next_sponsor_start(
    pos: float,
    sponsor_intervals: list[tuple[float, float, str]],
) -> float | None:
    """Возвращает время начала ближайшего спонсорского блока после позиции pos."""
    future = [s_start for s_start, _, _ in sponsor_intervals if s_start >= pos - 0.2]
    return min(future) if future else None


def _get_sponsor_end(
    pos: float,
    sponsor_intervals: list[tuple[float, float, str]],
) -> float:
    """Если pos внутри спонсорского блока, возвращает его конец, иначе pos."""
    for s_start, s_end, _ in sponsor_intervals:
        if s_start - 0.2 <= pos <= s_end + 0.2:
            return s_end
    return pos


def split_by_speech_pauses(  # noqa: PLR0912, PLR0915
    segments: list[Segment],
    target_duration: float = 60.0,
    min_duration: float = 20.0,
    max_duration: float = 85.0,
    sponsor_intervals: list[tuple[float, float, str]] | None = None,
    title_prefix: str = "Часть",
    video_title: str | None = None,
) -> list[Highlight]:
    """Разбивает транскрипт видео на последовательные части по границам предложений и пауз.

    Args:
        segments: Сегменты Whisper с пословными таймингами.
        target_duration: Целевая длительность одной части в секундах (по умолчанию 60с).
        min_duration: Минимальная допустимая длительность (по умолчанию 20с).
        max_duration: Максимальная допустимая длительность (по умолчанию 85с).
        sponsor_intervals: Интервалы спонсорских интеграций для пропуска (от SponsorBlock).
        title_prefix: Префикс названия клипа (например, "Часть").
        video_title: Название видео (если есть, используется в заголовках и описании).

    Returns:
        Список объектов Highlight, представляющих последовательные части.
    """
    if not segments:
        logger.warning("[SPEECH_SPLIT] Пустой список сегментов для нарезки")
        return []

    sponsors = sponsor_intervals or []

    words: list[Word] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                if w.text.strip():
                    words.append(w)
        elif seg.text.strip():
            words.append(
                Word(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    probability=1.0,
                )
            )

    if not words:
        logger.warning("[SPEECH_SPLIT] В сегментах не найдено слов")
        return []

    words.sort(key=lambda w: w.start)

    highlights: list[Highlight] = []
    curr_idx = 0
    total_words = len(words)
    part_number = 1

    while curr_idx < total_words:
        word_start = words[curr_idx].start
        sponsor_end = _get_sponsor_end(word_start, sponsors)
        if sponsor_end > word_start:
            logger.info(f"[SPEECH_SPLIT] Пропуск спонсорского блока до {sponsor_end:.1f}с")
            while curr_idx < total_words and words[curr_idx].start < sponsor_end:
                curr_idx += 1
            if curr_idx >= total_words:
                break

        start_word = words[curr_idx]
        clip_start = start_word.start
        next_sponsor_start = _get_next_sponsor_start(clip_start, sponsors)

        best_cut_idx = -1
        best_score = float("-inf")
        last_valid_idx = -1

        for idx in range(curr_idx, total_words):
            w = words[idx]
            cur_duration = w.end - clip_start

            if next_sponsor_start is not None and w.end >= next_sponsor_start - 0.2:
                if cur_duration >= min_duration:
                    best_cut_idx = last_valid_idx if last_valid_idx != -1 else idx
                break

            if cur_duration < min_duration:
                last_valid_idx = idx
                continue

            if cur_duration > max_duration:
                break

            last_valid_idx = idx

            clean_text = w.text.rstrip(_CLOSERS)
            ends_sentence = clean_text.endswith(_SENTENCE_END)
            ends_clause = clean_text.endswith(_CLAUSE_END)
            gap = max(0.0, words[idx + 1].start - w.end) if idx + 1 < total_words else 2.0

            dur_diff = abs(cur_duration - target_duration)
            score = -dur_diff * 1.5

            if ends_sentence:
                score += 35.0
            elif ends_clause:
                score += 15.0

            if gap >= 1.0:
                score += 30.0
            elif gap >= 0.5:
                score += 20.0
            elif gap >= 0.25:
                score += 10.0

            if score > best_score:
                best_score = score
                best_cut_idx = idx

        if best_cut_idx == -1:
            if last_valid_idx != -1 and (words[last_valid_idx].end - clip_start >= min_duration):
                best_cut_idx = last_valid_idx
            else:
                rem_dur = words[-1].end - clip_start
                if highlights and rem_dur < min_duration:
                    prev = highlights[-1]
                    if (words[-1].end - prev.start) <= max_duration:
                        logger.info(
                            f"[SPEECH_SPLIT] Хвост длительностью {rem_dur:.1f}с объединён с частью {part_number - 1}"
                        )
                        highlights[-1] = Highlight(
                            start=prev.start,
                            end=words[-1].end,
                            reason=prev.reason,
                            score=prev.score,
                            hook=prev.hook,
                            viral_title=prev.viral_title,
                            description=prev.description,
                        )
                        break

                best_cut_idx = total_words - 1

        end_word = words[best_cut_idx]
        clip_end = end_word.end

        padded_start = max(0.0, clip_start - 0.15)
        if highlights and padded_start < highlights[-1].end:
            padded_start = highlights[-1].end

        padded_end = clip_end + 0.25
        if best_cut_idx + 1 < total_words:
            padded_end = min(padded_end, words[best_cut_idx + 1].start)

        clip_words = words[curr_idx : best_cut_idx + 1]
        hook_text = " ".join(w.text for w in clip_words[:7]).strip()
        base_title = f"{video_title} - " if video_title else ""
        viral_title = f"{base_title}{title_prefix} {part_number}".strip()

        desc_title = f"{video_title} | " if video_title else ""
        description = (
            f"{desc_title}{title_prefix} {part_number} 🔥 "
            f"Смотри до конца! Подпишись, продолжение в профиле 👇 "
            f"#fyp #shorts #рекомендации #нарезка"
        )

        h = Highlight(
            start=padded_start,
            end=padded_end,
            reason=f"Последовательная нарезка по паузам ({title_prefix} {part_number})",
            score=10,
            hook=hook_text,
            viral_title=viral_title,
            description=description,
        )
        highlights.append(h)
        logger.info(
            f"[SPEECH_SPLIT] Создана часть {part_number}: {h.start:.1f}с - {h.end:.1f}с "
            f"({h.end - h.start:.1f}с) | Слова: {len(clip_words)}"
        )

        part_number += 1
        curr_idx = best_cut_idx + 1

    return highlights
