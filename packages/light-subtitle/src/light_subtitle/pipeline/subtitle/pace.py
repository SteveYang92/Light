"""Pace (对时) — assign each cue a comfortable display time window.

Adjusts cue timing for readable consumption:
  1. Duration fix — stretch too-short cues, cap overlong ones
  2. Gap resolution — resolve overlaps, distribute tiny gaps proportionally
  3. CPS enforcement — borrow time from gaps when reading speed too high
  4. Min-gap guard — ensure minimum separation between cues
  5. Reading padding — extend end times for viewer breathing room

NOTE: English layout splits text exceeding max_lines into multiple
cues with word-aligned timing.  Pace operates on whatever cue list
layout produces — it no longer performs its own splitting.
"""

from __future__ import annotations

from ... import logger


def correct(cues, config):
    """Adjust cue timing for readability.

    Runs duration fix, gap resolution, CPS enforcement, min-gap guard, reading padding.
    Scene B/C alignment is handled by run() before prepare.
    """
    return _apply_time_corrections(cues, config)


# ═══════════════════════════════════════════════════════════════════
# Main time-correction pipeline
# ═══════════════════════════════════════════════════════════════════


def _apply_time_corrections(cues, config):
    """Full time-correction pipeline.

    Order matters: each step assumes the previous step's output shape.
    """
    if not cues:
        return cues

    # Step 1: duration fix — stretch or cap each cue
    result = []
    for cue in cues:
        result.extend(_fix_cue_duration(cue, config))

    # Step 2: gap resolution — resolve overlaps, distribute tiny gaps
    MIN_GAP = 0.105
    for i in range(len(result) - 1):
        gap = result[i + 1].start - result[i].end
        if gap < 0:
            chars_i = sum(1 for _ in result[i].text.replace("\n", ""))
            chars_j = sum(1 for _ in result[i + 1].text.replace("\n", ""))
            total_chars = chars_i + chars_j
            if total_chars > 0:
                overlap = -gap
                i_share = overlap * (chars_i / total_chars)
                result[i].end = result[i + 1].start + i_share
            else:
                result[i].end = result[i + 1].start
        elif 0 < gap < 0.15:
            midpoint = result[i].end + gap * 0.5
            result[i].end = midpoint - MIN_GAP / 2
            result[i + 1].start = midpoint + MIN_GAP / 2

    # Step 4: CPS enforcement — borrow time (never split).
    # max_duration cap in _enforce_cps_ceiling ensures no post-hoc
    # re-check is needed; the cue list shape stays the same.
    result = _enforce_cps_ceiling(result, config)

    # Step 5: ensure all gaps >= MIN_GAP
    for i in range(1, len(result)):
        if result[i].start - result[i - 1].end < MIN_GAP:
            push = MIN_GAP - (result[i].start - result[i - 1].end)
            result[i].start = result[i - 1].end + MIN_GAP
            result[i].end = max(result[i].end, result[i].end + push)

    # Step 6: reading padding — breathing room at each cue's end
    for i in range(len(result)):
        pad = config.reading_padding
        if i + 1 < len(result):
            available = result[i + 1].start - result[i].end - MIN_GAP
            pad = min(pad, max(0, available))
        if pad > 0:
            result[i].end += pad

    # Final safety net: re-stretch any cue that fell below min_duration
    # after gap resolution / min-gap / padding adjustments.
    # Borrow from gaps on both sides; if insufficient, accept it.
    MIN_GAP = 0.105
    for i, cue in enumerate(result):
        dur = cue.end - cue.start
        if dur >= config.min_duration - 0.001:
            continue
        shortage = config.min_duration - dur

        gap_forward = (result[i + 1].start - cue.end - MIN_GAP) if i + 1 < len(result) else 0
        gap_backward = (cue.start - result[i - 1].end - MIN_GAP) if i > 0 else 0

        forward = min(shortage, max(0, gap_forward))
        backward = min(shortage - forward, max(0, gap_backward))

        cue.end += forward
        cue.start -= backward

    # Step 7: word-boundary alignment — after all pacing corrections,
    # ensure cue boundaries cover actual spoken words.  Only adjust when
    # there is sufficient headroom to adjacent cues, so we never create
    # TimelineGap (<0.1s gap) or GapFlash issues.  Use 0.105 (not 0.10)
    # to match steps 2–6 and stay above the QC min_gap threshold.
    MIN_GAP = 0.105
    MAX_EXIT_PADDING = config.reading_padding + 0.5  # reading_padding + 500ms grace

    for i, cue in enumerate(result):
        if not cue.words:
            continue

        word_start = min(w.start for w in cue.words)
        word_end = max(w.end for w in cue.words)

        prev_cue = result[i - 1] if i > 0 else None
        next_cue = result[i + 1] if i + 1 < len(result) else None

        # Pull start back toward word_start if it drifted and has headroom
        drift = cue.start - word_start
        if drift > 0.1:
            headroom = (word_start - prev_cue.end - MIN_GAP) if prev_cue else 999.0
            if headroom >= 0:
                cue.start = max(word_start, prev_cue.end + MIN_GAP) if prev_cue else word_start

        # Push end forward toward word_end if it falls short and has headroom
        shortfall = word_end - cue.end
        if shortfall > 0.1:
            headroom = (next_cue.start - word_end - MIN_GAP) if next_cue else 999.0
            if headroom >= 0:
                cue.end = min(word_end, next_cue.start - MIN_GAP) if next_cue else word_end

        # Cap excessive silent padding after last word
        excess = cue.end - word_end
        if excess > MAX_EXIT_PADDING:
            target_end = word_end + MAX_EXIT_PADDING
            if next_cue:
                gap_to_next = next_cue.start - target_end
                if gap_to_next >= MIN_GAP:
                    cue.end = target_end
                else:
                    cue.end = next_cue.start - MIN_GAP
            else:
                cue.end = target_end

    # Step 7b: entry-point confidence optimization.
    # Must run AFTER word-boundary alignment (step 7) so deliberate
    # entry shifts are not undone by the drift-correction logic.
    # Shift entry to first reliable word when wav2vec2 alignment
    # confidence is low at cue boundaries.
    if getattr(config, "optimize_entry_points", True) and result:
        words = getattr(config, "transcript_words", None)
        result = _optimize_entry_points(result, words)

    # After all boundary corrections, re-min-duration any cue that fell
    # below the minimum.
    for i, cue in enumerate(result):
        dur = cue.end - cue.start
        if dur >= config.min_duration - 0.001:
            continue
        shortage = config.min_duration - dur
        gap_forward = (result[i + 1].start - cue.end - MIN_GAP) if i + 1 < len(result) else 0.0
        gap_backward = (cue.start - result[i - 1].end - MIN_GAP) if i > 0 else 0.0

        forward = min(shortage, max(0.0, gap_forward))
        backward = min(shortage - forward, max(0.0, gap_backward))
        cue.end += forward
        cue.start -= backward

    return result


# ═══════════════════════════════════════════════════════════════════
# Duration helpers
# ═══════════════════════════════════════════════════════════════════


def _fix_cue_duration(cue, config):
    """Enforce min/max duration constraints.

    - Too-short cues: stretch to min_duration.
    - Overlong cues: cap end time to start + max_duration (merged cues exempt).
    """
    duration = cue.end - cue.start
    if duration < config.min_duration - 0.001:
        cue.end = cue.start + config.min_duration
    elif duration > config.max_duration:
        if cue.merged_from:
            logger.info(
                f"  Pace: merged cue max_duration exempt | {cue.unit_id} | "
                f"duration={duration:.2f}s (limit={config.max_duration}s) | "
                f"merged_from={cue.merged_from}"
            )
        else:
            cue.end = cue.start + config.max_duration
    return [cue]


def _enforce_cps_ceiling(cues, config):
    """For cues exceeding CPS reading-speed ceiling, extend time into
    available gaps (forward → backward → both). When gaps are insufficient,
    accept the over-limit CPS rather than splitting (splitting doesn't
    improve CPS and would damage semantic structure)."""
    result = []
    zh_cps = config.cps_limit
    en_cps = config.cps_limit_en
    MIN_GAP = 0.105

    for i, cue in enumerate(cues):
        chars = sum(1 for _ in cue.text.replace("\n", ""))
        duration = cue.end - cue.start
        if duration <= 0:
            result.append(cue)
            continue

        limit = zh_cps if cue.lang == "zh" else en_cps
        cps = chars / duration
        if cps <= limit:
            result.append(cue)
            continue

        needed = chars / limit
        shortage = needed - duration

        # Cap shortage so CPS extension never pulls duration past max_duration.
        if not cue.merged_from:
            shortage = min(shortage, max(0, config.max_duration - duration))

        if shortage <= 0:
            result.append(cue)
            continue

        gap_forward = max(0, (cues[i + 1].start - cue.end) - MIN_GAP) if i + 1 < len(cues) else 0
        gap_backward = max(0, (cue.start - cues[i - 1].end) - MIN_GAP) if i > 0 else 0

        # Strategy 1: extend forward
        if gap_forward > 0 and shortage <= gap_forward:
            cue.end = min(cue.end + shortage, cues[i + 1].start - MIN_GAP)
            result.append(cue)
            continue

        # Strategy 2: pull backward
        if gap_backward > 0 and shortage <= gap_backward:
            cue.start = max(cue.start - shortage, cues[i - 1].end + MIN_GAP)
            result.append(cue)
            continue

        # Strategy 3: borrow from both
        combined_gap = gap_forward + gap_backward
        if combined_gap > 0 and shortage <= combined_gap:
            from_backward = min(shortage, gap_backward)
            from_forward = shortage - from_backward
            if from_backward > 0:
                cue.start = max(
                    cue.start - from_backward, cues[i - 1].end + MIN_GAP if i > 0 else cue.start - from_backward
                )
            if from_forward > 0 and i + 1 < len(cues):
                cue.end = min(cue.end + from_forward, cues[i + 1].start - MIN_GAP)
            result.append(cue)
            continue

        # Borrow from both sides not enough → accept CPS over-limit.
        # Splitting doesn't change CPS (chars and total duration stay the same),
        # so it would only break semantic structure without improving readability.
        result.append(cue)

    return result


# ═══════════════════════════════════════════════════════════════════
# Entry-point confidence optimization
# ═══════════════════════════════════════════════════════════════════

_CLIFF_FIRST_MAX = 0.40
_CLIFF_PEAK_MIN = 0.70
_CLIFF_DIFF_MIN = 0.30
_STRETCH_MIN_MS = 800
_GAP_MIN_S = 0.50
_FIRST_GOOD_CONF = 0.50
_LEAD_IN_S = 0.10
_MAX_SHIFT_S = 1.50
_INTERNAL_GAP_THRESHOLD_S = 1.5  # gaps larger than this suggest word misalignment
_GAP_ANCHOR_MIN_CONF = 0.45  # anchor word for gap-fix must exceed this confidence


def _optimize_entry_points(cues, transcript_words=None):
    """Detect and fix low-confidence entry points.

    Uses *transcript_words* (flat list of Word objects) for confidence
    data.  If not provided, falls back to cue.words.
    """
    fixed = 0
    skipped = 0

    for i, cue in enumerate(cues):
        words = _get_aligned_words(cue, transcript_words) if transcript_words else cue.words
        if not words or len(words) < 2:
            continue

        # ── Pass 1: confidence cliff at entry ────────────────────
        first = words[0]
        second = words[1]
        later_confs = [w.confidence for w in words[1:6]]
        peak_later = max(later_confs) if later_confs else first.confidence
        conf_diff = peak_later - first.confidence
        cliff_detected = (
            first.confidence < _CLIFF_FIRST_MAX and peak_later >= _CLIFF_PEAK_MIN and conf_diff >= _CLIFF_DIFF_MIN
        )

        if cliff_detected:
            first_good = _find_first_good_word(words)
            if first_good is None:
                continue
            new_start = first_good.start - _LEAD_IN_S
            shift = new_start - cue.start
            if shift < 0.08:
                continue
            if shift > _MAX_SHIFT_S:
                skipped += 1
                _log_skip(i, cue, first, peak_later, first_good, shift, second)
                continue
            cue.start = new_start
            fixed += 1
            _log_fix(i, cue, first, peak_later, first_good, shift, second)
            continue

        # ── Pass 2: internal gap (aligned filler, then silence) ───
        gap_fixed = _fix_internal_gap(cue, words, i)
        if gap_fixed:
            fixed += 1

    if fixed or skipped:
        logger.info(f"  Entry optimization: {fixed} fixed, {skipped} skipped (shift > {_MAX_SHIFT_S}s)")

    return cues


# ── Entry-point helpers ─────────────────────────────────


def _find_first_good_word(words):
    for w in words:
        if w.confidence > _FIRST_GOOD_CONF:
            return w
    return None


def _fix_internal_gap(cue, words, cue_index):
    """Shift entry past filler words when a large internal gap signals misalignment.

    Walks consecutive word pairs looking for a gap > _INTERNAL_GAP_THRESHOLD_S.
    If found, shifts the cue start to the *next* word (minus lead-in),
    provided that word isn't itself followed by another large gap.
    """
    for k in range(len(words) - 1):
        w_a = words[k]
        w_b = words[k + 1]
        gap_s = w_b.start - w_a.end
        if gap_s <= _INTERNAL_GAP_THRESHOLD_S:
            continue
        # w_b should be a stable anchor: its successor must follow within a normal gap.
        if k + 2 < len(words):
            next_gap = words[k + 2].start - w_b.end
            if next_gap > _INTERNAL_GAP_THRESHOLD_S:
                continue  # w_b is also isolated — keep scanning
        new_start = w_b.start - _LEAD_IN_S
        shift = new_start - cue.start
        if shift < 0.08:
            continue
        if w_b.confidence < _GAP_ANCHOR_MIN_CONF:
            logger.info(
                f"  ⏭ skip gap fix cue #{cue_index + 1}: "
                f"gap={gap_s:.2f}s after '{w_a.text}' "
                f"shift=+{shift:.2f}s "
                f"→ '{w_b.text}' conf={w_b.confidence:.2f} < {_GAP_ANCHOR_MIN_CONF}"
            )
            continue
        cue.start = new_start
        logger.info(
            f"  ✔ gap fix cue #{cue_index + 1}: "
            f"gap={gap_s:.2f}s after '{w_a.text}' "
            f"shift=+{shift:.2f}s "
            f"→ '{w_b.text}' (conf={w_b.confidence:.2f})"
        )
        return True
    return False


def _log_fix(cue_index, cue, first, peak_later, first_good, shift, second):
    signals = _entry_signals(first, second)
    logger.info(
        f"  ✔ entry fix cue #{cue_index + 1}: "
        f"'{first.text}' conf={first.confidence:.2f}→{peak_later:.2f} "
        f"shift=+{shift:.2f}s "
        f"→ '{first_good.text}' (conf={first_good.confidence:.2f})"
        f"{', ' + ', '.join(signals) if signals else ''}"
    )


def _log_skip(cue_index, cue, first, peak_later, first_good, shift, second):
    signals = _entry_signals(first, second)
    logger.info(
        f"  ⏭ skip entry fix cue #{cue_index + 1}: "
        f"'{first.text}' conf={first.confidence:.2f}→{peak_later:.2f} "
        f"shift=+{shift:.2f}s > {_MAX_SHIFT_S}s "
        f"→ '{first_good.text}' (conf={first_good.confidence:.2f})"
        f"{', ' + ', '.join(signals) if signals else ''}"
    )


def _entry_signals(first, second):
    signals = []
    first_dur_ms = (first.end - first.start) * 1000
    first_text_len = len(first.text.strip())
    if first_text_len <= 5 and first_dur_ms > _STRETCH_MIN_MS:
        signals.append(f"stretch={first_dur_ms:.0f}ms")
    gap = second.start - first.end
    if gap > _GAP_MIN_S:
        signals.append(f"gap={gap:.2f}s")
    return signals


def _get_aligned_words(cue, transcript_words):
    """Return transcript words that fall within the cue's time range."""
    margin = 0.05
    ws = cue.start - margin
    we = cue.end + margin
    result = [w for w in transcript_words if ws <= w.start <= we]
    return sorted(result, key=lambda w: w.start)
