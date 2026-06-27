/** Parsed timed subtitle cue for custom overlay rendering. */
export interface TimedCue {
  start: number;
  end: number;
  text: string;
}

/** @deprecated Use TimedCue */
export type VttCue = TimedCue;

/** Convert WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds. */
export function vttTimeToSeconds(time: string): number {
  const [hms, msPart = "0"] = time.trim().split(".");
  const parts = hms.split(":").map(Number);
  const ms = Number(msPart.padEnd(3, "0").slice(0, 3));
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2] + ms / 1000;
  }
  return parts[0] * 60 + parts[1] + ms / 1000;
}

function srtTimeToSeconds(time: string): number {
  const [hms, msPart = "0"] = time.trim().split(",");
  const parts = hms.split(":").map(Number);
  return parts[0] * 3600 + parts[1] * 60 + parts[2] + Number(msPart) / 1000;
}

/** Convert ASS-style escaped line breaks into real text line breaks. */
export function normalizeSubtitleText(text: string): string {
  return text.replace(/\\[Nn]/g, "\n");
}

/** Parse WebVTT into timed cues (ignores cue settings line). */
export function parseVtt(content: string): TimedCue[] {
  const cues: TimedCue[] = [];
  const normalized = content.replace(/\r\n/g, "\n").replace(/^\uFEFF/, "");
  if (!normalized.startsWith("WEBVTT")) return cues;

  const blocks = normalized.slice(normalized.indexOf("\n") + 1).split(/\n\n+/);
  for (const block of blocks) {
    const lines = block.trim().split("\n").filter(Boolean);
    if (lines.length < 2) continue;

    let timingIdx = 0;
    if (!lines[0].includes("-->")) {
      timingIdx = 1;
      if (lines.length < 3) continue;
    }

    const timing = lines[timingIdx].match(
      /(\d{1,2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?\.\d{3})/,
    );
    if (!timing) continue;

    cues.push({
      start: vttTimeToSeconds(timing[1]),
      end: vttTimeToSeconds(timing[2]),
      text: normalizeSubtitleText(lines.slice(timingIdx + 1).join("\n")),
    });
  }

  return cues;
}

/** Alias for parseVtt — annotation files use the same format. */
export const parseAnnotationVtt = parseVtt;

/** Parse SRT into timed cues. */
export function parseSrt(content: string): TimedCue[] {
  const cues: TimedCue[] = [];
  const normalized = content.replace(/\r\n/g, "\n").replace(/^\uFEFF/, "");
  const blocks = normalized.trim().split(/\n\n+/);

  for (const block of blocks) {
    const lines = block.trim().split("\n").filter(Boolean);
    if (lines.length < 2) continue;

    let timingIdx = 0;
    if (!lines[0].includes("-->")) {
      timingIdx = 1;
      if (lines.length < 3) continue;
    }

    const timing = lines[timingIdx].match(
      /(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/,
    );
    if (!timing) continue;

    cues.push({
      start: srtTimeToSeconds(timing[1]),
      end: srtTimeToSeconds(timing[2]),
      text: normalizeSubtitleText(lines.slice(timingIdx + 1).join("\n")),
    });
  }

  return cues;
}

/** Remove leading ※ markers (legacy ASS→VTT conversions may duplicate them). */
export function stripAnnotationMarker(text: string): string {
  return normalizeSubtitleText(text).replace(/^\s*(?:※\s*)+/, "").trim();
}

/** Normalize annotation text to exactly one leading ※ marker. */
export function formatAnnotationDisplay(text: string): string {
  const body = stripAnnotationMarker(text);
  return body ? `※ ${body}` : "";
}

/** Return the active cue text at the given playback time, if any. */
export function cueTextAt(cues: TimedCue[], time: number): string {
  for (const cue of cues) {
    if (time >= cue.start && time < cue.end) return formatAnnotationDisplay(cue.text);
  }
  return "";
}

/** Like ``cueTextAt`` but without annotation ※ formatting (main/bilingual subs). */
export function cueTextAtRaw(cues: TimedCue[], time: number): string {
  for (const cue of cues) {
    if (time >= cue.start && time < cue.end) return normalizeSubtitleText(cue.text);
  }
  return "";
}

/** Marker line between ZH block and EN block in ``bilingual.vtt`` cues. */
export const BILINGUAL_CUE_MARKER = "<<EN>>";

function isPredominantlyLatin(text: string): boolean {
  const stripped = text.replace(/\s/g, "");
  if (!stripped) return false;
  const latin = (stripped.match(/[A-Za-z]/g) ?? []).length;
  return latin / stripped.length >= 0.5;
}

/** If the EN slot is actually Chinese (legacy mis-split), merge back into ZH. */
function reconcileBilingualSplit(zh: string, en: string): { zh: string; en: string } {
  if (en && !isPredominantlyLatin(en)) {
    return { zh: [zh, en].filter(Boolean).join("\n"), en: "" };
  }
  return { zh, en };
}

function splitLegacyBilingualCue(normalized: string): { zh: string; en: string } {
  const blank = normalized.indexOf("\n\n");
  if (blank >= 0) {
    return {
      zh: normalized.slice(0, blank).trim(),
      en: normalized.slice(blank + 2).trim(),
    };
  }
  const lines = normalized.split("\n");
  if (lines.length <= 1) {
    return { zh: normalized.trim(), en: "" };
  }
  const last = lines[lines.length - 1]?.trim() ?? "";
  if (isPredominantlyLatin(last)) {
    return {
      zh: lines.slice(0, -1).join("\n").trim(),
      en: last,
    };
  }
  return { zh: normalized.trim(), en: "" };
}

/** Split a bilingual VTT cue into ZH (first block) and EN (second block). */
export function splitBilingualCue(text: string): { zh: string; en: string } {
  const normalized = normalizeSubtitleText(text);
  const marker = `\n${BILINGUAL_CUE_MARKER}\n`;
  const idx = normalized.indexOf(marker);
  if (idx >= 0) {
    return reconcileBilingualSplit(
      normalized.slice(0, idx).trim(),
      normalized.slice(idx + marker.length).trim(),
    );
  }
  const legacy = splitLegacyBilingualCue(normalized);
  return reconcileBilingualSplit(legacy.zh, legacy.en);
}

/** Extract language code from a subtitle map key (zh.srt, video_p1.zh.vtt, …). */
export function extractSubLang(key: string): string | null {
  if (key.includes(".annotations.") || key.startsWith("annotations.")) return null;
  if (key.startsWith("bilingual")) return null;

  const simple = key.match(/^([a-z]{2})\.(?:srt|vtt|ass)$/i);
  if (simple) return simple[1].toLowerCase();

  const prefixed = key.match(/\.([a-z]{2})\.(?:srt|vtt|ass)$/i);
  if (prefixed) return prefixed[1].toLowerCase();

  return null;
}
