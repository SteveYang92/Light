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
