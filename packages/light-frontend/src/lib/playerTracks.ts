import type videojs from "video.js";
import type { Chunk } from "../types";

export const MAIN_TRACK_LABEL = "light-main";

type Player = ReturnType<typeof videojs>;

/** True when chunk has a web-playable bilingual VTT track. */
export function chunkHasBilingual(chunk: Chunk | null): boolean {
  if (!chunk?.subtitles?.length) return false;
  return chunk.subtitles.some((k) => k === "bilingual.vtt" || k.endsWith(".bilingual.vtt"));
}

/** Resolve bilingual VTT URL for custom overlay rendering. */
export function resolveBilingualVttUrl(chunk: Chunk): string | null {
  const subs = chunk.subtitles ?? [];
  if (subs.includes("bilingual.vtt") || subs.some((k) => k.endsWith(".bilingual.vtt"))) {
    return `/api/chunks/${chunk.id}/subtitles/bilingual.vtt`;
  }
  return null;
}

/** Resolve main subtitle URL — prefer VTT, fall back to SRT. */
export function resolveMainSubUrl(chunk: Chunk, lang: string): string | null {
  const subs = chunk.subtitles ?? [];
  const hasVtt = subs.includes(`${lang}.vtt`) || subs.some((k) => k.endsWith(`.${lang}.vtt`));
  if (hasVtt) return `/api/chunks/${chunk.id}/subtitles/${lang}.vtt`;

  const hasSrt = subs.includes(`${lang}.srt`) || subs.some((k) => k.endsWith(`.${lang}.srt`));
  if (hasSrt) return `/api/chunks/${chunk.id}/subtitles/${lang}.srt`;

  return null;
}

/** Resolve annotation VTT URL (WebVTT only). */
export function resolveAnnotationUrl(chunk: Chunk): string | null {
  const subs = chunk.subtitles ?? [];
  const hasVtt = subs.includes("annotations.vtt") || subs.some((k) => k.endsWith(".annotations.vtt"));
  if (hasVtt) return `/api/chunks/${chunk.id}/subtitles/annotations.vtt`;
  if (subs.some((k) => k.includes(".annotations."))) {
    return `/api/chunks/${chunk.id}/subtitles/annotations.vtt`;
  }
  return null;
}

export function chunkHasAnnotations(chunk: Chunk | null): boolean {
  if (!chunk?.subtitles?.length) return false;
  return chunk.subtitles.some((k) => k.includes("annotations."));
}

function removeManagedTracks(player: Player): void {
  const existing = player.remoteTextTracks();
  for (let i = existing.length - 1; i >= 0; i--) {
    const track = existing[i];
    if (track.label === MAIN_TRACK_LABEL) {
      player.removeRemoteTextTrack(track);
    }
  }
}

function applyMainTrackMode(player: Player, enabled: boolean): void {
  const tracks = player.textTracks();
  for (let i = 0; i < tracks.length; i++) {
    const t = tracks[i];
    if (t.label === MAIN_TRACK_LABEL) {
      t.mode = enabled ? "showing" : "disabled";
    }
  }
}

function watchTrackLoad(track: TextTrack, onLoad: () => void): () => void {
  track.addEventListener("load", onLoad);
  return () => track.removeEventListener("load", onLoad);
}

/** Sync Video.js remote text track for main subtitles only (single-language mode). */
export function syncPlayerTextTracks(
  player: Player,
  chunk: Chunk,
  lang: string,
  mainEnabled: boolean,
  useBilingualOverlay = false,
): () => void {
  removeManagedTracks(player);

  if (mainEnabled && !useBilingualOverlay) {
    const mainUrl = resolveMainSubUrl(chunk, lang);
    if (mainUrl) {
      player.addRemoteTextTrack(
        {
          kind: "subtitles",
          src: mainUrl,
          srclang: lang,
          label: MAIN_TRACK_LABEL,
          default: true,
        },
        false,
      );
    }
  }

  const apply = () => applyMainTrackMode(player, mainEnabled);
  apply();

  const list = player.textTracks();
  const onListChange = () => apply();
  list.addEventListener("addtrack", onListChange);

  const loadUnsubs: (() => void)[] = [];
  for (let i = 0; i < list.length; i++) {
    const t = list[i];
    if (t.label === MAIN_TRACK_LABEL) {
      loadUnsubs.push(watchTrackLoad(t, apply));
    }
  }

  const timers = [50, 200, 500, 1500, 3000].map((ms) => window.setTimeout(apply, ms));

  return () => {
    list.removeEventListener("addtrack", onListChange);
    loadUnsubs.forEach((fn) => fn());
    timers.forEach((id) => window.clearTimeout(id));
  };
}
