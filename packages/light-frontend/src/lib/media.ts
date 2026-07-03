import type { Chunk } from "../types";

const VIDEO_MIME: Record<string, string> = {
  mp4: "video/mp4",
  webm: "video/webm",
  mkv: "video/x-matroska",
};

const AUDIO_MIME: Record<string, string> = {
  mp3: "audio/mpeg",
  m4a: "audio/mp4",
  wav: "audio/wav",
  flac: "audio/flac",
  ogg: "audio/ogg",
  aac: "audio/aac",
  opus: "audio/opus",
  weba: "audio/webm",
};

export function isAudioChunk(chunk: Chunk): boolean {
  return chunk.media_kind === "audio";
}

export function isAudioVideo(video: { chunks: Chunk[] }): boolean {
  return video.chunks.some(isAudioChunk);
}

export function streamMimeType(ext: string, kind: "audio" | "video"): string {
  const normalized = ext.toLowerCase();
  if (kind === "audio") {
    return AUDIO_MIME[normalized] ?? "audio/mpeg";
  }
  return VIDEO_MIME[normalized] ?? "video/mp4";
}
