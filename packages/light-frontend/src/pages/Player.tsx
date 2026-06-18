import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useParams, Link } from "react-router-dom";
import useSWR from "swr";
import type videojs from "video.js";
import { fetchJSON, listenSSE, postJSON } from "../api/client";
import ProgressStepper from "../components/ProgressStepper";
import SubtitleControls from "../components/SubtitleControls";
import { extractSubLang, cueTextAt, parseVtt, type TimedCue } from "../lib/vtt";
import { attachMobilePlayerGestures } from "../lib/mobilePlayerGestures";
import { attachPauseIconOverlay } from "../lib/pauseIconOverlay";
import {
  chunkHasAnnotations,
  resolveAnnotationUrl,
  syncPlayerTextTracks,
} from "../lib/playerTracks";
import type { Video } from "../types";

interface SavedPlaybackPosition {
  version: number;
  chunkId: string;
  chunkIndex: number;
  time: number;
  muted: boolean;
  volume: number;
  wasPlaying: boolean;
  updatedAt: number;
}

const PLAYBACK_SAVE_INTERVAL_MS = 2000;
const PLAYBACK_RESTORE_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

function playbackStorageKey(videoId: string): string {
  return `light:player:${videoId}:position`;
}

function readSavedPlaybackPosition(videoId: string | undefined): SavedPlaybackPosition | null {
  if (!videoId || typeof window === "undefined") return null;

  try {
    const raw = window.localStorage.getItem(playbackStorageKey(videoId));
    if (!raw) return null;

    const parsed = JSON.parse(raw) as Partial<SavedPlaybackPosition>;
    if (
      typeof parsed.chunkId !== "string"
      || typeof parsed.chunkIndex !== "number"
      || typeof parsed.time !== "number"
      || typeof parsed.updatedAt !== "number"
    ) {
      return null;
    }

    if (Date.now() - parsed.updatedAt > PLAYBACK_RESTORE_MAX_AGE_MS) {
      window.localStorage.removeItem(playbackStorageKey(videoId));
      return null;
    }

    const hasTrustedAudioState = parsed.version === 2 || parsed.version === 3;
    return {
      version: 3,
      chunkId: parsed.chunkId,
      chunkIndex: Math.max(0, Math.floor(parsed.chunkIndex)),
      time: Math.max(0, parsed.time),
      muted: hasTrustedAudioState && typeof parsed.muted === "boolean" ? parsed.muted : false,
      volume: hasTrustedAudioState && typeof parsed.volume === "number" ? Math.min(1, Math.max(0, parsed.volume)) : 1,
      wasPlaying: parsed.version === 3 && typeof parsed.wasPlaying === "boolean" ? parsed.wasPlaying : true,
      updatedAt: parsed.updatedAt,
    };
  } catch {
    return null;
  }
}

function writeSavedPlaybackPosition(videoId: string, position: SavedPlaybackPosition): void {
  if (typeof window === "undefined") return;

  try {
    window.localStorage.setItem(playbackStorageKey(videoId), JSON.stringify(position));
  } catch {
    // Ignore storage failures on private browsing or quota errors.
  }
}

function BackLink() {
  return (
    <Link
      to="/"
      aria-label="返回"
      className="inline-flex items-center justify-center w-8 h-8 rounded-lg text-[#6b7280] hover:text-[#e5e5e5] hover:bg-[#1f1f1f] transition-colors shrink-0"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M15 18l-6-6 6-6" />
      </svg>
    </Link>
  );
}

function formatDuration(sec: number | null): string {
  if (!sec) return "00:00";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function Player() {
  const { id } = useParams<{ id: string }>();
  const { data: video, error, isLoading, mutate } = useSWR<Video>(
    id ? `/api/videos/${id}` : null,
    (path: string) => fetchJSON(path),
    {
      // Only poll while the pipeline is still running; done videos fetch once.
      refreshInterval: (latest) =>
        latest?.status === "processing" || latest?.status === "pending" ? 3000 : 0,
    },
  );

  const [currentChunkIdx, setCurrentChunkIdx] = useState(0);
  const [subLang, setSubLang] = useState<string>("zh");
  const [subEnabled, setSubEnabled] = useState(true);
  const [annotationsEnabled, setAnnotationsEnabled] = useState(false);
  const [playerEpoch, setPlayerEpoch] = useState(0);
  const [annotationText, setAnnotationText] = useState("");
  const [annotationOverlayEl, setAnnotationOverlayEl] = useState<HTMLDivElement | null>(null);
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<ReturnType<typeof videojs> | null>(null);
  const trackCleanupRef = useRef<(() => void) | null>(null);
  const annotationCuesRef = useRef<TimedCue[]>([]);
  const subSettingsRef = useRef({ subLang, subEnabled });
  const savedPlaybackRef = useRef<SavedPlaybackPosition | null>(null);
  const pendingSeekRef = useRef<{ chunkId: string; time: number } | null>(null);
  const restoredVideoIdRef = useRef<string | null>(null);
  const lastPlaybackSaveAtRef = useRef(0);
  const shouldAutoPlayRef = useRef(true);

  subSettingsRef.current = { subLang, subEnabled };

  // Restore local playback position when navigating to a video.
  useEffect(() => {
    const saved = readSavedPlaybackPosition(id);
    savedPlaybackRef.current = saved;
    pendingSeekRef.current = saved ? { chunkId: saved.chunkId, time: saved.time } : null;
    restoredVideoIdRef.current = null;
    lastPlaybackSaveAtRef.current = 0;
    shouldAutoPlayRef.current = saved?.wasPlaying ?? true;
    setCurrentChunkIdx(saved?.chunkIndex ?? 0);
    setSubEnabled(true);
  }, [id]);

  const chunks = video?.chunks ?? [];
  const currentChunk = chunks[currentChunkIdx] ?? null;

  useEffect(() => {
    if (!id || chunks.length === 0 || restoredVideoIdRef.current === id) return;
    restoredVideoIdRef.current = id;

    const saved = savedPlaybackRef.current;
    if (!saved) {
      if (currentChunkIdx >= chunks.length) setCurrentChunkIdx(0);
      return;
    }

    const byId = chunks.findIndex((chunk) => chunk.id === saved.chunkId);
    const nextIdx = byId >= 0 ? byId : Math.min(saved.chunkIndex, chunks.length - 1);
    const nextChunk = chunks[nextIdx];

    if (nextChunk) {
      pendingSeekRef.current = { chunkId: nextChunk.id, time: saved.time };
    }
    if (currentChunkIdx !== nextIdx) setCurrentChunkIdx(nextIdx);
  }, [id, chunks, currentChunkIdx]);

  // Default annotations on when current chunk has annotation files
  useEffect(() => {
    setAnnotationsEnabled(chunkHasAnnotations(currentChunk));
  }, [id, currentChunk?.id, currentChunk?.subtitles]);

  // Pick default language when opening a video
  useEffect(() => {
    if (!video?.chunks?.length) return;
    const langs = new Set<string>();
    for (const c of video.chunks) {
      for (const k of c.subtitles ?? []) {
        const lang = extractSubLang(k);
        if (lang) langs.add(lang);
      }
    }
    if (langs.size === 0) return;
    setSubLang(langs.has("zh") ? "zh" : Array.from(langs)[0]);
  }, [id, video?.id]);

  const syncTracks = useCallback(() => {
    const player = playerRef.current;
    const chunk = currentChunk;
    if (!player || !chunk) return;

    trackCleanupRef.current?.();
    const { subLang: lang, subEnabled: mainOn } = subSettingsRef.current;
    trackCleanupRef.current = syncPlayerTextTracks(player, chunk, lang, mainOn);
  }, [currentChunk]);

  const savePlaybackPosition = useCallback((chunkIdx = currentChunkIdx, time?: number, wasPlaying?: boolean) => {
    if (!id) return;

    const chunk = chunks[chunkIdx];
    if (!chunk) return;

    const player = playerRef.current;
    const position: SavedPlaybackPosition = {
      version: 3,
      chunkId: chunk.id,
      chunkIndex: chunkIdx,
      time: Math.max(0, time ?? player?.currentTime() ?? 0),
      muted: player?.muted() ?? savedPlaybackRef.current?.muted ?? false,
      volume: player?.volume() ?? savedPlaybackRef.current?.volume ?? 1,
      wasPlaying: wasPlaying ?? (player ? !player.paused() : savedPlaybackRef.current?.wasPlaying ?? false),
      updatedAt: Date.now(),
    };
    savedPlaybackRef.current = position;
    writeSavedPlaybackPosition(id, position);
  }, [chunks, currentChunkIdx, id]);

  const continuePlayback = useCallback(() => {
    const player = playerRef.current;
    if (!player) return;

    const result = player.play();
    if (result && typeof result.then === "function") {
      result.then(() => {
        savePlaybackPosition(currentChunkIdx, player.currentTime(), true);
      });
      return;
    }

    savePlaybackPosition(currentChunkIdx, player.currentTime(), true);
  }, [currentChunkIdx, savePlaybackPosition]);

  // ── SSE progress tracking while processing (reconnects on re-entry) ──
  const [pipelineEvents, setPipelineEvents] = useState<Record<string, unknown>[]>([]);
  const sseCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const isProcessing = video?.status === "processing" || video?.status === "pending";
    const hasNoChunks = !video?.chunks || video.chunks.length === 0;

    if (isProcessing && hasNoChunks && id) {
      sseCleanupRef.current?.();
      sseCleanupRef.current = listenSSE(
        id,
        (ev) => setPipelineEvents((prev) => [...prev, ev]),
        () => {
          mutate();
          sseCleanupRef.current = null;
        },
        () => {},
      );
    } else {
      sseCleanupRef.current?.();
      sseCleanupRef.current = null;
    }

    return () => {
      sseCleanupRef.current?.();
      sseCleanupRef.current = null;
    };
  }, [video?.status, video?.chunks?.length, id, mutate]);

  useEffect(() => {
    setPipelineEvents([]);
  }, [id]);

  // ── Init player — always create fresh on chunk switch ──
  useEffect(() => {
    if (!playerContainerRef.current || !currentChunk) return;

    const streamUrl = `/api/chunks/${currentChunk.id}/stream`;
    const ext = currentChunk.video_ext?.toLowerCase() ?? "mp4";
    const mimeType = ext === "webm" ? "video/webm"
      : ext === "mkv" ? "video/x-matroska"
      : "video/mp4";
    let cancelled = false;
    let autoPlayFallbackTimer: ReturnType<typeof setTimeout> | null = null;
    let removeMobileGestures: (() => void) | null = null;
    let removePauseIcon: (() => void) | null = null;

    async function init() {
      const { default: videojs } = await import("video.js");
      if (cancelled || !playerContainerRef.current) return;

      trackCleanupRef.current?.();
      trackCleanupRef.current = null;
      setAnnotationOverlayEl(null);

      if (playerRef.current) {
        playerRef.current.dispose();
        playerRef.current = null;
      }

      const container = playerContainerRef.current;
      container.innerHTML = "";
      const el = document.createElement("video");
      el.className = "video-js vjs-theme-dark vjs-big-play-centered";
      el.setAttribute("playsinline", "");
      el.setAttribute("webkit-playsinline", "");
      el.setAttribute("x5-playsinline", "");
      el.style.width = "100%";
      container.appendChild(el);

      const player = videojs(el, {
        controls: true,
        autoplay: false,
        // Eager buffering so the video decoder has a head start; otherwise the
        // audio clock (fast Opus decode) runs ahead of the slower video decode
        // (e.g. AV1 software decoding) and the first seconds show only the poster.
        preload: "auto",
        fluid: true,
        aspectRatio: "16:9",
        html5: {
          nativeTextTracks: false,
        },
        controlBar: {
          subsCapsButton: false,
        },
      });

      const savedPlayback = savedPlaybackRef.current;
      if (savedPlayback) {
        player.volume(savedPlayback.volume);
        player.muted(savedPlayback.muted);
      }

      player.src({ type: mimeType, src: streamUrl });
      playerRef.current = player;

      removePauseIcon = attachPauseIconOverlay(player);

      // Mobile: tap to play when paused, double-tap to pause when playing; horizontal drag to scrub.
      if (window.matchMedia("(pointer: coarse)").matches) {
        removeMobileGestures = attachMobilePlayerGestures(player);
      }

      const annotationOverlay = document.createElement("div");
      player.el().appendChild(annotationOverlay);
      setAnnotationOverlayEl(annotationOverlay);

      let autoPlayAttempted = false;

      const restorePosition = () => {
        const pendingSeek = pendingSeekRef.current;
        if (!pendingSeek || pendingSeek.chunkId !== currentChunk.id || pendingSeek.time <= 0) return;

        const duration = currentChunk.duration;
        const maxTime = duration && duration > 1 ? duration - 0.5 : pendingSeek.time;
        try {
          player.currentTime(Math.min(pendingSeek.time, maxTime));
          pendingSeekRef.current = null;
        } catch {
          // Some mobile browsers only allow seeking after more metadata arrives.
        }
      };

      const requestAutoPlay = () => {
        if (cancelled || !shouldAutoPlayRef.current || autoPlayAttempted) return;
        autoPlayAttempted = true;
        if (autoPlayFallbackTimer) {
          clearTimeout(autoPlayFallbackTimer);
          autoPlayFallbackTimer = null;
        }
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => {
            if (!cancelled) continuePlayback();
          });
        });
      };

      const markReady = () => setPlayerEpoch((e) => e + 1);
      player.one("loadedmetadata", () => {
        restorePosition();
        markReady();
      });
      player.one("loadeddata", () => {
        restorePosition();
        markReady();
      });
      // Prefer starting once enough is buffered (canplaythrough) so the video
      // decode does not lag behind audio at startup. Fall back to canplay after
      // a short grace period in case canplaythrough never fires on this source.
      player.one("canplaythrough", () => {
        restorePosition();
        requestAutoPlay();
      });
      player.one("canplay", () => {
        restorePosition();
        markReady();
        if (!autoPlayFallbackTimer) {
          autoPlayFallbackTimer = setTimeout(requestAutoPlay, 1500);
        }
      });
    }

    init();
    return () => {
      cancelled = true;
      removePauseIcon?.();
      removePauseIcon = null;
      removeMobileGestures?.();
      removeMobileGestures = null;
      if (autoPlayFallbackTimer) {
        clearTimeout(autoPlayFallbackTimer);
        autoPlayFallbackTimer = null;
      }
      setAnnotationOverlayEl(null);
    };
  }, [currentChunkIdx, currentChunk?.id]);

  // ── Sync Video.js text tracks when settings or player change ──
  useEffect(() => {
    syncTracks();
    return () => {
      trackCleanupRef.current?.();
      trackCleanupRef.current = null;
    };
  }, [subLang, subEnabled, playerEpoch, syncTracks]);

  // ── Annotation overlay (top-left) — independent of Video.js text tracks ──
  useEffect(() => {
    if (!annotationsEnabled || !currentChunk) {
      annotationCuesRef.current = [];
      setAnnotationText("");
      return;
    }

    const url = resolveAnnotationUrl(currentChunk);
    if (!url) {
      annotationCuesRef.current = [];
      setAnnotationText("");
      return;
    }

    let cancelled = false;
    fetch(url)
      .then((res) => (res.ok ? res.text() : Promise.reject(new Error(String(res.status)))))
      .then((text) => {
        if (cancelled) return;
        annotationCuesRef.current = parseVtt(text);
        const t = playerRef.current?.currentTime() ?? 0;
        setAnnotationText(cueTextAt(annotationCuesRef.current, t));
      })
      .catch(() => {
        if (!cancelled) {
          annotationCuesRef.current = [];
          setAnnotationText("");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [annotationsEnabled, currentChunk?.id, currentChunk?.subtitles]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player || !annotationsEnabled) {
      setAnnotationText("");
      return;
    }

    const onTime = () => {
      setAnnotationText(cueTextAt(annotationCuesRef.current, player.currentTime()));
    };

    player.on("timeupdate", onTime);
    player.on("seeked", onTime);
    onTime();

    return () => {
      player.off("timeupdate", onTime);
      player.off("seeked", onTime);
    };
  }, [annotationsEnabled, playerEpoch, currentChunk?.id]);

  // Persist playback position so mobile tab discard/background restores correctly.
  useEffect(() => {
    const player = playerRef.current;
    if (!player || !currentChunk) return;

    const saveNow = () => savePlaybackPosition(currentChunkIdx, player.currentTime());
    const saveThrottled = () => {
      const now = Date.now();
      if (now - lastPlaybackSaveAtRef.current < PLAYBACK_SAVE_INTERVAL_MS) return;
      lastPlaybackSaveAtRef.current = now;
      saveNow();
    };
    const saveWhenHidden = () => {
      if (document.visibilityState === "hidden") saveNow();
    };
    const onPlay = () => {
      saveNow();
    };

    player.on("timeupdate", saveThrottled);
    player.on("play", onPlay);
    player.on("pause", saveNow);
    player.on("seeked", saveNow);
    player.on("volumechange", saveNow);
    window.addEventListener("pagehide", saveNow);
    document.addEventListener("visibilitychange", saveWhenHidden);

    return () => {
      player.off("timeupdate", saveThrottled);
      player.off("play", onPlay);
      player.off("pause", saveNow);
      player.off("seeked", saveNow);
      player.off("volumechange", saveNow);
      window.removeEventListener("pagehide", saveNow);
      document.removeEventListener("visibilitychange", saveWhenHidden);
    };
  }, [currentChunk, currentChunkIdx, playerEpoch, savePlaybackPosition]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      trackCleanupRef.current?.();
      setAnnotationOverlayEl(null);
      if (playerRef.current) {
        playerRef.current.dispose();
        playerRef.current = null;
      }
    };
  }, []);

  // ── Auto-advance to next chunk ──
  useEffect(() => {
    const player = playerRef.current;
    if (!player) return;

    const onEnded = () => {
      if (currentChunkIdx < chunks.length - 1) {
        const nextIdx = currentChunkIdx + 1;
        pendingSeekRef.current = null;
        shouldAutoPlayRef.current = true;
        savePlaybackPosition(nextIdx, 0, true);
        setCurrentChunkIdx(nextIdx);
      }
    };

    player.on("ended", onEnded);
    return () => {
      player.off("ended", onEnded);
    };
  }, [currentChunkIdx, chunks.length, playerEpoch, savePlaybackPosition]);

  if (isLoading) {
    return <div className="flex items-center justify-center h-64 text-[#6b7280]">加载中...</div>;
  }

  if (error || !video) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-[#ef4444] gap-3">
        <p>加载失败</p>
        <button onClick={() => mutate()} className="text-xs text-[#3b82f6] hover:underline">重试</button>
      </div>
    );
  }

  const isMultiChunk = chunks.length > 1;
  const isProcessing = video.status === "processing" || video.status === "pending";
  const hasNoChunks = chunks.length === 0;
  const hasAnnotations = chunkHasAnnotations(currentChunk);

  if (isProcessing && hasNoChunks) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <BackLink />
          <h1 className="text-sm font-medium truncate">{video.title || "未命名"}</h1>
          <span className="text-xs text-[#3b82f6] animate-pulse">处理中...</span>
        </div>

        <div className="p-5 rounded-xl bg-[#141414] border border-[#1f1f1f] space-y-6 max-w-2xl">
          <ProgressStepper events={pipelineEvents} />
          <div className="text-sm text-[#888]">
            {pipelineEvents.length === 0 ? "正在连接..." : "可以离开此页面，处理不会中断。"}
          </div>
        </div>
      </div>
    );
  }

  if (video.status === "error") {
    const handleRetry = async () => {
      await postJSON(`/api/videos/${video.id}/retry`, {});
      mutate();
    };
    const errorMsg = video.run?.error_msg as string | undefined;

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <BackLink />
          <h1 className="text-sm font-medium truncate">{video.title || "未命名"}</h1>
          <span className="text-xs text-[#ef4444]">失败</span>
        </div>

        <div className="p-5 rounded-xl bg-[#141414] border border-[#1f1f1f] space-y-4 max-w-2xl">
          {errorMsg && (
            <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 rounded-lg px-4 py-3 font-mono">
              {errorMsg}
            </div>
          )}
          <button
            onClick={handleRetry}
            className="px-4 py-2 rounded-lg bg-[#3b82f6] text-white text-sm hover:bg-[#2563eb] transition-colors"
          >
            重新处理（复用已下载视频）
          </button>
        </div>
      </div>
    );
  }

  const subLangs = new Set<string>();
  for (const c of chunks) {
    for (const k of c.subtitles ?? []) {
      const lang = extractSubLang(k);
      if (lang) subLangs.add(lang);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <BackLink />
        <h1 className="text-sm font-medium truncate">{video.title || "未命名"}</h1>
        {video.status === "processing" && (
          <span className="text-xs text-[#3b82f6] animate-pulse">处理中...</span>
        )}
      </div>

      <div className="flex flex-col lg:flex-row gap-4">
        <div className="flex-1">
          <div
            ref={playerContainerRef}
            className="aspect-video max-h-[75vh] max-w-full mx-auto rounded-xl overflow-hidden bg-black relative"
          />
          {annotationOverlayEl && annotationsEnabled && annotationText && (
            createPortal(
              <div className="vjs-light-annotation-overlay">
                {annotationText}
              </div>,
              annotationOverlayEl,
            )
          )}

          <div className="mt-3">
            <SubtitleControls
              languages={Array.from(subLangs)}
              subLang={subLang}
              subEnabled={subEnabled}
              annotationsEnabled={annotationsEnabled}
              hasAnnotations={hasAnnotations}
              onSubEnabledChange={setSubEnabled}
              onSubLangChange={setSubLang}
              onAnnotationsChange={setAnnotationsEnabled}
            />
          </div>
        </div>

        {isMultiChunk && (
          <div className="lg:w-60 shrink-0">
            <h3 className="text-xs text-[#6b7280] mb-2 font-medium uppercase tracking-wider">
              片段
            </h3>
            <div className="space-y-1">
              {chunks.map((c, i) => (
                <button
                  key={c.id}
                  onClick={() => {
                    pendingSeekRef.current = null;
                    shouldAutoPlayRef.current = true;
                    savePlaybackPosition(i, 0, true);
                    setCurrentChunkIdx(i);
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-sm transition-colors ${
                    i === currentChunkIdx
                      ? "bg-[#3b82f6]/10 text-[#3b82f6]"
                      : "text-[#888] hover:bg-[#1f1f1f]"
                  }`}
                >
                  <span className="text-xs font-mono w-5 text-right">
                    {i === currentChunkIdx ? "\u25B6" : i + 1}
                  </span>
                  <span className="truncate">{formatDuration(c.duration)}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
