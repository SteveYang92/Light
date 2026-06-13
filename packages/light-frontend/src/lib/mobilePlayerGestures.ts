import type videojs from "video.js";

type Player = ReturnType<typeof videojs>;

const TAP_SLOP_PX = 10;
/** Horizontal movement before scrub mode engages. */
const SCRUB_START_PX = 5;
const DOUBLE_TAP_INTERVAL_MS = 300;
/** Full-width fast drag maps to this fraction of duration, clamped below. */
const SCRUB_DURATION_RATIO = 0.1;
const SCRUB_SPAN_MIN_SEC = 20;
const SCRUB_SPAN_MAX_SEC = 45;
/** ~0.8 px/ms (~800 px/s) treated as "fast"; slower drags scale span down. */
const SCRUB_REF_SPEED_PX_PER_MS = 0.8;
const SCRUB_SPEED_FACTOR_MIN = 0.4;
const SCRUB_SPEED_FACTOR_MAX = 1;

function formatClock(sec: number): string {
  const total = Math.max(0, Math.floor(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatDelta(sec: number): string {
  const sign = sec >= 0 ? "+" : "-";
  const total = Math.max(0, Math.floor(Math.abs(sec)));
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (m > 0) {
    return `${sign}${m}:${String(s).padStart(2, "0")}`;
  }
  return `${sign}${s}s`;
}

function isControlTarget(target: HTMLElement | null): boolean {
  return Boolean(target?.closest(".vjs-control-bar, .vjs-big-play-button, .vjs-menu"));
}

/** Attach tap-to-play (paused) / double-tap-to-pause (playing) and horizontal scrub gestures. */
export function attachMobilePlayerGestures(player: Player): () => void {
  const root = player.el();

  let startX = 0;
  let startY = 0;
  let anchorTime = 0;
  let scrubTime = 0;
  let mode: "idle" | "pending" | "scrub" = "idle";
  let overlay: HTMLDivElement | null = null;
  let lastTapTime = 0;
  let lastTapX = 0;
  let lastTapY = 0;
  let scrubEngagedMs = 0;

  const clampTime = (time: number): number => {
    const duration = player.duration();
    if (!duration || !Number.isFinite(duration)) {
      return Math.max(0, time);
    }
    return Math.max(0, Math.min(duration - 0.05, time));
  };

  const hideOverlay = () => {
    overlay?.remove();
    overlay = null;
  };

  const baseScrubSpanSec = (): number => {
    const duration = player.duration() || 0;
    if (!duration || !Number.isFinite(duration)) {
      return SCRUB_SPAN_MIN_SEC;
    }
    return Math.min(
      Math.max(duration * SCRUB_DURATION_RATIO, SCRUB_SPAN_MIN_SEC),
      SCRUB_SPAN_MAX_SEC,
    );
  };

  /** Scale seek range by drag speed: slow/precise drags move less time per pixel. */
  const scrubSpanSec = (absDx: number): number => {
    const elapsedMs = Math.max(Date.now() - scrubEngagedMs, 80);
    const speedPxPerMs = absDx / elapsedMs;
    const speedFactor = Math.min(
      SCRUB_SPEED_FACTOR_MAX,
      Math.max(SCRUB_SPEED_FACTOR_MIN, speedPxPerMs / SCRUB_REF_SPEED_PX_PER_MS),
    );
    return baseScrubSpanSec() * speedFactor;
  };

  const showOverlay = (deltaSec: number, targetSec: number) => {
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "vjs-light-seek-overlay";
      root.appendChild(overlay);
    }
    overlay.innerHTML = `<span class="vjs-light-seek-delta">${formatDelta(deltaSec)}</span>`
      + `<span class="vjs-light-seek-target">${formatClock(targetSec)}</span>`;
  };

  const onTouchStart = (event: TouchEvent) => {
    if (event.touches.length !== 1) return;
    if (isControlTarget(event.target as HTMLElement | null)) return;

    const touch = event.touches[0];
    startX = touch.clientX;
    startY = touch.clientY;
    anchorTime = player.currentTime() || 0;
    scrubTime = anchorTime;
    mode = "pending";
  };

  const onTouchMove = (event: TouchEvent) => {
    if (mode === "idle" || event.touches.length !== 1) return;

    const touch = event.touches[0];
    const dx = touch.clientX - startX;
    const dy = touch.clientY - startY;

    if (mode === "pending") {
      const adx = Math.abs(dx);
      const ady = Math.abs(dy);
      if (adx < TAP_SLOP_PX && ady < TAP_SLOP_PX) return;
      // Vertical scroll intent cancels scrub; tap/double-tap still handled on touchend.
      if (ady > adx && ady >= SCRUB_START_PX) {
        mode = "idle";
        return;
      }
      if (adx >= SCRUB_START_PX && adx >= ady) {
        mode = "scrub";
        scrubEngagedMs = Date.now();
        lastTapTime = 0;
      } else {
        return;
      }
    }

    if (mode !== "scrub") return;

    event.preventDefault();
    const width = root.clientWidth || window.innerWidth;
    const ratio = dx / width;
    scrubTime = clampTime(anchorTime + ratio * scrubSpanSec(Math.abs(dx)));
    showOverlay(scrubTime - anchorTime, scrubTime);
  };

  const onTouchEnd = (event: TouchEvent) => {
    if (mode === "scrub") {
      event.preventDefault();
      player.currentTime(scrubTime);
      hideOverlay();
      mode = "idle";
      return;
    }

    if (mode === "pending" && !isControlTarget(event.target as HTMLElement | null)) {
      event.preventDefault();
      const touch = event.changedTouches[0];
      if (!touch) {
        mode = "idle";
        hideOverlay();
        return;
      }

      const now = Date.now();
      if (player.paused()) {
        void player.play();
        lastTapTime = 0;
      } else {
        const isDoubleTap =
          now - lastTapTime <= DOUBLE_TAP_INTERVAL_MS
          && Math.abs(touch.clientX - lastTapX) <= TAP_SLOP_PX
          && Math.abs(touch.clientY - lastTapY) <= TAP_SLOP_PX;

        if (isDoubleTap) {
          player.pause();
          lastTapTime = 0;
        } else {
          lastTapTime = now;
          lastTapX = touch.clientX;
          lastTapY = touch.clientY;
        }
      }
    }

    mode = "idle";
    hideOverlay();
  };

  const onTouchCancel = () => {
    mode = "idle";
    lastTapTime = 0;
    hideOverlay();
  };

  root.addEventListener("touchstart", onTouchStart, { passive: true });
  root.addEventListener("touchmove", onTouchMove, { passive: false });
  root.addEventListener("touchend", onTouchEnd, { passive: false });
  root.addEventListener("touchcancel", onTouchCancel, { passive: true });

  return () => {
    root.removeEventListener("touchstart", onTouchStart);
    root.removeEventListener("touchmove", onTouchMove);
    root.removeEventListener("touchend", onTouchEnd);
    root.removeEventListener("touchcancel", onTouchCancel);
    hideOverlay();
  };
}
