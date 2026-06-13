import type videojs from "video.js";

type Player = ReturnType<typeof videojs>;

/** Show a centered play icon whenever the player is paused. */
export function attachPauseIconOverlay(player: Player): () => void {
  const overlay = document.createElement("div");
  overlay.className = "vjs-light-pause-overlay";

  const badge = document.createElement("div");
  badge.className = "vjs-light-pause-badge";
  badge.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M8 5.14v13.72L19 12 8 5.14z" fill="#fff" />
  </svg>`;
  overlay.appendChild(badge);
  player.el().appendChild(overlay);

  const sync = () => {
    overlay.style.display = player.paused() ? "flex" : "none";
  };

  player.on("play", sync);
  player.on("pause", sync);
  player.on("ended", sync);
  sync();

  return () => {
    player.off("play", sync);
    player.off("pause", sync);
    player.off("ended", sync);
    overlay.remove();
  };
}
