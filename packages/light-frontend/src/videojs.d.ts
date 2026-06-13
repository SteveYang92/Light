declare module "video.js" {
  import videojs from "video.js";
  export default videojs;
}

interface VideoJsPlayer {
  src(source: { type: string; src: string }): void;
  play(): Promise<void> | void;
  pause(): void;
  paused(): boolean;
  currentTime(seconds?: number): number;
  muted(value?: boolean): boolean;
  volume(value?: number): number;
  on(event: string, fn: () => void): void;
  off(event: string, fn: () => void): void;
  one(event: string, fn: () => void): void;
  ready(fn: () => void): void;
  dispose(): void;
  el(): Element;
  textTracks(): TextTrackList;
  remoteTextTracks(): TextTrackList;
  addRemoteTextTrack(track: {
    kind: string;
    src: string;
    srclang: string;
    label: string;
    default: boolean;
  }, manualCleanup: boolean): HTMLTrackElement;
  removeRemoteTextTrack(track: TextTrack): void;
}
