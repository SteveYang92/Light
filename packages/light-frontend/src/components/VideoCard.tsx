import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { Video } from "../types";

function formatDuration(sec: number | null): string {
  if (!sec) return "";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function statusBadge(status: string) {
  switch (status) {
    case "done":
      return <span className="text-xs text-[#22c55e]">&#10003; 就绪</span>;
    case "processing":
    case "pending":
      return <span className="text-xs text-[#3b82f6] animate-pulse">&#9881; 处理中</span>;
    case "error":
      return <span className="text-xs text-[#ef4444]">&#10007; 失败</span>;
    default:
      return null;
  }
}

export default function VideoCard({
  video,
  onDelete,
}: {
  video: Video;
  onDelete: (id: string) => void;
}) {
  const navigate = useNavigate();
  const [imgError, setImgError] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const src = video.thumbnail ?? null;

  const langs = new Set<string>();
  for (const c of video.chunks) {
    for (const k of c.subtitles) {
      const lang = k.split(".")[0];
      if (lang && lang !== "annotations" && lang !== "bilingual") langs.add(lang);
    }
  }

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirming) {
      onDelete(video.id);
    } else {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 3000);
    }
  };

  return (
    <div className="relative group">
      <button
        onClick={() => navigate(`/watch/${video.id}`)}
        className="w-full text-left rounded-xl overflow-hidden bg-[#141414] border border-[#1f1f1f]
          hover:border-[#333] transition-colors cursor-pointer"
      >
        <div className="aspect-video bg-[#1a1a1a] flex items-center justify-center overflow-hidden">
          {src && !imgError ? (
            <img
              src={src}
              alt={video.title}
              className="w-full h-full object-cover"
              onError={() => setImgError(true)}
            />
          ) : (
            <span className="text-[#6b7280] text-4xl font-light">&#9654;</span>
          )}
        </div>
        <div className="p-3 space-y-1">
          <h3 className="text-sm font-medium truncate">{video.title || "未命名"}</h3>
          <div className="flex items-center justify-between">
            <span className="text-xs text-[#6b7280]">{formatDuration(video.duration)}</span>
            {statusBadge(video.status)}
          </div>
          {langs.size > 0 && (
            <div className="flex gap-1 flex-wrap">
              {Array.from(langs).map((l) => (
                <span key={l} className="text-[10px] px-1.5 py-0.5 rounded bg-[#1f1f1f] text-[#888]">
                  {l}
                </span>
              ))}
            </div>
          )}
        </div>
      </button>

      {/* Delete button */}
      <button
        onClick={handleDelete}
        className={`absolute top-2 right-2 w-7 h-7 rounded-full flex items-center justify-center
          transition-all text-xs font-medium
          ${confirming
            ? "bg-[#ef4444] text-white scale-110"
            : "bg-black/60 text-[#888] opacity-0 group-hover:opacity-100 hover:text-[#ef4444]"
          }`}
        title={confirming ? "再次点击确认删除" : "删除"}
      >
        {confirming ? "?" : "\u2715"}
      </button>
    </div>
  );
}
