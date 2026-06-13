interface Step {
  key: string;
  label: string;
  progress: number;
  done: boolean;
  active: boolean;
  message?: string;
  chunk?: number;
  totalChunks?: number;
}

const STAGES: { key: string; label: string }[] = [
  { key: "download", label: "下载" },
  { key: "split", label: "切分" },
  { key: "chunk", label: "处理" },
  { key: "done", label: "完成" },
];

export default function ProgressStepper({ events }: { events: Record<string, unknown>[] }) {
  const last = events[events.length - 1];
  const currentStage = last?.stage as string;

  // Determine which step is active
  const activeIdx = STAGES.findIndex((s) => s.key === currentStage);
  const chunkInfo =
    currentStage === "chunk" && last?.chunk != null
      ? `第 ${(last.chunk as number) + 1}/${last.totalChunks ?? "?"} 段`
      : null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-0">
        {STAGES.map((s, i) => {
          const isDone = i < activeIdx || (s.key === "done" && currentStage === "done");
          const isActive = i === activeIdx;
          const isFuture = i > activeIdx && !(s.key === "done" && currentStage === "done");
          const isLast = i === STAGES.length - 1;

          return (
            <div key={s.key} className="flex items-center flex-1">
              <div className="flex items-center gap-2">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-colors
                    ${isDone ? "bg-[#22c55e] text-white" : ""}
                    ${isActive ? "bg-[#3b82f6] text-white ring-2 ring-[#3b82f6]/40" : ""}
                    ${isFuture ? "bg-[#1f1f1f] text-[#6b7280]" : ""}
                  `}
                >
                  {isDone ? "\u2713" : i + 1}
                </div>
                <span
                  className={`text-xs hidden sm:inline
                    ${isDone ? "text-[#22c55e]" : ""}
                    ${isActive ? "text-[#3b82f6]" : ""}
                    ${isFuture ? "text-[#6b7280]" : ""}
                  `}
                >
                  {s.label}
                </span>
              </div>
              {!isLast && (
                <div
                  className={`flex-1 h-px mx-2
                    ${isDone ? "bg-[#22c55e]" : "bg-[#1f1f1f]"}
                  `}
                />
              )}
            </div>
          );
        })}
      </div>

      {events.length > 0 && (
        <div className="text-sm text-[#888]">
          {chunkInfo && <span className="text-[#3b82f6]">{chunkInfo} </span>}
          <span>{last?.message as string}</span>
        </div>
      )}
    </div>
  );
}
