import { useMemo } from "react";

interface StageEntry {
  key: string;
  label: string;
  progress: number;
  message: string;
}

/** 英文 stage key → 中文标签（仅映射，不定义顺序） */
const LABELS: Record<string, string> = {
  download: "下载",
  split: "切分",
  asr: "识别",
  correct: "矫正",
  punct: "标点",
  segment: "断句",
  context: "上下文",
  translate: "翻译",
  annotate: "注解",
  format: "格式化",
  merge: "合并",
  done: "完成",
  error: "失败",
};

export default function ProgressStepper({ events }: { events: Record<string, unknown>[] }) {
  // 从事件流动态构建阶段列表（Map 保持插入顺序 → 自动反映管线实际阶段顺序）
  const stages: StageEntry[] = useMemo(() => {
    const seen = new Map<string, StageEntry>();
    for (const ev of events) {
      const key = ev.stage as string;
      if (!key) continue;
      seen.set(key, {
        key,
        label: LABELS[key] || key,
        progress: (ev.progress as number) ?? 0,
        message: (ev.message as string) ?? "",
      });
    }
    return Array.from(seen.values());
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="text-sm text-[#6b7280]">正在连接…</div>
    );
  }

  const last = events[events.length - 1];
  const currentKey = last?.stage as string;
  const isDone = currentKey === "done";
  const isError = currentKey === "error";
  const lastMessage = last?.message as string;

  const currentIdx = stages.length - 1;

  return (
    <div className="space-y-3">
      {/* 阶段步骤指示器 */}
      <div className="overflow-x-auto -mx-1 px-1">
        <div className="flex items-center gap-0 min-w-max">
          {stages.map((s, i) => {
            const isLast = i === stages.length - 1;
            const isActive = i === currentIdx && !isDone && !isError;
            const isDoneStep = i < currentIdx || (isDone && isLast);
            const isErrorStep = isError && isLast;

            return (
              <div key={s.key} className="flex items-center flex-1 min-w-0">
                <div className="flex items-center gap-1.5 shrink-0">
                  {/* 圆圈 */}
                  <div
                    className={[
                      "w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-colors shrink-0",
                      isErrorStep
                        ? "bg-[#ef4444] text-white ring-2 ring-[#ef4444]/40"
                        : isActive
                          ? "bg-[#3b82f6] text-white ring-2 ring-[#3b82f6]/40"
                          : isDoneStep
                            ? "bg-[#22c55e] text-white"
                            : "bg-[#1f1f1f] text-[#6b7280]",
                    ].join(" ")}
                  >
                    {isDoneStep ? "\u2713" : isErrorStep ? "!" : i + 1}
                  </div>
                  {/* 标签 */}
                  <span
                    className={[
                      "text-xs hidden sm:inline whitespace-nowrap",
                      isErrorStep
                        ? "text-[#ef4444]"
                        : isActive
                          ? "text-[#3b82f6]"
                          : isDoneStep
                            ? "text-[#22c55e]"
                            : "text-[#6b7280]",
                    ].join(" ")}
                  >
                    {s.label}
                  </span>
                </div>
                {/* 连接线 */}
                {!isLast && (
                  <div
                    className={[
                      "flex-1 h-px mx-1.5 min-w-[4px]",
                      isDoneStep || isActive ? "bg-[#22c55e]" : "bg-[#1f1f1f]",
                    ].join(" ")}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* 当前消息 */}
      <div className={["text-sm", isError ? "text-[#ef4444]" : "text-[#888]"].join(" ")}>
        {lastMessage}
      </div>
    </div>
  );
}
