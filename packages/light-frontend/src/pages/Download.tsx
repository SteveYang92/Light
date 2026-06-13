import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { postJSON, listenSSE } from "../api/client";
import ProgressStepper from "../components/ProgressStepper";

interface FormData {
  url: string;
  target_lang: string;
  video_format: string;
  bilingual: boolean;
  diarize: boolean;
  annotate: boolean;
  llm_model: string;
}

type Stage = "form" | "running" | "done" | "error";

export default function Download() {
  const [form, setForm] = useState<FormData>({
    url: "",
    target_lang: "zh",
    video_format: "",
    bilingual: false,
    diarize: false,
    annotate: false,
    llm_model: "deepseek-chat",
  });

  const [stage, setStage] = useState<Stage>("form");
  const [videoId, setVideoId] = useState<string | null>(null);
  const [events, setEvents] = useState<Record<string, unknown>[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();
  const cleanupRef = useRef<(() => void) | null>(null);

  const handleSubmit = useCallback(async () => {
    if (!form.url.trim()) return;
    setStage("running");
    setEvents([]);
    setErrorMsg("");

    try {
      const result = await postJSON<{ id: string }>("/api/videos/url", form);
      setVideoId(result.id);

      cleanupRef.current = listenSSE(
        result.id,
        (ev) => setEvents((prev) => [...prev, ev]),
        () => setStage("done"),
        (err) => {
          setErrorMsg(err.message);
          setStage("error");
        },
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "提交失败");
      setStage("error");
    }
  }, [form]);

  useEffect(() => {
    return () => cleanupRef.current?.();
  }, []);

  const onDoneClick = () => {
    if (videoId) navigate(`/watch/${videoId}`);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <h1 className="text-lg font-medium">下载视频</h1>

      {stage === "form" && (
        <div className="space-y-5">
          <div>
            <input
              type="url"
              value={form.url}
              onChange={(e) => setForm({ ...form, url: e.target.value })}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full px-4 py-3 rounded-xl bg-[#141414] border border-[#1f1f1f] text-sm
                placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6] transition-colors"
            />
            <p className="mt-1.5 text-xs text-[#6b7280]">
              支持 YouTube、B站、X 等 yt-dlp 兼容的链接
            </p>
          </div>

          <div className="rounded-xl bg-[#141414] border border-[#1f1f1f] p-4 space-y-3">
            <h2 className="text-sm font-medium text-[#e5e5e5]">视频画质</h2>
            <div>
              <label className="block text-xs text-[#6b7280] mb-1">下载画质</label>
              <select
                value={form.video_format}
                onChange={(e) => setForm({ ...form, video_format: e.target.value })}
                className="w-full max-w-xs px-3 py-2 rounded-lg bg-[#0a0a0a] border border-[#1f1f1f] text-sm
                  focus:outline-none focus:border-[#3b82f6]"
              >
                <option value="">自动</option>
                <option value="1080p">1080p</option>
                <option value="720p">720p</option>
                <option value="best">最佳</option>
              </select>
            </div>
          </div>

          <div className="rounded-xl bg-[#141414] border border-[#1f1f1f] p-4 space-y-4">
            <h2 className="text-sm font-medium text-[#e5e5e5]">字幕设置</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-[#6b7280] mb-1">翻译目标</label>
                <select
                  value={form.target_lang}
                  onChange={(e) => setForm({ ...form, target_lang: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-[#0a0a0a] border border-[#1f1f1f] text-sm
                    focus:outline-none focus:border-[#3b82f6]"
                >
                  <option value="zh">中文</option>
                  <option value="en">英文</option>
                  <option value="ja">日文</option>
                  <option value="">源语言（无需翻译）</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-[#6b7280] mb-1">LLM 模型</label>
                <select
                  value={form.llm_model}
                  onChange={(e) => setForm({ ...form, llm_model: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-[#0a0a0a] border border-[#1f1f1f] text-sm
                    focus:outline-none focus:border-[#3b82f6]"
                >
                  <option value="deepseek-chat">deepseek-chat (快速)</option>
                  <option value="deepseek-v4-flash">deepseek-v4-flash (高质量)</option>
                </select>
              </div>
            </div>
            <div className="flex flex-wrap gap-4 pt-1">
              <label className="flex items-center gap-2 text-sm text-[#888]">
                <input
                  type="checkbox"
                  checked={form.bilingual}
                  onChange={(e) => setForm({ ...form, bilingual: e.target.checked })}
                  className="accent-[#3b82f6]"
                />
                双语
              </label>
              <label className="flex items-center gap-2 text-sm text-[#888]">
                <input
                  type="checkbox"
                  checked={form.diarize}
                  onChange={(e) => setForm({ ...form, diarize: e.target.checked })}
                  className="accent-[#3b82f6]"
                />
                说话人分离
              </label>
              <label className="flex items-center gap-2 text-sm text-[#888]">
                <input
                  type="checkbox"
                  checked={form.annotate}
                  onChange={(e) => setForm({ ...form, annotate: e.target.checked })}
                  className="accent-[#3b82f6]"
                />
                术语注解
              </label>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!form.url.trim()}
            className="w-full py-2.5 rounded-xl bg-[#3b82f6] text-white text-sm font-medium
              hover:bg-[#2563eb] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            开始处理
          </button>
        </div>
      )}

      {(stage === "running" || stage === "done" || stage === "error") && (
        <div className="p-5 rounded-xl bg-[#141414] border border-[#1f1f1f] space-y-5">
          <ProgressStepper events={events} />

          {stage === "done" && (
            <button
              onClick={onDoneClick}
              className="w-full py-2.5 rounded-xl bg-[#22c55e] text-white text-sm font-medium
                hover:bg-[#16a34a] transition-colors"
            >
              去播放 &rarr;
            </button>
          )}

          {stage === "error" && (
            <div className="text-sm text-[#ef4444]">{errorMsg}</div>
          )}
        </div>
      )}
    </div>
  );
}
