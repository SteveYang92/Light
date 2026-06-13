import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { postJSON } from "../api/client";

interface ImportData {
  output_dir: string;
  video_path: string;
  title: string;
}

export default function Import() {
  const [importData, setImportData] = useState<ImportData>({
    output_dir: "",
    video_path: "",
    title: "",
  });
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  const handleSubmit = useCallback(async () => {
    if (!importData.output_dir.trim()) return;
    setErrorMsg("");

    try {
      const result = await postJSON<{ id: string; status: string; chunks: number }>(
        "/api/videos/import",
        importData,
      );
      navigate(`/watch/${result.id}`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "导入失败");
    }
  }, [importData, navigate]);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <h1 className="text-lg font-medium">导入字幕</h1>

      <div className="space-y-5">
        <p className="text-xs text-[#6b7280]">
          填入管线 output 目录路径，自动发现目录内的视频和字幕文件。
        </p>

        <div>
          <label className="block text-xs text-[#6b7280] mb-1">output 目录路径</label>
          <input
            type="text"
            value={importData.output_dir}
            onChange={(e) => setImportData({ ...importData, output_dir: e.target.value })}
            placeholder="/path/to/output/directory"
            className="w-full px-4 py-3 rounded-xl bg-[#141414] border border-[#1f1f1f] text-sm
              placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6]"
          />
        </div>

        <details className="text-xs text-[#6b7280]">
          <summary className="cursor-pointer hover:text-[#e5e5e5]">高级选项</summary>
          <div className="mt-3 space-y-3">
            <div>
              <label className="block text-xs text-[#6b7280] mb-1">视频文件路径（目录内无 mp4 时需指定）</label>
              <input
                type="text"
                value={importData.video_path}
                onChange={(e) => setImportData({ ...importData, video_path: e.target.value })}
                placeholder="可选，自动从 output 目录发现"
                className="w-full px-4 py-3 rounded-xl bg-[#141414] border border-[#1f1f1f] text-sm
                  placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6]"
              />
            </div>
            <div>
              <label className="block text-xs text-[#6b7280] mb-1">标题</label>
              <input
                type="text"
                value={importData.title}
                onChange={(e) => setImportData({ ...importData, title: e.target.value })}
                placeholder="自动从目录/文件名推断"
                className="w-full px-4 py-3 rounded-xl bg-[#141414] border border-[#1f1f1f] text-sm
                  placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6]"
              />
            </div>
          </div>
        </details>

        {errorMsg && (
          <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 rounded-lg px-4 py-3">
            {errorMsg}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={!importData.output_dir.trim()}
          className="w-full py-2.5 rounded-xl bg-[#3b82f6] text-white text-sm font-medium
            hover:bg-[#2563eb] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          导入
        </button>
      </div>
    </div>
  );
}
