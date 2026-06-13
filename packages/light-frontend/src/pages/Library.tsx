import useSWR from "swr";
import { fetchJSON } from "../api/client";
import VideoCard from "../components/VideoCard";
import type { Video } from "../types";

async function deleteVideo(id: string): Promise<void> {
  await fetchJSON(`/api/videos/${id}`, { method: "DELETE" });
}

export default function Library() {
  const { data, error, isLoading, mutate } = useSWR<{ videos: Video[] }>(
    "/api/videos",
    (path: string) => fetchJSON(path),
    { refreshInterval: 5000 },
  );

  const videos = data?.videos ?? [];

  const handleDelete = async (id: string) => {
    await deleteVideo(id);
    mutate();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-[#6b7280]">
        加载中...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-[#ef4444] gap-3">
        <p>连接失败</p>
        <p className="text-xs text-[#6b7280]">确保后端已启动 (uv run light-backend)</p>
        <button onClick={() => mutate()} className="text-xs text-[#3b82f6] hover:underline">
          重试
        </button>
      </div>
    );
  }

  if (videos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <div className="text-5xl text-[#6b7280] font-light">&#9654;</div>
        <p className="text-[#6b7280]">还没有视频</p>
        <p className="text-xs text-[#6b7280]">从顶部导航选择「下载」或「导入」</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-lg font-medium mb-6">视频库</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {videos.map((v) => (
          <VideoCard key={v.id} video={v} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  );
}
