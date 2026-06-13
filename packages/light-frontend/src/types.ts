export interface Chunk {
  id: string;
  chunk_index: number;
  duration: number | null;
  video_ext: string;
  subtitles: string[];
}

export interface Video {
  id: string;
  title: string;
  source: string;
  source_url: string | null;
  duration: number | null;
  status: string;
  thumbnail: string | null;
  chunks: Chunk[];
  run: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface PipelineEvent {
  stage: string;
  progress: number;
  message: string;
  chunk: number | null;
  total_chunks: number | null;
}
