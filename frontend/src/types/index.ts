export interface VideoInfo {
  id: string;
  filename: string;
  duration: number;
  resolution: string;
  fps: number;
  size: number;
}

export interface Message {
  id: string;
  text: string;
  sender: "user" | "assistant" | "system";
  timestamp: number;
  agentInfo?: {
    agentName: string;
    action: string;
    metadata?: Record<string, string>;
  };
  fileAttachment?: {
    filename: string;
    filePath: string;
    fileType: string;
    fileSize: number;
  };
}

export interface ChatMessage {
  message: string;
  sessionId: string;
  videoId?: string;
}

export interface AnalysisStatus {
  videoId: string;
  stage: string;
  progress: number;
  message: string;
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  video_id?: string;
  video_filename?: string;
  video_metadata?: {
    filename: string;
    duration_ms: number;
    width: number;
    height: number;
    fps: number;
    file_size: number;
  };
}

export interface SessionData {
  session: Session;
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
  }>;
  has_cache: boolean;
  cache_summary: Record<string, string>;
}
