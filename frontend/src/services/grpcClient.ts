/**
 * gRPC Client for Video Analysis Service
 * 
 * This is a placeholder implementation. In production, you would:
 * 1. Generate TypeScript types from protobuf using ts-proto
 * 2. Use @grpc/grpc-js for Node.js or grpc-web for browsers
 * 3. For Tauri, use Tauri's invoke API to call Rust backend which handles gRPC
 */

import { invoke } from "@tauri-apps/api/tauri";
import { ChatMessage, Message, VideoInfo, AnalysisStatus } from "../types";

class GrpcClient {
  private sessionId: string;

  constructor() {
    this.sessionId = this.generateSessionId();
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Send a chat message and receive streaming response
   */
  async *chat(message: string, videoId?: string): AsyncGenerator<Message> {
    try {
      const request: ChatMessage = {
        message,
        sessionId: this.sessionId,
        videoId,
      };

      // Call Tauri backend to invoke REST API
      try {
        const response: any = await invoke("grpc_chat", { 
          message: request.message,
          sessionId: request.sessionId,
          videoId: request.videoId 
        });

        // Response is an array of messages
        if (Array.isArray(response)) {
          for (const msg of response) {
            yield {
              id: `msg_${Date.now()}_${Math.random()}`,
              text: msg.text,
              sender: msg.sender || "assistant",
              timestamp: msg.timestamp || Date.now(),
            };
            // Small delay between messages for better UX
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        } else {
          // Single message response (shouldn't happen but handle it)
          yield {
            id: `msg_${Date.now()}`,
            text: response.text || JSON.stringify(response),
            sender: response.sender || "assistant",
            timestamp: response.timestamp || Date.now(),
          };
        }
      } catch (invokeError: any) {
        // Handle error from backend
        const errorMsg = invokeError.toString();
        console.error("Backend error:", errorMsg);
        
        let helpText = "";
        if (errorMsg.includes("connect") || errorMsg.includes("ECONNREFUSED")) {
          helpText = "\n\n💡 Backend not reachable. Please:\n1. Start the backend: start-backend-api.bat\n2. Ensure it's running on http://localhost:8000";
        } else if (errorMsg.includes("Ollama")) {
          helpText = "\n\n💡 Ollama issue. Please:\n1. Start Ollama: ollama serve\n2. Pull the model: ollama pull llama3.2";
        }
        
        yield {
          id: `msg_${Date.now()}`,
          text: `⚠️ Error: ${errorMsg}${helpText}`,
          sender: "system",
          timestamp: Date.now(),
        };
      }
    } catch (error) {
      console.error("Chat error:", error);
      yield {
        id: `msg_${Date.now()}`,
        text: `❌ Unexpected error: ${error}`,
        sender: "system",
        timestamp: Date.now(),
      };
    }
  }

  /**
   * Upload video file
   */
  async uploadVideo(file: File): Promise<VideoInfo> {
    try {
      console.log(`Uploading video: ${file.name} (${file.size} bytes)`);
      
      // Read file as array buffer
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Upload via Tauri invoke to backend
      const result = await invoke<{
        video_id: string;
        success: boolean;
        message: string;
        metadata?: {
          filename: string;
          duration_ms: number;
          width: number;
          height: number;
          fps: number;
          file_size: number;
        };
      }>("upload_video", {
        filename: file.name,
        data: Array.from(bytes),
        sessionId: this.sessionId, // Pass current session ID
      });

      if (!result.success) {
        throw new Error(result.message || "Upload failed");
      }

      console.log(`Upload successful: ${result.video_id}`);

      // Return video info
      const metadata = result.metadata || {
        filename: file.name,
        duration_ms: 0,
        width: 1920,
        height: 1080,
        fps: 30,
        file_size: file.size,
      };

      const videoInfo: VideoInfo = {
        id: result.video_id,
        filename: metadata.filename,
        duration: metadata.duration_ms / 1000, // Convert ms to seconds
        resolution: `${metadata.width}x${metadata.height}`,
        fps: metadata.fps,
        size: metadata.file_size,
      };

      return videoInfo;
    } catch (error) {
      console.error("Upload error:", error);
      throw new Error(
        `Failed to upload video: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Query video content
   */
  async *queryVideo(
    videoId: string,
    query: string
  ): AsyncGenerator<Message> {
    // Similar to chat but specifically for video queries
    yield* this.chat(query, videoId);
  }

  /**
   * Generate report
   */
  async generateReport(
    videoId: string,
    format: "pdf" | "pptx",
    sections?: string[]
  ): Promise<Uint8Array> {
    try {
      console.log(`Generating ${format} report for video: ${videoId}`);
      
      const result = await invoke<{
        success: boolean;
        message: string;
        report_data: number[];
        filename: string;
      }>("generate_report", {
        videoId,
        format,
        sections: sections || [],
      });
      
      if (!result.success) {
        throw new Error(result.message || "Report generation failed");
      }
      
      console.log(`Report generated: ${result.filename}`);
      
      // Convert number array back to Uint8Array
      return new Uint8Array(result.report_data);
    } catch (error) {
      console.error("Report generation error:", error);
      throw new Error(
        `Failed to generate report: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Get analysis status
   */
  async getAnalysisStatus(videoId: string): Promise<AnalysisStatus> {
    try {
      console.log(`Getting analysis status for video: ${videoId}`);
      
      const result = await invoke<{
        video_id: string;
        stage: string;
        progress: number;
        message: string;
      }>("get_analysis_status", { videoId });
      
      console.log(`Status: ${result.stage} (${result.progress * 100}%)`);
      
      return {
        videoId: result.video_id,
        stage: result.stage.toLowerCase(),
        progress: result.progress,
        message: result.message,
      };
    } catch (error) {
      console.error("Status check error:", error);
      throw new Error(
        `Failed to get analysis status: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Create a new chat session
   */
  async createSession(title: string = "New Chat"): Promise<any> {
    try {
      const result = await invoke<any>("create_session", { title });
      this.sessionId = result.session_id;
      return result;
    } catch (error) {
      console.error("Create session error:", error);
      throw error;
    }
  }

  /**
   * List all sessions
   */
  async listSessions(): Promise<any[]> {
    try {
      const result = await invoke<any>("list_sessions", {});
      return result.sessions || [];
    } catch (error) {
      console.error("List sessions error:", error);
      throw error;
    }
  }

  /**
   * Load a specific session
   */
  async loadSession(sessionId: string): Promise<any> {
    try {
      console.log("🔍 grpcClient: Loading session:", sessionId);
      const result = await invoke<any>("load_session", { sessionId });
      console.log("📦 grpcClient: Loaded session data:", result);
      console.log("📦 Messages count:", result.messages?.length || 0);
      this.sessionId = sessionId;
      return result;
    } catch (error) {
      console.error("❌ Load session error:", error);
      throw error;
    }
  }

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      const result = await invoke<any>("delete_session", { sessionId });
      return result.success || false;
    } catch (error) {
      console.error("Delete session error:", error);
      throw error;
    }
  }

  /**
   * Update a session (e.g., rename)
   */
  async updateSession(sessionId: string, title: string): Promise<boolean> {
    try {
      const result = await invoke<any>("update_session", { sessionId, title });
      return result.success || false;
    } catch (error) {
      console.error("Update session error:", error);
      throw error;
    }
  }

  /**
   * Get current session ID
   */
  getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Set session ID
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }
}

export const grpcClient = new GrpcClient();
