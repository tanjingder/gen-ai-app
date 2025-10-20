import { useState, useEffect } from "react";
import ChatInterface from "./components/ChatInterface";
import VideoUpload from "./components/VideoUpload";
import Header from "./components/Header";
import SessionList from "./components/SessionList";
import { VideoInfo, SessionData } from "./types";
import { grpcClient } from "./services/grpcClient";

function App() {
  const [currentVideo, setCurrentVideo] = useState<VideoInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [chatMessages, setChatMessages] = useState<any[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Initialize on mount - only once
    if (!isInitialized) {
      initializeApp();
    }
  }, [isInitialized]);

  const initializeApp = async () => {
    try {
      console.log("Initializing app...");
      
      // Mark as initialized immediately to prevent multiple calls
      setIsInitialized(true);
      
      // Simple connection check without creating sessions
      setIsConnected(true);
      
      // Check if there are existing sessions first
      const sessionList = await grpcClient.listSessions();
      
      if (sessionList.length > 0) {
        // Load the most recent session
        const mostRecent = sessionList[0]; // Sessions are sorted by updated_at descending
        console.log("Loading most recent session:", mostRecent.session_id);
        await loadSession(mostRecent.session_id);
      } else {
        // No existing sessions, create a new one
        console.log("No existing sessions, creating new one");
        await createNewSession();
      }
    } catch (error) {
      console.error("Failed to initialize app:", error);
      setIsConnected(false);
    }
  };

  const createNewSession = async () => {
    try {
      const session = await grpcClient.createSession("New Chat");
      console.log("Created new session:", session.session_id);
      
      // Update all state for the new session
      setCurrentSessionId(session.session_id);
      setSessionData(null);
      setChatMessages([]); // Clear messages for new session
      setCurrentVideo(null);
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  const loadSession = async (sessionId: string) => {
    try {
      console.log("🔄 Loading session:", sessionId);
      const data = await grpcClient.loadSession(sessionId);
      
      // Convert session messages to chat messages
      const messages = data.messages.map((msg: any, index: number) => ({
        id: `msg_${sessionId}_${index}`, // Include session ID to ensure uniqueness
        text: msg.content,
        sender: msg.role,
        timestamp: new Date(msg.timestamp).getTime(),
      }));
      
      console.log("✅ Loaded session with", messages.length, "messages:", messages);
      
      // Update all state - Force a new array reference
      setCurrentSessionId(sessionId);
      setSessionData(data);
      setChatMessages([...messages]); // Create new array to trigger re-render
      
      // If session has a video, reconstruct VideoInfo and set it
      if (data.session.video_id && data.session.video_filename) {
        console.log("📹 Session has video:", data.session.video_filename);
        
        // Reconstruct VideoInfo from session data
        if (data.session.video_metadata) {
          const metadata = data.session.video_metadata;
          const videoInfo: VideoInfo = {
            id: data.session.video_id,
            filename: data.session.video_filename,
            duration: metadata.duration_ms / 1000, // Convert ms to seconds
            resolution: `${metadata.width}x${metadata.height}`,
            fps: metadata.fps,
            size: metadata.file_size,
          };
          setCurrentVideo(videoInfo);
          console.log("✅ Restored video info:", videoInfo);
        } else {
          // Fallback: create minimal VideoInfo without full metadata
          const videoInfo: VideoInfo = {
            id: data.session.video_id,
            filename: data.session.video_filename,
            duration: 0,
            resolution: "Unknown",
            fps: 0,
            size: 0,
          };
          setCurrentVideo(videoInfo);
          console.log("⚠️ Restored video with minimal info (no metadata)");
        }
      } else {
        setCurrentVideo(null);
      }
    } catch (error) {
      console.error("❌ Failed to load session:", error);
    }
  };

  const handleSessionSelect = async (sessionId: string) => {
    console.log("👆 Session selected:", sessionId, "Current:", currentSessionId);
    if (sessionId === currentSessionId) {
      console.log("⏭️ Same session, skipping");
      return;
    }
    await loadSession(sessionId);
  };

  const handleNewChat = async () => {
    await createNewSession();
  };

  const handleDeleteSession = async (_sessionId: string) => {
    // Session deletion is handled by SessionList component
    // This is just a placeholder for App-level logic if needed
  };

  const handleVideoUploaded = (videoInfo: VideoInfo) => {
    setCurrentVideo(videoInfo);
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
      {/* Session Sidebar */}
      <SessionList
        currentSessionId={currentSessionId}
        onSessionSelect={handleSessionSelect}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header isConnected={isConnected} currentVideo={currentVideo} />
        
        <div className="flex-1 flex overflow-hidden min-w-0">
          {/* Video Upload Sidebar */}
          <div className="w-80 flex-shrink-0 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
            <VideoUpload onVideoUploaded={handleVideoUploaded} currentVideo={currentVideo} />
            
            {currentVideo && (
              <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                <h3 className="font-semibold mb-2">Current Video</h3>
                <p className="text-sm text-gray-300">{currentVideo.filename}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {currentVideo.duration}s • {currentVideo.resolution}
                </p>
              </div>
            )}
            
            {/* Session Info */}
            {currentSessionId && sessionData?.has_cache && (
              <div className="mt-4 p-4 bg-blue-900 bg-opacity-30 rounded-lg border border-blue-700">
                <h3 className="font-semibold mb-2 text-blue-300">📦 Cached Data</h3>
                <p className="text-xs text-blue-200">
                  This session has cached analysis results.
                  The AI will use them to avoid duplicate processing.
                </p>
                {sessionData.cache_summary && (
                  <div className="mt-2 text-xs text-blue-300">
                    <p className="font-semibold mb-1">Cached tools:</p>
                    <ul className="list-disc list-inside">
                      {Object.keys(sessionData.cache_summary).map((tool) => (
                        <li key={tool}>{tool}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Main Chat Area */}
          <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
            <ChatInterface 
              key={currentSessionId || 'no-session'} // Force re-render when session changes
              videoId={currentVideo?.id} 
              isConnected={isConnected}
              initialMessages={chatMessages}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
