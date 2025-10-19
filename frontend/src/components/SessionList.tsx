import { useState, useEffect } from "react";
import { Session } from "../types";
import { grpcClient } from "../services/grpcClient";

interface SessionListProps {
  currentSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteSession?: (sessionId: string) => void; // Make optional since not used internally
}

export default function SessionList({
  currentSessionId,
  onSessionSelect,
  onNewChat,
}: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  // Refresh sessions when currentSessionId changes (new session created)
  useEffect(() => {
    if (currentSessionId) {
      console.log("📋 SessionList: currentSessionId changed, refreshing list");
      loadSessions();
    }
  }, [currentSessionId]);

  const loadSessions = async () => {
    try {
      setIsLoading(true);
      const sessionList = await grpcClient.listSessions();
      setSessions(sessionList);
    } catch (error) {
      console.error("Failed to load sessions:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    try {
      const success = await grpcClient.deleteSession(sessionId);
      if (success) {
        const remainingSessions = sessions.filter((s) => s.session_id !== sessionId);
        setSessions(remainingSessions);
        setDeleteConfirm(null);
        
        // If deleted session was current, load another session or create new one
        if (sessionId === currentSessionId) {
          if (remainingSessions.length > 0) {
            // Load the most recent remaining session instead of creating new one
            console.log("📋 Deleted current session, loading most recent:", remainingSessions[0].session_id);
            onSessionSelect(remainingSessions[0].session_id);
          } else {
            // No sessions left, create a new one
            console.log("📋 Deleted last session, creating new one");
            onNewChat();
          }
        }
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  };

  const truncateTitle = (title: string, maxLength: number = 25) => {
    if (title.length <= maxLength) return title;
    return title.substring(0, maxLength) + "...";
  };

  return (
    <div className="session-sidebar">
      {/* Header */}
      <div className="session-header">
        <h2 className="session-title">Conversations</h2>
        <button
          className="btn-new-chat"
          onClick={onNewChat}
          title="Start new chat"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Chat
        </button>
      </div>

      {/* Session List */}
      <div className="session-list">
        {isLoading ? (
          <div className="session-loading">Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className="session-empty">
            <p>No conversations yet</p>
            <p className="session-empty-hint">Start a new chat to begin!</p>
          </div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.session_id}
              className={`session-item ${
                session.session_id === currentSessionId ? "active" : ""
              }`}
              onClick={() => {
                console.log("📋 SessionList: Clicked session:", session.session_id, session.title);
                onSessionSelect(session.session_id);
              }}
            >
              {deleteConfirm === session.session_id ? (
                <div className="session-delete-confirm">
                  <p className="delete-confirm-text">Delete this chat?</p>
                  <div className="delete-confirm-actions">
                    <button
                      className="btn-confirm-delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(session.session_id);
                      }}
                    >
                      Delete
                    </button>
                    <button
                      className="btn-cancel-delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteConfirm(null);
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="session-content">
                    <div className="session-info">
                      <h3 className="session-item-title">
                        {truncateTitle(session.title)}
                      </h3>
                      <p className="session-date">
                        {formatDate(session.updated_at)}
                      </p>
                    </div>
                    {session.video_filename && (
                      <div className="session-video-indicator" title={session.video_filename}>
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <polygon points="5 3 19 12 5 21 5 3" />
                        </svg>
                      </div>
                    )}
                  </div>
                  <button
                    className="btn-delete-session"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteConfirm(session.session_id);
                    }}
                    title="Delete conversation"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                </>
              )}
            </div>
          ))
        )}
      </div>

      {/* Refresh button at bottom */}
      <div className="session-footer">
        <button
          className="btn-refresh-sessions"
          onClick={loadSessions}
          title="Refresh sessions"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
          </svg>
          Refresh
        </button>
      </div>
    </div>
  );
}
