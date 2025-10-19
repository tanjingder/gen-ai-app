"""
Session Manager for handling chat sessions with persistent storage
Each session has its own folder containing:
- session.json: Session metadata
- messages.json: Chat history
- uploads/: Uploaded videos
- temp/: Extracted frames
- reports/: Generated PDFs/PPTs
- analysis.json: Cached analysis results
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
import shutil
import os
import stat
import time

from loguru import logger

from ..utils.config import settings


class SessionManager:
    """Manages chat sessions with persistent storage"""
    
    def __init__(self):
        self.sessions_dir = Path("./data/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session manager initialized: {self.sessions_dir}")
    
    def create_session(self, title: str = "New Chat") -> Dict[str, Any]:
        """
        Create a new chat session
        
        Returns:
            Session metadata
        """
        session_id = str(uuid.uuid4())
        session_dir = self.sessions_dir / session_id
        
        # Create session folder structure
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "uploads").mkdir(exist_ok=True)
        (session_dir / "temp").mkdir(exist_ok=True)
        (session_dir / "reports").mkdir(exist_ok=True)
        
        # Create session metadata
        session = {
            "session_id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "video_id": None,
            "video_filename": None
        }
        
        # Save session metadata
        self._save_session_metadata(session_id, session)
        
        # Initialize empty chat history
        self._save_messages(session_id, [])
        
        # Initialize empty analysis cache
        self._save_analysis_cache(session_id, {})
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions
        
        Returns:
            List of session metadata, sorted by updated_at (newest first)
        """
        sessions = []
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                try:
                    session = self._load_session_metadata(session_dir.name)
                    if session:
                        sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to load session {session_dir.name}: {e}")
        
        # Sort by updated_at (newest first)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return sessions
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session with all its data
        
        Returns:
            Complete session data including messages and cache
        """
        session = self._load_session_metadata(session_id)
        if not session:
            return None
        
        # Load chat history
        messages = self._load_messages(session_id)
        
        # Load analysis cache
        cache = self._load_analysis_cache(session_id)
        
        return {
            **session,
            "messages": messages,
            "cache": cache
        }
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data
        
        Returns:
            True if successful
        """
        session_dir = self.sessions_dir / session_id
        
        if not session_dir.exists():
            logger.warning(f"Session not found: {session_id}")
            return False
        
        try:
            # On Windows, files might be locked - retry with error handler
            def handle_remove_error(func, path, exc_info):
                """Error handler for rmtree on Windows"""
                # If access denied, try to change permissions
                if not os.access(path, os.W_OK):
                    try:
                        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
                        time.sleep(0.1)  # Brief pause
                        func(path)
                    except Exception as e:
                        logger.error(f"Could not remove {path}: {e}")
                else:
                    logger.error(f"Could not remove {path}: {exc_info}")
            
            # Delete entire session folder with error handler
            shutil.rmtree(session_dir, onerror=handle_remove_error)
            
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to delete session {session_id}: {e}")
            return False
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to session chat history"""
        messages = self._load_messages(session_id)
        
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_messages(session_id, messages)
        self._update_session_timestamp(session_id)
        
        return True
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        session = self._load_session_metadata(session_id)
        if not session:
            return False
        
        session["title"] = title
        session["updated_at"] = datetime.now().isoformat()
        
        self._save_session_metadata(session_id, session)
        return True
    
    def attach_video(self, session_id: str, video_id: str, filename: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Attach video to session with optional metadata"""
        session = self._load_session_metadata(session_id)
        if not session:
            return False
        
        session["video_id"] = video_id
        session["video_filename"] = filename
        session["updated_at"] = datetime.now().isoformat()
        
        # Store video metadata if provided
        if metadata:
            session["video_metadata"] = metadata
        
        self._save_session_metadata(session_id, session)
        return True
    
    def cache_analysis_result(self, session_id: str, tool_name: str, result: Any) -> bool:
        """
        Cache analysis result to avoid reprocessing
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool (e.g., "transcribe_audio", "extract_frames")
            result: Analysis result to cache
        """
        cache = self._load_analysis_cache(session_id)
        
        cache[tool_name] = {
            "result": result,
            "cached_at": datetime.now().isoformat()
        }
        
        self._save_analysis_cache(session_id, cache)
        logger.info(f"Cached {tool_name} result for session {session_id}")
        
        return True
    
    def get_cached_result(self, session_id: str, tool_name: str) -> Optional[Any]:
        """
        Get cached analysis result
        
        Returns:
            Cached result or None if not found
        """
        cache = self._load_analysis_cache(session_id)
        
        if tool_name in cache:
            logger.info(f"Cache hit for {tool_name} in session {session_id}")
            return cache[tool_name]["result"]
        
        return None
    
    def get_session_paths(self, session_id: str) -> Dict[str, Path]:
        """Get all relevant paths for a session"""
        session_dir = self.sessions_dir / session_id
        
        return {
            "session_dir": session_dir,
            "uploads_dir": session_dir / "uploads",
            "temp_dir": session_dir / "temp",
            "reports_dir": session_dir / "reports"
        }
    
    # Private helper methods
    
    def _get_session_file(self, session_id: str, filename: str) -> Path:
        """Get path to a session file"""
        return self.sessions_dir / session_id / filename
    
    def _save_session_metadata(self, session_id: str, session: Dict[str, Any]):
        """Save session metadata to disk"""
        session_file = self._get_session_file(session_id, "session.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
    
    def _load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata from disk"""
        session_file = self._get_session_file(session_id, "session.json")
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load session metadata: {e}")
            return None
    
    def _save_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """Save chat messages to disk"""
        messages_file = self._get_session_file(session_id, "messages.json")
        with open(messages_file, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    
    def _load_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load chat messages from disk"""
        messages_file = self._get_session_file(session_id, "messages.json")
        
        if not messages_file.exists():
            return []
        
        try:
            with open(messages_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load messages: {e}")
            return []
    
    def _save_analysis_cache(self, session_id: str, cache: Dict[str, Any]):
        """Save analysis cache to disk"""
        cache_file = self._get_session_file(session_id, "analysis.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    
    def _load_analysis_cache(self, session_id: str) -> Dict[str, Any]:
        """Load analysis cache from disk"""
        cache_file = self._get_session_file(session_id, "analysis.json")
        
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load analysis cache: {e}")
            return {}
    
    def _update_session_timestamp(self, session_id: str):
        """Update session's updated_at timestamp"""
        session = self._load_session_metadata(session_id)
        if session:
            session["updated_at"] = datetime.now().isoformat()
            self._save_session_metadata(session_id, session)


# Global session manager instance
session_manager = SessionManager()
