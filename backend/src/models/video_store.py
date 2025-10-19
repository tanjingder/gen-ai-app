"""
Video storage and management
"""
from typing import Dict, Any, Optional
from pathlib import Path


class VideoStore:
    """In-memory store for video information"""
    
    def __init__(self):
        self._videos: Dict[str, Dict[str, Any]] = {}
    
    def add_video(self, video_id: str, video_info: Dict[str, Any]):
        """Add video to store"""
        self._videos[video_id] = video_info
    
    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video info"""
        return self._videos.get(video_id)
    
    def update_video(self, video_id: str, updates: Dict[str, Any]):
        """Update video info"""
        if video_id in self._videos:
            self._videos[video_id].update(updates)
    
    def delete_video(self, video_id: str):
        """Remove video from store"""
        if video_id in self._videos:
            del self._videos[video_id]
    
    def list_videos(self) -> list[Dict[str, Any]]:
        """List all videos"""
        return list(self._videos.values())
