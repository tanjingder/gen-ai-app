"""
gRPC Service Implementation
"""
import asyncio
import uuid
import grpc
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional
import time

from loguru import logger
import ollama

from ..utils.config import settings
from ..mcp.orchestrator_v3 import orchestrator_v3 as orchestrator
from ..models.video_store import VideoStore


class VideoAnalysisServicer:
    """Implementation of VideoAnalysisService"""
    
    def __init__(self):
        self.video_store = VideoStore()
        self.active_sessions: Dict[str, Any] = {}
    
    def _get_or_reconstruct_video(self, video_id: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get video info from store, or reconstruct from session if backend restarted
        
        Args:
            video_id: Video identifier
            session_id: Session identifier (optional, used for reconstruction)
            
        Returns:
            Video info dict or None if not found
        """
        from ..utils.session_manager import session_manager
        
        # Try to get from in-memory store first
        video_info = self.video_store.get_video(video_id)
        
        if video_info:
            return video_info
        
        # If not in store and we have session_id, try to reconstruct
        if session_id:
            logger.info(f"Video {video_id} not in store, attempting to reconstruct from session {session_id}")
            session_data = session_manager.load_session(session_id)
            
            if session_data and session_data.get("video_id") == video_id:
                # Get session paths
                session_paths = session_manager.get_session_paths(session_id)
                uploads_dir = session_paths["uploads_dir"]
                
                # Find the video file (saved as {video_id}_{filename})
                video_filename = session_data.get("video_filename")
                if video_filename:
                    video_path = uploads_dir / f"{video_id}_{video_filename}"
                    
                    if video_path.exists():
                        logger.info(f"✅ Found and reconstructed video: {video_path}")
                        
                        # Reconstruct video info
                        video_info = {
                            "video_id": video_id,
                            "filename": video_filename,
                            "path": str(video_path),
                            "metadata": session_data.get("video_metadata", {})
                        }
                        
                        # Add back to store for future use
                        self.video_store.add_video(video_id, video_info)
                        return video_info
                    else:
                        logger.error(f"❌ Video file not found: {video_path}")
                else:
                    logger.error(f"❌ No video_filename in session data")
            else:
                logger.error(f"❌ Session has different or no video_id")
        
        return None
    
    async def Chat(self, request, context):
        """Handle chat messages"""
        # Import proto modules
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        message = request.message
        session_id = request.session_id or str(uuid.uuid4())
        video_id = request.video_id if request.HasField('video_id') else None
        
        logger.info(f"Chat request: session={session_id}, video={video_id}, message={message[:50]}...")
        
        # Load or create session
        session_data = session_manager.load_session(session_id)
        if not session_data:
            # Create new session
            session_data = session_manager.create_session("New Chat")
            session_id = session_data["session_id"]
        
        # Initialize session if needed (do this BEFORE adding the new message)
        if session_id not in self.active_sessions:
            # Load existing message history from persistent storage
            existing_messages = session_data.get("messages", [])
            history = [{"role": msg["role"], "content": msg["content"]} for msg in existing_messages]
            logger.info(f"Loaded {len(history)} existing messages for session {session_id}")
            
            self.active_sessions[session_id] = {
                "history": history,
                "video_id": video_id
            }
        
        # Add user message to both persistent storage and in-memory history
        session_manager.add_message(session_id, "user", message)
        
        session = self.active_sessions[session_id]
        session["history"].append({"role": "user", "content": message})
        
        # If video is attached and query is about video, use orchestrator
        if video_id:
            video_info = self._get_or_reconstruct_video(video_id, session_id)
            
            if video_info:
                # Stream initial status
                yield video_analysis_pb2.ChatResponse(
                    message=f"🎯 Analyzing your request...",
                    sender="system",
                    timestamp=int(time.time() * 1000)
                )
                
                # Use V3 orchestrator - single call handles everything
                try:
                    response_text = await orchestrator.process_query(
                        user_query=message,
                        video_id=video_id,
                        video_path=video_info["path"],
                        session_id=session_id
                    )
                except Exception as e:
                    logger.exception(f"Error in orchestrator V3: {e}")
                    response_text = f"❌ Error processing your request: {str(e)}"
                session["history"].append({"role": "assistant", "content": response_text})
                
                # Add assistant message to session
                session_manager.add_message(session_id, "assistant", response_text)
                
                yield video_analysis_pb2.ChatResponse(
                    message=response_text,
                    sender="assistant",
                    timestamp=int(time.time() * 1000)
                )
            else:
                yield video_analysis_pb2.ChatResponse(
                    message=f"Video {video_id} not found. Please upload a video first.",
                    sender="system",
                    timestamp=int(time.time() * 1000)
                )
        else:
            # Regular chat without video - direct Ollama
            try:
                logger.info(f"Sending message to Ollama: {message[:100]}")
                
                # Use async Ollama client
                from ollama import AsyncClient
                client = AsyncClient(host=settings.OLLAMA_HOST)
                
                response = await client.chat(
                    model=settings.OLLAMA_MODEL,
                    messages=session["history"]
                )
                
                response_text = response["message"]["content"]
                session["history"].append({"role": "assistant", "content": response_text})
                
                # Save assistant message to persistent storage
                session_manager.add_message(session_id, "assistant", response_text)
                
                logger.info(f"Ollama response: {response_text[:100]}")
                
                yield video_analysis_pb2.ChatResponse(
                    message=response_text,
                    sender="assistant",
                    timestamp=int(time.time() * 1000)
                )
            except Exception as e:
                logger.exception(f"Chat error: {e}")
                error_msg = f"⚠️ Error: {str(e)}\n\n"
                
                # Check if Ollama is running
                if "connection" in str(e).lower() or "refused" in str(e).lower():
                    error_msg += "💡 Is Ollama running? Start it with: ollama serve"
                elif "model" in str(e).lower():
                    error_msg += f"💡 Model '{settings.OLLAMA_MODEL}' not found. Pull it with: ollama pull {settings.OLLAMA_MODEL}"
                
                yield video_analysis_pb2.ChatResponse(
                    message=error_msg,
                    sender="system",
                    timestamp=int(time.time() * 1000)
                )
    
    async def UploadVideo(self, request_iterator, context):
        """Handle video upload"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        import cv2
        
        video_id = str(uuid.uuid4())
        filename = ""
        chunks = []
        session_id = None
        
        try:
            async for chunk in request_iterator:
                if chunk.filename:
                    filename = chunk.filename
                if chunk.HasField('session_id') and chunk.session_id:
                    session_id = chunk.session_id
                chunks.append(chunk.data)
                logger.debug(f"Received chunk {chunk.chunk_index} for {filename}")
            
            # Use session folder if session_id provided, otherwise use default
            if session_id:
                session_paths = session_manager.get_session_paths(session_id)
                upload_dir = Path(session_paths["uploads_dir"])
            else:
                upload_dir = settings.UPLOAD_DIR
            
            # Ensure upload directory exists
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save video
            video_path = upload_dir / f"{video_id}_{filename}"
            logger.info(f"Saving video to: {video_path}")
            
            with open(video_path, 'wb') as f:
                f.write(b''.join(chunks))
            
            logger.info(f"Video file size: {video_path.stat().st_size} bytes")
            
            logger.info(f"Video uploaded: {video_path} ({len(chunks)} chunks)")
            
            # Extract metadata
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0
            cap.release()
            
            file_size = video_path.stat().st_size
            
            # Prepare video metadata
            video_metadata = {
                "duration_ms": duration_ms,
                "width": width,
                "height": height,
                "fps": fps,
                "file_size": file_size
            }
            
            # Attach video to session if session_id provided
            if session_id:
                session_manager.attach_video(session_id, video_id, filename, video_metadata)
            
            # Store video info
            video_info = {
                "video_id": video_id,
                "filename": filename,
                "path": str(video_path),
                "metadata": video_metadata
            }
            self.video_store.add_video(video_id, video_info)
            
            return video_analysis_pb2.VideoUploadResponse(
                video_id=video_id,
                success=True,
                message=f"Video uploaded successfully: {filename}",
                metadata=video_analysis_pb2.VideoMetadata(
                    filename=filename,
                    duration_ms=duration_ms,
                    width=width,
                    height=height,
                    fps=fps,
                    file_size=file_size
                )
            )
            
        except Exception as e:
            logger.exception(f"Upload error: {e}")
            return video_analysis_pb2.VideoUploadResponse(
                video_id="",
                success=False,
                message=f"Upload failed: {str(e)}"
            )
    
    async def QueryVideo(self, request, context):
        """Handle video queries"""
        from . import video_analysis_pb2
        
        # Similar to Chat but specifically for video queries
        async for response in self.Chat(
            video_analysis_pb2.ChatRequest(
                message=request.query,
                session_id=request.session_id,
                video_id=request.video_id
            ),
            context
        ):
            yield video_analysis_pb2.QueryResponse(
                response_chunk=response.message,
                is_final=True,
                agent_results=[]
            )
    
    async def GenerateReport(self, request, context):
        """Generate report for video"""
        from . import video_analysis_pb2
        
        video_id = request.video_id
        report_format = "pdf" if request.format == video_analysis_pb2.ReportFormat.PDF else "pptx"
        session_id = request.session_id if hasattr(request, 'session_id') else None
        
        logger.info(f"Generating {report_format} report for video {video_id}")
        
        try:
            video_info = self._get_or_reconstruct_video(video_id, session_id)
            
            if not video_info:
                return video_analysis_pb2.ReportResponse(
                    success=False,
                    message=f"Video {video_id} not found"
                )
            
            # Create report content
            content = {
                "video_id": video_id,
                "metadata": video_info.get("metadata", {}),
                "sections": request.sections or ["summary", "transcription", "key_frames"]
            }
            
            output_path = settings.REPORTS_DIR / f"{video_id}_report.{report_format}"
            
            # Call report agent via orchestrator client
            if report_format == "pdf":
                result = await orchestrator.client.create_pdf_report(content, str(output_path))
            else:
                result = await orchestrator.client.create_ppt_report(content, str(output_path))
            
            # Read report file
            if output_path.exists():
                with open(output_path, 'rb') as f:
                    report_data = f.read()
                
                return video_analysis_pb2.ReportResponse(
                    success=True,
                    message=f"Report generated successfully",
                    report_data=report_data,
                    filename=output_path.name
                )
            else:
                return video_analysis_pb2.ReportResponse(
                    success=False,
                    message=f"Report generation failed: {result.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.exception(f"Report generation error: {e}")
            return video_analysis_pb2.ReportResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    async def GetAnalysisStatus(self, request, context):
        """Get analysis status"""
        from . import video_analysis_pb2
        
        video_id = request.video_id
        session_id = request.session_id if hasattr(request, 'session_id') else None
        
        video_info = self._get_or_reconstruct_video(video_id, session_id)
        
        if not video_info:
            return video_analysis_pb2.AnalysisStatus(
                video_id=video_id,
                stage=video_analysis_pb2.ProcessingStage.FAILED,
                progress=0.0,
                message="Video not found"
            )
        
        # Get current status from video store
        status = video_info.get("status", "uploaded")
        progress = video_info.get("progress", 0.0)
        
        stage_map = {
            "uploaded": video_analysis_pb2.ProcessingStage.UPLOADED,
            "extracting_audio": video_analysis_pb2.ProcessingStage.EXTRACTING_AUDIO,
            "transcribing": video_analysis_pb2.ProcessingStage.TRANSCRIBING,
            "analyzing_frames": video_analysis_pb2.ProcessingStage.ANALYZING_FRAMES,
            "completed": video_analysis_pb2.ProcessingStage.COMPLETED,
            "failed": video_analysis_pb2.ProcessingStage.FAILED,
        }
        
        return video_analysis_pb2.AnalysisStatus(
            video_id=video_id,
            stage=stage_map.get(status, video_analysis_pb2.ProcessingStage.UPLOADED),
            progress=progress,
            message=f"Video is {status}"
        )
    
    async def CreateSession(self, request, context):
        """Create a new chat session"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        try:
            title = request.title or "New Chat"
            session_data = session_manager.create_session(title)
            
            return video_analysis_pb2.SessionResponse(
                session_id=session_data["session_id"],
                title=session_data["title"],
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"],
                video_id=session_data.get("video_id", ""),
                video_filename=session_data.get("video_filename", "")
            )
        except Exception as e:
            logger.exception(f"CreateSession error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return video_analysis_pb2.SessionResponse()
    
    async def ListSessions(self, request, context):
        """List all chat sessions"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        try:
            sessions = session_manager.list_sessions()
            
            session_list = []
            for session in sessions:
                session_list.append(video_analysis_pb2.SessionResponse(
                    session_id=session["session_id"],
                    title=session["title"],
                    created_at=session["created_at"],
                    updated_at=session["updated_at"],
                    video_id=session.get("video_id", ""),
                    video_filename=session.get("video_filename", "")
                ))
            
            return video_analysis_pb2.SessionListResponse(sessions=session_list)
        except Exception as e:
            logger.exception(f"ListSessions error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return video_analysis_pb2.SessionListResponse()
    
    async def LoadSession(self, request, context):
        """Load a chat session with history and cache"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        try:
            session_id = request.session_id
            session_data = session_manager.load_session(session_id)
            
            if not session_data:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Session {session_id} not found")
                return video_analysis_pb2.SessionDataResponse()
            
            # Convert session data (SessionManager returns flat structure with spread)
            video_metadata = session_data.get("video_metadata")
            metadata_pb = None
            if video_metadata:
                metadata_pb = video_analysis_pb2.VideoMetadata(
                    filename=session_data.get("video_filename", ""),
                    duration_ms=video_metadata.get("duration_ms", 0),
                    width=video_metadata.get("width", 0),
                    height=video_metadata.get("height", 0),
                    fps=video_metadata.get("fps", 0.0),
                    file_size=video_metadata.get("file_size", 0)
                )
            
            session_info = video_analysis_pb2.SessionResponse(
                session_id=session_data["session_id"],
                title=session_data["title"],
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"],
                video_id=session_data.get("video_id", ""),
                video_filename=session_data.get("video_filename", "")
            )
            
            # Add video_metadata if available
            if metadata_pb:
                session_info.video_metadata.CopyFrom(metadata_pb)
            
            # Convert messages
            messages = []
            for msg in session_data.get("messages", []):
                messages.append(video_analysis_pb2.ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"]
                ))
            
            # Cache summary
            cache_data = session_data.get("cache", {})
            cache_summary = {}
            if cache_data:
                for tool_name in cache_data.keys():
                    cache_summary[tool_name] = "cached"
            
            return video_analysis_pb2.SessionDataResponse(
                session=session_info,
                messages=messages,
                has_cache=bool(cache_data),  # Compute from cache data
                cache_summary=cache_summary
            )
        except Exception as e:
            logger.exception(f"LoadSession error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return video_analysis_pb2.SessionDataResponse()
    
    async def DeleteSession(self, request, context):
        """Delete a chat session"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        try:
            session_id = request.session_id
            success = session_manager.delete_session(session_id)
            
            return video_analysis_pb2.DeleteSessionResponse(
                success=success,
                message="Session deleted successfully" if success else "Session not found"
            )
        except Exception as e:
            logger.exception(f"DeleteSession error: {e}")
            return video_analysis_pb2.DeleteSessionResponse(
                success=False,
                message=str(e)
            )
    
    async def UpdateSession(self, request, context):
        """Update a chat session (e.g., rename)"""
        from . import video_analysis_pb2
        from ..utils.session_manager import session_manager
        
        try:
            session_id = request.session_id
            title = request.title
            
            success = session_manager.update_session_title(session_id, title)
            
            return video_analysis_pb2.UpdateSessionResponse(
                success=success,
                message="Session updated successfully" if success else "Session not found"
            )
        except Exception as e:
            logger.exception(f"UpdateSession error: {e}")
            return video_analysis_pb2.UpdateSessionResponse(
                success=False,
                message=str(e)
            )
