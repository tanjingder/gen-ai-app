"""
MCP Orchestrator V2 - Cache-First, Intent-Driven Architecture

Flow:
[User Query] → [1] Analyze Intent → [2] Check Cache → [3] Plan Missing → 
[4] Execute Tools → [5] Synthesize → [6] Output & Cache
"""
from typing import Any, Dict, List, Optional, Literal
import json
import re
from pathlib import Path
from datetime import datetime
from loguru import logger
import ollama

from .client import mcp_client
from ..utils.config import settings


# Query intent types
QueryIntent = Literal[
    "summarize",           # Full video summary + PDF
    "transcribe",          # Audio transcription only
    "visual_analysis",     # Frame analysis only
    "detect_objects",      # Object detection
    "extract_text",        # OCR text extraction
    "chat",                # General question about video
    "report"               # Explicit report request
]


class SessionCache:
    """Manages session-level caching of analysis results"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.base_path = Path(f"data/sessions/{session_id}")
        self.cache_path = self.base_path / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data by key"""
        cache_file = self.cache_path / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ Cache hit: {key}")
                    return data
            except Exception as e:
                logger.error(f"Error reading cache {key}: {e}")
        return None
    
    def set(self, key: str, value: Dict[str, Any]):
        """Cache data by key"""
        cache_file = self.cache_path / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Cached: {key}")
        except Exception as e:
            logger.error(f"Error writing cache {key}: {e}")
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return (self.cache_path / f"{key}.json").exists()
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get multiple cached items at once"""
        return {key: self.get(key) for key in keys}


class MCPOrchestratorV2:
    """
    Intent-driven orchestrator with cache-first approach
    """
    
    def __init__(self):
        self.client = mcp_client
        self.ollama_model = settings.OLLAMA_MODEL
    
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text by removing encoding issues and alien characters"""
        if not text:
            return text
        
        # Remove common encoding artifacts
        text = text.replace('â€˜', "'")  # Single quote start
        text = text.replace('â€™', "'")  # Single quote end
        text = text.replace('â€œ', '"')  # Double quote start
        text = text.replace('â€', '"')   # Double quote end
        text = text.replace('â€"', '-')  # Em dash
        text = text.replace('â€"', '-')  # En dash
        text = text.replace('Â', '')     # Non-breaking space artifact
        text = text.replace('â€¦', '...')  # Ellipsis
        
        # Remove non-printable characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _format_transcript(self, transcription_data: Dict[str, Any]) -> str:
        """Format transcript by removing timestamps and creating clean paragraph"""
        if not transcription_data:
            return ""
        
        text = transcription_data.get("text", "") if isinstance(transcription_data, dict) else ""
        if not text:
            return ""
        
        # Remove timestamp patterns like [00:00:00.000 --> 00:00:07.160]
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]', '', text)
        
        # Remove any remaining square brackets
        text = re.sub(r'\[|\]', '', text)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # ============================================================
    # STEP 1: ANALYZE USER QUERY & INTENT
    # ============================================================
    
    async def analyze_intent(self, user_query: str, video_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step 1: Analyze user query to determine intent and required data
        
        Returns:
            {
                "intent": "summarize" | "transcribe" | "visual_analysis" | ...,
                "output_type": "text" | "pdf" | "structured",
                "required_data": ["transcription", "frames", "visual_analysis"],
                "reasoning": "..."
            }
        """
        prompt = f"""Analyze this user query about a video and determine the intent.

User Query: "{user_query}"

Return JSON with:
{{
  "intent": "summarize | transcribe | visual_analysis | detect_objects | extract_text | chat | report",
  "output_type": "text | pdf | structured | chat_response",
  "required_data": ["transcription", "frames", "visual_analysis", "objects", "text"],
  "reasoning": "brief explanation of what user wants"
}}

Intent Guidelines:
- "summarize" → User wants full video summary (may include PDF/PPT)
- "transcribe" → User wants audio transcription only
- "visual_analysis" → User wants to know what's shown visually
- "detect_objects" → User wants object detection
- "extract_text" → User wants OCR text from video
- "chat" → User asking a question about video content
- "report" → User explicitly asks for PDF/PPT report

Output Type:
- "text" → Return plain text response
- "pdf" → Generate PDF report
- "ppt" → Generate PowerPoint presentation
- "structured" → Return structured data
- "chat_response" → Natural language answer

Required Data (what we need to answer):
- "transcription" → Need audio transcription
- "frames" → Need extracted frames
- "visual_analysis" → Need frame analysis with descriptions
- "objects" → Need object detection
- "text" → Need OCR text extraction

Examples:
"Summarize this video" → intent: summarize, required_data: [transcription, visual_analysis]
"What is said in the video?" → intent: transcribe, required_data: [transcription]
"Create a PDF report" → intent: report, output_type: pdf, required_data: [transcription, visual_analysis]
"Generate a PowerPoint presentation" → intent: report, output_type: ppt, required_data: [transcription, visual_analysis]
"What objects are shown?" → intent: detect_objects, required_data: [frames, objects]

Return ONLY valid JSON."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a query analysis expert. Determine user intent accurately."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            intent_data = json.loads(response["message"]["content"])
            logger.info(f"📊 Intent Analysis: {intent_data.get('intent')} → {intent_data.get('output_type')}")
            logger.debug(f"Required data: {intent_data.get('required_data')}")
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            # Fallback to basic intent
            return {
                "intent": "chat",
                "output_type": "chat_response",
                "required_data": ["transcription"],
                "reasoning": "Fallback to basic chat intent"
            }
    
    # ============================================================
    # STEP 2: CHECK SESSION CACHE
    # ============================================================
    
    def check_cache_status(self, cache: SessionCache, required_data: List[str]) -> Dict[str, bool]:
        """
        Step 2: Check which required data is already cached
        
        Returns:
            {"transcription": True, "visual_analysis": False, ...}
        """
        status = {}
        
        # Map required data to cache keys
        cache_mapping = {
            "transcription": "transcription",
            "frames": "frames",
            "visual_analysis": "visual_analysis",
            "objects": "objects",
            "text": "extracted_text",
            "structured_summary": "structured_summary",
            "pdf_report": "pdf_report",
            "ppt_report": "ppt_report"
        }
        
        for data_type in required_data:
            cache_key = cache_mapping.get(data_type, data_type)
            status[data_type] = cache.has(cache_key)
        
        cached = [k for k, v in status.items() if v]
        missing = [k for k, v in status.items() if not v]
        
        logger.info(f"📦 Cache Status: {len(cached)} cached, {len(missing)} missing")
        if cached:
            logger.debug(f"  ✅ Cached: {cached}")
        if missing:
            logger.debug(f"  ❌ Missing: {missing}")
        
        return status
    
    # ============================================================
    # STEP 3: PLAN MISSING STEPS
    # ============================================================
    
    def plan_missing_tools(self, required_data: List[str], cache_status: Dict[str, bool]) -> List[Dict[str, str]]:
        """
        Step 3: Plan which tools to execute for missing data
        
        Returns:
            [{"agent": "transcription", "tool": "extract_audio", ...}, ...]
        """
        tools_plan = []
        
        # Transcription pipeline
        if "transcription" in required_data and not cache_status.get("transcription", False):
            tools_plan.extend([
                {"agent": "transcription", "tool": "extract_audio", "reason": "Extract audio track"},
                {"agent": "transcription", "tool": "transcribe_audio", "reason": "Transcribe audio to text"}
            ])
        
        # Frames extraction
        if any(d in required_data for d in ["frames", "visual_analysis", "objects", "text"]) and not cache_status.get("frames", False):
            tools_plan.append(
                {"agent": "vision", "tool": "extract_frames", "reason": "Extract video frames"}
            )
        
        # Visual analysis
        if "visual_analysis" in required_data and not cache_status.get("visual_analysis", False):
            tools_plan.append(
                {"agent": "vision", "tool": "analyze_frame", "reason": "Analyze frame content"}
            )
        
        # Object detection
        if "objects" in required_data and not cache_status.get("objects", False):
            tools_plan.append(
                {"agent": "vision", "tool": "detect_objects", "reason": "Detect objects in frames"}
            )
        
        # Text extraction
        if "text" in required_data and not cache_status.get("text", False):
            tools_plan.append(
                {"agent": "vision", "tool": "extract_text", "reason": "Extract text via OCR"}
            )
        
        logger.info(f"📋 Tools Plan: {len(tools_plan)} tools to execute")
        return tools_plan
    
    # ============================================================
    # STEP 4: EXECUTE TOOLS
    # ============================================================
    
    async def execute_tools(
        self, 
        tools_plan: List[Dict[str, str]], 
        video_id: str, 
        video_path: str,
        cache: SessionCache
    ) -> Dict[str, Any]:
        """
        Step 4: Execute planned tools and cache results
        
        Returns:
            {
                "transcription": {...},
                "frames": {...},
                "visual_analysis": {...},
                ...
            }
        """
        results = {}
        temp_dir = cache.base_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Track intermediate results for dependent tools
        intermediate_data = {}
        
        for tool_spec in tools_plan:
            agent = tool_spec["agent"]
            tool = tool_spec["tool"]
            reason = tool_spec.get("reason", "")
            
            logger.info(f"⚙️ Executing: {agent}.{tool} - {reason}")
            
            try:
                # Execute tool
                result = await self._execute_single_tool(
                    agent, tool, video_id, video_path, 
                    str(temp_dir), intermediate_data
                )
                
                # Cache result by tool type
                if tool == "transcribe_audio":
                    cache_key = "transcription"
                    results["transcription"] = result
                    intermediate_data["transcription"] = result.get("text", "")
                    cache.set(cache_key, result)
                    
                elif tool == "extract_frames":
                    cache_key = "frames"
                    results["frames"] = result
                    intermediate_data["frames"] = result.get("frames", [])
                    intermediate_data["total_frames"] = result.get("total_frames", 0)
                    cache.set(cache_key, result)
                    
                elif tool == "analyze_frame":
                    cache_key = "visual_analysis"
                    results["visual_analysis"] = result
                    cache.set(cache_key, result)
                    
                elif tool == "detect_objects":
                    cache_key = "objects"
                    results["objects"] = result
                    cache.set(cache_key, result)
                    
                elif tool == "extract_text":
                    cache_key = "extracted_text"
                    results["text"] = result
                    cache.set(cache_key, result)
                
                logger.info(f"✅ Completed: {agent}.{tool}")
                
            except Exception as e:
                logger.error(f"❌ Error executing {agent}.{tool}: {e}")
                results[f"error_{tool}"] = str(e)
        
        return results
    
    async def _execute_single_tool(
        self,
        agent: str,
        tool: str,
        video_id: str,
        video_path: str,
        temp_dir: str,
        intermediate_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single MCP tool"""
        
        if agent == "transcription":
            if tool == "extract_audio":
                audio_path = f"{temp_dir}/{video_id}_audio.wav"
                return await self.client.extract_audio(video_path, audio_path)
            elif tool == "transcribe_audio":
                audio_path = f"{temp_dir}/{video_id}_audio.wav"
                return await self.client.transcribe_audio(audio_path)
        
        elif agent == "vision":
            if tool == "extract_frames":
                return await self.client.extract_frames(video_path, output_dir=temp_dir)
            
            elif tool == "analyze_frame":
                # Comprehensive frame analysis
                frames_data = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if not frames_data:
                    frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                    return await self.client.analyze_frame(frame_path)
                
                # Sample frames
                sample_rate = max(1, total_frames // 5) if total_frames > 5 else 1
                frame_analyses = []
                
                for i, frame_info in enumerate(frames_data):
                    if i % sample_rate == 0:
                        frame_path = frame_info.get("path")
                        if frame_path:
                            try:
                                description_result = await self.client.analyze_frame(frame_path)
                                objects_result = await self.client.detect_objects(frame_path)
                                text_result = await self.client.extract_text(frame_path)
                                
                                frame_analyses.append({
                                    "frame": i,
                                    "timestamp": frame_info.get("timestamp", 0.0),
                                    "description": description_result.get("description", ""),
                                    "objects": [obj.get("class") for obj in objects_result.get("objects", [])],
                                    "text": text_result.get("text", ""),
                                    "confidence": description_result.get("confidence", 0.0)
                                })
                            except Exception as e:
                                logger.error(f"Error analyzing frame {i}: {e}")
                
                return {
                    "frame_analyses": frame_analyses,
                    "total_analyzed": len(frame_analyses),
                    "sample_rate": sample_rate
                }
            
            elif tool == "detect_objects":
                frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                return await self.client.detect_objects(frame_path)
            
            elif tool == "extract_text":
                frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                return await self.client.extract_text(frame_path)
        
        return {"error": f"Unknown tool: {agent}.{tool}"}
    
    # ============================================================
    # STEP 5: SYNTHESIZE WITH LLM
    # ============================================================
    
    async def synthesize_response(
        self,
        user_query: str,
        intent_data: Dict[str, Any],
        cached_data: Dict[str, Any],
        fresh_data: Dict[str, Any],
        cache: SessionCache
    ) -> str:
        """
        Step 5: Synthesize final response based on intent and available data
        """
        intent = intent_data.get("intent")
        output_type = intent_data.get("output_type")
        
        # Merge cached and fresh data
        all_data = {**cached_data, **fresh_data}
        
        # Handle different intents
        if intent == "transcribe":
            return await self._synthesize_transcription(all_data)
        
        elif intent in ["summarize", "report"]:
            return await self._synthesize_summary(user_query, all_data, cache, output_type == "pdf")
        
        elif intent == "visual_analysis":
            return await self._synthesize_visual_analysis(all_data)
        
        elif intent == "detect_objects":
            return await self._synthesize_objects(all_data)
        
        elif intent == "extract_text":
            return await self._synthesize_text(all_data)
        
        else:  # chat
            return await self._synthesize_chat(user_query, all_data)
    
    async def _synthesize_transcription(self, data: Dict[str, Any]) -> str:
        """Return clean transcription text"""
        transcription_data = data.get("transcription", {})
        
        if isinstance(transcription_data, dict):
            text = transcription_data.get("text", "")
        else:
            text = str(transcription_data)
        
        if not text:
            return "❌ No transcription available."
        
        # Clean up
        import re
        text_clean = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', text)
        text_clean = ' '.join(text_clean.split())
        
        return f"📝 **Video Transcription:**\n\n{text_clean}"
    
    async def _synthesize_summary(
        self, 
        user_query: str,
        data: Dict[str, Any], 
        cache: SessionCache,
        generate_pdf: bool = False
    ) -> str:
        """Generate structured summary and optionally PDF/PPT"""
        
        # Check if we already have structured summary
        structured_summary = cache.get("structured_summary")
        
        if not structured_summary:
            # Create new structured summary
            transcription_data = data.get("transcription", {})
            visual_data = data.get("visual_analysis", {})
            
            transcription_text = transcription_data.get("text", "") if isinstance(transcription_data, dict) else ""
            frame_analyses = visual_data.get("frame_analyses", []) if isinstance(visual_data, dict) else []
            
            structured_summary = await self._create_structured_summary(transcription_text, frame_analyses)
            cache.set("structured_summary", structured_summary)
        
        # Generate PDF if requested
        pdf_path = None
        if generate_pdf or "pdf" in user_query.lower():
            pdf_path = await self._generate_pdf_report(structured_summary, data, cache)
        
        # Generate PPT if requested
        ppt_path = None
        if "ppt" in user_query.lower() or "powerpoint" in user_query.lower():
            ppt_path = await self._generate_ppt_report(structured_summary, data, cache)
        
        # Format response
        return self._format_summary_response(structured_summary, pdf_path, ppt_path)
    
    async def _create_structured_summary(self, transcription: str, frame_analyses: List[Dict]) -> Dict[str, Any]:
        """Create structured summary using LLM"""
        prompt = f"""Create a structured video summary.

**Transcription:**
{transcription[:5000]}

**Visual Analysis:**
"""
        for frame in frame_analyses[:10]:
            prompt += f"\n[Frame {frame.get('frame')} @ {frame.get('timestamp', 0):.1f}s]: {frame.get('description', '')}"
        
        prompt += """

Return JSON:
{
  "title": "5-10 word title",
  "summary": "4-6 sentence summary",
  "key_moments": [{"time": "MM:SS", "event": "description"}],
  "takeaways": ["insight 1", "insight 2", ...]
}"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a video summarization expert."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            return json.loads(response["message"]["content"])
        except:
            return {
                "title": "Video Summary",
                "summary": "Analysis completed.",
                "key_moments": [],
                "takeaways": []
            }
    
    async def _generate_pdf_report(
        self,
        structured_summary: Dict[str, Any],
        data: Dict[str, Any],
        cache: SessionCache
    ) -> str:
        """Generate PDF report with comprehensive structured content"""
        logger.info("📄 Generating PDF report...")
        
        # Extract and format transcription (clean timestamps)
        transcription_data = data.get("transcription", {})
        transcription_text = self._format_transcript(transcription_data)
        
        # Extract visual analysis
        visual_data = data.get("visual_analysis", {})
        frame_analyses = visual_data.get("frame_analyses", []) if isinstance(visual_data, dict) else []
        
        # Format visual analysis section
        visual_summary = ""
        if frame_analyses:
            visual_summary = f"Analyzed {len(frame_analyses)} frames\n\n"
            # Include key frames with descriptions
            for i, frame in enumerate(frame_analyses[:8], 1):  # Top 8 frames
                timestamp = frame.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                # Get full description (don't truncate - LLaVA now gives concise descriptions)
                description = frame.get('description', 'No description')
                visual_summary += f"{time_str} - {description}\n"
                
                # Add detected objects if available
                objects = frame.get('objects', [])
                if objects:
                    visual_summary += f"  Objects: {', '.join(objects[:5])}\n"
                
                # Add extracted text if available (clean encoding issues)
                text = frame.get('text', '')
                if text:
                    cleaned_text = self._clean_ocr_text(text)
                    if cleaned_text and len(cleaned_text.strip()) > 0:  # Only show if there's meaningful text
                        visual_summary += f"  Text: \"{cleaned_text[:150]}\"\n"
                visual_summary += "\n"
        
        # Build comprehensive content
        content = {
            "title": structured_summary.get("title", "Video Analysis Report"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "summary": structured_summary.get("summary", ""),
            "key_moments": structured_summary.get("key_moments", []),
            "visual_analysis": visual_summary,
            "transcript": transcription_text,
            "takeaways": structured_summary.get("takeaways", [])
        }
        
        # Generate filename from title (sanitize for filesystem)
        title = structured_summary.get("title", "Video Analysis Report")
        safe_filename = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid chars
        safe_filename = safe_filename.strip()[:100]  # Limit length
        if not safe_filename:
            safe_filename = "Video_Analysis_Report"
        
        output_path = cache.base_path / "reports" / f"{safe_filename}.pdf"
        output_path.parent.mkdir(exist_ok=True)
        
        # Call report agent to generate PDF
        try:
            await self.client.create_pdf_report(content, str(output_path))
            cache.set("pdf_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return None
    
    async def _generate_ppt_report(
        self,
        structured_summary: Dict[str, Any],
        data: Dict[str, Any],
        cache: SessionCache
    ) -> str:
        """Generate PowerPoint report with same content as PDF"""
        logger.info("📊 Generating PowerPoint report...")
        
        # Extract and format transcription (clean timestamps)
        transcription_data = data.get("transcription", {})
        transcription_text = self._format_transcript(transcription_data)
        
        # Extract visual analysis
        visual_data = data.get("visual_analysis", {})
        frame_analyses = visual_data.get("frame_analyses", []) if isinstance(visual_data, dict) else []
        
        # Format visual analysis section
        visual_summary = ""
        if frame_analyses:
            visual_summary = f"Analyzed {len(frame_analyses)} frames\n\n"
            # Include key frames with descriptions
            for i, frame in enumerate(frame_analyses[:8], 1):  # Top 8 frames
                timestamp = frame.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                # Get full description (don't truncate - LLaVA now gives concise descriptions)
                description = frame.get('description', 'No description')
                visual_summary += f"{time_str} - {description}\n"
                
                # Add detected objects if available
                objects = frame.get('objects', [])
                if objects:
                    visual_summary += f"  Objects: {', '.join(objects[:5])}\n"
                
                # Add extracted text if available (clean encoding issues)
                text = frame.get('text', '')
                if text:
                    cleaned_text = self._clean_ocr_text(text)
                    if cleaned_text and len(cleaned_text.strip()) > 0:  # Only show if there's meaningful text
                        visual_summary += f"  Text: \"{cleaned_text[:150]}\"\n"
                visual_summary += "\n"
        
        # Build comprehensive content (same as PDF)
        content = {
            "title": structured_summary.get("title", "Video Analysis Report"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "summary": structured_summary.get("summary", ""),
            "key_moments": structured_summary.get("key_moments", []),
            "visual_analysis": visual_summary,
            "transcript": transcription_text,
            "takeaways": structured_summary.get("takeaways", [])
        }
        
        # Generate filename from title (sanitize for filesystem)
        title = structured_summary.get("title", "Video Analysis Report")
        safe_filename = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid chars
        safe_filename = safe_filename.strip()[:100]  # Limit length
        if not safe_filename:
            safe_filename = "Video_Analysis_Report"
        
        output_path = cache.base_path / "reports" / f"{safe_filename}.pptx"
        output_path.parent.mkdir(exist_ok=True)
        
        # Call report agent to generate PPT
        try:
            await self.client.create_ppt_report(content, str(output_path))
            cache.set("ppt_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate PPT report: {e}")
            return None
    
    def _format_visual_summary(self, visual_data: Dict[str, Any]) -> str:
        """Format visual analysis for report inclusion"""
        if not isinstance(visual_data, dict):
            return ""
        
        frame_analyses = visual_data.get("frame_analyses", [])
        if not frame_analyses:
            return ""
        
        summary = f"Analyzed {len(frame_analyses)} frames:\n"
        for frame in frame_analyses[:5]:  # Include top 5 frames
            summary += f"• Frame {frame.get('frame')} ({frame.get('timestamp', 0):.1f}s): {frame.get('description', '')[:100]}\n"
        
        return summary
    
    def _format_summary_response(self, summary: Dict[str, Any], pdf_path: Optional[str], ppt_path: Optional[str] = None) -> str:
        """Format summary for chat response"""
        response = f"✅ **Video Analysis Complete!**\n\n"
        response += f"**{summary.get('title', 'Video Summary')}**\n\n"
        response += f"{summary.get('summary', '')}\n\n"
        
        key_moments = summary.get("key_moments", [])
        if key_moments:
            response += "**Key Moments:**\n"
            for moment in key_moments:
                if isinstance(moment, dict):
                    response += f"• {moment.get('time', '00:00')} - {moment.get('event', '')}\n"
            response += "\n"
        
        takeaways = summary.get("takeaways", [])
        if takeaways:
            response += "**Key Takeaways:**\n"
            for i, takeaway in enumerate(takeaways, 1):
                response += f"{i}. {takeaway}\n"
            response += "\n"
        
        # Show available reports
        if pdf_path:
            response += f"📄 **Full PDF Report:** `{pdf_path}`\n"
        
        if ppt_path:
            response += f"📊 **PowerPoint Presentation:** `{ppt_path}`\n"
        
        return response
    
    async def _synthesize_visual_analysis(self, data: Dict[str, Any]) -> str:
        """Return visual analysis results"""
        visual_data = data.get("visual_analysis", {})
        frame_analyses = visual_data.get("frame_analyses", []) if isinstance(visual_data, dict) else []
        
        if not frame_analyses:
            return "❌ No visual analysis available."
        
        response = f"🎬 **Visual Analysis ({len(frame_analyses)} frames):**\n\n"
        for frame in frame_analyses[:10]:
            response += f"**Frame {frame.get('frame')} ({frame.get('timestamp', 0):.1f}s):**\n"
            response += f"{frame.get('description', 'No description')}\n\n"
        
        return response
    
    async def _synthesize_objects(self, data: Dict[str, Any]) -> str:
        """Return object detection results"""
        objects_data = data.get("objects", {})
        
        # Check if we have object detection data
        if not objects_data or not isinstance(objects_data, dict):
            return "❌ No object detection data available."
        
        detected_objects = objects_data.get("objects", [])
        
        if not detected_objects:
            return "❌ No objects detected in video."
        
        # Build response with object details
        response = f"🔍 **Detected Objects ({len(detected_objects)} total):**\n\n"
        
        # Group objects by class and count
        from collections import Counter
        object_counts = Counter()
        object_details = []
        
        for obj in detected_objects:
            if isinstance(obj, dict):
                obj_class = obj.get("class", obj.get("label", "unknown"))
                confidence = obj.get("confidence", 0.0)
                bbox = obj.get("bbox", obj.get("box", {}))
                
                object_counts[obj_class] += 1
                object_details.append({
                    "class": obj_class,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        # Show summary by class
        response += "**Object Summary:**\n"
        for obj_class, count in object_counts.most_common():
            response += f"• {obj_class}: {count} instance{'s' if count > 1 else ''}\n"
        
        response += "\n**Detailed Detections:**\n"
        for i, obj in enumerate(object_details[:20], 1):  # Show top 20
            conf_percent = obj["confidence"] * 100 if obj["confidence"] <= 1.0 else obj["confidence"]
            response += f"{i}. **{obj['class']}** - Confidence: {conf_percent:.1f}%\n"
            
            if obj.get("bbox"):
                bbox = obj["bbox"]
                if isinstance(bbox, dict):
                    response += f"   Location: x={bbox.get('x', 0):.0f}, y={bbox.get('y', 0):.0f}, w={bbox.get('width', 0):.0f}, h={bbox.get('height', 0):.0f}\n"
        
        if len(object_details) > 20:
            response += f"\n_...and {len(object_details) - 20} more objects_"
        
        return response
    
    async def _synthesize_text(self, data: Dict[str, Any]) -> str:
        """Return extracted text"""
        text_data = data.get("text", {})
        extracted_text = text_data.get("text", "") if isinstance(text_data, dict) else ""
        
        if not extracted_text:
            return "❌ No text detected in video."
        
        return f"📄 **Extracted Text:**\n\n{extracted_text}"
    
    async def _synthesize_chat(self, user_query: str, data: Dict[str, Any]) -> str:
        """Generate chat response using available data"""
        # Use LLM to answer question based on available data
        context = f"User asked: {user_query}\n\nAvailable data: {json.dumps(data, indent=2)[:2000]}"
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "Answer the user's question about the video based on available analysis data."},
                    {"role": "user", "content": context}
                ]
            )
            return response["message"]["content"]
        except:
            return "I couldn't generate a response. Please try again."
    
    # ============================================================
    # MAIN ORCHESTRATION FLOW
    # ============================================================
    
    async def process_query(
        self,
        user_query: str,
        video_id: str,
        video_path: str,
        session_id: str,
        video_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main orchestration flow:
        Query → Intent → Cache Check → Plan → Execute → Synthesize → Output
        """
        logger.info(f"🎯 Processing query: {user_query[:100]}")
        
        # Initialize session cache
        cache = SessionCache(session_id)
        
        # STEP 1: Analyze Intent
        intent_data = await self.analyze_intent(user_query, video_context)
        required_data = intent_data.get("required_data", [])
        
        # STEP 2: Check Cache
        cache_status = self.check_cache_status(cache, required_data)
        
        # Load cached data
        cached_data = {}
        for data_type, is_cached in cache_status.items():
            if is_cached:
                cache_key = data_type if data_type != "text" else "extracted_text"
                cached_data[data_type] = cache.get(cache_key)
        
        # STEP 3: Plan Missing Tools
        tools_plan = self.plan_missing_tools(required_data, cache_status)
        
        # STEP 4: Execute Tools (if needed)
        fresh_data = {}
        if tools_plan:
            logger.info(f"🚀 Executing {len(tools_plan)} tools...")
            fresh_data = await self.execute_tools(tools_plan, video_id, video_path, cache)
        else:
            logger.info("✅ All data available in cache!")
        
        # STEP 5 & 6: Synthesize & Output
        response = await self.synthesize_response(
            user_query,
            intent_data,
            cached_data,
            fresh_data,
            cache
        )
        
        logger.info("✅ Query processing complete")
        return response


# Global orchestrator instance
orchestrator = MCPOrchestratorV2()
