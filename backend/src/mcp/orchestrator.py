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
        
        # Remove sequences of repeating symbols/numbers (OCR noise)
        # Pattern: 8-8-8-8- or SSS or 888 (3+ repeating chars)
        text = re.sub(r'(\d[^\w\s]){3,}', ' ', text)  # Remove patterns like 8-8-8-
        text = re.sub(r'([A-Z])\1{2,}', ' ', text)     # Remove SSS, AAA, etc.
        text = re.sub(r'(\d)\1{2,}', ' ', text)        # Remove 888, 000, etc.
        
        # Remove non-printable characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Remove lines that are mostly non-alphabetic (OCR garbage)
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # Count alphabetic vs non-alphabetic characters
            alpha_count = sum(c.isalpha() for c in line)
            total_count = len(line.strip())
            
            # Keep line if it's at least 30% alphabetic or very short
            if total_count == 0 or (alpha_count / total_count >= 0.3) or total_count < 10:
                clean_lines.append(line)
        
        text = '\n'.join(clean_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        
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

Return JSON with EXACTLY ONE value for each field:
{{
  "intent": "(choose ONE: summarize, transcribe, visual_analysis, detect_objects, extract_text, chat, or report)",
  "output_type": "(choose ONE: text, pdf, ppt, structured, or chat_response)",
  "required_data": ["array of needed data types"],
  "reasoning": "brief explanation of what user wants"
}}

IMPORTANT: Select ONLY ONE intent value, not multiple. For output_type, use "ppt" for PowerPoint requests.

Intent Guidelines:
- "summarize" → User wants full video summary in TEXT format only (no PDF/PPT mentioned)
- "transcribe" → User wants audio transcription only
- "visual_analysis" → User wants to know what's shown visually (ANY question about graphs, charts, images, visuals, what's shown, what can be seen, etc.)
- "detect_objects" → User wants specific object detection/counting (e.g., "what objects?", "detect items", "count people")
- "extract_text" → User wants OCR text from video (e.g., "what text is shown?", "read the text")
- "chat" → User asking a general question about video content or asking for interpretation/explanation AFTER having data
- "report" → User explicitly asks for PDF/PPT/PowerPoint report or wants to generate a document

CRITICAL RULES:
1. If user asks about visual content (graphs, charts, what's shown, what's visible, etc.) → ALWAYS use "visual_analysis" intent
2. If query contains "PDF", "PPT", "PowerPoint", "presentation", or "report" → ALWAYS use "report" intent
3. If user asks to "summarize into [format]" where format is PDF/PPT/PowerPoint → use "report" intent, NOT "summarize"

Output Type Rules:
- "text" → Plain text response for most queries (summaries, visual descriptions, Q&A)
- "pdf" → ONLY if user explicitly asks for PDF ("create PDF", "generate PDF", "summarize to PDF")
- "ppt" → ONLY if user explicitly asks for PowerPoint/PPT ("create PPT", "PowerPoint", "presentation")
- "structured" → Return structured JSON data (rarely used)
- "chat_response" → Natural language answer to questions

IMPORTANT: Use "text" as default output_type. Only use "pdf" or "ppt" if explicitly requested.

Required Data (what we need to answer):
- "transcription" → Need audio transcription
- "frames" → Need extracted frames
- "visual_analysis" → Need frame analysis with descriptions
- "objects" → Need object detection
- "text" → Need OCR text extraction

Examples:
"Summarize this video" → intent: summarize, output_type: text, required_data: [transcription, visual_analysis]
"What is said in the video?" → intent: transcribe, output_type: text, required_data: [transcription]
"Analyze the visual content" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"What's shown in the video?" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"Describe what's happening visually" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"Are there any graphs in the video?" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"Are there any graphs or charts in this video?" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"Does the video show any charts?" → intent: visual_analysis, required_data: [frames, visual_analysis]
"What can you see in the video?" → intent: visual_analysis, required_data: [frames, visual_analysis]
"What's visible in the frames?" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"Show me what's in the video" → intent: visual_analysis, output_type: text, required_data: [frames, visual_analysis]
"What objects are shown?" → intent: detect_objects, output_type: text, required_data: [frames, objects]
"Create a PDF report" → intent: report, output_type: pdf, required_data: [transcription, frames, visual_analysis]
"Generate a PowerPoint presentation" → intent: report, output_type: ppt, required_data: [transcription, frames, visual_analysis]
"Summarize this video into PDF" → intent: report, output_type: pdf, required_data: [transcription, frames, visual_analysis]
"Can you summarize into PDF?" → intent: report, output_type: pdf, required_data: [transcription, frames, visual_analysis]
"Can you summarize into PowerPoint?" → intent: report, output_type: ppt, required_data: [transcription, frames, visual_analysis]
"Summarize into PPT" → intent: report, output_type: ppt, required_data: [transcription, frames, visual_analysis]
"Make a presentation" → intent: report, output_type: ppt, required_data: [transcription, frames, visual_analysis]
"Create a PPT" → intent: report, output_type: ppt, required_data: [transcription, frames, visual_analysis]

CRITICAL: 
- Questions about visual content (graphs, charts, what's shown) → intent: visual_analysis, output_type: text
- Report generation requests with PDF/PPT/PowerPoint → intent: report, output_type: pdf or ppt
- Default output_type is "text" unless explicitly requesting PDF or PPT file

Return ONLY valid JSON."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a query analysis expert. Return valid JSON with single values only - never return multiple options separated by pipes. For intent, choose exactly ONE value from: summarize, transcribe, visual_analysis, detect_objects, extract_text, chat, or report."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            intent_data = json.loads(response["message"]["content"])
            
            # Validate and fix intent if it contains multiple values (pipe-separated)
            intent_value = intent_data.get('intent', '')
            if '|' in intent_value or ' ' in intent_value:
                # LLM returned malformed intent - try to extract the correct one
                query_lower = user_query.lower()
                
                # Check for explicit format requests first (highest priority)
                if 'powerpoint' in query_lower or 'ppt' in query_lower or 'presentation' in query_lower:
                    intent_data['intent'] = 'report'
                    intent_data['output_type'] = 'ppt'
                    logger.warning(f"Fixed malformed intent. Detected PowerPoint request → intent: report")
                elif 'pdf' in query_lower:
                    intent_data['intent'] = 'report'
                    intent_data['output_type'] = 'pdf'
                    logger.warning(f"Fixed malformed intent. Detected PDF request → intent: report")
                elif 'summarize' in query_lower or 'summary' in query_lower:
                    intent_data['intent'] = 'summarize'
                    logger.warning(f"Fixed malformed intent. Detected summary request → intent: summarize")
                else:
                    intent_data['intent'] = 'chat'
                    logger.warning(f"Fixed malformed intent. Defaulting to: chat")
            
            # Validate output_type - fix common mistakes
            output_type = intent_data.get('output_type', 'text')
            intent = intent_data.get('intent', '')
            required_data = intent_data.get('required_data', [])
            query_lower = user_query.lower()
            
            # Fix output_type based on intent and query content
            if intent == 'visual_analysis' and output_type == 'ppt':
                # Visual analysis questions should return text, not PPT
                if not any(word in query_lower for word in ['ppt', 'powerpoint', 'presentation']):
                    intent_data['output_type'] = 'text'
                    logger.warning(f"Fixed output_type: visual_analysis should use 'text' not 'ppt'")
            
            elif intent == 'report':
                # Report intent should have pdf or ppt output_type
                if 'pdf' in query_lower:
                    intent_data['output_type'] = 'pdf'
                elif any(word in query_lower for word in ['ppt', 'powerpoint', 'presentation']):
                    intent_data['output_type'] = 'ppt'
            
            elif intent in ['summarize', 'transcribe', 'detect_objects', 'extract_text'] and output_type not in ['text', 'chat_response']:
                # These intents should default to text unless explicitly requesting a report
                if 'pdf' not in query_lower and 'ppt' not in query_lower and 'powerpoint' not in query_lower:
                    intent_data['output_type'] = 'text'
                    logger.warning(f"Fixed output_type: {intent} should use 'text' as default")
            
            # Fix required_data - ensure it matches the intent
            if intent == 'visual_analysis':
                # Visual analysis MUST have frames and visual_analysis in required_data
                if 'frames' not in required_data:
                    required_data.append('frames')
                    logger.warning(f"Fixed required_data: Added 'frames' for visual_analysis intent")
                if 'visual_analysis' not in required_data:
                    required_data.append('visual_analysis')
                    logger.warning(f"Fixed required_data: Added 'visual_analysis' for visual_analysis intent")
                intent_data['required_data'] = required_data
            
            elif intent == 'detect_objects':
                # Object detection needs frames and objects
                if 'frames' not in required_data:
                    required_data.append('frames')
                if 'objects' not in required_data:
                    required_data.append('objects')
                intent_data['required_data'] = required_data
            
            elif intent == 'report':
                # Reports typically need transcription, frames, and visual_analysis
                expected = ['transcription', 'frames', 'visual_analysis']
                for item in expected:
                    if item not in required_data:
                        required_data.append(item)
                intent_data['required_data'] = required_data
            
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
            logger.info(f"  ✅ Cached: {cached}")
        if missing:
            logger.info(f"  ❌ Missing: {missing}")
        
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
        
        # Frames extraction - only if frames needed AND not cached
        needs_frames = any(d in required_data for d in ["frames", "visual_analysis", "objects", "text"])
        frames_cached = cache_status.get("frames", False)
        
        if needs_frames and not frames_cached:
            logger.debug(f"  📌 Adding extract_frames: needs_frames={needs_frames}, cached={frames_cached}")
            tools_plan.append(
                {"agent": "vision", "tool": "extract_frames", "reason": "Extract video frames"}
            )
        elif needs_frames and frames_cached:
            logger.debug(f"  ✅ Skipping extract_frames: frames already cached")
        
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
        
        #JD
        # Load cached frames if available (for multi-frame operations)
        cached_frames = cache.get("frames")
        if cached_frames:
            intermediate_data["frames"] = cached_frames.get("frames", [])
            intermediate_data["total_frames"] = cached_frames.get("total_frames", 0)
            logger.info(f"📦 Loaded {intermediate_data['total_frames']} cached frames for multi-frame operations")
        #JD

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
                # Analyze 10 evenly-distributed frames for better coverage
                # Get frames list and total count from intermediate_data
                frames_list = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if total_frames == 0 or not frames_list:
                    return {"error": "No frames available for object detection"}
                
                # Sample 10 frames evenly distributed (or less if video is short)
                num_samples = min(10, total_frames)
                sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
                
                all_detections = []
                frame_detections = []
                
                for idx in sample_indices:
                    frame_path = f"{temp_dir}/{video_id}_frame_{idx}.jpg"
                    try:
                        result = await self.client.detect_objects(frame_path)
                        
                        if result.get("success"):
                            objects = result.get("objects", [])
                            # Add frame index to each detection
                            for obj in objects:
                                obj["frame_index"] = idx
                                obj["timestamp"] = idx  # Approximate timestamp
                            
                            all_detections.extend(objects)
                            frame_detections.append({
                                "frame_index": idx,
                                "timestamp": idx,
                                "objects_count": len(objects),
                                "objects": objects
                            })
                    except Exception as e:
                        logger.error(f"Error detecting objects in frame {idx}: {e}")
                
                # Aggregate object counts by class
                object_summary = {}
                for obj in all_detections:
                    obj_class = obj["class"]
                    if obj_class not in object_summary:
                        object_summary[obj_class] = {"count": 0, "max_confidence": 0.0}
                    object_summary[obj_class]["count"] += 1
                    object_summary[obj_class]["max_confidence"] = max(
                        object_summary[obj_class]["max_confidence"],
                        obj["confidence"]
                    )
                
                return {
                    "success": True,
                    "all_objects": all_detections,
                    "total_detected": len(all_detections),
                    "frames_analyzed": num_samples,
                    "frame_detections": frame_detections,
                    "object_summary": object_summary,
                    "model": "yolov8n"
                }
            
            elif tool == "extract_text":
                # Extract text from multiple frames for better coverage
                frames_list = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if total_frames == 0 or not frames_list:
                    # Fallback to single frame
                    frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                    return await self.client.extract_text(frame_path)
                
                # Sample 10 frames evenly distributed
                num_samples = min(10, total_frames)
                sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
                
                all_text = []
                frame_texts = []
                
                for idx in sample_indices:
                    frame_path = f"{temp_dir}/{video_id}_frame_{idx}.jpg"
                    try:
                        result = await self.client.extract_text(frame_path)
                        text = result.get("text", "")
                        
                        if text and len(text.strip()) > 0:
                            frame_texts.append({
                                "frame_index": idx,
                                "timestamp": idx,
                                "text": text
                            })
                            all_text.append(text)
                    except Exception as e:
                        logger.error(f"Error extracting text from frame {idx}: {e}")
                
                # Combine all text
                combined_text = "\n\n".join(all_text)
                
                return {
                    "success": True,
                    "text": combined_text,
                    "frames_analyzed": num_samples,
                    "frames_with_text": len(frame_texts),
                    "frame_texts": frame_texts
                }
        
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
            return await self._synthesize_visual_analysis(user_query, all_data)
        
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
        
        # Determine which format was requested (check for explicit format request)
        query_lower = user_query.lower()
        wants_ppt = "ppt" in query_lower or "powerpoint" in query_lower
        wants_pdf = "pdf" in query_lower or generate_pdf
        
        # Only generate the explicitly requested format
        # If both mentioned, prioritize the one mentioned first
        pdf_path = None
        ppt_path = None
        pdf_requested = False
        ppt_requested = False
        
        if wants_ppt and wants_pdf:
            # Both mentioned - check which comes first
            ppt_idx = min(query_lower.find("ppt"), query_lower.find("powerpoint")) if "ppt" in query_lower or "powerpoint" in query_lower else float('inf')
            pdf_idx = query_lower.find("pdf") if "pdf" in query_lower else float('inf')
            
            if ppt_idx < pdf_idx:
                ppt_requested = True
                ppt_path = await self._generate_ppt_report(structured_summary, data, cache)
            else:
                pdf_requested = True
                pdf_path = await self._generate_pdf_report(structured_summary, data, cache)
        elif wants_ppt:
            # Only PPT requested
            ppt_requested = True
            ppt_path = await self._generate_ppt_report(structured_summary, data, cache)
        elif wants_pdf:
            # Only PDF requested
            pdf_requested = True
            pdf_path = await self._generate_pdf_report(structured_summary, data, cache)
        
        # Format response - only show the path that was just generated
        return self._format_summary_response(structured_summary, pdf_path, ppt_path, pdf_requested, ppt_requested)
    
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
            result = await self.client.create_pdf_report(content, str(output_path))
            
            # Check if tool returned an error
            if isinstance(result, dict) and "error" in result:
                logger.error(f"❌ PDF report generation failed: {result['error']}")
                return None
            
            # Verify file was actually created
            if not output_path.exists():
                logger.error(f"❌ PDF report tool succeeded but file not found at: {output_path}")
                logger.error(f"Tool result: {result}")
                return None
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                logger.error(f"❌ PDF report created but file is empty: {output_path}")
                return None
                
            logger.info(f"✅ PDF report generated successfully: {output_path} ({file_size} bytes)")
            cache.set("pdf_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Failed to generate PDF report: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            result = await self.client.create_ppt_report(content, str(output_path))
            
            # Check if tool returned an error
            if isinstance(result, dict) and "error" in result:
                logger.error(f"❌ PPT report generation failed: {result['error']}")
                return None
            
            # Verify file was actually created
            if not output_path.exists():
                logger.error(f"❌ PPT report tool succeeded but file not found at: {output_path}")
                logger.error(f"Tool result: {result}")
                return None
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                logger.error(f"❌ PPT report created but file is empty: {output_path}")
                return None
                
            logger.info(f"✅ PPT report generated successfully: {output_path} ({file_size} bytes)")
            cache.set("ppt_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Failed to generate PPT report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _format_visual_summary(self, visual_data: Dict[str, Any]) -> str:
        """Format visual analysis for report inclusion with enhanced details"""
        if not isinstance(visual_data, dict):
            return ""
        
        frame_analyses = visual_data.get("frame_analyses", [])
        if not frame_analyses:
            return ""
        
        summary = f"Analyzed {len(frame_analyses)} frames with comprehensive visual analysis\n\n"
        
        # Include more detailed frame information for reports
        for i, frame in enumerate(frame_analyses[:10], 1):  # Include top 10 frames
            timestamp = frame.get('timestamp', 0)
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            description = frame.get('description', 'No description')
            objects = frame.get('objects', [])
            text_content = frame.get('text', '')
            
            summary += f"{i}. [{time_str}] {description}\n"
            
            if objects:
                summary += f"   Objects detected: {', '.join(objects[:8])}\n"
            
            if text_content:
                cleaned_text = self._clean_ocr_text(text_content)[:150]
                if cleaned_text:
                    summary += f"   On-screen text: \"{cleaned_text}\"\n"
            
            summary += "\n"
        
        if len(frame_analyses) > 10:
            summary += f"...and {len(frame_analyses) - 10} additional frames analyzed\n"
        
        return summary
    
    def _format_summary_response(self, summary: Dict[str, Any], pdf_path: Optional[str], ppt_path: Optional[str] = None, pdf_requested: bool = False, ppt_requested: bool = False) -> str:
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
        
        # Show available reports or error messages
        if pdf_path:
            response += f"📄 **Full PDF Report:** `{pdf_path}`\n"
        elif pdf_requested:
            response += f"❌ **PDF generation failed** - Please check the logs for details\n"
        
        if ppt_path:
            response += f"📊 **PowerPoint Presentation:** `{ppt_path}`\n"
        elif ppt_requested:
            response += f"❌ **PowerPoint generation failed** - Please check the logs for details\n"
        
        return response
    
    async def _synthesize_visual_analysis(self, user_query: str, data: Dict[str, Any]) -> str:
        """Return visual analysis results with LLM-generated insights that directly answer the user's question"""
        visual_data = data.get("visual_analysis", {})
        frame_analyses = visual_data.get("frame_analyses", []) if isinstance(visual_data, dict) else []
        
        if not frame_analyses:
            return "❌ No visual analysis available."
        
        # Prepare frame descriptions for LLM analysis
        frames_text = ""
        for i, frame in enumerate(frame_analyses[:15], 1):  # Use up to 15 frames
            timestamp = frame.get('timestamp', 0)
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            description = frame.get('description', 'No description')
            objects = frame.get('objects', [])
            text_content = frame.get('text', '')
            
            frames_text += f"{i}. [{time_str}] {description}"
            if objects:
                frames_text += f" | Objects: {', '.join(objects[:5])}"
            if text_content:
                cleaned = self._clean_ocr_text(text_content)[:100]
                if cleaned:
                    frames_text += f" | Text: \"{cleaned}\""
            frames_text += "\n"
        
        # Use LLM to generate comprehensive visual summary
        prompt = f"""You are analyzing a video to answer the user's question.

User's Question: "{user_query}"

Total frames analyzed: {len(frame_analyses)}

Frame descriptions:
{frames_text}

IMPORTANT: 
1. First, carefully read through ALL frame descriptions
2. Look for evidence that answers the user's question (e.g., if they ask about graphs/charts, look for mentions of "bar graph", "chart", "pie chart", "line graph", "data visualization", etc.)
3. Answer the question directly and honestly based on what you find
4. Make sure your Direct Answer is consistent with your Brief Summary and Key Visual Elements

Generate a response in this format:

**Direct Answer:**
(Answer the user's specific question directly based on visual evidence. Be specific - mention which frames if found, or clearly state if not found. Examples: "Yes, bar graphs are visible in frames 3 and 4" OR "No graphs or charts detected in the analyzed frames")

**Brief Summary:**
(2-3 sentences describing what the video shows overall - must be consistent with your Direct Answer)

**Key Visual Elements:**
(List 3-5 most important visual elements observed - must be consistent with your Direct Answer)

CRITICAL: Ensure your Direct Answer, Brief Summary, and Key Visual Elements are all logically consistent. If you say "No" in Direct Answer, don't mention the item in the summary."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a video analysis expert. Answer questions directly and accurately based on visual evidence. Always ensure your responses are internally consistent - if you say something exists in your answer, mention it in the summary, and vice versa. Read carefully before answering."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            llm_summary = response["message"]["content"]
            
            # Format final response - start with LLM analysis (includes direct answer)
            result = f"🎬 **Visual Analysis**\n\n"
            result += f"_Analyzed {len(frame_analyses)} frames_\n\n"
            result += llm_summary
            
            # Add all frame details in a compact format
            result += f"\n\n---\n\n"
            result += f"**📋 Frame-by-Frame Details:**\n\n"
            
            # Show all frames with compact formatting
            for i, frame in enumerate(frame_analyses, 1):
                timestamp = frame.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                description = frame.get('description', 'No description')
                
                # Compact format: Frame number + timestamp + description
                result += f"**{i}. [{time_str}]** {description}\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating visual summary with LLM: {e}")
            # Fallback to simple list
            response = f"🎬 **Visual Analysis ({len(frame_analyses)} frames):**\n\n"
            for frame in frame_analyses[:10]:
                timestamp = frame.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                response += f"**{time_str}** - {frame.get('description', 'No description')}\n\n"
            return response
    
    async def _synthesize_objects(self, data: Dict[str, Any]) -> str:
        """Return object detection results with LLM-generated summary"""
        objects_data = data.get("objects", {})
        
        # Check if we have object detection data
        if not objects_data or not isinstance(objects_data, dict):
            return "❌ No object detection data available."
        
        # Handle new multi-frame format
        all_objects = objects_data.get("all_objects", [])
        object_summary = objects_data.get("object_summary", {})
        frames_analyzed = objects_data.get("frames_analyzed", 1)
        frame_detections = objects_data.get("frame_detections", [])
        
        # Fallback to old single-frame format
        if not all_objects:
            all_objects = objects_data.get("objects", [])
        
        if not all_objects:
            return "❌ No objects detected in video."
        
        # Prepare data for LLM analysis
        sorted_summary = sorted(object_summary.items(), key=lambda x: x[1]["count"], reverse=True) if object_summary else []
        
        # Build object summary text for LLM
        objects_text = ""
        for obj_class, obj_data in sorted_summary:
            count = obj_data["count"]
            max_conf = obj_data["max_confidence"] * 100
            objects_text += f"- {obj_class}: {count} instances (max confidence: {max_conf:.1f}%)\n"
        
        # Get frame distribution info
        frame_distribution = ""
        for frame_det in frame_detections[:5]:  # First 5 frames
            frame_idx = frame_det.get("frame_index", "?")
            obj_count = frame_det.get("objects_count", 0)
            frame_objs = frame_det.get("objects", [])
            obj_types = list(set([o.get("class", "unknown") for o in frame_objs]))
            frame_distribution += f"Frame {frame_idx}: {', '.join(obj_types[:3])}\n"
        
        # Use LLM to generate intelligent summary
        prompt = f"""Analyze these object detection results from a video and provide a natural, insightful summary.

Detection Data:
- Total detections: {len(all_objects)}
- Frames analyzed: {frames_analyzed}
- Unique object types: {len(object_summary)}

Objects found:
{objects_text}

Sample frame distribution:
{frame_distribution}

Generate a response with these sections:

**Summary:**
(2-3 sentences describing what the video shows based on detected objects)

**Objects Detected:**
(List only the most significant/relevant objects, not all. Group related items if appropriate)

**Observations:**
(1-2 insightful observations about the scene, patterns, or context)

Keep it concise, natural, and focus on what matters most."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a video analysis expert. Provide clear, natural summaries of object detection results."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            llm_summary = response["message"]["content"]
            
            # Add technical details at the end
            result = f"🔍 **Object Detection Analysis**\n\n"
            result += f"_Analyzed {frames_analyzed} frames • {len(all_objects)} total detections • {len(object_summary)} unique object types_\n\n"
            result += llm_summary
            result += f"\n\n---\n\n"
            result += f"**Technical Details:**\n"
            result += f"• Most common: {sorted_summary[0][0]} ({sorted_summary[0][1]['count']} instances)\n" if sorted_summary else ""
            result += f"• Detection confidence: {sorted_summary[0][1]['max_confidence']*100:.1f}% average\n" if sorted_summary else ""
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating object summary with LLM: {e}")
            # Fallback to simple list if LLM fails
            response = f"🔍 **Detected Objects ({len(all_objects)} total):**\n\n"
            response += f"_Analyzed {frames_analyzed} frames_\n\n"
            for obj_class, obj_data in sorted_summary[:10]:
                count = obj_data["count"]
                response += f"• {obj_class}: {count} instance{'s' if count > 1 else ''}\n"
            return response
    
    async def _synthesize_text(self, data: Dict[str, Any]) -> str:
        """Return extracted text with LLM-generated analysis"""
        text_data = data.get("text", {})
        
        # Handle different data formats
        if isinstance(text_data, dict):
            extracted_text = text_data.get("text", "")
            frames_analyzed = text_data.get("frames_analyzed", 1)
            frames_with_text = text_data.get("frames_with_text", 0)
        else:
            extracted_text = str(text_data) if text_data else ""
            frames_analyzed = 1
            frames_with_text = 1 if extracted_text else 0
        
        # Clean the text
        cleaned_text = self._clean_ocr_text(extracted_text)
        
        if not cleaned_text or len(cleaned_text.strip()) < 5:
            return f"❌ No text detected in video.\n\n_Analyzed {frames_analyzed} frames, found text in {frames_with_text} frames_"
        
        # Use LLM to analyze and format the extracted text
        prompt = f"""Analyze this text extracted from a video via OCR and provide insights.

Note: This text has been cleaned of OCR artifacts and noise. If the remaining text seems fragmented or unclear, it may indicate poor OCR quality or complex visual content.

Extracted Text:
{cleaned_text[:2000]}

Generate a response with these sections:

**Text Summary:**
(1-2 sentences describing what type of content this text represents - e.g., presentation slides, subtitles, on-screen text, document, etc.)

**Key Information:**
(List the most important information found in the text - main points, titles, key terms. If the text is too fragmented to extract meaningful information, say so.)

**Content Type:**
(Identify the likely source/context - presentation, tutorial, interview, advertisement, etc.)

**OCR Quality Note:**
(If the text appears heavily fragmented or contains many errors, briefly note this. Otherwise, skip this section.)

Keep it concise and practical. If the OCR quality is very poor and text is meaningless, be honest about it."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are an OCR text analysis expert. Provide clear, actionable insights from extracted text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            llm_analysis = response["message"]["content"]
            
            # Format response
            result = f"📄 **Extracted Text Analysis**\n\n"
            result += f"_Analyzed {frames_analyzed} frames • Found text in {frames_with_text} frames • {len(cleaned_text)} characters total_\n\n"
            result += llm_analysis
            result += f"\n\n---\n\n"
            result += f"**📝 Raw Extracted Text:**\n\n"
            
            # Show text with smart truncation
            if len(cleaned_text) <= 1000:
                result += f"```\n{cleaned_text}\n```\n"
            else:
                result += f"```\n{cleaned_text[:800]}\n```\n"
                result += f"\n_...and {len(cleaned_text) - 800} more characters. Ask 'Show full extracted text' to see complete text._\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing extracted text with LLM: {e}")
            # Fallback to simple display
            return f"📄 **Extracted Text:**\n\n{cleaned_text[:1000]}"
    
    async def _synthesize_chat(self, user_query: str, data: Dict[str, Any]) -> str:
        """Generate chat response using available data"""
        # Build comprehensive context from all available data
        context_parts = []
        
        # Add transcription if available
        transcription_data = data.get("transcription", {})
        if transcription_data:
            transcript_text = self._format_transcript(transcription_data)
            if transcript_text:
                context_parts.append(f"**Video Transcript:**\n{transcript_text[:1500]}")
        
        # Add visual analysis if available
        visual_data = data.get("visual_analysis", {})
        if visual_data:
            frame_analyses = visual_data.get("frame_analyses", [])
            if frame_analyses:
                visual_summary = "**Visual Content:**\n"
                for frame in frame_analyses[:8]:
                    timestamp = frame.get('timestamp', 0)
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    visual_summary += f"[{minutes:02d}:{seconds:02d}] {frame.get('description', '')}; "
                context_parts.append(visual_summary[:800])
        
        # Add object detection if available
        objects_data = data.get("objects", {})
        if objects_data:
            object_summary = objects_data.get("object_summary", {})
            if object_summary:
                objects_text = "**Detected Objects:** "
                sorted_objs = sorted(object_summary.items(), key=lambda x: x[1]["count"], reverse=True)
                objects_text += ", ".join([f"{obj}({count['count']})" for obj, count in sorted_objs[:10]])
                context_parts.append(objects_text)
        
        # Add extracted text if available
        text_data = data.get("text", {})
        if text_data:
            extracted = text_data.get("text", "")
            if extracted:
                cleaned = self._clean_ocr_text(extracted)[:500]
                if cleaned:
                    context_parts.append(f"**On-screen Text:** {cleaned}")
        
        context = f"""User Question: "{user_query}"

Video Analysis Data:
{chr(10).join(context_parts)}

Answer the user's question naturally based on the available video analysis data. Be specific and cite timestamps when relevant. If the data doesn't contain enough information to answer, say so honestly."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a helpful video analysis assistant. Answer questions about videos based on transcripts, visual analysis, and detected objects. Be conversational but accurate."},
                    {"role": "user", "content": context}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I encountered an error while analyzing the video data. Please try rephrasing your question or ask for a different type of analysis."
    
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
