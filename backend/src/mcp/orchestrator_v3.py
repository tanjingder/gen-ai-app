"""
Orchestrator V3 - LLM-Driven Video Analysis
============================================

Flow:
1. Interpret Intent - What is the user asking?
2. Retrieve Context - What data do we already have?
3. Plan Actions - Which tools should we call?
4. Execute Tools - Get fresh data from agents
5. Fuse Results - Combine all available data
6. Reason Freely - Let LLM think without constraints
7. Generate Response - Format appropriately

Key Principles:
- Minimal rigid prompts
- Let LLM reason freely on data
- Only constrain format at the final step
- Use layered prompting strategy
"""

import json
import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import ollama
from loguru import logger

from ..utils.config import settings
from .client import mcp_client


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


class MCPOrchestratorV3:
    """
    LLM-driven orchestrator with flexible reasoning
    """
    
    def __init__(self):
        self.client = mcp_client
        self.ollama_model = settings.OLLAMA_MODEL
        
        # System identity - defines WHO the LLM is
        self.system_identity = """You are an intelligent video analysis orchestrator.

Your role is to:
- Understand what users want to know about videos
- Coordinate multiple analysis tools (transcription, visual analysis, object detection, OCR)
- Reason about the results and provide insightful answers
- Adapt your response style to the user's needs

You have access to session memory and can leverage previously analyzed data."""
    
    # ============================================================
    # LAYER 1: INTENT INTERPRETATION
    # ============================================================
    
    async def interpret_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Layer 1: Let LLM freely interpret what the user wants
        
        Returns:
            {
                "user_wants": "natural language description",
                "required_data": ["transcription", "visual_analysis", "objects", "text"],
                "output_preference": "text" | "report" | "structured"
            }
        """
        prompt = f"""Analyze this user query about a video:

User Query: "{user_query}"

Think about:
1. What is the user asking for?
2. What data would you need to answer this question?
3. Do they want a report/document, or just an answer?

Return JSON with your interpretation:
{{
  "user_wants": "describe what they want in plain English",
  "required_data": ["array of data types needed"],
  "output_preference": "text|report|structured"
}}

Available data types:
- "transcription" - Audio/speech content
- "visual_analysis" - What's shown in the video frames
- "objects" - Specific objects detected
- "text" - Text extracted via OCR
- "charts" - Charts and graphs analysis

Examples:
- "What's shown in the video?" → required_data: ["visual_analysis"]
- "What is being said?" → required_data: ["transcription"]
- "Summarize this video" → required_data: ["transcription", "visual_analysis"]
- "Are there any graphs?" → required_data: ["charts"]
- "Analyze the charts" → required_data: ["charts"]
- "What do the graphs show?" → required_data: ["charts"]
- "How many people appear?" → required_data: ["objects"]
- "What text is shown?" → required_data: ["text"]
- "Tell me about the presentation" → required_data: ["transcription", "visual_analysis", "text"]

Examples of output_preference:
- "Create a PDF report" → report
- "Make a PowerPoint" → report
- "Tell me about..." → text
- "List all..." → structured
- Most questions → text (default)

Return ONLY valid JSON."""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": self.system_identity},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            intent = json.loads(response["message"]["content"])
            logger.info(f"🎯 Intent: {intent.get('user_wants')}")
            logger.info(f"📊 Required Data: {intent.get('required_data')} | Output: {intent.get('output_preference')}")
            
            return intent
            
        except Exception as e:
            logger.error(f"Error interpreting intent: {e}")
            return {
                "user_wants": "answer question about video",
                "required_data": ["transcription", "visual_analysis"],
                "output_preference": "text"
            }
    
    # ============================================================
    # LAYER 2: RETRIEVE CONTEXT
    # ============================================================
    
    def retrieve_context(self, cache: SessionCache, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Layer 2: Check what data we already have
        
        Returns:
            {
                "has_transcription": true/false,
                "has_frames": true/false,
                "has_visual_analysis": true/false,
                "has_objects": true/false,
                "has_text": true/false,
                "cached_data": {...}
            }
        """
        context = {
            "has_transcription": cache.has("transcription"),
            "has_frames": cache.has("frames"),
            "has_visual_analysis": cache.has("visual_analysis"),
            "has_objects": cache.has("objects"),
            "has_text": cache.has("extracted_text"),
            "has_charts": cache.has("chart_analysis"),
            "cached_data": {}
        }
        
        # Load cached data
        if context["has_transcription"]:
            context["cached_data"]["transcription"] = cache.get("transcription")
        if context["has_frames"]:
            context["cached_data"]["frames"] = cache.get("frames")
        if context["has_visual_analysis"]:
            context["cached_data"]["visual_analysis"] = cache.get("visual_analysis")
        if context["has_objects"]:
            context["cached_data"]["objects"] = cache.get("objects")
        if context["has_text"]:
            context["cached_data"]["text"] = cache.get("extracted_text")
        if context["has_charts"]:
            context["cached_data"]["charts"] = cache.get("chart_analysis")
        
        # Check what's required vs what's available
        required_data = intent.get('required_data', [])
        
        # Map required data to cache keys
        cache_mapping = {
            "transcription": "has_transcription",
            "visual_analysis": "has_visual_analysis",
            "objects": "has_objects",
            "text": "has_text"
        }
        
        # Determine which required items are cached vs missing
        available_required = []
        missing_required = []
        
        for data_type in required_data:
            cache_key = cache_mapping.get(data_type)
            if cache_key:
                if context.get(cache_key, False):
                    available_required.append(data_type)
                else:
                    missing_required.append(data_type)
        
        # Log status focused on what's required
        logger.info(f"📦 Context: {len(available_required)} required cached, {len(missing_required)} required missing")
        if available_required:
            logger.info(f"  ✅ Available: {available_required}")
        if missing_required:
            logger.info(f"  ❌ Missing: {missing_required}")
        
        return context
    
    # ============================================================
    # LAYER 3: PLAN ACTIONS
    # ============================================================
    
    async def plan_actions(
        self,
        user_query: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Layer 3: Determine which tools to call based on required data and what's cached
        
        Returns:
            [
                {"agent": "vision", "tool": "extract_frames", "reason": "Need frames for visual analysis"},
                {"agent": "vision", "tool": "analyze_frame", "reason": "User wants to know what's shown"},
                ...
            ]
        """
        required_data = intent.get('required_data', [])
        
        # Map required data to cache keys
        cache_mapping = {
            "transcription": "has_transcription",
            "visual_analysis": "has_visual_analysis",
            "objects": "has_objects",
            "text": "has_text",
            "charts": "has_charts"
        }
        
        # Determine what's missing
        missing_data = []
        for data_type in required_data:
            cache_key = cache_mapping.get(data_type)
            if cache_key and not context.get(cache_key, False):
                missing_data.append(data_type)
        
        # If everything is cached, no tools needed
        if not missing_data:
            logger.info("📋 Action Plan: All required data cached, no tools needed")
            return []
        
        # Map missing data to tool sequences
        tools_plan = []
        
        # Check if we need frames (required for visual analysis, objects, text extraction, or charts)
        needs_frames = any(d in missing_data for d in ["visual_analysis", "objects", "text", "charts"])
        has_frames = context.get("has_frames", False)
        
        if needs_frames and not has_frames:
            tools_plan.append({
                "agent": "vision",
                "tool": "extract_frames",
                "reason": "Extract frames for visual analysis"
            })
        
        # Add tools for each missing data type
        if "transcription" in missing_data:
            tools_plan.extend([
                {
                    "agent": "transcription",
                    "tool": "extract_audio",
                    "reason": "Extract audio for transcription"
                },
                {
                    "agent": "transcription",
                    "tool": "transcribe_audio",
                    "reason": "Convert audio to text"
                }
            ])
        
        if "visual_analysis" in missing_data:
            tools_plan.append({
                "agent": "vision",
                "tool": "analyze_frame",
                "reason": "Analyze visual content of frames"
            })
        
        if "objects" in missing_data:
            tools_plan.append({
                "agent": "vision",
                "tool": "detect_objects",
                "reason": "Detect and count objects in frames"
            })
        
        if "text" in missing_data:
            tools_plan.append({
                "agent": "vision",
                "tool": "extract_text",
                "reason": "Extract text via OCR"
            })
        
        if "charts" in missing_data:
            tools_plan.append({
                "agent": "vision",
                "tool": "analyze_chart",
                "reason": "Analyze charts and graphs"
            })
        
        logger.info(f"📋 Action Plan: {len(tools_plan)} tools to execute for missing data: {missing_data}")
        for step in tools_plan:
            logger.info(f"  → {step.get('agent')}.{step.get('tool')}: {step.get('reason')}")
        
        return tools_plan
    
    # ============================================================
    # LAYER 4: EXECUTE TOOLS
    # ============================================================
    
    async def execute_tools(
        self,
        tools_plan: List[Dict[str, str]],
        video_id: str,
        video_path: str,
        cache: SessionCache,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Layer 4: Execute planned tools and cache results
        
        Returns fresh data collected from tools
        """
        if not tools_plan:
            logger.info("⏭️ No tools to execute")
            return {}
        
        results = {}
        temp_dir = cache.base_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Track intermediate data for dependent tools
        intermediate_data = {}
        
        # Load cached frames if available
        cached_frames = context.get("cached_data", {}).get("frames")
        if cached_frames:
            intermediate_data["frames"] = cached_frames.get("frames", [])
            intermediate_data["total_frames"] = cached_frames.get("total_frames", 0)
        
        for tool_spec in tools_plan:
            agent = tool_spec.get("agent")
            tool = tool_spec.get("tool")
            reason = tool_spec.get("reason", "")
            
            logger.info(f"⚙️ Executing: {agent}.{tool}")
            if reason:
                logger.info(f"   Reason: {reason}")
            
            try:
                result = await self._execute_single_tool(
                    agent, tool, video_id, video_path,
                    str(temp_dir), intermediate_data
                )
                
                # Cache and store results
                if tool == "transcribe_audio":
                    cache.set("transcription", result)
                    results["transcription"] = result
                    intermediate_data["transcription"] = result.get("text", "")
                
                elif tool == "extract_frames":
                    cache.set("frames", result)
                    results["frames"] = result
                    intermediate_data["frames"] = result.get("frames", [])
                    intermediate_data["total_frames"] = result.get("total_frames", 0)
                
                elif tool == "analyze_frame":
                    cache.set("visual_analysis", result)
                    results["visual_analysis"] = result
                
                elif tool == "detect_objects":
                    cache.set("objects", result)
                    results["objects"] = result
                
                elif tool == "extract_text":
                    cache.set("extracted_text", result)
                    results["text"] = result
                
                elif tool == "analyze_chart":
                    cache.set("chart_analysis", result)
                    results["charts"] = result
                
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
                # Multi-frame analysis
                frames_data = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if not frames_data:
                    # Fallback to single frame
                    frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                    return await self.client.analyze_frame(frame_path)
                
                # Sample frames intelligently
                sample_rate = max(1, total_frames // 8) if total_frames > 8 else 1
                frame_analyses = []
                
                for i, frame_info in enumerate(frames_data):
                    if i % sample_rate == 0:
                        frame_path = frame_info.get("path")
                        if frame_path:
                            try:
                                analysis = await self.client.analyze_frame(frame_path)
                                frame_analyses.append({
                                    "frame": i,
                                    "timestamp": frame_info.get("timestamp", 0.0),
                                    "description": analysis.get("description", ""),
                                    "confidence": analysis.get("confidence", 0.0)
                                })
                            except Exception as e:
                                logger.error(f"Error analyzing frame {i}: {e}")
                
                return {
                    "frame_analyses": frame_analyses,
                    "total_analyzed": len(frame_analyses),
                    "sample_rate": sample_rate
                }
            
            elif tool == "detect_objects":
                # Multi-frame object detection
                frames_list = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if total_frames == 0:
                    return {"error": "No frames available"}
                
                num_samples = min(10, total_frames)
                sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
                
                all_detections = []
                object_counts = {}
                
                for idx in sample_indices:
                    frame_path = f"{temp_dir}/{video_id}_frame_{idx}.jpg"
                    try:
                        result = await self.client.detect_objects(frame_path)
                        objects = result.get("objects", [])
                        
                        for obj in objects:
                            obj_class = obj.get("class")
                            if obj_class not in object_counts:
                                object_counts[obj_class] = 0
                            object_counts[obj_class] += 1
                        
                        all_detections.extend(objects)
                    except Exception as e:
                        logger.error(f"Error detecting objects in frame {idx}: {e}")
                
                return {
                    "all_objects": all_detections,
                    "object_counts": object_counts,
                    "frames_analyzed": num_samples
                }
            
            elif tool == "extract_text":
                # Multi-frame text extraction
                frames_list = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if total_frames == 0:
                    frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                    return await self.client.extract_text(frame_path)
                
                num_samples = min(10, total_frames)
                sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
                
                all_text = []
                for idx in sample_indices:
                    frame_path = f"{temp_dir}/{video_id}_frame_{idx}.jpg"
                    try:
                        result = await self.client.extract_text(frame_path)
                        text = result.get("text", "")
                        if text.strip():
                            all_text.append(text)
                    except Exception as e:
                        logger.error(f"Error extracting text from frame {idx}: {e}")
                
                return {
                    "text": "\n\n".join(all_text),
                    "frames_analyzed": num_samples
                }
            
            elif tool == "analyze_chart":
                # Multi-frame chart analysis
                frames_list = intermediate_data.get("frames", [])
                total_frames = intermediate_data.get("total_frames", 0)
                
                if total_frames == 0:
                    return {"error": "No frames available"}
                
                num_samples = min(10, total_frames)
                sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
                
                logger.info(f"🔍 Chart Analysis: Sampling {num_samples} frames from {total_frames} total frames")
                logger.info(f"📊 Frame indices to analyze: {sample_indices}")
                
                chart_analyses = []
                
                for idx in sample_indices:
                    frame_path = f"{temp_dir}/{video_id}_frame_{idx}.jpg"
                    logger.info(f"🖼️  Analyzing frame {idx}: {frame_path}")
                    
                    try:
                        result = await self.client.analyze_chart(frame_path)
                        has_chart = result.get("has_chart", False)
                        
                        logger.info(f"   {'✅ Chart detected!' if has_chart else '❌ No chart'} - Frame {idx}")
                        
                        if has_chart:
                            chart_type = result.get("chart_type", "unknown")
                            logger.info(f"   📈 Chart type: {chart_type}")
                            chart_analyses.append({
                                "frame": idx,
                                "chart_type": chart_type,
                                "description": result.get("description", ""),
                                "data_points": result.get("data_points", []),
                                "insights": result.get("insights", "")
                            })
                    except Exception as e:
                        logger.error(f"❌ Error analyzing chart in frame {idx}: {e}")
                
                return {
                    "chart_analyses": chart_analyses,
                    "frames_analyzed": num_samples,
                    "charts_found": len(chart_analyses)
                }
        
        return {"error": f"Unknown tool: {agent}.{tool}"}
    
    # ============================================================
    # LAYER 5: FUSE RESULTS
    # ============================================================
    
    def fuse_results(
        self,
        cached_data: Dict[str, Any],
        fresh_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Layer 5: Combine all available data into unified context
        
        Returns complete dataset for reasoning
        """
        fused = {**cached_data, **fresh_data}
        
        # Count what we have
        data_types = []
        if fused.get("transcription"):
            data_types.append("transcription")
        if fused.get("visual_analysis"):
            data_types.append("visual_analysis")
        if fused.get("objects"):
            data_types.append("objects")
        if fused.get("text"):
            data_types.append("text")
        if fused.get("charts"):
            data_types.append("charts")
        if fused.get("text"):
            data_types.append("extracted_text")
        
        logger.info(f"🔗 Fused Data: {len(data_types)} types available: {data_types}")
        
        return fused
    
    # ============================================================
    # LAYER 6: REASON FREELY
    # ============================================================
    
    async def reason_freely(
        self,
        user_query: str,
        intent: Dict[str, Any],
        fused_data: Dict[str, Any]
    ) -> str:
        """
        Layer 6: Let LLM reason freely about all data WITHOUT format constraints
        
        This is where the magic happens - no rigid structure, just thinking
        """
        # Build context for LLM
        context_parts = []
        
        # Transcription
        transcription = fused_data.get("transcription", {})
        if transcription:
            text = transcription.get("text", "")
            if text:
                # Clean timestamps
                text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]', '', text)
                text = ' '.join(text.split())
                context_parts.append(f"**Audio Transcription:**\n{text[:3000]}")
        
        # Visual Analysis
        visual = fused_data.get("visual_analysis", {})
        if visual:
            frame_analyses = visual.get("frame_analyses", [])
            if frame_analyses:
                context_parts.append(f"\n**Visual Analysis ({len(frame_analyses)} frames):**")
                for i, frame in enumerate(frame_analyses[:15], 1):
                    timestamp = frame.get("timestamp", 0)
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    desc = frame.get("description", "")
                    context_parts.append(f"{i}. [{minutes:02d}:{seconds:02d}] {desc}")
        
        # Object Detection
        objects = fused_data.get("objects", {})
        if objects:
            object_counts = objects.get("object_counts", {})
            if object_counts:
                context_parts.append(f"\n**Objects Detected:**")
                sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
                for obj_class, count in sorted_objects[:15]:
                    context_parts.append(f"- {obj_class}: {count}")
        
        # Extracted Text
        extracted = fused_data.get("text", {})
        if extracted:
            text = extracted.get("text", "")
            if text and len(text.strip()) > 5:
                # Clean OCR artifacts
                text = self._clean_ocr_text(text)
                context_parts.append(f"\n**Text Extracted (OCR):**\n{text[:1000]}")
        
        # Chart Analysis
        charts = fused_data.get("charts", {})
        if charts:
            chart_analyses = charts.get("chart_analyses", [])
            frames_analyzed = charts.get("frames_analyzed", 0)
            charts_found = charts.get("charts_found", 0)
            
            logger.info(f"Chart analysis: {charts_found} charts found in {frames_analyzed} frames")
            
            if chart_analyses:
                context_parts.append(f"\n**Chart Analysis ({len(chart_analyses)} charts found):**")
                for i, chart in enumerate(chart_analyses, 1):
                    chart_type = chart.get("chart_type", "unknown")
                    desc = chart.get("description", "")
                    insights = chart.get("insights", "")
                    context_parts.append(f"{i}. {chart_type.upper()}: {desc}")
                    if insights:
                        context_parts.append(f"   Insights: {insights}")
            else:
                # No charts detected - still provide this info to the LLM
                context_parts.append(f"\n**Chart Analysis:** Analyzed {frames_analyzed} frames. No charts, graphs, or data visualizations were detected in the video.")
                logger.warning(f"No charts detected despite analyzing {frames_analyzed} frames")
        
        # Build full context
        full_context = "\n".join(context_parts)
        
        if not full_context:
            return "I don't have enough data to answer that question. Please try analyzing the video first."
        
        # Free reasoning prompt
        prompt = f"""You are analyzing a video to answer the user's question.

User's Question: "{user_query}"

What they want: {intent.get('user_wants')}

Available Data:
{full_context}

---

Now, think freely and deeply about this data. Consider:
1. What patterns do you see?
2. What's the main message or content?
3. How does this answer the user's question?
4. What insights can you provide?

Write your analysis naturally - don't force a structure. Think out loud, explore connections, be insightful.
If the user asked a specific question, answer it directly and thoroughly.
If they want a summary, provide a comprehensive understanding of the video.

Your response:"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_identity + "\n\nIMPORTANT: Reason freely and naturally. Don't force rigid structures unless specifically asked. Be insightful and thorough."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            reasoning = response["message"]["content"]
            logger.info(f"💭 Free reasoning complete ({len(reasoning)} chars)")
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in free reasoning: {e}")
            return "I encountered an error while analyzing the video data."
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text artifacts"""
        if not text:
            return ""
        
        # Remove common encoding artifacts
        replacements = {
            'â€˜': "'", 'â€™': "'", 'â€œ': '"', 'â€': '"',
            'â€"': '-', 'â€"': '-', 'Â': '', 'â€¦': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove OCR noise patterns
        text = re.sub(r'(\d[^\w\s]){3,}', ' ', text)
        text = re.sub(r'([A-Z])\1{2,}', ' ', text)
        text = re.sub(r'(\d)\1{2,}', ' ', text)
        
        # Remove non-printable chars
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    # ============================================================
    # LAYER 7: GENERATE RESPONSE
    # ============================================================
    
    async def generate_response(
        self,
        user_query: str,
        reasoning: str,
        intent: Dict[str, Any],
        fused_data: Dict[str, Any],
        cache: SessionCache
    ) -> str:
        """
        Layer 7: Format response based on output preference
        
        This is the ONLY layer where we apply formatting constraints
        """
        output_pref = intent.get("output_preference", "text")
        
        # For text responses, return reasoning as-is (maybe with light formatting)
        if output_pref == "text":
            return self._format_text_response(reasoning, fused_data)
        
        # For reports, generate structured document
        elif output_pref == "report":
            return await self._generate_report(user_query, reasoning, fused_data, cache)
        
        # For structured output
        elif output_pref == "structured":
            return await self._format_structured(reasoning, fused_data)
        
        else:
            return reasoning
    
    def _format_text_response(self, reasoning: str, data: Dict[str, Any]) -> str:
        """Light formatting for text responses"""
        response = f"{reasoning}\n\n"
        
        # Add data source indicators
        sources = []
        if data.get("transcription"):
            sources.append("📝 Transcription")
        if data.get("visual_analysis"):
            sources.append("🎬 Visual Analysis")
        if data.get("objects"):
            sources.append("🔍 Object Detection")
        if data.get("text"):
            sources.append("📄 Text Extraction")
        
        if sources:
            response += f"---\n_Analysis based on: {', '.join(sources)}_"
        
        return response
    
    async def _generate_report(
        self,
        user_query: str,
        reasoning: str,
        data: Dict[str, Any],
        cache: SessionCache
    ) -> str:
        """Generate PDF or PPT report"""
        query_lower = user_query.lower()
        wants_ppt = "powerpoint" in query_lower or "ppt" in query_lower
        wants_pdf = "pdf" in query_lower
        
        # Ask LLM to structure the reasoning for a report
        prompt = f"""Convert this analysis into a structured report format.

Analysis:
{reasoning}

Create a JSON structure suitable for a professional report:
{{
  "title": "short descriptive title (5-10 words)",
  "summary": "executive summary (3-5 sentences)",
  "key_points": ["point 1", "point 2", ...],
  "details": "detailed analysis",
  "takeaways": ["insight 1", "insight 2", ...]
}}"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You structure analysis into professional reports."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            structured = json.loads(response["message"]["content"])
            
            # Build report content
            transcription = data.get("transcription", {})
            transcript_text = transcription.get("text", "") if transcription else ""
            if transcript_text:
                transcript_text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]', '', transcript_text)
                transcript_text = ' '.join(transcript_text.split())
            
            visual = data.get("visual_analysis", {})
            visual_summary = ""
            if visual:
                frame_analyses = visual.get("frame_analyses", [])
                if frame_analyses:
                    visual_summary = f"Analyzed {len(frame_analyses)} frames\n\n"
                    for frame in frame_analyses[:10]:
                        ts = frame.get("timestamp", 0)
                        mins = int(ts // 60)
                        secs = int(ts % 60)
                        visual_summary += f"[{mins:02d}:{secs:02d}] {frame.get('description', '')}\n"
            
            content = {
                "title": structured.get("title", "Video Analysis Report"),
                "generated_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                "summary": structured.get("summary", reasoning[:500]),
                "key_moments": structured.get("key_points", []),
                "visual_analysis": visual_summary,
                "transcript": transcript_text,
                "takeaways": structured.get("takeaways", [])
            }
            
            # Generate filename
            title = structured.get("title", "Video_Analysis")
            safe_filename = re.sub(r'[<>:"/\\|?*]', '', title)[:100].strip()
            if not safe_filename:
                safe_filename = "Video_Analysis_Report"
            
            output_path = None
            
            if wants_ppt:
                output_path = cache.base_path / "reports" / f"{safe_filename}.pptx"
                output_path.parent.mkdir(exist_ok=True)
                result = await self.client.create_ppt_report(content, str(output_path))
                if output_path.exists():
                    cache.set("ppt_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
                    # Return both reasoning and file metadata in a parseable format
                    file_info = json.dumps({
                        "filename": output_path.name,
                        "file_path": str(output_path.absolute()),
                        "file_type": "pptx",
                        "file_size": output_path.stat().st_size
                    })
                    return f"{reasoning}\n\n---\n\n📊 **PowerPoint Presentation Generated**\n\n<!-- FILE_ATTACHMENT: {file_info} -->"
            
            else:  # Default to PDF
                output_path = cache.base_path / "reports" / f"{safe_filename}.pdf"
                output_path.parent.mkdir(exist_ok=True)
                result = await self.client.create_pdf_report(content, str(output_path))
                if output_path.exists():
                    cache.set("pdf_report", {"path": str(output_path), "generated_at": datetime.now().isoformat()})
                    # Return both reasoning and file metadata in a parseable format
                    file_info = json.dumps({
                        "filename": output_path.name,
                        "file_path": str(output_path.absolute()),
                        "file_type": "pdf",
                        "file_size": output_path.stat().st_size
                    })
                    return f"{reasoning}\n\n---\n\n📄 **PDF Report Generated**\n\n<!-- FILE_ATTACHMENT: {file_info} -->"
            
            return f"{reasoning}\n\n❌ Report generation failed - please check logs"
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"{reasoning}\n\n❌ Report generation failed: {str(e)}"
    
    async def _format_structured(self, reasoning: str, data: Dict[str, Any]) -> str:
        """Format as structured JSON"""
        return json.dumps({
            "analysis": reasoning,
            "data_sources": list(data.keys()),
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    
    # ============================================================
    # MAIN ORCHESTRATION FLOW
    # ============================================================
    
    async def process_query(
        self,
        user_query: str,
        video_id: str,
        video_path: str,
        session_id: str
    ) -> str:
        """
        Main orchestration flow - all 7 layers
        """
        logger.info(f"🎬 Starting orchestration for: {user_query}")
        
        cache = SessionCache(session_id)
        
        try:
            # Layer 1: Interpret Intent
            intent = await self.interpret_intent(user_query)
            
            # Layer 2: Retrieve Context
            context = self.retrieve_context(cache, intent)
            
            # Layer 3: Plan Actions
            action_plan = await self.plan_actions(user_query, intent, context)
            
            # Layer 4: Execute Tools
            fresh_data = await self.execute_tools(
                action_plan, video_id, video_path, cache, context
            )
            
            # Layer 5: Fuse Results
            fused_data = self.fuse_results(context.get("cached_data", {}), fresh_data)
            
            # Layer 6: Reason Freely
            reasoning = await self.reason_freely(user_query, intent, fused_data)
            
            # Layer 7: Generate Response
            final_response = await self.generate_response(
                user_query, reasoning, intent, fused_data, cache
            )
            
            logger.info("✅ Orchestration complete")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Orchestration error: {e}")
            logger.error(traceback.format_exc())
            return f"I encountered an error while processing your request: {str(e)}"


# Create singleton instance
orchestrator_v3 = MCPOrchestratorV3()
