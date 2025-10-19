"""
MCP Orchestrator for coordinating agent actions
"""
from typing import Any, Dict, List, Optional
import json
from datetime import datetime
from loguru import logger
import ollama

from .client import mcp_client
from ..utils.config import settings


class MCPOrchestrator:
    """
    Orchestrates MCP agents using Ollama as the planning LLM
    """
    
    def __init__(self):
        self.client = mcp_client
        self.ollama_model = settings.OLLAMA_MODEL
    
    async def get_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available tools from all MCP servers"""
        tools = {
            "transcription": await self.client.list_tools("transcription"),
            "vision": await self.client.list_tools("vision"),
            "report": await self.client.list_tools("report"),
        }
        return tools
    
    def _create_system_prompt(self, tools: Dict[str, List[Dict[str, Any]]],  cached_tools: Optional[List[str]] = None) -> str:
        """
        Create system prompt with available tools
        
        Args:
            tools: Available tools from MCP servers
            cached_tools: List of tool names that have cached results
        """
        cache_info = ""
        if cached_tools:
            cached_tools_str = ', '.join(cached_tools)
            cache_info = """
**IMPORTANT - CACHED RESULTS AVAILABLE:**
The following tools have already been run and their results are cached:
""" + cached_tools_str + """

DO NOT run these tools again unless the user explicitly asks to re-analyze.
Use the cached results instead to save time and resources!
"""
        
        prompt = cache_info + """
You are an AI assistant specialized in analyzing video content. You have access to the following agents and their tools:

**Transcription Agent** - For audio and speech analysis:
- extract_audio: Extract audio track from video
- transcribe_audio: Convert speech to text with timestamps

**Vision Agent** - For visual content analysis:
- extract_frames: Extract key frames from video at intervals
- analyze_frame: Describe what's in a specific frame using AI vision model
- detect_objects: Identify objects in frames using YOLO
- extract_text: OCR text from frames (for slides, captions, code snippets)
- analyze_chart: Detect and analyze charts/graphs in frames

**Report Agent** - For generating summaries:
- create_pdf_report: Generate PDF summary (REQUIRES analysis data first!)
- create_ppt_report: Generate PowerPoint presentation (REQUIRES analysis data first!)

**IMPORTANT RULES:**

1. **For Video Summarization/Report Requests (PDF/PowerPoint)**: 
   - DO NOT include create_pdf_report or create_ppt_report in your plan!
   - Reports are generated AUTOMATICALLY after analysis is complete
   - **ALWAYS include BOTH audio AND visual analysis** for comprehensive summaries:
     a) extract_audio → transcribe_audio (REQUIRED for spoken content, timestamps, context)
     b) extract_frames → analyze_frame (REQUIRED for visual understanding)
   - Optional: detect_objects, extract_text (if slides/text visible)
   - The system will automatically synthesize ALL data and create the report
   
2. **For Specific Analysis Requests**: Use relevant tools
   - "What's in this video?" → transcribe_audio + extract_frames + analyze_frame
   - "Summarize this video" → transcribe_audio + extract_frames + analyze_frame
   - "What objects are shown?" → extract_frames + detect_objects
   - "What text appears?" → extract_frames + extract_text
   - "Describe the video" → transcribe_audio + extract_frames + analyze_frame

3. **Order Matters**: 
   - Transcription: extract_audio → transcribe_audio
   - Vision: extract_frames → (analyze_frame | detect_objects | extract_text)
   - NEVER include report tools - they run automatically!

When a user asks to summarize or analyze a video, determine which ANALYSIS tools to use. 

**For comprehensive summaries/reports, ALWAYS include transcription + visual analysis:**

Example plan for "Summarize this video into PDF":
{
  "reasoning": "To create a comprehensive video summary, we need both audio transcription and visual analysis",
  "agents": [
    {
      "agent": "transcription",
      "tool": "extract_audio",
      "reason": "Extract audio track from video"
    },
    {
      "agent": "transcription",
      "tool": "transcribe_audio",
      "reason": "Get spoken content with timestamps for context and key moments"
    },
    {
      "agent": "vision", 
      "tool": "extract_frames",
      "reason": "Extract visual frames for analysis"
    },
    {
      "agent": "vision",
      "tool": "analyze_frame", 
      "reason": "Understand visual content and scenes"
    }
  ],
  "answer_plan": "System will automatically synthesize transcription + visual data and generate PDF"
}

**CRITICAL**: 
- ALWAYS include transcribe_audio for video summaries (provides context, key moments, timeline)
- ALWAYS include analyze_frame for visual understanding
- Do NOT include create_pdf_report - it runs automatically after analysis!"""
        
        return prompt
    
    async def plan_execution(self, user_query: str, video_context: Optional[Dict[str, Any]] = None, cached_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Use Ollama to create an execution plan for the user query
        
        Args:
            user_query: User's question or request
            video_context: Context about the video (metadata, previous results, cached data)
            cached_tools: List of tool names that have cached results
            
        Returns:
            Execution plan with agents to invoke
        """
        tools = await self.get_available_tools()
        system_prompt = self._create_system_prompt(tools, cached_tools)
        
        context_info = ""
        if video_context:
            # Use string concatenation to avoid f-string format issues with JSON
            video_context_json = json.dumps(video_context, indent=2)
            context_info = "\n\nVideo Context:\n" + video_context_json
        
        # Use string concatenation to avoid f-string format issues
        user_prompt = "User Query: " + user_query + context_info + "\n\nCreate an execution plan to answer this query."
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json"
            )
            
            # Log raw response for debugging
            raw_content = response["message"]["content"]
            logger.debug(f"Raw Ollama response: {raw_content[:500]}")  # First 500 chars
            
            plan = json.loads(raw_content)
            logger.info(f"Execution plan created: {plan.get('reasoning', 'No reasoning provided')}")
            logger.debug(f"Full plan structure: {json.dumps(plan, indent=2)}")
            
            # Validate plan structure
            if "agents" not in plan:
                logger.error("Plan missing 'agents' field")
                plan["agents"] = []
            elif not isinstance(plan["agents"], list):
                logger.error(f"Plan 'agents' is not a list: {type(plan['agents'])}")
                plan["agents"] = []
            
            return plan
            
        except Exception as e:
            logger.exception(f"Error creating execution plan: {e}")
            return {
                "reasoning": "Error in planning",
                "agents": [],
                "error": str(e)
            }
    
    async def execute_plan(self, plan: Dict[str, Any], video_id: str, video_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the agent plan with caching support
        
        Args:
            plan: Execution plan from plan_execution
            video_id: Video identifier
            video_path: Path to video file
            session_id: Optional session ID for caching
            
        Returns:
            Aggregated results from all agents
        """
        results = {
            "video_id": video_id,
            "agent_results": [],
            "errors": [],
            "analysis_data": {}  # Collect analysis data for reports
        }
        
        for agent_task in plan.get("agents", []):
            # Ensure agent_task is a dict, not a string
            if isinstance(agent_task, str):
                logger.error(f"Invalid agent_task (string instead of dict): {agent_task}")
                continue
                
            agent_name = agent_task.get("agent")
            tool_name = agent_task.get("tool")
            reason = agent_task.get("reason", "")
            
            # Validate required fields
            if not agent_name or not tool_name:
                logger.error(f"Invalid agent_task missing required fields: {agent_task}")
                continue
            
            logger.info(f"Executing: {agent_name}.{tool_name} - {reason}")
            
            try:
                # Check cache first if session_id provided
                cached_result = None
                if session_id:
                    from ..utils.session_manager import session_manager
                    cached_result = session_manager.get_cached_result(session_id, tool_name)
                    if cached_result:
                        logger.info(f"Using cached result for {tool_name}")
                        result = cached_result
                    
                # Execute the tool if not cached
                if not cached_result:
                    logger.info(f"Calling MCP tool {agent_name}.{tool_name}...")
                    result = await self._execute_tool(agent_name, tool_name, video_id, video_path, results["analysis_data"], session_id)
                    logger.info(f"Completed {agent_name}.{tool_name}")
                    
                    # Cache the result if session_id provided
                    if session_id:
                        session_manager.cache_analysis_result(session_id, tool_name, result)
                
                # Store result
                results["agent_results"].append({
                    "agent": agent_name,
                    "tool": tool_name,
                    "reason": reason,
                    "result": result
                })
                
                # Accumulate analysis data for report generation
                if agent_name == "transcription" and tool_name == "transcribe_audio":
                    transcription_text = result.get("text", "")
                    results["analysis_data"]["transcription"] = transcription_text
                    results["analysis_data"]["has_transcription"] = True
                    logger.info(f"Stored transcription: {len(transcription_text)} characters")
                elif agent_name == "vision" and tool_name == "extract_frames":
                    results["analysis_data"]["frames"] = result.get("frames", [])
                    results["analysis_data"]["total_frames"] = result.get("total_frames", 0)
                elif agent_name == "vision" and tool_name == "analyze_frame":
                    # Store comprehensive frame-by-frame analyses
                    frame_analyses = result.get("frame_analyses", [])
                    results["analysis_data"]["frame_analyses"] = frame_analyses
                    results["analysis_data"]["total_analyzed_frames"] = result.get("total_analyzed", 0)
                    logger.info(f"Stored {len(frame_analyses)} comprehensive frame analyses")
                    logger.debug(f"Frame analyses sample: {frame_analyses[0] if frame_analyses else 'empty'}")
                elif agent_name == "vision" and tool_name == "detect_objects":
                    results["analysis_data"]["detected_objects"] = result.get("objects", [])
                elif agent_name == "vision" and tool_name == "extract_text":
                    results["analysis_data"]["extracted_text"] = result.get("text", "")
                elif agent_name == "vision" and tool_name == "analyze_chart":
                    results["analysis_data"]["chart_analysis"] = result.get("analysis", {})
                    
            except Exception as e:
                logger.exception(f"Error executing {agent_name}.{tool_name}: {e}")
                results["errors"].append({
                    "agent": agent_name,
                    "tool": tool_name,
                    "error": str(e)
                })
        
        logger.info(f"🔄 All agent tasks completed. Checking for synthesis...")
        logger.debug(f"Analysis data keys: {list(results['analysis_data'].keys())}")
        
        # Step 4: LLM Synthesis - Create structured summary from all data
        has_transcription = bool(results["analysis_data"].get("transcription"))
        has_frame_analyses = bool(results["analysis_data"].get("frame_analyses"))
        
        logger.info(f"Analysis complete - Transcription: {has_transcription}, Frame analyses: {has_frame_analyses}")
        
        # Check if user explicitly asked for a report/summary (comprehensive analysis)
        # We only auto-generate PDF if the plan includes BOTH transcription AND visual analysis
        # This indicates the user wants a full summary/report, not just raw transcription
        has_both_analyses = has_transcription and has_frame_analyses
        
        # Create structured summary only if we have comprehensive data (both audio and visual)
        if has_both_analyses:
            logger.info("Creating structured summary with LLM (comprehensive analysis detected)...")
            structured_summary = await self.synthesize_structured_summary(results["analysis_data"])
            results["analysis_data"]["structured_summary"] = structured_summary
            logger.info(f"Structured summary created: {structured_summary.get('title')}")
            
            # Step 5: Automatically generate PDF report with structured summary
            try:
                logger.info("Generating PDF report with structured summary...")
                pdf_result = await self._execute_tool(
                    "report",
                    "create_pdf_report",
                    video_id,
                    "",  # video_path not needed for report
                    results["analysis_data"],
                    session_id
                )
                results["agent_results"].append({
                    "agent": "report",
                    "tool": "create_pdf_report",
                    "reason": "Generate PDF with structured summary",
                    "result": pdf_result
                })
                logger.info(f"✅ PDF report generated: {pdf_result.get('output_path')}")
            except Exception as e:
                logger.exception(f"Error generating PDF report: {e}")
                results["errors"].append({
                    "agent": "report",
                    "tool": "create_pdf_report",
                    "error": str(e)
                })
        else:
            logger.info("Skipping structured summary - user requested specific analysis, not comprehensive report")
            logger.debug(f"Has transcription only: {has_transcription and not has_frame_analyses}, Has frames only: {has_frame_analyses and not has_transcription}")
        
        return results
    
    async def _execute_tool(self, agent: str, tool: str, video_id: str, video_path: str, analysis_data: Dict[str, Any] = None, session_id: str = None) -> Any:
        """
        Execute a specific tool
        
        Args:
            agent: Agent name
            tool: Tool name  
            video_id: Video ID
            video_path: Path to video
            analysis_data: Accumulated analysis data for report generation
            session_id: Session ID for temp folder location
        """
        if analysis_data is None:
            analysis_data = {}
        
        # Determine temp directory - use session temp if available, otherwise global temp
        if session_id:
            temp_dir = f"data/sessions/{session_id}/temp"
        else:
            temp_dir = str(settings.TEMP_DIR)
            
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
                # Comprehensive multi-frame analysis with ALL vision tools
                frames_data = analysis_data.get("frames", [])
                total_frames = analysis_data.get("total_frames", 0)
                
                if not frames_data:
                    # Fallback: analyze just frame 0
                    frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                    return await self.client.analyze_frame(frame_path)
                
                # Analyze every 5th frame (or adjust based on total frames)
                sample_rate = max(1, total_frames // 5) if total_frames > 5 else 1
                frame_analyses = []
                
                logger.info(f"Comprehensive frame analysis: sampling every {sample_rate} frame(s) from {total_frames} total")
                
                for i, frame_info in enumerate(frames_data):
                    if i % sample_rate == 0:  # Sample frames
                        frame_path = frame_info.get("path")
                        if frame_path:
                            try:
                                # Comprehensive analysis with ALL tools per frame
                                description_result = await self.client.analyze_frame(frame_path)
                                objects_result = await self.client.detect_objects(frame_path)
                                text_result = await self.client.extract_text(frame_path)
                                
                                frame_analysis = {
                                    "frame": i,
                                    "timestamp": frame_info.get("timestamp", 0.0),
                                    "description": description_result.get("description", description_result.get("analysis", "")),
                                    "objects": [obj.get("class") for obj in objects_result.get("objects", [])],
                                    "text": text_result.get("text", ""),
                                    "confidence": description_result.get("confidence", 0.0)
                                }
                                
                                frame_analyses.append(frame_analysis)
                                logger.info(f"Analyzed frame {i}/{total_frames}: {len(frame_analysis['objects'])} objects, {len(frame_analysis['text'])} chars text")
                            except Exception as e:
                                logger.error(f"Error analyzing frame {i}: {e}")
                
                # Return comprehensive visual summary
                return {
                    "frame_analyses": frame_analyses,
                    "total_analyzed": len(frame_analyses),
                    "sample_rate": sample_rate
                }
                
            elif tool == "detect_objects":
                frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                logger.info(f"Calling detect_objects on frame: {frame_path}")
                result = await self.client.detect_objects(frame_path)
                logger.info(f"detect_objects completed, found {len(result.get('objects', []))} objects")
                return result
            elif tool == "extract_text":
                frame_path = f"{temp_dir}/{video_id}_frame_0.jpg"
                return await self.client.extract_text(frame_path)
        
        elif agent == "report":
            if tool == "create_pdf_report":
                # Use structured summary from LLM
                structured_summary = analysis_data.get("structured_summary", {})
                
                # Debug: Log the structured summary to see what LLM returned
                logger.debug(f"Structured summary keys: {list(structured_summary.keys())}")
                logger.debug(f"Key moments type: {type(structured_summary.get('key_moments'))}")
                logger.debug(f"Visual summary type: {type(structured_summary.get('visual_summary'))}")
                logger.debug(f"Takeaways type: {type(structured_summary.get('takeaways'))}")
                
                # Get timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                
                # Build formatted content sections
                title = structured_summary.get("title", "Video Analysis Report")
                
                # Summary section - Use LLM-generated summary with more detail
                summary_text = structured_summary.get("summary", "Comprehensive video analysis completed.")
                
                # Key moments timeline - Each on new line
                key_moments_text = ""
                key_moments = structured_summary.get("key_moments", [])
                if key_moments and isinstance(key_moments, list):
                    for moment in key_moments:
                        if isinstance(moment, dict):
                            time = moment.get('time', '00:00')
                            event = moment.get('event', '')
                            key_moments_text += f"{time} - {event}\n"
                        elif isinstance(moment, str):
                            # Fallback: if moment is a string, just add it
                            key_moments_text += f"• {moment}\n"
                else:
                    key_moments_text = "No key moments identified"
                
                # Visual analysis - Use LLM visual_summary for cleaner output
                visual_analysis_text = ""
                visual_summary = structured_summary.get("visual_summary", [])
                frame_analyses = analysis_data.get("frame_analyses", [])
                
                if visual_summary and isinstance(visual_summary, list):
                    # Use LLM-generated visual summary (cleaner)
                    visual_analysis_text = f"Analyzed {len(frame_analyses)} key frames:\n\n"
                    for vis in visual_summary:
                        if isinstance(vis, dict):
                            frame_num = vis.get('frame', 'N/A')
                            desc = vis.get('desc', '')
                            visual_analysis_text += f"Frame {frame_num}: {desc}\n"
                        elif isinstance(vis, str):
                            # Fallback: if vis is a string, just add it
                            visual_analysis_text += f"• {vis}\n"
                elif frame_analyses:
                    # Fallback: Format frame descriptions cleanly
                    visual_analysis_text = f"Analyzed {len(frame_analyses)} key frames:\n\n"
                    for frame in frame_analyses:
                        frame_num = frame.get('frame', 0)
                        desc = frame.get('description', 'No description')
                        visual_analysis_text += f"Frame {frame_num}: {desc}\n"
                        
                        # Add extracted text if available (clean format)
                        text = frame.get('text', '').strip()
                        if text and len(text) > 10:
                            # Clean up the text - remove extra spaces and newlines
                            text_clean = ' '.join(text.split())[:150]
                            visual_analysis_text += f"  Text visible: {text_clean}\n"
                        visual_analysis_text += "\n"
                else:
                    visual_analysis_text = "No visual analysis available"
                
                # Transcript summary - Remove timestamps and show full content
                transcription = analysis_data.get("transcription", "")
                if transcription:
                    # Remove timestamp markers like [00:00:00.000 --> 00:00:07.160]
                    import re
                    transcript_clean = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', transcription)
                    # Clean up extra whitespace
                    transcript_clean = ' '.join(transcript_clean.split())
                    transcript_summary = transcript_clean  # No character limit - show full transcript
                else:
                    transcript_summary = "No transcription available"
                
                # Key takeaways
                takeaways_text = ""
                takeaways = structured_summary.get("takeaways", [])
                if takeaways and isinstance(takeaways, list):
                    for i, takeaway in enumerate(takeaways, 1):
                        if isinstance(takeaway, str):
                            takeaways_text += f"- {takeaway}\n"
                        elif isinstance(takeaway, dict):
                            # If takeaway is a dict, try to get a 'text' or 'takeaway' field
                            text = takeaway.get('text') or takeaway.get('takeaway') or str(takeaway)
                            takeaways_text += f"- {text}\n"
                else:
                    takeaways_text = "No key takeaways identified"
                
                # Build final content structure for PDF
                content = {
                    "video_id": video_id,
                    "title": f"{title}\nGenerated: {timestamp}",
                    "sections": ["summary", "key_moments", "visual_analysis", "transcript_summary", "takeaways"],
                    "summary": summary_text,
                    "key_moments": key_moments_text,
                    "visual_analysis": visual_analysis_text,
                    "transcript_summary": transcript_summary,
                    "takeaways": takeaways_text
                }
                
                # Save report to session reports folder if session_id provided
                if session_id:
                    output_path = f"data/sessions/{session_id}/reports/{video_id}_report.pdf"
                else:
                    output_path = f"{settings.REPORTS_DIR}/{video_id}_report.pdf"
                return await self.client.create_pdf_report(content, output_path)
                
            elif tool == "create_ppt_report":
                # Prepare comprehensive content for PPT report (similar to PDF)
                sections = ["summary", "transcription", "visual_analysis"]
                
                if analysis_data.get("detected_objects"):
                    sections.append("detected_objects")
                if analysis_data.get("extracted_text"):
                    sections.append("extracted_text")
                
                content = {
                    "video_id": video_id,
                    "sections": sections,
                    "summary": {
                        "total_frames": analysis_data.get("total_frames", 0),
                        "has_transcription": analysis_data.get("has_transcription", False),
                        "has_visual_analysis": bool(analysis_data.get("frame_analysis")),
                        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "transcription": analysis_data.get("transcription", "No transcription available"),
                    "visual_analysis": analysis_data.get("frame_analysis", "No visual analysis available"),
                    "detected_objects": self._format_detected_objects(analysis_data.get("detected_objects", [])),
                    "extracted_text": analysis_data.get("extracted_text", "No text detected"),
                    "metadata": {}
                }
                
                # Save report to session reports folder if session_id provided
                if session_id:
                    output_path = f"data/sessions/{session_id}/reports/{video_id}_report.pptx"
                else:
                    output_path = f"{settings.REPORTS_DIR}/{video_id}_report.pptx"
                return await self.client.create_ppt_report(content, output_path)
        
        return {"error": f"Unknown tool: {agent}.{tool}"}
    
    def _format_detected_objects(self, objects: List[Dict[str, Any]]) -> str:
        """Format detected objects for report"""
        if not objects:
            return "No objects detected"
        
        # Group by class and count
        object_counts = {}
        for obj in objects:
            class_name = obj.get("class", "unknown")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Format as text
        lines = []
        for class_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {class_name}: {count} instance(s)")
        
        return "\n".join(lines)
    
    def _format_chart_analysis(self, chart_data: Dict[str, Any]) -> str:
        """Format chart analysis for report"""
        if not chart_data:
            return "No charts detected"
        
        chart_type = chart_data.get("chart_type", "unknown")
        has_axes = chart_data.get("has_axes", False)
        text_labels = chart_data.get("text_labels", [])
        
        result = f"Chart Type: {chart_type}\n"
        result += f"Has Axes: {'Yes' if has_axes else 'No'}\n"
        
        if text_labels:
            result += f"\nDetected Labels:\n"
            for label in text_labels[:10]:  # Limit to 10 labels
                result += f"- {label.get('text', '')}\n"
        
        return result
    
    async def synthesize_structured_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to create structured summary from all analysis data
        
        Args:
            analysis_data: Complete analysis data (transcription, frame analyses, etc.)
            
        Returns:
            Structured summary with title, key_moments, visual_summary, takeaways
        """
        try:
            # Prepare data for LLM
            transcription = analysis_data.get("transcription", "No transcription available")
            frame_analyses = analysis_data.get("frame_analyses", [])
            total_frames = analysis_data.get("total_frames", 0)
            
            # Build structured prompt with more context
            prompt = """You are a video analysis assistant. Given the transcription and visual analysis of a video, create a comprehensive structured summary.

**Full Transcription:**
""" + transcription[:5000] + """

**Visual Analysis (Key Frames):**
"""
            
            for frame in frame_analyses[:10]:  # Limit to first 10 for prompt size
                prompt += f"\n[Frame {frame['frame']} @ {frame['timestamp']:.1f}s]"
                prompt += f"\n- Scene: {frame['description']}"
                if frame.get('objects'):
                    prompt += f"\n- Objects: {', '.join(frame['objects'][:5])}"
                if frame.get('text'):
                    prompt += f"\n- Text: {frame['text'][:100]}"
                prompt += "\n"
            
            prompt += """

**Task:** Create a detailed JSON response with:
1. "title": A concise, descriptive title for the video (5-10 words)
2. "summary": A comprehensive 4-6 sentence summary covering main topics, key concepts, and overall purpose
3. "key_moments": Array of 5-10 important moments with {"time": "MM:SS", "event": "brief description"}
   - Extract these from the transcription timestamps
   - Focus on topic changes, key explanations, examples, or demonstrations
4. "visual_summary": Array of key visual observations with {"frame": number, "desc": "what was shown"}
   - Describe what was visually presented (slides, demonstrations, graphics, text)
5. "takeaways": Array of 4-6 key insights, learnings, or main points
   - What should viewers remember from this video?

Return ONLY valid JSON, no markdown formatting."""

            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a video summarization expert. Analyze and extract key insights."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            structured_summary = json.loads(response["message"]["content"])
            logger.info(f"Created structured summary: {structured_summary.get('title', 'Untitled')}")
            return structured_summary
            
        except Exception as e:
            logger.exception(f"Error creating structured summary: {e}")
            # Return fallback structure
            return {
                "title": "Video Analysis",
                "summary": "Analysis completed with transcription and visual data.",
                "key_moments": [],
                "visual_summary": [],
                "takeaways": ["See detailed sections below for full analysis."]
            }
    
    async def synthesize_response(self, user_query: str, results: Dict[str, Any]) -> str:
        """
        Use Ollama to synthesize final response from agent results
        
        Args:
            user_query: Original user query
            results: Results from execute_plan
            
        Returns:
            Natural language response
        """
        try:
            # Check if a report was generated
            report_path = None
            for agent_result in results.get("agent_results", []):
                if agent_result.get("agent") == "report":  # Changed from "report_agent" to "report"
                    result_data = agent_result.get("result", {})
                    if result_data.get("success"):
                        report_path = result_data.get("output_path")
                        break
            
            # Get structured summary if available
            structured_summary = results.get("analysis_data", {}).get("structured_summary", {})
            
            # If we have a structured summary and report, create a user-friendly response
            if structured_summary and report_path:
                title = structured_summary.get("title", "Video Analysis")
                summary = structured_summary.get("summary", "Analysis completed.")
                key_moments = structured_summary.get("key_moments", [])
                takeaways = structured_summary.get("takeaways", [])
                
                # Build a comprehensive summary for the user
                response_text = f"✅ **Video Analysis Complete!**\n\n"
                response_text += f"**Title:** {title}\n\n"
                response_text += f"**Summary:** {summary}\n\n"
                
                if key_moments and isinstance(key_moments, list):
                    response_text += f"**Key Moments:**\n"
                    for moment in key_moments:  # Show ALL key moments
                        if isinstance(moment, dict):
                            response_text += f"• {moment.get('time', '00:00')} - {moment.get('event', '')}\n"
                        elif isinstance(moment, str):
                            response_text += f"• {moment}\n"
                    response_text += "\n"
                
                if takeaways and isinstance(takeaways, list):
                    response_text += f"**Key Takeaways:**\n"
                    for i, takeaway in enumerate(takeaways, 1):  # Show ALL takeaways
                        if isinstance(takeaway, str):
                            response_text += f"{i}. {takeaway}\n"
                        elif isinstance(takeaway, dict):
                            text = takeaway.get('text') or takeaway.get('takeaway') or str(takeaway)
                            response_text += f"{i}. {text}\n"
                    response_text += "\n"
                
                response_text += f"📄 **Full PDF Report:** `{report_path}`"
                return response_text
            
            # Check if user asked for specific data (transcription only, frames only, etc.)
            analysis_data = results.get("analysis_data", {})
            transcription = analysis_data.get("transcription", "")
            
            # If user asked for transcription and we have it, return it directly
            if "transcribe" in user_query.lower() and transcription and not structured_summary:
                logger.info("User requested transcription only - returning raw transcript")
                # Clean up the transcription (remove timestamps if present)
                import re
                transcript_clean = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', transcription)
                transcript_clean = ' '.join(transcript_clean.split())
                
                return f"📝 **Video Transcription:**\n\n{transcript_clean}"
            
            # Fallback: Use LLM to synthesize (if no structured summary)
            results_summary = {
                "agent_results": [
                    {"agent": r.get("agent"), "tool": r.get("tool"), "reason": r.get("reason")}
                    for r in results.get("agent_results", [])
                ],
                "errors": results.get("errors", [])
            }
            
            context = "User asked: " + user_query + "\n\nAgent results:\n" + json.dumps(results_summary, indent=2)

            if report_path:
                context += "\n\n**IMPORTANT**: A report file was successfully created at: " + report_path
                context += "\nMake sure to inform the user about this file path in your response!"

            prompt = context + "\n\nProvide a clear, helpful answer to the user's question based on these results."

            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing video content. Synthesize the agent results into a clear answer. If a report file was created, prominently mention the file path to the user."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response["message"]["content"]
            
            # Ensure report path is mentioned if it exists
            if report_path and report_path not in response_text:
                response_text += f"\n\n📄 **Report generated**: `{report_path}`"
            
            return response_text
            
        except Exception as e:
            logger.exception(f"Error synthesizing response: {e}")
            
            # Fallback: check if report was created
            for agent_result in results.get("agent_results", []):
                if agent_result.get("agent") == "report":
                    result_data = agent_result.get("result", {})
                    if result_data.get("success"):
                        report_path = result_data.get("output_path")
                        return f"✅ Report successfully generated!\n\n📄 File location: `{report_path}`\n\nYou can find your PDF report at this location."
            
            return f"I encountered results but had trouble formatting the answer: {str(e)}"


# Global orchestrator instance
orchestrator = MCPOrchestrator()
