"""
Vision Agent MCP Server
Handles frame extraction, object detection, OCR, and image analysis
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import base64

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import cv2
import numpy as np


class VisionAgent:
    """Agent for vision and image analysis tasks"""
    
    def __init__(self):
        self.temp_dir = Path("./temp")
    
    async def extract_frames(self, video_path: str, interval_seconds: float = 1.0, output_dir: str = None) -> Dict[str, Any]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            interval_seconds: Interval between frames in seconds
            output_dir: Optional output directory for frames (defaults to ./temp)
            
        Returns:
            List of extracted frame paths
        """
        try:
            video_file = Path(video_path)
            if not video_file.exists():
                return {"error": f"Video file not found: {video_path}"}
            
            # Extract video_id from filename (format: video_id_originalname.ext)
            # If filename doesn't match pattern, fall back to using stem
            filename_stem = video_file.stem
            video_id = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem
            
            # Use provided output_dir or fall back to default temp_dir
            target_dir = Path(output_dir) if output_dir else self.temp_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)
            
            frames = []
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Use standardized naming: video_id_frame_N.jpg
                    frame_path = target_dir / f"{video_id}_frame_{saved_count}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frames.append({
                        "frame_index": saved_count,
                        "timestamp": frame_count / fps,
                        "path": str(frame_path)
                    })
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            
            return {
                "success": True,
                "frames": frames,
                "total_frames": saved_count
            }
            
        except Exception as e:
            return {"error": f"Frame extraction failed: {str(e)}"}
    
    async def analyze_frame(self, frame_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Analyze frame content using Ollama LLaVA vision model
        
        Args:
            frame_path: Path to frame image
            prompt: Optional prompt to guide analysis
            
        Returns:
            Description of frame content
        """
        try:
            frame_file = Path(frame_path)
            if not frame_file.exists():
                return {"error": f"Frame file not found: {frame_path}"}
            
            # Read image and get basic info
            image = cv2.imread(str(frame_file))
            height, width = image.shape[:2]
            
            try:
                # Use Ollama with LLaVA for vision analysis
                import ollama
                
                # Encode image to base64 for Ollama
                with open(frame_file, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Prepare prompt
                analysis_prompt = prompt if prompt else "Describe what you see in this image concisely in 1-2 sentences. Focus on the main subject, key objects, and any visible text."
                
                # Call Ollama LLaVA
                response = ollama.chat(
                    model='llava',
                    messages=[{
                        'role': 'user',
                        'content': analysis_prompt,
                        'images': [img_base64]
                    }]
                )
                
                description = response['message']['content']
                
                return {
                    "success": True,
                    "description": description,
                    "frame_info": {
                        "width": width,
                        "height": height,
                        "path": str(frame_file)
                    },
                    "model": "ollama/llava"
                }
                
            except ImportError:
                return {
                    "error": "Ollama package not installed. Run: pip install ollama"
                }
            except Exception as ollama_error:
                # Fallback to basic analysis if Ollama fails
                mean_color = image.mean(axis=(0, 1))
                return {
                    "success": True,
                    "description": f"Frame shows an image of size {width}x{height}. Vision model unavailable: {str(ollama_error)}",
                    "frame_info": {
                        "width": width,
                        "height": height,
                        "mean_brightness": float(mean_color.mean())
                    },
                    "note": "Ollama LLaVA model not available. Ensure Ollama is running and llava model is pulled: ollama pull llava"
                }
            
        except Exception as e:
            return {"error": f"Frame analysis failed: {str(e)}"}
    
    async def detect_objects(self, frame_path: str) -> Dict[str, Any]:
        """
        Detect objects in frame using YOLO (Ultralytics)
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            List of detected objects with bounding boxes
        """
        try:
            frame_file = Path(frame_path)
            if not frame_file.exists():
                return {"error": f"Frame file not found: {frame_path}"}
            
            try:
                # Use Ultralytics YOLO for object detection
                import torch
                
                # Fix for PyTorch 2.6+ weights_only default change
                # Temporarily override torch.load to use weights_only=False for YOLO models
                _original_torch_load = torch.load
                
                def _patched_torch_load(*args, **kwargs):
                    # Force weights_only=False for loading YOLO models (safe for official models)
                    kwargs['weights_only'] = False
                    return _original_torch_load(*args, **kwargs)
                
                torch.load = _patched_torch_load
                
                try:
                    from ultralytics import YOLO
                    # Load YOLOv8 model (will auto-download on first use)
                    model = YOLO('yolov8n.pt')  # nano model for speed
                finally:
                    # Restore original torch.load
                    torch.load = _original_torch_load
                
                # Run inference
                results = model(str(frame_file), verbose=False)
                
                # Parse results
                detected_objects = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        detected_objects.append({
                            "class": result.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        })
                
                return {
                    "success": True,
                    "objects": detected_objects,
                    "total_detected": len(detected_objects),
                    "model": "yolov8n"
                }
                
            except ImportError:
                return {
                    "error": "Ultralytics not installed. Run: pip install ultralytics"
                }
            except Exception as yolo_error:
                # Fallback to basic info
                image = cv2.imread(str(frame_file))
                height, width = image.shape[:2]
                
                return {
                    "success": False,
                    "error": f"YOLO detection failed: {str(yolo_error)}",
                    "objects": [],
                    "note": "Install ultralytics: pip install ultralytics"
                }
            
        except Exception as e:
            return {"error": f"Object detection failed: {str(e)}"}
    
    async def extract_text(self, frame_path: str) -> Dict[str, Any]:
        """
        Extract text from frame using OCR
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Extracted text with bounding boxes
        """
        try:
            frame_file = Path(frame_path)
            if not frame_file.exists():
                return {"error": f"Frame file not found: {frame_path}"}
            
            # Placeholder for OCR (would use Tesseract or PaddleOCR)
            # You could integrate with pytesseract or paddleocr
            
            try:
                import pytesseract
                image = cv2.imread(str(frame_file))
                text = pytesseract.image_to_string(image)
                
                return {
                    "success": True,
                    "text": text,
                    "note": "Using pytesseract for OCR"
                }
            except ImportError:
                return {
                    "success": True,
                    "text": "",
                    "note": "Install pytesseract for OCR: pip install pytesseract"
                }
            
        except Exception as e:
            return {"error": f"Text extraction failed: {str(e)}"}
    
    async def analyze_chart(self, frame_path: str) -> Dict[str, Any]:
        """
        Analyze charts and graphs in frame using OpenCV + pytesseract
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Chart analysis results with type, text labels, and structure
        """
        try:
            frame_file = Path(frame_path)
            if not frame_file.exists():
                return {"error": f"Frame file not found: {frame_path}"}
            
            # Read image
            image = cv2.imread(str(frame_file))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape[:2]
            
            # Detect chart type and structure using OpenCV
            chart_analysis = {
                "width": width,
                "height": height,
                "chart_type": "unknown",
                "has_axes": False,
                "has_bars": False,
                "has_lines": False,
                "text_labels": []
            }
            
            # Edge detection for finding chart elements
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines (for line charts and axes)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) > 0:
                chart_analysis["has_lines"] = True
                
                # Check for perpendicular lines (axes)
                horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
                vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 10)
                
                if horizontal_lines > 0 and vertical_lines > 0:
                    chart_analysis["has_axes"] = True
            
            # Detect rectangles/bars (for bar charts)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangular_contours = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter small contours
                    if w > 20 and h > 20:
                        rectangular_contours.append((x, y, w, h))
            
            if len(rectangular_contours) >= 2:
                chart_analysis["has_bars"] = True
            
            # Determine chart type
            if chart_analysis["has_bars"]:
                chart_analysis["chart_type"] = "bar_chart"
            elif chart_analysis["has_lines"] and chart_analysis["has_axes"]:
                chart_analysis["chart_type"] = "line_chart"
            elif chart_analysis["has_axes"]:
                chart_analysis["chart_type"] = "plot"
            
            # Extract text labels using OCR
            try:
                import pytesseract
                
                # Extract all text
                text = pytesseract.image_to_string(image)
                
                # Get text with bounding boxes
                text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                labels = []
                for i in range(len(text_data['text'])):
                    txt = text_data['text'][i].strip()
                    if txt and len(txt) > 0:
                        labels.append({
                            "text": txt,
                            "confidence": text_data['conf'][i],
                            "bbox": [
                                text_data['left'][i],
                                text_data['top'][i],
                                text_data['width'][i],
                                text_data['height'][i]
                            ]
                        })
                
                chart_analysis["text_labels"] = labels
                chart_analysis["full_text"] = text
                
            except ImportError:
                chart_analysis["note"] = "Install pytesseract for text extraction: pip install pytesseract"
            except Exception as ocr_error:
                chart_analysis["ocr_error"] = str(ocr_error)
            
            return {
                "success": True,
                "analysis": chart_analysis
            }
            
        except Exception as e:
            return {"error": f"Chart analysis failed: {str(e)}"}


async def main():
    """Main entry point for the MCP server"""
    agent = VisionAgent()
    server = Server("vision-agent")
    
    # Register tools
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="extract_frames",
                description="Extract key frames from video at specified intervals",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to the video file"
                        },
                        "interval_seconds": {
                            "type": "number",
                            "description": "Interval between frames in seconds",
                            "default": 1.0
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional output directory for extracted frames"
                        }
                    },
                    "required": ["video_path"]
                }
            ),
            Tool(
                name="analyze_frame",
                description="Analyze a frame and describe its content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "frame_path": {
                            "type": "string",
                            "description": "Path to the frame image"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional prompt to guide analysis"
                        }
                    },
                    "required": ["frame_path"]
                }
            ),
            Tool(
                name="detect_objects",
                description="Detect and identify objects in a frame",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "frame_path": {
                            "type": "string",
                            "description": "Path to the frame image"
                        }
                    },
                    "required": ["frame_path"]
                }
            ),
            Tool(
                name="extract_text",
                description="Extract text from frame using OCR",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "frame_path": {
                            "type": "string",
                            "description": "Path to the frame image"
                        }
                    },
                    "required": ["frame_path"]
                }
            ),
            Tool(
                name="analyze_chart",
                description="Analyze charts and graphs in frame",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "frame_path": {
                            "type": "string",
                            "description": "Path to the frame image"
                        }
                    },
                    "required": ["frame_path"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        if name == "extract_frames":
            result = await agent.extract_frames(
                arguments["video_path"],
                arguments.get("interval_seconds", 1.0),
                arguments.get("output_dir")
            )
        elif name == "analyze_frame":
            result = await agent.analyze_frame(
                arguments["frame_path"],
                arguments.get("prompt")
            )
        elif name == "detect_objects":
            result = await agent.detect_objects(arguments["frame_path"])
        elif name == "extract_text":
            result = await agent.extract_text(arguments["frame_path"])
        elif name == "analyze_chart":
            result = await agent.analyze_chart(arguments["frame_path"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
