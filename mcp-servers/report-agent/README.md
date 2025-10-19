# Report Agent MCP Server

An MCP server for generating PDF and PowerPoint reports from video analysis results.

## Features

- Generate PDF reports with analysis summaries
- Create PowerPoint presentations
- Format and structure analysis content
- Professional report layouts

## Prerequisites

- Python 3.11+

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

```powershell
python server.py
```

## Tools

### create_pdf_report
Generate a PDF report from video analysis.

**Input:**
- `content` (object): Report content including metadata and sections
- `output_path` (string): Output PDF file path

**Content Structure:**
```json
{
  "video_id": "uuid",
  "metadata": {
    "duration_ms": 10000,
    "width": 1920,
    "height": 1080,
    "fps": 30
  },
  "sections": ["summary", "transcription", "key_frames"],
  "summary": "Video summary text",
  "transcription": "Full transcription",
  "key_frames": "Frame analysis"
}
```

### create_ppt_report
Generate a PowerPoint presentation.

**Input:**
- `content` (object): Same structure as PDF report
- `output_path` (string): Output PPTX file path

### format_content
Format raw analysis results into structured report content.

**Input:**
- `raw_content` (object): Raw analysis results from other agents

## Example Usage

The report agent integrates with other agents to create comprehensive reports combining:
- Video metadata
- Transcription results
- Frame analysis
- Object detection results
- OCR extracted text

Reports are automatically formatted with professional layouts, tables, and styling.
