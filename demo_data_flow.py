"""
Demo: Show how files flow through the system
This demonstrates the complete data flow from upload to report
"""
from pathlib import Path

# Simple configuration (without importing backend settings)
class SimpleSettings:
    UPLOAD_DIR = Path("backend/uploads")
    TEMP_DIR = Path("backend/temp")
    REPORTS_DIR = Path("backend/reports")
    
    TRANSCRIPTION_MCP_HOST = "localhost"
    TRANSCRIPTION_MCP_PORT = 8001
    VISION_MCP_HOST = "localhost"
    VISION_MCP_PORT = 8002
    REPORT_MCP_HOST = "localhost"
    REPORT_MCP_PORT = 8003
    
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1"
    OLLAMA_VISION_MODEL = "llava"

settings = SimpleSettings()


def show_folder_structure():
    """Display current folder structure and file counts"""
    
    print("=" * 60)
    print("📁 CURRENT FOLDER STRUCTURE")
    print("=" * 60)
    
    folders = {
        "uploads": (settings.UPLOAD_DIR, "🎥 User Uploads (Original Videos)"),
        "temp": (settings.TEMP_DIR, "🔄 Temporary Processing Files"),
        "reports": (settings.REPORTS_DIR, "📊 Final Reports for Users")
    }
    
    for name, (path, description) in folders.items():
        path = Path(path)
        print(f"\n{description}")
        print(f"Path: {path.absolute()}")
        
        if not path.exists():
            print(f"  ⚠️  Folder does not exist yet")
        else:
            files = list(path.iterdir())
            print(f"  📂 Files: {len(files)}")
            
            if files:
                for i, file in enumerate(files[:5]):  # Show first 5
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"     {i+1}. {file.name} ({size_mb:.2f} MB)")
                
                if len(files) > 5:
                    print(f"     ... and {len(files) - 5} more files")
            else:
                print(f"     (empty)")
    
    print("\n" + "=" * 60)


def show_data_flow():
    """Show the data flow diagram"""
    
    print("\n" + "=" * 60)
    print("🔄 DATA FLOW DIAGRAM")
    print("=" * 60)
    
    print("""
1. USER UPLOADS VIDEO
   └→ Saved to: uploads/video_123456.mp4
   
2. TRANSCRIPTION AGENT (Port 8001)
   ├→ Input:  uploads/video_123456.mp4
   ├→ Output: temp/video_123456_audio.wav
   └→ Result: Transcription text + timestamps
   
3. VISION AGENT (Port 8002)
   ├→ Input:  uploads/video_123456.mp4
   ├→ Output: temp/video_123456_frame_0.jpg
   │          temp/video_123456_frame_1.jpg
   └→ Result: Visual analysis + object detection
   
4. REPORT AGENT (Port 8003)
   ├→ Input:  All analysis results (JSON)
   ├→ Output: reports/video_123456_report.pdf
   └→ Result: Final report for user download

5. CLEANUP (Optional)
   ├→ Delete: temp/video_123456_audio.wav
   ├→ Delete: temp/video_123456_frame_*.jpg
   ├→ Keep:   uploads/video_123456.mp4 ✅
   └→ Keep:   reports/video_123456_report.pdf ✅
""")


def estimate_disk_usage():
    """Estimate disk usage for each folder"""
    
    print("\n" + "=" * 60)
    print("💾 DISK USAGE ESTIMATION")
    print("=" * 60)
    
    folders = [
        (settings.UPLOAD_DIR, "uploads"),
        (settings.TEMP_DIR, "temp"),
        (settings.REPORTS_DIR, "reports")
    ]
    
    total_size = 0
    
    for path, name in folders:
        path = Path(path)
        
        if not path.exists():
            print(f"\n{name.upper()}: Not created yet")
            continue
        
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        total_size += size
        
        print(f"\n{name.upper()}: {size_mb:.2f} MB")
        
        # Show breakdown
        files = [f for f in path.rglob('*') if f.is_file()]
        if files:
            print(f"  Files: {len(files)}")
            
            # Group by extension
            extensions = {}
            for f in files:
                ext = f.suffix.lower() or 'no_ext'
                extensions[ext] = extensions.get(ext, 0) + f.stat().st_size
            
            for ext, ext_size in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
                ext_mb = ext_size / (1024 * 1024)
                print(f"    {ext}: {ext_mb:.2f} MB")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n{'TOTAL'}: {total_mb:.2f} MB")


def show_cleanup_recommendations():
    """Show recommendations for cleanup"""
    
    print("\n" + "=" * 60)
    print("🧹 CLEANUP RECOMMENDATIONS")
    print("=" * 60)
    
    temp_dir = Path(settings.TEMP_DIR)
    
    if not temp_dir.exists() or not list(temp_dir.iterdir()):
        print("\n✅ temp/ folder is empty - no cleanup needed!")
        return
    
    files = [f for f in temp_dir.rglob('*') if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n⚠️  temp/ folder contains {len(files)} files ({size_mb:.2f} MB)")
    print("\nThese files can be safely deleted:")
    
    for i, file in enumerate(files[:10], 1):
        file_size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({file_size_mb:.2f} MB)")
    
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    print("\nTo clean up:")
    print("  Option 1: Manual deletion")
    print(f"    > Remove-Item {temp_dir.absolute()}\\* -Force")
    print("\n  Option 2: Implement automatic cleanup after video processing")


def show_config():
    """Show current configuration"""
    
    print("\n" + "=" * 60)
    print("⚙️  CURRENT CONFIGURATION")
    print("=" * 60)
    
    print(f"""
Folder Paths (from config.py):
  UPLOAD_DIR:  {settings.UPLOAD_DIR}
  TEMP_DIR:    {settings.TEMP_DIR}
  REPORTS_DIR: {settings.REPORTS_DIR}

MCP Agents:
  Transcription: {settings.TRANSCRIPTION_MCP_HOST}:{settings.TRANSCRIPTION_MCP_PORT}
  Vision:        {settings.VISION_MCP_HOST}:{settings.VISION_MCP_PORT}
  Report:        {settings.REPORT_MCP_HOST}:{settings.REPORT_MCP_PORT}

AI Models:
  Ollama Host:   {settings.OLLAMA_HOST}
  Main Model:    {settings.OLLAMA_MODEL}
  Vision Model:  {settings.OLLAMA_VISION_MODEL}
""")


def main():
    """Main demo function"""
    
    print("\n" + "=" * 60)
    print("🎬 VIDEO ANALYSIS SYSTEM - DATA FLOW DEMO")
    print("=" * 60)
    
    # Show configuration
    show_config()
    
    # Show current folder structure
    show_folder_structure()
    
    # Show data flow
    show_data_flow()
    
    # Show disk usage
    estimate_disk_usage()
    
    # Show cleanup recommendations
    show_cleanup_recommendations()
    
    print("\n" + "=" * 60)
    print("✅ For more details, see: DATA_FLOW_ARCHITECTURE.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
