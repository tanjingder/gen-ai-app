"""
Test MCP Stdio Client Integration
Tests the true MCP architecture with stdio communication
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from src.mcp.client import mcp_client
from loguru import logger


async def test_transcription_agent():
    """Test transcription agent connection and tools"""
    print("\n" + "="*70)
    print("🎵 TESTING TRANSCRIPTION AGENT")
    print("="*70)
    
    try:
        # List available tools
        print("\n1. Listing available tools...")
        tools = await mcp_client.list_tools("transcription")
        
        if tools:
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print("❌ No tools found or connection failed")
            return False
        
        print("\n✅ Transcription agent is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Transcription agent test failed")
        return False


async def test_vision_agent():
    """Test vision agent connection and tools"""
    print("\n" + "="*70)
    print("👁️  TESTING VISION AGENT")
    print("="*70)
    
    try:
        # List available tools
        print("\n1. Listing available tools...")
        tools = await mcp_client.list_tools("vision")
        
        if tools:
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print("❌ No tools found or connection failed")
            return False
        
        print("\n✅ Vision agent is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Vision agent test failed")
        return False


async def test_report_agent():
    """Test report agent connection and tools"""
    print("\n" + "="*70)
    print("📊 TESTING REPORT AGENT")
    print("="*70)
    
    try:
        # List available tools
        print("\n1. Listing available tools...")
        tools = await mcp_client.list_tools("report")
        
        if tools:
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print("❌ No tools found or connection failed")
            return False
        
        print("\n✅ Report agent is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Report agent test failed")
        return False


async def main():
    """Main test function"""
    print("\n" + "="*70)
    print("🧪 MCP STDIO CLIENT INTEGRATION TEST")
    print("="*70)
    print("\nThis test verifies that the backend can communicate with")
    print("MCP agents via stdio (standard input/output) protocol.")
    print("\nThe backend will spawn each agent as a subprocess and")
    print("communicate using the official MCP protocol.")
    print("="*70)
    
    results = {}
    
    try:
        # Test each agent
        results["transcription"] = await test_transcription_agent()
        results["vision"] = await test_vision_agent()
        results["report"] = await test_report_agent()
    finally:
        # Cleanup always runs, even if tests fail
        print("\n" + "="*70)
        print("🧹 CLEANING UP")
        print("="*70)
        
        try:
            await mcp_client.cleanup()
            print("✅ All connections closed")
        except Exception as e:
            print(f"⚠️  Cleanup encountered issues (agents may still close): {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for agent, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{agent.upper():15s}: {status}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n✨ The MCP stdio architecture is working correctly!")
        print("Your backend can now communicate with all agents.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\n⚠️  Troubleshooting:")
        print("1. Make sure agent server.py files exist in mcp-servers/*/")
        print("2. Check that backend/venv/Scripts/python.exe exists")
        print("3. Verify mcp package is installed: pip install mcp==0.9.1")
        print("4. Check agent server.py files have no syntax errors")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit_code = 130
    
    sys.exit(exit_code)
