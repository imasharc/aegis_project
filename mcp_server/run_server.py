"""
Startup script for AEGIS MCP Server
This version handles imports gracefully whether run directly or as a module
"""

import asyncio
import sys
from pathlib import Path

# Smart import handling - this is a professional pattern you'll use often
try:
    # Try importing as if we're running as a package module
    from .server import create_server
    print("📦 Imported server as package module")
except ImportError:
    # If that fails, we're probably running directly, so add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        from server import create_server
        print("📁 Imported server from current directory")
    except ImportError as e:
        print(f"❌ Could not import server module: {e}")
        print("🔧 Make sure server.py exists in the same directory")
        sys.exit(1)


async def main():
    """Start the MCP server using stdio transport."""
    
    print("🚀 AEGIS MCP Server Starting...")
    print("=" * 50)
    print("📁 Reports will be saved to: summarization_reports/")
    print("🔒 Server configured for safe tool calls")
    print("🎯 Ready to handle compliance report saving requests")
    print("=" * 50)
    
    try:
        # Create the server instance
        server = create_server()
        print("✅ MCP server instance created successfully")
        
        # For now, we'll simulate the server running since we don't have a full MCP client
        # In a real deployment, this would connect to the actual MCP transport
        print("🔄 Server would normally start listening here...")
        print("💡 To test functionality, run the test script instead")
        
        # Simulate server readiness
        await asyncio.sleep(1)
        print("🎉 MCP server startup completed successfully!")
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🎬 Starting AEGIS MCP Server directly...")
    asyncio.run(main())