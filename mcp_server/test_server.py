"""
Test script to verify MCP server works correctly
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the current directory to Python's path so it can find our modules
sys.path.insert(0, str(Path(__file__).parent))

# Now we can import our server module
from server import AEGISReportServer


async def test_report_saving():
    """Test that our MCP server can save reports safely."""
    
    server = AEGISReportServer()
    
    # Test data similar to what your summarization agent would send
    test_arguments = {
        "query": "How should we handle GDPR data breach notifications?",
        "action_plan": "1. Notify supervisory authority within 72 hours\n2. Document the incident\n3. Assess risk to data subjects",
        "citations": {"total_citations": 5, "gdpr_citations": 3},
        "metadata": {"processing_time": 2.5, "precision_score": 87.3}
    }
    
    # Test the safe saving function
    result = await server._save_report_safely(test_arguments)
    
    print("🧪 MCP Server Test Results:")
    print(f"Response: {result[0].text}")
    
    # Verify file was created
    reports_dir = server.reports_dir
    report_files = list(reports_dir.glob("*.json"))
    
    if report_files:
        print(f"✅ Found {len(report_files)} report files")
        print(f"📁 Reports directory: {reports_dir}")
        
        # Let's also show what's in the latest report
        latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
        print(f"📄 Latest report: {latest_report.name}")
    else:
        print("❌ No report files found")


if __name__ == "__main__":
    print("🚀 Starting MCP Server Test...")
    print("=" * 50)
    asyncio.run(test_report_saving())
    print("=" * 50)
    print("✅ Test completed!")