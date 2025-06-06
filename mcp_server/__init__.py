"""
AEGIS MCP Server Package

A simple Model Context Protocol server for safely saving compliance analysis reports.
This package demonstrates safe tool calls and custom MCP server implementation for
academic project requirements.

The server provides controlled, validated file operations for the AEGIS multi-agent
compliance system, ensuring that AI agents can save reports through a secure,
standardized interface rather than direct file system access.

Usage:
    from mcp_server import create_server, AEGISReportServer
    
    # Create and run the server
    server = create_server()
    
    # Or create a custom instance
    report_server = AEGISReportServer()

Author: AEGIS Project Team
Version: 1.0.0
Purpose: Academic demonstration of MCP integration with multi-agent systems
"""

# Import the main components that other parts of your system might need
from .server import AEGISReportServer, create_server

# Package metadata that helps with debugging and documentation
__version__ = "1.0.0"
__author__ = "AEGIS Project Team"
__description__ = "Simple MCP server for compliance report management"

# Define what gets imported when someone does "from mcp_server import *"
# This is good practice because it explicitly controls the public interface
__all__ = [
    "AEGISReportServer",    # The main server class
    "create_server",        # The simple server creation function
    "get_reports_directory", # Utility function for finding saved reports
    "list_saved_reports"    # Utility function for browsing reports
]

# Add some useful utility functions that make your package more complete
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


def get_reports_directory() -> Path:
    """
    Get the path to the directory where compliance reports are saved.
    
    This utility function helps other parts of your AEGIS system find the
    reports that the MCP server has saved, enabling features like report
    browsing or analysis of historical compliance queries.
    
    Returns:
        Path: The directory containing saved reports
    """
    return Path(__file__).parent / "summarization_reports"


def list_saved_reports() -> List[Dict[str, Any]]:
    """
    List all compliance reports saved by the MCP server.
    
    This function provides a way to programmatically access the reports
    that your MCP server has saved, which could be useful for building
    report browsing features or analyzing historical compliance patterns.
    
    Returns:
        List[Dict]: Information about each saved report including filename,
                   creation time, and basic metadata
    """
    reports_dir = get_reports_directory()
    
    if not reports_dir.exists():
        return []
    
    reports = []
    
    # Scan for JSON report files
    for report_file in reports_dir.glob("*.json"):
        try:
            # Get file metadata
            file_stats = report_file.stat()
            file_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            # Try to read basic report information
            report_info = {
                "filename": report_file.name,
                "file_path": str(report_file),
                "file_size_bytes": file_size,
                "modified_time": modified_time.isoformat(),
                "query_preview": None,
                "citations_count": None
            }
            
            # Try to extract preview information from the report
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                    # Add preview information if available
                    if "query" in report_data:
                        query = report_data["query"]
                        report_info["query_preview"] = query[:100] + "..." if len(query) > 100 else query
                    
                    if "citations" in report_data and "total_citations" in report_data["citations"]:
                        report_info["citations_count"] = report_data["citations"]["total_citations"]
                        
            except (json.JSONDecodeError, KeyError):
                # If we can't read the file content, that's okay - we still have file metadata
                pass
            
            reports.append(report_info)
            
        except (OSError, IOError):
            # Skip files we can't read
            continue
    
    # Sort by modification time, newest first
    reports.sort(key=lambda r: r["modified_time"], reverse=True)
    
    return reports


def cleanup_old_reports(days_to_keep: int = 30) -> int:
    """
    Clean up old report files to prevent unlimited disk usage.
    
    This utility function provides a way to maintain your reports directory
    by removing old files. This is good practice for any system that saves
    files regularly, as it prevents disk space issues over time.
    
    Args:
        days_to_keep: Number of days of reports to retain (default: 30)
        
    Returns:
        int: Number of files cleaned up
    """
    reports_dir = get_reports_directory()
    
    if not reports_dir.exists():
        return 0
    
    # Calculate cutoff date
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    cleaned_count = 0
    
    for report_file in reports_dir.glob("*.json"):
        try:
            file_stats = report_file.stat()
            
            if file_stats.st_mtime < cutoff_time:
                report_file.unlink()  # Delete the file
                cleaned_count += 1
                
        except (OSError, IOError):
            # Skip files we can't process
            continue
    
    return cleaned_count


# Package initialization logging (optional, but helpful for debugging)
def _log_package_initialization():
    """Log that the package has been imported successfully."""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"AEGIS MCP Server package loaded (version {__version__})")
    
    # Check if reports directory exists and log status
    reports_dir = get_reports_directory()
    if reports_dir.exists():
        report_count = len(list(reports_dir.glob("*.json")))
        logger.info(f"Reports directory found with {report_count} existing reports")
    else:
        logger.info("Reports directory will be created on first use")


# Automatically log package initialization when imported
# This helps with debugging and shows the package is working
try:
    _log_package_initialization()
except Exception:
    # Don't fail package import if logging fails
    pass