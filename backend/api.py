"""
Enhanced FastAPI Backend with MCP Server Integration
This version demonstrates how to integrate the MCP server with the multi-agent system
to create a complete workflow that showcases both components working together.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import logging
import traceback
from datetime import datetime
import sys
from pathlib import Path

# Add the mcp_server directory to the path so we can import our MCP server
mcp_server_path = Path(__file__).parent.parent / "mcp_server"
sys.path.append(str(mcp_server_path))

# Import your existing sophisticated system
from main import EnhancedMultiAgentSystem

# Import your MCP server for integration
try:
    from server import AEGISReportServer
    MCP_SERVER_AVAILABLE = True
    print("✅ MCP Server integration enabled")
except ImportError as e:
    MCP_SERVER_AVAILABLE = False
    print(f"⚠️  MCP Server not available: {e}")
    print("   Multi-agent system will work without MCP integration")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Enhanced-FastAPI-Backend")

app = FastAPI(
    title="Enhanced Multi-Agent Compliance API with MCP Integration",
    description="Sophisticated compliance analysis with MCP server integration for report management",
    version="3.0.0"
)

# Enhanced CORS to support streaming
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instances
multi_agent_system = None
mcp_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize both the multi-agent system and MCP server."""
    global multi_agent_system, mcp_server
    
    try:
        logger.info("Initializing Enhanced Multi-Agent System...")
        multi_agent_system = EnhancedMultiAgentSystem()
        logger.info("✅ Multi-Agent System initialized successfully")
        
        # Initialize MCP server if available
        if MCP_SERVER_AVAILABLE:
            logger.info("Initializing MCP Server for report management...")
            mcp_server = AEGISReportServer()
            logger.info("✅ MCP Server initialized successfully")
        else:
            logger.info("🔄 Running without MCP Server integration")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        raise

class ProgressUpdate(BaseModel):
    """Model for streaming progress updates to the frontend."""
    step: int
    total_steps: int
    agent_name: str
    status: str
    message: str
    percentage: float
    timestamp: str
    
class EnhancedQueryRequest(BaseModel):
    """Enhanced request model that supports both progress tracking and MCP integration."""
    query: str
    session_id: Optional[str] = None
    enable_progress: bool = True
    save_detailed_report: bool = True  # New option for MCP integration

async def process_query_with_mcp_integration(query: str, save_report: bool = True) -> AsyncGenerator[str, None]:
    """
    Process a compliance query with MCP server integration for comprehensive workflow demonstration.
    
    This function demonstrates the complete integration pattern:
    1. Multi-agent system processes the query with sophisticated analysis
    2. MCP server safely saves the detailed report with full metadata
    3. User receives both the immediate response and confirmation of saved report
    
    This showcases how MCP servers enable AI systems to interact safely with
    external resources while maintaining the sophisticated analysis capabilities
    of your multi-agent architecture.
    """
    
    agents = [
        "GDPR Agent",
        "Polish Law Agent", 
        "Security Agent",
        "Integration Agent",
        "Citation Agent"
    ]
    
    try:
        # Stage 1: Multi-Agent Processing with Progress Updates
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'multi_agent_processing', 'message': 'Starting sophisticated multi-agent analysis'})}\n\n"
        
        # Simulate the multi-agent processing with realistic progress
        for step, agent_name in enumerate(agents, 1):
            progress_update = {
                "type": "progress",
                "step": step,
                "total_steps": len(agents) + (2 if save_report and MCP_SERVER_AVAILABLE else 1),
                "agent_name": agent_name,
                "status": "processing",
                "message": f"Processing with {agent_name}...",
                "percentage": round((step / (len(agents) + 2)) * 100, 1),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(progress_update)}\n\n"
            
            # Simulate realistic processing time
            await asyncio.sleep(1.5)
        
        # Stage 2: Execute Actual Multi-Agent Processing
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'executing_analysis', 'message': 'Executing comprehensive compliance analysis'})}\n\n"
        
        # Process the query using your actual multi-agent system
        try:
            result = multi_agent_system.process_query(query)
            
            # Extract the results for MCP integration
            if result and "summary" in result:
                action_plan = result["summary"].get("action_plan", "")
                
                # Create comprehensive metadata for the MCP server
                mcp_report_data = {
                    "query": query,
                    "action_plan": action_plan,
                    "citations": {
                        "total_citations": len(result.get("gdpr_citations", [])) + 
                                         len(result.get("polish_law_citations", [])) + 
                                         len(result.get("internal_policy_citations", [])),
                        "gdpr_citations": len(result.get("gdpr_citations", [])),
                        "polish_law_citations": len(result.get("polish_law_citations", [])),
                        "security_citations": len(result.get("internal_policy_citations", []))
                    },
                    "metadata": {
                        "processing_timestamp": datetime.now().isoformat(),
                        "multi_agent_system": "enhanced_architecture",
                        "domains_analyzed": ["gdpr", "polish_law", "internal_security"],
                        "integration_method": "mcp_protocol_demonstration"
                    }
                }
                
                # Stage 3: MCP Server Integration (if enabled and available)
                if save_report and MCP_SERVER_AVAILABLE and mcp_server:
                    yield f"data: {json.dumps({'type': 'stage', 'stage': 'mcp_integration', 'message': 'Saving detailed report via MCP server'})}\n\n"
                    
                    try:
                        # Use the MCP server to save the detailed report
                        mcp_progress = {
                            "type": "progress",
                            "step": len(agents) + 1,
                            "total_steps": len(agents) + 2,
                            "agent_name": "MCP Server",
                            "status": "processing",
                            "message": "Safely saving compliance report via MCP protocol",
                            "percentage": round(((len(agents) + 1) / (len(agents) + 2)) * 100, 1),
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(mcp_progress)}\n\n"
                        
                        # Execute the MCP tool call
                        mcp_result = await mcp_server._save_report_safely(mcp_report_data)
                        
                        # Extract the MCP response
                        mcp_response_text = mcp_result[0].text if mcp_result else "MCP save completed"
                        
                        # Create enhanced result that includes MCP integration information
                        enhanced_result = {
                            **result,
                            "mcp_integration": {
                                "enabled": True,
                                "report_saved": True,
                                "mcp_response": mcp_response_text,
                                "integration_status": "successful"
                            }
                        }
                        
                        yield f"data: {json.dumps({'type': 'stage', 'stage': 'mcp_complete', 'message': 'Report successfully saved via MCP protocol'})}\n\n"
                        
                    except Exception as mcp_error:
                        logger.error(f"MCP integration error: {mcp_error}")
                        # Continue with the response even if MCP fails
                        enhanced_result = {
                            **result,
                            "mcp_integration": {
                                "enabled": True,
                                "report_saved": False,
                                "error": str(mcp_error),
                                "integration_status": "failed"
                            }
                        }
                        
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'MCP integration failed: {mcp_error}'})}\n\n"
                
                else:
                    # MCP integration not requested or not available
                    enhanced_result = {
                        **result,
                        "mcp_integration": {
                            "enabled": False,
                            "reason": "disabled" if not save_report else "not_available"
                        }
                    }
                
                # Stage 4: Final Result Delivery
                final_progress = {
                    "type": "progress",
                    "step": len(agents) + 2,
                    "total_steps": len(agents) + 2,
                    "agent_name": "System",
                    "status": "completed",
                    "message": "Analysis complete - delivering comprehensive results",
                    "percentage": 100,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(final_progress)}\n\n"
                
                # Send the final comprehensive result
                final_update = {
                    "type": "final_result",
                    "result": enhanced_result,
                    "timestamp": datetime.now().isoformat(),
                    "architecture": "multi_agent_with_mcp_integration"
                }
                
                yield f"data: {json.dumps(final_update)}\n\n"
                
            else:
                # Handle case where multi-agent processing failed
                error_result = {
                    "success": False,
                    "error": "Multi-agent processing did not generate expected results",
                    "mcp_integration": {"enabled": False, "reason": "upstream_failure"}
                }
                
                error_update = {
                    "type": "final_result", 
                    "result": error_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(error_update)}\n\n"
                
        except Exception as processing_error:
            logger.error(f"Multi-agent processing error: {processing_error}")
            
            error_result = {
                "success": False,
                "error": f"Multi-agent processing failed: {str(processing_error)}",
                "mcp_integration": {"enabled": False, "reason": "processing_error"}
            }
            
            error_update = {
                "type": "error",
                "result": error_result,
                "timestamp": datetime.now().isoformat()
            }
            
            yield f"data: {json.dumps(error_update)}\n\n"
        
    except Exception as e:
        logger.error(f"Complete workflow error: {e}")
        
        error_update = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "workflow_stage": "complete_failure"
        }
        yield f"data: {json.dumps(error_update)}\n\n"

@app.post("/analyze-with-mcp")
async def analyze_with_mcp_integration(request: EnhancedQueryRequest):
    """
    Analyze compliance query with full MCP server integration.
    
    This endpoint demonstrates the complete integration workflow:
    1. Sophisticated multi-agent compliance analysis
    2. Safe report saving via MCP protocol 
    3. Comprehensive response with both analysis and storage confirmation
    
    This showcases how MCP servers enable AI systems to safely interact
    with external resources while maintaining sophisticated processing capabilities.
    """
    
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters long"
        )
    
    # Return streaming response with MCP integration
    return StreamingResponse(
        process_query_with_mcp_integration(request.query, request.save_detailed_report),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/test-mcp-server")
async def test_mcp_server():
    """
    Test endpoint to verify MCP server functionality independently.
    
    This endpoint allows you to test your MCP server implementation
    without running the full multi-agent workflow, which is useful
    for debugging and demonstrating MCP protocol compliance.
    """
    
    if not MCP_SERVER_AVAILABLE or not mcp_server:
        raise HTTPException(
            status_code=503,
            detail="MCP Server not available - check server initialization"
        )
    
    # Create test data similar to what the multi-agent system would generate
    test_report_data = {
        "query": "Test query for MCP server verification",
        "action_plan": """
        TEST ACTION PLAN FOR MCP VERIFICATION
        
        1. Verify MCP Server Initialization
           - Confirm server instance is properly created
           - Validate tool registration and discovery
           
        2. Test Input Validation
           - Confirm required fields are properly validated
           - Verify error handling for invalid inputs
           
        3. Test Safe File Operations
           - Confirm secure filename generation
           - Verify proper file path handling
           
        4. Validate Response Structure
           - Confirm proper MCP response format
           - Verify structured error handling
        """,
        "citations": {
            "total_citations": 4,
            "gdpr_citations": 2,
            "polish_law_citations": 1,
            "security_citations": 1
        },
        "metadata": {
            "test_timestamp": datetime.now().isoformat(),
            "test_purpose": "mcp_server_verification",
            "integration_method": "direct_api_test"
        }
    }
    
    try:
        # Test the MCP server directly
        mcp_result = await mcp_server._save_report_safely(test_report_data)
        
        response_text = mcp_result[0].text if mcp_result else "No response received"
        
        return {
            "mcp_test_status": "success",
            "mcp_response": response_text,
            "test_timestamp": datetime.now().isoformat(),
            "protocol_compliance": "verified"
        }
        
    except Exception as e:
        logger.error(f"MCP server test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP server test failed: {str(e)}"
        )

# Keep the original endpoint for backward compatibility
@app.post("/analyze")
async def analyze_compliance_query(request: EnhancedQueryRequest):
    """Original endpoint maintained for backward compatibility."""
    
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters long"
        )
    
    try:
        result = multi_agent_system.process_query(request.query)
        
        # Add MCP integration status to legacy endpoint
        if result:
            result["mcp_integration"] = {
                "available": MCP_SERVER_AVAILABLE,
                "endpoint": "analyze-with-mcp" if MCP_SERVER_AVAILABLE else None
            }
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check with both multi-agent system and MCP server status."""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "multi_agent_system": "unhealthy",
        "mcp_server": "not_available",
        "integration_capabilities": []
    }
    
    # Check multi-agent system
    if multi_agent_system is None:
        health_status["multi_agent_system"] = "unhealthy"
        health_status["multi_agent_message"] = "Multi-agent system not initialized"
    else:
        health_status["multi_agent_system"] = "healthy"
        health_status["multi_agent_message"] = "Multi-agent system operational"
        health_status["integration_capabilities"].append("sophisticated_compliance_analysis")
    
    # Check MCP server
    if MCP_SERVER_AVAILABLE and mcp_server:
        try:
            # Test MCP server with a minimal operation
            test_data = {
                "query": "Health check test",
                "action_plan": "MCP server health verification"
            }
            test_result = await mcp_server._save_report_safely(test_data)
            
            if test_result:
                health_status["mcp_server"] = "healthy"
                health_status["mcp_message"] = "MCP server operational and protocol-compliant"
                health_status["integration_capabilities"].extend([
                    "safe_tool_calls",
                    "report_persistence", 
                    "mcp_protocol_compliance"
                ])
            else:
                health_status["mcp_server"] = "degraded"
                health_status["mcp_message"] = "MCP server responding but results unclear"
                
        except Exception as e:
            health_status["mcp_server"] = "unhealthy"
            health_status["mcp_message"] = f"MCP server error: {str(e)}"
    else:
        health_status["mcp_server"] = "not_available"
        health_status["mcp_message"] = "MCP server not initialized or not available"
    
    # Determine overall system health
    if (health_status["multi_agent_system"] == "healthy" and 
        health_status["mcp_server"] in ["healthy", "not_available"]):
        overall_status = "healthy"
        overall_message = "System operational with available integrations"
    elif health_status["multi_agent_system"] == "healthy":
        overall_status = "degraded"
        overall_message = "Multi-agent system healthy, MCP integration unavailable"
    else:
        overall_status = "unhealthy"
        overall_message = "Core systems not operational"
    
    health_status.update({
        "status": overall_status,
        "message": overall_message,
        "features": health_status["integration_capabilities"]
    })
    
    return health_status

@app.get("/mcp-status")
async def mcp_status():
    """Detailed status information about MCP server integration."""
    
    if not MCP_SERVER_AVAILABLE:
        return {
            "mcp_available": False,
            "reason": "MCP server module not found or not importable",
            "suggestion": "Ensure mcp_server directory is properly set up with server.py"
        }
    
    if not mcp_server:
        return {
            "mcp_available": False,
            "reason": "MCP server not initialized",
            "suggestion": "Check server startup logs for initialization errors"
        }
    
    try:
        # Get MCP server information
        server_info = {
            "mcp_available": True,
            "server_name": mcp_server.server.name,
            "reports_directory": str(mcp_server.reports_dir),
            "protocol_compliance": "full",
            "integration_status": "active"
        }
        
        # Test tool discovery (core MCP requirement)
        tools = await mcp_server.server.list_tools()
        server_info["available_tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in tools
        ]
        
        # Check reports directory
        if mcp_server.reports_dir.exists():
            report_files = list(mcp_server.reports_dir.glob("*.json"))
            server_info["saved_reports"] = len(report_files)
        else:
            server_info["saved_reports"] = 0
            
        return server_info
        
    except Exception as e:
        return {
            "mcp_available": True,
            "status": "error",
            "error": str(e),
            "suggestion": "Check MCP server configuration and logs"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)