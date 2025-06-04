"""
FastAPI Backend for Enhanced Multi-Agent Compliance System

This module creates a web API interface for the sophisticated multi-agent system
while preserving all the architectural sophistication of the underlying components.
The API acts as a bridge between web requests and the complex agent orchestration,
demonstrating how to expose sophisticated functionality through simple interfaces.

Note: Citation formatting is now handled by the frontend to allow for flexible
presentation styles (numbered lists, structured displays, etc.)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import traceback
from datetime import datetime

# Import your existing sophisticated system
from main import EnhancedMultiAgentSystem

# Set up logging for the API layer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI-Backend")

app = FastAPI(
    title="Enhanced Multi-Agent Compliance API",
    description="Sophisticated compliance analysis using GDPR, Polish Law, and Internal Security agents",
    version="1.1.0"  # Updated version to reflect citation improvements
)

# Enable CORS for frontend communication
# This allows your Streamlit frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sophisticated multi-agent system once at startup
# This preserves the expensive initialization (loading vector stores, etc.)
multi_agent_system = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the multi-agent system at startup.
    
    This approach follows the principle of "initialize once, use many times"
    which is crucial for maintaining performance with your sophisticated
    vector store connections and agent architectures.
    """
    global multi_agent_system
    try:
        logger.info("Initializing Enhanced Multi-Agent System...")
        multi_agent_system = EnhancedMultiAgentSystem()
        logger.info("✅ Multi-Agent System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Multi-Agent System: {e}")
        # In production, you might want to exit here
        raise

# Request/Response models for clean API contracts
class QueryRequest(BaseModel):
    """
    Input model for compliance queries.
    
    This Pydantic model ensures type safety and automatic API documentation.
    It demonstrates how to create clean interfaces for complex systems.
    """
    query: str
    session_id: Optional[str] = None  # For future session tracking
    citation_style: Optional[str] = "numbered"  # New: Allow frontend to specify preferred citation style
    
    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "query": "We need to implement employee monitoring software in our Warsaw office that tracks productivity metrics. What GDPR compliance steps do we need?",
                "session_id": "optional-session-identifier",
                "citation_style": "numbered"
            }
        }

class ComplianceResponse(BaseModel):
    """
    Response model for compliance analysis results.
    
    This structured approach ensures consistent API responses while
    preserving all the sophisticated information your system generates.
    Enhanced to include raw citation data for flexible frontend formatting.
    """
    success: bool
    action_plan: str
    citations: Dict[str, Any]
    raw_citations: Optional[List[Dict[str, Any]]] = None  # New: Raw citation data for formatting
    metadata: Dict[str, Any]
    processing_time: float
    session_id: Optional[str] = None

@app.post("/analyze", response_model=ComplianceResponse)
async def analyze_compliance_query(request: QueryRequest):
    """
    Analyze a compliance query using the sophisticated multi-agent system.
    
    This endpoint demonstrates how to expose complex processing through a simple,
    clean API interface. The key insight is that we preserve all the sophistication
    of your system while making it accessible via standard HTTP requests.
    
    Enhanced to provide both summary citation metrics and raw citation data
    that the frontend can format according to user preferences.
    
    The processing flow:
    1. Validate the input query
    2. Execute the sophisticated multi-agent analysis
    3. Extract and format the results
    4. Return structured response with both summary and detailed citation information
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.query or len(request.query.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Query must be at least 10 characters long and contain meaningful content"
            )
        
        # Log the incoming request for monitoring
        logger.info(f"Processing compliance query: {request.query[:100]}...")
        logger.info(f"Requested citation style: {request.citation_style}")
        
        # Execute the sophisticated multi-agent analysis
        # This is where your existing sophisticated system does its magic
        result = multi_agent_system.process_query(request.query)
        
        # Extract the sophisticated results
        action_plan = result.get("summary", {}).get("action_plan", "No action plan generated")
        
        # Process citations for structured response
        # Your system generates sophisticated citation information that we preserve
        citations_info = {
            "total_citations": result.get("summary", {}).get("total_citations", 0),
            "gdpr_citations": len(result.get("gdpr_citations", [])),
            "polish_law_citations": len(result.get("polish_law_citations", [])),
            "security_citations": len(result.get("internal_policy_citations", [])),
            "precision_rate": result.get("summary", {}).get("overall_precision_rate", 0)
        }
        
        # Extract raw citation data for frontend formatting
        # This allows the frontend to implement different citation styles
        raw_citations = []
        
        # Collect GDPR citations
        for citation in result.get("gdpr_citations", []):
            raw_citations.append({
                "source": "European Data Protection Regulation (GDPR)",
                "text": citation.get("text", ""),
                "article": citation.get("article", ""),
                "chapter": citation.get("chapter", ""),
                "type": "gdpr"
            })
        
        # Collect Polish law citations
        for citation in result.get("polish_law_citations", []):
            raw_citations.append({
                "source": "Polish Data Protection Implementation",
                "text": citation.get("text", ""),
                "article": citation.get("article", ""),
                "law": citation.get("law", ""),
                "type": "polish_law"
            })
        
        # Collect internal policy citations
        for citation in result.get("internal_policy_citations", []):
            raw_citations.append({
                "source": "Internal Security Procedures",
                "text": citation.get("text", ""),
                "procedure": citation.get("procedure", ""),
                "section": citation.get("section", ""),
                "type": "internal_policy"
            })
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create metadata about the analysis
        metadata = {
            "agent_coordination": "enhanced_multi_agent_system",
            "domains_analyzed": ["gdpr", "polish_law", "internal_security"],
            "analysis_timestamp": start_time.isoformat(),
            "processing_time_seconds": processing_time,
            "architecture": "component-based_with_sophisticated_orchestration",
            "citation_style_requested": request.citation_style
        }
        
        logger.info(f"✅ Query processed successfully in {processing_time:.3f} seconds")
        logger.info(f"Generated {len(raw_citations)} detailed citations")
        
        return ComplianceResponse(
            success=True,
            action_plan=action_plan,
            citations=citations_info,
            raw_citations=raw_citations,
            metadata=metadata,
            processing_time=processing_time,
            session_id=request.session_id
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        # Handle unexpected errors gracefully
        processing_time = (datetime.now() - start_time).total_seconds()
        error_trace = traceback.format_exc()
        
        logger.error(f"❌ Error processing query: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal processing error",
                "message": str(e),
                "processing_time": processing_time,
                "suggestion": "Please check your query format and try again"
            }
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring system status.
    
    This provides insight into whether the sophisticated multi-agent system
    is properly initialized and ready to handle queries.
    """
    if multi_agent_system is None:
        return {"status": "unhealthy", "message": "Multi-agent system not initialized"}
    
    try:
        # Get system summary to verify all agents are working
        system_summary = multi_agent_system.get_system_summary()
        
        return {
            "status": "healthy",
            "message": "Enhanced multi-agent system operational",
            "agents": system_summary.get("agents", {}),
            "timestamp": datetime.now().isoformat(),
            "features": ["numbered_citations", "raw_citation_data", "flexible_formatting"]
        }
    except Exception as e:
        return {
            "status": "degraded", 
            "message": f"System partially operational: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Enhanced Multi-Agent Compliance API",
        "description": "Sophisticated GDPR, Polish Law, and Internal Security analysis",
        "endpoints": {
            "/analyze": "POST - Analyze compliance queries with flexible citation formatting",
            "/health": "GET - System health check",
            "/docs": "GET - Interactive API documentation"
        },
        "architecture": "Component-based multi-agent system with sophisticated orchestration",
        "features": ["numbered_citations", "citation_style_selection", "raw_citation_access"]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    # The reload=True helps during development
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )