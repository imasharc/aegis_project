"""
Backend Client Module for Enhanced Multi-Agent Compliance System - Updated Version

This updated version demonstrates how to integrate flexible response handling
into an existing system without breaking backwards compatibility. The key
insight here is that we add the normalization layer while preserving all
existing functionality.

This approach shows you how to evolve software systems gracefully over time,
which is a crucial skill in professional software development where requirements
and data formats change but existing functionality must continue working.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from config import BACKEND_URL, BACKEND_TIMEOUT, HEALTH_CHECK_TIMEOUT

# Import our new response handling capabilities
from enhanced_response_handler import (
    normalize_backend_response,
    validate_normalized_response
)

# Set up logging for debugging purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_backend_health() -> bool:
    """
    Check if the sophisticated backend system is operational.
    Enhanced with detailed logging to help diagnose connection issues.
    
    This function remains unchanged because health checks use a different,
    simpler format that doesn't need normalization. This demonstrates
    how to apply changes selectively - only where they're needed.
    """
    try:
        logger.info(f"Checking backend health at {BACKEND_URL}/health")
        response = requests.get(
            f"{BACKEND_URL}/health", 
            timeout=HEALTH_CHECK_TIMEOUT
        )
        
        logger.info(f"Health check response: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            is_healthy = health_data.get("status") == "healthy"
            logger.info(f"Backend health status: {health_data}")
            return is_healthy
        else:
            logger.warning(f"Backend health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Backend health check failed with exception: {e}")
        return False


def get_backend_system_info() -> Optional[Dict[str, Any]]:
    """
    Retrieve detailed system information from the backend.
    Enhanced with response logging for troubleshooting.
    
    This function also remains unchanged because system info uses the same
    simple format as health checks. We only apply normalization where we
    need to handle the complex multi-agent response format.
    """
    try:
        logger.info("Fetching backend system information")
        response = requests.get(
            f"{BACKEND_URL}/health", 
            timeout=HEALTH_CHECK_TIMEOUT
        )
        
        if response.status_code == 200:
            system_info = response.json()
            logger.info(f"System info retrieved: {system_info}")
            return system_info
        else:
            logger.warning(f"System info request failed with status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"System info request failed: {e}")
        return None


def send_query_to_backend(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Send a compliance query to the sophisticated backend system.
    Enhanced with response normalization to handle multiple backend formats.
    
    This is where the magic happens. We've enhanced this function to automatically
    convert whatever format the backend sends into the format that your frontend
    expects. This means your frontend code doesn't need to change at all, but it
    can now work with the new backend format.
    
    The key insight here is that we handle the complexity at the communication
    boundary, keeping the rest of your application simple and focused on its
    core responsibilities rather than format conversion.
    
    Args:
        query: The compliance question to analyze
        session_id: Optional session identifier for tracking
        
    Returns:
        Dict containing either success results or detailed error information,
        guaranteed to be in the format expected by the frontend
    """
    logger.info(f"Sending query to backend: {query[:100]}...")  # Log first 100 chars
    
    try:
        # Prepare the request payload
        payload = {
            "query": query,
            "citation_style": "numbered"
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        logger.info(f"Request payload prepared: {payload}")
        
        # Make the request with detailed logging
        logger.info(f"Making POST request to {BACKEND_URL}/analyze")
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json=payload,
            timeout=BACKEND_TIMEOUT
        )
        
        logger.info(f"Received response with status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        # Log the raw response text for inspection
        raw_response_text = response.text
        logger.info(f"Raw response length: {len(raw_response_text)} characters")
        logger.info(f"Raw response preview: {raw_response_text[:500]}...")  # First 500 chars
        
        # Handle successful HTTP responses
        if response.status_code == 200:
            try:
                # Parse the JSON response
                raw_response_data = response.json()
                logger.info("Successfully parsed JSON response")
                logger.info(f"Raw response keys: {list(raw_response_data.keys())}")
                
                # Here's the key enhancement: normalize the response format
                # This is where we convert whatever format the backend sent
                # into the format that your frontend expects
                logger.info("Normalizing response format for frontend compatibility")
                normalized_response = normalize_backend_response(raw_response_data)
                
                # Validate that the normalization worked correctly
                if validate_normalized_response(normalized_response):
                    logger.info("Response normalization successful")
                    logger.info(f"Normalized response keys: {list(normalized_response.keys())}")
                    
                    # Log some key metrics about what we converted
                    if normalized_response.get("citations"):
                        total_citations = normalized_response["citations"].get("total_citations", 0)
                        logger.info(f"Successfully normalized {total_citations} citations")
                    
                    return normalized_response
                else:
                    logger.error("Response normalization failed validation")
                    return {
                        "success": False,
                        "error": "Failed to normalize backend response to expected format",
                        "backend_response": raw_response_data
                    }
                        
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response causing JSON error: {raw_response_text}")
                
                return {
                    "success": False,
                    "error": f"Backend returned invalid JSON: {str(e)}",
                    "raw_response": raw_response_text[:1000],  # First 1000 chars for debugging
                    "json_error": True
                }
        
        # Handle HTTP error responses
        else:
            logger.error(f"HTTP error response: {response.status_code}")
            
            try:
                error_detail = response.json().get("detail", f"HTTP {response.status_code} error")
                logger.error(f"Error detail from backend: {error_detail}")
            except ValueError:
                error_detail = f"HTTP {response.status_code} error (no JSON detail)"
                logger.error("Backend error response was not valid JSON")
            
            return {
                "success": False,
                "error": f"Backend error: {error_detail}",
                "status_code": response.status_code,
                "http_error": True,
                "raw_response": response.text[:1000]  # For debugging
            }
            
    except requests.exceptions.Timeout:
        logger.error("Request timeout occurred")
        return {
            "success": False,
            "error": "Analysis timeout - your query might be very complex. Please try a more specific question.",
            "timeout": True,
            "suggestion": "Consider breaking your question into smaller, more focused parts."
        }
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return {
            "success": False,
            "error": "Cannot connect to backend service. Please check if the API server is running.",
            "connection_error": True,
            "suggestion": "Ensure the FastAPI backend is running on the correct port."
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return {
            "success": False,
            "error": f"Network error: {str(e)}",
            "connection_error": True,
            "suggestion": "Please check your network connection and try again."
        }


def validate_query_locally(query: str) -> Dict[str, Any]:
    """
    Perform client-side validation before sending to backend.
    Enhanced with more detailed validation feedback.
    
    This function remains unchanged because local validation doesn't depend
    on response format. This demonstrates how changes can be surgical -
    we only modify the parts that need to change.
    """
    logger.info(f"Validating query locally: {len(query)} characters")
    
    # Check for empty or whitespace-only queries
    if not query or not query.strip():
        logger.warning("Query validation failed: empty query")
        return {
            "valid": False,
            "error": "Please enter a compliance question to analyze.",
            "error_type": "empty_query"
        }
    
    # Check minimum length for meaningful analysis
    if len(query.strip()) < 10:
        logger.warning(f"Query validation failed: too short ({len(query.strip())} chars)")
        return {
            "valid": False,
            "error": "Please provide a more detailed question for accurate analysis.",
            "error_type": "too_short",
            "suggestion": "Include specific details about your business scenario, location, and data types."
        }
    
    # Check for extremely long queries that might cause issues
    if len(query) > 5000:
        logger.warning(f"Query validation failed: too long ({len(query)} chars)")
        return {
            "valid": False,
            "error": "Query is too long. Please break it into smaller, more focused questions.",
            "error_type": "too_long",
            "suggestion": "Consider focusing on one specific compliance aspect at a time."
        }
    
    # Query passes local validation
    logger.info("Query validation passed")
    return {
        "valid": True,
        "message": "Query ready for backend analysis"
    }


def format_backend_error(error_response: Dict[str, Any]) -> str:
    """
    Format backend error responses into user-friendly messages.
    Enhanced with debugging information when available.
    
    This function also remains largely unchanged, but now it might receive
    normalized error responses, which it can handle without modification.
    This demonstrates how good error handling design is resilient to
    changes in other parts of the system.
    """
    base_error = error_response.get("error", "An unknown error occurred.")
    
    # Include debugging information if available (for development)
    debug_info = []
    
    if error_response.get("backend_response"):
        debug_info.append("Original backend response available in logs")
    
    if error_response.get("raw_response"):
        debug_info.append("Raw response data available in logs")
    
    # Build the formatted error message
    formatted_error = base_error
    
    # Add specific suggestions based on error type
    if error_response.get("timeout"):
        formatted_error += f"\n\n💡 **Suggestion:** {error_response.get('suggestion', 'Try a more specific question.')}"
    
    elif error_response.get("connection_error"):
        formatted_error += f"\n\n🔧 **Technical Note:** {error_response.get('suggestion', 'Check backend service status.')}"
    
    elif error_response.get("http_error"):
        status_code = error_response.get("status_code", "unknown")
        formatted_error += f"\n\n📋 **HTTP Status:** {status_code}"
    
    elif error_response.get("json_error"):
        formatted_error += "\n\n🔍 **Debug:** The backend response was not valid JSON. Check the backend logs for details."
    
    # Add debug information for development (you can remove this in production)
    if debug_info:
        formatted_error += f"\n\n🐛 **Debug Info:** {' | '.join(debug_info)}"
    
    return formatted_error