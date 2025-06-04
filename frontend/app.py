"""
Enhanced Multi-Agent Compliance System - Streamlit Frontend

This frontend demonstrates how to create an intuitive interface for sophisticated
AI systems. The design philosophy is to hide complexity while showcasing capability -
users see a simple chat interface, but behind the scenes they're accessing a
sophisticated multi-agent architecture with GDPR, Polish Law, and Security expertise.

The interface design principles:
- Progressive disclosure: Show simple interface first, reveal sophistication in results
- Clear feedback: Users understand what's happening during processing
- Citation preservation: Complex citation structures are presented in accessible formats
- Educational value: Results help users understand compliance requirements

Enhanced with sophisticated citation formatting that converts complex citation text
into clean, numbered reference lists for improved readability and professional presentation.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional

# Import our modular components - fixed dependency chain
from config import PAGE_CONFIG, UI_MESSAGES, VALIDATION
from backend_client import (
    check_backend_health, 
    get_backend_system_info,
    send_query_to_backend,
    validate_query_locally,
    format_backend_error
)
from ui_components import (
    display_app_header,
    display_sidebar_info,
    display_query_interface,
    display_processing_feedback,
    display_success_results,
    display_error_results,
    display_chat_history
)


def initialize_session_state():
    """
    Initialize Streamlit session state variables for chat history and session tracking.
    
    This centralized initialization makes it easy to understand what persistent
    data our application maintains across user interactions. Session state is
    crucial for compliance analysis where users build understanding through
    multiple related queries.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        # Create a simple session ID for potential backend session tracking
        st.session_state.session_id = f"session_{int(time.time())}"


def check_system_readiness() -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify that the backend system is ready for processing.
    
    This implements the "fail fast" principle - we check system readiness
    before allowing users to invest time in formulating questions. This
    prevents frustration and provides clear feedback when systems are unavailable.
    
    Returns:
        tuple: (system_ready: bool, system_info: Optional[Dict])
    """
    backend_healthy = check_backend_health()
    
    if not backend_healthy:
        return False, None
    
    # Get detailed system information for the sidebar display
    system_info = get_backend_system_info()
    return True, system_info


def process_user_query(query: str) -> Dict[str, Any]:
    """
    Process a user query through the backend system with proper error handling.
    
    This function demonstrates clean separation of concerns - it orchestrates
    the query processing workflow without getting involved in UI details or
    backend communication specifics. This makes it easy to modify the
    processing logic without affecting other parts of the application.
    
    Args:
        query: The validated user compliance question
        
    Returns:
        Dict containing processing results or error information
    """
    # Show processing feedback to keep users informed
    display_processing_feedback()
    
    # Send query to the sophisticated backend system
    with st.spinner("🧠 Analyzing your compliance query with multi-agent system..."):
        result = send_query_to_backend(query, st.session_state.session_id)
    
    # Store successful queries in chat history for session context
    if result.get("success", False):
        st.session_state.chat_history.append({
            "query": query,
            "response": result,
            "timestamp": time.time()
        })
    
    return result


def handle_query_submission() -> Optional[str]:
    """
    Handle the query interface and validation logic.
    
    This function encapsulates the query input and validation workflow,
    demonstrating how to build reusable interaction patterns. By separating
    this logic, we can easily modify validation rules or input methods
    without affecting the main application flow.
    
    Returns:
        Optional[str]: Validated query ready for processing, or None if invalid
    """
    st.header("💬 Compliance Query Interface")
    
    # Query input with sophisticated placeholder and help text
    query = st.text_area(
        "Enter your compliance question:",
        height=100,
        placeholder=UI_MESSAGES["query_placeholder"],
        help=UI_MESSAGES["query_help"]
    )
    
    # Analysis button
    if st.button("🔍 Analyze Compliance Requirements", type="primary"):
        # Perform local validation before processing
        validation_result = validate_query_locally(query)
        
        if not validation_result.get("valid", False):
            # Display validation error with helpful guidance
            st.warning(validation_result.get("error", "Invalid query"))
            
            if validation_result.get("suggestion"):
                st.info(f"💡 **Suggestion:** {validation_result['suggestion']}")
            
            return None
        
        return query
    
    return None


def main():
    """
    Main application flow orchestrating all components.
    
    This simplified main function demonstrates how to build robust applications
    by composing smaller, focused functions. Each function has a single
    responsibility, making the code easier to understand, test, and maintain.
    
    The flow follows a clear pattern:
    1. Initialize application state
    2. Check system readiness  
    3. Display header and navigation
    4. Handle user interactions
    5. Process and display results
    6. Maintain session context
    
    This pattern can be applied to any complex application to maintain
    clarity and reduce cognitive load for developers.
    """
    # Configure Streamlit page settings
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize session management
    initialize_session_state()
    
    # Check system readiness before proceeding
    system_ready, system_info = check_system_readiness()
    
    # Render main application header
    display_app_header()
    
    # Render sidebar with system status and guidance
    display_sidebar_info(system_ready, system_info)
    
    # Stop execution if system is not ready
    if not system_ready:
        st.error("🚫 **System Not Ready:** Please ensure the FastAPI backend is running on port 8000.")
        st.info("💡 **Quick Fix:** Check that your backend server is started and accessible.")
        st.stop()
    
    # Handle query input and validation
    user_query = handle_query_submission()
    
    # Process query if user submitted one
    if user_query:
        # Process the query using the backend system
        processing_result = process_user_query(user_query)
        
        # Display results based on success or failure
        if processing_result.get("success", False):
            display_success_results(processing_result)
        else:
            display_error_results(processing_result)
    
    # Display session history for context and learning
    display_chat_history(st.session_state.chat_history)


# Application entry point with clean module structure
if __name__ == "__main__":
    main()