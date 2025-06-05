"""
Enhanced Multi-Agent Compliance System - Streamlit Frontend with Sample Query Integration

This enhanced version demonstrates how to create progressive user interfaces that guide
users from simple examples to sophisticated custom queries. The key insight here is
that complex AI systems become more accessible when users can see what's possible
through concrete examples before diving into custom scenarios.

The interface design principles enhanced in this version:
- Guided discovery: Users explore capabilities through curated examples
- Progressive disclosure: Simple examples first, then customization options
- Educational scaffolding: Users learn effective query patterns through interaction
- Seamless workflow: Sample selection integrates smoothly with custom query input
- Clear feedback: Users understand system capabilities and how to leverage them

This approach teaches several important UX design concepts:
- Reducing cognitive load through progressive enhancement
- Providing multiple entry points for different user experience levels
- Creating educational experiences that build user confidence
- Balancing guided workflows with expert-level flexibility

The sample query integration transforms a potentially intimidating AI interface
into an approachable tool that teaches users how to formulate effective compliance
questions while demonstrating the sophisticated capabilities of the multi-agent system.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional

# Import our enhanced modular components with sample query support
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
    display_enhanced_query_interface,  # Updated function with sample queries
    display_processing_feedback,
    display_success_results,
    display_error_results,
    display_chat_history
)
# Import the sample query system for validation and stats
from sample_queries import validate_sample_queries, get_category_stats


def initialize_session_state():
    """
    Initialize Streamlit session state variables for chat history and session tracking.
    
    Enhanced to include sample query system state management, ensuring that
    user interactions with sample queries are properly tracked and maintained
    across page refreshes. This provides a smoother user experience and enables
    better analytics about which sample queries are most effective.
    
    The session state design here demonstrates important principles for stateful
    web applications:
    - Clear separation between user data and system state
    - Defensive initialization that handles missing or corrupted state
    - Meaningful default values that provide good user experience
    - Future-proofing for additional features like query history analysis
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        # Create a simple session ID for potential backend session tracking
        st.session_state.session_id = f"session_{int(time.time())}"
    
    # Enhanced session state for sample query functionality
    if "selected_query_text" not in st.session_state:
        st.session_state.selected_query_text = ""
    
    if "last_sample_category" not in st.session_state:
        st.session_state.last_sample_category = None
    
    if "sample_usage_count" not in st.session_state:
        st.session_state.sample_usage_count = 0


def check_system_readiness() -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify that both the backend system and sample query system are ready for processing.
    
    Enhanced to include validation of the sample query system, ensuring that users
    have access to both the AI processing capabilities and the guided example system.
    This comprehensive readiness check prevents users from encountering issues
    after they've invested time in query formulation.
    
    The enhanced readiness check demonstrates several important system design principles:
    - Fail-fast validation to prevent downstream issues
    - Comprehensive system health monitoring across all components
    - Clear communication to users about system capabilities and limitations
    - Graceful degradation when some features are unavailable
    
    Returns:
        tuple: (system_ready: bool, system_info: Optional[Dict])
    """
    # Check backend health as before
    backend_healthy = check_backend_health()
    
    if not backend_healthy:
        return False, None
    
    # Validate the sample query system
    sample_validation = validate_sample_queries()
    if not sample_validation["valid"]:
        st.warning("⚠️ Sample query system has validation issues, but backend is operational.")
        # Log the issues for debugging
        for issue in sample_validation["issues"]:
            st.sidebar.error(f"Sample Query Issue: {issue}")
    
    # Get detailed system information for the sidebar display
    system_info = get_backend_system_info()
    
    # Add sample query statistics to system info
    if system_info:
        sample_stats = get_category_stats()
        system_info["sample_system"] = {
            "categories": sample_stats["total_categories"],
            "queries": sample_stats["total_queries"],
            "validation_status": "valid" if sample_validation["valid"] else "issues_found"
        }
    
    return True, system_info


def process_user_query(query: str) -> Dict[str, Any]:
    """
    Process a user query through the backend system with proper error handling.
    
    Enhanced to track whether the query originated from a sample selection,
    enabling better analytics and user guidance. Understanding query patterns
    helps improve both the sample collection and the AI system's performance.
    
    This enhanced processing function demonstrates several important patterns:
    - Transparent user feedback during long-running operations
    - Comprehensive error handling with actionable guidance
    - User behavior analytics for continuous system improvement
    - Context preservation for better user experience
    
    Args:
        query: The validated user compliance question
        
    Returns:
        Dict containing processing results or error information with enhanced metadata
    """
    # Show processing feedback to keep users informed
    display_processing_feedback()
    
    # Track whether this query came from a sample (for analytics)
    query_metadata = {
        "is_sample_based": bool(st.session_state.selected_query_text and 
                              st.session_state.selected_query_text.strip() in query),
        "session_sample_count": st.session_state.sample_usage_count,
        "last_sample_category": st.session_state.last_sample_category
    }
    
    # Send query to the sophisticated backend system
    with st.spinner("🧠 Analyzing your compliance query with multi-agent system..."):
        result = send_query_to_backend(query, st.session_state.session_id)
    
    # Enhance result with query metadata for better user experience
    if "metadata" not in result:
        result["metadata"] = {}
    result["metadata"].update(query_metadata)
    
    # Store successful queries in chat history for session context
    if result.get("success", False):
        st.session_state.chat_history.append({
            "query": query,
            "response": result,
            "timestamp": time.time(),
            "query_source": "sample" if query_metadata["is_sample_based"] else "custom"
        })
        
        # Update sample usage tracking
        if query_metadata["is_sample_based"]:
            st.session_state.sample_usage_count += 1
    
    return result


def display_system_welcome():
    """
    Display an enhanced welcome section that introduces both custom and sample query capabilities.
    
    This welcome section teaches users about the progressive approach to using the system:
    starting with examples to understand capabilities, then moving to custom scenarios.
    The educational design helps users understand not just how to use the system,
    but why different approaches work better for different situations.
    
    This demonstrates important principles for complex system onboarding:
    - Multiple learning pathways for different user comfort levels
    - Clear explanation of system capabilities and expected outcomes
    - Gentle guidance toward best practices without overwhelming new users
    - Progressive disclosure of advanced features as users gain confidence
    """
    
    # Enhanced welcome message with usage guidance
    st.markdown("""
    ### 🚀 Welcome to Enhanced Compliance Analysis
    
    Our sophisticated multi-agent system combines **GDPR expertise**, **Polish law knowledge**, 
    and **internal security procedures** to provide comprehensive compliance guidance for European businesses.
    
    **🎯 Quick Start Options:**
    
    **New to the system?** Browse our **sample business scenarios** below to see what's possible. 
    Each sample demonstrates different aspects of EU compliance analysis and can be used directly 
    or modified for your specific situation.
    
    **Experienced user?** Skip to the **custom query section** below to write your own detailed 
    compliance question, or use sample queries as starting templates.
    
    **💡 Pro Tip:** Even experienced users often find that starting with a relevant sample 
    and modifying it produces better results than writing from scratch, since our samples 
    include the specific details and context that our AI agents need for comprehensive analysis.
    """)


def handle_enhanced_query_submission() -> Optional[str]:
    """
    Handle the enhanced query interface with sample integration and validation logic.
    
    This function replaces the original query submission handler with an enhanced version
    that seamlessly integrates sample query selection with custom query input. The design
    demonstrates how to create interfaces that support multiple user workflows without
    adding complexity or confusion.
    
    Key enhancements include:
    - Integrated sample query selector with custom input
    - Preserved validation logic for quality assurance
    - Enhanced user guidance based on query source (sample vs custom)
    - Better error messaging that guides users toward successful patterns
    
    Returns:
        Optional[str]: Validated query ready for processing, or None if invalid
    """
    
    # Display the enhanced query interface with sample integration
    user_query = display_enhanced_query_interface()
    
    # If a query was submitted, validate it before processing
    if user_query:
        # Perform local validation before processing
        validation_result = validate_query_locally(user_query)
        
        if not validation_result.get("valid", False):
            # Enhanced validation error display with sample query guidance
            st.warning(validation_result.get("error", "Invalid query"))
            
            if validation_result.get("suggestion"):
                st.info(f"💡 **Suggestion:** {validation_result['suggestion']}")
            
            # Provide additional guidance for common validation issues
            if validation_result.get("error_type") == "too_short":
                st.info("""
                📝 **Quick Fix:** Try selecting one of our sample queries above to see the level of 
                detail that works best, then modify it for your specific scenario. Our samples 
                include the context and specificity that our AI agents need for accurate analysis.
                """)
            elif validation_result.get("error_type") == "empty_query":
                st.info("""
                🎯 **Getting Started:** Browse the sample business scenarios above to see examples 
                of effective compliance questions, or click "Use This Query" on any sample that's 
                similar to your situation.
                """)
            
            return None
        
        return user_query
    
    return None


def display_session_analytics():
    """
    Display helpful analytics about the user's session to encourage effective usage patterns.
    
    This analytics section teaches users about their own usage patterns and helps them
    understand which approaches tend to work best. The goal is educational rather than
    judgmental - we want users to learn from their experience and become more effective
    at formulating compliance questions.
    
    This demonstrates important principles for user education through analytics:
    - Positive reinforcement for effective behaviors
    - Gentle guidance toward best practices
    - Transparency about system capabilities and user success patterns
    - Privacy-focused analytics that help users without compromising their data
    """
    
    if len(st.session_state.chat_history) > 0:
        with st.expander("📊 Your Session Analytics", expanded=False):
            successful_queries = [q for q in st.session_state.chat_history if q['response'].get('success')]
            sample_based_queries = [q for q in st.session_state.chat_history if q.get('query_source') == 'sample']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Queries This Session",
                    len(st.session_state.chat_history),
                    help="Total number of compliance questions analyzed"
                )
            
            with col2:
                success_rate = len(successful_queries) / len(st.session_state.chat_history) * 100
                st.metric(
                    "Success Rate",
                    f"{success_rate:.0f}%",
                    help="Percentage of queries that generated successful analysis"
                )
            
            with col3:
                sample_usage = len(sample_based_queries) / len(st.session_state.chat_history) * 100
                st.metric(
                    "Sample-Based Queries",
                    f"{sample_usage:.0f}%",
                    help="Percentage of queries based on sample scenarios"
                )
            
            # Provide personalized guidance based on session patterns
            if success_rate < 70 and len(st.session_state.chat_history) > 2:
                st.info("""
                💡 **Tip:** Try using our sample queries as starting points. Users who begin with 
                samples and modify them tend to get better results than those who write entirely 
                custom queries from scratch.
                """)
            elif success_rate > 90 and len(st.session_state.chat_history) > 3:
                st.success("""
                🎉 **Excellent!** You're getting great results from the system. Consider sharing 
                your effective query patterns with colleagues who might benefit from similar analysis.
                """)


def main():
    """
    Enhanced main application flow orchestrating all components with sample query integration.
    
    This enhanced main function demonstrates how to evolve complex applications by adding
    new capabilities while maintaining existing functionality. The key insight is that
    we enhance the user experience without breaking existing workflows, allowing both
    novice and expert users to be successful.
    
    The enhanced flow demonstrates several important application design principles:
    - Backward compatibility: Existing functionality continues to work
    - Progressive enhancement: New features improve the experience without complexity
    - Multiple user pathways: Different approaches for different experience levels
    - Educational design: The interface teaches users how to be more effective
    - Comprehensive error handling: Issues are resolved rather than just reported
    
    Enhanced flow pattern:
    1. Initialize enhanced session state with sample query support
    2. Check comprehensive system readiness (backend + sample system)
    3. Display enhanced header with usage guidance
    4. Provide sample query integration in sidebar
    5. Handle enhanced query interface with sample integration
    6. Process queries with enhanced metadata and analytics
    7. Display results with context about query source and effectiveness
    8. Maintain enhanced session context with usage pattern insights
    """
    # Configure Streamlit page settings
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize enhanced session management with sample query support
    initialize_session_state()
    
    # Check comprehensive system readiness (backend + sample system)
    system_ready, system_info = check_system_readiness()
    
    # Render enhanced application header with usage guidance
    display_app_header()
    
    # Display enhanced welcome section with progressive usage guidance
    display_system_welcome()
    
    # Render enhanced sidebar with system status and sample query statistics
    display_sidebar_info(system_ready, system_info)
    
    # Stop execution if core system is not ready
    if not system_ready:
        st.error("🚫 **System Not Ready:** Please ensure the FastAPI backend is running on port 8000.")
        st.info("💡 **Quick Fix:** Check that your backend server is started and accessible.")
        
        # Even if backend is down, show sample query system status for transparency
        sample_validation = validate_sample_queries()
        if sample_validation["valid"]:
            st.info("✅ **Sample Query System:** Available for browsing examples (backend required for analysis)")
        else:
            st.warning("⚠️ **Sample Query System:** Also experiencing issues")
        
        st.stop()
    
    # Handle enhanced query input with sample integration and validation
    user_query = handle_enhanced_query_submission()
    
    # Process query if user submitted one
    if user_query:
        # Process the query using the backend system with enhanced tracking
        processing_result = process_user_query(user_query)
        
        # Display results based on success or failure with enhanced context
        if processing_result.get("success", False):
            display_success_results(processing_result)
        else:
            display_error_results(processing_result)
    
    # Display enhanced session history with usage pattern insights
    display_chat_history(st.session_state.chat_history)
    
    # Show session analytics to help users understand their usage patterns
    display_session_analytics()


# Enhanced application entry point with comprehensive error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 **Application Error:** {str(e)}")
        st.info("""
        **Troubleshooting Steps:**
        1. Refresh the page to restart the application
        2. Check that all required files are present in the project directory
        3. Verify that the FastAPI backend is running on port 8000
        4. Contact support if the issue persists
        """)
        
        # Show technical details for debugging in expander
        with st.expander("🔧 Technical Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())