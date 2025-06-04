"""
Real-Time Progress Tracking Components for Enhanced Multi-Agent Compliance System

This module implements sophisticated progress tracking for long-running AI processes,
demonstrating how to create engaging user experiences during complex operations.

The architecture here solves a fundamental challenge in AI applications: how to keep
users engaged and informed during processing that can take anywhere from 10 seconds
to several minutes. Rather than showing a static spinner, we provide detailed,
real-time feedback about what's happening behind the scenes.

Key concepts demonstrated:
- Server-Sent Events (SSE) for real-time communication
- Progressive UI updates with meaningful feedback
- Graceful handling of network interruptions
- Professional progress visualization for enterprise applications
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from config import BACKEND_URL

@dataclass
class AgentProgress:
    """
    Data structure for tracking individual agent progress.
    
    Using a dataclass here demonstrates clean data modeling - we define
    exactly what information we need to track for each agent, making the
    code more maintainable and self-documenting.
    """
    name: str
    status: str = "pending"  # pending, processing, completed, error
    progress: float = 0.0
    current_stage: str = ""
    message: str = ""
    completion_time: Optional[float] = None

class ProgressTracker:
    """
    Manages progress tracking state and UI updates for the compliance analysis process.
    
    This class encapsulates the complexity of managing real-time progress updates
    while providing a clean interface for the UI components. The separation of
    concerns here makes it easy to modify the progress tracking logic without
    affecting the display components.
    
    The class demonstrates several important patterns:
    - State management for complex UI interactions
    - Event handling for streaming data
    - Progressive enhancement of user feedback
    """
    
    def __init__(self):
        self.agents: List[AgentProgress] = []
        self.overall_progress: float = 0.0
        self.start_time: Optional[float] = None
        self.is_complete: bool = False
        self.error_message: Optional[str] = None
        
        # Initialize agent list with expected agents
        agent_names = [
            "GDPR Agent",
            "Polish Law Agent", 
            "Security Agent",
            "Integration Agent",
            "Citation Agent"
        ]
        
        for name in agent_names:
            self.agents.append(AgentProgress(name=name))
    
    def update_agent_progress(self, agent_name: str, progress_data: Dict[str, Any]):
        """
        Update progress for a specific agent based on streaming data.
        
        This method demonstrates how to handle real-time data updates while
        maintaining UI consistency. The key insight is to update state first,
        then trigger UI refreshes, rather than trying to update the UI directly.
        """
        
        for agent in self.agents:
            if agent.name == agent_name:
                agent.status = progress_data.get("status", "processing")
                agent.current_stage = progress_data.get("stage", "")
                agent.message = progress_data.get("message", "")
                
                if agent.status == "completed":
                    agent.progress = 100.0
                    agent.completion_time = time.time()
                else:
                    # Calculate agent-specific progress
                    step = progress_data.get("step", 1)
                    total_steps = progress_data.get("total_steps", 5)
                    agent.progress = (step / total_steps) * 100
                
                break
        
        # Update overall progress
        self.overall_progress = progress_data.get("percentage", 0.0)
    
    def mark_error(self, error_message: str):
        """Handle error states gracefully with user-friendly messaging."""
        self.error_message = error_message
        self.is_complete = True
    
    def mark_complete(self):
        """Mark the entire process as complete."""
        self.is_complete = True
        self.overall_progress = 100.0
        
        # Ensure all agents are marked as completed
        for agent in self.agents:
            if agent.status != "completed":
                agent.status = "completed"
                agent.progress = 100.0

def create_progress_display(tracker: ProgressTracker) -> None:
    """
    Create a sophisticated progress display with real-time updates.
    
    This function demonstrates how to build engaging progress interfaces that
    provide users with detailed insight into complex processing. The design
    follows principles of progressive disclosure - showing overview information
    immediately, with detailed agent status available on demand.
    
    The visual design here balances information density with clarity, ensuring
    users understand both overall progress and specific agent activity without
    feeling overwhelmed by technical details.
    """
    
    # Overall progress header with time estimation
    st.markdown("### 🧠 Analyzing Your Compliance Query")
    
    # Main progress bar with sophisticated styling
    progress_container = st.container()
    with progress_container:
        # Custom CSS for enhanced progress bar styling
        st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00C851 0%, #007E33 100%);
        }
        .agent-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .agent-completed {
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .agent-processing {
            background-color: #fff3cd;
            border-color: #ffeaa7;
        }
        .agent-pending {
            background-color: #f8f9fa;
            border-color: #e9ecef;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Overall progress bar
        st.progress(tracker.overall_progress / 100.0)
        
        # Progress statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completed_agents = len([a for a in tracker.agents if a.status == "completed"])
            st.metric("Agents Complete", f"{completed_agents}/{len(tracker.agents)}")
        
        with col2:
            if tracker.start_time:
                elapsed = time.time() - tracker.start_time
                st.metric("Elapsed Time", f"{elapsed:.1f}s")
            else:
                st.metric("Elapsed Time", "0.0s")
        
        with col3:
            st.metric("Overall Progress", f"{tracker.overall_progress:.1f}%")
    
    # Agent-specific progress with expandable details
    st.markdown("#### 🤖 Agent Activity")
    
    for agent in tracker.agents:
        # Determine visual styling based on status
        if agent.status == "completed":
            status_emoji = "✅"
            card_class = "agent-completed"
        elif agent.status == "processing":
            status_emoji = "⚙️"
            card_class = "agent-processing"
        elif agent.status == "error":
            status_emoji = "❌"
            card_class = "agent-card"
        else:
            status_emoji = "⏳"
            card_class = "agent-pending"
        
        # Create agent progress card
        with st.expander(f"{status_emoji} {agent.name} - {agent.status.title()}", 
                        expanded=(agent.status == "processing")):
            
            if agent.status == "processing":
                st.progress(agent.progress / 100.0)
                st.write(f"**Current Stage:** {agent.current_stage}")
                st.write(f"**Activity:** {agent.message}")
                
                # Show spinning indicator for active processing
                st.markdown("🔄 *Processing...*")
                
            elif agent.status == "completed":
                st.progress(1.0)
                st.success(f"✅ {agent.message}")
                
                if agent.completion_time and tracker.start_time:
                    completion_duration = agent.completion_time - tracker.start_time
                    st.write(f"⏱️ Completed in {completion_duration:.1f} seconds")
                    
            elif agent.status == "pending":
                st.progress(0.0)
                st.info("⏳ Waiting to start...")
                
            else:  # error status
                st.progress(0.0)
                st.error(f"❌ Error: {agent.message}")

def stream_progress_updates(query: str, session_id: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Handle Server-Sent Events (SSE) streaming from the backend.
    
    This function implements the client-side SSE handling, demonstrating how to
    consume real-time data streams from your backend API. SSE is perfect for
    progress tracking because it provides automatic reconnection and is much
    simpler than WebSockets for this use case.
    
    The implementation here includes robust error handling and fallback mechanisms,
    ensuring the application gracefully handles network interruptions or backend
    issues without leaving users in an undefined state.
    
    Args:
        query: The compliance question to analyze
        session_id: Optional session identifier
        
    Yields:
        Dict containing progress updates or final results
    """
    
    try:
        # Prepare the request for streaming analysis
        payload = {
            "query": query,
            "session_id": session_id,
            "enable_progress": True
        }
        
        # Make streaming request to the enhanced backend
        response = requests.post(
            f"{BACKEND_URL}/analyze-progressive",
            json=payload,
            stream=True,
            timeout=300  # 5 minute timeout for complex queries
        )
        
        if response.status_code != 200:
            yield {
                "type": "error",
                "error": f"Backend error: HTTP {response.status_code}",
                "details": response.text
            }
            return
        
        # Process the streaming response line by line
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    # Parse the JSON data from the SSE stream
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    yield data
                    
                except json.JSONDecodeError as e:
                    # Handle malformed JSON gracefully
                    st.warning(f"Received malformed data from backend: {line}")
                    continue
                    
            elif line.strip() == "":
                # Empty lines are part of SSE protocol, ignore them
                continue
                
    except requests.exceptions.Timeout:
        yield {
            "type": "error", 
            "error": "Request timeout - the analysis is taking longer than expected.",
            "suggestion": "Please try again with a more specific question."
        }
        
    except requests.exceptions.ConnectionError:
        yield {
            "type": "error",
            "error": "Cannot connect to the backend service.",
            "suggestion": "Please ensure the API server is running and try again."
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please try again or contact support if the issue persists."
        }

def handle_progressive_analysis(query: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Orchestrate the complete progressive analysis workflow with real-time updates.
    
    This function demonstrates how to coordinate complex real-time user interfaces
    with streaming backend processes. The key insight is to separate the streaming
    data handling from the UI updates, allowing for clean error handling and
    graceful degradation when network issues occur.
    
    The pattern here can be applied to any long-running AI process where you want
    to provide users with meaningful feedback rather than just waiting for a
    final result.
    
    Args:
        query: The compliance question to analyze
        session_id: Optional session identifier for tracking
        
    Returns:
        Final analysis result or None if cancelled/failed
    """
    
    # Initialize progress tracking
    tracker = ProgressTracker()
    tracker.start_time = time.time()
    
    # Create progress display container
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Add cancellation button for user control
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⏹️ Cancel Analysis", type="secondary"):
            st.warning("Analysis cancelled by user")
            return None
    
    try:
        # Process streaming updates
        for update_data in stream_progress_updates(query, session_id):
            
            # Handle different types of updates
            if update_data.get("type") == "error":
                tracker.mark_error(update_data.get("error", "Unknown error"))
                
                with status_placeholder.container():
                    st.error(f"❌ Analysis Error: {tracker.error_message}")
                    
                    suggestion = update_data.get("suggestion")
                    if suggestion:
                        st.info(f"💡 **Suggestion:** {suggestion}")
                
                return None
                
            elif update_data.get("type") == "final_result":
                # Analysis complete - return the final result
                tracker.mark_complete()
                
                with progress_placeholder.container():
                    create_progress_display(tracker)
                
                with status_placeholder.container():
                    st.success("🎉 Analysis Complete! Generating comprehensive compliance guidance...")
                
                return update_data.get("result")
                
            else:
                # Regular progress update
                agent_name = update_data.get("agent_name", "Unknown Agent")
                tracker.update_agent_progress(agent_name, update_data)
                
                # Update the progress display
                with progress_placeholder.container():
                    create_progress_display(tracker)
                
                # Show current activity in status area
                current_message = update_data.get("message", "Processing...")
                with status_placeholder.container():
                    st.info(f"🔄 **{agent_name}:** {current_message}")
                
                # Small delay to make updates visible (adjust based on your needs)
                time.sleep(0.1)
                
    except Exception as e:
        tracker.mark_error(f"Unexpected error during analysis: {str(e)}")
        
        with status_placeholder.container():
            st.error(f"❌ Analysis Failed: {tracker.error_message}")
            st.info("💡 **Suggestion:** Please try again or contact support if the issue persists.")
        
        return None
    
    # Should not reach here in normal flow
    return None