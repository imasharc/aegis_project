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
import requests
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# Configuration for the backend API
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Enhanced Compliance Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_citations_from_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Advanced citation parser that extracts citations from action plan text
    and converts them to a structured format for numbered display.
    
    This function demonstrates sophisticated text processing techniques,
    using regular expressions to identify citation patterns and extract
    the relevant information for clean presentation.
    
    Args:
        text: The action plan text containing embedded citations
        
    Returns:
        tuple: (cleaned_text_without_citations, list_of_citation_objects)
    """
    citations = []
    citation_counter = 1
    
    # Advanced regex pattern to match citation sections
    # This pattern identifies the citation header and captures the detailed citation content
    citation_pattern = r'AUTHORITATIVE SOURCE CITATIONS:\s*\n\n(.*?)(?=\n\n[A-Z]|\Z)'
    
    # Find the main citation section
    citation_match = re.search(citation_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if citation_match:
        citation_section = citation_match.group(1)
        
        # Pattern to match individual source blocks (e.g., "European Data Protection Regulation (GDPR):")
        source_pattern = r'([^:]+:)\s*\[(\d+)\s+with[^]]*\]\s*(.*?)(?=\n\n[^:]+:|\Z)'
        
        source_matches = re.findall(source_pattern, citation_section, re.DOTALL)
        
        for source_title, count, content in source_matches:
            # Clean up the source title
            clean_source = source_title.strip().rstrip(':')
            
            # Extract individual citations within this source
            # Pattern to match [1], [2], etc. with their content
            individual_pattern = r'\[(\d+)\]\s*([^[]+?)(?=\[|\Z)'
            individual_matches = re.findall(individual_pattern, content, re.DOTALL)
            
            for orig_num, citation_text in individual_matches:
                # Clean up the citation text
                clean_text = re.sub(r'\s*✓\s*$', '', citation_text.strip())
                clean_text = re.sub(r'\s+', ' ', clean_text)
                
                citations.append({
                    'number': citation_counter,
                    'source': clean_source,
                    'text': clean_text,
                    'original_number': orig_num
                })
                citation_counter += 1
    
    # Remove the entire citation section from the main text
    cleaned_text = re.sub(citation_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any remaining citation references in the main text
    cleaned_text = re.sub(r'\[(\d+)\s+with[^]]*\]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text, citations

def format_citations_as_numbered_list(citations: List[Dict[str, str]]) -> str:
    """
    Convert structured citation data into a clean numbered list format.
    
    This function takes the parsed citation data and creates a professional,
    easy-to-read numbered list that maintains all the important legal and
    regulatory reference information while improving readability.
    
    Args:
        citations: List of citation dictionaries with source, text, etc.
        
    Returns:
        str: Formatted HTML string with numbered citations
    """
    if not citations:
        return ""
    
    # Group citations by source for organized presentation
    grouped_citations = {}
    for citation in citations:
        source = citation['source']
        if source not in grouped_citations:
            grouped_citations[source] = []
        grouped_citations[source].append(citation)
    
    # Build the formatted citation list
    formatted_html = "<div style='font-size: 16px; line-height: 1.6; margin-top: 20px;'>"
    formatted_html += "<h4>📋 AUTHORITATIVE SOURCE CITATIONS:</h4>\n\n"
    
    for source, source_citations in grouped_citations.items():
        formatted_html += f"<p><strong>{source}:</strong> [{len(source_citations)} citations]</p>\n"
        
        for citation in source_citations:
            formatted_html += f"<p style='margin-left: 20px; margin-bottom: 10px;'>"
            formatted_html += f"<strong>[{citation['number']}]</strong> {citation['text']}"
            formatted_html += "</p>\n"
        
        formatted_html += "\n"
    
    formatted_html += "</div>"
    
    return formatted_html

def format_raw_citations(raw_citations: List[Dict[str, Any]]) -> str:
    """
    Format raw citation data from the backend into numbered lists.
    
    This function handles the structured citation data returned by the
    enhanced backend API, converting it into the same clean numbered
    format for consistent presentation.
    
    Args:
        raw_citations: List of raw citation objects from backend
        
    Returns:
        str: Formatted HTML string with numbered citations
    """
    if not raw_citations:
        return ""
    
    # Group by source type
    grouped = {}
    for i, citation in enumerate(raw_citations, 1):
        source = citation.get('source', 'Unknown Source')
        if source not in grouped:
            grouped[source] = []
        
        # Add citation number for sequential referencing
        citation_copy = citation.copy()
        citation_copy['number'] = i
        grouped[source].append(citation_copy)
    
    # Build formatted HTML
    formatted_html = "<div style='font-size: 16px; line-height: 1.6; margin-top: 20px;'>"
    formatted_html += "<h4>📋 AUTHORITATIVE SOURCE CITATIONS:</h4>\n\n"
    
    for source, citations in grouped.items():
        formatted_html += f"<p><strong>{source}:</strong> [{len(citations)} citations]</p>\n"
        
        for citation in citations:
            formatted_html += f"<p style='margin-left: 20px; margin-bottom: 10px;'>"
            formatted_html += f"<strong>[{citation['number']}]</strong> "
            
            # Format based on citation type
            if citation.get('type') == 'gdpr':
                if citation.get('article'):
                    formatted_html += f"Article {citation['article']} "
                if citation.get('chapter'):
                    formatted_html += f"({citation['chapter']}): "
            elif citation.get('type') == 'polish_law':
                if citation.get('article'):
                    formatted_html += f"Article {citation['article']} "
                if citation.get('law'):
                    formatted_html += f"({citation['law']}): "
            elif citation.get('type') == 'internal_policy':
                if citation.get('procedure'):
                    formatted_html += f"Procedure {citation['procedure']} "
                if citation.get('section'):
                    formatted_html += f"(Section {citation['section']}): "
            
            formatted_html += citation.get('text', '')
            formatted_html += "</p>\n"
        
        formatted_html += "\n"
    
    formatted_html += "</div>"
    
    return formatted_html

def check_backend_health() -> bool:
    """
    Check if the sophisticated backend system is operational.
    
    This function provides immediate feedback about system availability,
    which is crucial for user experience with complex AI systems.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except requests.exceptions.RequestException:
        return False

def send_query_to_backend(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Send a compliance query to the sophisticated backend system.
    
    This function handles the communication with your multi-agent system,
    providing proper error handling and user feedback for the complex
    processing that happens behind the scenes.
    
    Enhanced to request numbered citation formatting from the backend.
    """
    try:
        payload = {
            "query": query,
            "citation_style": "numbered"  # Request numbered citation style
        }
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json=payload,
            timeout=60  # Allow time for sophisticated analysis
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return {
                "success": False,
                "error": f"Backend error: {error_detail}",
                "status_code": response.status_code
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Analysis timeout - your query might be very complex. Please try a more specific question.",
            "timeout": True
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Connection error: {str(e)}",
            "connection_error": True
        }

def display_citation_analysis(citations: Dict[str, Any]):
    """
    Display the sophisticated citation analysis in an accessible format.
    
    This function demonstrates how to present complex technical information
    (your sophisticated citation system) in a way that's valuable to users
    without overwhelming them with technical details.
    """
    st.subheader("📊 Citation Analysis")
    
    # Create metrics display for immediate insight
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Citations", 
            citations.get("total_citations", 0),
            help="Total number of authoritative sources analyzed"
        )
    
    with col2:
        st.metric(
            "GDPR References", 
            citations.get("gdpr_citations", 0),
            help="European data protection regulation citations"
        )
    
    with col3:
        st.metric(
            "Polish Law", 
            citations.get("polish_law_citations", 0),
            help="Polish data protection implementation citations"
        )
    
    with col4:
        st.metric(
            "Security Procedures", 
            citations.get("security_citations", 0),
            help="Internal security procedure references"
        )
    
    # Precision quality indicator
    precision_rate = citations.get("precision_rate", 0)
    if precision_rate > 0:
        st.metric(
            "Analysis Precision", 
            f"{precision_rate}%",
            help="Quality score of the citation analysis - higher values indicate more detailed structural analysis"
        )

def display_system_metadata(metadata: Dict[str, Any]):
    """
    Display metadata about the sophisticated analysis process.
    
    This helps users understand the sophistication they're receiving
    while building confidence in the system's capabilities.
    """
    with st.expander("🔍 Analysis Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Analysis Architecture:**")
            st.write(f"• System: {metadata.get('agent_coordination', 'Multi-agent')}")
            st.write(f"• Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
            st.write(f"• Timestamp: {metadata.get('analysis_timestamp', 'Unknown')}")
            st.write(f"• Citation Style: {metadata.get('citation_style_requested', 'numbered')}")
        
        with col2:
            st.write("**Domains Analyzed:**")
            domains = metadata.get('domains_analyzed', [])
            for domain in domains:
                st.write(f"• {domain.replace('_', ' ').title()}")
            
            features = metadata.get('features', [])
            if features:
                st.write("**Enhanced Features:**")
                for feature in features:
                    st.write(f"• {feature.replace('_', ' ').title()}")

def render_action_plan_with_citations(action_plan_text: str, raw_citations: List[Dict[str, Any]] = None):
    """
    Process and render action plan with sophisticated citation formatting.
    
    This function demonstrates advanced text processing techniques for
    professional document presentation. It separates the main content
    from citations and formats them appropriately for legal and
    compliance documentation standards.
    
    The approach here teaches an important concept: when dealing with
    complex information systems, it's often better to separate content
    processing from presentation logic. This makes the system more
    maintainable and allows for different formatting styles.
    
    Args:
        action_plan_text: The main action plan content
        raw_citations: Optional structured citation data from backend
    """
    
    # First, try to use structured citation data if available
    if raw_citations and len(raw_citations) > 0:
        st.markdown("### 📋 Comprehensive Action Plan")
        
        # Display the main action plan (should be clean of citation formatting)
        clean_text = re.sub(r'AUTHORITATIVE SOURCE CITATIONS:.*', '', action_plan_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{clean_text}</div>', 
                   unsafe_allow_html=True)
        
        # Display formatted citations separately
        formatted_citations = format_raw_citations(raw_citations)
        st.markdown(formatted_citations, unsafe_allow_html=True)
        
    else:
        # Fallback: Parse citations from the text itself
        st.markdown("### 📋 Comprehensive Action Plan")
        
        # Use our sophisticated citation parser
        cleaned_text, parsed_citations = parse_citations_from_text(action_plan_text)
        
        # Display the cleaned action plan
        st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{cleaned_text}</div>', 
                   unsafe_allow_html=True)
        
        # Display formatted citations if any were found
        if parsed_citations:
            formatted_citations = format_citations_as_numbered_list(parsed_citations)
            st.markdown(formatted_citations, unsafe_allow_html=True)
        else:
            # If no structured citations found, display original text
            st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{action_plan_text}</div>', 
                       unsafe_allow_html=True)

def main():
    """
    Main application interface that provides sophisticated compliance analysis
    through an intuitive chat-like interface.
    
    Enhanced with advanced citation formatting capabilities that demonstrate
    how to present complex legal and regulatory information in accessible,
    professional formats.
    """
    # Header with system branding
    st.title("🛡️ Enhanced Compliance Analysis System")
    st.markdown("""
    **Sophisticated multi-agent analysis** combining GDPR expertise, Polish law knowledge, 
    and internal security procedures to provide comprehensive compliance guidance.
    
    *Now featuring advanced citation formatting for professional documentation.*
    """)
    
    # Sidebar for system information and settings
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check backend health with visual feedback
        if check_backend_health():
            st.success("✅ Backend System Operational")
            st.info("All agents connected and ready for analysis")
        else:
            st.error("❌ Backend System Unavailable")
            st.warning("Please ensure the FastAPI backend is running on port 8000")
            st.stop()
        
        st.header("📋 Query Guidelines")
        st.markdown("""
        **Effective queries should include:**
        - Specific business scenario
        - Geographic context (EU/Poland)
        - Data types involved
        - Technology or vendors mentioned
        
        **Example topics:**
        - Employee monitoring systems
        - Cross-border data transfers
        - Cloud service implementations
        - Incident response procedures
        """)
        
        st.header("📄 Citation Features")
        st.info("""
        **Enhanced citation system:**
        - Numbered reference lists
        - Source categorization
        - Clean, professional formatting
        - Legal document standards
        """)
    
    # Main chat interface
    st.header("💬 Compliance Query Interface")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input with sophisticated placeholder
    query = st.text_area(
        "Enter your compliance question:",
        height=100,
        placeholder="""Example: "We're implementing employee monitoring software in our Warsaw office that tracks productivity metrics and integrates with our German cloud service provider. What specific GDPR compliance steps do we need, and how do our internal security procedures apply to this cross-border data processing scenario?""",
        help="Describe your business scenario with specific details about location, technology, and data types for the most accurate analysis with properly formatted citations."
    )
    
    # Process query button with clear call-to-action
    if st.button("🔍 Analyze Compliance Requirements", type="primary"):
        if not query.strip():
            st.warning("Please enter a compliance question to analyze.")
            return
        
        if len(query.strip()) < 10:
            st.warning("Please provide a more detailed question for accurate analysis.")
            return
        
        # Show processing feedback
        with st.spinner("🧠 Analyzing your query with sophisticated multi-agent system..."):
            st.info("**Processing Steps:**\n"
                   "1. 🇪🇺 GDPR Agent: Analyzing European data protection requirements\n"
                   "2. 🇵🇱 Polish Law Agent: Reviewing Polish implementation specifics\n"
                   "3. 🔒 Security Agent: Evaluating internal procedure requirements\n"
                   "4. 📊 Integration Agent: Creating comprehensive action plan\n"
                   "5. 🔗 Citation Agent: Formatting authoritative references")
            
            # Send query to sophisticated backend
            result = send_query_to_backend(query)
        
        # Handle results with comprehensive feedback
        if result.get("success", False):
            # Store in chat history for session continuity
            st.session_state.chat_history.append({
                "query": query,
                "response": result,
                "timestamp": time.time()
            })
            
            # Display the sophisticated analysis results
            st.success("✅ Analysis Complete - Comprehensive compliance guidance generated with formatted citations")
            
            # Main action plan with sophisticated citation formatting
            action_plan = result.get("action_plan", "No action plan generated")
            raw_citations = result.get("raw_citations", [])
            
            # Use our enhanced citation rendering
            render_action_plan_with_citations(action_plan, raw_citations)
            
            # Citation analysis display
            citations = result.get("citations", {})
            if citations.get("total_citations", 0) > 0:
                display_citation_analysis(citations)
            
            # System metadata for transparency
            metadata = result.get("metadata", {})
            if metadata:
                display_system_metadata(metadata)
            
        else:
            # Error handling with helpful guidance
            st.error("❌ Analysis Error")
            error_message = result.get("error", "Unknown error occurred")
            st.write(f"**Error Details:** {error_message}")
            
            if result.get("timeout"):
                st.info("**Suggestion:** Try breaking your question into smaller, more specific parts.")
            elif result.get("connection_error"):
                st.info("**Suggestion:** Check that the backend server is running and accessible.")
    
    # Chat history display for session context
    if st.session_state.chat_history:
        st.header("📝 Session History")
        
        for i, chat_item in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat_item['query'][:60]}..."):
                st.write("**Query:**", chat_item['query'])
                
                response = chat_item['response']
                if response.get('success'):
                    st.write("**Action Plan:**")
                    
                    # Use enhanced citation formatting for history too
                    action_plan = response.get('action_plan', '')
                    raw_citations = response.get('raw_citations', [])
                    
                    if raw_citations:
                        # Clean text and show citations separately
                        clean_text = re.sub(r'AUTHORITATIVE SOURCE CITATIONS:.*', '', action_plan, flags=re.DOTALL | re.IGNORECASE)
                        st.markdown(clean_text.strip())
                        
                        st.write("**Citations:**")
                        st.write(f"- {len(raw_citations)} authoritative sources referenced")
                    else:
                        # Parse and display
                        cleaned_text, parsed_citations = parse_citations_from_text(action_plan)
                        st.markdown(cleaned_text)
                        
                        if parsed_citations:
                            st.write(f"**Citations:** {len(parsed_citations)} authoritative sources")
                    
                    citations = response.get('citations', {})
                    if citations.get('total_citations', 0) > 0:
                        st.write(f"**Total Sources:** {citations['total_citations']}")
                else:
                    st.write("**Error:**", response.get('error', 'Unknown error'))

if __name__ == "__main__":
    main()