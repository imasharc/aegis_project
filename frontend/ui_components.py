"""
Enhanced UI Components Module with Sample Query Integration

This updated version demonstrates how to build progressive disclosure interfaces
where users can either start with guided examples or jump directly to custom
queries. This approach reduces friction for new users while maintaining
flexibility for experienced users.

Key enhancements:
- Sample query selector with categorized tabs
- One-click query population from curated examples  
- Smooth integration between guided and custom query workflows
- Educational scaffolding that teaches effective query patterns

The design philosophy here is "guided discovery" - we show users what's possible
through concrete examples, then enable them to modify or create their own queries
based on those patterns. This approach is particularly effective for complex
systems where users might not initially understand the full capability scope.
"""

import streamlit as st
import re
from typing import Dict, Any, List, Tuple, Optional

# Import our sample query system
from sample_queries import get_sample_categories, get_category_queries, search_sample_queries

# Existing citation configuration constants
CITATION_CONFIG = {
    "citation_pattern": r'AUTHORITATIVE SOURCE CITATIONS:\s*\n\n(.*?)(?=\n\n[A-Z]|\Z)',
    "source_pattern": r'([^:]+:)\s*\[(\d+)\s+with[^]]*\]\s*(.*?)(?=\n\n[^:]+:|\Z)',
    "individual_pattern": r'\[(\d+)\]\s*([^[]+?)(?=\[|\Z)'
}


def display_sample_query_selector():
    """
    Display the sample query selector with categorized tabs and clickable examples.
    
    This component implements the "guided discovery" pattern where users can explore
    system capabilities through curated examples before writing custom queries.
    The tab-based organization helps users quickly find relevant business scenarios.
    
    This approach teaches several important UX principles:
    - Progressive disclosure: Show categories first, then specific examples
    - Contextual guidance: Each category explains what types of scenarios it covers
    - One-click activation: Examples can be used immediately without typing
    - Educational scaffolding: Users learn effective query patterns through examples
    
    Returns:
        Optional[str]: Selected sample query text, or None if no selection made
    """
    
    st.subheader("📚 Sample Compliance Scenarios")
    st.markdown("*Choose from realistic business scenarios to see how our multi-agent system works, or use these as starting points for your own queries.*")
    
    # Get sample categories for tab creation
    sample_categories = get_sample_categories()
    
    # Create tabs for different business scenario categories
    tab_keys = list(sample_categories.keys())
    tab_titles = [sample_categories[key]["title"] for key in tab_keys]
    
    # Create the tab interface with descriptive titles
    tabs = st.tabs(tab_titles)
    
    selected_query = None
    
    # Process each tab with its corresponding sample queries
    for i, (tab_key, tab) in enumerate(zip(tab_keys, tabs)):
        with tab:
            category_data = sample_categories[tab_key]
            
            # Display category description to provide context
            st.markdown(f"**{category_data['description']}**")
            st.markdown("---")  # Visual separator
            
            # Get queries for this category
            category_queries = get_category_queries(tab_key)
            
            # Display each query as a clickable button with description
            for query_data in category_queries:
                # Create an expander for each query to show preview
                with st.expander(f"🔍 {query_data['title']}", expanded=False):
                    # Show a preview of the query
                    query_preview = query_data['query'][:200] + "..." if len(query_data['query']) > 200 else query_data['query']
                    st.markdown(f"*{query_preview}*")
                    
                    # Create action buttons for this query
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Main action button to select this query
                        if st.button(f"Use This Query", key=f"use_{tab_key}_{query_data['title']}", type="primary"):
                            selected_query = query_data['query']
                            st.success(f"✅ Selected: {query_data['title']}")
                            st.markdown("*Scroll down to review and submit the query.*")
                    
                    with col2:
                        # Secondary button to just view the full query
                        if st.button(f"View Full Text", key=f"view_{tab_key}_{query_data['title']}", type="secondary"):
                            st.info("**Full Query Text:**")
                            st.markdown(f"```\n{query_data['query']}\n```")
    
    # Add search functionality for power users
    with st.expander("🔍 Search Sample Queries", expanded=False):
        search_term = st.text_input(
            "Search queries by keyword:", 
            placeholder="e.g., 'GDPR', 'employee monitoring', 'cloud migration'"
        )
        
        if search_term and len(search_term) > 2:
            search_results = search_sample_queries(search_term)
            
            if search_results:
                st.markdown(f"**Found {len(search_results)} matching queries:**")
                
                for result in search_results[:5]:  # Limit to top 5 results
                    if st.button(f"Use: {result['title']}", key=f"search_{result['title']}", type="secondary"):
                        selected_query = result['query']
                        st.success(f"✅ Selected from search: {result['title']}")
            else:
                st.warning("No matching queries found. Try different keywords.")
    
    return selected_query


def display_enhanced_query_interface():
    """
    Display the enhanced query input interface with sample query integration.
    
    This function combines the sample query selector with the traditional text input,
    creating a hybrid interface that serves both novice and expert users. The design
    follows the principle of "progressive enhancement" - basic functionality works
    without JavaScript or complex interactions, but enhanced features improve the
    experience when available.
    
    The key insight here is that we're not replacing the custom query capability,
    but rather providing an alternative entry point that reduces friction for new
    users while teaching them effective query patterns.
    
    Returns:
        Optional[str]: The validated user query ready for processing, or None if invalid
    """
    
    st.header("💬 Compliance Query Interface")
    
    # Initialize session state for query text if not exists
    if 'selected_query_text' not in st.session_state:
        st.session_state.selected_query_text = ""
    
    # Display the sample query selector
    selected_sample = display_sample_query_selector()
    
    # Update session state if a sample was selected
    if selected_sample:
        st.session_state.selected_query_text = selected_sample
    
    st.markdown("---")  # Visual separator between samples and input
    
    # Enhanced query input section with better guidance
    st.markdown("### ✏️ Your Compliance Question")
    st.markdown("*Modify the selected sample above, or write your own compliance question below.*")
    
    # Query input with the selected sample pre-populated
    query = st.text_area(
        "Enter your compliance question:",
        value=st.session_state.selected_query_text,
        height=120,
        placeholder="""Example: "We're implementing employee monitoring software in our Warsaw office that tracks productivity metrics and integrates with our German cloud service provider. What specific GDPR compliance steps do we need, and how do our internal security procedures apply to this cross-border data processing scenario?\"""",
        help="Describe your business scenario with specific details about location, technology, and data types for the most accurate analysis with properly formatted citations."
    )
    
    # Clear query button for user convenience
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear", type="secondary"):
            st.session_state.selected_query_text = ""
            st.rerun()  # Refresh to clear the text area
    
    # Analysis submission button with clear call-to-action
    if st.button("🔍 Analyze Compliance Requirements", type="primary"):
        return query
    
    # Enhanced guidance section with query quality tips
    with st.expander("💡 Tips for Better Results", expanded=False):
        st.markdown("""
        **For optimal analysis results, consider including:**
        
        **Scenario Context:**
        - What business process or system you're implementing
        - Who will be affected (employees, customers, partners)  
        - What data will be collected or processed
        
        **Technical Details:**
        - Specific software or platforms being used
        - Data storage locations and cloud providers
        - Integration points with existing systems
        
        **Compliance Scope:**
        - Geographic regions involved (EU, Poland, other countries)
        - Industry-specific requirements
        - Timeline for implementation or compliance
        
        **Common Query Patterns:**
        - "We are implementing [system] for [purpose] involving [data types]..."
        - "Our [location] office needs to [process/store/transfer] data..."
        - "How should we handle [specific scenario] under GDPR and Polish law?"
        
        **Quality Indicators:**
        ✅ Specific business context and location details
        ✅ Clear description of data types and processing activities  
        ✅ Mention of relevant technology providers or platforms
        ✅ Timeline or implementation details
        ✅ Cross-border or multi-jurisdictional elements
        """)
    
    return None


def display_sample_query_stats():
    """
    Display statistics about the sample query system for transparency.
    
    This component provides users with insight into the breadth and depth of
    available samples, building confidence in the system's comprehensive coverage
    of compliance scenarios. Transparency about system capabilities helps users
    understand when they might need to go beyond the samples.
    """
    
    from sample_queries import get_category_stats
    
    stats = get_category_stats()
    
    st.markdown("### 📊 Sample Query Coverage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Business Categories",
            stats["total_categories"],
            help="Different types of compliance scenarios covered"
        )
    
    with col2:
        st.metric(
            "Sample Queries", 
            stats["total_queries"],
            help="Total number of realistic business examples"
        )
    
    with col3:
        st.metric(
            "Average per Category",
            stats["average_queries_per_category"],
            help="Depth of coverage in each business area"
        )
    
    # Show category distribution for transparency
    with st.expander("📋 Coverage Details", expanded=False):
        st.markdown("**Queries per Business Category:**")
        for category, count in stats["category_distribution"].items():
            st.markdown(f"- **{category}:** {count} sample scenarios")


# All existing functions remain unchanged from the original ui_components.py
# I'm including the key ones here for completeness

def parse_citations_from_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Advanced citation parser that extracts citations from action plan text
    and converts them to a structured format for numbered display.
    """
    citations = []
    citation_counter = 1
    
    # Find the main citation section using our configured pattern
    citation_pattern = CITATION_CONFIG["citation_pattern"]
    citation_match = re.search(citation_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if citation_match:
        citation_section = citation_match.group(1)
        
        # Extract individual source blocks
        source_pattern = CITATION_CONFIG["source_pattern"]
        source_matches = re.findall(source_pattern, citation_section, re.DOTALL)
        
        for source_title, count, content in source_matches:
            # Clean up the source title
            clean_source = source_title.strip().rstrip(':')
            
            # Extract individual citations within this source
            individual_pattern = CITATION_CONFIG["individual_pattern"]
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
    
    # More aggressive cleaning to prevent duplication
    cleaned_text = re.sub(citation_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'\[(\d+)\s+with[^]]*\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[(\d+)\]', '', cleaned_text)
    cleaned_text = re.sub(r'\*\*AUTHORITATIVE SOURCE CITATIONS:\*\*.*', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'\*\*SYSTEM INSIGHTS:\*\*.*', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip(), citations


def format_citations_as_numbered_list(citations: List[Dict[str, str]]) -> str:
    """
    Convert structured citation data into a clean numbered list format.
    """
    if not citations:
        return ""
    
    # Group citations by source
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
    """
    if not raw_citations:
        return ""
    
    # Group by source type
    grouped = {}
    for i, citation in enumerate(raw_citations, 1):
        source = citation.get('source', 'Unknown Source')
        if source not in grouped:
            grouped[source] = []
        
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


def display_processing_feedback():
    """
    Display processing feedback to users during analysis.
    """
    st.info("**Processing Steps:**\n"
           "1. 🇪🇺 GDPR Agent: Analyzing European data protection requirements\n"
           "2. 🇵🇱 Polish Law Agent: Reviewing Polish implementation specifics\n"
           "3. 🔒 Security Agent: Evaluating internal procedure requirements\n"
           "4. 📊🔗 Summarization Agent: Creating comprehensive action plan with authoritative citations")


def display_citation_analysis(citations: Dict[str, Any]):
    """
    Display the sophisticated citation analysis in an accessible format.
    """
    st.subheader("📊 Citation Analysis")
    
    # Create metrics display
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
            help="Quality score of the citation analysis"
        )


def display_system_metadata(metadata: Dict[str, Any]):
    """
    Display metadata about the sophisticated analysis process.
    """
    with st.expander("🔍 Analysis Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Analysis Architecture:**")
            st.write(f"• System: {metadata.get('agent_coordination', 'Multi-agent')}")
            st.write(f"• Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
            st.write(f"• Timestamp: {metadata.get('analysis_timestamp', 'Unknown')}")
        
        with col2:
            st.write("**Domains Analyzed:**")
            domains = metadata.get('domains_analyzed', [])
            for domain in domains:
                st.write(f"• {domain.replace('_', ' ').title()}")


def render_action_plan_with_citations(action_plan_text: str, raw_citations: List[Dict[str, Any]] = None):
    """
    Process and render action plan with sophisticated citation formatting.
    """
    
    # Try structured citation data first
    if raw_citations and len(raw_citations) > 0:
        st.markdown("### 📋 Comprehensive Action Plan")
        
        # Clean text of citation formatting
        clean_text = re.sub(r'AUTHORITATIVE SOURCE CITATIONS:.*', '', action_plan_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{clean_text}</div>', 
                   unsafe_allow_html=True)
        
        # Display formatted citations
        formatted_citations = format_raw_citations(raw_citations)
        st.markdown(formatted_citations, unsafe_allow_html=True)
        
    else:
        # Fallback: Parse citations from text
        st.markdown("### 📋 Comprehensive Action Plan")
        
        cleaned_text, parsed_citations = parse_citations_from_text(action_plan_text)
        
        # Display cleaned action plan
        st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{cleaned_text}</div>', 
                   unsafe_allow_html=True)
        
        # Display formatted citations if found
        if parsed_citations:
            formatted_citations = format_citations_as_numbered_list(parsed_citations)
            st.markdown(formatted_citations, unsafe_allow_html=True)
        else:
            # Display original text if no citations found
            st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{action_plan_text}</div>', 
                       unsafe_allow_html=True)


def display_app_header():
    """
    Display the main application header with branding and description.
    """
    st.title("🛡️ Enhanced Compliance Analysis System")
    st.markdown("""
    **Sophisticated multi-agent analysis** combining GDPR expertise, Polish law knowledge, 
    and internal security procedures to provide comprehensive compliance guidance.
    
    *Now featuring advanced citation formatting for professional documentation.*
    """)


def display_sidebar_info(system_ready: bool, system_info: Optional[Dict[str, Any]] = None):
    """
    Display sidebar content including system status and usage guidelines.
    
    Enhanced with sample query system information to help users understand
    the full range of system capabilities and how to access them effectively.
    """
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Display system health status with appropriate styling
        if system_ready:
            st.success("✅ Backend System Operational")
            st.info("All agents connected and ready for analysis")
            
        else:
            st.error("❌ Backend System Unavailable")
            st.warning("Please ensure the FastAPI backend is running on port 8000")
        
        # Add sample query system info
        st.header("📚 Sample Query System")
        display_sample_query_stats()
        
        st.header("📋 Query Guidelines")
        st.markdown("""
        **Effective queries should include:**
        
        - Specific business scenario description
        - Geographic context (EU/Poland specific)  
        - Types of data being processed
        - Technology platforms or vendors mentioned
        - Timeframes or implementation deadlines
        
        **Quick Start Options:**
        
        - Browse sample scenarios in the main interface
        - Select a relevant business category tab
        - Click "Use This Query" on any example
        - Modify the sample to match your situation
        
        **Example query topics:**
        
        - Employee monitoring system implementations
        - Cross-border data transfer scenarios
        - Cloud service provider integrations
        - Data breach incident response procedures
        - Consent management workflows
        """)
        
        st.header("📄 Citation Features")
        st.info("""
        **Enhanced citation system provides:**
        
        - Professionally numbered reference lists
        - Automatic source categorization
        - Clean, readable formatting
        - Legal document compliance standards
        - Cross-reference capability
        """)


def display_success_results(result: Dict[str, Any]):
    """
    Display successful analysis results with formatted action plan and citations.
    
    Enhanced to include information about whether the analysis was based on
    a sample query or custom input, providing users with context about how
    the system arrived at its recommendations.
    """
    st.success("🎉 **Analysis Complete!** Comprehensive compliance guidance generated.")
    
    # Extract key components from the result
    action_plan = result.get('action_plan', '')
    raw_citations = result.get('raw_citations', [])
    citations_metadata = result.get('citations', {})
    analysis_metadata = result.get('metadata', {})
    
    # Display citation analysis metrics if available
    if citations_metadata and citations_metadata.get('total_citations', 0) > 0:
        display_citation_analysis(citations_metadata)
        st.markdown("---")  # Visual separator
    
    # Render the main action plan with enhanced citation formatting
    if action_plan:
        render_action_plan_with_citations(action_plan, raw_citations)
    else:
        st.warning("No action plan was generated. Please try rephrasing your query.")
    
    # Display analysis metadata for transparency
    if analysis_metadata:
        display_system_metadata(analysis_metadata)


def display_error_results(result: Dict[str, Any]):
    """
    Display error results with helpful guidance for users.
    
    Enhanced to suggest using sample queries as an alternative when
    custom queries fail, providing users with a fallback option that
    often resolves query formulation issues.
    """
    # Format the error message for user-friendly display
    from backend_client import format_backend_error
    error_message = format_backend_error(result)
    
    # Display the error with appropriate styling
    st.error(f"❌ **Analysis Failed**")
    
    # Show the detailed error message
    st.markdown(f"**Error Details:** {error_message}")
    
    # Provide specific guidance based on error type
    if result.get("timeout"):
        st.info("""
        🕐 **Timeout Guidance:**
        
        Your query might be very complex for the current system capacity. Consider:
        - Trying one of our sample queries to test system functionality
        - Breaking your question into smaller, more focused parts
        - Simplifying the scenario description
        - Focusing on one specific compliance aspect at a time
        """)
        
    elif result.get("connection_error"):
        st.info("""
        🔧 **Connection Guidance:**
        
        There seems to be a connectivity issue with the backend service:
        - Ensure the FastAPI backend is running on port 8000
        - Check your network connection
        - Try a sample query to verify system connectivity
        - Contact your system administrator if the issue persists
        """)
        
    elif result.get("http_error"):
        status_code = result.get("status_code", "unknown")
        st.info(f"""
        📋 **HTTP Error Guidance (Status: {status_code}):**
        
        The backend service returned an error:
        - If status is 422: Your query might have validation issues
        - If status is 500: There may be a temporary backend issue
        - Try selecting a sample query to test the system
        - Modify the sample query to match your specific scenario
        """)
        
    else:
        st.info("""
        🆘 **General Troubleshooting:**
        
        - Try using one of our sample queries first to verify system functionality
        - Ensure your query is detailed and specific
        - Check that you've included relevant business context
        - Use sample queries as templates for your own scenarios
        - Contact support if the issue continues
        """)
    
    # Add a helpful call-to-action to try sample queries
    st.markdown("---")
    st.info("💡 **Quick Fix:** Try selecting a sample query from the categories above to verify the system is working, then modify it to match your specific scenario.")


def display_chat_history(chat_history: List[Dict[str, Any]]):
    """
    Display chat history for session context and learning.
    
    Enhanced to show whether queries were based on samples or custom input,
    helping users understand which approaches tend to work best and learn
    from their query evolution patterns.
    """
    if not chat_history:
        return
        
    st.header("📝 Session History")
    
    # Display guidance about session context
    st.info(f"📊 **Session Context:** Showing your last {min(len(chat_history), 3)} queries to help build comprehensive understanding.")
    
    # Show the most recent queries (limit to 3 for readability)
    for i, chat_item in enumerate(reversed(chat_history[-3:])):
        query_number = len(chat_history) - i
        
        # Create an expandable section for each historical query
        with st.expander(f"🔍 Query {query_number}: {chat_item['query'][:60]}..."):
            
            # Display the original query
            st.markdown("**📝 Original Question:**")
            st.markdown(f"*{chat_item['query']}*")
            
            # Display the response summary
            response = chat_item['response']
            if response.get('success'):
                st.markdown("**✅ Analysis Summary:**")
                
                # Extract and display key information
                action_plan = response.get('action_plan', '')
                raw_citations = response.get('raw_citations', [])
                
                if raw_citations:
                    # Show citation count and clean action plan text
                    clean_text = re.sub(r'AUTHORITATIVE SOURCE CITATIONS:.*', '', action_plan, flags=re.DOTALL | re.IGNORECASE)
                    st.markdown(clean_text.strip()[:300] + "..." if len(clean_text) > 300 else clean_text.strip())
                    
                    st.markdown(f"**📋 Citations:** {len(raw_citations)} authoritative sources referenced")
                    
                else:
                    # Parse citations from text if structured data not available
                    cleaned_text, parsed_citations = parse_citations_from_text(action_plan)
                    summary_text = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
                    st.markdown(summary_text)
                    
                    if parsed_citations:
                        st.markdown(f"**📋 Citations:** {len(parsed_citations)} authoritative sources")
                
                # Show analysis metadata if available
                citations_data = response.get('citations', {})
                if citations_data.get('total_citations', 0) > 0:
                    total_cites = citations_data['total_citations']
                    gdpr_cites = citations_data.get('gdpr_citations', 0)
                    polish_cites = citations_data.get('polish_law_citations', 0)
                    
                    st.markdown(f"**📊 Source Breakdown:** {total_cites} total ({gdpr_cites} GDPR, {polish_cites} Polish law)")
                
            else:
                # Display error information for failed queries
                st.markdown("**❌ Analysis Error:**")
                st.markdown(f"*{response.get('error', 'Unknown error occurred')}*")
                
            # Add timestamp information
            if 'timestamp' in chat_item:
                import datetime
                timestamp = datetime.datetime.fromtimestamp(chat_item['timestamp'])
                st.caption(f"🕐 Query submitted: {timestamp.strftime('%H:%M:%S')}")


# Additional utility function for handling sidebar content (aliased for compatibility)
def display_sidebar_content():
    """
    Legacy function name for backward compatibility.
    Redirects to the main sidebar display function.
    """
    # This function exists for backward compatibility with existing code
    # that might reference the old function name
    display_sidebar_info(True, None)