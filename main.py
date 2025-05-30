import os
import random
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from gdpr_agent import GDPRAgent
from polish_law_agent import PolishLawAgent
# Updated import: Using the sophisticated Internal Security Agent instead of the basic Internal Policy Agent
# This agent provides precise procedure citations with implementation step details
from internal_security_agent import InternalSecurityAgent
from summarization_agent import SummarizationAgent
from test_queries import (
    get_all_eu_business_queries,
    get_eu_business_queries_by_category, 
    get_eu_business_categories,
    get_complex_eu_multi_domain_queries,
    get_intra_eu_coordination_queries
)

# Load environment variables
load_dotenv()

# Define the state structure
# Note: The state structure remains exactly the same because the new Internal Security Agent
# was designed to be a drop-in replacement that still updates "internal_policy_citations"
# This demonstrates good system design - we can upgrade individual components without
# breaking the overall workflow structure
class AgentState(TypedDict):
    user_query: str
    gdpr_citations: List[Dict[str, Any]]
    polish_law_citations: List[Dict[str, Any]]
    polish_law_analysis: str
    internal_policy_citations: List[Dict[str, Any]]  # Still the same key name for compatibility
    summary: Dict[str, Any]
    
# Initialize agents
# The GDPR and Polish Law agents use sophisticated metadata flattening approaches
gdpr_agent = GDPRAgent()
polish_law_agent = PolishLawAgent()

# Updated agent initialization: Using the new Internal Security Agent
# This agent brings the same level of sophistication to internal security procedures
# that the Polish Law Agent brings to legal document analysis
# It can create precise citations like "Procedure 3.1: User Account Management Process, Step 2 - Initial Access Provisioning"
internal_security_agent = InternalSecurityAgent()

# The summarization agent has been updated to handle the enhanced citation formats
# from all three sophisticated agents
summarization_agent = SummarizationAgent()

# Create a new graph
# The workflow structure remains identical because we designed the new security agent
# to integrate seamlessly with the existing pipeline
workflow = StateGraph(AgentState)

# Add nodes for each agent
# Notice how we're using the same node name "internal_policy" even though we're now
# using the Internal Security Agent - this maintains compatibility with existing code
# while upgrading the underlying functionality
workflow.add_node("gdpr", gdpr_agent.process)
workflow.add_node("polish_law", polish_law_agent.process)
workflow.add_node("internal_policy", internal_security_agent.process)  # Updated agent but same node name
workflow.add_node("summarization", summarization_agent.process)

# Define the edges of the graph (the flow)
# The workflow remains a linear pipeline:
# 1. GDPR Agent finds relevant GDPR articles with precise paragraph citations
# 2. Polish Law Agent finds relevant Polish law provisions with detailed structural references
# 3. Internal Security Agent finds relevant security procedures with implementation step citations
# 4. Summarization Agent combines all three into a unified action plan
workflow.add_edge("gdpr", "polish_law")
workflow.add_edge("polish_law", "internal_policy")
workflow.add_edge("internal_policy", "summarization")
workflow.add_edge("summarization", END)

# Set the entry point
workflow.set_entry_point("gdpr")

# Compile the graph
app = workflow.compile()

def get_random_test_query(focus_area="by_category"):
    """
    Randomly select an EU test query with configurable focus for targeted system validation.
    
    This enhanced function allows you to test specific aspects of EU GDPR compliance
    by selecting queries that match particular operational areas or complexity levels.
    This targeted approach helps validate system performance across different EU
    business scenarios and identify areas needing refinement.
    
    Parameters:
    focus_area (str): Controls the type of EU queries selected
        - "mixed": Random selection from all EU business scenarios (default)
        - "complex": Multi-member state queries requiring sophisticated coordination
        - "intra_eu": Focus on cross-border coordination within EU
        - "by_category": Rotate through different operational areas systematically
        - "emerging_tech": EU-based AI, IoT, and new technology challenges
    
    This approach demonstrates how thoughtful test case selection within the EU
    regulatory framework can provide more actionable insights into system performance.
    """
    
    if focus_area == "complex":
        # Use only the most challenging multi-member state scenarios
        # These test sophisticated coordination across EU jurisdictions
        complex_eu_queries = get_complex_eu_multi_domain_queries()
        return random.choice(complex_eu_queries)
    
    elif focus_area == "intra_eu":
        # Focus on cross-border coordination within EU
        # Tests understanding of member state implementation differences
        coordination_queries = get_intra_eu_coordination_queries()
        return random.choice(coordination_queries)
    
    elif focus_area == "by_category":
        # Systematically test different EU operational areas
        # Ensures comprehensive coverage of EU business domains
        categories = get_eu_business_categories()
        random_category = random.choice(categories)
        category_queries = get_eu_business_queries_by_category(random_category)
        selected_query = random.choice(category_queries)
        
        # Print category info to track EU testing coverage
        print(f"Testing EU {random_category} scenario...")
        return selected_query
    
    elif focus_area == "emerging_tech":
        # Focus on EU-based emerging technology scenarios
        # Tests application of GDPR to new technologies within EU framework
        tech_queries = get_eu_business_queries_by_category("emerging_technology_scenarios")
        return random.choice(tech_queries)
    
    else:  # "mixed" or any other value
        # Default: random selection from all EU business scenarios
        # Provides comprehensive EU coverage while maintaining unpredictability
        all_eu_business_queries = get_all_eu_business_queries()
        return random.choice(all_eu_business_queries)

def main():
    """
    Main execution function that demonstrates the enhanced multi-agent system.
    
    This function orchestrates the complete workflow from initial query through
    final action plan generation. The workflow now includes three sophisticated
    agents that each use metadata flattening approaches to create precise citations
    from their respective knowledge bases.
    
    The beauty of this design is that users get comprehensive compliance guidance
    that seamlessly integrates GDPR requirements, Polish law specifics, and 
    internal security procedure implementations - all with precise, verifiable citations.
    """
    # Select a random query to test the enhanced system capabilities
    user_query = get_random_test_query()
    
    # Print workflow start information with enhanced context
    print("\n===== STARTING ENHANCED LANGGRAPH WORKFLOW =====")
    print(f"Test query: \"{user_query}\"")
    print("Enhanced System Features:")
    print("  • GDPR Agent: Precise article and paragraph citations")
    print("  • Polish Law Agent: Detailed structural references with metadata flattening")
    print("  • Internal Security Agent: Implementation step-level procedure citations")
    print("  • Summarization Agent: Unified action plans with precise legal traceability")
    print("=" * 50)
    
    # Initial state setup
    # The state structure accommodates all the enhanced citation formats
    # while maintaining backward compatibility
    initial_state = {
        "user_query": user_query,
        "gdpr_citations": [],
        "polish_law_citations": [],
        "internal_policy_citations": [],  # Will be populated by Internal Security Agent
        "summary": {}
    }
    
    # Run the enhanced graph workflow
    print("Executing enhanced graph workflow with sophisticated agents...")
    result = app.invoke(initial_state)
    print("\n===== ENHANCED WORKFLOW COMPLETED =====")
    
    # Print the final summary with enhanced formatting
    print("\n" + "=" * 50)
    print("COMPREHENSIVE COMPLIANCE ACTION PLAN")
    print("=" * 50)
    
    if "summary" in result and "action_plan" in result["summary"]:
        # Display the action plan created by combining all three sophisticated agents
        print(result["summary"]["action_plan"])
        
        # Show processing statistics if available
        if "total_citations" in result["summary"]:
            print(f"\nProcessing Summary:")
            print(f"  • Total Citations: {result['summary']['total_citations']}")
            print(f"  • GDPR Citations: {result['summary']['citations_by_source']['gdpr']}")
            print(f"  • Polish Law Citations: {result['summary']['citations_by_source']['polish_law']}")
            print(f"  • Security Procedure Citations: {result['summary']['citations_by_source']['internal_policy']}")
            print(f"  • Overall Precision Rate: {result['summary']['overall_precision_rate']}%")
            
            # Explain what the precision rate means
            print(f"\nPrecision Rate Explanation:")
            print(f"This rate shows how many citations include detailed structural references")
            print(f"(like specific paragraphs, sub-paragraphs, or implementation steps)")
            print(f"rather than just basic article or procedure numbers.")
    else:
        print("No summary generated - this may indicate an issue with the workflow.")
        print("Check the individual agent outputs for debugging information.")

if __name__ == "__main__":
    main()