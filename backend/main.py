import os
import random
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Direct imports to enhanced agents - clean and focused
from agent.gdpr import create_enhanced_gdpr_agent
from agent.polish_law import create_enhanced_polish_law_agent
from agent.internal_security import create_enhanced_internal_security_agent
from agent.summarization import create_enhanced_summarization_agent

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
class AgentState(TypedDict):
    user_query: str
    gdpr_citations: List[Dict[str, Any]]
    polish_law_citations: List[Dict[str, Any]]
    polish_law_analysis: str
    internal_policy_citations: List[Dict[str, Any]]
    summary: Dict[str, Any]

class EnhancedMultiAgentSystem:
    """
    Clean orchestrator for enhanced multi-agent system.
    
    This class demonstrates the principle of focused responsibility:
    - Orchestrates agent coordination without getting bogged down in details
    - Lets each agent handle its own sophisticated logging and monitoring
    - Maintains a clear, readable workflow that anyone can understand
    
    The beauty of this approach is that complexity is pushed down to where
    it belongs - in the specialized components that know how to handle it.
    """
    
    def __init__(self):
        """Initialize the system with minimal setup complexity."""
        # Create enhanced agents with their sophisticated internals
        # Each agent handles its own logging, validation, and error management
        self.gdpr_agent = create_enhanced_gdpr_agent(
            self._get_db_path("gdpr_db"), 
            self._create_simple_logger("GDPR")
        )
        
        self.polish_law_agent = create_enhanced_polish_law_agent(
            self._get_db_path("polish_law_db"),
            self._create_simple_logger("PolishLaw")
        )
        
        self.internal_security_agent = create_enhanced_internal_security_agent(
            self._get_db_path("internal_security_db"),
            self._create_simple_logger("InternalSecurity")
        )
        
        self.summarization_agent = create_enhanced_summarization_agent(
            self._create_simple_logger("Summarization")
        )
        
        print("🔌 Connecting to vector stores...")
    
        if not self.gdpr_agent.connect_and_validate():
            print("⚠️  GDPR agent connection failed")
        else:
            print("✅ GDPR agent connected successfully")
            
        if not self.polish_law_agent.connect_and_validate():
            print("⚠️  Polish Law agent connection failed") 
        else:
            print("✅ Polish Law agent connected successfully")
            
        if not self.internal_security_agent.connect_and_validate():
            print("⚠️  Internal Security agent connection failed")
        else:
            print("✅ Internal Security agent connected successfully")

        # Create the workflow - clean and simple
        self._setup_workflow()
    
    def _get_db_path(self, db_name: str) -> str:
        """
        Simple path resolution when data is in the same directory as the script.
        
        Since main.py is in backend/ and data/ is also in backend/, we just need
        to look for data/ relative to where the script is located, not go up
        any directory levels.
        """
        # Get the directory where this script is located (backend/)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Build path to database - data is in the same parent directory as the script
        database_path = os.path.join(script_directory, "data", db_name)
        
        # Provide clear feedback about what we found
        if os.path.exists(database_path):
            print(f"✅ Found database: {db_name}")
            print(f"   Location: {database_path}")
        else:
            print(f"❌ Database not found: {db_name}")
            print(f"   Expected location: {database_path}")
        
        return database_path
    
    def _create_simple_logger(self, name: str):
        """Create a basic logger - agents handle their own detailed logging."""
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)  # Only show important messages
        return logger
    
    def _setup_workflow(self):
        """Set up the workflow graph - clean and focused."""
        workflow = StateGraph(AgentState)
        
        # Add nodes - each agent handles its own complexity
        workflow.add_node("gdpr", self.gdpr_agent.process)
        workflow.add_node("polish_law", self.polish_law_agent.process)
        workflow.add_node("internal_policy", self.internal_security_agent.process)
        workflow.add_node("summarization", self.summarization_agent.process)
        
        # Define workflow - clear linear progression
        workflow.add_edge("gdpr", "polish_law")
        workflow.add_edge("polish_law", "internal_policy")
        workflow.add_edge("internal_policy", "summarization")
        workflow.add_edge("summarization", END)
        
        workflow.set_entry_point("gdpr")
        self.app = workflow.compile()
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a query with clean, focused execution.
        
        The orchestrator's job is coordination, not detailed monitoring.
        Each agent handles its own sophisticated processing and logging.
        """
        # Create initial state
        initial_state = {
            "user_query": user_query,
            "gdpr_citations": [],
            "polish_law_citations": [],
            "internal_policy_citations": [],
            "summary": {}
        }
        
        # Execute workflow - let agents handle the complexity
        try:
            return self.app.invoke(initial_state)
        except Exception as e:
            # Simple error handling - agents provide detailed error info
            return {
                **initial_state,
                "summary": {
                    "action_plan": f"Workflow error: {str(e)}",
                    "error_details": {"message": str(e)}
                }
            }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a high-level system summary without overwhelming detail.
        
        For detailed statistics, users can access individual agents directly.
        This method provides just the essential information for orchestration.
        """
        summary = {"system_status": "operational", "agents": {}}
        
        # Collect high-level status from each agent
        for name, agent in [
            ("gdpr", self.gdpr_agent),
            ("polish_law", self.polish_law_agent), 
            ("internal_security", self.internal_security_agent),
            ("summarization", self.summarization_agent)
        ]:
            # Each agent can provide its own summary if it supports it
            if hasattr(agent, 'get_agent_statistics'):
                stats = agent.get_agent_statistics()
                summary["agents"][name] = {
                    "status": "operational",
                    "success_rate": stats.get('overall_success_rate_percent', 'unknown')
                }
            else:
                summary["agents"][name] = {"status": "operational"}
        
        return summary

def get_random_test_query(focus_area="by_category"):
    """Select a test query - clean and simple."""
    if focus_area == "complex":
        return random.choice(get_complex_eu_multi_domain_queries())
    elif focus_area == "intra_eu":
        return random.choice(get_intra_eu_coordination_queries())
    elif focus_area == "by_category":
        category = random.choice(get_eu_business_categories())
        queries = get_eu_business_queries_by_category(category)
        print(f"Testing {category} scenario...")
        return random.choice(queries)
    elif focus_area == "emerging_tech":
        return random.choice(get_eu_business_queries_by_category("emerging_technology_scenarios"))
    else:
        return random.choice(get_all_eu_business_queries())

def main():
    """
    Clean main execution focused on the essential workflow.
    
    This demonstrates how removing complexity from the orchestration layer
    makes the system much easier to understand and maintain. The sophisticated
    functionality is still there - it's just properly encapsulated in the
    components that know how to handle it.
    """
    print("\n🚀 Enhanced Multi-Agent System - Clean Orchestration")
    print("=" * 60)
    
    # Initialize system - agents handle their own setup complexity
    system = EnhancedMultiAgentSystem()
    
    # Get test query
    user_query = get_random_test_query()
    print(f"Query: {user_query}")
    print("-" * 60)
    
    # Process query - clean execution flow
    result = system.process_query(user_query)
    
    # Display results - focus on what matters to users
    print("\n📋 Results:")
    print("=" * 60)
    
    if "summary" in result and "action_plan" in result["summary"]:
        print(result["summary"]["action_plan"])
        
        # Show key metrics if available
        summary = result["summary"]
        if "total_citations" in summary:
            print(f"\n📊 Summary:")
            print(f"   Total Citations: {summary['total_citations']}")
            
            citations_by_source = summary.get('citations_by_source', {})
            for source, count in citations_by_source.items():
                print(f"   {source.replace('_', ' ').title()}: {count}")
            
            if "overall_precision_rate" in summary:
                print(f"   Precision Rate: {summary['overall_precision_rate']}%")
    else:
        print("No results generated - check agent logs for details")
    
    # Optional: Show high-level system status
    if "--verbose" in os.sys.argv:
        print(f"\n🔧 System Summary:")
        summary = system.get_system_summary()
        for agent_name, agent_info in summary["agents"].items():
            status = agent_info.get("status", "unknown")
            success_rate = agent_info.get("success_rate", "unknown")
            print(f"   {agent_name.title()}: {status} ({success_rate}% success)")

if __name__ == "__main__":
    main()