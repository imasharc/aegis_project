import os
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from gdpr_agent import GDPRAgent
from polish_law_agent import PolishLawAgent
from internal_policy_agent import InternalPolicyAgent
from summarization_agent import SummarizationAgent

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
    
# Initialize agents
gdpr_agent = GDPRAgent()
polish_law_agent = PolishLawAgent()
internal_policy_agent = InternalPolicyAgent()
summarization_agent = SummarizationAgent()

# Create a new graph
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("gdpr", gdpr_agent.process)
workflow.add_node("polish_law", polish_law_agent.process)
workflow.add_node("internal_policy", internal_policy_agent.process)
workflow.add_node("summarization", summarization_agent.process)

# Define the edges of the graph (the flow)
workflow.add_edge("gdpr", "polish_law")
workflow.add_edge("polish_law", "internal_policy")
workflow.add_edge("internal_policy", "summarization")
workflow.add_edge("summarization", END)

# Set the entry point
workflow.set_entry_point("gdpr")

# Compile the graph
app = workflow.compile()

def main():
    # Hardcoded user query for now (will be replaced with Streamlit input later)
    print("\n===== STARTING LANGGRAPH WORKFLOW =====")
    user_query = "What are the requirements for processing sensitive personal data under GDPR in our Polish branch?"
    print("=======================================\n")
    
    # Initial state
    initial_state = {
        "user_query": user_query,
        "gdpr_citations": [],
        "polish_law_citations": [],
        "internal_policy_citations": [],
        "summary": {}
    }
    
    # Run the graph
    print("Executing graph workflow...")
    result = app.invoke(initial_state)
    print("\n===== WORKFLOW COMPLETED =====")
    
    # Print the final summary
    print("\n----- FINAL SUMMARY -----")
    if "summary" in result and "action_plan" in result["summary"]:
        print(result["summary"]["action_plan"])
    else:
        print("No summary generated.")

if __name__ == "__main__":
    main()