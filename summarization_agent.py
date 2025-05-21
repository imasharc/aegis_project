import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class SummarizationAgent:
    def __init__(self):
        # Initialize the OpenAI model
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the prompt template for summarization
        self.prompt = ChatPromptTemplate.from_template(
            """You are a legal compliance expert who provides clear, actionable guidance.
            
            Given the user query: {user_query}
            
            And considering all of these citations:
            
            GDPR Citations:
            {gdpr_citations}
            
            Polish Law Citations:
            {polish_law_citations}
            
            Internal Policy Citations:
            {internal_policy_citations}
            
            Create a comprehensive yet concise step-by-step action plan that addresses the user's query.
            For each step:
            1. Provide a clear, actionable instruction
            2. Include specific references to the legal requirements and internal policies that support this step
            3. Note any specific considerations for Polish operations
            
            Format your response as a bulleted action plan with proper citations to all the legal sources.
            """
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all citations and create a summarized action plan."""
        print("\n📊 [STEP 4/4] SUMMARIZATION AGENT: Creating final action plan with citations...")
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        polish_law_citations = state["polish_law_citations"]
        internal_policy_citations = state["internal_policy_citations"]
        
        # Format citations for the prompt
        gdpr_citations_text = "\n".join([
            f"- {cite['article']}: {cite['quote']} - {cite['explanation']}"
            for cite in gdpr_citations
        ])
        
        polish_law_citations_text = "\n".join([
            f"- {cite['article']}: {cite['quote']} - {cite['explanation']}"
            for cite in polish_law_citations
        ])
        
        internal_policy_citations_text = "\n".join([
            f"- {cite['source']}, {cite['section']}: {cite['quote']} - {cite['explanation']}"
            for cite in internal_policy_citations
        ])
        
        # Create the chain
        chain = self.prompt | self.model
        
        # Invoke the chain
        response = chain.invoke({
            "user_query": user_query,
            "gdpr_citations": gdpr_citations_text,
            "polish_law_citations": polish_law_citations_text,
            "internal_policy_citations": internal_policy_citations_text
        })
        
        # Create a structured summary with the action plan
        summary = {
            "action_plan": response.content,
            "references": {
                "gdpr": gdpr_citations,
                "polish_law": polish_law_citations,
                "internal_policy": internal_policy_citations
            }
        }
        
        # Update the state with the summary
        state["summary"] = summary
        print("✅ Completed: Action plan generated")
        
        return state