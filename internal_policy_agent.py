import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class InternalPolicyAgent:
    def __init__(self):
        # Initialize the OpenAI model
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the prompt template for internal policy analysis
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert on internal company data protection policies.
            
            Given the user query: {user_query}
            
            And considering these GDPR citations:
            {gdpr_citations}
            
            And these Polish law citations:
            {polish_law_citations}
            
            Identify the relevant internal company policies, procedures, and guidelines that would apply.
            For each policy, provide:
            1. The policy name/document reference
            2. The specific section or requirement
            3. A brief explanation of how it implements the legal requirements
            
            Format your response as a structured list of internal policies.
            """
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the query with legal context and extract internal policy guidelines."""
        print("\n📋 [STEP 3/4] INTERNAL POLICY AGENT: Matching with company policies...")
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        polish_law_citations = state["polish_law_citations"]
        
        # Format citations for the prompt
        gdpr_citations_text = "\n".join([
            f"- {cite['article']}: {cite['quote']} - {cite['explanation']}"
            for cite in gdpr_citations
        ])
        
        polish_law_citations_text = "\n".join([
            f"- {cite['article']}: {cite['quote']} - {cite['explanation']}"
            for cite in polish_law_citations
        ])
        
        # Create the chain
        chain = self.prompt | self.model
        
        # Invoke the chain
        response = chain.invoke({
            "user_query": user_query,
            "gdpr_citations": gdpr_citations_text,
            "polish_law_citations": polish_law_citations_text
        })
        
        # Later here will be a call this would be a RAG-based retrieval
        # For now, I'll just hardcode some articles
        
        citations = [
            {
                "source": "Company Data Protection Policy",
                "section": "Section 4.3: Sensitive Data Handling",
                "text": "Procedures for processing special categories of data",
                "quote": "All processing of sensitive personal data requires explicit documentation of the legal basis, proper consent collection, and approval from the Data Protection Officer.",
                "explanation": "This policy implements both GDPR Article 9 and Polish Data Protection Act Article 27 requirements."
            },
            {
                "source": "Polish Branch Procedure Manual",
                "section": "PR-PL-021: Employee Data Processing",
                "text": "Specific procedures for the Polish branch",
                "quote": "The HR department must use the standardized consent forms in both Polish and English, and must maintain records in the centralized HR system with appropriate access controls.",
                "explanation": "This procedure addresses the specific requirements of Polish Labor Code Article 221."
            }
        ]
        
        # Update the state with internal policy citations
        state["internal_policy_citations"] = citations
        print(f"✅ Completed: Found {len(state['internal_policy_citations'])} internal policy guidelines")
        
        return state