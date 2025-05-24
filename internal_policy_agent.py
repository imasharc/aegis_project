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
        """Process the user query with GDPR and Polish law context."""
        print("\n📋 [STEP 3/4] INTERNAL POLICY AGENT: Matching with company policies...")
        user_query = state["user_query"]
        
        # Pass the original citation variables directly
        chain = self.prompt | self.model
        
        # Invoke the chain with the expected variable names
        response = chain.invoke({
            "user_query": user_query,
            "gdpr_citations": state["gdpr_citations"],
            "polish_law_citations": state["polish_law_citations"]
        })
        
        # Hardcoded internal policies for now
        internal_policies = [
            {
                "policy_id": "DP-001",
                "title": "Data Processing Policy",
                "content": "Guidelines for processing sensitive personal data in compliance with GDPR and Polish data protection laws."
            },
            {
                "policy_id": "DP-002",
                "title": "Data Subject Rights Policy",
                "content": "Procedures for handling data subject requests in Poland."
            }
        ]
        
        state["internal_policy_citations"] = internal_policies
        print(f"✅ Completed: Found {len(internal_policies)} relevant internal policies")
        
        return state