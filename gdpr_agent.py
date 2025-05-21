import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class GDPRAgent:
    def __init__(self):
        # Initialize the OpenAI model
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the prompt template for GDPR analysis
        self.prompt = ChatPromptTemplate.from_template(
            """You are a specialized GDPR legal expert.
            
            Given the user query: {user_query}
            
            Identify the most relevant GDPR articles, recitals, and provisions that apply to this query.
            For each citation, provide:
            1. The specific article/recital number
            2. A direct quote of the relevant text
            3. A brief explanation of its relevance to the query
            
            Format your response as a structured list of citations.
            """
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query and extract GDPR citations."""
        print("\n🔍 [STEP 1/4] GDPR AGENT: Analyzing query for relevant GDPR provisions...")
        user_query = state["user_query"]
        
        # Create the chain
        chain = self.prompt | self.model
        
        # Invoke the chain
        response = chain.invoke({"user_query": user_query})
        
        # Later here will be a call this would be a RAG-based retrieval
        # For now, I'll just hardcode some articles
        citations = [
            {
                "source": "GDPR",
                "article": "Article 9",
                "text": "Processing of special categories of personal data",
                "quote": "Processing of personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, or trade union membership, and the processing of genetic data, biometric data for the purpose of uniquely identifying a natural person, data concerning health or data concerning a natural person's sex life or sexual orientation shall be prohibited.",
                "explanation": "Article 9 specifically addresses the processing of sensitive data categories, which are subject to stricter conditions."
            },
            {
                "source": "GDPR",
                "article": "Article 9(2)",
                "text": "Exceptions for processing sensitive data",
                "quote": "Paragraph 1 shall not apply if one of the following applies: (a) the data subject has given explicit consent...",
                "explanation": "This section outlines the conditions under which sensitive data can be lawfully processed."
            }
        ]
        
        # Update the state with GDPR citations
        state["gdpr_citations"] = citations
        print(f"✅ Completed: Found {len(state['gdpr_citations'])} GDPR citations")
        
        return state