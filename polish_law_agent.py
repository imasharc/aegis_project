import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class PolishLawAgent:
    def __init__(self):
        # Initialize the OpenAI model
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the prompt template for Polish law analysis
        self.prompt = ChatPromptTemplate.from_template(
            """You are a specialized Polish data protection law expert.
            
            Given the user query: {user_query}
            
            And considering these GDPR citations:
            {gdpr_citations}
            
            Identify the most relevant Polish implementation laws, regulations, and provisions that apply to this query.
            For each citation, provide:
            1. The specific law/article number
            2. A direct quote of the relevant text
            3. A brief explanation of its relevance to the query and how it relates to the GDPR provisions
            
            Format your response as a structured list of citations.
            """
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query with GDPR context and extract Polish law citations."""
        print("\n📜 [STEP 2/4] POLISH LAW AGENT: Identifying Polish implementation laws...")
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        
        # Format GDPR citations for the prompt
        gdpr_citations_text = "\n".join([
            f"- {cite['article']}: {cite['quote']} - {cite['explanation']}"
            for cite in gdpr_citations
        ])
        
        # Create the chain
        chain = self.prompt | self.model
        
        # Invoke the chain
        response = chain.invoke({
            "user_query": user_query,
            "gdpr_citations": gdpr_citations_text
        })
        
        # Later here will be a call this would be a RAG-based retrieval
        # For now, I'll just hardcode some articles
        
        citations = [
            {
                "source": "Polish Data Protection Act",
                "article": "Article 27",
                "text": "Processing of sensitive personal data",
                "quote": "The processing of sensitive personal data is subject to additional safeguards under Polish law, including notification requirements to the Polish Data Protection Authority.",
                "explanation": "This article implements GDPR Article 9 requirements in the Polish legal context."
            },
            {
                "source": "Polish Labor Code",
                "article": "Article 221",
                "text": "Processing of employee personal data",
                "quote": "Employers are entitled to process specific categories of personal data of job candidates and employees as listed in this provision.",
                "explanation": "This article provides specific rules for handling sensitive employee data in the Polish workplace."
            }
        ]
        
        # Update the state with Polish law citations
        state["polish_law_citations"] = citations
        print(f"✅ Completed: Found {len(state['polish_law_citations'])} Polish law citations")
        
        return state