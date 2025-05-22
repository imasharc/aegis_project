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
        
        # Define the prompt template for structured summarization
        self.prompt = ChatPromptTemplate.from_template(
            """You are a legal compliance expert who provides clear, actionable guidance.
            
            Given the user query: {user_query}
            
            And considering all legal citations provided, create a step-by-step action plan.
            
            FORMAT REQUIREMENTS:
            1. Each action item should be an ordered number starting with an action verb
            2. Include numbered citations in brackets [1] [2] after relevant statements
            3. Keep action items concise but complete
            4. Focus on practical, implementable steps

            Format each point like this:
            [number]. [Action Step] - [Detailed explanation with specific requirements] [citation number 1] [citation number 2]
            
            Available Legal Sources:
            {all_citations_with_numbers}
            
            Create a numbered action plan that addresses the user's query with proper numbered citations.
            """
        )
    
    def create_numbered_citations(self, gdpr_citations, polish_law_citations, internal_policy_citations):
        """Create a numbered list of all citations and return both the numbered text and citation mapping."""
        all_citations = []
        citation_number = 1
        
        # Add GDPR citations
        for citation in gdpr_citations:
            numbered_citation = {
                "number": citation_number,
                "source_type": "GDPR",
                "source": citation.get("source", "GDPR"),
                "article": citation.get("article", ""),
                "quote": citation.get("quote", ""),
                "explanation": citation.get("explanation", "")
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Add Polish law citations
        for citation in polish_law_citations:
            numbered_citation = {
                "number": citation_number,
                "source_type": "Polish Law",
                "source": citation.get("source", "Polish Law"),
                "article": citation.get("article", ""),
                "quote": citation.get("quote", ""),
                "explanation": citation.get("explanation", "")
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Add internal policy citations
        for citation in internal_policy_citations:
            numbered_citation = {
                "number": citation_number,
                "source_type": "Internal Policy",
                "source": citation.get("source", "Internal Policy"),
                "article": citation.get("section", citation.get("article", "")),
                "quote": citation.get("quote", ""),
                "explanation": citation.get("explanation", "")
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Create the formatted text for the prompt
        citations_text = "\n".join([
            f"[{cite['number']}] {cite['source']} {cite['article']}: {cite['explanation']}"
            for cite in all_citations
        ])
        
        return all_citations, citations_text
    
    def format_final_response(self, action_plan, all_citations):
        """Format the final response with action plan and categorized citations."""
        
        # Group citations by source document
        citations_by_source = {}
        for citation in all_citations:
            source = citation["source"]
            if source not in citations_by_source:
                citations_by_source[source] = []
            citations_by_source[source].append(citation)
        
        # Build the final formatted response
        final_response = action_plan + "\n\n"
        
        # Add categorized citations
        final_response += "**LEGAL CITATIONS:**\n\n"
        
        for source, citations in citations_by_source.items():
            final_response += f"**{source}:**\n"
            for citation in citations:
                final_response += f"[{citation['number']}] {citation['article']}: \"{citation['quote']}\"\n"
            final_response += "\n"
        
        return final_response
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all citations and create a structured action plan with numbered citations."""
        print("\n📊 [STEP 4/4] SUMMARIZATION AGENT: Creating structured action plan with numbered citations...")
        
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        polish_law_citations = state["polish_law_citations"]
        internal_policy_citations = state["internal_policy_citations"]
        
        # Create numbered citations
        all_citations, citations_text = self.create_numbered_citations(
            gdpr_citations, polish_law_citations, internal_policy_citations
        )
        
        print(f"📝 Processed {len(all_citations)} total citations for numbering")
        
        # Create the chain and get action plan
        chain = self.prompt | self.model
        response = chain.invoke({
            "user_query": user_query,
            "all_citations_with_numbers": citations_text
        })
        
        # Format the final response with categorized citations
        formatted_response = self.format_final_response(response.content, all_citations)
        
        # Create a structured summary
        summary = {
            "action_plan": formatted_response,
            "total_citations": len(all_citations),
            "citations_by_source": {
                "gdpr": len(gdpr_citations),
                "polish_law": len(polish_law_citations), 
                "internal_policy": len(internal_policy_citations)
            }
        }
        
        state["summary"] = summary
        print("✅ Completed: Structured action plan with numbered citations generated")
        
        return state