import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class SummarizationAgent:
    """
    Enhanced Summarization Agent with precise citation formatting support.
    
    This agent receives citations from the enhanced GDPR, Polish law, and internal policy
    agents and formats them using the unified precision citation format. It creates
    professional action plans with numbered citations that maintain the sophisticated
    structural information preserved by the enhanced agent system.
    
    The agent demonstrates how the complete pipeline works together to create
    comprehensive legal guidance with verifiable, precise citations.
    """
    
    def __init__(self):
        # Initialize the OpenAI model for sophisticated summarization
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the enhanced prompt template for structured summarization with precise citations
        self.prompt = ChatPromptTemplate.from_template(
            """You are a legal compliance expert who provides clear, actionable guidance with precise legal citations.
            
            Given the user query: {user_query}
            
            And considering all legal citations provided with their precise structural references, create a step-by-step action plan.
            
            FORMAT REQUIREMENTS:
            1. Each action item should be an ordered number starting with an action verb
            2. Include numbered citations in brackets [1] [2] after relevant statements
            3. Keep action items concise but complete with specific legal requirements
            4. Focus on practical, implementable steps that ensure compliance
            5. Reference the precise citation numbers to maintain legal traceability
            
            Format each point like this:
            [number]. [Action Step] - [Detailed explanation with specific compliance requirements] [citation number 1] [citation number 2]
            
            Available Legal Sources with Precise Citations:
            {all_citations_with_numbers}
            
            Create a comprehensive numbered action plan that addresses the user's query with proper numbered citations that reference the precise legal provisions identified by the enhanced agent system.
            """
        )
    
    def create_numbered_citations(self, gdpr_citations, polish_law_citations, internal_policy_citations):
        """
        Create a numbered list of all citations with enhanced formatting support.
        
        This method processes citations from all three enhanced agents and creates
        a unified numbering system while preserving the precise citation format
        that each agent created. The result maintains the sophisticated structural
        information while making it accessible for action plan creation.
        """
        all_citations = []
        citation_number = 1
        
        # Process GDPR citations with enhanced formatting
        for citation in gdpr_citations:
            # Extract the precise article reference created by the enhanced GDPR agent
            article_reference = citation.get("article", "")
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "GDPR",
                "source": "GDPR",
                "article": article_reference,  # This now contains precise format like "Article 1, paragraph 2(c) (Chapter I: General provisions)"
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Process Polish law citations with enhanced formatting
        for citation in polish_law_citations:
            # Extract the precise article reference created by the enhanced Polish law agent
            article_reference = citation.get("article", "")
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "Polish Law",
                "source": "Polish Data Protection Act",
                "article": article_reference,  # This now contains precise format like "Article 3, paragraph 1(2) (Chapter 2: Obligations)"
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Process internal policy citations with enhanced formatting
        for citation in internal_policy_citations:
            # Extract the precise section reference (adapt to internal policy structure)
            article_reference = citation.get("article", citation.get("section", ""))
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "Internal Policy",
                "source": "Internal Data Protection Policy",
                "article": article_reference,  # This should contain precise format when internal policy agent is enhanced
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Create the formatted text for the LLM prompt
        # This provides context for creating the action plan while preserving citation precision
        citations_text = "\n".join([
            f"[{cite['number']}] {cite['source']} {cite['article']}: {cite['explanation']}"
            for cite in all_citations
        ])
        
        return all_citations, citations_text
    
    def format_enhanced_citations_by_source(self, all_citations):
        """
        Format citations using the enhanced precision format grouped by legal source.
        
        This method creates the exact citation format requested, showcasing the
        precision achieved by the enhanced agent system. It demonstrates how
        sophisticated legal document structure is preserved through the entire
        pipeline from JSON processing to final user display.
        """
        formatted_citations = ""
        
        # Group citations by source type while preserving order
        source_groups = {
            "GDPR": [],
            "Polish Law": [],
            "Internal Policy": []
        }
        
        # Group citations while maintaining the enhanced formatting
        for citation in all_citations:
            source_type = citation["source_type"]
            if source_type in source_groups:
                source_groups[source_type].append(citation)
        
        # Format each source group with enhanced precision
        for source_type, citations in source_groups.items():
            if citations:  # Only show sections that have citations
                if source_type == "GDPR":
                    formatted_citations += "**GDPR:**\n"
                elif source_type == "Polish Law":
                    formatted_citations += "**Polish Data Protection Act:**\n"
                elif source_type == "Internal Policy":
                    formatted_citations += "**Internal Data Protection Policy:**\n"
                
                # Add each citation with enhanced precision formatting
                for citation in citations:
                    # Use the precise article reference created by enhanced agents
                    precise_reference = citation["article"]
                    quote = citation["quote"]
                    citation_number = citation["number"]
                    
                    # Format in the exact style requested with precise structural references
                    formatted_citations += f"[{citation_number}] {precise_reference}: \"{quote}\"\n"
                
                formatted_citations += "\n"  # Add spacing between source groups
        
        return formatted_citations.strip()  # Remove trailing whitespace
    
    def format_final_response(self, action_plan, all_citations):
        """
        Format the final response with action plan and enhanced precision citations.
        
        This method creates the complete response that showcases the sophisticated
        capabilities of the enhanced agent system. The final output demonstrates
        how complex legal document structure is preserved and made actionable
        through the entire pipeline.
        """
        # Start with the action plan created by the LLM
        final_response = action_plan + "\n\n"
        
        # Add the enhanced precision citations grouped by source
        final_response += "**LEGAL CITATIONS:**\n\n"
        
        # Use the enhanced citation formatting method
        enhanced_citations = self.format_enhanced_citations_by_source(all_citations)
        final_response += enhanced_citations
        
        return final_response
    
    def create_processing_summary(self, all_citations, gdpr_citations, polish_law_citations, internal_policy_citations):
        """
        Create a comprehensive processing summary showing the enhanced system's capabilities.
        
        This method provides transparency into how well the enhanced system worked
        and what level of precision was achieved across different legal sources.
        """
        # Calculate precision statistics
        enhanced_gdpr_count = sum(1 for cite in gdpr_citations 
                                 if "paragraph" in cite.get("article", "").lower())
        enhanced_polish_count = sum(1 for cite in polish_law_citations 
                                   if "paragraph" in cite.get("article", "").lower())
        enhanced_internal_count = sum(1 for cite in internal_policy_citations 
                                     if "paragraph" in cite.get("article", "").lower() or 
                                        "section" in cite.get("article", "").lower())
        
        summary = {
            "total_citations": len(all_citations),
            "citations_by_source": {
                "gdpr": len(gdpr_citations),
                "polish_law": len(polish_law_citations), 
                "internal_policy": len(internal_policy_citations)
            },
            "precision_statistics": {
                "enhanced_gdpr_citations": enhanced_gdpr_count,
                "enhanced_polish_citations": enhanced_polish_count,
                "enhanced_internal_citations": enhanced_internal_count,
                "total_enhanced": enhanced_gdpr_count + enhanced_polish_count + enhanced_internal_count
            }
        }
        
        # Calculate overall precision rate
        if len(all_citations) > 0:
            precision_rate = (summary["precision_statistics"]["total_enhanced"] / len(all_citations)) * 100
            summary["overall_precision_rate"] = round(precision_rate, 1)
        else:
            summary["overall_precision_rate"] = 0
        
        return summary
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all citations and create a structured action plan with enhanced precision citations.
        
        This method represents the culmination of the enhanced agent system, taking
        the sophisticated citations created by the enhanced GDPR and Polish law agents
        and presenting them in a comprehensive, actionable format that maintains
        the precision achieved through the metadata flattening approach.
        """
        print("\n📊 [STEP 4/4] ENHANCED SUMMARIZATION AGENT: Creating structured action plan with precision citations...")
        
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        polish_law_citations = state["polish_law_citations"]
        internal_policy_citations = state["internal_policy_citations"]
        
        print(f"📝 Processing citations from enhanced agents:")
        print(f"   - GDPR: {len(gdpr_citations)} citations")
        print(f"   - Polish Law: {len(polish_law_citations)} citations")
        print(f"   - Internal Policy: {len(internal_policy_citations)} citations")
        
        # Create numbered citations with enhanced formatting preservation
        all_citations, citations_text = self.create_numbered_citations(
            gdpr_citations, polish_law_citations, internal_policy_citations
        )
        
        print(f"📋 Created unified numbering system for {len(all_citations)} total citations")
        
        # Create the chain and get action plan using enhanced citation context
        chain = self.prompt | self.model
        response = chain.invoke({
            "user_query": user_query,
            "all_citations_with_numbers": citations_text
        })
        
        print("🎯 Generated action plan with precise legal citations")
        
        # Format the final response with enhanced precision citations
        formatted_response = self.format_final_response(response.content, all_citations)
        
        # Create a comprehensive processing summary
        processing_summary = self.create_processing_summary(
            all_citations, gdpr_citations, polish_law_citations, internal_policy_citations
        )
        
        # Create the final summary structure
        summary = {
            "action_plan": formatted_response,
            **processing_summary
        }
        
        state["summary"] = summary
        
        print("✅ Completed: Enhanced action plan with precision citations generated")
        print(f"📊 Precision rate: {processing_summary['overall_precision_rate']}% of citations include paragraph-level detail")
        
        return state