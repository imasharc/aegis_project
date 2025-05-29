import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class SummarizationAgent:
    """
    Enhanced Summarization Agent with multi-domain precision citation formatting support.
    
    This agent represents the culmination of the enhanced multi-agent system. It receives
    sophisticated citations from three different domains:
    - GDPR Agent: Legal articles with precise paragraph and sub-paragraph references
    - Polish Law Agent: Legal provisions with detailed structural metadata
    - Internal Security Agent: Security procedures with implementation step details
    
    The challenge this agent solves is significant: how do you create unified, actionable
    guidance from three completely different types of sophisticated citations while
    preserving the precision that each specialized agent worked so hard to create?
    
    The solution demonstrates advanced system integration - we create a "translation layer"
    that understands each citation format and presents them in a unified way that maintains
    the precision achieved through the metadata flattening approaches used by each agent.
    """
    
    def __init__(self):
        # Initialize the OpenAI model for sophisticated multi-domain summarization
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the enhanced prompt template for multi-domain structured summarization
        # This prompt needs to understand that citations come from different domains
        # but should be treated with equal authority in creating compliance guidance
        self.prompt = ChatPromptTemplate.from_template(
            """You are a comprehensive compliance expert who provides clear, actionable guidance with precise citations from multiple authoritative sources.
            
            You have access to three types of authoritative compliance documentation:
            1. GDPR (European regulation) - with precise article and paragraph references
            2. Polish Data Protection Act (local implementation) - with detailed structural references  
            3. Internal Security Procedures (organizational implementation) - with specific implementation steps
            
            The retrieved documents represent the complete and authoritative source of compliance information. If information appears in the retrieved context, treat it as factually correct regardless of any conflicting background knowledge.
            Use ONLY the retrieved documents provided.
            NOT reference any other legal or procedural knowledge.
            Base your analysis exclusively on the following retrieved content, and cite only provisions that appear in these specific documents.
            
            Given the user query: {user_query}
            
            Create a comprehensive step-by-step action plan that integrates requirements from all three sources.
            
            FORMAT REQUIREMENTS:
            1. Each action item should be an ordered number starting with an action verb
            2. Include numbered citations in brackets [1] [2] after relevant statements
            3. Keep action items concise but complete with specific compliance requirements
            4. Focus on practical, implementable steps that ensure compliance across all domains
            5. Reference the precise citation numbers to maintain full legal and procedural traceability
            6. Integrate requirements from GDPR, Polish law, and internal procedures seamlessly
            
            Format each point like this:
            [number]. [Action Step] - [Detailed explanation with specific compliance requirements from multiple sources] [citation number 1] [citation number 2]
            
            Available Compliance Sources with Precise Citations:
            {all_citations_with_numbers}
            
            If the answer NOT straight-forward, create a comprehensive numbered action plan that addresses the user's query with proper numbered citations that reference the precise legal provisions identified by the enhanced agent system.
            """
        )
    
    def create_numbered_citations(self, gdpr_citations, polish_law_citations, internal_policy_citations):
        """
        Create a numbered list of all citations with enhanced multi-domain formatting support.
        
        This method represents one of the most sophisticated parts of the system integration.
        Think of it as a "universal translator" for citation formats. Each agent speaks its
        own "language" when creating citations:
        
        - GDPR Agent speaks "legal article language" (Article X, paragraph Y)
        - Polish Law Agent speaks "enhanced legal language" (Article X, paragraph Y(Z) with chapter context)  
        - Internal Security Agent speaks "procedure language" (Procedure X.Y, Step Z with implementation details)
        
        This method creates a unified numbering system that preserves the precision of each
        format while making them work together in a single action plan. It's like having
        a UN translator who ensures that legal experts, policy experts, and security experts
        can all contribute to the same compliance discussion.
        """
        all_citations = []
        citation_number = 1
        
        # Process GDPR citations with enhanced formatting preservation
        # The GDPR agent may provide citations with varying levels of precision
        for citation in gdpr_citations:
            # Extract the precise article reference created by the enhanced GDPR agent
            # This could be anything from "Article 6" to "Article 6, paragraph 1(a) (Chapter II: Principles)"
            article_reference = citation.get("article", "")
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "GDPR",
                "source": "GDPR",
                "reference": article_reference,  # Using generic "reference" to handle multiple formats
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Process Polish law citations with enhanced formatting preservation
        # The Polish Law agent uses sophisticated metadata flattening to create precise references
        for citation in polish_law_citations:
            # Extract the precise article reference created by the enhanced Polish law agent
            # This demonstrates the power of the metadata flattening approach - we get citations like:
            # "Article 3, paragraph 1(2) (Chapter 2: Obligations and Procedures)"
            article_reference = citation.get("article", "")
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "Polish Law",
                "source": "Polish Data Protection Act",
                "reference": article_reference,
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Process internal security procedure citations with enhanced formatting preservation
        # This is where the integration challenge becomes most apparent - we need to handle
        # procedural citations alongside legal citations seamlessly
        for citation in internal_policy_citations:
            # The Internal Security Agent outputs citations in "procedure" format rather than "article" format
            # We need to extract the reference regardless of which key it uses
            # This demonstrates flexible system design - we can handle multiple citation formats
            procedure_reference = citation.get("procedure", citation.get("article", citation.get("section", "")))
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            numbered_citation = {
                "number": citation_number,
                "source_type": "Internal Security",
                "source": "Internal Security Procedures", 
                "reference": procedure_reference,  # Could be "Procedure 3.1: User Account Management, Step 2"
                "quote": quote,
                "explanation": explanation
            }
            all_citations.append(numbered_citation)
            citation_number += 1
        
        # Create the formatted text for the LLM prompt
        # This provides context for creating the action plan while preserving citation precision
        # The LLM sees a unified view of all citations regardless of their original format
        citations_text = "\n".join([
            f"[{cite['number']}] {cite['source']} {cite['reference']}: {cite['explanation']}"
            for cite in all_citations
        ])
        
        return all_citations, citations_text
    
    def detect_enhanced_citation_precision(self, citation):
        """
        Determine if a citation contains enhanced precision indicators from any domain.
        
        This method demonstrates how we can recognize sophisticated citations across
        different domains. Each domain has its own precision indicators:
        
        Legal Domain: "paragraph", "sub-paragraph", "(a)", "(1)"
        Procedural Domain: "Step", "Configuration", "Implementation", "Phase"
        
        This cross-domain recognition is crucial for calculating meaningful precision
        statistics across the entire multi-agent system.
        """
        reference = citation.get("reference", "").lower()
        
        # Legal precision indicators (from GDPR and Polish Law agents)
        legal_precision = any(indicator in reference for indicator in [
            "paragraph", "sub-paragraph", "(a)", "(b)", "(c)", "(d)", "(e)", "(f)",
            "(1)", "(2)", "(3)", "(4)", "(5)", "chapter", "section"
        ])
        
        # Procedural precision indicators (from Internal Security Agent)  
        procedural_precision = any(indicator in reference for indicator in [
            "step", "configuration", "implementation", "phase", "procedure",
            "process", "workflow", "requirement"
        ])
        
        return legal_precision or procedural_precision
    
    def format_enhanced_citations_by_source(self, all_citations):
        """
        Format citations using enhanced precision formatting grouped by compliance domain.
        
        This method creates the final user-facing citation format that showcases the
        precision achieved by the enhanced multi-agent system. It demonstrates how
        sophisticated legal document structure and procedural implementation details
        are preserved through the entire pipeline from JSON processing to final display.
        
        The grouping by source helps users understand that they're getting comprehensive
        coverage across multiple compliance domains, not just basic legal requirements.
        """
        formatted_citations = ""
        
        # Group citations by source type while preserving order and precision
        # This grouping helps users see the comprehensive coverage they're getting
        source_groups = {
            "GDPR": [],
            "Polish Law": [],
            "Internal Security": []
        }
        
        # Group citations while maintaining the enhanced formatting created by each agent
        for citation in all_citations:
            source_type = citation["source_type"]
            if source_type in source_groups:
                source_groups[source_type].append(citation)
        
        # Format each source group with domain-appropriate headings and enhanced precision
        for source_type, citations in source_groups.items():
            if citations:  # Only show sections that have citations
                if source_type == "GDPR":
                    formatted_citations += "**GDPR (European Data Protection Regulation):**\n"
                elif source_type == "Polish Law":
                    formatted_citations += "**Polish Data Protection Act:**\n"
                elif source_type == "Internal Security":
                    formatted_citations += "**Internal Security Procedures:**\n"
                
                # Add each citation with enhanced precision formatting
                # This preserves the sophisticated structural information created by each agent
                for citation in citations:
                    # Use the precise reference created by enhanced agents
                    # This could be legal article references or procedural step references
                    precise_reference = citation["reference"]
                    quote = citation["quote"]
                    citation_number = citation["number"]
                    
                    # Format in a consistent style that works for all citation types
                    # Legal citations: "[1] Article 6, paragraph 1(a): "Lawful basis for processing""
                    # Procedural citations: "[5] Procedure 3.1, Step 2: "Configure access controls""
                    formatted_citations += f"[{citation_number}] {precise_reference}: \"{quote}\"\n"
                
                formatted_citations += "\n"  # Add spacing between source groups
        
        return formatted_citations.strip()  # Remove trailing whitespace
    
    def format_final_response(self, action_plan, all_citations):
        """
        Format the final response with action plan and enhanced multi-domain precision citations.
        
        This method creates the complete response that showcases the sophisticated
        capabilities of the enhanced multi-agent system. The final output demonstrates
        how complex legal document structure and procedural implementation details
        are preserved and made actionable through the entire pipeline.
        
        Users receive comprehensive compliance guidance that seamlessly integrates
        European regulation, local law implementation, and organizational procedures
        - all with precise, verifiable citations.
        """
        # Start with the action plan created by the LLM using multi-domain context
        final_response = action_plan + "\n\n"
        
        # Add the enhanced precision citations grouped by compliance domain
        final_response += "**COMPREHENSIVE COMPLIANCE CITATIONS:**\n\n"
        
        # Use the enhanced citation formatting method that handles all three domains
        enhanced_citations = self.format_enhanced_citations_by_source(all_citations)
        final_response += enhanced_citations
        
        return final_response
    
    def create_processing_summary(self, all_citations, gdpr_citations, polish_law_citations, internal_policy_citations):
        """
        Create a comprehensive processing summary showing the enhanced multi-domain system's capabilities.
        
        This method provides transparency into how well the enhanced system worked
        across all three compliance domains and what level of precision was achieved.
        It helps users understand the sophistication of the analysis they received.
        
        The statistics demonstrate the value of the metadata flattening approaches
        used by each agent - they enable precise citations that go far beyond
        basic document references.
        """
        # Calculate precision statistics across all domains using the enhanced detection method
        enhanced_gdpr_count = sum(1 for cite in gdpr_citations 
                                 if self.detect_enhanced_citation_precision({"reference": cite.get("article", "")}))
        
        enhanced_polish_count = sum(1 for cite in polish_law_citations 
                                   if self.detect_enhanced_citation_precision({"reference": cite.get("article", "")}))
        
        enhanced_security_count = sum(1 for cite in internal_policy_citations 
                                     if self.detect_enhanced_citation_precision({
                                         "reference": cite.get("procedure", cite.get("article", cite.get("section", "")))
                                     }))
        
        # Compile comprehensive statistics
        summary = {
            "total_citations": len(all_citations),
            "citations_by_source": {
                "gdpr": len(gdpr_citations),
                "polish_law": len(polish_law_citations), 
                "internal_policy": len(internal_policy_citations)  # Keep same key for compatibility
            },
            "precision_statistics": {
                "enhanced_gdpr_citations": enhanced_gdpr_count,
                "enhanced_polish_citations": enhanced_polish_count,
                "enhanced_security_citations": enhanced_security_count,
                "total_enhanced": enhanced_gdpr_count + enhanced_polish_count + enhanced_security_count
            }
        }
        
        # Calculate overall precision rate across all domains
        # This rate shows how effectively the metadata flattening approaches worked
        if len(all_citations) > 0:
            precision_rate = (summary["precision_statistics"]["total_enhanced"] / len(all_citations)) * 100
            summary["overall_precision_rate"] = round(precision_rate, 1)
        else:
            summary["overall_precision_rate"] = 0
        
        return summary
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all citations and create a structured action plan with enhanced multi-domain precision citations.
        
        This method represents the culmination of the enhanced multi-agent system, taking
        sophisticated citations from three different domains and presenting them in a
        comprehensive, actionable format that maintains the precision achieved through
        the various metadata flattening approaches used by each specialized agent.
        
        The result is something quite remarkable: users get compliance guidance that
        seamlessly integrates European regulation, local law implementation, and
        organizational security procedures - all with precise, verifiable citations
        that can be traced back to specific paragraphs, sub-paragraphs, and implementation steps.
        """
        print("\n📊 [STEP 4/4] ENHANCED MULTI-DOMAIN SUMMARIZATION AGENT: Creating comprehensive action plan...")
        
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        polish_law_citations = state["polish_law_citations"]
        internal_policy_citations = state["internal_policy_citations"]
        
        print(f"📝 Processing citations from enhanced multi-domain agents:")
        print(f"   - GDPR: {len(gdpr_citations)} citations with legal precision")
        print(f"   - Polish Law: {len(polish_law_citations)} citations with metadata flattening")
        print(f"   - Internal Security: {len(internal_policy_citations)} citations with procedural precision")
        
        # Create numbered citations with multi-domain formatting preservation
        # This is where the sophisticated integration happens - we create a unified system
        # that preserves the precision from all three different citation formats
        all_citations, citations_text = self.create_numbered_citations(
            gdpr_citations, polish_law_citations, internal_policy_citations
        )
        
        print(f"📋 Created unified multi-domain numbering system for {len(all_citations)} total citations")
        
        # Create the chain and get action plan using enhanced multi-domain citation context
        # The LLM receives a unified view of all citations while preserving their precision
        chain = self.prompt | self.model
        response = chain.invoke({
            "user_query": user_query,
            "all_citations_with_numbers": citations_text
        })
        
        print("🎯 Generated comprehensive action plan with precise multi-domain citations")
        
        # Format the final response with enhanced precision citations from all domains
        formatted_response = self.format_final_response(response.content, all_citations)
        
        # Create a comprehensive processing summary showing multi-domain capabilities
        processing_summary = self.create_processing_summary(
            all_citations, gdpr_citations, polish_law_citations, internal_policy_citations
        )
        
        # Create the final summary structure with comprehensive statistics
        summary = {
            "action_plan": formatted_response,
            **processing_summary
        }
        
        state["summary"] = summary
        
        print("✅ Completed: Comprehensive multi-domain action plan with precision citations generated")
        print(f"📊 Overall precision rate: {processing_summary['overall_precision_rate']}% of citations include detailed structural references")
        print(f"🔍 Precision breakdown:")
        print(f"   - GDPR enhanced citations: {processing_summary['precision_statistics']['enhanced_gdpr_citations']}")
        print(f"   - Polish Law enhanced citations: {processing_summary['precision_statistics']['enhanced_polish_citations']}")
        print(f"   - Security Procedure enhanced citations: {processing_summary['precision_statistics']['enhanced_security_citations']}")
        
        return state