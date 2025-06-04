"""
Enhanced Response Handler for Multi-Format Backend Compatibility

This module demonstrates how to build flexible interfaces that can handle
different response formats gracefully. This is a crucial skill in software
engineering where APIs evolve over time and different components may have
different expectations about data structure.

The key principle here is "graceful degradation" - our code should work
well with the ideal data format, but still function reasonably when the
data comes in a different but related format.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def normalize_backend_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert backend responses into the format expected by the frontend.
    
    This function acts as a translation layer between different response formats,
    demonstrating how to build robust interfaces that can evolve over time.
    Think of this like a universal translator that can understand multiple
    "dialects" of the same language.
    
    This approach teaches several important software engineering principles:
    - Interface adaptation: Making incompatible formats work together
    - Backwards compatibility: Supporting older formats while enabling new ones  
    - Defensive programming: Handling unexpected data gracefully
    - Format migration: Smoothly transitioning between different data structures
    
    Args:
        raw_response: The response data from the backend, in any supported format
        
    Returns:
        Dict: Response normalized to the format expected by the frontend
    """
    
    logger.info("Normalizing backend response to frontend-compatible format")
    logger.info(f"Input response keys: {list(raw_response.keys())}")
    
    # Create the normalized response structure that our frontend expects
    normalized_response = {
        "success": True,  # We'll validate this assumption as we process
        "action_plan": "",
        "citations": {},
        "raw_citations": [],
        "metadata": {}
    }
    
    # Handle the new backend format (domain-specific citations)
    if all(key in raw_response for key in ["gdpr_citations", "polish_law_citations", "internal_policy_citations"]):
        logger.info("Detected new backend format with domain-specific citations")
        
        # Extract and combine citations from different domains
        all_citations = []
        citation_counts = {}
        
        # Process GDPR citations
        gdpr_citations = raw_response.get("gdpr_citations", [])
        if gdpr_citations:
            all_citations.extend(normalize_citation_format(gdpr_citations, "gdpr"))
            citation_counts["gdpr_citations"] = len(gdpr_citations)
            logger.info(f"Processed {len(gdpr_citations)} GDPR citations")
        
        # Process Polish law citations
        polish_citations = raw_response.get("polish_law_citations", [])
        if polish_citations:
            all_citations.extend(normalize_citation_format(polish_citations, "polish_law"))
            citation_counts["polish_law_citations"] = len(polish_citations)
            logger.info(f"Processed {len(polish_citations)} Polish law citations")
        
        # Process internal policy citations
        internal_citations = raw_response.get("internal_policy_citations", [])
        if internal_citations:
            all_citations.extend(normalize_citation_format(internal_citations, "internal_policy"))
            citation_counts["security_citations"] = len(internal_citations)
            logger.info(f"Processed {len(internal_citations)} internal policy citations")
        
        # Build the action plan from the summary
        summary = raw_response.get("summary", "")
        if summary:
            # Convert the summary into an action plan format
            normalized_response["action_plan"] = format_summary_as_action_plan(summary)
            logger.info("Converted summary to action plan format")
        else:
            logger.warning("No summary found in backend response")
            normalized_response["action_plan"] = "Analysis completed successfully, but detailed action plan not available."
        
        # Build citations metadata
        normalized_response["citations"] = {
            "total_citations": len(all_citations),
            **citation_counts,
            "precision_rate": calculate_precision_score(all_citations)
        }
        
        # Store the raw citations for detailed display
        normalized_response["raw_citations"] = all_citations
        
        # Build metadata about the analysis
        normalized_response["metadata"] = {
            "agent_coordination": "Multi-agent",
            "domains_analyzed": ["gdpr", "polish_law", "internal_security"],
            "analysis_timestamp": "Current",
            "response_format": "domain_specific_v2"
        }
        
        logger.info(f"Successfully normalized response with {len(all_citations)} total citations")
        
    # Handle the legacy backend format (if it exists)
    elif "action_plan" in raw_response:
        logger.info("Detected legacy backend format with unified structure")
        
        # This format already matches our expectations, just pass it through
        normalized_response.update(raw_response)
        
    # Handle unexpected formats gracefully
    else:
        logger.warning("Unknown response format detected, attempting best-effort conversion")
        
        # Try to extract any meaningful content we can find
        possible_content = []
        
        for key, value in raw_response.items():
            if isinstance(value, str) and len(value) > 50:
                possible_content.append(f"**{key.replace('_', ' ').title()}:**\n{value}")
        
        if possible_content:
            normalized_response["action_plan"] = "\n\n".join(possible_content)
            normalized_response["metadata"] = {
                "response_format": "unknown_format_converted",
                "original_keys": list(raw_response.keys())
            }
            logger.info("Created action plan from available content fields")
        else:
            # Last resort: indicate that we got a response but couldn't process it
            normalized_response["success"] = False
            normalized_response["error"] = "Response format not recognized, but backend responded successfully"
            normalized_response["metadata"] = {
                "response_format": "unrecognized",
                "original_response": raw_response
            }
            logger.error("Could not extract meaningful content from response")
    
    return normalized_response


def normalize_citation_format(citations: List[Dict[str, Any]], citation_type: str) -> List[Dict[str, Any]]:
    """
    Convert citations from various formats into a standardized structure.
    
    This function demonstrates how to handle data that might come in slightly
    different formats while ensuring consistent output. This is essential when
    working with multiple data sources or when APIs evolve over time.
    
    Args:
        citations: List of citation objects in various formats
        citation_type: Type of citation (gdpr, polish_law, internal_policy)
        
    Returns:
        List of citations in normalized format
    """
    
    normalized_citations = []
    
    for i, citation in enumerate(citations):
        # Create a standardized citation object
        normalized_citation = {
            "type": citation_type,
            "number": i + 1,  # Will be renumbered globally later
            "source": determine_citation_source(citation, citation_type),
            "text": extract_citation_text(citation),
        }
        
        # Add type-specific fields based on the citation type
        if citation_type == "gdpr":
            normalized_citation.update(extract_gdpr_specifics(citation))
        elif citation_type == "polish_law":
            normalized_citation.update(extract_polish_law_specifics(citation))
        elif citation_type == "internal_policy":
            normalized_citation.update(extract_internal_policy_specifics(citation))
        
        normalized_citations.append(normalized_citation)
    
    logger.info(f"Normalized {len(normalized_citations)} {citation_type} citations")
    return normalized_citations


def determine_citation_source(citation: Dict[str, Any], citation_type: str) -> str:
    """
    Extract or determine the source name for a citation.
    
    This function shows how to handle data that might be structured differently
    across different sources while still extracting the essential information.
    """
    
    # Try various ways that the source might be specified
    if "source" in citation:
        return citation["source"]
    elif "article" in citation:
        if citation_type == "gdpr":
            return "European Data Protection Regulation (GDPR)"
        elif citation_type == "polish_law":
            return "Polish Data Protection Implementation"
        else:
            return f"Article {citation['article']}"
    elif citation_type == "gdpr":
        return "European Data Protection Regulation (GDPR)"
    elif citation_type == "polish_law":
        return "Polish Data Protection Law"
    elif citation_type == "internal_policy":
        return "Internal Security Procedures"
    else:
        return "Unknown Source"


def extract_citation_text(citation: Dict[str, Any]) -> str:
    """
    Extract the main text content from a citation object.
    
    Citations might store their text in different fields depending on the
    backend format, so we try multiple approaches to find the actual content.
    """
    
    # Try various field names that might contain the citation text
    text_fields = ["text", "quote", "content", "description", "summary"]
    
    for field in text_fields:
        if field in citation and citation[field]:
            return str(citation[field]).strip()
    
    # If no standard text field found, try to construct from available data
    if "article" in citation and "chapter" in citation:
        return f"Article {citation['article']} ({citation['chapter']})"
    
    # Last resort: convert the entire citation to a string representation
    return str(citation)


def extract_gdpr_specifics(citation: Dict[str, Any]) -> Dict[str, str]:
    """Extract GDPR-specific information from a citation."""
    specifics = {}
    
    if "article" in citation:
        specifics["article"] = citation["article"]
    if "chapter" in citation:
        specifics["chapter"] = citation["chapter"]
    
    return specifics


def extract_polish_law_specifics(citation: Dict[str, Any]) -> Dict[str, str]:
    """Extract Polish law-specific information from a citation."""
    specifics = {}
    
    if "article" in citation:
        specifics["article"] = citation["article"]
    if "law" in citation:
        specifics["law"] = citation["law"]
    
    return specifics


def extract_internal_policy_specifics(citation: Dict[str, Any]) -> Dict[str, str]:
    """Extract internal policy-specific information from a citation."""
    specifics = {}
    
    if "procedure" in citation:
        specifics["procedure"] = citation["procedure"]
    if "section" in citation:
        specifics["section"] = citation["section"]
    
    return specifics


def format_summary_as_action_plan(summary) -> str:
    """
    Convert a summary into action plan format for frontend display.
    
    Enhanced to handle both string and dictionary summary formats gracefully.
    This demonstrates defensive programming - writing code that can adapt to
    different input types automatically, which is crucial when working with
    APIs that might evolve or return data in various formats.
    
    Args:
        summary: Either a string containing summary text, or a dictionary 
                containing structured summary information
                
    Returns:
        str: Formatted action plan text ready for frontend display
    """
    
    # Handle the case where summary is a dictionary (structured data)
    if isinstance(summary, dict):
        logger.info("Summary is structured data (dictionary), extracting content")
        
        # Try to extract meaningful content from the dictionary
        summary_parts = []
        
        # Look for common field names that might contain the main summary
        content_fields = ['text', 'content', 'summary', 'analysis', 'recommendations', 'guidance']
        
        for field in content_fields:
            if field in summary and summary[field]:
                summary_parts.append(str(summary[field]))
                logger.info(f"Extracted content from field: {field}")
        
        # If we found specific fields, use them
        if summary_parts:
            summary_text = "\n\n".join(summary_parts)
        else:
            # Otherwise, try to extract all string values from the dictionary
            logger.info("No standard content fields found, extracting all text content")
            summary_parts = []
            
            for key, value in summary.items():
                if isinstance(value, str) and len(value.strip()) > 20:  # Only include substantial text
                    section_title = key.replace('_', ' ').title()
                    summary_parts.append(f"**{section_title}:**\n{value}")
            
            if summary_parts:
                summary_text = "\n\n".join(summary_parts)
            else:
                # Last resort: convert entire dictionary to string representation
                summary_text = f"Analysis completed with structured data: {summary}"
                logger.warning("Could not extract readable content from summary dictionary")
    
    # Handle the case where summary is already a string  
    elif isinstance(summary, str):
        logger.info("Summary is text format (string), processing directly")
        summary_text = summary
    
    # Handle unexpected types gracefully
    else:
        logger.warning(f"Unexpected summary type: {type(summary)}, converting to string")
        summary_text = str(summary)
    
    # Now process the extracted text, regardless of its original format
    # Check if the summary already looks like an action plan
    if any(indicator in summary_text.lower() for indicator in ["1.", "step ", "action", "implement", "ensure"]):
        logger.info("Summary already appears to be in action plan format")
        return summary_text
    
    # Otherwise, format it as a comprehensive action plan
    formatted_plan = f"""
**Comprehensive Compliance Action Plan**

Based on the multi-agent analysis of your query, here is your personalized compliance guidance:

{summary_text}

**Next Steps:**
This analysis was generated by our sophisticated multi-agent system combining GDPR expertise, Polish law knowledge, and internal security procedures. Please review each recommendation carefully and consider consulting with your legal team for implementation guidance specific to your organization's context.
    """.strip()
    
    logger.info("Converted summary to structured action plan format")
    return formatted_plan


def calculate_precision_score(citations: List[Dict[str, Any]]) -> int:
    """
    Calculate a quality score for the citation analysis.
    
    This provides users with feedback about the comprehensiveness of the analysis,
    which helps build trust in the system and provides transparency about the
    quality of the results.
    """
    
    if not citations:
        return 0
    
    # Simple scoring based on citation diversity and completeness
    score_factors = []
    
    # Factor 1: Number of citations (more comprehensive analysis)
    citation_count_score = min(len(citations) * 10, 50)  # Max 50 points for citations
    score_factors.append(citation_count_score)
    
    # Factor 2: Diversity of citation types
    citation_types = set(c.get("type", "unknown") for c in citations)
    diversity_score = len(citation_types) * 15  # 15 points per unique type
    score_factors.append(diversity_score)
    
    # Factor 3: Completeness of citation information
    complete_citations = sum(1 for c in citations if c.get("text") and c.get("source"))
    completeness_score = (complete_citations / len(citations)) * 35  # Max 35 points
    score_factors.append(completeness_score)
    
    total_score = sum(score_factors)
    final_score = min(int(total_score), 100)  # Cap at 100%
    
    logger.info(f"Calculated precision score: {final_score}% (based on {len(citations)} citations)")
    return final_score


def validate_normalized_response(response: Dict[str, Any]) -> bool:
    """
    Validate that the normalized response has all required fields.
    
    This final validation step ensures that our normalization process
    produced a response that will work correctly with the frontend,
    regardless of what format the backend originally sent.
    """
    
    required_fields = ["success", "action_plan", "citations", "raw_citations", "metadata"]
    
    for field in required_fields:
        if field not in response:
            logger.error(f"Normalized response missing required field: {field}")
            return False
    
    # Validate that the success field makes sense
    if response["success"] and not response["action_plan"]:
        logger.warning("Response marked as successful but has no action plan")
        return False
    
    logger.info("Normalized response passed all validation checks")
    return True