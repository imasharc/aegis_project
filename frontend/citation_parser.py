"""
Citation Processing Module for Enhanced Multi-Agent Compliance System

This module handles all citation-related functionality, demonstrating a key software
engineering principle: separating complex domain logic into focused modules.

Why separate citation processing?
- Complex regular expressions are isolated and testable
- Citation formatting can be modified without affecting other code
- Different citation styles can be easily added
- Business logic is separated from presentation logic

The functions here implement sophisticated text processing techniques that convert
complex legal citation formats into clean, numbered reference lists suitable
for professional documentation.
"""

import re
from typing import Dict, Any, List, Tuple

# Citation configuration constants
CITATION_CONFIG = {
    "citation_pattern": r'AUTHORITATIVE SOURCE CITATIONS:\s*\n\n(.*?)(?=\n\n[A-Z]|\Z)',
    "source_pattern": r'([^:]+:)\s*\[(\d+)\s+with[^]]*\]\s*(.*?)(?=\n\n[^:]+:|\Z)',
    "individual_pattern": r'\[(\d+)\]\s*([^[]+?)(?=\[|\Z)'
}


def parse_citations_from_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Advanced citation parser that extracts citations from action plan text
    and converts them to a structured format for numbered display.
    
    Enhanced to prevent citation duplication by more aggressively cleaning
    citation references from the main text while preserving the detailed
    citation information for separate display.
    
    Args:
        text: The action plan text containing embedded citations
        
    Returns:
        tuple: (cleaned_text_without_citations, list_of_citation_objects)
    """
    citations = []
    citation_counter = 1
    
    # Find the main citation section using our configured pattern
    citation_pattern = CITATION_CONFIG["citation_pattern"]
    citation_match = re.search(citation_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if citation_match:
        citation_section = citation_match.group(1)
        
        # Extract individual source blocks (e.g., "European Data Protection Regulation (GDPR):")
        source_pattern = CITATION_CONFIG["source_pattern"]
        source_matches = re.findall(source_pattern, citation_section, re.DOTALL)
        
        for source_title, count, content in source_matches:
            # Clean up the source title by removing colons and extra whitespace
            clean_source = source_title.strip().rstrip(':')
            
            # Extract individual citations within this source using numbered references
            individual_pattern = CITATION_CONFIG["individual_pattern"]
            individual_matches = re.findall(individual_pattern, content, re.DOTALL)
            
            for orig_num, citation_text in individual_matches:
                # Clean up the citation text by removing checkmarks and normalizing whitespace
                clean_text = re.sub(r'\s*✓\s*$', '', citation_text.strip())
                clean_text = re.sub(r'\s+', ' ', clean_text)
                
                # Create structured citation object
                citations.append({
                    'number': citation_counter,
                    'source': clean_source,
                    'text': clean_text,
                    'original_number': orig_num
                })
                citation_counter += 1
    
    # More aggressive cleaning to prevent duplication
    # Remove the entire citation section
    cleaned_text = re.sub(citation_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining citation references in brackets
    cleaned_text = re.sub(r'\[(\d+)\s+with[^]]*\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[(\d+)\]', '', cleaned_text)  # Remove simple [1] references
    
    # Remove "AUTHORITATIVE SOURCE CITATIONS" headers that might remain
    cleaned_text = re.sub(r'\*\*AUTHORITATIVE SOURCE CITATIONS:\*\*.*', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove system insights that appear in raw text
    cleaned_text = re.sub(r'\*\*SYSTEM INSIGHTS:\*\*.*', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up multiple spaces and normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Normalize paragraph breaks
    
    return cleaned_text.strip(), citations


def format_citations_as_numbered_list(citations: List[Dict[str, str]]) -> str:
    """
    Convert structured citation data into a clean numbered list format.
    
    This function takes the parsed citation data and creates a professional,
    easy-to-read numbered list that maintains all the important legal and
    regulatory reference information while improving readability.
    
    The formatting strategy:
    1. Group citations by source for organized presentation
    2. Create HTML structure with proper styling
    3. Maintain legal document formatting standards
    4. Ensure accessibility and readability
    
    Args:
        citations: List of citation dictionaries with source, text, etc.
        
    Returns:
        str: Formatted HTML string with numbered citations
    """
    if not citations:
        return ""
    
    # Group citations by source for organized presentation
    # This makes it easier for users to understand which regulations apply
    grouped_citations = {}
    for citation in citations:
        source = citation['source']
        if source not in grouped_citations:
            grouped_citations[source] = []
        grouped_citations[source].append(citation)
    
    # Build the formatted citation list with professional styling
    formatted_html = "<div style='font-size: 16px; line-height: 1.6; margin-top: 20px;'>"
    formatted_html += "<h4>📋 AUTHORITATIVE SOURCE CITATIONS:</h4>\n\n"
    
    for source, source_citations in grouped_citations.items():
        # Display source header with citation count for transparency
        formatted_html += f"<p><strong>{source}:</strong> [{len(source_citations)} citations]</p>\n"
        
        # Format each individual citation with proper indentation
        for citation in source_citations:
            formatted_html += f"<p style='margin-left: 20px; margin-bottom: 10px;'>"
            formatted_html += f"<strong>[{citation['number']}]</strong> {citation['text']}"
            formatted_html += "</p>\n"
        
        formatted_html += "\n"  # Add spacing between source groups
    
    formatted_html += "</div>"
    
    return formatted_html


def format_raw_citations(raw_citations: List[Dict[str, Any]]) -> str:
    """
    Format raw citation data from the backend into numbered lists.
    
    This function handles the structured citation data returned by the
    enhanced backend API, converting it into the same clean numbered
    format for consistent presentation across different data sources.
    
    The key difference from parse_citations_from_text is that this function
    works with already-structured data rather than parsing text, demonstrating
    how to handle different input formats while maintaining consistent output.
    
    Args:
        raw_citations: List of raw citation objects from backend
        
    Returns:
        str: Formatted HTML string with numbered citations
    """
    if not raw_citations:
        return ""
    
    # Group by source type for logical organization
    grouped = {}
    for i, citation in enumerate(raw_citations, 1):
        source = citation.get('source', 'Unknown Source')
        if source not in grouped:
            grouped[source] = []
        
        # Add citation number for sequential referencing
        citation_copy = citation.copy()
        citation_copy['number'] = i
        grouped[source].append(citation_copy)
    
    # Build formatted HTML with consistent styling
    formatted_html = "<div style='font-size: 16px; line-height: 1.6; margin-top: 20px;'>"
    formatted_html += "<h4>📋 AUTHORITATIVE SOURCE CITATIONS:</h4>\n\n"
    
    for source, citations in grouped.items():
        formatted_html += f"<p><strong>{source}:</strong> [{len(citations)} citations]</p>\n"
        
        for citation in citations:
            formatted_html += f"<p style='margin-left: 20px; margin-bottom: 10px;'>"
            formatted_html += f"<strong>[{citation['number']}]</strong> "
            
            # Format based on citation type for better clarity
            # This demonstrates how to handle different structured data formats
            if citation.get('type') == 'gdpr':
                if citation.get('article'):
                    formatted_html += f"Article {citation['article']} "
                if citation.get('chapter'):
                    formatted_html += f"({citation['chapter']}): "
            elif citation.get('type') == 'polish_law':
                if citation.get('article'):
                    formatted_html += f"Article {citation['article']} "
                if citation.get('law'):
                    formatted_html += f"({citation['law']}): "
            elif citation.get('type') == 'internal_policy':
                if citation.get('procedure'):
                    formatted_html += f"Procedure {citation['procedure']} "
                if citation.get('section'):
                    formatted_html += f"(Section {citation['section']}): "
            
            formatted_html += citation.get('text', '')
            formatted_html += "</p>\n"
        
        formatted_html += "\n"
    
    formatted_html += "</div>"
    
    return formatted_html


def clean_action_plan_text(text: str) -> str:
    """
    Remove citation sections from action plan text for clean display.
    
    This utility function separates content from citations, allowing us to
    display the main action plan cleanly while handling citations separately.
    This is a common pattern in document processing where you want to separate
    different types of content.
    
    Args:
        text: Original action plan text with embedded citations
        
    Returns:
        str: Clean action plan text without citation sections
    """
    # Remove the citation section using our configured pattern
    citation_pattern = CITATION_CONFIG["citation_pattern"]
    clean_text = re.sub(citation_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any remaining citation references
    clean_text = re.sub(r'\[(\d+)\s+with[^]]*\]', '', clean_text)
    
    # Normalize whitespace for consistent presentation
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()


def format_action_plan_as_ordered_list(text: str) -> str:
    """
    Convert numbered action plan text into proper HTML ordered lists.
    
    Enhanced to handle various formatting patterns and edge cases that can occur
    in AI-generated content. This function now better recognizes different
    numbering styles and formats them consistently.
    
    Args:
        text: Action plan text with numbered items
        
    Returns:
        str: HTML formatted ordered list
    """
    
    # First, clean up common formatting issues
    text = re.sub(r'={3,}', '', text)  # Remove separator lines
    text = re.sub(r'\*\*MULTI-DOMAIN.*?\*\*', '', text, flags=re.DOTALL | re.IGNORECASE)  # Remove metadata headers
    text = re.sub(r'Here is a clear.*?:', '', text, flags=re.IGNORECASE)  # Remove intro text
    
    # Enhanced pattern to match various numbered formats:
    # "1. **Title**" or "1. Title" or "**1. Title**" 
    patterns = [
        # Pattern 1: "1. **Bold Title** Content"
        r'(\d+)\.\s*\*\*(.*?)\*\*(.*?)(?=\d+\.\s*\*\*|\d+\.\s*[A-Z]|\Z)',
        # Pattern 2: "**1. Bold Title** Content" 
        r'\*\*(\d+)\.\s*(.*?)\*\*(.*?)(?=\*\*\d+\.|\d+\.\s*[A-Z]|\Z)',
        # Pattern 3: "1. Title without bold" 
        r'(\d+)\.\s*([^*\n]+?)(?=\d+\.\s*|\Z)'
    ]
    
    matches = []
    
    # Try each pattern until we find matches
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            break
    
    # If no structured patterns found, try to split by number markers
    if not matches:
        # Split by number patterns and try to structure
        lines = text.split('\n')
        current_item = None
        items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number
            number_match = re.match(r'^(\d+)\.\s*(.*)', line)
            if number_match:
                # Save previous item if exists
                if current_item:
                    items.append(current_item)
                
                # Start new item
                number, content = number_match.groups()
                current_item = {
                    'number': number,
                    'title': '',
                    'content': content
                }
            elif current_item:
                # Continue previous item
                current_item['content'] += ' ' + line
        
        # Add the last item
        if current_item:
            items.append(current_item)
        
        # Convert to HTML if we found items
        if items:
            html_content = '<ol style="font-size: 18px; line-height: 1.8; margin: 20px 0;">\n'
            
            for item in items:
                content = item['content'].strip()
                
                # Try to separate title from content (look for sentences)
                sentences = content.split('. ')
                if len(sentences) > 1:
                    title = sentences[0]
                    rest_content = '. '.join(sentences[1:])
                    
                    html_content += f'<li style="margin-bottom: 20px;">\n'
                    html_content += f'<strong>{title}</strong>\n'
                    if rest_content:
                        html_content += f'<div style="margin-top: 8px;">{rest_content}</div>\n'
                    html_content += '</li>\n'
                else:
                    html_content += f'<li style="margin-bottom: 20px;">{content}</li>\n'
            
            html_content += '</ol>'
            return html_content
        else:
            # No structure found, return formatted text
            return f'<div style="font-size: 18px; line-height: 1.6; margin: 20px 0;">{text}</div>'
    
    # Process structured matches
    html_content = '<ol style="font-size: 18px; line-height: 1.8; margin: 20px 0;">\n'
    
    for match in matches:
        if len(match) == 3:  # Pattern 1 or 2
            number, title, content = match
        else:  # Pattern 3 (no bold formatting)
            number, combined = match
            title = combined
            content = ""
        
        # Clean up the content
        clean_title = re.sub(r'\s+', ' ', title.strip())
        clean_content = re.sub(r'\s+', ' ', content.strip()) if content else ""
        
        # Remove any remaining asterisks or formatting artifacts
        clean_title = re.sub(r'\*+', '', clean_title)
        clean_content = re.sub(r'\*+', '', clean_content)
        
        # Create list item with proper styling
        html_content += f'<li style="margin-bottom: 20px;">\n'
        
        if clean_title:
            html_content += f'<strong style="color: #2c3e50;">{clean_title}</strong>\n'
        
        if clean_content:
            html_content += f'<div style="margin-top: 8px; color: #34495e;">{clean_content}</div>\n'
        
        html_content += '</li>\n'
    
    html_content += '</ol>'
    
    return html_content