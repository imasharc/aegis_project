"""
Summarization Citation Manager

This module handles the sophisticated challenge of unifying citations from three completely
different domains (GDPR legal articles, Polish law provisions, and security procedures) while
preserving the precision that each specialized agent worked hard to achieve.

Think of this as a "universal translator" for citation formats. Each domain has its own
"language" for precise references:
- GDPR: "Article 6, paragraph 1(a) (Chapter II: Principles)"
- Polish Law: "Article 3, paragraph 1(2) (Chapter 2: Obligations)"  
- Security: "Procedure 3.1: User Management, Step 2 - Configure Access Controls"

The citation manager creates a unified numbering system that allows all three types to
work together in a single action plan while maintaining the structural precision that
makes your system stand out from basic document retrieval approaches.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple


class SummarizationCitationManager:
    """
    Manages the unification of citations from multiple specialized domains.
    
    This class solves one of the most challenging integration problems in your system:
    how do you take precise citations from three completely different domains and make
    them work together seamlessly? Each domain has its own citation standards and
    precision indicators, but users need to see them as a unified, coherent set.
    
    The manager demonstrates advanced system integration - it creates a "translation layer"
    that understands each citation format while enabling them to work together in
    unified action plans with consistent numbering and presentation.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summarization citation manager.
        
        Args:
            logger: Configured logger for tracking citation management operations
        """
        self.logger = logger
        self.logger.info("Summarization Citation Manager initialized")
        
        # Track citation management statistics across all domains
        self.citation_stats = {
            'total_citation_sets_processed': 0,
            'total_citations_unified': 0,
            'gdpr_citations_processed': 0,
            'polish_law_citations_processed': 0,
            'security_citations_processed': 0,
            'cross_domain_precision_preserved': 0,
            'unified_numbering_success_rate': 0.0,
            'domain_distribution': {'gdpr': 0, 'polish_law': 0, 'security': 0},
            'precision_preservation_rate': 0.0
        }
    
    def create_unified_citation_system(self, gdpr_citations: List[Dict[str, Any]], 
                                     polish_law_citations: List[Dict[str, Any]], 
                                     internal_policy_citations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Create a unified citation numbering system that preserves precision across all domains.
        
        This method represents the sophisticated integration challenge at the heart of your
        system. You have three agents that each create precise citations using different
        metadata flattening approaches and structural analysis. Now you need to present
        them as a unified system without losing any of the precision.
        
        The method demonstrates how architectural sophistication enables seamless integration
        while preserving the specialized work done by each domain expert.
        
        Args:
            gdpr_citations: Citations from GDPR agent with legal precision
            polish_law_citations: Citations from Polish Law agent with enhanced metadata
            internal_policy_citations: Citations from Security agent with procedural precision
            
        Returns:
            Tuple of (unified_citations_list, formatted_citations_text)
        """
        self.logger.info("Creating unified citation system from multi-domain inputs")
        self.citation_stats['total_citation_sets_processed'] += 1
        
        # Update domain statistics
        self.citation_stats['gdpr_citations_processed'] += len(gdpr_citations)
        self.citation_stats['polish_law_citations_processed'] += len(polish_law_citations)
        self.citation_stats['security_citations_processed'] += len(internal_policy_citations)
        
        # Update domain distribution
        self.citation_stats['domain_distribution']['gdpr'] += len(gdpr_citations)
        self.citation_stats['domain_distribution']['polish_law'] += len(polish_law_citations)
        self.citation_stats['domain_distribution']['security'] += len(internal_policy_citations)
        
        try:
            # Create the unified citation list with domain-aware processing
            unified_citations = self._process_multi_domain_citations(
                gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Generate formatted text for LLM consumption
            formatted_text = self._create_llm_formatted_citations(unified_citations)
            
            # Update success statistics
            self.citation_stats['total_citations_unified'] += len(unified_citations)
            self._calculate_and_update_success_rates(unified_citations)
            
            self.logger.info(f"Successfully unified {len(unified_citations)} citations from {self._count_active_domains(gdpr_citations, polish_law_citations, internal_policy_citations)} domains")
            
            return unified_citations, formatted_text
            
        except Exception as e:
            self.logger.error(f"Error creating unified citation system: {e}")
            # Return fallback system to maintain workflow
            return self._create_fallback_citation_system(gdpr_citations, polish_law_citations, internal_policy_citations)
    
    def _process_multi_domain_citations(self, gdpr_citations: List[Dict[str, Any]], 
                                       polish_law_citations: List[Dict[str, Any]], 
                                       internal_policy_citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process citations from all domains while preserving their individual precision characteristics.
        
        This method demonstrates the "universal translator" concept - it understands the
        precision indicators and structural information from each domain and preserves
        them in a unified format that works across domain boundaries.
        """
        unified_citations = []
        citation_number = 1
        
        # Process GDPR citations with legal domain awareness
        for citation in gdpr_citations:
            unified_citation = self._create_unified_legal_citation(citation, citation_number, 'GDPR')
            unified_citations.append(unified_citation)
            citation_number += 1
            
            self.logger.debug(f"Processed GDPR citation {citation_number - 1}: {citation.get('article', 'Unknown')}")
        
        # Process Polish law citations with enhanced legal domain awareness
        for citation in polish_law_citations:
            unified_citation = self._create_unified_legal_citation(citation, citation_number, 'Polish Law')
            unified_citations.append(unified_citation)
            citation_number += 1
            
            self.logger.debug(f"Processed Polish Law citation {citation_number - 1}: {citation.get('article', 'Unknown')}")
        
        # Process security procedure citations with procedural domain awareness
        for citation in internal_policy_citations:
            unified_citation = self._create_unified_procedural_citation(citation, citation_number)
            unified_citations.append(unified_citation)
            citation_number += 1
            
            self.logger.debug(f"Processed Security citation {citation_number - 1}: {citation.get('procedure', citation.get('article', 'Unknown'))}")
        
        self.logger.info(f"Multi-domain processing complete: {len(unified_citations)} citations unified with preserved precision")
        
        return unified_citations
    
    def _create_unified_legal_citation(self, citation: Dict[str, Any], number: int, domain: str) -> Dict[str, Any]:
        """
        Create a unified citation from legal domain sources (GDPR or Polish Law).
        
        This method handles the translation from legal citation formats to the unified
        system while preserving all the precision indicators that the legal agents
        worked to create through their metadata flattening approaches.
        """
        # Extract the legal reference (both domains use 'article' key)
        legal_reference = citation.get("article", "")
        quote = citation.get("quote", "")
        explanation = citation.get("explanation", "")
        
        # Determine precision level based on legal domain indicators
        precision_level = self._analyze_legal_citation_precision(legal_reference, domain)
        
        # Create the unified citation structure
        unified_citation = {
            "number": number,
            "source_type": domain,
            "source": "GDPR" if domain == "GDPR" else "Polish Data Protection Act",
            "reference": legal_reference,
            "quote": quote,
            "explanation": explanation,
            "domain": "legal",
            "precision_level": precision_level,
            "precision_preserved": len(legal_reference) > 10,  # Basic precision indicator
        }
        
        # Track precision preservation
        if unified_citation["precision_preserved"]:
            self.citation_stats['cross_domain_precision_preserved'] += 1
        
        return unified_citation
    
    def _create_unified_procedural_citation(self, citation: Dict[str, Any], number: int) -> Dict[str, Any]:
        """
        Create a unified citation from procedural domain sources (Security).
        
        This method handles the unique challenge of translating procedural citations
        (which use different keys and precision indicators) into the unified system
        while preserving the implementation step precision that makes security citations valuable.
        """
        # Security citations may use 'procedure', 'article', or 'section' keys
        procedural_reference = citation.get("procedure", citation.get("article", citation.get("section", "")))
        quote = citation.get("quote", "")
        explanation = citation.get("explanation", "")
        
        # Determine precision level based on procedural domain indicators
        precision_level = self._analyze_procedural_citation_precision(procedural_reference)
        
        # Create the unified citation structure
        unified_citation = {
            "number": number,
            "source_type": "Internal Security",
            "source": "Internal Security Procedures", 
            "reference": procedural_reference,
            "quote": quote,
            "explanation": explanation,
            "domain": "procedural",
            "precision_level": precision_level,
            "precision_preserved": self._has_procedural_precision(procedural_reference),
        }
        
        # Track precision preservation
        if unified_citation["precision_preserved"]:
            self.citation_stats['cross_domain_precision_preserved'] += 1
        
        return unified_citation
    
    def _analyze_legal_citation_precision(self, reference: str, domain: str) -> str:
        """
        Analyze the precision level of legal citations across GDPR and Polish Law domains.
        
        This method understands the precision indicators that each legal agent uses
        and translates them into consistent precision levels for unified reporting.
        """
        if not reference:
            return "minimal"
        
        reference_lower = reference.lower()
        
        # Check for maximum legal precision indicators
        if any(indicator in reference_lower for indicator in [
            "paragraph", "sub-paragraph", "(a)", "(b)", "(c)", "(1)", "(2)", "chapter"
        ]):
            return "maximum"
        
        # Check for medium precision (article with basic structure)
        if "article" in reference_lower and len(reference) > 15:
            return "medium"
        
        # Basic precision (article number only)
        if "article" in reference_lower:
            return "basic"
        
        return "minimal"
    
    def _analyze_procedural_citation_precision(self, reference: str) -> str:
        """
        Analyze the precision level of procedural citations from the security domain.
        
        This method understands the unique precision indicators used in security
        procedure citations and maps them to the unified precision scale.
        """
        if not reference:
            return "minimal"
        
        reference_lower = reference.lower()
        
        # Check for maximum procedural precision (procedure + step + sub-step)
        if any(indicator in reference_lower for indicator in [
            "step", "configuration", "implementation", "phase"
        ]) and "procedure" in reference_lower:
            return "maximum"
        
        # Check for medium precision (procedure with title or section)
        if "procedure" in reference_lower and ":" in reference:
            return "medium"
        
        # Basic precision (procedure number only)
        if "procedure" in reference_lower:
            return "basic"
        
        return "minimal"
    
    def _has_procedural_precision(self, reference: str) -> bool:
        """
        Determine if a procedural reference contains implementation-level precision.
        
        This method identifies when security procedure citations include the kind of
        implementation step details that make them actionable for security workflows.
        """
        if not reference:
            return False
        
        reference_lower = reference.lower()
        
        # Look for procedural precision indicators
        precision_indicators = [
            "step", "configure", "implementation", "phase", "procedure", 
            "process", "workflow", "requirement"
        ]
        
        return any(indicator in reference_lower for indicator in precision_indicators)
    
    def _create_llm_formatted_citations(self, unified_citations: List[Dict[str, Any]]) -> str:
        """
        Create formatted citation text optimized for LLM consumption.
        
        This method produces the text that will be included in the LLM prompt,
        ensuring that the LLM can reference the unified citations effectively
        while maintaining the precision information from all domains.
        """
        if not unified_citations:
            return "No citations available"
        
        formatted_lines = []
        
        for citation in unified_citations:
            number = citation["number"]
            source = citation["source"]
            reference = citation["reference"]
            explanation = citation["explanation"]
            
            # Create a consistent format that works across all domains
            line = f"[{number}] {source} {reference}: {explanation}"
            formatted_lines.append(line)
        
        formatted_text = "\n".join(formatted_lines)
        
        self.logger.debug(f"Created LLM-formatted citations: {len(formatted_lines)} lines, {len(formatted_text)} characters")
        
        return formatted_text
    
    def _calculate_and_update_success_rates(self, unified_citations: List[Dict[str, Any]]) -> None:
        """
        Calculate and update success rates based on the unified citation results.
        
        This method tracks how well the unification process preserves precision
        and maintains quality across domain boundaries.
        """
        if not unified_citations:
            return
        
        # Calculate unified numbering success rate
        correctly_numbered = sum(1 for i, cite in enumerate(unified_citations) 
                               if cite["number"] == i + 1)
        numbering_success_rate = (correctly_numbered / len(unified_citations)) * 100
        
        # Update rolling average
        old_rate = self.citation_stats['unified_numbering_success_rate']
        sets_processed = self.citation_stats['total_citation_sets_processed']
        new_rate = ((old_rate * (sets_processed - 1)) + numbering_success_rate) / sets_processed
        self.citation_stats['unified_numbering_success_rate'] = new_rate
        
        # Calculate precision preservation rate
        preserved_precision = sum(1 for cite in unified_citations if cite.get("precision_preserved", False))
        preservation_rate = (preserved_precision / len(unified_citations)) * 100
        
        # Update rolling average for precision preservation
        old_precision_rate = self.citation_stats['precision_preservation_rate']
        new_precision_rate = ((old_precision_rate * (sets_processed - 1)) + preservation_rate) / sets_processed
        self.citation_stats['precision_preservation_rate'] = new_precision_rate
        
        self.logger.debug(f"Updated success rates: numbering {new_rate:.1f}%, precision preservation {new_precision_rate:.1f}%")
    
    def _count_active_domains(self, gdpr_citations: List, polish_law_citations: List, 
                            internal_policy_citations: List) -> int:
        """
        Count how many domains contributed citations to this unification.
        
        This helps track the complexity of the unification task and provides
        insights into which combinations of domains are most common.
        """
        active_domains = 0
        
        if gdpr_citations:
            active_domains += 1
        if polish_law_citations:
            active_domains += 1
        if internal_policy_citations:
            active_domains += 1
        
        return active_domains
    
    def _create_fallback_citation_system(self, gdpr_citations: List[Dict[str, Any]], 
                                       polish_law_citations: List[Dict[str, Any]], 
                                       internal_policy_citations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Create a fallback citation system when unified processing fails.
        
        This ensures the system continues to function even when the sophisticated
        unification process encounters issues, demonstrating graceful degradation.
        """
        self.logger.warning("Creating fallback citation system due to unification error")
        
        fallback_citations = []
        citation_number = 1
        
        # Add basic citations from each domain that has data
        for domain_name, citations in [
            ("GDPR", gdpr_citations),
            ("Polish Law", polish_law_citations), 
            ("Internal Security", internal_policy_citations)
        ]:
            for citation in citations:
                fallback_citation = {
                    "number": citation_number,
                    "source_type": domain_name,
                    "source": f"{domain_name} (Fallback)",
                    "reference": str(citation.get("article", citation.get("procedure", "Unknown"))),
                    "quote": citation.get("quote", "Citation available"),
                    "explanation": citation.get("explanation", "Relevant content found"),
                    "domain": "mixed",
                    "precision_level": "fallback",
                    "precision_preserved": False
                }
                fallback_citations.append(fallback_citation)
                citation_number += 1
        
        # Create simple formatted text
        formatted_text = "\n".join([
            f"[{cite['number']}] {cite['source']} {cite['reference']}: {cite['explanation']}"
            for cite in fallback_citations
        ])
        
        return fallback_citations, formatted_text
    
    def get_citation_management_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about citation management operations.
        
        Returns:
            Dictionary containing detailed citation management metrics and performance data
        """
        stats = dict(self.citation_stats)
        
        # Calculate additional derived statistics
        if stats['total_citations_unified'] > 0:
            # Domain distribution percentages
            total_domain_citations = sum(stats['domain_distribution'].values())
            if total_domain_citations > 0:
                for domain in stats['domain_distribution']:
                    count = stats['domain_distribution'][domain]
                    percentage = (count / total_domain_citations) * 100
                    stats[f'{domain}_percentage'] = round(percentage, 1)
            
            # Average citations per domain per set
            if stats['total_citation_sets_processed'] > 0:
                avg_per_set = stats['total_citations_unified'] / stats['total_citation_sets_processed']
                stats['average_citations_per_set'] = round(avg_per_set, 1)
        
        return stats
    
    def log_citation_management_summary(self) -> None:
        """
        Log a comprehensive summary of all citation management operations.
        
        This provides insights into how well the multi-domain unification process
        is working and helps identify patterns in citation complexity and domain usage.
        """
        stats = self.get_citation_management_statistics()
        
        self.logger.info("=== MULTI-DOMAIN CITATION MANAGEMENT SUMMARY ===")
        self.logger.info(f"Citation sets processed: {stats['total_citation_sets_processed']}")
        self.logger.info(f"Total citations unified: {stats['total_citations_unified']}")
        self.logger.info(f"Average citations per set: {stats.get('average_citations_per_set', 0)}")
        self.logger.info(f"Unified numbering success rate: {stats['unified_numbering_success_rate']:.1f}%")
        self.logger.info(f"Precision preservation rate: {stats['precision_preservation_rate']:.1f}%")
        
        # Log domain distribution
        self.logger.info("Domain distribution:")
        for domain, count in stats['domain_distribution'].items():
            percentage = stats.get(f'{domain}_percentage', 0)
            self.logger.info(f"  - {domain}: {count} citations ({percentage}%)")
        
        # Provide interpretation of performance
        precision_rate = stats['precision_preservation_rate']
        if precision_rate >= 90:
            self.logger.info("Excellent precision preservation - multi-domain unification working optimally")
        elif precision_rate >= 75:
            self.logger.info("Good precision preservation - most domain precision maintained")
        elif precision_rate >= 50:
            self.logger.info("Moderate precision preservation - review domain-specific processing")
        else:
            self.logger.info("Low precision preservation - investigate unification pipeline issues")


def create_summarization_citation_manager(logger: logging.Logger) -> SummarizationCitationManager:
    """
    Factory function to create a configured summarization citation manager.
    
    This provides a clean interface for creating manager instances with
    proper dependency injection of the logger.
    """
    return SummarizationCitationManager(logger)