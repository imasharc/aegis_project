"""
Summarization Formatter

This module handles the sophisticated challenge of presenting multi-domain citations and
comprehensive action plans in a way that showcases the precision of your system while
remaining accessible and actionable for users. Think of this as the "presentation layer"
that transforms technical sophistication into user value.

The formatter demonstrates how architectural excellence can be made visible to users.
All the complex work done by metadata flattening, structural analysis, and cross-domain
integration needs to be presented in a way that makes users understand they're receiving
something special compared to basic document retrieval systems.

The challenge is balancing sophistication with usability - showing enough detail to
demonstrate the system's capabilities while organizing it in a way that guides users
toward actionable compliance and implementation steps.
"""

import logging
from typing import Dict, List, Any, Optional


class SummarizationFormatter:
    """
    Handles sophisticated formatting and presentation of multi-domain citation systems.
    
    This class solves the presentation challenge that sophisticated systems often face:
    how do you show users the value of complex processing without overwhelming them?
    Your system does remarkable things with metadata flattening, structural analysis,
    and cross-domain integration - but users need to see that value clearly.
    
    The formatter demonstrates how good design can make sophisticated technology
    accessible. It creates presentations that highlight precision and comprehensiveness
    while guiding users toward actionable outcomes that justify the architectural
    sophistication behind the scenes.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summarization formatter.
        
        Args:
            logger: Configured logger for tracking formatting operations
        """
        self.logger = logger
        self.logger.info("Summarization Formatter initialized")
        
        # Track formatting statistics to understand usage patterns
        self.formatting_stats = {
            'total_responses_formatted': 0,
            'multi_domain_responses': 0,
            'single_domain_responses': 0,
            'average_citations_per_response': 0.0,
            'domain_combinations_formatted': {},
            'formatting_errors': 0,
            'precision_showcasing_instances': 0
        }
    
    def format_comprehensive_response(self, action_plan: str, unified_citations: List[Dict[str, Any]], 
                                    precision_analysis: Dict[str, Any]) -> str:
        """
        Format a comprehensive response that showcases the sophistication of the multi-agent system.
        
        This method creates the final user-facing output that demonstrates the value of your
        architectural approach. Users should immediately understand that they're receiving
        something far more sophisticated than basic document search results.
        
        The method balances multiple objectives:
        - Showcase the precision achieved through metadata flattening
        - Demonstrate cross-domain integration capabilities  
        - Provide actionable guidance that justifies the complexity
        - Present information in a professional, scannable format
        
        Args:
            action_plan: The LLM-generated action plan with numbered citations
            unified_citations: The unified citation system from all domains
            precision_analysis: Analysis results showing system precision metrics
            
        Returns:
            Professionally formatted response showcasing system sophistication
        """
        self.logger.info("Formatting comprehensive multi-domain response with precision showcasing")
        self.formatting_stats['total_responses_formatted'] += 1
        
        try:
            # Analyze the response to understand its complexity and domain coverage
            response_characteristics = self._analyze_response_characteristics(unified_citations, precision_analysis)
            
            # Build the formatted response with sophisticated presentation
            formatted_response = self._build_sophisticated_response_structure(
                action_plan, unified_citations, precision_analysis, response_characteristics
            )
            
            # Update formatting statistics based on the response
            self._update_formatting_statistics(response_characteristics, unified_citations)
            
            self.logger.info(f"Comprehensive response formatted: {response_characteristics['domain_count']} domains, "
                           f"{response_characteristics['citation_count']} citations, "
                           f"{response_characteristics['precision_level']} precision")
            
            return formatted_response
            
        except Exception as e:
            self.formatting_stats['formatting_errors'] += 1
            self.logger.error(f"Error formatting comprehensive response: {e}")
            # Return fallback formatting to maintain workflow
            return self._create_fallback_response_format(action_plan, unified_citations)
    
    def _analyze_response_characteristics(self, unified_citations: List[Dict[str, Any]], 
                                        precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the response to understand how to best showcase its sophistication.
        
        This method examines what the system accomplished to determine the most
        effective way to present the results. Different combinations of domain
        coverage and precision levels call for different presentation strategies.
        """
        # Count domains and analyze coverage
        domain_coverage = self._analyze_domain_coverage(unified_citations)
        
        # Assess overall precision level achieved
        precision_assessment = self._assess_overall_precision_level(unified_citations, precision_analysis)
        
        # Identify sophistication highlights to emphasize
        sophistication_highlights = self._identify_sophistication_highlights(unified_citations, precision_analysis)
        
        # Determine optimal presentation strategy
        presentation_strategy = self._determine_presentation_strategy(domain_coverage, precision_assessment)
        
        characteristics = {
            'domain_count': len(domain_coverage['active_domains']),
            'active_domains': domain_coverage['active_domains'],
            'citation_count': len(unified_citations),
            'precision_level': precision_assessment['overall_level'],
            'precision_score': precision_assessment['overall_score'],
            'sophistication_highlights': sophistication_highlights,
            'presentation_strategy': presentation_strategy,
            'domain_balance': domain_coverage['balance_assessment']
        }
        
        self.logger.debug(f"Response characteristics: {characteristics['domain_count']} domains, "
                        f"{characteristics['precision_level']} precision, "
                        f"strategy: {characteristics['presentation_strategy']}")
        
        return characteristics
    
    def _analyze_domain_coverage(self, unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze which domains contributed to the response and how balanced the coverage is.
        
        This helps determine whether to emphasize multi-domain integration or
        focus on the depth achieved within specific domains.
        """
        domain_counts = {}
        active_domains = set()
        
        for citation in unified_citations:
            domain = citation.get('source_type', 'Unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            active_domains.add(domain)
        
        # Assess balance of domain representation
        if len(domain_counts) > 1:
            total_citations = sum(domain_counts.values())
            max_domain_percentage = max(domain_counts.values()) / total_citations
            
            if max_domain_percentage < 0.6:
                balance_assessment = 'well_balanced'
            elif max_domain_percentage < 0.8:
                balance_assessment = 'moderately_balanced'
            else:
                balance_assessment = 'domain_dominant'
        else:
            balance_assessment = 'single_domain'
        
        return {
            'domain_counts': domain_counts,
            'active_domains': list(active_domains),
            'balance_assessment': balance_assessment
        }
    
    def _assess_overall_precision_level(self, unified_citations: List[Dict[str, Any]], 
                                      precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the overall precision level achieved to determine presentation emphasis.
        
        High precision responses should emphasize the detailed structural analysis,
        while lower precision responses should focus on comprehensive coverage.
        """
        # Get system metrics from precision analysis
        system_metrics = precision_analysis.get('system_metrics', {})
        overall_score = system_metrics.get('overall_precision_score', 0)
        
        # Analyze precision levels in unified citations
        precision_distribution = {}
        for citation in unified_citations:
            level = citation.get('precision_level', 'minimal')
            precision_distribution[level] = precision_distribution.get(level, 0) + 1
        
        # Determine overall precision category
        if overall_score >= 85:
            overall_level = 'exceptional'
        elif overall_score >= 75:
            overall_level = 'high'
        elif overall_score >= 65:
            overall_level = 'good'
        elif overall_score >= 55:
            overall_level = 'moderate'
        else:
            overall_level = 'basic'
        
        return {
            'overall_score': overall_score,
            'overall_level': overall_level,
            'precision_distribution': precision_distribution
        }
    
    def _identify_sophistication_highlights(self, unified_citations: List[Dict[str, Any]], 
                                          precision_analysis: Dict[str, Any]) -> List[str]:
        """
        Identify specific aspects of sophistication that should be highlighted to users.
        
        This method looks for concrete evidence of the advanced processing that
        users should understand they're receiving compared to basic systems.
        """
        highlights = []
        
        # Check for enhanced citation precision
        enhanced_citations = sum(1 for cite in unified_citations 
                               if cite.get('precision_level') in ['maximum', 'high'])
        
        if enhanced_citations > 0:
            highlights.append(f"Enhanced structural analysis on {enhanced_citations} citations")
        
        # Check for multi-domain integration
        domains = set(cite.get('source_type', 'Unknown') for cite in unified_citations)
        if len(domains) > 1:
            highlights.append(f"Integrated guidance from {len(domains)} specialized domains")
        
        # Check for procedural implementation detail
        procedural_detail = sum(1 for cite in unified_citations 
                              if cite.get('domain') == 'procedural' and 
                                 cite.get('precision_level') in ['maximum', 'high'])
        
        if procedural_detail > 0:
            highlights.append(f"Implementation-level detail on {procedural_detail} security procedures")
        
        # Check for legal precision indicators
        legal_precision = sum(1 for cite in unified_citations 
                            if cite.get('domain') == 'legal' and 
                               cite.get('precision_level') in ['maximum', 'high'])
        
        if legal_precision > 0:
            highlights.append(f"Paragraph-level legal precision on {legal_precision} provisions")
        
        # Check integration quality
        integration_analysis = precision_analysis.get('integration_analysis', {})
        integration_score = integration_analysis.get('integration_score', 0)
        
        if integration_score >= 85:
            highlights.append("Seamless cross-domain integration with precision preservation")
        
        # Check for comprehensive coverage
        total_citations = len(unified_citations)
        if total_citations >= 6:
            highlights.append(f"Comprehensive analysis across {total_citations} authoritative sources")
        
        return highlights
    
    def _determine_presentation_strategy(self, domain_coverage: Dict[str, Any], 
                                       precision_assessment: Dict[str, Any]) -> str:
        """
        Determine the optimal strategy for presenting the response to maximize impact.
        
        Different response characteristics call for different presentation approaches
        to best showcase the value users are receiving from the sophisticated system.
        """
        domain_count = len(domain_coverage['active_domains'])
        balance = domain_coverage['balance_assessment']
        precision_level = precision_assessment['overall_level']
        
        # Multi-domain responses with good precision
        if domain_count >= 3 and precision_level in ['exceptional', 'high']:
            return 'showcase_integration_and_precision'
        
        # Multi-domain responses with moderate precision  
        elif domain_count >= 2 and precision_level in ['good', 'moderate']:
            return 'emphasize_comprehensive_coverage'
        
        # Single domain with exceptional precision
        elif domain_count == 1 and precision_level in ['exceptional', 'high']:
            return 'highlight_domain_expertise'
        
        # Balanced multi-domain responses
        elif domain_count >= 2 and balance == 'well_balanced':
            return 'emphasize_balanced_integration'
        
        # Fallback strategy
        else:
            return 'standard_professional_format'
    
    def _build_sophisticated_response_structure(self, action_plan: str, 
                                              unified_citations: List[Dict[str, Any]], 
                                              precision_analysis: Dict[str, Any], 
                                              characteristics: Dict[str, Any]) -> str:
        """
        Build the sophisticated response structure based on the analysis.
        
        This method creates the actual formatted output using the presentation strategy
        determined by the characteristic analysis. The goal is to make the sophistication
        visible and valuable to users while maintaining professional readability.
        """
        response_parts = []
        
        # Add executive summary that highlights sophistication
        executive_summary = self._create_executive_summary(characteristics, precision_analysis)
        if executive_summary:
            response_parts.append(executive_summary)
        
        # Add the main action plan
        response_parts.append(action_plan)
        
        # Add sophisticated citation display
        citation_display = self._create_sophisticated_citation_display(
            unified_citations, characteristics, precision_analysis
        )
        response_parts.append(citation_display)
        
        # Add system insights footer if appropriate
        if characteristics['presentation_strategy'] in ['showcase_integration_and_precision', 'emphasize_comprehensive_coverage']:
            system_insights = self._create_system_insights_footer(characteristics, precision_analysis)
            if system_insights:
                response_parts.append(system_insights)
        
        return "\n\n".join(response_parts)
    
    def _create_executive_summary(self, characteristics: Dict[str, Any], 
                                precision_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Create an executive summary that immediately communicates system sophistication.
        
        This summary helps users understand the quality and comprehensiveness of the
        analysis they're receiving, setting appropriate expectations for the value.
        """
        strategy = characteristics['presentation_strategy']
        highlights = characteristics['sophistication_highlights']
        
        # Only create executive summary for sophisticated responses
        if strategy not in ['showcase_integration_and_precision', 'emphasize_comprehensive_coverage']:
            return None
        
        summary_parts = []
        
        # Opening statement based on sophistication level
        if characteristics['precision_level'] in ['exceptional', 'high']:
            summary_parts.append("**COMPREHENSIVE COMPLIANCE ANALYSIS**")
        else:
            summary_parts.append("**MULTI-DOMAIN COMPLIANCE GUIDANCE**")
        
        # Highlight key achievements
        if highlights:
            summary_parts.append("This analysis provides:")
            for highlight in highlights[:3]:  # Top 3 highlights
                summary_parts.append(f"• {highlight}")
        
        # Add integration quality note if relevant
        integration_analysis = precision_analysis.get('integration_analysis', {})
        if integration_analysis.get('integration_score', 0) >= 80:
            preservation_quality = integration_analysis.get('preservation_quality', 'unknown')
            if preservation_quality in ['excellent', 'very good']:
                summary_parts.append(f"• Cross-domain precision preservation: {preservation_quality}")
        
        return "\n".join(summary_parts) + "\n" + "="*50
    
    def _create_sophisticated_citation_display(self, unified_citations: List[Dict[str, Any]], 
                                             characteristics: Dict[str, Any], 
                                             precision_analysis: Dict[str, Any]) -> str:
        """
        Create a sophisticated citation display that showcases the precision achieved.
        
        This method formats citations in a way that makes the structural detail and
        cross-domain integration visible to users, demonstrating the value of the
        sophisticated processing pipeline.
        """
        citation_parts = []
        citation_parts.append("**AUTHORITATIVE SOURCE CITATIONS:**\n")
        
        # Group citations by domain for sophisticated presentation
        grouped_citations = self._group_citations_by_domain(unified_citations)
        
        # Format each domain group with appropriate sophistication emphasis
        for domain_name, domain_citations in grouped_citations.items():
            domain_display = self._format_domain_citation_group(
                domain_name, domain_citations, characteristics
            )
            citation_parts.append(domain_display)
        
        # Add precision summary if appropriate
        if characteristics['presentation_strategy'] == 'showcase_integration_and_precision':
            precision_summary = self._create_precision_summary(unified_citations, precision_analysis)
            if precision_summary:
                citation_parts.append(precision_summary)
        
        return "\n".join(citation_parts)
    
    def _group_citations_by_domain(self, unified_citations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group citations by domain for sophisticated presentation.
        
        This grouping helps users understand the comprehensive coverage they're
        receiving and see how different types of expertise contribute to the guidance.
        """
        grouped = {}
        
        for citation in unified_citations:
            domain = citation.get('source_type', 'Unknown')
            if domain not in grouped:
                grouped[domain] = []
            grouped[domain].append(citation)
        
        # Sort domains in logical order for presentation
        domain_order = ['GDPR', 'Polish Law', 'Internal Security']
        ordered_grouped = {}
        
        for domain in domain_order:
            if domain in grouped:
                ordered_grouped[domain] = grouped[domain]
        
        # Add any other domains not in the standard order
        for domain, citations in grouped.items():
            if domain not in ordered_grouped:
                ordered_grouped[domain] = citations
        
        return ordered_grouped
    
    def _format_domain_citation_group(self, domain_name: str, domain_citations: List[Dict[str, Any]], 
                                    characteristics: Dict[str, Any]) -> str:
        """
        Format a group of citations from a specific domain with appropriate emphasis.
        
        This method showcases domain-specific sophistication while maintaining
        consistency across the overall citation presentation.
        """
        group_parts = []
        
        # Create domain header with sophistication indicators
        header = self._create_domain_header(domain_name, domain_citations)
        group_parts.append(header)
        
        # Format individual citations with precision emphasis
        for citation in domain_citations:
            citation_line = self._format_individual_citation_with_precision(citation, characteristics)
            group_parts.append(citation_line)
        
        group_parts.append("")  # Add spacing between domains
        
        return "\n".join(group_parts)
    
    def _create_domain_header(self, domain_name: str, domain_citations: List[Dict[str, Any]]) -> str:
        """
        Create a domain header that communicates the type of expertise being provided.
        
        This helps users understand the different types of authoritative sources
        contributing to their comprehensive guidance.
        """
        # Count precision citations in this domain
        enhanced_count = sum(1 for cite in domain_citations 
                           if cite.get('precision_level') in ['maximum', 'high'])
        
        if domain_name == 'GDPR':
            base_header = "**European Data Protection Regulation (GDPR):**"
        elif domain_name == 'Polish Law':
            base_header = "**Polish Data Protection Implementation:**"
        elif domain_name == 'Internal Security':
            base_header = "**Internal Security Procedures:**"
        else:
            base_header = f"**{domain_name}:**"
        
        # Add precision indicator if significant enhanced citations
        if enhanced_count > 0:
            precision_note = f" [{enhanced_count} with enhanced structural detail]"
            return base_header + precision_note
        
        return base_header
    
    def _format_individual_citation_with_precision(self, citation: Dict[str, Any], 
                                                 characteristics: Dict[str, Any]) -> str:
        """
        Format an individual citation with emphasis on precision when appropriate.
        
        This method showcases the structural detail achieved while maintaining
        readability and professional presentation standards.
        """
        number = citation.get('number', '?')
        reference = citation.get('reference', 'Unknown')
        quote = citation.get('quote', '')
        precision_level = citation.get('precision_level', 'basic')
        
        # Base citation format
        citation_line = f"[{number}] {reference}: \"{quote}\""
        
        # Add precision indicators for sophisticated responses
        if characteristics['presentation_strategy'] in ['showcase_integration_and_precision', 'highlight_domain_expertise']:
            if precision_level == 'maximum':
                citation_line += " ✓"  # Indicator for maximum precision
            elif precision_level == 'high':
                citation_line += " •"  # Indicator for high precision
        
        return citation_line
    
    def _create_precision_summary(self, unified_citations: List[Dict[str, Any]], 
                                precision_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Create a precision summary that showcases system capabilities.
        
        This summary helps users understand the sophisticated analysis they received
        and builds confidence in the quality of the guidance.
        """
        system_metrics = precision_analysis.get('system_metrics', {})
        overall_score = system_metrics.get('overall_precision_score', 0)
        enhanced_count = system_metrics.get('total_enhanced_citations', 0)
        
        if overall_score < 70:  # Only show for high-quality responses
            return None
        
        summary_parts = []
        summary_parts.append("---")
        summary_parts.append("**ANALYSIS QUALITY METRICS:**")
        
        # Overall precision
        summary_parts.append(f"• Overall precision score: {overall_score:.1f}%")
        
        # Enhanced citations
        if enhanced_count > 0:
            total_citations = len(unified_citations)
            enhancement_rate = (enhanced_count / total_citations) * 100
            summary_parts.append(f"• Enhanced structural analysis: {enhanced_count}/{total_citations} citations ({enhancement_rate:.1f}%)")
        
        # Integration quality
        integration_analysis = precision_analysis.get('integration_analysis', {})
        integration_score = integration_analysis.get('integration_score', 0)
        if integration_score >= 80:
            summary_parts.append(f"• Cross-domain integration quality: {integration_score:.1f}%")
        
        return "\n".join(summary_parts)
    
    def _create_system_insights_footer(self, characteristics: Dict[str, Any], 
                                     precision_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Create a system insights footer for highly sophisticated responses.
        
        This footer helps users understand the comprehensive analysis they received
        and encourages appreciation for the system's capabilities.
        """
        insights = precision_analysis.get('insights', [])
        if not insights:
            return None
        
        footer_parts = []
        footer_parts.append("---")
        footer_parts.append("**SYSTEM INSIGHTS:**")
        
        # Show the most relevant insights (max 3)
        relevant_insights = [insight for insight in insights 
                           if any(keyword in insight.lower() for keyword in 
                                ['excellent', 'exceptional', 'optimal', 'sophisticated'])]
        
        display_insights = relevant_insights[:2] if relevant_insights else insights[:2]
        
        for insight in display_insights:
            footer_parts.append(f"• {insight}")
        
        return "\n".join(footer_parts)
    
    def _create_fallback_response_format(self, action_plan: str, 
                                       unified_citations: List[Dict[str, Any]]) -> str:
        """
        Create a fallback response format when sophisticated formatting fails.
        
        This ensures users still receive professional output even when the
        advanced formatting features encounter errors.
        """
        self.logger.warning("Using fallback response formatting due to error")
        
        fallback_parts = []
        
        # Add the action plan
        fallback_parts.append(action_plan)
        
        # Add basic citation formatting
        if unified_citations:
            fallback_parts.append("\n**CITATIONS:**\n")
            for citation in unified_citations:
                number = citation.get('number', '?')
                source = citation.get('source', 'Unknown')
                reference = citation.get('reference', 'Unknown')
                quote = citation.get('quote', '')
                
                citation_line = f"[{number}] {source} {reference}: \"{quote}\""
                fallback_parts.append(citation_line)
        
        return "\n".join(fallback_parts)
    
    def _update_formatting_statistics(self, characteristics: Dict[str, Any], 
                                    unified_citations: List[Dict[str, Any]]) -> None:
        """
        Update formatting statistics based on the response characteristics.
        
        This tracking helps understand usage patterns and optimize presentation
        strategies for different types of responses over time.
        """
        domain_count = characteristics['domain_count']
        
        # Update domain statistics
        if domain_count > 1:
            self.formatting_stats['multi_domain_responses'] += 1
        else:
            self.formatting_stats['single_domain_responses'] += 1
        
        # Track domain combinations
        domains = tuple(sorted(characteristics['active_domains']))
        combination_key = '_'.join(domains)
        self.formatting_stats['domain_combinations_formatted'][combination_key] = \
            self.formatting_stats['domain_combinations_formatted'].get(combination_key, 0) + 1
        
        # Update average citations per response
        citation_count = len(unified_citations)
        current_avg = self.formatting_stats['average_citations_per_response']
        response_count = self.formatting_stats['total_responses_formatted']
        
        new_avg = ((current_avg * (response_count - 1)) + citation_count) / response_count
        self.formatting_stats['average_citations_per_response'] = new_avg
        
        # Track precision showcasing
        if characteristics['presentation_strategy'] in ['showcase_integration_and_precision', 'highlight_domain_expertise']:
            self.formatting_stats['precision_showcasing_instances'] += 1
    
    def get_formatting_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about formatting operations.
        
        Returns:
            Dictionary containing detailed formatting metrics and usage patterns
        """
        stats = dict(self.formatting_stats)
        
        # Calculate derived statistics
        if stats['total_responses_formatted'] > 0:
            multi_domain_rate = (stats['multi_domain_responses'] / stats['total_responses_formatted']) * 100
            stats['multi_domain_rate_percent'] = round(multi_domain_rate, 1)
            
            precision_showcase_rate = (stats['precision_showcasing_instances'] / stats['total_responses_formatted']) * 100
            stats['precision_showcase_rate_percent'] = round(precision_showcase_rate, 1)
        
        return stats
    
    def log_formatting_summary(self) -> None:
        """
        Log a comprehensive summary of all formatting operations.
        
        This provides insights into how the sophisticated presentation features
        are being used and helps optimize the user experience over time.
        """
        stats = self.get_formatting_statistics()
        
        self.logger.info("=== SOPHISTICATED RESPONSE FORMATTING SUMMARY ===")
        self.logger.info(f"Total responses formatted: {stats['total_responses_formatted']}")
        self.logger.info(f"Multi-domain responses: {stats['multi_domain_responses']} ({stats.get('multi_domain_rate_percent', 0)}%)")
        self.logger.info(f"Precision showcasing instances: {stats['precision_showcasing_instances']} ({stats.get('precision_showcase_rate_percent', 0)}%)")
        self.logger.info(f"Average citations per response: {stats['average_citations_per_response']:.1f}")
        self.logger.info(f"Formatting errors: {stats['formatting_errors']}")
        
        # Log popular domain combinations
        if stats['domain_combinations_formatted']:
            self.logger.info("Popular domain combinations:")
            sorted_combinations = sorted(stats['domain_combinations_formatted'].items(), 
                                       key=lambda x: x[1], reverse=True)
            for combination, count in sorted_combinations[:5]:
                self.logger.info(f"  - {combination}: {count} responses")


def create_summarization_formatter(logger: logging.Logger) -> SummarizationFormatter:
    """
    Factory function to create a configured summarization formatter.
    
    This provides a clean interface for creating formatter instances with
    proper dependency injection of the logger.
    """
    return SummarizationFormatter(logger)