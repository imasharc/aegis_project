"""
Summarization Precision Analyzer

This module tackles one of the most sophisticated analytical challenges in your system:
how do you measure and compare "precision" across three completely different domains
that each have their own standards for what constitutes excellent citation quality?

Think of this as a "quality assessment expert" who understands that:
- Legal precision means specific paragraphs and sub-paragraphs within articles
- Procedural precision means implementation steps and configuration details
- System precision means how well all domains work together

The analyzer demonstrates how architectural sophistication enables meaningful
quality measurement across domain boundaries, providing insights that help
optimize the entire multi-agent system for maximum citation value.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple


class SummarizationPrecisionAnalyzer:
    """
    Analyzes and measures citation precision across multiple specialized domains.
    
    This class solves a fascinating analytical challenge: how do you create meaningful
    quality metrics when your system integrates work from three agents that each
    have completely different precision standards? Legal citations use paragraph
    references, security procedures use implementation steps, but users need to
    understand the overall quality of the integrated result.
    
    The analyzer demonstrates advanced cross-domain analytics - it understands what
    constitutes excellence in each domain and creates unified quality metrics that
    help users understand the sophistication they're receiving.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summarization precision analyzer.
        
        Args:
            logger: Configured logger for tracking precision analysis operations
        """
        self.logger = logger
        self.logger.info("Summarization Precision Analyzer initialized")
        
        # Track precision analysis statistics across all operations
        self.analysis_stats = {
            'total_citation_sets_analyzed': 0,
            'cross_domain_analyses_performed': 0,
            'precision_assessments_completed': 0,
            'domain_specific_metrics': {
                'gdpr': {'total': 0, 'enhanced': 0, 'precision_scores': []},
                'polish_law': {'total': 0, 'enhanced': 0, 'precision_scores': []},
                'security': {'total': 0, 'enhanced': 0, 'precision_scores': []}
            },
            'overall_system_scores': [],
            'precision_distribution': {
                'maximum': 0, 'high': 0, 'medium': 0, 'basic': 0, 'minimal': 0
            }
        }
    
    def analyze_multi_domain_precision(self, unified_citations: List[Dict[str, Any]], 
                                     gdpr_citations: List[Dict[str, Any]], 
                                     polish_law_citations: List[Dict[str, Any]], 
                                     internal_policy_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive precision analysis across all domains and their integration.
        
        This method represents the culmination of cross-domain quality assessment. It
        analyzes not just how well each individual domain performed, but how successfully
        the integration process preserved and enhanced the overall precision of the system.
        
        The method demonstrates how sophisticated analytics can provide meaningful
        insights into system performance across domain boundaries, enabling continuous
        optimization of the entire multi-agent architecture.
        
        Args:
            unified_citations: The final integrated citations from all domains
            gdpr_citations: Original GDPR citations for domain-specific analysis
            polish_law_citations: Original Polish law citations for domain-specific analysis  
            internal_policy_citations: Original security citations for domain-specific analysis
            
        Returns:
            Comprehensive precision analysis results with cross-domain insights
        """
        self.logger.info("Starting comprehensive multi-domain precision analysis")
        self.analysis_stats['total_citation_sets_analyzed'] += 1
        self.analysis_stats['cross_domain_analyses_performed'] += 1
        
        try:
            # Analyze precision within each domain using domain-specific expertise
            domain_analyses = self._analyze_individual_domains(
                gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Analyze the quality of the cross-domain integration
            integration_analysis = self._analyze_integration_quality(unified_citations, domain_analyses)
            
            # Calculate comprehensive system-wide precision metrics
            system_metrics = self._calculate_system_precision_metrics(domain_analyses, integration_analysis)
            
            # Generate insights and recommendations for optimization
            insights = self._generate_precision_insights(domain_analyses, integration_analysis, system_metrics)
            
            # Compile the complete analysis results
            analysis_results = {
                'domain_analyses': domain_analyses,
                'integration_analysis': integration_analysis,
                'system_metrics': system_metrics,
                'insights': insights,
                'analysis_timestamp': self._get_analysis_timestamp(),
                'analysis_quality': self._assess_analysis_completeness(domain_analyses, integration_analysis)
            }
            
            # Update internal statistics and tracking
            self._update_analysis_statistics(analysis_results)
            
            self.logger.info(f"Multi-domain precision analysis completed: {system_metrics.get('overall_precision_score', 0):.1f}% system precision")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during multi-domain precision analysis: {e}")
            return self._create_fallback_analysis(unified_citations)
    
    def _analyze_individual_domains(self, gdpr_citations: List[Dict[str, Any]], 
                                   polish_law_citations: List[Dict[str, Any]], 
                                   internal_policy_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze precision within each domain using domain-specific quality standards.
        
        This method demonstrates deep domain expertise - it understands what constitutes
        excellent citations in each specialized area and applies appropriate quality
        standards rather than using generic metrics across all domains.
        """
        domain_analyses = {}
        
        # Analyze GDPR domain with legal precision standards
        if gdpr_citations:
            gdpr_analysis = self._analyze_legal_domain_precision(gdpr_citations, 'GDPR')
            domain_analyses['gdpr'] = gdpr_analysis
            
            # Update domain-specific statistics
            self.analysis_stats['domain_specific_metrics']['gdpr']['total'] += len(gdpr_citations)
            self.analysis_stats['domain_specific_metrics']['gdpr']['enhanced'] += gdpr_analysis['enhanced_citations']
            self.analysis_stats['domain_specific_metrics']['gdpr']['precision_scores'].append(gdpr_analysis['domain_precision_score'])
            
            self.logger.debug(f"GDPR domain analysis: {gdpr_analysis['domain_precision_score']:.1f}% precision")
        
        # Analyze Polish Law domain with enhanced legal precision standards
        if polish_law_citations:
            polish_analysis = self._analyze_legal_domain_precision(polish_law_citations, 'Polish Law')
            domain_analyses['polish_law'] = polish_analysis
            
            # Update domain-specific statistics
            self.analysis_stats['domain_specific_metrics']['polish_law']['total'] += len(polish_law_citations)
            self.analysis_stats['domain_specific_metrics']['polish_law']['enhanced'] += polish_analysis['enhanced_citations']
            self.analysis_stats['domain_specific_metrics']['polish_law']['precision_scores'].append(polish_analysis['domain_precision_score'])
            
            self.logger.debug(f"Polish Law domain analysis: {polish_analysis['domain_precision_score']:.1f}% precision")
        
        # Analyze Security domain with procedural precision standards
        if internal_policy_citations:
            security_analysis = self._analyze_procedural_domain_precision(internal_policy_citations)
            domain_analyses['security'] = security_analysis
            
            # Update domain-specific statistics
            self.analysis_stats['domain_specific_metrics']['security']['total'] += len(internal_policy_citations)
            self.analysis_stats['domain_specific_metrics']['security']['enhanced'] += security_analysis['enhanced_citations']
            self.analysis_stats['domain_specific_metrics']['security']['precision_scores'].append(security_analysis['domain_precision_score'])
            
            self.logger.debug(f"Security domain analysis: {security_analysis['domain_precision_score']:.1f}% precision")
        
        self.logger.info(f"Individual domain analyses completed for {len(domain_analyses)} active domains")
        
        return domain_analyses
    
    def _analyze_legal_domain_precision(self, citations: List[Dict[str, Any]], domain_name: str) -> Dict[str, Any]:
        """
        Analyze precision using legal domain expertise for GDPR and Polish Law citations.
        
        This method applies sophisticated understanding of what makes legal citations
        excellent - specific paragraph references, chapter context, and structural
        detail that enables precise legal compliance guidance.
        """
        if not citations:
            return self._create_empty_domain_analysis(domain_name)
        
        # Analyze each citation for legal precision indicators
        precision_scores = []
        enhanced_count = 0
        precision_distribution = {'maximum': 0, 'high': 0, 'medium': 0, 'basic': 0, 'minimal': 0}
        
        for citation in citations:
            # Extract the legal reference for analysis
            article_reference = citation.get('article', '')
            explanation = citation.get('explanation', '')
            quote = citation.get('quote', '')
            
            # Calculate citation precision score using legal domain expertise
            precision_score, precision_level = self._calculate_legal_citation_score(
                article_reference, explanation, quote, domain_name
            )
            
            precision_scores.append(precision_score)
            precision_distribution[precision_level] += 1
            
            # Count enhanced citations (those with structural detail)
            if self._is_enhanced_legal_citation(article_reference, domain_name):
                enhanced_count += 1
        
        # Calculate domain-wide metrics
        domain_precision_score = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        enhancement_rate = (enhanced_count / len(citations)) * 100 if citations else 0
        
        analysis = {
            'domain_name': domain_name,
            'total_citations': len(citations),
            'enhanced_citations': enhanced_count,
            'enhancement_rate': round(enhancement_rate, 1),
            'domain_precision_score': round(domain_precision_score, 1),
            'precision_distribution': precision_distribution,
            'quality_assessment': self._assess_legal_domain_quality(domain_precision_score, enhancement_rate),
            'domain_type': 'legal'
        }
        
        return analysis
    
    def _analyze_procedural_domain_precision(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze precision using procedural domain expertise for security procedure citations.
        
        This method applies specialized understanding of what makes security procedure
        citations excellent - implementation step detail, configuration specificity,
        and workflow precision that enables actionable security guidance.
        """
        if not citations:
            return self._create_empty_domain_analysis('Security')
        
        # Analyze each citation for procedural precision indicators
        precision_scores = []
        enhanced_count = 0
        precision_distribution = {'maximum': 0, 'high': 0, 'medium': 0, 'basic': 0, 'minimal': 0}
        
        for citation in citations:
            # Extract the procedural reference for analysis (flexible key handling)
            procedure_reference = citation.get('procedure', citation.get('article', citation.get('section', '')))
            explanation = citation.get('explanation', '')
            quote = citation.get('quote', '')
            
            # Calculate citation precision score using procedural domain expertise
            precision_score, precision_level = self._calculate_procedural_citation_score(
                procedure_reference, explanation, quote
            )
            
            precision_scores.append(precision_score)
            precision_distribution[precision_level] += 1
            
            # Count enhanced citations (those with implementation detail)
            if self._is_enhanced_procedural_citation(procedure_reference):
                enhanced_count += 1
        
        # Calculate domain-wide metrics
        domain_precision_score = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        enhancement_rate = (enhanced_count / len(citations)) * 100 if citations else 0
        
        analysis = {
            'domain_name': 'Security',
            'total_citations': len(citations),
            'enhanced_citations': enhanced_count,
            'enhancement_rate': round(enhancement_rate, 1),
            'domain_precision_score': round(domain_precision_score, 1),
            'precision_distribution': precision_distribution,
            'quality_assessment': self._assess_procedural_domain_quality(domain_precision_score, enhancement_rate),
            'domain_type': 'procedural'
        }
        
        return analysis
    
    def _calculate_legal_citation_score(self, article_reference: str, explanation: str, 
                                      quote: str, domain_name: str) -> Tuple[float, str]:
        """
        Calculate a precision score for legal citations using domain-specific expertise.
        
        This method demonstrates sophisticated quality assessment - it understands the
        specific indicators that make legal citations valuable and assigns scores
        based on the structural detail and legal precision present.
        """
        score = 0
        precision_level = 'minimal'
        
        if not article_reference:
            return 0, 'minimal'
        
        reference_lower = article_reference.lower()
        
        # Base score for having an article reference
        score = 20
        precision_level = 'basic'
        
        # Enhanced scoring for legal structural details
        if 'paragraph' in reference_lower:
            score += 25
            precision_level = 'medium'
            
            # Bonus for sub-paragraph detail
            if any(indicator in reference_lower for indicator in ['(a)', '(b)', '(c)', '(1)', '(2)', '(3)']):
                score += 20
                precision_level = 'high'
        
        # Bonus for chapter or section context
        if 'chapter' in reference_lower or 'section' in reference_lower:
            score += 15
            
            # Maximum precision for detailed structural context
            if precision_level == 'high':
                precision_level = 'maximum'
                score += 10
        
        # Quality indicators from explanation and quote
        if explanation and len(explanation) > 50:
            score += 10  # Substantial explanation
        
        if quote and len(quote) > 30:
            score += 10  # Substantial quote
        
        # Domain-specific bonuses
        if domain_name == 'Polish Law':
            # Polish Law agent's enhanced metadata deserves recognition
            if len(article_reference) > 25:  # Detailed reference indicates enhanced processing
                score += 5
        
        # Normalize score to 0-100 range
        final_score = min(score, 100)
        
        return final_score, precision_level
    
    def _calculate_procedural_citation_score(self, procedure_reference: str, explanation: str, 
                                           quote: str) -> Tuple[float, str]:
        """
        Calculate a precision score for procedural citations using domain-specific expertise.
        
        This method applies specialized understanding of security procedure quality,
        recognizing implementation detail and actionable specificity as key indicators
        of citation value for operational security guidance.
        """
        score = 0
        precision_level = 'minimal'
        
        if not procedure_reference:
            return 0, 'minimal'
        
        reference_lower = procedure_reference.lower()
        
        # Base score for having a procedure reference
        score = 20
        precision_level = 'basic'
        
        # Enhanced scoring for procedural implementation details
        if 'step' in reference_lower:
            score += 30
            precision_level = 'medium'
            
            # Bonus for specific step details
            if any(indicator in reference_lower for indicator in [
                'configure', 'implementation', 'phase', 'workflow'
            ]):
                score += 20
                precision_level = 'high'
        
        # Bonus for procedure structure and context
        if 'procedure' in reference_lower and ':' in procedure_reference:
            score += 15  # Procedure with title/description
            
            # Maximum precision for detailed implementation guidance
            if precision_level == 'high':
                precision_level = 'maximum'
                score += 15
        
        # Security-specific quality indicators
        security_indicators = ['access', 'security', 'authentication', 'authorization', 'monitor', 'audit']
        security_relevance = sum(1 for indicator in security_indicators 
                               if indicator in reference_lower or indicator in explanation.lower())
        
        if security_relevance > 0:
            score += min(security_relevance * 5, 15)  # Bonus for security relevance
        
        # Quality indicators from explanation and quote
        if explanation and len(explanation) > 50:
            score += 10  # Substantial explanation
        
        if quote and len(quote) > 30:
            score += 10  # Substantial quote
        
        # Normalize score to 0-100 range
        final_score = min(score, 100)
        
        return final_score, precision_level
    
    def _is_enhanced_legal_citation(self, article_reference: str, domain_name: str) -> bool:
        """
        Determine if a legal citation demonstrates enhanced precision from metadata flattening.
        
        This method recognizes when legal citations include the sophisticated structural
        detail that indicates the agent successfully used metadata flattening approaches.
        """
        if not article_reference:
            return False
        
        reference_lower = article_reference.lower()
        
        # Legal enhancement indicators
        enhancement_indicators = [
            'paragraph', 'sub-paragraph', '(a)', '(b)', '(c)', '(d)', 
            '(1)', '(2)', '(3)', '(4)', 'chapter', 'section'
        ]
        
        return any(indicator in reference_lower for indicator in enhancement_indicators)
    
    def _is_enhanced_procedural_citation(self, procedure_reference: str) -> bool:
        """
        Determine if a procedural citation demonstrates enhanced precision from component integration.
        
        This method recognizes when security citations include the implementation detail
        that indicates successful component architecture and metadata processing.
        """
        if not procedure_reference:
            return False
        
        reference_lower = procedure_reference.lower()
        
        # Procedural enhancement indicators
        enhancement_indicators = [
            'step', 'configure', 'implementation', 'phase', 'procedure',
            'process', 'workflow', 'requirement'
        ]
        
        return any(indicator in reference_lower for indicator in enhancement_indicators)
    
    def _analyze_integration_quality(self, unified_citations: List[Dict[str, Any]], 
                                   domain_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how well the multi-domain integration preserved and enhanced overall precision.
        
        This method evaluates the sophisticated challenge of unifying three different
        citation formats while maintaining their individual precision characteristics.
        It measures whether the integration process added value or lost precision.
        """
        if not unified_citations:
            return {'integration_score': 0, 'preservation_quality': 'poor', 'integration_insights': []}
        
        # Calculate integration preservation metrics
        preservation_metrics = self._calculate_preservation_metrics(unified_citations, domain_analyses)
        
        # Assess unified numbering and organization quality
        organization_quality = self._assess_organization_quality(unified_citations)
        
        # Evaluate cross-domain coherence
        coherence_assessment = self._assess_cross_domain_coherence(unified_citations)
        
        # Calculate overall integration score
        integration_score = self._calculate_integration_score(
            preservation_metrics, organization_quality, coherence_assessment
        )
        
        # Generate integration insights
        integration_insights = self._generate_integration_insights(
            preservation_metrics, organization_quality, coherence_assessment, integration_score
        )
        
        integration_analysis = {
            'integration_score': round(integration_score, 1),
            'preservation_metrics': preservation_metrics,
            'organization_quality': organization_quality,
            'coherence_assessment': coherence_assessment,
            'preservation_quality': self._categorize_integration_quality(integration_score),
            'integration_insights': integration_insights,
            'total_unified_citations': len(unified_citations)
        }
        
        self.logger.info(f"Integration analysis completed: {integration_score:.1f}% integration quality")
        
        return integration_analysis
    
    def _calculate_preservation_metrics(self, unified_citations: List[Dict[str, Any]], 
                                      domain_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics showing how well domain precision was preserved during integration.
        
        This measures whether the sophisticated work done by individual agents was
        maintained through the unification process or lost during integration.
        """
        preserved_precision = sum(1 for cite in unified_citations 
                                if cite.get('precision_preserved', False))
        
        total_citations = len(unified_citations)
        preservation_rate = (preserved_precision / total_citations * 100) if total_citations > 0 else 0
        
        # Analyze precision level distribution in unified citations
        precision_distribution = {}
        for citation in unified_citations:
            level = citation.get('precision_level', 'unknown')
            precision_distribution[level] = precision_distribution.get(level, 0) + 1
        
        return {
            'preservation_rate': round(preservation_rate, 1),
            'preserved_citations': preserved_precision,
            'total_citations': total_citations,
            'precision_distribution': precision_distribution
        }
    
    def _assess_organization_quality(self, unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of unified citation organization and numbering.
        
        This evaluates whether the unification process created a coherent,
        well-organized presentation that enhances user understanding.
        """
        # Check numbering consistency
        correctly_numbered = sum(1 for i, cite in enumerate(unified_citations)
                               if cite.get('number') == i + 1)
        numbering_accuracy = (correctly_numbered / len(unified_citations) * 100) if unified_citations else 0
        
        # Check source diversity
        unique_sources = len(set(cite.get('source_type', 'unknown') for cite in unified_citations))
        
        # Check reference quality
        complete_references = sum(1 for cite in unified_citations
                                if cite.get('reference') and cite.get('quote') and cite.get('explanation'))
        completeness_rate = (complete_references / len(unified_citations) * 100) if unified_citations else 0
        
        return {
            'numbering_accuracy': round(numbering_accuracy, 1),
            'source_diversity': unique_sources,
            'completeness_rate': round(completeness_rate, 1),
            'organization_score': round((numbering_accuracy + completeness_rate) / 2, 1)
        }
    
    def _assess_cross_domain_coherence(self, unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess how well citations from different domains work together coherently.
        
        This evaluates whether the multi-domain integration creates a unified
        experience or feels like disconnected pieces from different systems.
        """
        domain_counts = {}
        domain_quality_scores = {}
        
        for citation in unified_citations:
            domain = citation.get('domain', 'unknown')
            precision_level = citation.get('precision_level', 'minimal')
            
            # Count domains
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Track quality by domain
            if domain not in domain_quality_scores:
                domain_quality_scores[domain] = []
            
            # Convert precision level to score
            level_scores = {'maximum': 100, 'high': 80, 'medium': 60, 'basic': 40, 'minimal': 20}
            score = level_scores.get(precision_level, 20)
            domain_quality_scores[domain].append(score)
        
        # Calculate coherence metrics
        domain_balance = self._calculate_domain_balance(domain_counts)
        quality_consistency = self._calculate_quality_consistency(domain_quality_scores)
        
        return {
            'domain_counts': domain_counts,
            'domain_balance': round(domain_balance, 1),
            'quality_consistency': round(quality_consistency, 1),
            'coherence_score': round((domain_balance + quality_consistency) / 2, 1)
        }
    
    def _calculate_domain_balance(self, domain_counts: Dict[str, int]) -> float:
        """
        Calculate how well-balanced the domain representation is in unified citations.
        
        Perfect balance isn't always desired (queries might naturally favor one domain),
        but extreme imbalance might indicate retrieval or processing issues.
        """
        if not domain_counts:
            return 0
        
        total_citations = sum(domain_counts.values())
        if total_citations == 0:
            return 0
        
        # Calculate distribution entropy as a balance measure
        import math
        balance_score = 0
        
        for count in domain_counts.values():
            if count > 0:
                proportion = count / total_citations
                balance_score -= proportion * math.log2(proportion)
        
        # Normalize to 0-100 scale (log2(3) ≈ 1.58 is maximum for 3 domains)
        max_entropy = math.log2(len(domain_counts)) if len(domain_counts) > 1 else 1
        normalized_score = (balance_score / max_entropy) * 100 if max_entropy > 0 else 100
        
        return min(normalized_score, 100)
    
    def _calculate_quality_consistency(self, domain_quality_scores: Dict[str, List[float]]) -> float:
        """
        Calculate how consistent the quality is across different domains.
        
        High consistency indicates that all domains are performing well and
        the integration process isn't favoring one domain over others.
        """
        if not domain_quality_scores:
            return 0
        
        # Calculate average quality for each domain
        domain_averages = {}
        for domain, scores in domain_quality_scores.items():
            if scores:
                domain_averages[domain] = sum(scores) / len(scores)
        
        if not domain_averages:
            return 0
        
        # Calculate consistency as inverse of standard deviation
        average_scores = list(domain_averages.values())
        if len(average_scores) < 2:
            return 100  # Single domain is perfectly consistent
        
        mean_score = sum(average_scores) / len(average_scores)
        variance = sum((score - mean_score) ** 2 for score in average_scores) / len(average_scores)
        std_dev = variance ** 0.5
        
        # Convert to consistency score (lower std_dev = higher consistency)
        # Maximum possible std_dev is around 40 (range 20-100 with mean 60)
        consistency_score = max(0, 100 - (std_dev * 2.5))
        
        return consistency_score
    
    def _calculate_integration_score(self, preservation_metrics: Dict[str, Any], 
                                   organization_quality: Dict[str, Any], 
                                   coherence_assessment: Dict[str, Any]) -> float:
        """
        Calculate an overall integration score from all quality metrics.
        
        This provides a single number that represents how well the multi-domain
        integration performed across all dimensions of quality assessment.
        """
        # Weight different aspects of integration quality
        preservation_weight = 0.4  # Most important: did we preserve domain precision?
        organization_weight = 0.3  # Important: is the result well-organized?
        coherence_weight = 0.3     # Important: do domains work well together?
        
        preservation_score = preservation_metrics.get('preservation_rate', 0)
        organization_score = organization_quality.get('organization_score', 0)
        coherence_score = coherence_assessment.get('coherence_score', 0)
        
        integration_score = (
            preservation_score * preservation_weight +
            organization_score * organization_weight +
            coherence_score * coherence_weight
        )
        
        return integration_score
    
    def _generate_integration_insights(self, preservation_metrics: Dict[str, Any], 
                                     organization_quality: Dict[str, Any], 
                                     coherence_assessment: Dict[str, Any], 
                                     integration_score: float) -> List[str]:
        """
        Generate actionable insights about integration quality.
        
        These insights help understand what's working well and what could be
        optimized in the multi-domain integration process.
        """
        insights = []
        
        # Preservation insights
        preservation_rate = preservation_metrics.get('preservation_rate', 0)
        if preservation_rate >= 90:
            insights.append("Excellent precision preservation - domain expertise successfully maintained")
        elif preservation_rate >= 75:
            insights.append("Good precision preservation - most domain detail retained through integration")
        elif preservation_rate >= 50:
            insights.append("Moderate precision preservation - review integration process for optimization")
        else:
            insights.append("Low precision preservation - integration may be losing valuable domain detail")
        
        # Organization insights
        organization_score = organization_quality.get('organization_score', 0)
        if organization_score >= 95:
            insights.append("Perfect citation organization - unified presentation is well-structured")
        elif organization_score < 80:
            insights.append("Citation organization could be improved - check numbering and completeness")
        
        # Coherence insights
        coherence_score = coherence_assessment.get('coherence_score', 0)
        domain_balance = coherence_assessment.get('domain_balance', 0)
        
        if coherence_score >= 80:
            insights.append("Strong cross-domain coherence - domains integrate smoothly")
        elif coherence_score < 60:
            insights.append("Cross-domain coherence needs attention - domains may feel disconnected")
        
        if domain_balance < 30:
            insights.append("Domain representation is highly skewed - consider query or retrieval tuning")
        
        # Overall insights
        if integration_score >= 90:
            insights.append("Outstanding multi-domain integration - system performing at optimal level")
        elif integration_score >= 75:
            insights.append("Strong multi-domain integration - minor optimizations could enhance performance")
        elif integration_score >= 60:
            insights.append("Adequate multi-domain integration - several areas have optimization potential")
        else:
            insights.append("Multi-domain integration needs significant improvement - review entire pipeline")
        
        return insights
    
    def _categorize_integration_quality(self, integration_score: float) -> str:
        """
        Categorize integration quality based on the overall score.
        
        This provides a human-readable assessment of how well the integration performed.
        """
        if integration_score >= 90:
            return "excellent"
        elif integration_score >= 80:
            return "very good"
        elif integration_score >= 70:
            return "good"
        elif integration_score >= 60:
            return "fair"
        elif integration_score >= 50:
            return "poor"
        else:
            return "very poor"
    
    def _calculate_system_precision_metrics(self, domain_analyses: Dict[str, Any], 
                                          integration_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive system-wide precision metrics.
        
        This provides the "big picture" view of how well the entire multi-agent
        system performed across all domains and integration challenges.
        """
        # Calculate weighted overall precision score
        total_citations = 0
        weighted_precision_sum = 0
        
        for domain_analysis in domain_analyses.values():
            domain_citations = domain_analysis.get('total_citations', 0)
            domain_score = domain_analysis.get('domain_precision_score', 0)
            
            total_citations += domain_citations
            weighted_precision_sum += domain_citations * domain_score
        
        overall_precision_score = (weighted_precision_sum / total_citations) if total_citations > 0 else 0
        
        # Calculate system enhancement rate
        total_enhanced = sum(analysis.get('enhanced_citations', 0) for analysis in domain_analyses.values())
        system_enhancement_rate = (total_enhanced / total_citations * 100) if total_citations > 0 else 0
        
        # Integration impact assessment
        integration_score = integration_analysis.get('integration_score', 0)
        integration_impact = self._assess_integration_impact(overall_precision_score, integration_score)
        
        system_metrics = {
            'overall_precision_score': round(overall_precision_score, 1),
            'system_enhancement_rate': round(system_enhancement_rate, 1),
            'integration_score': integration_score,
            'integration_impact': integration_impact,
            'total_citations': total_citations,
            'total_enhanced_citations': total_enhanced,
            'active_domains': len(domain_analyses),
            'system_quality_category': self._categorize_system_quality(overall_precision_score, integration_score)
        }
        
        # Add to running statistics
        self.analysis_stats['overall_system_scores'].append(overall_precision_score)
        
        return system_metrics
    
    def _assess_integration_impact(self, precision_score: float, integration_score: float) -> str:
        """
        Assess whether integration enhanced or degraded overall system performance.
        
        This helps understand if the multi-domain approach is adding value or
        creating unnecessary complexity compared to individual domain performance.
        """
        score_difference = integration_score - precision_score
        
        if score_difference > 10:
            return "significantly_positive"
        elif score_difference > 5:
            return "positive"
        elif score_difference > -5:
            return "neutral"
        elif score_difference > -10:
            return "negative"
        else:
            return "significantly_negative"
    
    def _categorize_system_quality(self, precision_score: float, integration_score: float) -> str:
        """
        Categorize overall system quality based on both precision and integration performance.
        
        This provides a comprehensive assessment of the entire multi-agent system's
        performance across all dimensions of quality.
        """
        average_score = (precision_score + integration_score) / 2
        
        if average_score >= 90 and min(precision_score, integration_score) >= 85:
            return "exceptional"
        elif average_score >= 80 and min(precision_score, integration_score) >= 70:
            return "excellent"
        elif average_score >= 70 and min(precision_score, integration_score) >= 60:
            return "good"
        elif average_score >= 60:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _generate_precision_insights(self, domain_analyses: Dict[str, Any], 
                                   integration_analysis: Dict[str, Any], 
                                   system_metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable insights about overall system precision performance.
        
        These insights help understand system strengths and identify specific
        areas where optimization efforts would be most beneficial.
        """
        insights = []
        
        # System-level insights
        overall_score = system_metrics['overall_precision_score']
        if overall_score >= 85:
            insights.append("Exceptional system precision - multi-agent architecture performing optimally")
        elif overall_score >= 75:
            insights.append("Strong system precision - minor optimizations could enhance performance")
        elif overall_score >= 65:
            insights.append("Good system precision - focus optimization on lowest-performing domains")
        else:
            insights.append("System precision needs improvement - consider component-level optimization")
        
        # Domain-specific insights
        domain_scores = {name: analysis.get('domain_precision_score', 0) 
                        for name, analysis in domain_analyses.items()}
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            worst_domain = min(domain_scores, key=domain_scores.get)
            
            best_score = domain_scores[best_domain]
            worst_score = domain_scores[worst_domain]
            
            if best_score - worst_score > 20:
                insights.append(f"Large performance gap between domains - {worst_domain} domain needs attention")
            
            if best_score >= 85:
                insights.append(f"{best_domain} domain demonstrates excellent precision - consider applying patterns to other domains")
        
        # Enhancement insights
        enhancement_rate = system_metrics['system_enhancement_rate']
        if enhancement_rate >= 80:
            insights.append("Excellent citation enhancement - metadata flattening approaches highly effective")
        elif enhancement_rate >= 60:
            insights.append("Good citation enhancement - most citations include structural detail")
        elif enhancement_rate < 40:
            insights.append("Low citation enhancement - review metadata processing and citation building")
        
        # Integration insights
        integration_impact = system_metrics['integration_impact']
        if integration_impact in ['positive', 'significantly_positive']:
            insights.append("Multi-domain integration adds value - unified approach outperforms individual domains")
        elif integration_impact in ['negative', 'significantly_negative']:
            insights.append("Integration may be degrading performance - review unification process")
        
        return insights
    
    def _create_empty_domain_analysis(self, domain_name: str) -> Dict[str, Any]:
        """
        Create an empty domain analysis structure for domains with no citations.
        
        This ensures consistent analysis structure even when some domains don't
        contribute citations to a particular query response.
        """
        return {
            'domain_name': domain_name,
            'total_citations': 0,
            'enhanced_citations': 0,
            'enhancement_rate': 0,
            'domain_precision_score': 0,
            'precision_distribution': {},
            'quality_assessment': 'no_data',
            'domain_type': 'legal' if domain_name in ['GDPR', 'Polish Law'] else 'procedural'
        }
    
    def _assess_legal_domain_quality(self, precision_score: float, enhancement_rate: float) -> str:
        """
        Assess the quality of legal domain performance using domain-specific standards.
        
        This applies legal domain expertise to evaluate whether the citations meet
        the standards expected for legal compliance guidance.
        """
        if precision_score >= 85 and enhancement_rate >= 75:
            return "excellent"
        elif precision_score >= 75 and enhancement_rate >= 60:
            return "very_good"
        elif precision_score >= 65 and enhancement_rate >= 45:
            return "good"
        elif precision_score >= 55:
            return "adequate"
        else:
            return "needs_improvement"
    
    def _assess_procedural_domain_quality(self, precision_score: float, enhancement_rate: float) -> str:
        """
        Assess the quality of procedural domain performance using domain-specific standards.
        
        This applies procedural domain expertise to evaluate whether the citations meet
        the standards expected for actionable security implementation guidance.
        """
        if precision_score >= 80 and enhancement_rate >= 70:
            return "excellent"
        elif precision_score >= 70 and enhancement_rate >= 55:
            return "very_good"
        elif precision_score >= 60 and enhancement_rate >= 40:
            return "good"
        elif precision_score >= 50:
            return "adequate"
        else:
            return "needs_improvement"
    
    def _create_fallback_analysis(self, unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a fallback analysis when comprehensive analysis fails.
        
        This ensures the system continues to provide some precision insights
        even when the sophisticated analysis encounters errors.
        """
        return {
            'domain_analyses': {},
            'integration_analysis': {'integration_score': 0, 'preservation_quality': 'error'},
            'system_metrics': {
                'overall_precision_score': 0,
                'system_enhancement_rate': 0,
                'integration_score': 0,
                'total_citations': len(unified_citations)
            },
            'insights': ['Precision analysis failed - using fallback assessment'],
            'analysis_quality': 'fallback'
        }
    
    def _get_analysis_timestamp(self) -> str:
        """Get a timestamp for the analysis operation."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _assess_analysis_completeness(self, domain_analyses: Dict[str, Any], 
                                    integration_analysis: Dict[str, Any]) -> str:
        """
        Assess the completeness and reliability of the precision analysis.
        
        This provides confidence information about the analysis results to help
        users understand how much trust to place in the precision metrics.
        """
        completeness_score = 0
        
        # Domain analysis completeness
        if domain_analyses:
            completeness_score += 40
            if len(domain_analyses) >= 2:
                completeness_score += 20  # Multi-domain analysis
        
        # Integration analysis completeness
        if integration_analysis.get('integration_score', 0) > 0:
            completeness_score += 30
        
        # Additional quality indicators
        if integration_analysis.get('integration_insights'):
            completeness_score += 10
        
        if completeness_score >= 90:
            return "complete"
        elif completeness_score >= 70:
            return "mostly_complete"
        elif completeness_score >= 50:
            return "partial"
        else:
            return "limited"
    
    def _update_analysis_statistics(self, analysis_results: Dict[str, Any]) -> None:
        """
        Update internal statistics based on analysis results.
        
        This tracks long-term patterns in precision analysis to enable
        system optimization and performance monitoring over time.
        """
        self.analysis_stats['precision_assessments_completed'] += 1
        
        # Update precision distribution
        system_metrics = analysis_results.get('system_metrics', {})
        overall_score = system_metrics.get('overall_precision_score', 0)
        
        if overall_score >= 90:
            category = 'maximum'
        elif overall_score >= 80:
            category = 'high'
        elif overall_score >= 70:
            category = 'medium'
        elif overall_score >= 60:
            category = 'basic'
        else:
            category = 'minimal'
        
        self.analysis_stats['precision_distribution'][category] += 1
    
    def get_precision_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about precision analysis operations.
        
        Returns:
            Dictionary containing detailed precision analysis metrics and trends
        """
        stats = dict(self.analysis_stats)
        
        # Calculate rolling averages and trends
        if stats['overall_system_scores']:
            scores = stats['overall_system_scores']
            stats['average_system_score'] = round(sum(scores) / len(scores), 1)
            
            # Calculate trend (recent vs older scores)
            if len(scores) >= 4:
                recent_avg = sum(scores[-2:]) / 2 if len(scores) >= 2 else scores[-1]
                older_avg = sum(scores[:-2]) / len(scores[:-2]) if len(scores) > 2 else recent_avg
                trend = recent_avg - older_avg
                
                if trend > 5:
                    stats['performance_trend'] = 'improving'
                elif trend < -5:
                    stats['performance_trend'] = 'declining'
                else:
                    stats['performance_trend'] = 'stable'
            else:
                stats['performance_trend'] = 'insufficient_data'
        
        # Calculate domain-specific averages
        for domain, metrics in stats['domain_specific_metrics'].items():
            if metrics['precision_scores']:
                avg_score = sum(metrics['precision_scores']) / len(metrics['precision_scores'])
                stats[f'{domain}_average_score'] = round(avg_score, 1)
                
                if metrics['total'] > 0:
                    enhancement_rate = (metrics['enhanced'] / metrics['total']) * 100
                    stats[f'{domain}_enhancement_rate'] = round(enhancement_rate, 1)
        
        return stats
    
    def log_precision_analysis_summary(self) -> None:
        """
        Log a comprehensive summary of all precision analysis operations.
        
        This provides insights into system performance trends and helps identify
        opportunities for precision optimization across all domains.
        """
        stats = self.get_precision_analysis_statistics()
        
        self.logger.info("=== MULTI-DOMAIN PRECISION ANALYSIS SUMMARY ===")
        self.logger.info(f"Citation sets analyzed: {stats['total_citation_sets_analyzed']}")
        self.logger.info(f"Precision assessments completed: {stats['precision_assessments_completed']}")
        self.logger.info(f"Average system score: {stats.get('average_system_score', 'N/A')}")
        self.logger.info(f"Performance trend: {stats.get('performance_trend', 'N/A')}")
        
        # Log domain-specific performance
        self.logger.info("Domain-specific performance:")
        for domain in ['gdpr', 'polish_law', 'security']:
            avg_score = stats.get(f'{domain}_average_score', 'N/A')
            enhancement_rate = stats.get(f'{domain}_enhancement_rate', 'N/A')
            total_citations = stats['domain_specific_metrics'][domain]['total']
            
            self.logger.info(f"  - {domain}: {total_citations} citations, avg score: {avg_score}, enhancement: {enhancement_rate}%")
        
        # Log precision distribution
        distribution = stats['precision_distribution']
        total_analyses = sum(distribution.values())
        if total_analyses > 0:
            self.logger.info("Precision distribution:")
            for level, count in distribution.items():
                percentage = (count / total_analyses) * 100
                self.logger.info(f"  - {level}: {count} ({percentage:.1f}%)")


def create_summarization_precision_analyzer(logger: logging.Logger) -> SummarizationPrecisionAnalyzer:
    """
    Factory function to create a configured summarization precision analyzer.
    
    This provides a clean interface for creating analyzer instances with
    proper dependency injection of the logger.
    """
    return SummarizationPrecisionAnalyzer(logger)