"""
Summarization Statistics Collector

This module handles the sophisticated challenge of collecting and aggregating performance
statistics from multiple specialized agents and processing components to provide comprehensive
insights about the entire multi-agent system's performance and capabilities.

Think of this as the "system health monitor" that understands how all the different
components are performing individually and how well they work together. It provides
the analytics that help optimize the entire pipeline and demonstrate the value of
the architectural sophistication to users and system administrators.

The collector demonstrates how good architecture enables comprehensive observability.
Each component tracks its own performance, but the collector aggregates everything
into meaningful system-wide insights that guide optimization and showcase capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class SummarizationStatisticsCollector:
    """
    Collects and analyzes comprehensive statistics from the entire multi-agent system.
    
    This class solves the observability challenge that sophisticated systems often face:
    how do you get meaningful insights when you have multiple agents each doing complex
    processing with their own performance characteristics? Individual component stats
    are useful, but users and administrators need to understand overall system health.
    
    The collector demonstrates advanced system analytics - it understands what metrics
    matter across different domains and creates unified insights that help optimize
    the entire multi-agent architecture while showcasing its sophistication.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summarization statistics collector.
        
        Args:
            logger: Configured logger for tracking statistics collection operations
        """
        self.logger = logger
        self.logger.info("Summarization Statistics Collector initialized")
        
        # Track collection operations and system insights
        self.collection_stats = {
            'total_collections_performed': 0,
            'system_health_assessments': 0,
            'performance_trend_analyses': 0,
            'optimization_recommendations_generated': 0,
            'cross_agent_correlations_identified': 0,
            'collection_errors': 0
        }
        
        # Store historical data for trend analysis
        self.historical_metrics = {
            'overall_precision_scores': [],
            'processing_times': [],
            'citation_counts': [],
            'enhancement_rates': [],
            'integration_scores': []
        }
    
    def collect_comprehensive_system_statistics(self, gdpr_citations: List[Dict[str, Any]], 
                                              polish_law_citations: List[Dict[str, Any]], 
                                              internal_policy_citations: List[Dict[str, Any]], 
                                              unified_citations: List[Dict[str, Any]], 
                                              precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect comprehensive statistics from all system components and analyze overall performance.
        
        This method represents the culmination of system observability - it gathers performance
        data from every component, analyzes cross-agent patterns, and creates insights that
        help understand how well the entire architectural sophistication is working.
        
        The method demonstrates how good observability enables continuous optimization
        by providing actionable insights into system performance across all domains.
        
        Args:
            gdpr_citations: Citations from GDPR agent for domain analysis
            polish_law_citations: Citations from Polish Law agent for domain analysis
            internal_policy_citations: Citations from Internal Security agent for domain analysis
            unified_citations: Final unified citations for integration analysis
            precision_analysis: Precision analysis results for quality assessment
            
        Returns:
            Comprehensive system statistics with performance insights and recommendations
        """
        self.logger.info("Collecting comprehensive system statistics across all agents and components")
        self.collection_stats['total_collections_performed'] += 1
        
        try:
            # Collect basic system metrics across all components
            basic_metrics = self._collect_basic_system_metrics(
                gdpr_citations, polish_law_citations, internal_policy_citations, unified_citations
            )
            
            # Analyze cross-agent performance patterns
            cross_agent_analysis = self._analyze_cross_agent_performance(
                gdpr_citations, polish_law_citations, internal_policy_citations, precision_analysis
            )
            
            # Calculate system health indicators
            system_health = self._assess_system_health(basic_metrics, cross_agent_analysis, precision_analysis)
            
            # Generate performance trends and insights
            performance_insights = self._generate_performance_insights(basic_metrics, system_health)
            
            # Create optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                system_health, cross_agent_analysis, performance_insights
            )
            
            # Compile comprehensive statistics
            comprehensive_stats = {
                'collection_timestamp': datetime.now().isoformat(),
                'basic_metrics': basic_metrics,
                'cross_agent_analysis': cross_agent_analysis,
                'system_health': system_health,
                'performance_insights': performance_insights,
                'optimization_recommendations': optimization_recommendations,
                'collection_quality': 'comprehensive'
            }
            
            # Update historical tracking for trend analysis
            self._update_historical_metrics(comprehensive_stats)
            
            # Update collection statistics
            self.collection_stats['system_health_assessments'] += 1
            if performance_insights.get('trends_identified', 0) > 0:
                self.collection_stats['performance_trend_analyses'] += 1
            if optimization_recommendations:
                self.collection_stats['optimization_recommendations_generated'] += 1
            
            self.logger.info(f"Comprehensive statistics collection completed: "
                           f"{system_health.get('overall_health_score', 0):.1f}% system health")
            
            return comprehensive_stats
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error collecting comprehensive system statistics: {e}")
            return self._create_fallback_statistics(gdpr_citations, polish_law_citations, 
                                                  internal_policy_citations, unified_citations)
    
    def _collect_basic_system_metrics(self, gdpr_citations: List[Dict[str, Any]], 
                                    polish_law_citations: List[Dict[str, Any]], 
                                    internal_policy_citations: List[Dict[str, Any]], 
                                    unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect fundamental metrics about system performance across all agents.
        
        This method gathers the essential numbers that describe how the system
        performed: citation counts, processing success rates, and domain coverage.
        These metrics form the foundation for more sophisticated analysis.
        """
        # Count citations by source with detailed analysis
        citation_counts = {
            'gdpr': len(gdpr_citations),
            'polish_law': len(polish_law_citations),
            'internal_security': len(internal_policy_citations),
            'total_individual': len(gdpr_citations) + len(polish_law_citations) + len(internal_policy_citations),
            'unified': len(unified_citations)
        }
        
        # Analyze citation source distribution
        source_distribution = self._analyze_citation_source_distribution(citation_counts)
        
        # Calculate processing efficiency metrics
        processing_efficiency = self._calculate_processing_efficiency(citation_counts)
        
        # Analyze domain coverage and balance
        domain_coverage = self._analyze_domain_coverage_metrics(citation_counts)
        
        basic_metrics = {
            'citation_counts': citation_counts,
            'source_distribution': source_distribution,
            'processing_efficiency': processing_efficiency,
            'domain_coverage': domain_coverage,
            'active_domains': self._count_active_domains(citation_counts),
            'system_scale': self._assess_system_scale(citation_counts)
        }
        
        self.logger.debug(f"Basic metrics collected: {citation_counts['total_individual']} individual citations, "
                        f"{citation_counts['unified']} unified, {basic_metrics['active_domains']} active domains")
        
        return basic_metrics
    
    def _analyze_citation_source_distribution(self, citation_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze how citations are distributed across different sources.
        
        This helps understand whether the system is providing balanced coverage
        or if certain domains are dominating the responses.
        """
        total_individual = citation_counts['total_individual']
        
        if total_individual == 0:
            return {'balance_type': 'no_citations', 'distribution_score': 0}
        
        # Calculate percentages for each domain
        distribution_percentages = {}
        for domain in ['gdpr', 'polish_law', 'internal_security']:
            count = citation_counts[domain]
            percentage = (count / total_individual) * 100 if total_individual > 0 else 0
            distribution_percentages[domain] = round(percentage, 1)
        
        # Assess balance quality
        balance_assessment = self._assess_distribution_balance(distribution_percentages)
        
        return {
            'percentages': distribution_percentages,
            'balance_type': balance_assessment['type'],
            'distribution_score': balance_assessment['score'],
            'dominant_domain': balance_assessment.get('dominant_domain'),
            'underrepresented_domains': balance_assessment.get('underrepresented_domains', [])
        }
    
    def _assess_distribution_balance(self, percentages: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess the quality of citation distribution balance across domains.
        
        This analysis helps identify whether queries are naturally favoring
        certain domains or if there might be retrieval or processing issues.
        """
        # Get domains with citations
        active_percentages = {domain: pct for domain, pct in percentages.items() if pct > 0}
        
        if not active_percentages:
            return {'type': 'no_data', 'score': 0}
        
        if len(active_percentages) == 1:
            dominant_domain = list(active_percentages.keys())[0]
            return {
                'type': 'single_domain',
                'score': 100,  # Perfect for single domain
                'dominant_domain': dominant_domain
            }
        
        # Calculate balance metrics for multi-domain responses
        max_percentage = max(active_percentages.values())
        min_percentage = min(active_percentages.values())
        percentage_range = max_percentage - min_percentage
        
        # Identify dominant and underrepresented domains
        dominant_domain = max(active_percentages, key=active_percentages.get) if max_percentage > 60 else None
        underrepresented = [domain for domain, pct in active_percentages.items() if pct < 15]
        
        # Categorize balance type
        if percentage_range <= 20:
            balance_type = 'well_balanced'
            score = 100 - percentage_range  # Higher score for better balance
        elif percentage_range <= 40:
            balance_type = 'moderately_balanced'
            score = 80 - percentage_range
        elif max_percentage >= 80:
            balance_type = 'heavily_skewed'
            score = max(20, 60 - percentage_range)
        else:
            balance_type = 'unbalanced'
            score = max(10, 50 - percentage_range)
        
        return {
            'type': balance_type,
            'score': round(score, 1),
            'dominant_domain': dominant_domain,
            'underrepresented_domains': underrepresented,
            'percentage_range': round(percentage_range, 1)
        }
    
    def _calculate_processing_efficiency(self, citation_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate metrics that indicate how efficiently the system processed the request.
        
        This includes measures like citation preservation rate through unification
        and overall system throughput indicators.
        """
        total_individual = citation_counts['total_individual']
        unified_count = citation_counts['unified']
        
        # Calculate unification efficiency
        if total_individual > 0:
            unification_rate = (unified_count / total_individual) * 100
            
            if unification_rate > 100:
                efficiency_assessment = 'enhanced'  # System added value through integration
            elif unification_rate >= 95:
                efficiency_assessment = 'excellent'
            elif unification_rate >= 85:
                efficiency_assessment = 'good'
            elif unification_rate >= 70:
                efficiency_assessment = 'fair'
            else:
                efficiency_assessment = 'poor'
        else:
            unification_rate = 0
            efficiency_assessment = 'no_data'
        
        # Calculate overall processing scale
        if unified_count >= 10:
            scale_assessment = 'comprehensive'
        elif unified_count >= 6:
            scale_assessment = 'substantial'
        elif unified_count >= 3:
            scale_assessment = 'adequate'
        elif unified_count >= 1:
            scale_assessment = 'minimal'
        else:
            scale_assessment = 'none'
        
        return {
            'unification_rate': round(unification_rate, 1),
            'efficiency_assessment': efficiency_assessment,
            'scale_assessment': scale_assessment,
            'citation_density': round(unified_count / max(1, total_individual), 2)
        }
    
    def _analyze_domain_coverage_metrics(self, citation_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze domain coverage patterns to understand system comprehensiveness.
        
        This helps assess whether the system is providing users with comprehensive
        guidance across all relevant compliance domains.
        """
        # Identify active domains
        active_domains = []
        for domain in ['gdpr', 'polish_law', 'internal_security']:
            if citation_counts[domain] > 0:
                active_domains.append(domain)
        
        # Assess coverage completeness
        coverage_completeness = len(active_domains) / 3.0  # 3 total possible domains
        
        # Determine coverage pattern
        if len(active_domains) == 3:
            coverage_pattern = 'full_spectrum'
        elif len(active_domains) == 2:
            coverage_pattern = 'dual_domain'
        elif len(active_domains) == 1:
            coverage_pattern = 'single_domain'
        else:
            coverage_pattern = 'no_coverage'
        
        # Identify coverage gaps
        inactive_domains = [domain for domain in ['gdpr', 'polish_law', 'internal_security'] 
                          if citation_counts[domain] == 0]
        
        return {
            'active_domains': active_domains,
            'coverage_completeness': round(coverage_completeness, 2),
            'coverage_pattern': coverage_pattern,
            'inactive_domains': inactive_domains,
            'breadth_score': round(coverage_completeness * 100, 1)
        }
    
    def _count_active_domains(self, citation_counts: Dict[str, int]) -> int:
        """Count the number of domains that contributed citations to the response."""
        return sum(1 for domain in ['gdpr', 'polish_law', 'internal_security'] 
                  if citation_counts[domain] > 0)
    
    def _assess_system_scale(self, citation_counts: Dict[str, int]) -> str:
        """
        Assess the overall scale of system processing based on citation volumes.
        
        This helps understand whether the system handled a simple query or
        performed comprehensive analysis across multiple complex domains.
        """
        total_citations = citation_counts['total_individual']
        active_domains = self._count_active_domains(citation_counts)
        
        # Calculate scale score based on citations and domain coverage
        scale_score = total_citations + (active_domains * 2)
        
        if scale_score >= 15:
            return 'comprehensive_analysis'
        elif scale_score >= 10:
            return 'substantial_analysis'
        elif scale_score >= 6:
            return 'moderate_analysis'
        elif scale_score >= 3:
            return 'basic_analysis'
        else:
            return 'minimal_analysis'
    
    def _analyze_cross_agent_performance(self, gdpr_citations: List[Dict[str, Any]], 
                                       polish_law_citations: List[Dict[str, Any]], 
                                       internal_policy_citations: List[Dict[str, Any]], 
                                       precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance patterns across different agents to identify optimization opportunities.
        
        This method looks for correlations and patterns in how different agents
        perform to understand system-wide strengths and improvement opportunities.
        """
        # Analyze individual agent performance
        agent_performance = self._assess_individual_agent_performance(
            gdpr_citations, polish_law_citations, internal_policy_citations, precision_analysis
        )
        
        # Identify performance correlations
        performance_correlations = self._identify_performance_correlations(agent_performance)
        
        # Assess agent synergy quality
        synergy_assessment = self._assess_agent_synergy(agent_performance, precision_analysis)
        
        # Calculate overall agent coordination score
        coordination_score = self._calculate_agent_coordination_score(agent_performance, synergy_assessment)
        
        cross_agent_analysis = {
            'agent_performance': agent_performance,
            'performance_correlations': performance_correlations,
            'synergy_assessment': synergy_assessment,
            'coordination_score': coordination_score,
            'system_balance': self._assess_system_balance(agent_performance)
        }
        
        # Track correlation discoveries
        if performance_correlations.get('significant_correlations', 0) > 0:
            self.collection_stats['cross_agent_correlations_identified'] += 1
        
        self.logger.debug(f"Cross-agent analysis completed: {coordination_score:.1f}% coordination score")
        
        return cross_agent_analysis
    
    def _assess_individual_agent_performance(self, gdpr_citations: List[Dict[str, Any]], 
                                           polish_law_citations: List[Dict[str, Any]], 
                                           internal_policy_citations: List[Dict[str, Any]], 
                                           precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the performance of each individual agent using domain-specific metrics.
        
        This provides agent-specific insights that help understand which components
        are performing well and which might need optimization attention.
        """
        performance = {}
        
        # Get domain analyses from precision analysis
        domain_analyses = precision_analysis.get('domain_analyses', {})
        
        # Analyze GDPR agent performance
        gdpr_analysis = domain_analyses.get('gdpr', {})
        performance['gdpr'] = {
            'citation_count': len(gdpr_citations),
            'precision_score': gdpr_analysis.get('domain_precision_score', 0),
            'enhancement_rate': gdpr_analysis.get('enhancement_rate', 0),
            'quality_assessment': gdpr_analysis.get('quality_assessment', 'unknown'),
            'performance_category': self._categorize_agent_performance(
                len(gdpr_citations), gdpr_analysis.get('domain_precision_score', 0)
            )
        }
        
        # Analyze Polish Law agent performance
        polish_analysis = domain_analyses.get('polish_law', {})
        performance['polish_law'] = {
            'citation_count': len(polish_law_citations),
            'precision_score': polish_analysis.get('domain_precision_score', 0),
            'enhancement_rate': polish_analysis.get('enhancement_rate', 0),
            'quality_assessment': polish_analysis.get('quality_assessment', 'unknown'),
            'performance_category': self._categorize_agent_performance(
                len(polish_law_citations), polish_analysis.get('domain_precision_score', 0)
            )
        }
        
        # Analyze Internal Security agent performance
        security_analysis = domain_analyses.get('security', {})
        performance['security'] = {
            'citation_count': len(internal_policy_citations),
            'precision_score': security_analysis.get('domain_precision_score', 0),
            'enhancement_rate': security_analysis.get('enhancement_rate', 0),
            'quality_assessment': security_analysis.get('quality_assessment', 'unknown'),
            'performance_category': self._categorize_agent_performance(
                len(internal_policy_citations), security_analysis.get('domain_precision_score', 0)
            )
        }
        
        return performance
    
    def _categorize_agent_performance(self, citation_count: int, precision_score: float) -> str:
        """
        Categorize an agent's performance based on citation volume and precision quality.
        
        This provides a simple categorization that helps quickly identify which
        agents are performing optimally and which need attention.
        """
        # Weight both quantity and quality
        if citation_count >= 3 and precision_score >= 85:
            return 'excellent'
        elif citation_count >= 2 and precision_score >= 75:
            return 'very_good'
        elif citation_count >= 1 and precision_score >= 65:
            return 'good'
        elif citation_count >= 1 and precision_score >= 50:
            return 'adequate'
        elif citation_count >= 1:
            return 'needs_improvement'
        else:
            return 'inactive'
    
    def _identify_performance_correlations(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify correlations in agent performance that might indicate system-wide patterns.
        
        This analysis helps understand whether agent performance issues are isolated
        or reflect broader system patterns that need systematic attention.
        """
        # Collect performance metrics for correlation analysis
        precision_scores = []
        enhancement_rates = []
        citation_counts = []
        
        for agent_name, metrics in agent_performance.items():
            if metrics['citation_count'] > 0:  # Only include active agents
                precision_scores.append(metrics['precision_score'])
                enhancement_rates.append(metrics['enhancement_rate'])
                citation_counts.append(metrics['citation_count'])
        
        correlations = {
            'precision_consistency': self._calculate_consistency(precision_scores),
            'enhancement_consistency': self._calculate_consistency(enhancement_rates),
            'volume_balance': self._calculate_consistency(citation_counts),
            'significant_correlations': 0
        }
        
        # Identify significant patterns
        if correlations['precision_consistency'] > 80:
            correlations['significant_correlations'] += 1
            correlations['precision_pattern'] = 'highly_consistent'
        elif correlations['precision_consistency'] < 50:
            correlations['precision_pattern'] = 'highly_variable'
        
        if correlations['enhancement_consistency'] > 75:
            correlations['significant_correlations'] += 1
            correlations['enhancement_pattern'] = 'consistent_enhancement'
        
        return correlations
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """
        Calculate a consistency score for a list of values.
        
        This measures how similar the values are, with higher scores indicating
        more consistent performance across agents.
        """
        if len(values) <= 1:
            return 100  # Single value is perfectly consistent
        
        if not values:
            return 0
        
        # Calculate coefficient of variation (inverse of consistency)
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return 100 if all(v == 0 for v in values) else 0
        
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / mean_val
        
        # Convert to consistency score (0-100)
        consistency_score = max(0, 100 - (coefficient_of_variation * 100))
        return round(consistency_score, 1)
    
    def _assess_agent_synergy(self, agent_performance: Dict[str, Any], 
                            precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how well agents work together to create integrated value.
        
        This measures whether the multi-agent approach creates synergistic value
        or if agents are working in isolation without meaningful integration.
        """
        # Get integration quality metrics
        integration_analysis = precision_analysis.get('integration_analysis', {})
        integration_score = integration_analysis.get('integration_score', 0)
        
        # Count active agents
        active_agents = sum(1 for metrics in agent_performance.values() 
                          if metrics['citation_count'] > 0)
        
        # Assess synergy based on integration success and agent participation
        if active_agents >= 3 and integration_score >= 85:
            synergy_level = 'exceptional'
            synergy_score = 95
        elif active_agents >= 2 and integration_score >= 75:
            synergy_level = 'strong'
            synergy_score = 85
        elif active_agents >= 2 and integration_score >= 60:
            synergy_level = 'moderate'
            synergy_score = 70
        elif active_agents >= 2:
            synergy_level = 'weak'
            synergy_score = 50
        else:
            synergy_level = 'none'
            synergy_score = 0
        
        return {
            'synergy_level': synergy_level,
            'synergy_score': synergy_score,
            'active_agents': active_agents,
            'integration_effectiveness': integration_score,
            'collaboration_quality': self._assess_collaboration_quality(agent_performance, integration_score)
        }
    
    def _assess_collaboration_quality(self, agent_performance: Dict[str, Any], 
                                    integration_score: float) -> str:
        """
        Assess the quality of collaboration between agents.
        
        This determines whether agents are effectively complementing each other
        or working in isolation without meaningful coordination.
        """
        active_count = sum(1 for metrics in agent_performance.values() 
                         if metrics['citation_count'] > 0)
        
        # Get average performance quality
        active_performances = [metrics for metrics in agent_performance.values() 
                             if metrics['citation_count'] > 0]
        
        if not active_performances:
            return 'no_collaboration'
        
        avg_precision = sum(perf['precision_score'] for perf in active_performances) / len(active_performances)
        
        # Assess collaboration based on agent participation and integration success
        if active_count >= 3 and integration_score >= 80 and avg_precision >= 75:
            return 'excellent_collaboration'
        elif active_count >= 2 and integration_score >= 70:
            return 'good_collaboration'
        elif active_count >= 2 and integration_score >= 50:
            return 'fair_collaboration'
        elif active_count >= 2:
            return 'poor_collaboration'
        else:
            return 'isolated_processing'
    
    def _calculate_agent_coordination_score(self, agent_performance: Dict[str, Any], 
                                          synergy_assessment: Dict[str, Any]) -> float:
        """
        Calculate an overall score representing how well agents coordinate.
        
        This provides a single metric that captures the effectiveness of the
        multi-agent architecture in creating integrated, valuable outputs.
        """
        # Base score from synergy assessment
        base_score = synergy_assessment['synergy_score']
        
        # Adjust based on individual agent performance consistency
        active_agents = [metrics for metrics in agent_performance.values() 
                        if metrics['citation_count'] > 0]
        
        if active_agents:
            # Bonus for consistent high performance across agents
            precision_scores = [agent['precision_score'] for agent in active_agents]
            min_precision = min(precision_scores)
            avg_precision = sum(precision_scores) / len(precision_scores)
            
            # Penalty for large performance gaps
            precision_range = max(precision_scores) - min_precision
            consistency_bonus = max(0, 20 - precision_range)
            
            # Bonus for high minimum performance
            quality_bonus = max(0, (min_precision - 50) / 5)
            
            coordination_score = min(100, base_score + consistency_bonus + quality_bonus)
        else:
            coordination_score = 0
        
        return round(coordination_score, 1)
    
    def _assess_system_balance(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess whether the system provides balanced performance across all agents.
        
        This helps identify if certain agents consistently outperform others,
        which might indicate optimization opportunities or architectural issues.
        """
        # Collect performance categories
        performance_categories = {}
        for agent, metrics in agent_performance.items():
            category = metrics['performance_category']
            performance_categories[agent] = category
        
        # Identify balance patterns
        unique_categories = set(performance_categories.values())
        
        if len(unique_categories) == 1:
            balance_type = 'uniform_performance'
        elif 'excellent' in unique_categories and 'needs_improvement' in unique_categories:
            balance_type = 'highly_unbalanced'
        elif len(unique_categories) <= 2:
            balance_type = 'moderately_balanced'
        else:
            balance_type = 'variable_performance'
        
        # Identify strongest and weakest performers
        active_agents = {agent: metrics for agent, metrics in agent_performance.items() 
                        if metrics['citation_count'] > 0}
        
        if active_agents:
            strongest_agent = max(active_agents.keys(), 
                                key=lambda a: active_agents[a]['precision_score'])
            weakest_agent = min(active_agents.keys(), 
                              key=lambda a: active_agents[a]['precision_score'])
        else:
            strongest_agent = None
            weakest_agent = None
        
        return {
            'balance_type': balance_type,
            'performance_categories': performance_categories,
            'strongest_performer': strongest_agent,
            'weakest_performer': weakest_agent,
            'needs_attention': [agent for agent, cat in performance_categories.items() 
                              if cat in ['needs_improvement', 'inactive']]
        }
    
    def _assess_system_health(self, basic_metrics: Dict[str, Any], 
                            cross_agent_analysis: Dict[str, Any], 
                            precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall system health by combining metrics from all components.
        
        This provides a comprehensive view of how well the entire multi-agent
        system is functioning across all dimensions of performance.
        """
        # Extract key health indicators
        citation_health = self._assess_citation_health(basic_metrics)
        precision_health = self._assess_precision_health(precision_analysis)
        integration_health = self._assess_integration_health(precision_analysis)
        coordination_health = self._assess_coordination_health(cross_agent_analysis)
        
        # Calculate weighted overall health score
        health_components = {
            'citation_processing': citation_health,
            'precision_quality': precision_health,
            'domain_integration': integration_health,
            'agent_coordination': coordination_health
        }
        
        # Weight different aspects of system health
        weights = {
            'citation_processing': 0.25,
            'precision_quality': 0.35,
            'domain_integration': 0.25,
            'agent_coordination': 0.15
        }
        
        overall_score = sum(health_components[component] * weights[component] 
                          for component in health_components)
        
        # Determine overall health category
        health_category = self._categorize_system_health(overall_score)
        
        # Identify critical issues
        critical_issues = self._identify_critical_health_issues(health_components)
        
        system_health = {
            'overall_health_score': round(overall_score, 1),
            'health_category': health_category,
            'health_components': health_components,
            'critical_issues': critical_issues,
            'health_trend': self._assess_health_trend(),
            'system_stability': self._assess_system_stability(health_components)
        }
        
        return system_health
    
    def _assess_citation_health(self, basic_metrics: Dict[str, Any]) -> float:
        """
        Assess the health of citation processing and generation.
        
        This measures how effectively the system is retrieving and processing
        citations across all domains.
        """
        citation_counts = basic_metrics['citation_counts']
        processing_efficiency = basic_metrics['processing_efficiency']
        domain_coverage = basic_metrics['domain_coverage']
        
        # Base score from citation volume
        total_citations = citation_counts['unified']
        if total_citations >= 8:
            volume_score = 100
        elif total_citations >= 5:
            volume_score = 80
        elif total_citations >= 3:
            volume_score = 60
        elif total_citations >= 1:
            volume_score = 40
        else:
            volume_score = 0
        
        # Adjust for processing efficiency
        efficiency_score = processing_efficiency.get('unification_rate', 0)
        
        # Adjust for domain coverage
        coverage_score = domain_coverage.get('breadth_score', 0)
        
        # Calculate weighted citation health
        citation_health = (volume_score * 0.4 + efficiency_score * 0.3 + coverage_score * 0.3)
        
        return round(citation_health, 1)
    
    def _assess_precision_health(self, precision_analysis: Dict[str, Any]) -> float:
        """
        Assess the health of precision and quality across the system.
        
        This measures how well the system achieves high-quality, detailed
        citations that demonstrate the value of the sophisticated processing.
        """
        system_metrics = precision_analysis.get('system_metrics', {})
        overall_precision = system_metrics.get('overall_precision_score', 0)
        enhancement_rate = system_metrics.get('system_enhancement_rate', 0)
        
        # Weight precision score and enhancement rate
        precision_health = (overall_precision * 0.7 + enhancement_rate * 0.3)
        
        return round(precision_health, 1)
    
    def _assess_integration_health(self, precision_analysis: Dict[str, Any]) -> float:
        """
        Assess the health of multi-domain integration capabilities.
        
        This measures how well the system unifies work from different agents
        into coherent, valuable integrated outputs.
        """
        integration_analysis = precision_analysis.get('integration_analysis', {})
        integration_score = integration_analysis.get('integration_score', 0)
        
        return integration_score
    
    def _assess_coordination_health(self, cross_agent_analysis: Dict[str, Any]) -> float:
        """
        Assess the health of agent coordination and collaboration.
        
        This measures how well different agents work together to create
        synergistic value greater than the sum of individual contributions.
        """
        coordination_score = cross_agent_analysis.get('coordination_score', 0)
        
        return coordination_score
    
    def _categorize_system_health(self, overall_score: float) -> str:
        """
        Categorize overall system health based on the comprehensive score.
        
        This provides a human-readable assessment of system performance
        that helps quickly understand operational status.
        """
        if overall_score >= 90:
            return 'excellent'
        elif overall_score >= 80:
            return 'very_good'
        elif overall_score >= 70:
            return 'good'
        elif overall_score >= 60:
            return 'fair'
        elif overall_score >= 50:
            return 'poor'
        else:
            return 'critical'
    
    def _identify_critical_health_issues(self, health_components: Dict[str, float]) -> List[str]:
        """
        Identify critical issues that need immediate attention.
        
        This helps prioritize optimization efforts by highlighting the most
        serious performance problems affecting system effectiveness.
        """
        critical_issues = []
        
        for component, score in health_components.items():
            if score < 50:
                if component == 'citation_processing':
                    critical_issues.append("Citation processing is failing - check agent retrieval and vector stores")
                elif component == 'precision_quality':
                    critical_issues.append("Precision quality is poor - review metadata processing and enhancement")
                elif component == 'domain_integration':
                    critical_issues.append("Domain integration is failing - check unification and citation management")
                elif component == 'agent_coordination':
                    critical_issues.append("Agent coordination is poor - review multi-agent workflow and communication")
        
        # Add system-wide critical issues
        overall_avg = sum(health_components.values()) / len(health_components)
        if overall_avg < 40:
            critical_issues.append("System-wide performance is critically low - comprehensive review needed")
        
        return critical_issues
    
    def _assess_health_trend(self) -> str:
        """
        Assess whether system health is improving, declining, or stable over time.
        
        This requires historical data and helps understand whether optimization
        efforts are having positive effects on system performance.
        """
        if len(self.historical_metrics['overall_precision_scores']) < 3:
            return 'insufficient_data'
        
        recent_scores = self.historical_metrics['overall_precision_scores'][-3:]
        
        # Simple trend analysis
        if recent_scores[-1] > recent_scores[0] + 5:
            return 'improving'
        elif recent_scores[-1] < recent_scores[0] - 5:
            return 'declining'
        else:
            return 'stable'
    
    def _assess_system_stability(self, health_components: Dict[str, float]) -> str:
        """
        Assess the stability and reliability of system performance.
        
        This measures whether the system provides consistent performance
        or has high variability that might indicate reliability issues.
        """
        # Check for consistent performance across components
        min_score = min(health_components.values())
        max_score = max(health_components.values())
        score_range = max_score - min_score
        
        if score_range <= 15 and min_score >= 75:
            return 'highly_stable'
        elif score_range <= 25 and min_score >= 60:
            return 'stable'
        elif score_range <= 40:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _generate_performance_insights(self, basic_metrics: Dict[str, Any], 
                                     system_health: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable insights about system performance patterns.
        
        These insights help understand what's working well and what needs
        optimization attention for maximum system effectiveness.
        """
        insights = {
            'key_strengths': [],
            'improvement_areas': [],
            'performance_patterns': [],
            'trends_identified': 0
        }
        
        # Identify key strengths
        health_score = system_health['overall_health_score']
        if health_score >= 85:
            insights['key_strengths'].append("Exceptional overall system performance across all components")
        
        if basic_metrics['domain_coverage']['coverage_completeness'] >= 0.8:
            insights['key_strengths'].append("Excellent multi-domain coverage providing comprehensive guidance")
        
        if basic_metrics['processing_efficiency']['unification_rate'] >= 90:
            insights['key_strengths'].append("Highly efficient citation processing and unification")
        
        # Identify improvement areas
        health_components = system_health['health_components']
        for component, score in health_components.items():
            if score < 70:
                if component == 'citation_processing':
                    insights['improvement_areas'].append("Citation processing efficiency needs optimization")
                elif component == 'precision_quality':
                    insights['improvement_areas'].append("Precision quality could be enhanced through better metadata processing")
                elif component == 'domain_integration':
                    insights['improvement_areas'].append("Domain integration processes need refinement")
                elif component == 'agent_coordination':
                    insights['improvement_areas'].append("Agent coordination and collaboration could be improved")
        
        # Identify performance patterns
        coverage_pattern = basic_metrics['domain_coverage']['coverage_pattern']
        if coverage_pattern == 'full_spectrum':
            insights['performance_patterns'].append("System consistently provides full-spectrum compliance coverage")
        elif coverage_pattern == 'single_domain':
            insights['performance_patterns'].append("System tends toward single-domain responses - consider query analysis")
        
        # Track trend identification
        health_trend = system_health.get('health_trend', 'unknown')
        if health_trend in ['improving', 'declining']:
            insights['trends_identified'] += 1
            insights['performance_patterns'].append(f"System health trend: {health_trend}")
        
        return insights
    
    def _generate_optimization_recommendations(self, system_health: Dict[str, Any], 
                                             cross_agent_analysis: Dict[str, Any], 
                                             performance_insights: Dict[str, Any]) -> List[str]:
        """
        Generate specific optimization recommendations based on system analysis.
        
        These recommendations provide actionable guidance for improving system
        performance and maximizing the value of the architectural sophistication.
        """
        recommendations = []
        
        # Health-based recommendations
        critical_issues = system_health.get('critical_issues', [])
        if critical_issues:
            recommendations.append(f"CRITICAL: Address {len(critical_issues)} critical system issues immediately")
        
        health_score = system_health['overall_health_score']
        if health_score < 70:
            recommendations.append("Overall system health needs improvement - prioritize component optimization")
        
        # Agent coordination recommendations
        coordination_score = cross_agent_analysis.get('coordination_score', 0)
        if coordination_score < 70:
            recommendations.append("Improve agent coordination - review workflow integration and communication")
        
        system_balance = cross_agent_analysis.get('system_balance', {})
        needs_attention = system_balance.get('needs_attention', [])
        if needs_attention:
            agent_list = ', '.join(needs_attention)
            recommendations.append(f"Focus optimization on underperforming agents: {agent_list}")
        
        # Performance pattern recommendations
        improvement_areas = performance_insights.get('improvement_areas', [])
        for area in improvement_areas[:3]:  # Top 3 improvement areas
            recommendations.append(f"Optimize: {area}")
        
        # Trend-based recommendations
        health_trend = system_health.get('health_trend', 'unknown')
        if health_trend == 'declining':
            recommendations.append("ATTENTION: System performance is declining - investigate recent changes")
        elif health_trend == 'improving':
            recommendations.append("Continue current optimization efforts - positive trend detected")
        
        # Specific technical recommendations
        stability = system_health.get('system_stability', 'unknown')
        if stability == 'unstable':
            recommendations.append("Improve system stability - address performance variability across components")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _update_historical_metrics(self, comprehensive_stats: Dict[str, Any]) -> None:
        """
        Update historical metrics for trend analysis and performance monitoring.
        
        This maintains a rolling window of performance data that enables
        trend detection and long-term system optimization insights.
        """
        # Extract key metrics for historical tracking
        system_metrics = comprehensive_stats.get('basic_metrics', {}).get('citation_counts', {})
        system_health = comprehensive_stats.get('system_health', {})
        precision_analysis = comprehensive_stats.get('basic_metrics', {})  # Note: this might need adjustment based on actual structure
        
        # Update historical data (maintain last 20 data points)
        max_history = 20
        
        # Add new data points
        if system_health.get('overall_health_score'):
            self.historical_metrics['overall_precision_scores'].append(system_health['overall_health_score'])
            if len(self.historical_metrics['overall_precision_scores']) > max_history:
                self.historical_metrics['overall_precision_scores'].pop(0)
        
        citation_count = system_metrics.get('unified', 0)
        if citation_count > 0:
            self.historical_metrics['citation_counts'].append(citation_count)
            if len(self.historical_metrics['citation_counts']) > max_history:
                self.historical_metrics['citation_counts'].pop(0)
        
        # Note: Other metrics would be added here based on available data structure
    
    def _create_fallback_statistics(self, gdpr_citations: List[Dict[str, Any]], 
                                  polish_law_citations: List[Dict[str, Any]], 
                                  internal_policy_citations: List[Dict[str, Any]], 
                                  unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create fallback statistics when comprehensive collection fails.
        
        This ensures the system continues to provide some statistical insights
        even when the sophisticated analysis encounters errors.
        """
        return {
            'collection_timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'citation_counts': {
                    'gdpr': len(gdpr_citations),
                    'polish_law': len(polish_law_citations),
                    'internal_security': len(internal_policy_citations),
                    'unified': len(unified_citations)
                }
            },
            'system_health': {
                'overall_health_score': 0,
                'health_category': 'error'
            },
            'collection_quality': 'fallback'
        }
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the statistics collection process itself.
        
        Returns:
            Dictionary containing collection operation metrics and performance data
        """
        stats = dict(self.collection_stats)
        
        # Calculate collection success rate
        if stats['total_collections_performed'] > 0:
            success_rate = ((stats['total_collections_performed'] - stats['collection_errors']) / 
                          stats['total_collections_performed']) * 100
            stats['collection_success_rate'] = round(success_rate, 1)
        
        # Add historical metrics summary
        stats['historical_data_points'] = {
            'precision_scores': len(self.historical_metrics['overall_precision_scores']),
            'citation_counts': len(self.historical_metrics['citation_counts'])
        }
        
        return stats
    
    def log_statistics_collection_summary(self) -> None:
        """
        Log a comprehensive summary of all statistics collection operations.
        
        This provides insights into the observability system's own performance
        and helps ensure comprehensive system monitoring capabilities.
        """
        stats = self.get_collection_statistics()
        
        self.logger.info("=== SYSTEM STATISTICS COLLECTION SUMMARY ===")
        self.logger.info(f"Total collections performed: {stats['total_collections_performed']}")
        self.logger.info(f"Collection success rate: {stats.get('collection_success_rate', 0)}%")
        self.logger.info(f"System health assessments: {stats['system_health_assessments']}")
        self.logger.info(f"Performance trend analyses: {stats['performance_trend_analyses']}")
        self.logger.info(f"Optimization recommendations generated: {stats['optimization_recommendations_generated']}")
        self.logger.info(f"Cross-agent correlations identified: {stats['cross_agent_correlations_identified']}")
        self.logger.info(f"Collection errors: {stats['collection_errors']}")
        
        # Log historical data status
        historical_status = stats['historical_data_points']
        self.logger.info(f"Historical trend data: {historical_status['precision_scores']} precision points, "
                        f"{historical_status['citation_counts']} citation counts")


def create_summarization_statistics_collector(logger: logging.Logger) -> SummarizationStatisticsCollector:
    """
    Factory function to create a configured summarization statistics collector.
    
    This provides a clean interface for creating collector instances with
    proper dependency injection of the logger.
    """
    return SummarizationStatisticsCollector(logger)