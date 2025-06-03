"""
Enhanced Summarization Agent - Sophisticated Orchestrator

This module represents the culmination of the summarization agent refactoring, combining all the
specialized components into a clean, maintainable orchestrator. Like your enhanced GDPR, Polish Law,
and Internal Security agents, this agent demonstrates how architectural sophistication creates 
more reliable, maintainable, and powerful systems for multi-domain integration.

The agent follows the same design patterns as your other refactored agents:
- Single Responsibility Principle with focused components
- Dependency injection for clean interfaces
- Comprehensive error handling and graceful degradation
- Detailed logging and statistics for monitoring
- Factory pattern for clean instantiation

This refactored agent integrates seamlessly with your enhanced multi-agent pipeline,
demonstrating how consistent architectural patterns create system-wide reliability and excellence.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from .summarization_citation_manager import SummarizationCitationManager, create_summarization_citation_manager
from .summarization_precision_analyzer import SummarizationPrecisionAnalyzer, create_summarization_precision_analyzer
from .summarization_formatter import SummarizationFormatter, create_summarization_formatter
from .summarization_statistics_collector import SummarizationStatisticsCollector, create_summarization_statistics_collector
from .summarization_response_builder import SummarizationResponseBuilder, create_summarization_response_builder


class SummarizationAgent:
    """
    Enhanced Summarization Agent with sophisticated modular architecture.
    
    This class represents the complete solution for multi-domain citation integration and 
    comprehensive response generation using the same architectural excellence demonstrated 
    in your other agent refactors. Instead of a monolithic agent, we now have a sophisticated 
    orchestrator that coordinates specialized components through clean interfaces.
    
    The agent demonstrates how your architectural patterns create:
    - More reliable processing through specialized components
    - Better maintainability through single responsibility modules
    - Enhanced functionality through component synergy
    - Easier testing through dependency injection
    - Improved monitoring through comprehensive statistics
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the enhanced summarization agent with sophisticated component architecture.
        
        Args:
            logger: Configured logger for comprehensive operation tracking
        """
        self.logger = logger
        
        # Initialize all specialized components using dependency injection
        self._initialize_components()
        
        # Track session statistics across all operations
        self.session_stats = {
            'total_queries_processed': 0,
            'successful_integrations': 0,
            'component_errors': 0,
            'fallback_operations': 0,
            'average_processing_time': 0.0,
            'total_unified_citations': 0,
            'multi_domain_responses': 0,
            'single_domain_responses': 0
        }
        
        self.logger.info("Enhanced Summarization Agent initialized with sophisticated component architecture")
    
    def _initialize_components(self) -> None:
        """
        Initialize all specialized components using factory functions and dependency injection.
        
        This method demonstrates the same architectural patterns as your other enhanced agents,
        creating focused components that work together through clean interfaces for multi-domain integration.
        """
        self.logger.info("Initializing sophisticated summarization components...")
        
        # Initialize all components with proper dependency injection
        self.citation_manager = create_summarization_citation_manager(self.logger)
        self.precision_analyzer = create_summarization_precision_analyzer(self.logger)
        self.formatter = create_summarization_formatter(self.logger)
        self.statistics_collector = create_summarization_statistics_collector(self.logger)
        self.response_builder = create_summarization_response_builder(self.logger)
        
        self.logger.info("All summarization components initialized successfully")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method demonstrating sophisticated component orchestration.
        
        This method represents the enhanced approach to multi-domain summarization, orchestrating
        all the specialized components to create the most comprehensive and precise response
        possible. The method demonstrates how architectural sophistication enables reliable,
        maintainable operations that showcase the value of the entire multi-agent system.
        
        Args:
            state: Processing state dictionary from the multi-agent workflow
            
        Returns:
            Updated state with comprehensive multi-domain summary and statistics
        """
        session_start = time.time()
        user_query = state["user_query"]
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED SUMMARIZATION AGENT SESSION WITH COMPONENT ARCHITECTURE")
        self.logger.info(f"User query: {user_query}")
        self.logger.info("Using sophisticated component orchestration for maximum integration precision")
        self.logger.info("=" * 80)
        
        print("\n📊 [STEP 4/4] ENHANCED SUMMARIZATION AGENT: Sophisticated multi-domain integration...")
        
        self.session_stats['total_queries_processed'] += 1
        
        try:
            # Stage 1: Multi-Domain Citation Unification
            unified_citations, formatted_citations = self._execute_citation_unification_stage(state)
            
            # Stage 2: Comprehensive Precision Analysis
            precision_analysis = self._execute_precision_analysis_stage(state, unified_citations)
            
            # Stage 3: Sophisticated Response Building
            action_plan = self._execute_response_building_stage(user_query, formatted_citations, precision_analysis)
            
            # Stage 4: Professional Response Formatting
            formatted_response = self._execute_response_formatting_stage(action_plan, unified_citations, precision_analysis)
            
            # Stage 5: Comprehensive Statistics Collection
            comprehensive_stats = self._execute_statistics_collection_stage(state, unified_citations, precision_analysis)
            
            # Stage 6: Final Integration and State Update
            self._execute_completion_stage(state, formatted_response, comprehensive_stats, session_start)
            
            return state
            
        except Exception as e:
            return self._handle_processing_error(state, e, session_start)
    
    def _execute_citation_unification_stage(self, state: Dict[str, Any]) -> tuple:
        """
        Execute multi-domain citation unification using the sophisticated citation manager.
        
        This stage demonstrates how component specialization enables reliable multi-domain
        integration. The citation manager handles all the complexity of unifying citations
        from three different domains while preserving their individual precision.
        """
        self.logger.info("STAGE 1: Multi-Domain Citation Unification with Precision Preservation")
        
        try:
            # Extract citations from all domains
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Use sophisticated citation manager for unification
            unified_citations, formatted_citations = self.citation_manager.create_unified_citation_system(
                gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Log unification success with domain analysis
            domain_count = sum(1 for citations in [gdpr_citations, polish_law_citations, internal_policy_citations] 
                             if citations)
            
            self.logger.info(f"✅ Citation unification successful: {len(unified_citations)} citations from {domain_count} domains")
            
            # Update session statistics
            self.session_stats['total_unified_citations'] += len(unified_citations)
            if domain_count > 1:
                self.session_stats['multi_domain_responses'] += 1
            else:
                self.session_stats['single_domain_responses'] += 1
            
            return unified_citations, formatted_citations
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in citation unification stage: {e}")
            # Return empty results to continue processing
            return [], "No citations available due to unification error"
    
    def _execute_precision_analysis_stage(self, state: Dict[str, Any], unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute comprehensive precision analysis using the sophisticated precision analyzer.
        
        This stage demonstrates the power of cross-domain analytics - the precision analyzer
        understands quality standards across different domains and creates unified metrics
        that help users understand the sophistication they're receiving.
        """
        self.logger.info("STAGE 2: Comprehensive Cross-Domain Precision Analysis")
        
        try:
            # Extract domain-specific citations for analysis
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Execute sophisticated precision analysis
            precision_analysis = self.precision_analyzer.analyze_multi_domain_precision(
                unified_citations, gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Log analysis results
            system_metrics = precision_analysis.get('system_metrics', {})
            overall_score = system_metrics.get('overall_precision_score', 0)
            
            self.logger.info(f"✅ Precision analysis completed: {overall_score:.1f}% overall system precision")
            
            return precision_analysis
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in precision analysis stage: {e}")
            # Return basic analysis to continue processing
            return {
                'system_metrics': {'overall_precision_score': 0},
                'integration_analysis': {'integration_score': 0},
                'insights': ['Precision analysis failed - using fallback assessment']
            }
    
    def _execute_response_building_stage(self, user_query: str, formatted_citations: str, 
                                       precision_analysis: Dict[str, Any]) -> str:
        """
        Execute sophisticated response building using the enhanced response builder.
        
        This stage demonstrates how AI integration can be enhanced by sophisticated
        components. The response builder uses structured prompts and precision context
        to guide the LLM toward creating responses that showcase system capabilities.
        """
        self.logger.info("STAGE 3: Sophisticated Response Building with AI Integration")
        
        try:
            # Analyze response characteristics for optimal building strategy
            response_characteristics = self._analyze_response_context(precision_analysis)
            
            # Use sophisticated response builder
            action_plan = self.response_builder.build_comprehensive_action_plan(
                user_query, formatted_citations, precision_analysis, response_characteristics
            )
            
            self.logger.info("✅ Sophisticated response building completed successfully")
            
            return action_plan
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in response building stage: {e}")
            # Return basic response to continue processing
            return f"Based on available compliance documentation for '{user_query}', please review applicable requirements and implement necessary compliance measures. Refer to the provided citations for specific guidance."
    
    def _execute_response_formatting_stage(self, action_plan: str, unified_citations: List[Dict[str, Any]], 
                                         precision_analysis: Dict[str, Any]) -> str:
        """
        Execute professional response formatting using the sophisticated formatter.
        
        This stage demonstrates how presentation sophistication makes technical excellence
        visible to users. The formatter creates responses that showcase the system's
        capabilities while maintaining professional readability and actionable guidance.
        """
        self.logger.info("STAGE 4: Professional Response Formatting with Sophistication Showcase")
        
        try:
            # Use sophisticated formatter for comprehensive presentation
            formatted_response = self.formatter.format_comprehensive_response(
                action_plan, unified_citations, precision_analysis
            )
            
            self.logger.info("✅ Professional response formatting completed")
            
            return formatted_response
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in response formatting stage: {e}")
            # Return basic formatting to continue processing
            return self._create_basic_formatted_response(action_plan, unified_citations)
    
    def _execute_statistics_collection_stage(self, state: Dict[str, Any], unified_citations: List[Dict[str, Any]], 
                                           precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive statistics collection using the sophisticated collector.
        
        This stage demonstrates how sophisticated observability enables continuous
        optimization. The statistics collector provides insights into system performance
        across all components and domains for optimization guidance.
        """
        self.logger.info("STAGE 5: Comprehensive System Statistics Collection")
        
        try:
            # Extract domain-specific citations for comprehensive analysis
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Execute comprehensive statistics collection
            comprehensive_stats = self.statistics_collector.collect_comprehensive_system_statistics(
                gdpr_citations, polish_law_citations, internal_policy_citations, 
                unified_citations, precision_analysis
            )
            
            # Log statistics collection success
            system_health = comprehensive_stats.get('system_health', {})
            health_score = system_health.get('overall_health_score', 0)
            
            self.logger.info(f"✅ Statistics collection completed: {health_score:.1f}% system health")
            
            return comprehensive_stats
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in statistics collection stage: {e}")
            # Return basic statistics to continue processing
            return {
                'basic_metrics': {'citation_counts': {'unified': len(unified_citations)}},
                'system_health': {'overall_health_score': 0}
            }
    
    def _execute_completion_stage(self, state: Dict[str, Any], formatted_response: str, 
                                comprehensive_stats: Dict[str, Any], session_start: float) -> None:
        """
        Execute completion stage with comprehensive statistics and state update.
        
        This stage demonstrates how component architecture enables comprehensive
        monitoring and statistics collection across all system operations.
        """
        self.logger.info("STAGE 6: Session Completion with Component Statistics")
        
        # Update processing statistics
        session_time = time.time() - session_start
        self.session_stats['average_processing_time'] = \
            (self.session_stats['average_processing_time'] * (self.session_stats['total_queries_processed'] - 1) + session_time) / \
            self.session_stats['total_queries_processed']
        
        # Mark as successful integration
        self.session_stats['successful_integrations'] += 1
        
        # Create final summary structure with comprehensive information
        summary = {
            "action_plan": formatted_response,
            **comprehensive_stats.get('basic_metrics', {}),
            **comprehensive_stats.get('system_health', {}),
            'component_architecture': 'enhanced',
            'processing_time_seconds': round(session_time, 3)
        }
        
        # Update state with comprehensive results
        state["summary"] = summary
        
        # Log comprehensive session completion
        self._log_session_completion(comprehensive_stats, session_time)
        
        citation_count = comprehensive_stats.get('basic_metrics', {}).get('citation_counts', {}).get('unified', 0)
        print(f"✅ Completed: {citation_count} sophisticated citations integrated from multiple domains")
        print(f"⏱️  Processing time: {session_time:.3f} seconds")
    
    def _analyze_response_context(self, precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze response context to determine optimal processing strategy.
        
        This method examines the precision analysis results to understand what
        level of sophistication should be applied in response building and formatting.
        """
        system_metrics = precision_analysis.get('system_metrics', {})
        integration_analysis = precision_analysis.get('integration_analysis', {})
        
        # Extract key indicators for response strategy
        overall_precision = system_metrics.get('overall_precision_score', 0)
        integration_score = integration_analysis.get('integration_score', 0)
        total_citations = system_metrics.get('total_citations', 0)
        active_domains = system_metrics.get('active_domains', 0)
        
        # Determine precision level
        if overall_precision >= 85:
            precision_level = 'exceptional'
        elif overall_precision >= 75:
            precision_level = 'high'
        elif overall_precision >= 65:
            precision_level = 'good'
        else:
            precision_level = 'moderate'
        
        # Generate sophistication highlights
        highlights = []
        if overall_precision >= 80:
            highlights.append(f"High-precision analysis achieving {overall_precision:.1f}% accuracy")
        if integration_score >= 80:
            highlights.append(f"Excellent multi-domain integration with {integration_score:.1f}% effectiveness")
        if active_domains >= 3:
            highlights.append("Comprehensive coverage across all compliance domains")
        if total_citations >= 6:
            highlights.append(f"Extensive source analysis with {total_citations} authoritative references")
        
        return {
            'precision_level': precision_level,
            'citation_count': total_citations,
            'domain_count': active_domains,
            'sophistication_highlights': highlights,
            'integration_quality': integration_score
        }
    
    def _create_basic_formatted_response(self, action_plan: str, unified_citations: List[Dict[str, Any]]) -> str:
        """
        Create basic formatted response when sophisticated formatting fails.
        
        This ensures users receive professional output even when the advanced
        formatting features encounter errors, demonstrating graceful degradation.
        """
        basic_parts = []
        
        # Add the action plan
        basic_parts.append(action_plan)
        
        # Add basic citation display
        if unified_citations:
            basic_parts.append("\n**AUTHORITATIVE SOURCE CITATIONS:**\n")
            
            # Group by domain for basic organization
            domain_groups = {}
            for citation in unified_citations:
                domain = citation.get('source_type', 'Unknown')
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(citation)
            
            # Display each domain group
            for domain, citations in domain_groups.items():
                basic_parts.append(f"**{domain}:**")
                for citation in citations:
                    number = citation.get('number', '?')
                    reference = citation.get('reference', 'Unknown')
                    quote = citation.get('quote', '')
                    basic_parts.append(f"[{number}] {reference}: \"{quote}\"")
                basic_parts.append("")  # Add spacing
        
        return "\n".join(basic_parts)
    
    def _handle_processing_error(self, state: Dict[str, Any], error: Exception, 
                                session_start: float) -> Dict[str, Any]:
        """
        Handle processing errors with comprehensive error information and graceful degradation.
        
        This method ensures the workflow continues even when the enhanced agent
        encounters issues, demonstrating robust error handling for multi-domain integration.
        """
        session_time = time.time() - session_start
        self.session_stats['component_errors'] += 1
        
        self.logger.error("=" * 80)
        self.logger.error("ENHANCED SUMMARIZATION AGENT SESSION FAILED")
        self.logger.error(f"Error after {session_time:.3f} seconds: {error}")
        self.logger.error("=" * 80)
        
        # Provide error information in summary to maintain workflow
        error_summary = {
            "action_plan": f"Error occurred during multi-domain summarization: {str(error)}\n\nPlease review individual agent outputs for available guidance.",
            "total_citations": 0,
            "overall_precision_rate": 0,
            "processing_error": True,
            "component_architecture": "error_state"
        }
        
        state["summary"] = error_summary
        
        print(f"❌ Enhanced multi-domain summarization error: {error}")
        return state
    
    def _log_session_completion(self, comprehensive_stats: Dict[str, Any], session_time: float) -> None:
        """
        Log comprehensive session completion with component statistics.
        
        This method provides detailed insights into how well the component
        architecture performed and identifies any areas needing attention.
        """
        self.logger.info("=" * 80)
        self.logger.info("ENHANCED SUMMARIZATION AGENT SESSION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Session processing time: {session_time:.3f} seconds")
        
        # Log comprehensive statistics
        basic_metrics = comprehensive_stats.get('basic_metrics', {})
        system_health = comprehensive_stats.get('system_health', {})
        
        citation_counts = basic_metrics.get('citation_counts', {})
        self.logger.info(f"Citations unified: {citation_counts.get('unified', 0)}")
        self.logger.info(f"System health score: {system_health.get('overall_health_score', 0):.1f}%")
        
        # Log component performance summary
        self._log_component_performance_summary()
        
        # Log optimization recommendations if available
        optimization_recs = comprehensive_stats.get('optimization_recommendations', [])
        if optimization_recs:
            self.logger.info("System optimization recommendations:")
            for rec in optimization_recs[:3]:  # Show top 3
                self.logger.info(f"  - {rec}")
        
        self.logger.info("Component architecture demonstrated sophisticated multi-domain integration capabilities")
        self.logger.info("=" * 80)
    
    def _log_component_performance_summary(self) -> None:
        """
        Log detailed performance summary from all components.
        
        This method aggregates performance data from all components to provide
        a comprehensive view of how well the sophisticated architecture performed.
        """
        self.logger.info("=== COMPONENT PERFORMANCE SUMMARY ===")
        
        # Citation Manager statistics
        if hasattr(self.citation_manager, 'get_citation_management_statistics'):
            citation_stats = self.citation_manager.get_citation_management_statistics()
            preservation_rate = citation_stats.get('precision_preservation_rate', 0)
            self.logger.info(f"Citation Manager: {preservation_rate:.1f}% precision preservation rate")
        
        # Precision Analyzer statistics
        if hasattr(self.precision_analyzer, 'get_precision_analysis_statistics'):
            precision_stats = self.precision_analyzer.get_precision_analysis_statistics()
            avg_score = precision_stats.get('average_system_score', 0)
            self.logger.info(f"Precision Analyzer: {avg_score:.1f}% average system score")
        
        # Response Builder statistics
        if hasattr(self.response_builder, 'get_building_statistics'):
            builder_stats = self.response_builder.get_building_statistics()
            success_rate = builder_stats.get('llm_success_rate_percent', 0)
            self.logger.info(f"Response Builder: {success_rate:.1f}% LLM integration success rate")
        
        # Formatter statistics
        if hasattr(self.formatter, 'get_formatting_statistics'):
            format_stats = self.formatter.get_formatting_statistics()
            multi_domain_rate = format_stats.get('multi_domain_rate_percent', 0)
            self.logger.info(f"Formatter: {multi_domain_rate:.1f}% multi-domain response rate")
        
        # Statistics Collector statistics
        if hasattr(self.statistics_collector, 'get_collection_statistics'):
            collection_stats = self.statistics_collector.get_collection_statistics()
            collection_success_rate = collection_stats.get('collection_success_rate', 0)
            self.logger.info(f"Statistics Collector: {collection_success_rate:.1f}% collection success rate")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the enhanced agent's performance.
        
        Returns:
            Dictionary containing detailed agent performance metrics and component statistics
        """
        agent_stats = dict(self.session_stats)
        
        # Add component statistics
        component_stats = {}
        
        if hasattr(self.citation_manager, 'get_citation_management_statistics'):
            component_stats['citation_manager'] = self.citation_manager.get_citation_management_statistics()
        
        if hasattr(self.precision_analyzer, 'get_precision_analysis_statistics'):
            component_stats['precision_analyzer'] = self.precision_analyzer.get_precision_analysis_statistics()
        
        if hasattr(self.formatter, 'get_formatting_statistics'):
            component_stats['formatter'] = self.formatter.get_formatting_statistics()
        
        if hasattr(self.statistics_collector, 'get_collection_statistics'):
            component_stats['statistics_collector'] = self.statistics_collector.get_collection_statistics()
        
        if hasattr(self.response_builder, 'get_building_statistics'):
            component_stats['response_builder'] = self.response_builder.get_building_statistics()
        
        agent_stats['component_statistics'] = component_stats
        
        # Calculate overall performance metrics
        if agent_stats['total_queries_processed'] > 0:
            success_rate = (agent_stats['successful_integrations'] / agent_stats['total_queries_processed']) * 100
            agent_stats['overall_success_rate_percent'] = round(success_rate, 1)
            
            multi_domain_rate = (agent_stats['multi_domain_responses'] / agent_stats['total_queries_processed']) * 100
            agent_stats['multi_domain_rate_percent'] = round(multi_domain_rate, 1)
        else:
            agent_stats['overall_success_rate_percent'] = 0
            agent_stats['multi_domain_rate_percent'] = 0
        
        return agent_stats
    
    def log_agent_summary(self) -> None:
        """
        Log a comprehensive summary of the enhanced agent's performance.
        
        This provides a complete picture of how well the component architecture
        is working and helps identify opportunities for optimization.
        """
        stats = self.get_agent_statistics()
        
        self.logger.info("=== ENHANCED SUMMARIZATION AGENT PERFORMANCE SUMMARY ===")
        self.logger.info(f"Total queries processed: {stats['total_queries_processed']}")
        self.logger.info(f"Successful integrations: {stats['successful_integrations']}")
        self.logger.info(f"Overall success rate: {stats['overall_success_rate_percent']}%")
        self.logger.info(f"Multi-domain response rate: {stats['multi_domain_rate_percent']}%")
        self.logger.info(f"Average processing time: {stats['average_processing_time']:.3f} seconds")
        self.logger.info(f"Total unified citations: {stats['total_unified_citations']}")
        self.logger.info(f"Component errors: {stats['component_errors']}")
        self.logger.info(f"Fallback operations: {stats['fallback_operations']}")
        
        # Summarize component performance
        component_stats = stats.get('component_statistics', {})
        if component_stats:
            self.logger.info("Component architecture demonstrating sophisticated multi-domain integration capabilities")
            for component, component_data in component_stats.items():
                if isinstance(component_data, dict) and component_data:
                    key_metric = list(component_data.keys())[0]
                    self.logger.info(f"  - {component}: operational with {key_metric} tracking")


def create_enhanced_summarization_agent(logger: logging.Logger) -> SummarizationAgent:
    """
    Factory function to create a configured enhanced summarization agent.
    
    This provides a clean interface for creating agent instances with proper
    dependency injection, following the same patterns as your other enhanced agents.
    """
    return SummarizationAgent(logger)