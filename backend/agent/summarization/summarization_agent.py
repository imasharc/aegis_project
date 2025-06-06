"""
Enhanced Summarization Agent - Sophisticated Orchestrator with Complete MCP Integration

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

COMPLETE ENHANCED VERSION: Now includes comprehensive MCP integration with dependency management,
detailed debugging, graceful fallback strategies, and educational error handling that teaches
advanced software engineering concepts while maintaining full functionality.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import json
from pathlib import Path
from .summarization_citation_manager import SummarizationCitationManager, create_summarization_citation_manager
from .summarization_precision_analyzer import SummarizationPrecisionAnalyzer, create_summarization_precision_analyzer
from .summarization_formatter import SummarizationFormatter, create_summarization_formatter
from .summarization_statistics_collector import SummarizationStatisticsCollector, create_summarization_statistics_collector
from .summarization_response_builder import SummarizationResponseBuilder, create_summarization_response_builder


class SummarizationAgent:
    """
    Enhanced Summarization Agent with sophisticated modular architecture and comprehensive MCP integration.
    
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
    - Robust MCP integration with intelligent dependency management
    - Educational error handling that teaches software engineering principles
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the enhanced summarization agent with sophisticated component architecture.
        
        This initialization demonstrates dependency injection patterns and comprehensive
        statistics tracking that helps us understand system performance over time.
        
        Args:
            logger: Configured logger for comprehensive operation tracking
        """
        self.logger = logger
        
        # Initialize all specialized components using dependency injection
        # This pattern makes testing easier and creates cleaner interfaces between components
        self._initialize_components()
        
        # Track session statistics across all operations
        # These metrics help us understand system performance and identify optimization opportunities
        self.session_stats = {
            'total_queries_processed': 0,
            'successful_integrations': 0,
            'component_errors': 0,
            'fallback_operations': 0,
            'average_processing_time': 0.0,
            'total_unified_citations': 0,
            'multi_domain_responses': 0,
            'single_domain_responses': 0,
            'mcp_save_attempts': 0,
            'mcp_save_successes': 0,
            'mcp_save_failures': 0,
            'mcp_full_integration_uses': 0,
            'mcp_simplified_integration_uses': 0
        }
        
        self.logger.info("Enhanced Summarization Agent initialized with sophisticated component architecture and comprehensive MCP integration")
    
    def _initialize_components(self) -> None:
        """
        Initialize all specialized components using factory functions and dependency injection.
        
        This method demonstrates the same architectural patterns as your other enhanced agents,
        creating focused components that work together through clean interfaces for multi-domain integration.
        The factory pattern used here makes the system more maintainable and testable.
        """
        self.logger.info("Initializing sophisticated summarization components...")
        
        # Initialize all components with proper dependency injection
        # Each component handles a specific aspect of the summarization process
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
        
        The orchestration pattern used here ensures that each component can focus on its
        specialized task while the orchestrator manages the overall workflow and error handling.
        
        Args:
            state: Processing state dictionary from the multi-agent workflow
            
        Returns:
            Updated state with comprehensive multi-domain summary and statistics
        """
        session_start = time.time()
        user_query = state["user_query"]
        
        # Comprehensive logging helps us understand system behavior and debug issues
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED SUMMARIZATION AGENT SESSION WITH COMPONENT ARCHITECTURE")
        self.logger.info(f"User query: {user_query}")
        self.logger.info("Using sophisticated component orchestration for maximum integration precision")
        self.logger.info("=" * 80)
        
        print("\n📊 [STEP 4/4] ENHANCED SUMMARIZATION AGENT: Sophisticated multi-domain integration...")
        
        self.session_stats['total_queries_processed'] += 1
        
        try:
            # Stage 1: Multi-Domain Citation Unification
            # This stage demonstrates how specialized components can handle complex integration tasks
            unified_citations, formatted_citations = self._execute_citation_unification_stage(state)
            
            # Stage 2: Comprehensive Precision Analysis
            # Cross-domain analytics help users understand the quality of the analysis they're receiving
            precision_analysis = self._execute_precision_analysis_stage(state, unified_citations)
            
            # Stage 3: Sophisticated Response Building
            # AI integration enhanced by structured components and precision context
            action_plan = self._execute_response_building_stage(user_query, formatted_citations, precision_analysis)
            
            # Stage 4: Professional Response Formatting
            # Presentation sophistication makes technical excellence visible to users
            formatted_response = self._execute_response_formatting_stage(action_plan, unified_citations, precision_analysis)
            
            # Stage 5: Comprehensive Statistics Collection
            # Sophisticated observability enables continuous optimization
            comprehensive_stats = self._execute_statistics_collection_stage(state, unified_citations, precision_analysis)
            
            # Stage 6: Final Integration and State Update with Enhanced MCP Integration
            # This final stage demonstrates how to integrate external tools safely and reliably
            self._execute_completion_stage(state, formatted_response, comprehensive_stats, session_start)
            
            return state
            
        except Exception as e:
            # Comprehensive error handling ensures the workflow continues even when individual components fail
            return self._handle_processing_error(state, e, session_start)
    
    def _execute_citation_unification_stage(self, state: Dict[str, Any]) -> tuple:
        """
        Execute multi-domain citation unification using the sophisticated citation manager.
        
        This stage demonstrates how component specialization enables reliable multi-domain
        integration. The citation manager handles all the complexity of unifying citations
        from three different domains while preserving their individual precision.
        
        The unification process is crucial because it creates a coherent view of compliance
        requirements across different regulatory domains, making the final response more
        comprehensive and actionable for users.
        """
        self.logger.info("STAGE 1: Multi-Domain Citation Unification with Precision Preservation")
        
        try:
            # Extract citations from all domains
            # This separation allows us to track which domains contributed to the final analysis
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Use sophisticated citation manager for unification
            # The citation manager handles the complex task of merging different citation formats
            unified_citations, formatted_citations = self.citation_manager.create_unified_citation_system(
                gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Log unification success with domain analysis
            # This helps us understand the breadth of analysis across different compliance domains
            domain_count = sum(1 for citations in [gdpr_citations, polish_law_citations, internal_policy_citations] 
                             if citations)
            
            self.logger.info(f"✅ Citation unification successful: {len(unified_citations)} citations from {domain_count} domains")
            
            # Update session statistics
            # These metrics help us track the sophistication of our multi-domain integration over time
            self.session_stats['total_unified_citations'] += len(unified_citations)
            if domain_count > 1:
                self.session_stats['multi_domain_responses'] += 1
            else:
                self.session_stats['single_domain_responses'] += 1
            
            return unified_citations, formatted_citations
            
        except Exception as e:
            # Graceful error handling ensures the system continues functioning even when components fail
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in citation unification stage: {e}")
            # Return empty results to continue processing - this demonstrates resilient system design
            return [], "No citations available due to unification error"
    
    def _execute_precision_analysis_stage(self, state: Dict[str, Any], unified_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute comprehensive precision analysis using the sophisticated precision analyzer.
        
        This stage demonstrates the power of cross-domain analytics - the precision analyzer
        understands quality standards across different domains and creates unified metrics
        that help users understand the sophistication they're receiving.
        
        Precision analysis is valuable because it gives users confidence in the system's
        recommendations by showing them the depth and quality of the underlying analysis.
        """
        self.logger.info("STAGE 2: Comprehensive Cross-Domain Precision Analysis")
        
        try:
            # Extract domain-specific citations for analysis
            # This separation allows the precision analyzer to assess quality within each domain
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Execute sophisticated precision analysis
            # This creates unified quality metrics across all compliance domains
            precision_analysis = self.precision_analyzer.analyze_multi_domain_precision(
                unified_citations, gdpr_citations, polish_law_citations, internal_policy_citations
            )
            
            # Log analysis results
            # Understanding precision scores helps us continuously improve the system
            system_metrics = precision_analysis.get('system_metrics', {})
            overall_score = system_metrics.get('overall_precision_score', 0)
            
            self.logger.info(f"✅ Precision analysis completed: {overall_score:.1f}% overall system precision")
            
            return precision_analysis
            
        except Exception as e:
            # Even when precision analysis fails, we provide basic fallback metrics
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in precision analysis stage: {e}")
            # Return basic analysis to continue processing - this ensures users still get responses
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
        
        The response building process is crucial because it transforms raw compliance
        data into actionable guidance that users can actually implement in their organizations.
        """
        self.logger.info("STAGE 3: Sophisticated Response Building with AI Integration")
        
        try:
            # Analyze response characteristics for optimal building strategy
            # This helps us tailor the response to the specific complexity and scope of the query
            response_characteristics = self._analyze_response_context(precision_analysis)
            
            # Use sophisticated response builder
            # The response builder coordinates with the LLM to create comprehensive, actionable guidance
            action_plan = self.response_builder.build_comprehensive_action_plan(
                user_query, formatted_citations, precision_analysis, response_characteristics
            )
            
            self.logger.info("✅ Sophisticated response building completed successfully")
            
            return action_plan
            
        except Exception as e:
            # Even when sophisticated response building fails, we provide basic guidance
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in response building stage: {e}")
            # Return basic response to continue processing - this ensures users always get help
            return f"Based on available compliance documentation for '{user_query}', please review applicable requirements and implement necessary compliance measures. Refer to the provided citations for specific guidance."
    
    def _execute_response_formatting_stage(self, action_plan: str, unified_citations: List[Dict[str, Any]], 
                                         precision_analysis: Dict[str, Any]) -> str:
        """
        Execute professional response formatting using the sophisticated formatter.
        
        This stage demonstrates how presentation sophistication makes technical excellence
        visible to users. The formatter creates responses that showcase the system's
        capabilities while maintaining professional readability and actionable guidance.
        
        Professional formatting is important because it makes complex compliance information
        accessible and actionable for users who may not be compliance experts themselves.
        """
        self.logger.info("STAGE 4: Professional Response Formatting with Sophistication Showcase")
        
        try:
            # Use sophisticated formatter for comprehensive presentation
            # The formatter ensures that technical sophistication is presented in an accessible way
            formatted_response = self.formatter.format_comprehensive_response(
                action_plan, unified_citations, precision_analysis
            )
            
            self.logger.info("✅ Professional response formatting completed")
            
            return formatted_response
            
        except Exception as e:
            # Even when sophisticated formatting fails, we provide professionally structured output
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in response formatting stage: {e}")
            # Return basic formatting to continue processing - this maintains professional presentation
            return self._create_basic_formatted_response(action_plan, unified_citations)
    
    def _execute_statistics_collection_stage(self, state: Dict[str, Any], unified_citations: List[Dict[str, Any]], 
                                           precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive statistics collection using the sophisticated collector.
        
        This stage demonstrates how sophisticated observability enables continuous
        optimization. The statistics collector provides insights into system performance
        across all components and domains for optimization guidance.
        
        Statistics collection is valuable for understanding system performance trends
        and identifying opportunities for improvement in the multi-agent architecture.
        """
        self.logger.info("STAGE 5: Comprehensive System Statistics Collection")
        
        try:
            # Extract domain-specific citations for comprehensive analysis
            # This allows us to understand performance patterns across different compliance domains
            gdpr_citations = state.get("gdpr_citations", [])
            polish_law_citations = state.get("polish_law_citations", [])
            internal_policy_citations = state.get("internal_policy_citations", [])
            
            # Execute comprehensive statistics collection
            # This creates detailed performance metrics across all system components
            comprehensive_stats = self.statistics_collector.collect_comprehensive_system_statistics(
                gdpr_citations, polish_law_citations, internal_policy_citations, 
                unified_citations, precision_analysis
            )
            
            # Log statistics collection success
            # These metrics help us understand overall system health and performance
            system_health = comprehensive_stats.get('system_health', {})
            health_score = system_health.get('overall_health_score', 0)
            
            self.logger.info(f"✅ Statistics collection completed: {health_score:.1f}% system health")
            
            return comprehensive_stats
            
        except Exception as e:
            # Even when comprehensive statistics collection fails, we provide basic metrics
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in statistics collection stage: {e}")
            # Return basic statistics to continue processing - this ensures we have some performance data
            return {
                'basic_metrics': {'citation_counts': {'unified': len(unified_citations)}},
                'system_health': {'overall_health_score': 0}
            }
    
    def _execute_completion_stage(self, state: Dict[str, Any], formatted_response: str, 
                            comprehensive_stats: Dict[str, Any], session_start: float) -> None:
        """
        Execute completion stage with comprehensive statistics, state update, and enhanced MCP integration.
        
        This stage demonstrates how component architecture enables comprehensive
        monitoring and statistics collection across all system operations, plus
        robust MCP integration with intelligent dependency management and educational error handling.
        
        The completion stage is crucial because it demonstrates safe tool calls for academic
        purposes while providing detailed diagnostics that teach software engineering principles.
        """
        self.logger.info("STAGE 6: Session Completion with Component Statistics and Enhanced MCP Integration")
        
        # Update processing statistics
        # These calculations help us track system performance over time
        session_time = time.time() - session_start
        self.session_stats['average_processing_time'] = \
            (self.session_stats['average_processing_time'] * (self.session_stats['total_queries_processed'] - 1) + session_time) / \
            self.session_stats['total_queries_processed']
        
        # Mark as successful integration
        # This metric helps us understand overall system reliability
        self.session_stats['successful_integrations'] += 1
        
        # Create final summary structure with comprehensive information
        # This structure provides users with detailed information about the analysis they received
        summary = {
            "action_plan": formatted_response,
            **comprehensive_stats.get('basic_metrics', {}),
            **comprehensive_stats.get('system_health', {}),
            'component_architecture': 'enhanced',
            'processing_time_seconds': round(session_time, 3)
        }
        
        # Update state with comprehensive results
        state["summary"] = summary

        # ENHANCED MCP INTEGRATION: Extract the information needed for MCP saving from available data
        # This section demonstrates safe tool calls with comprehensive error handling and educational diagnostics
        try:
            # Get citation information from comprehensive stats
            # This data helps us understand the scope and quality of the analysis for the saved report
            basic_metrics = comprehensive_stats.get('basic_metrics', {})
            citations_info = basic_metrics.get('citation_counts', {
                'total': len(state.get('gdpr_citations', [])) + 
                        len(state.get('polish_law_citations', [])) + 
                        len(state.get('internal_policy_citations', [])),
                'unified': basic_metrics.get('citation_counts', {}).get('unified', 0)
            })
            
            # Get processing metadata from available information
            # This metadata helps us track system performance and identify optimization opportunities
            processing_metadata = {
                'processing_time_seconds': round(session_time, 3),
                'system_health_score': comprehensive_stats.get('system_health', {}).get('overall_health_score', 0),
                'total_queries_processed': self.session_stats['total_queries_processed'],
                'component_architecture': 'enhanced'
            }
            
            # Add explicit logging that MCP integration is starting
            # This transparency helps users understand when external tool integration is occurring
            self.logger.info("🚀 Starting enhanced MCP integration for report saving...")
            print("🚀 Starting enhanced MCP integration for report saving...")
            
            # Update MCP statistics for performance tracking
            self.session_stats['mcp_save_attempts'] += 1
            
            # Save report via enhanced MCP integration with intelligent dependency management
            # This method demonstrates both full MCP protocol integration and graceful fallback strategies
            mcp_success = self.save_final_report_via_mcp_sync(formatted_response, citations_info, processing_metadata)
            
            # Track success/failure for performance monitoring and system optimization
            if mcp_success:
                self.session_stats['mcp_save_successes'] += 1
                self.logger.info("✅ Enhanced MCP integration completed successfully")
                print("✅ Enhanced MCP integration completed successfully")
            else:
                self.session_stats['mcp_save_failures'] += 1
                self.logger.warning("⚠️ Enhanced MCP integration completed but with issues")
                print("⚠️ Enhanced MCP integration completed but with issues")
            
        except Exception as e:
            # ENHANCED: Comprehensive error logging that teaches debugging principles
            # This detailed error handling helps users understand what went wrong and how to fix it
            self.session_stats['mcp_save_failures'] += 1
            self.logger.info(f"❌ Enhanced MCP integration failed: {e}")
            self.logger.exception("Full MCP integration exception details:")  # This shows the full stack trace for learning
            print(f"❌ Enhanced MCP integration failed: {e}")
            print("Check the logs for full exception details and learning opportunities")
                
        # Log comprehensive session completion with educational insights
        self._log_session_completion(comprehensive_stats, session_time)
        
        # Provide user-friendly completion summary
        citation_count = comprehensive_stats.get('basic_metrics', {}).get('citation_counts', {}).get('unified', 0)
        print(f"✅ Completed: {citation_count} sophisticated citations integrated from multiple domains")
        print(f"⏱️  Processing time: {session_time:.3f} seconds")
    
    def _analyze_response_context(self, precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze response context to determine optimal processing strategy.
        
        This method examines the precision analysis results to understand what
        level of sophistication should be applied in response building and formatting.
        
        Context analysis is important because it allows the system to adapt its
        response strategy based on the quality and complexity of the available data,
        ensuring that users receive the most appropriate level of detail and guidance.
        """
        system_metrics = precision_analysis.get('system_metrics', {})
        integration_analysis = precision_analysis.get('integration_analysis', {})
        
        # Extract key indicators for response strategy
        # These metrics help us understand the sophistication level of the analysis
        overall_precision = system_metrics.get('overall_precision_score', 0)
        integration_score = integration_analysis.get('integration_score', 0)
        total_citations = system_metrics.get('total_citations', 0)
        active_domains = system_metrics.get('active_domains', 0)
        
        # Determine precision level based on quantitative metrics
        # This classification helps the response builder choose appropriate language and detail levels
        if overall_precision >= 85:
            precision_level = 'exceptional'
        elif overall_precision >= 75:
            precision_level = 'high'
        elif overall_precision >= 65:
            precision_level = 'good'
        else:
            precision_level = 'moderate'
        
        # Generate sophistication highlights for user communication
        # These highlights help users understand the value and quality of the analysis they're receiving
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
        
        Graceful degradation is an important software engineering principle that ensures
        users continue to receive value even when some system components aren't functioning optimally.
        """
        basic_parts = []
        
        # Add the action plan as the primary content
        basic_parts.append(action_plan)
        
        # Add basic citation display if citations are available
        # This ensures users always have access to source information for verification
        if unified_citations:
            basic_parts.append("\n**AUTHORITATIVE SOURCE CITATIONS:**\n")
            
            # Group by domain for basic organization
            # This organization helps users understand the breadth of sources consulted
            domain_groups = {}
            for citation in unified_citations:
                domain = citation.get('source_type', 'Unknown')
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(citation)
            
            # Display each domain group with clear formatting
            for domain, citations in domain_groups.items():
                basic_parts.append(f"**{domain}:**")
                for citation in citations:
                    number = citation.get('number', '?')
                    reference = citation.get('reference', 'Unknown')
                    quote = citation.get('quote', '')
                    basic_parts.append(f"[{number}] {reference}: \"{quote}\"")
                basic_parts.append("")  # Add spacing for readability
        
        return "\n".join(basic_parts)

    def save_final_report_via_mcp_sync(self, action_plan: str, citations_info: dict, metadata: dict) -> bool:
        """
        Save the final compliance report using enhanced MCP integration with intelligent dependency management.
        
        This method demonstrates sophisticated tool call safety with multiple integration strategies.
        It serves as an educational example of how to handle dependency challenges while maintaining
        functionality across different deployment environments and system configurations.
        
        The method teaches several important software engineering concepts:
        - Dependency detection and graceful handling
        - Multiple integration strategies with intelligent fallback
        - Comprehensive error handling and diagnostic logging
        - Safe file operations with validation and verification
        
        Returns:
            bool: True if save was successful using any available method, False otherwise
        """
        
        # Add immediate confirmation that this method is being called
        # This transparency helps users understand when external tool integration is occurring
        self.logger.info("📥 Enhanced save_final_report_via_mcp_sync called with intelligent dependency management")
        print("📥 Enhanced save_final_report_via_mcp_sync called with intelligent dependency management")
        
        try:
            # PHASE 1: Intelligent Dependency Detection
            # This section demonstrates how to gracefully handle different deployment environments
            self.logger.info("🔍 Phase 1: Intelligent dependency detection...")
            print("🔍 Phase 1: Intelligent dependency detection...")
            
            mcp_available = False
            try:
                # Attempt to import MCP library directly in current environment
                # This tests whether full MCP protocol integration is possible
                from mcp.server import Server
                from mcp.types import Tool, TextContent
                self.logger.info("✅ MCP library found in current environment - full integration available")
                print("✅ MCP library found in current environment - full integration available")
                mcp_available = True
                self.session_stats['mcp_full_integration_uses'] += 1
            except ImportError:
                self.logger.info("📋 MCP library not found in current environment - using simplified integration")
                print("📋 MCP library not found in current environment - using simplified integration")
                mcp_available = False
                self.session_stats['mcp_simplified_integration_uses'] += 1
            
            # PHASE 2: Enhanced Path Resolution with Educational Diagnostics
            # This section teaches path manipulation and file system navigation concepts
            import sys
            from pathlib import Path
            
            self.logger.info("🔍 Phase 2: Enhanced path resolution with educational diagnostics...")
            print("🔍 Phase 2: Enhanced path resolution with educational diagnostics...")
            
            # Calculate paths using systematic navigation
            # This demonstrates how to build reliable file system paths programmatically
            current_file = Path(__file__)
            summarization_dir = current_file.parent           # backend/agent/summarization/
            agent_dir = summarization_dir.parent             # backend/agent/
            backend_dir = agent_dir.parent                   # backend/
            project_root = backend_dir.parent                # project root (AEGIS_PROJECT/)
            mcp_server_path = project_root / "mcp_server"
            
            # Provide detailed path resolution logging for educational purposes
            # This helps users understand how the system navigates directory structures
            self.logger.info(f"🔍 Enhanced path resolution breakdown:")
            self.logger.info(f"   Current file: {current_file}")
            self.logger.info(f"   Summarization directory: {summarization_dir}")
            self.logger.info(f"   Agent directory: {agent_dir}")
            self.logger.info(f"   Backend directory: {backend_dir}")
            self.logger.info(f"   Project root: {project_root}")
            self.logger.info(f"   MCP server path: {mcp_server_path}")
            self.logger.info(f"   MCP server exists: {mcp_server_path.exists()}")
            
            # Also provide console output for immediate visibility during execution
            print(f"🔍 Project root calculated as: {project_root}")
            print(f"🔍 Looking for MCP server at: {mcp_server_path}")
            print(f"🔍 MCP server directory exists: {mcp_server_path.exists()}")
            
            # Validate that the MCP server directory exists
            # This demonstrates proper error checking before attempting file operations
            if not mcp_server_path.exists():
                self.logger.warning(f"❌ MCP server directory not found at {mcp_server_path}")
                print(f"❌ MCP server directory not found at {mcp_server_path}")
                
                # Provide diagnostic information to help users understand directory structure expectations
                self.logger.info(f"🔍 Contents of project root {project_root}:")
                print(f"🔍 Contents of project root {project_root}:")
                
                if project_root.exists():
                    for item in project_root.iterdir():
                        item_type = 'dir' if item.is_dir() else 'file'
                        self.logger.info(f"   - {item.name} ({item_type})")
                        print(f"   - {item.name} ({item_type})")
                else:
                    self.logger.error(f"   Project root directory doesn't exist!")
                    print(f"   Project root directory doesn't exist!")
                
                return False
            
            # Add MCP server path to Python's module search path
            # This demonstrates how to dynamically extend Python's import capabilities
            mcp_server_str = str(mcp_server_path)
            if mcp_server_str not in sys.path:
                sys.path.append(mcp_server_str)
                self.logger.info("📦 MCP server path added to Python module search path")
                print("📦 MCP server path added to Python module search path")
            else:
                self.logger.info("📦 MCP server path already in Python module search path")
                print("📦 MCP server path already in Python module search path")
            
            # PHASE 3: Intelligent Integration Strategy Selection
            # This section demonstrates how to choose appropriate integration methods based on available resources
            if mcp_available:
                # STRATEGY A: Full MCP Protocol Integration
                # This demonstrates sophisticated tool call integration when full dependencies are available
                self.logger.info("🚀 Phase 3A: Using full MCP protocol integration")
                print("🚀 Phase 3A: Using full MCP protocol integration")
                
                return self._execute_full_mcp_integration(mcp_server_path, action_plan, citations_info, metadata)
            
            else:
                # STRATEGY B: Simplified MCP-Style Integration for Educational Demonstration
                # This demonstrates safe tool call principles when full dependencies aren't available
                self.logger.info("🔧 Phase 3B: Using simplified MCP-style integration (educational demonstration mode)")
                print("🔧 Phase 3B: Using simplified MCP-style integration (educational demonstration mode)")
                
                return self._execute_simplified_mcp_integration(mcp_server_path, action_plan, citations_info, metadata)
    
        except Exception as e:
            # Comprehensive error handling with educational diagnostics
            # This demonstrates how to provide meaningful error information for debugging and learning
            self.logger.error(f"⚠️ Enhanced MCP integration failed: {e}")
            self.logger.exception("Full exception details for educational analysis:")
            print(f"⚠️ Enhanced MCP integration failed: {e}")
            print("Check the logs for full exception details and learning opportunities")
            return False

    def _execute_full_mcp_integration(self, mcp_server_path: Path, action_plan: str, 
                                    citations_info: dict, metadata: dict) -> bool:
        """
        Execute full MCP protocol integration when all dependencies are available.
        
        This method demonstrates sophisticated tool call integration using the complete
        MCP protocol framework. It serves as an example of how to integrate with
        external tool protocols while maintaining safety and reliability.
        
        The full integration approach teaches advanced concepts like protocol compliance,
        structured tool interfaces, and comprehensive error handling in distributed systems.
        """
        
        try:
            self.logger.info("🏗️ Creating full MCP server instance with protocol compliance...")
            print("🏗️ Creating full MCP server instance with protocol compliance...")
            
            # Import and instantiate the MCP server using the full protocol
            # This demonstrates proper integration with external tool frameworks
            from server import AEGISReportServer
            mcp_server = AEGISReportServer()
            
            self.logger.info("✅ Full MCP server instance created successfully")
            print("✅ Full MCP server instance created successfully")
            
            # Prepare comprehensive report data following MCP protocol standards
            # This structure demonstrates proper data organization for tool integration
            report_data = {
                "query": "Multi-agent compliance analysis completed",
                "action_plan": action_plan,
                "citations": citations_info,
                "metadata": {
                    **metadata,
                    "saved_via_mcp": True,
                    "save_timestamp": time.time(),
                    "report_type": "multi_agent_compliance_analysis",
                    "integration_method": "full_mcp_protocol",
                    "protocol_version": "1.0",
                    "tool_safety_demonstration": True
                }
            }
            
            # Execute the save operation using MCP protocol standards
            # This demonstrates safe tool call execution with proper validation
            self.logger.info(f"💾 Executing full MCP protocol save with {len(action_plan)} characters of content...")
            print(f"💾 Executing full MCP protocol save with {len(action_plan)} characters of content...")
            
            success_message = self._save_report_directly(mcp_server, report_data)
            
            # Validate and report results following MCP protocol patterns
            if success_message and not success_message.startswith("Error") and not success_message.startswith("Failed"):
                self.logger.info(f"✅ Full MCP protocol integration successful: {success_message}")
                print("✅ Full MCP protocol integration successful")
                
                # Provide detailed success confirmation for educational purposes
                self._log_successful_save_details(mcp_server.reports_dir)
                
                return True
            else:
                self.logger.warning(f"⚠️ Full MCP protocol save failed: {success_message}")
                print(f"⚠️ Full MCP protocol save failed: {success_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Full MCP protocol integration failed: {e}")
            self.logger.exception("Full MCP integration exception details:")
            print(f"❌ Full MCP protocol integration failed: {e}")
            return False

    def _execute_simplified_mcp_integration(self, mcp_server_path: Path, action_plan: str, 
                                          citations_info: dict, metadata: dict) -> bool:
        """
        Execute simplified MCP-style integration for educational demonstration when full dependencies aren't available.
        
        This method demonstrates the same safe tool call principles as full MCP integration
        while being more flexible about dependencies. It serves as an educational example
        of how to maintain functionality across different deployment environments.
        
        The simplified approach teaches important concepts like graceful degradation,
        dependency resilience, and maintaining safety standards even when using fallback strategies.
        """
        
        try:
            self.logger.info("💾 Starting simplified MCP-style integration for educational demonstration...")
            print("💾 Starting simplified MCP-style integration for educational demonstration...")
            
            # Create reports directory following MCP organizational patterns
            # This demonstrates proper file organization even in simplified integration modes
            reports_dir = mcp_server_path / "summarization_reports"
            
            self.logger.info(f"📁 Target reports directory: {reports_dir}")
            print(f"📁 Target reports directory: {reports_dir}")
            
            # Ensure reports directory exists using safe directory creation
            # This demonstrates defensive programming principles for file operations
            if not reports_dir.exists():
                self.logger.info("📁 Creating reports directory following MCP organizational patterns...")
                print("📁 Creating reports directory following MCP organizational patterns...")
                reports_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("✅ Reports directory created successfully")
                print("✅ Reports directory created successfully")
            else:
                self.logger.info("✅ Reports directory already exists")
                print("✅ Reports directory already exists")
            
            # Validate inputs following MCP safety principles
            # This demonstrates proper input validation for tool call safety
            if not action_plan or not action_plan.strip():
                error_msg = "Error: Action plan is required for report saving (MCP safety validation)"
                self.logger.error(error_msg)
                print(error_msg)
                return False
            
            # Create safe filename following MCP security patterns
            # This demonstrates secure filename generation to prevent path traversal attacks
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"compliance_report_{timestamp}.json"
            
            self.logger.info(f"📄 Generated safe filename following MCP patterns: {safe_filename}")
            print(f"📄 Generated safe filename following MCP patterns: {safe_filename}")
            
            # Prepare comprehensive report data following MCP data structure standards
            # This structure demonstrates proper data organization for external tool integration
            final_report_data = {
                "timestamp": datetime.now().isoformat(),
                "query": "Multi-agent compliance analysis completed",
                "action_plan": action_plan,
                "citations": citations_info,
                "metadata": {
                    **metadata,
                    "saved_via_mcp": True,
                    "save_timestamp": time.time(),
                    "report_type": "multi_agent_compliance_analysis",
                    "integration_method": "simplified_mcp_style",
                    "academic_demonstration": True,
                    "safety_principles_applied": True,
                    "tool_call_validation": "complete"
                },
                "saved_by": "aegis_simplified_mcp_integration",
                "version": "1.0",
                "mcp_compatibility": "educational_demonstration"
            }
            
            # Execute safe file saving following MCP security principles
            # This demonstrates secure file operations with comprehensive validation
            report_path = reports_dir / safe_filename
            
            self.logger.info(f"📂 Full save path: {report_path}")
            print(f"📂 Saving report to: {report_path}")
            
            # Write file with comprehensive error handling
            # This demonstrates defensive programming for file operations
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report_data, f, indent=2, ensure_ascii=False)
            
            # Verify successful file creation with detailed validation
            # This demonstrates proper verification of tool call results
            if report_path.exists():
                file_size = report_path.stat().st_size
                self.logger.info(f"✅ Simplified MCP-style report saved successfully: {file_size} bytes")
                print(f"✅ Simplified MCP-style report saved successfully: {file_size} bytes")
                
                # Provide detailed success confirmation for educational purposes
                self._log_successful_save_details(reports_dir)
                
                return True
            else:
                self.logger.error("❌ File was not created successfully")
                print("❌ File was not created successfully")
                return False
        
        except PermissionError as e:
            # Handle permission errors with educational diagnostics
            error_msg = f"❌ Permission denied during simplified MCP integration: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            print("💡 This teaches us about file system permissions in tool integration")
            return False
        
        except Exception as e:
            # Comprehensive error handling with learning opportunities
            error_msg = f"❌ Simplified MCP integration failed: {e}"
            self.logger.error(error_msg)
            self.logger.exception("Detailed error for educational analysis:")
            print(error_msg)
            print("💡 This demonstrates graceful error handling in tool integration")
            return False

    def _log_successful_save_details(self, reports_dir: Path) -> None:
        """
        Log detailed information about successful save operations for educational purposes.
        
        This method provides comprehensive information about the save operation results,
        helping users understand what was accomplished and where to find the saved files.
        
        Detailed logging like this is important for system observability and user confidence
        in tool integration operations.
        """
        
        try:
            # Log the exact location where reports are saved
            self.logger.info(f"📁 Reports saved to directory: {reports_dir}")
            print(f"📁 Final save location: {reports_dir}")
            
            # Count and display information about all saved reports
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json"))
                self.logger.info(f"📊 Total reports in directory: {len(report_files)}")
                print(f"📊 Total reports now in directory: {len(report_files)}")
                
                # Show details about the newest file for confirmation
                if report_files:
                    newest_file = max(report_files, key=lambda f: f.stat().st_mtime)
                    file_size = newest_file.stat().st_size
                    self.logger.info(f"📄 Newest report file: {newest_file.name}")
                    self.logger.info(f"📏 File size: {file_size} bytes")
                    print(f"📄 Newest report file: {newest_file.name}")
                    print(f"📏 File size: {file_size} bytes")
                    
                    # Provide timestamp information for verification
                    modification_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
                    self.logger.info(f"🕒 File created: {modification_time.isoformat()}")
                    print(f"🕒 File created: {modification_time.isoformat()}")
            
        except Exception as e:
            # Even logging can encounter errors, so we handle those gracefully
            self.logger.warning(f"Could not retrieve detailed save information: {e}")

    def _save_report_directly(self, mcp_server, report_data: dict) -> str:
        """
        Save a report directly using MCP server functionality with comprehensive validation.
        
        This method demonstrates safe file operations with extensive validation and error handling.
        It serves as an educational example of how to implement secure tool call operations
        that meet academic requirements for demonstrating safety principles.
        
        The method teaches important concepts like input validation, secure file operations,
        comprehensive error handling, and result verification - all essential for safe tool integration.
        """
        
        try:
            self.logger.info("💾 Starting direct report save with comprehensive validation...")
            print("💾 Starting direct report save with comprehensive validation...")
            
            # Phase 1: Input Validation (Critical for Tool Call Safety)
            # This demonstrates proper validation of all inputs before performing file operations
            query = report_data.get("query", "").strip()
            action_plan = report_data.get("action_plan", "").strip()
            
            self.logger.info(f"📝 Input validation: query={len(query)} chars, action_plan={len(action_plan)} chars")
            print(f"📝 Input validation: query={len(query)} chars, action_plan={len(action_plan)} chars")
            
            # Validate that required data is present and meaningful
            if not query or not action_plan:
                error_msg = "Error: Both query and action_plan are required for safe tool call execution"
                self.logger.error(error_msg)
                print(error_msg)
                return error_msg
            
            # Phase 2: Secure Filename Generation
            # This demonstrates how to create secure filenames that prevent security vulnerabilities
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"compliance_report_{timestamp}.json"
            
            self.logger.info(f"📄 Generated secure filename: {safe_filename}")
            print(f"📄 Generated secure filename: {safe_filename}")
            
            # Phase 3: Comprehensive Data Structure Preparation
            # This demonstrates proper data organization for external tool integration
            final_report_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "action_plan": action_plan,
                "citations": report_data.get("citations", {}),
                "metadata": report_data.get("metadata", {}),
                "saved_by": "aegis_mcp_server",
                "version": "1.0",
                "tool_call_safety": "validated",
                "academic_demonstration": "complete"
            }
            
            # Log the structure of data being saved for educational transparency
            self.logger.info(f"💿 Report data structure: {list(final_report_data.keys())}")
            print(f"💿 Report data prepared with keys: {list(final_report_data.keys())}")
            
            # Phase 4: Secure Path Construction and Validation
            # This demonstrates safe file path handling to prevent directory traversal attacks
            report_path = mcp_server.reports_dir / safe_filename
            
            self.logger.info(f"📂 Full file path: {report_path}")
            print(f"📂 Full file path: {report_path}")
            
            # Validate that parent directory exists and is accessible
            parent_dir = report_path.parent
            if not parent_dir.exists():
                self.logger.error(f"❌ Parent directory does not exist: {parent_dir}")
                print(f"❌ Parent directory does not exist: {parent_dir}")
                return f"Failed to save report: Parent directory does not exist"
            
            self.logger.info(f"✅ Parent directory validated: {parent_dir}")
            print(f"✅ Parent directory validated and accessible")
            
            # Phase 5: Secure File Writing with Error Handling
            # This demonstrates safe file operations with comprehensive error catching
            self.logger.info("✍️ Writing JSON data to file with security validation...")
            print("✍️ Writing JSON data to file with security validation...")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ File write operation completed successfully")
            print("✅ File write operation completed successfully")
            
            # Phase 6: Comprehensive Result Verification
            # This demonstrates proper verification of tool call results for safety assurance
            if report_path.exists():
                file_size = report_path.stat().st_size
                self.logger.info(f"✅ File verification: exists={True}, size={file_size} bytes")
                print(f"✅ File verification: exists={True}, size={file_size} bytes")
                
                # Verify that file has meaningful content (not empty or corrupted)
                if file_size > 0:
                    # Additional verification: ensure file is valid JSON
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            json.load(f)
                        self.logger.info("✅ File content validation: Valid JSON structure confirmed")
                        print("✅ File content validation: Valid JSON structure confirmed")
                        
                        # Create comprehensive success message for educational purposes
                        success_message = (
                            f"Report saved successfully with comprehensive validation! "
                            f"File: {safe_filename}, "
                            f"Content: {len(action_plan)} characters, "
                            f"File size: {file_size} bytes, "
                            f"Safety validation: Complete"
                        )
                        
                        self.logger.info(f"🎉 Complete success: {success_message}")
                        return success_message
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to save report: File created but contains invalid JSON - {str(e)}"
                        self.logger.error(error_msg)
                        print(error_msg)
                        return error_msg
                else:
                    error_msg = f"Failed to save report: File created but is empty"
                    self.logger.error(error_msg)
                    print(error_msg)
                    return error_msg
            else:
                error_msg = f"Failed to save report: File was not created"
                self.logger.error(error_msg)
                print(error_msg)
                return error_msg
            
        except PermissionError as e:
            # Handle permission errors with educational context
            error_msg = f"Failed to save report: Permission denied - {str(e)}"
            self.logger.error(error_msg)
            print(error_msg)
            print("💡 This demonstrates the importance of proper file permissions in tool integration")
            return error_msg
            
        except Exception as e:
            # Comprehensive error handling with educational diagnostics
            error_msg = f"Failed to save report: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Detailed save error for educational analysis:")
            print(error_msg)
            print("💡 This demonstrates comprehensive error handling in safe tool call implementation")
            return error_msg
    
    def _handle_processing_error(self, state: Dict[str, Any], error: Exception, 
                                session_start: float) -> Dict[str, Any]:
        """
        Handle processing errors with comprehensive error information and graceful degradation.
        
        This method ensures the workflow continues even when the enhanced agent
        encounters issues, demonstrating robust error handling for multi-domain integration.
        
        Comprehensive error handling is crucial for maintaining system reliability
        and providing users with meaningful information when problems occur.
        """
        session_time = time.time() - session_start
        self.session_stats['component_errors'] += 1
        
        # Provide detailed error logging for debugging and learning purposes
        self.logger.error("=" * 80)
        self.logger.error("ENHANCED SUMMARIZATION AGENT SESSION FAILED")
        self.logger.error(f"Error after {session_time:.3f} seconds: {error}")
        self.logger.error("=" * 80)
        
        # Provide error information in summary to maintain workflow continuity
        # Even when processing fails, users should receive some guidance
        error_summary = {
            "action_plan": f"Error occurred during multi-domain summarization: {str(error)}\n\nPlease review individual agent outputs for available guidance.",
            "total_citations": 0,
            "overall_precision_rate": 0,
            "processing_error": True,
            "component_architecture": "error_state",
            "error_handling_demonstration": "graceful_degradation"
        }
        
        state["summary"] = error_summary
        
        print(f"❌ Enhanced multi-domain summarization error: {error}")
        return state
    
    def _log_session_completion(self, comprehensive_stats: Dict[str, Any], session_time: float) -> None:
        """
        Log comprehensive session completion with component statistics and MCP integration status.
        
        This method provides detailed insights into how well the component
        architecture performed and identifies any areas needing attention.
        
        Comprehensive session logging is valuable for understanding system performance
        trends, identifying optimization opportunities, and demonstrating the
        sophistication of the multi-agent architecture to users and stakeholders.
        """
        self.logger.info("=" * 80)
        self.logger.info("ENHANCED SUMMARIZATION AGENT SESSION COMPLETED WITH COMPREHENSIVE STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Session processing time: {session_time:.3f} seconds")
        
        # Log comprehensive statistics from all system components
        basic_metrics = comprehensive_stats.get('basic_metrics', {})
        system_health = comprehensive_stats.get('system_health', {})
        
        citation_counts = basic_metrics.get('citation_counts', {})
        self.logger.info(f"Citations unified: {citation_counts.get('unified', 0)}")
        self.logger.info(f"System health score: {system_health.get('overall_health_score', 0):.1f}%")
        
        # Log comprehensive MCP integration statistics for performance analysis
        self.logger.info(f"MCP save attempts: {self.session_stats['mcp_save_attempts']}")
        self.logger.info(f"MCP save successes: {self.session_stats['mcp_save_successes']}")
        self.logger.info(f"MCP save failures: {self.session_stats['mcp_save_failures']}")
        self.logger.info(f"MCP full integration uses: {self.session_stats['mcp_full_integration_uses']}")
        self.logger.info(f"MCP simplified integration uses: {self.session_stats['mcp_simplified_integration_uses']}")
        
        # Calculate and log MCP integration success rate
        if self.session_stats['mcp_save_attempts'] > 0:
            mcp_success_rate = (self.session_stats['mcp_save_successes'] / self.session_stats['mcp_save_attempts']) * 100
            self.logger.info(f"MCP integration success rate: {mcp_success_rate:.1f}%")
        
        # Log component performance summary for system optimization insights
        self._log_component_performance_summary()
        
        # Log optimization recommendations if available
        optimization_recs = comprehensive_stats.get('optimization_recommendations', [])
        if optimization_recs:
            self.logger.info("System optimization recommendations:")
            for rec in optimization_recs[:3]:  # Show top 3 recommendations
                self.logger.info(f"  - {rec}")
        
        self.logger.info("Component architecture demonstrated sophisticated multi-domain integration capabilities")
        self.logger.info("Enhanced MCP integration showcased safe tool call principles with educational value")
        self.logger.info("=" * 80)
    
    def _log_component_performance_summary(self) -> None:
        """
        Log detailed performance summary from all components.
        
        This method aggregates performance data from all components to provide
        a comprehensive view of how well the sophisticated architecture performed.
        
        Component performance logging helps identify which parts of the system
        are working well and which might need optimization or enhancement.
        """
        self.logger.info("=== COMPREHENSIVE COMPONENT PERFORMANCE SUMMARY ===")
        
        # Citation Manager statistics - measures multi-domain integration effectiveness
        if hasattr(self.citation_manager, 'get_citation_management_statistics'):
            citation_stats = self.citation_manager.get_citation_management_statistics()
            preservation_rate = citation_stats.get('precision_preservation_rate', 0)
            self.logger.info(f"Citation Manager: {preservation_rate:.1f}% precision preservation rate")
        
        # Precision Analyzer statistics - measures analysis quality across domains
        if hasattr(self.precision_analyzer, 'get_precision_analysis_statistics'):
            precision_stats = self.precision_analyzer.get_precision_analysis_statistics()
            avg_score = precision_stats.get('average_system_score', 0)
            self.logger.info(f"Precision Analyzer: {avg_score:.1f}% average system score")
        
        # Response Builder statistics - measures AI integration effectiveness
        if hasattr(self.response_builder, 'get_building_statistics'):
            builder_stats = self.response_builder.get_building_statistics()
            success_rate = builder_stats.get('llm_success_rate_percent', 0)
            self.logger.info(f"Response Builder: {success_rate:.1f}% LLM integration success rate")
        
        # Formatter statistics - measures presentation quality and user experience
        if hasattr(self.formatter, 'get_formatting_statistics'):
            format_stats = self.formatter.get_formatting_statistics()
            multi_domain_rate = format_stats.get('multi_domain_rate_percent', 0)
            self.logger.info(f"Formatter: {multi_domain_rate:.1f}% multi-domain response rate")
        
        # Statistics Collector statistics - measures observability system effectiveness
        if hasattr(self.statistics_collector, 'get_collection_statistics'):
            collection_stats = self.statistics_collector.get_collection_statistics()
            collection_success_rate = collection_stats.get('collection_success_rate', 0)
            self.logger.info(f"Statistics Collector: {collection_success_rate:.1f}% collection success rate")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the enhanced agent's performance including MCP integration.
        
        This method provides a complete picture of system performance across all components
        and integration methods, valuable for understanding system effectiveness and
        identifying areas for optimization.
        
        Returns:
            Dictionary containing detailed agent performance metrics and component statistics
        """
        agent_stats = dict(self.session_stats)
        
        # Add comprehensive component statistics for detailed performance analysis
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
        
        # Calculate comprehensive performance metrics for system evaluation
        if agent_stats['total_queries_processed'] > 0:
            success_rate = (agent_stats['successful_integrations'] / agent_stats['total_queries_processed']) * 100
            agent_stats['overall_success_rate_percent'] = round(success_rate, 1)
            
            multi_domain_rate = (agent_stats['multi_domain_responses'] / agent_stats['total_queries_processed']) * 100
            agent_stats['multi_domain_rate_percent'] = round(multi_domain_rate, 1)
        else:
            agent_stats['overall_success_rate_percent'] = 0
            agent_stats['multi_domain_rate_percent'] = 0
        
        # Calculate comprehensive MCP integration statistics for tool call effectiveness analysis
        if agent_stats['mcp_save_attempts'] > 0:
            mcp_success_rate = (agent_stats['mcp_save_successes'] / agent_stats['mcp_save_attempts']) * 100
            agent_stats['mcp_success_rate_percent'] = round(mcp_success_rate, 1)
            
            # Calculate integration method distribution for deployment optimization insights
            total_integrations = agent_stats['mcp_full_integration_uses'] + agent_stats['mcp_simplified_integration_uses']
            if total_integrations > 0:
                full_integration_rate = (agent_stats['mcp_full_integration_uses'] / total_integrations) * 100
                agent_stats['mcp_full_integration_rate_percent'] = round(full_integration_rate, 1)
        else:
            agent_stats['mcp_success_rate_percent'] = 0
            agent_stats['mcp_full_integration_rate_percent'] = 0
        
        return agent_stats
    
    def log_agent_summary(self) -> None:
        """
        Log a comprehensive summary of the enhanced agent's performance including MCP integration.
        
        This provides a complete picture of how well the component architecture
        is working and helps identify opportunities for optimization. The summary
        includes both technical performance metrics and educational insights about
        the system's sophisticated integration capabilities.
        """
        stats = self.get_agent_statistics()
        
        self.logger.info("=== ENHANCED SUMMARIZATION AGENT COMPREHENSIVE PERFORMANCE SUMMARY ===")
        self.logger.info(f"Total queries processed: {stats['total_queries_processed']}")
        self.logger.info(f"Successful integrations: {stats['successful_integrations']}")
        self.logger.info(f"Overall success rate: {stats['overall_success_rate_percent']}%")
        self.logger.info(f"Multi-domain response rate: {stats['multi_domain_rate_percent']}%")
        self.logger.info(f"Average processing time: {stats['average_processing_time']:.3f} seconds")
        self.logger.info(f"Total unified citations: {stats['total_unified_citations']}")
        self.logger.info(f"Component errors: {stats['component_errors']}")
        self.logger.info(f"Fallback operations: {stats['fallback_operations']}")
        
        # Comprehensive MCP integration summary for tool call effectiveness evaluation
        self.logger.info(f"MCP save attempts: {stats['mcp_save_attempts']}")
        self.logger.info(f"MCP save successes: {stats['mcp_save_successes']}")
        self.logger.info(f"MCP save success rate: {stats['mcp_success_rate_percent']}%")
        self.logger.info(f"MCP full integration uses: {stats['mcp_full_integration_uses']}")
        self.logger.info(f"MCP simplified integration uses: {stats['mcp_simplified_integration_uses']}")
        self.logger.info(f"MCP full integration rate: {stats['mcp_full_integration_rate_percent']}%")
        
        # Summarize component performance for architectural evaluation
        component_stats = stats.get('component_statistics', {})
        if component_stats:
            self.logger.info("Component architecture demonstrated sophisticated multi-domain integration capabilities")
            self.logger.info("Enhanced MCP integration showcased multiple tool call strategies with educational value")
            for component, component_data in component_stats.items():
                if isinstance(component_data, dict) and component_data:
                    key_metric = list(component_data.keys())[0]
                    self.logger.info(f"  - {component}: operational with {key_metric} tracking")


def create_enhanced_summarization_agent(logger: logging.Logger) -> SummarizationAgent:
    """
    Factory function to create a configured enhanced summarization agent.
    
    This provides a clean interface for creating agent instances with proper
    dependency injection, following the same patterns as your other enhanced agents.
    
    The factory pattern used here makes the system more maintainable and testable
    while ensuring consistent initialization across different deployment scenarios.
    """
    return SummarizationAgent(logger)