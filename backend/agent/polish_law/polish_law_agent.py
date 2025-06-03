"""
Enhanced Polish Law Agent - Sophisticated Orchestrator

This module represents the culmination of the Polish law agent refactoring, combining all the
specialized components into a clean, maintainable orchestrator. Like your enhanced
GDPR agent, this agent demonstrates how architectural sophistication creates
more reliable, maintainable, and powerful systems specifically adapted for Polish law.

Polish Law Specific Features:
- Section-aware document analysis and citation creation
- Polish legal terminology recognition and validation
- Gazette reference integration for legal authenticity
- Parliament session and amendment context tracking
- Polish legal numbering pattern recognition and processing
- Enhanced support for Polish legal document organizational patterns

The agent now follows the same design patterns as your GDPR agent while respecting
Polish legal document conventions:
- Single Responsibility Principle with focused components
- Dependency injection for clean interfaces
- Comprehensive error handling and graceful degradation  
- Detailed logging and statistics for monitoring
- Factory pattern for clean instantiation

This refactored agent integrates seamlessly with your enhanced Polish law processing pipeline,
using the same vector database and metadata structure that your processing modules create,
while providing sophisticated analysis capabilities specifically adapted for Polish legal documents.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .polish_law_vector_store_connector import PolishLawVectorStoreConnector, create_polish_law_vector_store_connector
from .polish_law_metadata_processor import PolishLawMetadataProcessor, create_polish_law_metadata_processor
from .polish_law_content_analyzer import PolishLawContentAnalyzer, create_polish_law_content_analyzer
from .polish_law_citation_builder import PolishLawCitationBuilder, create_polish_law_citation_builder
from .polish_law_response_parser import PolishLawResponseParser, create_polish_law_response_parser


class PolishLawAgent:
    """
    Enhanced Polish Law Agent with sophisticated modular architecture.
    
    This class represents the complete solution for Polish law analysis using the same
    architectural excellence demonstrated in your GDPR agent refactor, but specifically
    adapted for Polish legal document patterns and citation requirements.
    Instead of a monolithic agent, we now have a sophisticated orchestrator that
    coordinates specialized components through clean interfaces.
    
    The agent demonstrates how your architectural patterns create:
    - More reliable processing through specialized components adapted for Polish law
    - Better maintainability through single responsibility modules
    - Enhanced functionality through component synergy with Polish legal features
    - Easier testing through dependency injection
    - Improved monitoring through comprehensive statistics including Polish law metrics
    """
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize the enhanced Polish law agent with sophisticated component architecture.
        
        Args:
            db_path: Path to the Polish law vector database created by your processing pipeline
            logger: Configured logger for comprehensive operation tracking
        """
        self.db_path = db_path
        self.logger = logger
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize all specialized components using dependency injection
        self._initialize_polish_law_components()
        
        # Configure LLM prompts for enhanced Polish law analysis
        self._setup_polish_law_prompts()
        
        # Track session statistics across all operations with Polish law specifics
        self.session_stats = {
            'total_queries_processed': 0,
            'successful_analyses': 0,
            'component_errors': 0,
            'fallback_operations': 0,
            'average_processing_time': 0.0,
            'total_citations_created': 0,
            'polish_law_features': {
                'section_aware_analyses': 0,
                'gazette_reference_enriched': 0,
                'polish_terminology_validated': 0,
                'parliament_context_included': 0
            }
        }
        
        self.logger.info("Enhanced Polish Law Agent initialized with sophisticated component architecture")
    
    def _initialize_polish_law_components(self) -> None:
        """
        Initialize all specialized components using factory functions and dependency injection.
        
        This method demonstrates the same architectural patterns as your GDPR agent,
        creating focused components that work together through clean interfaces while
        being specifically adapted for Polish legal document analysis.
        """
        self.logger.info("Initializing sophisticated Polish law analysis components...")
        
        # Initialize all components with proper dependency injection
        self.vector_connector = create_polish_law_vector_store_connector(self.db_path, self.logger)
        self.metadata_processor = create_polish_law_metadata_processor(self.logger)
        self.content_analyzer = create_polish_law_content_analyzer(self.logger)
        self.citation_builder = create_polish_law_citation_builder(self.logger)
        
        # Initialize response parser with all required dependencies
        self.response_parser = create_polish_law_response_parser(
            self.metadata_processor, self.content_analyzer, 
            self.citation_builder, self.logger
        )
        
        self.logger.info("All Polish law analysis components initialized successfully")
    
    def _setup_polish_law_prompts(self) -> None:
        """
        Configure LLM prompts optimized for the enhanced component architecture and Polish law.
        
        These prompts work with the sophisticated analysis pipeline to encourage
        the LLM to provide information that our components can enhance and perfect,
        while being specifically adapted for Polish legal document analysis.
        """
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a specialized Polish data protection law expert analyzing retrieved regulation content with enhanced structural understanding.
            
            User Query: {user_query}
            
            Based on the following retrieved Polish law content, identify the most relevant provisions:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. Basic article information (precise formatting will be enhanced automatically using structural metadata)
            2. A direct, specific quote of the relevant text from the retrieved context
            3. A brief explanation of its relevance to the query and how it applies to data protection under Polish law
            
            ENHANCED POLISH LAW CITATION GUIDANCE:
            - Choose quotes that represent complete legal requirements or principles from Polish data protection law
            - Prefer quotes that include structural indicators like "1)", "(a)", "(b)" when present
            - Be aware that Polish law may have sections and different organizational patterns than EU regulations
            - The system will automatically determine precise paragraph, sub-paragraph, and section references
            - Focus on the legal substance rather than structural formatting in your explanations
            - Consider Polish legal context and how provisions relate to Polish data protection principles
            
            Format your response as a structured list of citations in this exact format:
            
            CITATION 1:
            - Article: [Basic article info - precise structure will be determined automatically]
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including Polish data protection law relevance]
            
            CITATION 2:
            - Article: [Basic article info - precise structure will be determined automatically]
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including Polish data protection law relevance]
            """
        )
        
        self.logger.info("Enhanced LLM prompts configured for Polish law component integration")
    
    def connect_and_validate(self) -> bool:
        """
        Connect to the vector store and validate that all components are ready for Polish law analysis.
        
        This method performs comprehensive validation to ensure that the agent
        can work with the vector database created by your Polish law processing pipeline.
        
        Returns:
            True if all components are ready for operation, False otherwise
        """
        self.logger.info("Connecting and validating enhanced Polish law agent components...")
        
        try:
            # Connect to vector store using the sophisticated connector
            if not self.vector_connector.connect_and_validate():
                self.logger.error("Polish law vector store connection failed - agent cannot operate")
                return False
            
            self.logger.info("✅ All Polish law agent components connected and validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during Polish law agent initialization: {e}")
            return False
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method demonstrating sophisticated component orchestration for Polish law.
        
        This method represents the enhanced approach to Polish law analysis, orchestrating
        all the specialized components to create the most precise citations possible while
        following Polish legal citation conventions. The method demonstrates how architectural 
        sophistication enables reliable, maintainable operations specifically adapted for Polish law.
        
        Args:
            state: Processing state dictionary from the multi-agent workflow
            
        Returns:
            Updated state with enhanced Polish law citations
        """
        session_start = time.time()
        user_query = state["user_query"]
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED POLISH LAW AGENT SESSION WITH COMPONENT ARCHITECTURE")
        self.logger.info(f"User query: {user_query}")
        self.logger.info("Using sophisticated component orchestration for maximum precision in Polish law analysis")
        self.logger.info("=" * 80)
        
        print("\n🇵🇱 [STEP 2/4] ENHANCED POLISH LAW AGENT: Sophisticated analysis with component architecture...")
        
        self.session_stats['total_queries_processed'] += 1
        
        try:
            # Stage 1: Document Retrieval with Enhanced Vector Store Connector
            retrieved_docs, retrieved_context, document_metadata = self._execute_polish_law_retrieval_stage(user_query)
            
            if not retrieved_docs:
                return self._handle_retrieval_failure(state)
            
            # Stage 2: LLM Analysis with Enhanced Prompts
            llm_response = self._execute_llm_analysis_stage(user_query, retrieved_context)
            
            # Stage 3: Sophisticated Response Parsing and Citation Creation
            polish_law_citations = self._execute_citation_creation_stage(llm_response, document_metadata)
            
            # Stage 4: Final Validation and State Update
            self._execute_completion_stage(state, polish_law_citations, session_start)
            
            return state
            
        except Exception as e:
            return self._handle_processing_error(state, e, session_start)
    
    def _execute_polish_law_retrieval_stage(self, user_query: str) -> tuple:
        """
        Execute document retrieval using the sophisticated Polish law vector store connector.
        
        This stage demonstrates how component specialization improves reliability for Polish law.
        The vector store connector handles all the complexity of document retrieval
        and metadata validation specifically adapted for Polish legal documents.
        """
        self.logger.info("STAGE 1: Enhanced Document Retrieval with Polish Law Vector Store Connector")
        
        try:
            retrieved_docs, retrieved_context, document_metadata = \
                self.vector_connector.retrieve_relevant_documents(user_query)
            
            if retrieved_docs:
                self.logger.info(f"✅ Retrieved {len(retrieved_docs)} Polish law documents with enhanced metadata")
                # Log retrieval statistics with Polish law specifics
                retrieval_stats = self.vector_connector.get_retrieval_statistics()
                enhancement_rate = retrieval_stats.get('enhancement_rate_percent', 0)
                section_rate = retrieval_stats.get('section_coverage_rate_percent', 0)
                gazette_rate = retrieval_stats.get('gazette_reference_rate_percent', 0)
                
                self.logger.info(f"📊 Polish law document enhancement rate: {enhancement_rate}%")
                self.logger.info(f"📊 Section coverage rate: {section_rate}%")
                self.logger.info(f"📊 Gazette reference rate: {gazette_rate}%")
                
                # Track Polish law-specific features
                if section_rate > 0:
                    self.session_stats['polish_law_features']['section_aware_analyses'] += 1
                if gazette_rate > 0:
                    self.session_stats['polish_law_features']['gazette_reference_enriched'] += 1
            else:
                self.logger.warning("❌ No Polish law documents retrieved - query may be too specific or database empty")
            
            return retrieved_docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in Polish law retrieval stage: {e}")
            return [], "", []
    
    def _execute_llm_analysis_stage(self, user_query: str, retrieved_context: str) -> str:
        """
        Execute LLM analysis using enhanced prompts optimized for Polish law component processing.
        
        This stage uses prompts designed to work with the sophisticated parsing
        and analysis components that will process the LLM's response for Polish law.
        """
        self.logger.info("STAGE 2: Enhanced LLM Analysis with Polish Law Component-Optimized Prompts")
        
        try:
            rag_chain = self.rag_prompt | self.model
            start_time = time.time()
            
            response = rag_chain.invoke({
                "user_query": user_query,
                "retrieved_context": retrieved_context
            })
            
            analysis_time = time.time() - start_time
            self.logger.info(f"✅ LLM analysis completed in {analysis_time:.3f} seconds")
            self.logger.info(f"📝 Response length: {len(response.content)} characters")
            
            return response.content
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in LLM analysis stage: {e}")
            return f"ERROR: Failed to analyze retrieved Polish law content - {str(e)}"
    
    def _execute_citation_creation_stage(self, llm_response: str, 
                                        document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Execute sophisticated citation creation using the response parser and all Polish law components.
        
        This stage demonstrates the power of the component architecture for Polish law. The response
        parser orchestrates the metadata processor, content analyzer, and citation builder
        to create the most precise citations possible while following Polish legal conventions.
        """
        self.logger.info("STAGE 3: Sophisticated Citation Creation with Polish Law Component Orchestration")
        
        try:
            # Use the sophisticated response parser with all components
            polish_law_citations = self.response_parser.parse_llm_response_to_citations(
                llm_response, document_metadata
            )
            
            if polish_law_citations:
                self.session_stats['total_citations_created'] += len(polish_law_citations)
                self.logger.info(f"✅ Created {len(polish_law_citations)} enhanced Polish law citations")
                
                # Log enhancement statistics with Polish law specifics
                enhanced_count = sum(1 for cite in polish_law_citations 
                                   if cite.get('enhancement_level') == 'sophisticated')
                basic_count = sum(1 for cite in polish_law_citations 
                                if cite.get('enhancement_level') == 'basic')
                
                self.logger.info(f"📊 Citation enhancement: {enhanced_count} sophisticated, {basic_count} basic")
                
                # Track Polish law-specific enhancements
                section_enhanced = sum(1 for cite in polish_law_citations 
                                     if cite.get('analysis_metadata', {}).get('polish_law_features', {}).get('section_aware', False))
                gazette_validated = sum(1 for cite in polish_law_citations 
                                      if cite.get('analysis_metadata', {}).get('polish_law_features', {}).get('gazette_reference', False))
                
                if section_enhanced > 0:
                    self.session_stats['polish_law_features']['section_aware_analyses'] += section_enhanced
                    self.logger.info(f"📊 Section-enhanced citations: {section_enhanced}")
                
                if gazette_validated > 0:
                    self.session_stats['polish_law_features']['gazette_reference_enriched'] += gazette_validated
                    self.logger.info(f"📊 Gazette-validated citations: {gazette_validated}")
                    
            else:
                self.logger.warning("❌ No citations created - using component fallback mechanisms")
                polish_law_citations = self._create_emergency_fallback_citations(document_metadata)
            
            return polish_law_citations
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in Polish law citation creation stage: {e}")
            return self._create_emergency_fallback_citations(document_metadata)
    
    def _execute_completion_stage(self, state: Dict[str, Any], polish_law_citations: List[Dict[str, Any]], 
                                 session_start: float) -> None:
        """
        Execute completion stage with comprehensive statistics and state update.
        
        This stage demonstrates how component architecture enables comprehensive
        monitoring and statistics collection across all system operations for Polish law.
        """
        self.logger.info("STAGE 4: Session Completion with Polish Law Component Statistics")
        
        # Update processing statistics
        session_time = time.time() - session_start
        self.session_stats['average_processing_time'] = \
            (self.session_stats['average_processing_time'] * (self.session_stats['total_queries_processed'] - 1) + session_time) / \
            self.session_stats['total_queries_processed']
        
        if polish_law_citations:
            self.session_stats['successful_analyses'] += 1
        
        # Update state with results
        state["polish_law_citations"] = polish_law_citations
        
        # Log comprehensive session completion
        self._log_session_completion(polish_law_citations, session_time)
        
        print(f"✅ Completed: {len(polish_law_citations)} sophisticated Polish law citations created")
        print(f"⏱️  Processing time: {session_time:.3f} seconds")
    
    def _handle_retrieval_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle retrieval failure with graceful degradation for Polish law.
        
        This method demonstrates how component architecture enables reliable
        error handling and fallback mechanisms specifically adapted for Polish law.
        """
        self.logger.error("Polish law document retrieval failed - creating minimal fallback citations")
        self.session_stats['fallback_operations'] += 1
        
        fallback_citations = [{
            "article": "Polish Data Protection Law",
            "quote": "Polish law vector store is not available or empty",
            "explanation": "Could not retrieve Polish data protection law documents for analysis",
            "enhancement_level": "error"
        }]
        
        state["polish_law_citations"] = fallback_citations
        print("❌ Polish law document retrieval failed - provided fallback citations")
        
        return state
    
    def _handle_processing_error(self, state: Dict[str, Any], error: Exception, 
                                session_start: float) -> Dict[str, Any]:
        """
        Handle processing errors with comprehensive error information and graceful degradation.
        
        This method ensures the workflow continues even when the enhanced Polish law agent
        encounters issues, demonstrating robust error handling.
        """
        session_time = time.time() - session_start
        self.session_stats['component_errors'] += 1
        
        self.logger.error("=" * 80)
        self.logger.error("ENHANCED POLISH LAW AGENT SESSION FAILED")
        self.logger.error(f"Error after {session_time:.3f} seconds: {error}")
        self.logger.error("=" * 80)
        
        # Provide error information in citations to maintain workflow
        error_citations = [{
            "article": "Polish Data Protection Law (Enhanced Agent Error)",
            "quote": "Could not complete enhanced Polish law analysis",
            "explanation": f"Error occurred during sophisticated processing: {str(error)}"
        }]
        
        state["polish_law_citations"] = error_citations
        
        print(f"❌ Enhanced Polish law analysis error: {error}")
        return state
    
    def _create_emergency_fallback_citations(self, document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create emergency fallback citations when all sophisticated processing fails for Polish law.
        
        This method uses basic citation building capabilities to ensure the system
        provides some output even when the advanced features encounter issues,
        while maintaining Polish law citation conventions.
        """
        self.logger.warning("Creating emergency fallback citations using basic Polish law methods")
        self.session_stats['fallback_operations'] += 1
        
        try:
            if document_metadata:
                # Use basic citation builder for simple fallback
                fallback_citation = self.citation_builder.create_basic_citation_from_metadata(
                    document_metadata[0]['metadata']
                )
                
                return [{
                    "article": fallback_citation,
                    "quote": "Emergency fallback - retrieved relevant Polish law content",
                    "explanation": "Created using emergency fallback when enhanced processing failed",
                    "enhancement_level": "emergency"
                }]
            else:
                return [{
                    "article": "Polish Data Protection Law",
                    "quote": "No Polish law documents available for analysis",
                    "explanation": "Emergency fallback when no documents could be retrieved",
                    "enhancement_level": "emergency"
                }]
                
        except Exception as e:
            self.logger.error(f"Emergency fallback creation failed: {e}")
            return [{
                "article": "Polish Data Protection Law (System Error)",
                "quote": "All Polish law analysis methods failed",
                "explanation": "System error - could not create any citations",
                "enhancement_level": "error"
            }]
    
    def _log_session_completion(self, polish_law_citations: List[Dict[str, Any]], session_time: float) -> None:
        """
        Log comprehensive session completion with component statistics.
        
        This method provides detailed insights into how well the component
        architecture performed for Polish law and identifies any areas needing attention.
        """
        self.logger.info("=" * 80)
        self.logger.info("ENHANCED POLISH LAW AGENT SESSION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Session processing time: {session_time:.3f} seconds")
        self.logger.info(f"Citations created: {len(polish_law_citations)}")
        
        # Log component performance statistics
        self._log_component_statistics()
        
        # Log citation quality analysis with Polish law specifics
        self._log_citation_quality_analysis(polish_law_citations)
        
        self.logger.info("Component architecture demonstrated sophisticated Polish law analysis capabilities")
        self.logger.info("=" * 80)
    
    def _log_component_statistics(self) -> None:
        """
        Log detailed statistics from all components to understand system performance.
        
        This method aggregates statistics from all components to provide a comprehensive
        view of how well the sophisticated architecture is performing for Polish law.
        """
        self.logger.info("=== POLISH LAW COMPONENT PERFORMANCE STATISTICS ===")
        
        # Vector store connector statistics
        if hasattr(self.vector_connector, 'get_retrieval_statistics'):
            retrieval_stats = self.vector_connector.get_retrieval_statistics()
            self.logger.info(f"Vector Store: {retrieval_stats.get('total_documents_retrieved', 0)} documents, "
                           f"{retrieval_stats.get('enhancement_rate_percent', 0)}% enhanced, "
                           f"{retrieval_stats.get('section_coverage_rate_percent', 0)}% section coverage")
        
        # Metadata processor statistics
        if hasattr(self.metadata_processor, 'get_processing_statistics'):
            metadata_stats = self.metadata_processor.get_processing_statistics()
            self.logger.info(f"Metadata Processor: {metadata_stats.get('enhancement_rate_percent', 0)}% reconstruction rate, "
                           f"{metadata_stats.get('section_processing_rate_percent', 0)}% section processing")
        
        # Content analyzer statistics
        if hasattr(self.content_analyzer, 'get_analysis_statistics'):
            analysis_stats = self.content_analyzer.get_analysis_statistics()
            self.logger.info(f"Content Analyzer: {analysis_stats.get('guided_analysis_rate_percent', 0)}% guided analysis, "
                           f"{analysis_stats.get('section_aware_parsing_rate_percent', 0)}% section-aware parsing")
        
        # Citation builder statistics
        if hasattr(self.citation_builder, 'get_citation_statistics'):
            citation_stats = self.citation_builder.get_citation_statistics()
            self.logger.info(f"Citation Builder: {citation_stats.get('overall_precision_score', 0)}% precision score, "
                           f"{citation_stats.get('section_citation_rate', 0)}% section citations")
        
        # Response parser statistics
        if hasattr(self.response_parser, 'get_parsing_statistics'):
            parsing_stats = self.response_parser.get_parsing_statistics()
            self.logger.info(f"Response Parser: {parsing_stats.get('overall_success_rate_percent', 0)}% success rate, "
                           f"{parsing_stats.get('section_enhancement_rate_percent', 0)}% section enhancements")
    
    def _log_citation_quality_analysis(self, polish_law_citations: List[Dict[str, Any]]) -> None:
        """
        Log detailed analysis of citation quality and enhancement levels for Polish law.
        
        This analysis helps understand how well the component architecture
        is achieving the sophisticated citation creation goals for Polish legal documents.
        """
        if not polish_law_citations:
            return
        
        # Analyze enhancement levels
        enhancement_distribution = {}
        polish_law_features = {
            'section_aware': 0,
            'gazette_reference': 0,
            'parliament_info': 0
        }
        
        for citation in polish_law_citations:
            level = citation.get('enhancement_level', 'unknown')
            enhancement_distribution[level] = enhancement_distribution.get(level, 0) + 1
            
            # Track Polish law-specific features
            polish_features = citation.get('analysis_metadata', {}).get('polish_law_features', {})
            if polish_features.get('section_aware', False):
                polish_law_features['section_aware'] += 1
            if polish_features.get('gazette_reference', False):
                polish_law_features['gazette_reference'] += 1
            if polish_features.get('parliament_info', False):
                polish_law_features['parliament_info'] += 1
        
        self.logger.info("=== POLISH LAW CITATION QUALITY ANALYSIS ===")
        for level, count in sorted(enhancement_distribution.items()):
            percentage = (count / len(polish_law_citations)) * 100
            self.logger.info(f"{level.title()} citations: {count} ({percentage:.1f}%)")
        
        # Log Polish law-specific features
        self.logger.info("Polish law-specific citation features:")
        for feature, count in polish_law_features.items():
            percentage = (count / len(polish_law_citations)) * 100
            self.logger.info(f"  - {feature.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Log sample enhanced citation for verification
        sophisticated_citations = [c for c in polish_law_citations if c.get('enhancement_level') == 'sophisticated']
        if sophisticated_citations:
            sample = sophisticated_citations[0]
            self.logger.info(f"Sample enhanced Polish law citation: {sample.get('article', 'Unknown')}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the enhanced Polish law agent's performance.
        
        Returns:
            Dictionary containing detailed agent performance metrics and component statistics for Polish law
        """
        agent_stats = dict(self.session_stats)
        
        # Add component statistics
        component_stats = {}
        
        if hasattr(self.vector_connector, 'get_retrieval_statistics'):
            component_stats['vector_connector'] = self.vector_connector.get_retrieval_statistics()
        
        if hasattr(self.metadata_processor, 'get_processing_statistics'):
            component_stats['metadata_processor'] = self.metadata_processor.get_processing_statistics()
        
        if hasattr(self.content_analyzer, 'get_analysis_statistics'):
            component_stats['content_analyzer'] = self.content_analyzer.get_analysis_statistics()
        
        if hasattr(self.citation_builder, 'get_citation_statistics'):
            component_stats['citation_builder'] = self.citation_builder.get_citation_statistics()
        
        if hasattr(self.response_parser, 'get_parsing_statistics'):
            component_stats['response_parser'] = self.response_parser.get_parsing_statistics()
        
        agent_stats['component_statistics'] = component_stats
        
        # Calculate overall performance metrics
        if agent_stats['total_queries_processed'] > 0:
            success_rate = (agent_stats['successful_analyses'] / agent_stats['total_queries_processed']) * 100
            agent_stats['overall_success_rate_percent'] = round(success_rate, 1)
        else:
            agent_stats['overall_success_rate_percent'] = 0
        
        return agent_stats
    
    def log_agent_summary(self) -> None:
        """
        Log a comprehensive summary of the enhanced Polish law agent's performance.
        
        This provides a complete picture of how well the component architecture
        is working for Polish law and helps identify opportunities for optimization.
        """
        stats = self.get_agent_statistics()
        
        self.logger.info("=== ENHANCED POLISH LAW AGENT PERFORMANCE SUMMARY ===")
        self.logger.info(f"Total queries processed: {stats['total_queries_processed']}")
        self.logger.info(f"Successful analyses: {stats['successful_analyses']}")
        self.logger.info(f"Overall success rate: {stats['overall_success_rate_percent']}%")
        self.logger.info(f"Average processing time: {stats['average_processing_time']:.3f} seconds")
        self.logger.info(f"Total citations created: {stats['total_citations_created']}")
        self.logger.info(f"Component errors: {stats['component_errors']}")
        self.logger.info(f"Fallback operations: {stats['fallback_operations']}")
        
        # Summarize Polish law-specific performance
        polish_features = stats['polish_law_features']
        self.logger.info("Polish law-specific performance metrics:")
        self.logger.info(f"  - Section-aware analyses: {polish_features['section_aware_analyses']}")
        self.logger.info(f"  - Gazette reference enriched: {polish_features['gazette_reference_enriched']}")
        self.logger.info(f"  - Polish terminology validated: {polish_features['polish_terminology_validated']}")
        self.logger.info(f"  - Parliament context included: {polish_features['parliament_context_included']}")
        
        # Summarize component performance
        component_stats = stats.get('component_statistics', {})
        if component_stats:
            self.logger.info("Component architecture demonstrating sophisticated Polish law analysis capabilities")
            for component, component_data in component_stats.items():
                if isinstance(component_data, dict) and component_data:
                    key_metric = list(component_data.keys())[0]
                    self.logger.info(f"  - {component}: operational with {key_metric} tracking")


def create_enhanced_polish_law_agent(db_path: str, logger: logging.Logger) -> PolishLawAgent:
    """
    Factory function to create a configured enhanced Polish law agent.
    
    This provides a clean interface for creating agent instances with proper
    dependency injection, following the same patterns as your GDPR agent.
    """
    return PolishLawAgent(db_path, logger)