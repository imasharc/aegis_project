"""
Enhanced Internal Security Agent - Sophisticated Orchestrator

This module represents the culmination of the internal security agent refactoring, combining all the
specialized components into a clean, maintainable orchestrator. Like your enhanced GDPR agent,
this agent demonstrates how architectural sophistication creates more reliable, maintainable,
and powerful systems for security procedure analysis.

The agent now follows the same design patterns as your GDPR agent:
- Single Responsibility Principle with focused components
- Dependency injection for clean interfaces
- Comprehensive error handling and graceful degradation  
- Detailed logging and statistics for monitoring
- Factory pattern for clean instantiation

This refactored agent integrates seamlessly with your enhanced processing pipeline,
using the same vector database and procedural metadata structure that your processing modules create.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .internal_security_vector_store_connector import InternalSecurityVectorStoreConnector, create_internal_security_vector_store_connector
from .internal_security_metadata_processor import InternalSecurityMetadataProcessor, create_internal_security_metadata_processor
from .internal_security_content_analyzer import InternalSecurityContentAnalyzer, create_internal_security_content_analyzer
from .internal_security_citation_builder import InternalSecurityCitationBuilder, create_internal_security_citation_builder
from .internal_security_response_parser import InternalSecurityResponseParser, create_internal_security_response_parser


class InternalSecurityAgent:
    """
    Enhanced Internal Security Agent with sophisticated modular architecture.
    
    This class represents the complete solution for security procedure analysis using the same
    architectural excellence demonstrated in your GDPR agent refactor. Instead of a monolithic
    agent, we now have a sophisticated orchestrator that coordinates specialized components
    through clean interfaces for security procedure implementation guidance.
    
    The agent demonstrates how your architectural patterns create:
    - More reliable processing through specialized components
    - Better maintainability through single responsibility modules
    - Enhanced functionality through component synergy
    - Easier testing through dependency injection
    - Improved monitoring through comprehensive statistics
    """
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize the enhanced internal security agent with sophisticated component architecture.
        
        Args:
            db_path: Path to the internal security vector database created by your processing pipeline
            logger: Configured logger for comprehensive operation tracking
        """
        self.db_path = db_path
        self.logger = logger
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize all specialized components using dependency injection
        self._initialize_components()
        
        # Configure LLM prompts for enhanced analysis
        self._setup_prompts()
        
        # Track session statistics across all operations
        self.session_stats = {
            'total_queries_processed': 0,
            'successful_analyses': 0,
            'component_errors': 0,
            'fallback_operations': 0,
            'average_processing_time': 0.0,
            'total_citations_created': 0
        }
        
        self.logger.info("Enhanced Internal Security Agent initialized with sophisticated component architecture")
    
    def _initialize_components(self) -> None:
        """
        Initialize all specialized components using factory functions and dependency injection.
        
        This method demonstrates the same architectural patterns as your GDPR agent,
        creating focused components that work together through clean interfaces for security procedures.
        """
        self.logger.info("Initializing sophisticated internal security analysis components...")
        
        # Initialize all components with proper dependency injection
        self.vector_connector = create_internal_security_vector_store_connector(self.db_path, self.logger)
        self.metadata_processor = create_internal_security_metadata_processor(self.logger)
        self.content_analyzer = create_internal_security_content_analyzer(self.logger)
        self.citation_builder = create_internal_security_citation_builder(self.logger)
        
        # Initialize response parser with all required dependencies
        self.response_parser = create_internal_security_response_parser(
            self.metadata_processor, self.content_analyzer, 
            self.citation_builder, self.logger
        )
        
        self.logger.info("All internal security analysis components initialized successfully")
    
    def _setup_prompts(self) -> None:
        """
        Configure LLM prompts optimized for the enhanced component architecture.
        
        These prompts work with the sophisticated analysis pipeline to encourage
        the LLM to provide information that our components can enhance and perfect
        for security procedure implementation guidance.
        """
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a specialized internal security procedure expert analyzing retrieved procedural content with enhanced implementation understanding.
            
            User Query: {user_query}
            
            Based on the following retrieved internal security procedure content, identify the most relevant procedures and implementation steps:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. Basic procedure information (precise formatting will be enhanced automatically using implementation metadata)
            2. A direct, specific quote of the relevant text from the retrieved context
            3. A brief explanation of its relevance to the query and how it addresses the security requirement
            
            ENHANCED PROCEDURE CITATION GUIDANCE:
            - Choose quotes that represent complete security procedures or implementation requirements
            - Prefer quotes that include implementation indicators like "Step 1:" or "Configure" when present
            - The system will automatically determine precise procedure and step references
            - Focus on the security implementation substance rather than structural formatting in your explanations
            
            Format your response as a structured list of citations in this exact format:
            
            CITATION 1:
            - Procedure: [Basic procedure info - precise structure will be determined automatically]
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including security implementation relevance]
            
            CITATION 2:
            - Procedure: [Basic procedure info - precise structure will be determined automatically] 
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including security implementation relevance]
            """
        )
        
        self.logger.info("Enhanced security procedure prompt templates configured for component integration")
    
    def connect_and_validate(self) -> bool:
        """
        Connect to the vector store and validate that all components are ready.
        
        This method performs comprehensive validation to ensure that the agent
        can work with the vector database created by your processing pipeline for security procedures.
        
        Returns:
            True if all components are ready for operation, False otherwise
        """
        self.logger.info("Connecting and validating enhanced internal security agent components...")
        
        try:
            # Connect to vector store using the sophisticated connector
            if not self.vector_connector.connect_and_validate():
                self.logger.error("Vector store connection failed - agent cannot operate")
                return False
            
            self.logger.info("✅ All internal security agent components connected and validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during internal security agent initialization: {e}")
            return False
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method demonstrating sophisticated component orchestration.
        
        This method represents the enhanced approach to security procedure analysis, orchestrating
        all the specialized components to create the most precise citations possible for implementation guidance.
        The method demonstrates how architectural sophistication enables reliable,
        maintainable operations.
        
        Args:
            state: Processing state dictionary from the multi-agent workflow
            
        Returns:
            Updated state with enhanced security procedure citations
        """
        session_start = time.time()
        user_query = state["user_query"]
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED INTERNAL SECURITY AGENT SESSION WITH COMPONENT ARCHITECTURE")
        self.logger.info(f"User query: {user_query}")
        self.logger.info("Using sophisticated component orchestration for maximum precision")
        self.logger.info("=" * 80)
        
        print("\n🔒 [STEP 3/4] ENHANCED INTERNAL SECURITY AGENT: Sophisticated analysis with component architecture...")
        
        self.session_stats['total_queries_processed'] += 1
        
        try:
            # Stage 1: Document Retrieval with Enhanced Vector Store Connector
            retrieved_docs, retrieved_context, document_metadata = self._execute_retrieval_stage(user_query)
            
            if not retrieved_docs:
                return self._handle_retrieval_failure(state)
            
            # Stage 2: LLM Analysis with Enhanced Prompts
            llm_response = self._execute_llm_analysis_stage(user_query, retrieved_context)
            
            # Stage 3: Sophisticated Response Parsing and Citation Creation
            security_citations = self._execute_citation_creation_stage(llm_response, document_metadata)
            
            # Stage 4: Final Validation and State Update
            self._execute_completion_stage(state, security_citations, session_start)
            
            return state
            
        except Exception as e:
            return self._handle_processing_error(state, e, session_start)
    
    def _execute_retrieval_stage(self, user_query: str) -> tuple:
        """
        Execute document retrieval using the sophisticated vector store connector.
        
        This stage demonstrates how component specialization improves reliability for security procedures.
        The vector store connector handles all the complexity of document retrieval
        and procedural metadata validation.
        """
        self.logger.info("STAGE 1: Enhanced Document Retrieval with Vector Store Connector")
        
        try:
            retrieved_docs, retrieved_context, document_metadata = \
                self.vector_connector.retrieve_relevant_procedures(user_query)
            
            if retrieved_docs:
                self.logger.info(f"✅ Retrieved {len(retrieved_docs)} documents with enhanced procedural metadata")
                # Log retrieval statistics
                retrieval_stats = self.vector_connector.get_retrieval_statistics()
                enhancement_rate = retrieval_stats.get('enhancement_rate_percent', 0)
                self.logger.info(f"📊 Document enhancement rate: {enhancement_rate}%")
            else:
                self.logger.warning("❌ No documents retrieved - query may be too specific or database empty")
            
            return retrieved_docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in retrieval stage: {e}")
            return [], "", []
    
    def _execute_llm_analysis_stage(self, user_query: str, retrieved_context: str) -> str:
        """
        Execute LLM analysis using enhanced prompts optimized for component processing.
        
        This stage uses prompts designed to work with the sophisticated parsing
        and analysis components that will process the LLM's response for security procedures.
        """
        self.logger.info("STAGE 2: Enhanced LLM Analysis with Component-Optimized Prompts")
        
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
            return f"ERROR: Failed to analyze retrieved security procedure content - {str(e)}"
    
    def _execute_citation_creation_stage(self, llm_response: str, 
                                        document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Execute sophisticated citation creation using the response parser and all components.
        
        This stage demonstrates the power of the component architecture for security procedures. The response
        parser orchestrates the metadata processor, content analyzer, and citation builder
        to create the most precise citations possible for implementation guidance.
        """
        self.logger.info("STAGE 3: Sophisticated Citation Creation with Component Orchestration")
        
        try:
            # Use the sophisticated response parser with all components
            security_citations = self.response_parser.parse_llm_response_to_procedure_citations(
                llm_response, document_metadata
            )
            
            if security_citations:
                self.session_stats['total_citations_created'] += len(security_citations)
                self.logger.info(f"✅ Created {len(security_citations)} enhanced security procedure citations")
                
                # Log enhancement statistics
                enhanced_count = sum(1 for cite in security_citations 
                                   if cite.get('enhancement_level') == 'sophisticated')
                basic_count = sum(1 for cite in security_citations 
                                if cite.get('enhancement_level') == 'basic')
                
                self.logger.info(f"📊 Citation enhancement: {enhanced_count} sophisticated, {basic_count} basic")
            else:
                self.logger.warning("❌ No citations created - using component fallback mechanisms")
                security_citations = self._create_emergency_fallback_citations(document_metadata)
            
            return security_citations
            
        except Exception as e:
            self.session_stats['component_errors'] += 1
            self.logger.error(f"Error in citation creation stage: {e}")
            return self._create_emergency_fallback_citations(document_metadata)
    
    def _execute_completion_stage(self, state: Dict[str, Any], security_citations: List[Dict[str, Any]], 
                                 session_start: float) -> None:
        """
        Execute completion stage with comprehensive statistics and state update.
        
        This stage demonstrates how component architecture enables comprehensive
        monitoring and statistics collection across all system operations for security procedures.
        """
        self.logger.info("STAGE 4: Session Completion with Component Statistics")
        
        # Update processing statistics
        session_time = time.time() - session_start
        self.session_stats['average_processing_time'] = \
            (self.session_stats['average_processing_time'] * (self.session_stats['total_queries_processed'] - 1) + session_time) / \
            self.session_stats['total_queries_processed']
        
        if security_citations:
            self.session_stats['successful_analyses'] += 1
        
        # Update state with results (using the same key name for compatibility)
        state["internal_policy_citations"] = security_citations
        
        # Log comprehensive session completion
        self._log_session_completion(security_citations, session_time)
        
        print(f"✅ Completed: {len(security_citations)} sophisticated security procedure citations created")
        print(f"⏱️  Processing time: {session_time:.3f} seconds")
    
    def _handle_retrieval_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle retrieval failure with graceful degradation.
        
        This method demonstrates how component architecture enables reliable
        error handling and fallback mechanisms for security procedures.
        """
        self.logger.error("Document retrieval failed - creating minimal fallback citations")
        self.session_stats['fallback_operations'] += 1
        
        fallback_citations = [{
            "procedure": "Internal Security Procedures",
            "quote": "Internal security vector store is not available or empty",
            "explanation": "Could not retrieve security procedure documents for analysis"
        }]
        
        state["internal_policy_citations"] = fallback_citations
        print("❌ Document retrieval failed - provided fallback citations")
        
        return state
    
    def _handle_processing_error(self, state: Dict[str, Any], error: Exception, 
                                session_start: float) -> Dict[str, Any]:
        """
        Handle processing errors with comprehensive error information and graceful degradation.
        
        This method ensures the workflow continues even when the enhanced agent
        encounters issues, demonstrating robust error handling for security procedures.
        """
        session_time = time.time() - session_start
        self.session_stats['component_errors'] += 1
        
        self.logger.error("=" * 80)
        self.logger.error("ENHANCED INTERNAL SECURITY AGENT SESSION FAILED")
        self.logger.error(f"Error after {session_time:.3f} seconds: {error}")
        self.logger.error("=" * 80)
        
        # Provide error information in citations to maintain workflow
        error_citations = [{
            "procedure": "Internal Security Procedures (Enhanced Agent Error)",
            "quote": "Could not complete enhanced security procedure analysis",
            "explanation": f"Error occurred during sophisticated processing: {str(error)}"
        }]
        
        state["internal_policy_citations"] = error_citations
        
        print(f"❌ Enhanced security procedure analysis error: {error}")
        return state
    
    def _create_emergency_fallback_citations(self, document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create emergency fallback citations when all sophisticated processing fails.
        
        This method uses basic citation building capabilities to ensure the system
        provides some output even when the advanced features encounter issues for security procedures.
        """
        self.logger.warning("Creating emergency fallback citations using basic methods")
        self.session_stats['fallback_operations'] += 1
        
        try:
            if document_metadata:
                # Use basic citation builder for simple fallback
                fallback_citation = self.citation_builder.create_basic_procedure_citation_from_metadata(
                    document_metadata[0]['metadata']
                )
                
                return [{
                    "procedure": fallback_citation,
                    "quote": "Emergency fallback - retrieved relevant security procedure content",
                    "explanation": "Created using emergency fallback when enhanced processing failed"
                }]
            else:
                return [{
                    "procedure": "Internal Security Procedures",
                    "quote": "No security procedure documents available for analysis",
                    "explanation": "Emergency fallback when no documents could be retrieved"
                }]
                
        except Exception as e:
            self.logger.error(f"Emergency fallback creation failed: {e}")
            return [{
                "procedure": "Internal Security Procedures (System Error)",
                "quote": "All security procedure analysis methods failed",
                "explanation": "System error - could not create any citations"
            }]
    
    def _log_session_completion(self, security_citations: List[Dict[str, Any]], session_time: float) -> None:
        """
        Log comprehensive session completion with component statistics.
        
        This method provides detailed insights into how well the component
        architecture performed and identifies any areas needing attention for security procedures.
        """
        self.logger.info("=" * 80)
        self.logger.info("ENHANCED INTERNAL SECURITY AGENT SESSION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Session processing time: {session_time:.3f} seconds")
        self.logger.info(f"Citations created: {len(security_citations)}")
        
        # Log component performance statistics
        self._log_component_statistics()
        
        # Log citation quality analysis
        self._log_citation_quality_analysis(security_citations)
        
        self.logger.info("Component architecture demonstrated sophisticated security procedure analysis capabilities")
        self.logger.info("=" * 80)
    
    def _log_component_statistics(self) -> None:
        """
        Log detailed statistics from all components to understand system performance.
        
        This method aggregates statistics from all components to provide a comprehensive
        view of how well the sophisticated architecture is performing for security procedures.
        """
        self.logger.info("=== COMPONENT PERFORMANCE STATISTICS ===")
        
        # Vector store connector statistics
        if hasattr(self.vector_connector, 'get_retrieval_statistics'):
            retrieval_stats = self.vector_connector.get_retrieval_statistics()
            self.logger.info(f"Vector Store: {retrieval_stats.get('total_documents_retrieved', 0)} documents, "
                           f"{retrieval_stats.get('enhancement_rate_percent', 0)}% enhanced")
        
        # Metadata processor statistics
        if hasattr(self.metadata_processor, 'get_processing_statistics'):
            metadata_stats = self.metadata_processor.get_processing_statistics()
            self.logger.info(f"Metadata Processor: {metadata_stats.get('enhancement_rate_percent', 0)}% reconstruction rate")
        
        # Content analyzer statistics
        if hasattr(self.content_analyzer, 'get_analysis_statistics'):
            analysis_stats = self.content_analyzer.get_analysis_statistics()
            self.logger.info(f"Content Analyzer: {analysis_stats.get('guided_analysis_rate_percent', 0)}% guided analysis")
        
        # Citation builder statistics
        if hasattr(self.citation_builder, 'get_citation_statistics'):
            citation_stats = self.citation_builder.get_citation_statistics()
            self.logger.info(f"Citation Builder: {citation_stats.get('overall_precision_score', 0)}% precision score")
        
        # Response parser statistics
        if hasattr(self.response_parser, 'get_parsing_statistics'):
            parsing_stats = self.response_parser.get_parsing_statistics()
            self.logger.info(f"Response Parser: {parsing_stats.get('overall_success_rate_percent', 0)}% success rate")
    
    def _log_citation_quality_analysis(self, security_citations: List[Dict[str, Any]]) -> None:
        """
        Log detailed analysis of citation quality and enhancement levels.
        
        This analysis helps understand how well the component architecture
        is achieving the sophisticated citation creation goals for security procedures.
        """
        if not security_citations:
            return
        
        # Analyze enhancement levels
        enhancement_distribution = {}
        for citation in security_citations:
            level = citation.get('enhancement_level', 'unknown')
            enhancement_distribution[level] = enhancement_distribution.get(level, 0) + 1
        
        self.logger.info("=== CITATION QUALITY ANALYSIS ===")
        for level, count in sorted(enhancement_distribution.items()):
            percentage = (count / len(security_citations)) * 100
            self.logger.info(f"{level.title()} citations: {count} ({percentage:.1f}%)")
        
        # Log sample enhanced citation for verification
        sophisticated_citations = [c for c in security_citations if c.get('enhancement_level') == 'sophisticated']
        if sophisticated_citations:
            sample = sophisticated_citations[0]
            self.logger.info(f"Sample enhanced citation: {sample.get('procedure', 'Unknown')}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the enhanced agent's performance.
        
        Returns:
            Dictionary containing detailed agent performance metrics and component statistics
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
        Log a comprehensive summary of the enhanced agent's performance.
        
        This provides a complete picture of how well the component architecture
        is working and helps identify opportunities for optimization for security procedures.
        """
        stats = self.get_agent_statistics()
        
        self.logger.info("=== ENHANCED INTERNAL SECURITY AGENT PERFORMANCE SUMMARY ===")
        self.logger.info(f"Total queries processed: {stats['total_queries_processed']}")
        self.logger.info(f"Successful analyses: {stats['successful_analyses']}")
        self.logger.info(f"Overall success rate: {stats['overall_success_rate_percent']}%")
        self.logger.info(f"Average processing time: {stats['average_processing_time']:.3f} seconds")
        self.logger.info(f"Total citations created: {stats['total_citations_created']}")
        self.logger.info(f"Component errors: {stats['component_errors']}")
        self.logger.info(f"Fallback operations: {stats['fallback_operations']}")
        
        # Summarize component performance
        component_stats = stats.get('component_statistics', {})
        if component_stats:
            self.logger.info("Component architecture demonstrating sophisticated analysis capabilities")
            for component, component_data in component_stats.items():
                if isinstance(component_data, dict) and component_data:
                    key_metric = list(component_data.keys())[0]
                    self.logger.info(f"  - {component}: operational with {key_metric} tracking")


def create_enhanced_internal_security_agent(db_path: str, logger: logging.Logger) -> InternalSecurityAgent:
    """
    Factory function to create a configured enhanced internal security agent.
    
    This provides a clean interface for creating agent instances with proper
    dependency injection, following the same patterns as your GDPR agent.
    """
    return InternalSecurityAgent(db_path, logger)