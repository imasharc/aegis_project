"""
Internal Security Vector Store Connector

This module handles all interactions with the Chroma vector store for internal security procedure documents.
Following the Single Responsibility Principle established in the GDPR refactor, this module's only job is to
safely connect to the vector store, validate procedural metadata structure, and retrieve relevant documents
with their flattened metadata intact.

Key responsibilities:
- Connect to and validate the internal security procedure vector store
- Perform similarity searches with proper filtering for procedural documents
- Validate flattened procedural metadata structure in retrieved documents
- Provide comprehensive logging and error handling for vector operations
- Track retrieval statistics specific to security procedure complexity patterns

This component demonstrates how the same architectural patterns from legal document processing
can be successfully adapted to procedural document workflows while maintaining the same
reliability and maintainability benefits.
"""

import os
import time
import logging
from typing import Dict, List, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class InternalSecurityVectorStoreConnector:
    """
    Handles connection and interaction with the internal security procedure vector store.
    
    This class encapsulates all the logic for safely connecting to the vector store
    and retrieving procedural documents while validating that the flattened metadata approach
    is working correctly for security procedures. It provides a clean interface for document
    retrieval operations while handling all the complexity of vector store management.
    
    The connector is specifically adapted for procedural documents, understanding that
    security procedures have different organizational patterns compared to legal documents
    while maintaining the same reliability and error handling standards.
    """
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize the internal security vector store connector.
        
        Args:
            db_path: Path to the Chroma database directory
            logger: Configured logger instance for detailed operation tracking
        """
        self.db_path = db_path
        self.logger = logger
        self.security_db = None
        
        # Initialize retrieval statistics for procedural documents
        self.retrieval_stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'enhanced_procedure_documents': 0,
            'retrieval_errors': 0,
            'average_retrieval_time': 0.0,
            'procedure_complexity_distribution': {},
            'implementation_steps_found': 0
        }
        
        self.logger.info("Internal Security Vector Store Connector initialized")
    
    def connect_and_validate(self) -> bool:
        """
        Connect to the internal security vector store and validate its procedural metadata structure.
        
        This method establishes the connection to the vector store and performs
        comprehensive validation to ensure that the flattened procedural metadata approach
        is working correctly with the stored security procedure documents.
        
        Returns:
            True if connection and validation succeed, False otherwise
        """
        self.logger.info("Connecting to internal security vector store with procedural metadata validation...")
        
        try:
            # Initialize embeddings using the same model as processing
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.logger.info("Embeddings model initialized: text-embedding-3-large")
            
            # Connect to the internal security vector store
            if not os.path.exists(self.db_path):
                self.logger.error(f"Internal security vector store not found at: {self.db_path}")
                return False
            
            self.security_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=embeddings,
                collection_name="internal_security_procedures"
            )
            
            # Validate the connection and procedural metadata structure
            return self._validate_procedural_store_structure()
            
        except Exception as e:
            self.logger.error(f"Error connecting to internal security vector store: {e}")
            return False
    
    def _validate_procedural_store_structure(self) -> bool:
        """
        Validate that the vector store contains expected procedural documents with flattened metadata.
        
        This validation ensures that the documents were processed correctly by the
        enhanced processing pipeline and contain the procedural metadata structure needed for
        sophisticated procedure citation creation.
        """
        try:
            collection_count = self.security_db._collection.count()
            self.logger.info(f"Internal security vector store validation: {collection_count} documents found")
            
            if collection_count == 0:
                self.logger.warning("Internal security vector store is empty - no documents to validate")
                return True  # Empty store is valid, just not useful
            
            # Test retrieval and procedural metadata structure
            test_docs = self.security_db.similarity_search("access control", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                
                # Validate presence of expected flattened fields for security procedures
                expected_fields = [
                    'document_type', 'procedure_number', 'has_enhanced_procedure',
                    'implementation_step_count', 'procedure_complexity'
                ]
                
                present_fields = []
                missing_fields = []
                
                for field in expected_fields:
                    if field in test_metadata:
                        present_fields.append(field)
                    else:
                        missing_fields.append(field)
                
                self.logger.info(f"Procedural metadata validation: {len(present_fields)}/{len(expected_fields)} expected fields present")
                
                if missing_fields:
                    self.logger.warning(f"Missing expected procedural metadata fields: {missing_fields}")
                    self.logger.warning("Store may be from older processing - basic functionality will work")
                
                # Test enhanced procedural structure if available
                if test_metadata.get('has_enhanced_procedure', False):
                    self._validate_enhanced_procedural_structure(test_metadata)
                
                self.logger.info("✅ Internal security vector store structure validation completed")
                return True
            else:
                self.logger.warning("Could not retrieve test documents for validation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating internal security vector store structure: {e}")
            return False
    
    def _validate_enhanced_procedural_structure(self, metadata: Dict[str, Any]) -> None:
        """
        Validate enhanced procedural metadata structure in a sample document.
        
        This checks that the sophisticated flattened procedural metadata is properly preserved
        and can be accessed for precise procedure citation creation with implementation details.
        """
        # Check for JSON structure preservation
        json_str = metadata.get('procedure_structure_json', '')
        if json_str:
            try:
                import json
                deserialized = json.loads(json_str)
                self.logger.info("✅ Enhanced procedural structure JSON deserialization successful")
            except Exception as e:
                self.logger.warning(f"⚠️  Enhanced procedural structure JSON deserialization failed: {e}")
        else:
            self.logger.info("No JSON structure found in sample procedural document")
        
        # Log procedural structural indicators
        step_count = metadata.get('implementation_step_count', 0)
        has_sub_steps = metadata.get('has_sub_steps', False)
        complexity = metadata.get('procedure_complexity', 'unknown')
        
        self.logger.info(f"Enhanced procedural structure indicators: {step_count} implementation steps, "
                        f"sub-steps: {has_sub_steps}, complexity: {complexity}")
    
    def retrieve_relevant_procedures(self, query: str, k: int = 8) -> Tuple[List, str, List[Dict]]:
        """
        Retrieve relevant security procedure documents with comprehensive procedural metadata tracking.
        
        This method performs similarity search while tracking statistics about
        the retrieved procedural documents and their metadata quality. It provides detailed
        logging for debugging and optimization purposes specific to security procedures.
        
        Args:
            query: Search query for similarity search
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, formatted_context, document_metadata_list)
        """
        if not self.security_db:
            error_msg = "Internal security vector store is not connected"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
        
        self.logger.info(f"Retrieving security procedure documents for query: '{query[:100]}...'")
        self.logger.info(f"Requesting top {k} documents with procedural metadata analysis")
        
        self.retrieval_stats['total_queries'] += 1
        
        try:
            # Perform similarity search with timing and procedural filtering
            start_time = time.time()
            docs = self.security_db.similarity_search(
                query, 
                k=k,
                filter={"document_type": "internal_security_procedures"}  # Ensure we only get security procedures
            )
            retrieval_time = time.time() - start_time
            
            # Update statistics
            self.retrieval_stats['total_documents_retrieved'] += len(docs)
            self.retrieval_stats['average_retrieval_time'] = \
                (self.retrieval_stats['average_retrieval_time'] * (self.retrieval_stats['total_queries'] - 1) + retrieval_time) / \
                self.retrieval_stats['total_queries']
            
            self.logger.info(f"Retrieved {len(docs)} security procedure documents in {retrieval_time:.3f} seconds")
            
            # Process and analyze retrieved procedural documents
            context_pieces, document_metadata = self._process_retrieved_procedures(docs)
            
            # Create formatted context for LLM
            retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
            
            # Log comprehensive retrieval statistics
            self._log_procedural_retrieval_statistics(docs, document_metadata)
            
            return docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.retrieval_stats['retrieval_errors'] += 1
            error_msg = f"Error during security procedure document retrieval: {e}"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
    
    def _process_retrieved_procedures(self, docs: List) -> Tuple[List[str], List[Dict]]:
        """
        Process retrieved procedural documents and extract metadata for analysis.
        
        This method analyzes each retrieved procedural document to understand its metadata
        quality and creates formatted context pieces for LLM processing with security
        procedure-specific information.
        """
        context_pieces = []
        document_metadata = []
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            
            # Extract and validate procedural metadata structure
            has_enhanced = metadata.get('has_enhanced_procedure', False)
            complexity = metadata.get('procedure_complexity', 'unknown')
            step_count = metadata.get('implementation_step_count', 0)
            
            # Track enhanced procedural metadata statistics
            if has_enhanced:
                self.retrieval_stats['enhanced_procedure_documents'] += 1
                self.retrieval_stats['implementation_steps_found'] += step_count
                
                # Track complexity distribution
                self.retrieval_stats['procedure_complexity_distribution'][complexity] = \
                    self.retrieval_stats['procedure_complexity_distribution'].get(complexity, 0) + 1
            
            # Create document metadata entry for further processing
            doc_metadata = {
                'index': i,
                'metadata': metadata,
                'content': doc.page_content,
                'has_enhanced_procedure': has_enhanced,
                'procedure_complexity': complexity,
                'implementation_step_count': step_count
            }
            document_metadata.append(doc_metadata)
            
            # Build enhanced reference for context display
            reference = self._build_procedure_reference(metadata, has_enhanced, complexity, step_count)
            
            # Create formatted context piece
            context_piece = f"[Document {i+1} - {reference}]\n{doc.page_content}"
            context_pieces.append(context_piece)
            
            # Log document details for verification
            self._log_procedure_details(i, metadata, has_enhanced, complexity, step_count, doc.page_content)
        
        return context_pieces, document_metadata
    
    def _build_procedure_reference(self, metadata: Dict[str, Any], 
                                  has_enhanced: bool, complexity: str, step_count: int) -> str:
        """
        Build a comprehensive reference string for a procedural document.
        
        This creates a human-readable reference that shows the procedure's
        structure and metadata quality for logging and debugging purposes.
        """
        procedure_num = metadata.get('procedure_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        section_num = metadata.get('section_number', 'N/A')
        procedure_title = metadata.get('procedure_title', '')
        
        reference = f"Internal Security Procedures - Procedure {procedure_num}"
        
        if procedure_title:
            reference += f": {procedure_title}"
        
        if metadata.get('section_title'):
            reference += f" (Section {section_num}: {metadata['section_title']})"
        
        reference += f" - {doc_type}"
        
        if has_enhanced:
            reference += f" [Enhanced: {complexity}, {step_count} steps]"
        
        # Add security-specific metadata
        if metadata.get('classification_level'):
            reference += f" [Classification: {metadata['classification_level']}]"
        
        return reference
    
    def _log_procedure_details(self, index: int, metadata: Dict[str, Any], 
                              has_enhanced: bool, complexity: str, step_count: int, content: str) -> None:
        """
        Log detailed information about a retrieved procedural document.
        
        This provides comprehensive visibility into what procedural documents were retrieved
        and their metadata quality for debugging and optimization.
        """
        procedure_num = metadata.get('procedure_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        section_num = metadata.get('section_number', 'N/A')
        classification = metadata.get('classification_level', 'N/A')
        
        self.logger.info(f"Security Procedure Document {index+1}: Procedure {procedure_num} ({doc_type}), "
                        f"Section {section_num}, Enhanced: {'✓' if has_enhanced else '✗'}, "
                        f"Complexity: {complexity}, Steps: {step_count}, "
                        f"Classification: {classification}, Content: {len(content)} chars")
        
        # Log content preview for verification
        if index < 3:  # Only log previews for first few documents
            content_preview = content[:150].replace('\n', ' ')
            self.logger.debug(f"Security Procedure Document {index+1} preview: {content_preview}...")
    
    def _log_procedural_retrieval_statistics(self, docs: List, document_metadata: List[Dict]) -> None:
        """
        Log comprehensive statistics about the retrieval operation.
        
        This provides insights into the quality and patterns of retrieved procedural documents
        to help optimize the retrieval and citation processes for security procedures.
        """
        enhanced_count = sum(1 for doc_meta in document_metadata 
                           if doc_meta.get('has_enhanced_procedure', False))
        
        # Calculate complexity distribution
        complexity_distribution = {}
        for doc_meta in document_metadata:
            complexity = doc_meta.get('procedure_complexity', 'unknown')
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        # Calculate procedure and type distributions
        procedure_counts = {}
        type_counts = {}
        classification_counts = {}
        
        for doc in docs:
            procedure = doc.metadata.get('procedure_number', 'unknown')
            doc_type = doc.metadata.get('type', 'unknown')
            classification = doc.metadata.get('classification_level', 'unclassified')
            
            procedure_counts[procedure] = procedure_counts.get(procedure, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        enhancement_rate = (enhanced_count / len(docs) * 100) if docs else 0
        total_steps = sum(doc_meta.get('implementation_step_count', 0) for doc_meta in document_metadata)
        
        # Log comprehensive statistics
        self.logger.info("=" * 60)
        self.logger.info("SECURITY PROCEDURE DOCUMENT RETRIEVAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Documents retrieved: {len(docs)}")
        self.logger.info(f"Enhanced procedural metadata: {enhanced_count}/{len(docs)} documents")
        self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
        self.logger.info(f"Total implementation steps: {total_steps}")
        self.logger.info(f"Complexity distribution: {dict(complexity_distribution)}")
        self.logger.info(f"Procedure distribution: {dict(procedure_counts)}")
        self.logger.info(f"Document type distribution: {dict(type_counts)}")
        self.logger.info(f"Classification distribution: {dict(classification_counts)}")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all retrieval operations.
        
        Returns:
            Dictionary containing detailed retrieval statistics for procedural documents
        """
        stats = dict(self.retrieval_stats)
        
        # Calculate enhancement rate across all retrievals
        if stats['total_documents_retrieved'] > 0:
            enhancement_rate = (stats['enhanced_procedure_documents'] / stats['total_documents_retrieved']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
            
            # Calculate average steps per enhanced procedure
            if stats['enhanced_procedure_documents'] > 0:
                avg_steps = stats['implementation_steps_found'] / stats['enhanced_procedure_documents']
                stats['average_steps_per_procedure'] = round(avg_steps, 1)
            else:
                stats['average_steps_per_procedure'] = 0
        else:
            stats['enhancement_rate_percent'] = 0
            stats['average_steps_per_procedure'] = 0
        
        return stats
    
    def is_connected(self) -> bool:
        """
        Check if the vector store connection is active.
        
        Returns:
            True if connected and ready for operations, False otherwise
        """
        return self.security_db is not None


def create_internal_security_vector_store_connector(db_path: str, logger: logging.Logger) -> InternalSecurityVectorStoreConnector:
    """
    Factory function to create a configured internal security vector store connector.
    
    This provides a clean interface for creating connector instances with
    proper dependency injection of configuration and logger.
    """
    return InternalSecurityVectorStoreConnector(db_path, logger)