import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import logging

class InternalSecurityAgent:
    """
    Enhanced Internal Security Agent with intelligent flattened procedural metadata support.
    
    This agent represents the complete solution for working with sophisticated internal
    security procedure structures while respecting vector database constraints. It can work
    with flattened procedural metadata for efficiency while accessing complete implementation
    information when needed for precise procedure citation creation.
    
    The agent demonstrates how sophisticated security procedure functionality can be preserved
    even when adapting to technical limitations through intelligent design patterns adapted
    from the Polish law approach but tailored for internal security procedures.
    """
    
    def __init__(self):
        # Set up comprehensive logging for the enhanced security system
        self._setup_logging()
        self.logger.info("Initializing Enhanced Internal Security Agent with flattened procedural metadata support...")
        
        # Initialize language model for sophisticated security procedure analysis
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.logger.info("Language model initialized: gpt-4o-mini")
        
        # Load vector store with enhanced validation
        self._initialize_vector_store()
        
        # Configure prompts for enhanced security procedure analysis
        self._setup_prompts()
        
        self.logger.info("Enhanced Internal Security Agent initialization completed successfully")
        self.logger.info("System ready for precise procedure citation creation with flattened metadata")
    
    def _setup_logging(self):
        """
        Initialize comprehensive logging system for enhanced security agent operations.
        
        This logging system provides complete visibility into how the agent processes
        flattened procedural metadata and reconstructs sophisticated implementation information
        for precise procedure citation creation. Understanding this flow is crucial for
        system optimization and debugging of security procedure workflows.
        """
        # Create logs directory structure
        DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        log_base_dir = os.path.join(DATA_DIR, "logs")
        self.log_dir = os.path.join(log_base_dir, "internal_security_agent")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file for this enhanced session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"enhanced_security_agent_session_{timestamp}.log")
        
        # Configure detailed logger with enhanced formatting for debugging
        self.logger = logging.getLogger(f"EnhancedInternalSecurityAgent_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if logger already exists
        if not self.logger.handlers:
            # File handler for persistent detailed logs
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Enhanced formatter showing function context for debugging
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Enhanced security agent logging initialized. Log file: {log_file}")
    
    def _initialize_vector_store(self):
        """
        Initialize connection to vector store with flattened procedural metadata validation.
        
        This method not only connects to the vector store but also validates that documents
        contain the expected flattened procedural metadata structure. This validation helps
        ensure that our procedural metadata flattening approach is working correctly for
        internal security procedures.
        """
        try:
            # Initialize embeddings using the same model as processing
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.logger.info("Embeddings model initialized: text-embedding-3-large")
            
            # Load internal security procedures vector store
            security_db_path = os.path.join(os.path.dirname(__file__), "data/internal_security_db")
            if os.path.exists(security_db_path):
                self.security_db = Chroma(
                    persist_directory=security_db_path,
                    embedding_function=embeddings,
                    collection_name="internal_security_procedures"
                )
                
                # Validate flattened procedural metadata structure with enhanced testing
                try:
                    collection_count = self.security_db._collection.count()
                    self.logger.info(f"Internal security vector store loaded: {collection_count} documents")
                    
                    # Test flattened procedural metadata structure
                    if collection_count > 0:
                        test_docs = self.security_db.similarity_search("access control", k=1)
                        if test_docs:
                            test_metadata = test_docs[0].metadata
                            
                            # Validate presence of expected flattened procedural fields
                            flattened_fields = [
                                'has_enhanced_procedure', 'implementation_step_count', 'has_sub_steps',
                                'procedure_complexity', 'procedure_structure_json'
                            ]
                            
                            missing_fields = []
                            present_fields = []
                            
                            for field in flattened_fields:
                                if field in test_metadata:
                                    present_fields.append(field)
                                else:
                                    missing_fields.append(field)
                            
                            if present_fields:
                                self.logger.info(f"Flattened procedural metadata validation: {len(present_fields)}/{len(flattened_fields)} fields present")
                                self.logger.info(f"Present fields: {present_fields}")
                                
                                # Test JSON deserialization if available
                                if 'procedure_structure_json' in test_metadata:
                                    json_str = test_metadata['procedure_structure_json']
                                    if json_str:
                                        try:
                                            deserialized = json.loads(json_str)
                                            self.logger.info("✅ Procedural JSON deserialization test successful")
                                        except Exception as e:
                                            self.logger.warning(f"⚠️  Procedural JSON deserialization test failed: {e}")
                                    else:
                                        self.logger.info("Empty JSON structure found (document has no enhanced procedural metadata)")
                            
                            if missing_fields:
                                self.logger.warning(f"Missing flattened procedural metadata fields: {missing_fields}")
                                self.logger.warning("System will work but may fall back to basic citation mode")
                            else:
                                self.logger.info("✅ All expected flattened procedural metadata fields present and validated")
                        
                except Exception as e:
                    self.logger.warning(f"Could not validate flattened procedural metadata structure: {e}")
                
            else:
                self.security_db = None
                self.logger.error(f"Internal security vector store not found at: {security_db_path}")
                raise FileNotFoundError(f"Required internal security vector store not found")
                
        except Exception as e:
            self.logger.error(f"Error initializing enhanced security vector store: {e}")
            raise
    
    def _setup_prompts(self):
        """
        Configure prompt templates for enhanced security procedure analysis with implementation awareness.
        
        These prompts are designed to work with our flattened procedural metadata approach while
        encouraging the LLM to identify quotes that can benefit from precise implementation step
        citation formatting. The focus is on procedural context rather than legal context.
        """
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a specialized internal security procedure expert analyzing retrieved procedural content with enhanced implementation understanding.
            
            User Query: {user_query}
            
            Based on the following retrieved internal security procedure content, identify the most relevant procedures and implementation steps:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. Basic procedure information (precise formatting will be handled automatically using implementation metadata)
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
        
        self.logger.info("Enhanced security procedure prompt templates configured for flattened metadata processing")
    
    def _extract_flattened_procedural_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate flattened procedural metadata from vector store documents.
        
        This method processes the flattened procedural metadata created by our processing script
        and prepares it for use in sophisticated procedure citation creation. It demonstrates
        how flattened procedural metadata can be efficiently processed while preserving access
        to complete implementation information when needed.
        """
        self.logger.debug("Extracting flattened procedural metadata from vector store document")
        
        # Initialize metadata structure with safe defaults for security procedures
        extracted_info = {
            'procedure_number': metadata.get('procedure_number', ''),
            'section_number': metadata.get('section_number', ''),
            'section_title': metadata.get('section_title', ''),
            'procedure_title': metadata.get('procedure_title', ''),
            'policy_reference': metadata.get('policy_reference', ''),
            'has_enhanced_procedure': metadata.get('has_enhanced_procedure', False),
            'quick_indicators': {},
            'full_structure': None
        }
        
        # Extract quick procedural indicators for efficient processing
        if extracted_info['has_enhanced_procedure']:
            extracted_info['quick_indicators'] = {
                'implementation_step_count': metadata.get('implementation_step_count', 0),
                'has_sub_steps': metadata.get('has_sub_steps', False),
                'required_tools_count': metadata.get('required_tools_count', 0),
                'procedure_complexity': metadata.get('procedure_complexity', 'simple')
            }
            
            # Deserialize complete structure when available
            json_str = metadata.get('procedure_structure_json', '')
            if json_str:
                try:
                    extracted_info['full_structure'] = json.loads(json_str)
                    self.logger.debug(f"Successfully deserialized complete procedure structure: "
                                    f"{extracted_info['quick_indicators']['implementation_step_count']} steps, "
                                    f"complexity: {extracted_info['quick_indicators']['procedure_complexity']}")
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize complete procedure structure: {e}")
                    # Continue with quick indicators only
            
            self.logger.debug(f"Enhanced procedural metadata extracted: Procedure {extracted_info['procedure_number']}, "
                            f"{extracted_info['quick_indicators']['implementation_step_count']} steps, "
                            f"sub-steps: {extracted_info['quick_indicators']['has_sub_steps']}")
        else:
            self.logger.debug(f"Basic metadata extracted: Procedure {extracted_info['procedure_number']} "
                            f"(no enhanced structure)")
        
        return extracted_info
    
    def _parse_procedure_structure_with_hints(self, content: str, structural_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse security procedure content structure using flattened metadata hints for guidance.
        
        This method demonstrates the power of our flattened procedural metadata approach. Instead
        of parsing content blindly, we use the structural hints to guide our analysis, making it
        much more accurate and efficient than generic text parsing for security procedures.
        
        Think of this like having a blueprint while navigating a complex security implementation -
        the hints tell us what implementation steps to expect, making navigation much more reliable.
        """
        self.logger.debug("Parsing procedure content structure using flattened procedural metadata hints")
        
        content_map = {
            'steps': {},
            'parsing_successful': False,
            'used_hints': True
        }
        
        # If we don't have structural hints, fall back to basic parsing
        if not structural_hints.get('has_sub_steps', False):
            self.logger.debug("No sub-steps indicated by hints - using simplified parsing")
            return self._parse_simple_procedure_structure(content)
        
        try:
            # Use hints to guide parsing strategy for security procedures
            step_count = structural_hints.get('implementation_step_count', 0)
            complexity = structural_hints.get('procedure_complexity', 'simple')
            
            self.logger.debug(f"Using hints: expecting {step_count} implementation steps "
                            f"with {complexity} complexity")
            
            # Parse with guided expectations for security procedures
            lines = content.split('\n')
            current_step = None
            current_sub_step = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for step markers guided by expected count
                step_match = re.match(r'^Step\s+(\d+):\s+(.+)', line)
                if step_match:
                    step_num = step_match.group(1)
                    step_title = step_match.group(2)
                    
                    # Validate against expected count
                    if int(step_num) <= step_count:
                        content_map['steps'][step_num] = {
                            'start_line': line_idx,
                            'title': step_title,
                            'full_text': line,
                            'sub_steps': {}
                        }
                        current_step = step_num
                        current_sub_step = None
                        
                        self.logger.debug(f"Found expected step {step_num} at line {line_idx}")
                        continue
                
                # Look for sub-step markers (common in security procedures)
                if current_step:
                    # Look for configuration steps like "Configure", "Set", "Enable"
                    config_match = re.match(r'^(Configure|Set|Enable|Create|Deploy)\s+(.+)', line)
                    if config_match:
                        action = config_match.group(1)
                        description = config_match.group(2)
                        
                        sub_step_key = f"{action.lower()}_{len(content_map['steps'][current_step]['sub_steps']) + 1}"
                        content_map['steps'][current_step]['sub_steps'][sub_step_key] = {
                            'start_line': line_idx,
                            'action': action,
                            'description': description,
                            'full_line': line
                        }
                        current_sub_step = sub_step_key
                        
                        self.logger.debug(f"Found configuration sub-step {current_step}.{sub_step_key} at line {line_idx}")
                        continue
                
                # Handle continuation text
                if current_step:
                    if current_sub_step:
                        content_map['steps'][current_step]['sub_steps'][current_sub_step]['description'] += ' ' + line
                    else:
                        content_map['steps'][current_step]['full_text'] += ' ' + line
            
            content_map['parsing_successful'] = len(content_map['steps']) > 0
            self.logger.debug(f"Guided procedural parsing completed: found {len(content_map['steps'])} steps")
            
        except Exception as e:
            self.logger.warning(f"Error in guided procedure content parsing: {e}")
            # Fall back to basic parsing
            return self._parse_simple_procedure_structure(content)
        
        return content_map
    
    def _parse_simple_procedure_structure(self, content: str) -> Dict[str, Any]:
        """
        Parse procedure content structure without hints - fallback for documents without enhanced metadata.
        
        This method provides reliable procedure citation creation even for documents that don't
        have enhanced procedural metadata, ensuring the system works gracefully across all
        document types in your security procedure collection.
        """
        self.logger.debug("Using simple procedure parsing (no procedural hints available)")
        
        content_map = {
            'steps': {},
            'parsing_successful': False,
            'used_hints': False
        }
        
        try:
            # Basic step detection without guidance for security procedures
            lines = content.split('\n')
            current_step = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for basic step patterns
                step_match = re.match(r'^Step\s+(\d+):\s+(.+)', line)
                if step_match:
                    step_num = step_match.group(1)
                    step_title = step_match.group(2)
                    
                    content_map['steps'][step_num] = {
                        'start_line': line_idx,
                        'title': step_title,
                        'full_text': line,
                        'sub_steps': {}
                    }
                    current_step = step_num
                    continue
                
                # Simple configuration detection
                if current_step:
                    config_match = re.match(r'^(Configure|Set|Enable|Create|Deploy)\s+(.+)', line)
                    if config_match:
                        action = config_match.group(1)
                        description = config_match.group(2)
                        
                        sub_step_key = f"{action.lower()}_{len(content_map['steps'][current_step]['sub_steps']) + 1}"
                        content_map['steps'][current_step]['sub_steps'][sub_step_key] = {
                            'start_line': line_idx,
                            'action': action,
                            'description': description,
                            'full_line': line
                        }
            
            content_map['parsing_successful'] = len(content_map['steps']) > 0
            self.logger.debug(f"Simple procedural parsing completed: found {len(content_map['steps'])} steps")
            
        except Exception as e:
            self.logger.warning(f"Error in simple procedure content parsing: {e}")
        
        return content_map
    
    def _locate_quote_in_procedure_structure(self, quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Determine exactly where a specific quote appears within the security procedure structure.
        
        This method represents the core intelligence of the enhanced procedure citation system,
        working with both hint-guided and simple parsing results to create the most precise
        procedure citations possible from the available implementation information.
        """
        self.logger.debug(f"Locating quote in procedure structure: '{quote[:50]}...'")
        
        if not content_map.get('parsing_successful', False):
            self.logger.warning("Procedure parsing was not successful, cannot locate quote precisely")
            return None
        
        # Clean and normalize quote for better matching
        clean_quote = ' '.join(quote.split()).lower()
        if len(clean_quote) < 10:
            self.logger.warning(f"Quote too short for reliable matching: '{quote}'")
            return None
        
        # Search through parsed procedure structure with enhanced logging
        for step_num, step_data in content_map['steps'].items():
            # Check main step text
            step_text = ' '.join(step_data['full_text'].split()).lower()
            if clean_quote in step_text:
                # Found in step - check for sub-step specificity
                for sub_step_key, sub_step_data in step_data.get('sub_steps', {}).items():
                    sub_step_text = ' '.join(sub_step_data.get('description', '').split()).lower()
                    if clean_quote in sub_step_text:
                        self.logger.info(f"Quote precisely located: step {step_num}, sub-step {sub_step_key}")
                        return {
                            'step': step_num,
                            'sub_step': sub_step_key,
                            'location_type': 'sub_step',
                            'confidence': 'high'
                        }
                
                # Quote found in main step but not in specific sub-step
                self.logger.info(f"Quote located in main step {step_num}")
                return {
                    'step': step_num,
                    'sub_step': None,
                    'location_type': 'main_step',
                    'confidence': 'medium'
                }
        
        self.logger.warning("Could not locate quote in procedure structure")
        return None
    
    def _create_precise_procedure_citation_reference(self, metadata: Dict[str, Any], content: str, quote: str) -> str:
        """
        Create a precise security procedure citation using flattened metadata and intelligent content analysis.
        
        This method represents the culmination of our enhanced procedural approach, demonstrating how
        sophisticated procedure citation creation can be achieved even when working within vector
        database constraints. It produces citations like "Procedure 3.1: User Account Management Process, Step 2 - Initial Access Provisioning".
        """
        self.logger.info("Creating precise procedure citation using flattened metadata and content analysis")
        
        # Step 1: Extract flattened procedural metadata with enhanced validation
        extracted_info = self._extract_flattened_procedural_metadata(metadata)
        
        # Step 2: Use structural hints to guide content parsing
        if extracted_info['has_enhanced_procedure']:
            content_map = self._parse_procedure_structure_with_hints(content, extracted_info['quick_indicators'])
            self.logger.debug("Using hint-guided parsing for enhanced precision")
        else:
            content_map = self._parse_simple_procedure_structure(content)
            self.logger.debug("Using simple parsing due to limited metadata")
        
        # Step 3: Locate quote within parsed structure
        quote_location = self._locate_quote_in_procedure_structure(quote, content_map)
        
        # Step 4: Build precise procedure citation reference
        reference_parts = []
        
        # Add procedure information (always available)
        procedure_num = extracted_info['procedure_number']
        procedure_title = extracted_info['procedure_title']
        section_title = extracted_info['section_title']
        
        if procedure_num and procedure_title:
            if quote_location:
                # Create detailed reference based on location analysis
                step_num = quote_location['step']
                sub_step_key = quote_location.get('sub_step')
                
                if sub_step_key:
                    # Most precise: procedure, step, and sub-step
                    reference_parts.append(f"Procedure {procedure_num}: {procedure_title}, Step {step_num} - {sub_step_key}")
                    self.logger.info(f"Created maximum precision citation: Procedure {procedure_num}, Step {step_num}, Sub-step {sub_step_key}")
                else:
                    # Medium precision: procedure and step
                    reference_parts.append(f"Procedure {procedure_num}: {procedure_title}, Step {step_num}")
                    self.logger.info(f"Created step-level citation: Procedure {procedure_num}, Step {step_num}")
            else:
                # Basic precision: procedure only
                reference_parts.append(f"Procedure {procedure_num}: {procedure_title}")
                self.logger.info(f"Created procedure-level citation: Procedure {procedure_num}")
        elif section_title:
            # Fallback to section level
            section_num = extracted_info['section_number']
            if section_num:
                reference_parts.append(f"Section {section_num}: {section_title}")
            else:
                reference_parts.append(section_title)
        
        # Add policy reference for complete context
        policy_ref = extracted_info['policy_reference']
        if policy_ref:
            reference_parts.append(f"(Policy Reference: {policy_ref})")
            self.logger.debug(f"Added policy context: {policy_ref}")
        
        # Combine all parts into final citation
        final_citation = " ".join(reference_parts)
        
        # Log citation creation success with metadata about precision level
        precision_level = "maximum" if quote_location and quote_location.get('sub_step') else \
                         "step" if quote_location else "procedure"
        
        self.logger.info(f"Final procedure citation created with {precision_level} precision: {final_citation}")
        
        return final_citation if reference_parts else f"Internal Security Procedures - Section {extracted_info.get('section_number', 'Unknown')}"
    
    def _retrieve_relevant_security_documents(self, query: str, k: int = 8) -> Tuple[List, str, List[Dict]]:
        """
        Retrieve relevant security procedure documents with flattened metadata and prepare for enhanced analysis.
        
        This method handles document retrieval while validating that our flattened procedural
        metadata approach is working correctly. It prepares the retrieved information for
        sophisticated procedure citation creation while providing comprehensive logging.
        """
        self.logger.info(f"Starting enhanced security document retrieval for query: '{query[:100]}...'")
        self.logger.info(f"Retrieving top {k} documents with flattened procedural metadata validation")
        
        if not self.security_db:
            self.logger.error("Internal security vector store is not available")
            return [], "ERROR: Internal security vector store is not available", []
        
        try:
            # Perform similarity search with metadata filtering
            start_time = time.time()
            docs = self.security_db.similarity_search(
                query, 
                k=k,
                filter={"document_type": "internal_security_procedures"}
            )
            retrieval_time = time.time() - start_time
            
            self.logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.3f} seconds")
            
            # Prepare enhanced context and validate flattened procedural metadata
            context_pieces = []
            document_metadata = []
            
            # Statistics for flattened procedural metadata validation
            enhanced_count = 0
            complexity_distribution = {}
            
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                
                # Validate flattened procedural metadata structure
                has_enhanced = metadata.get('has_enhanced_procedure', False)
                complexity = metadata.get('procedure_complexity', 'unknown')
                
                document_metadata.append({
                    'index': i,
                    'metadata': metadata,
                    'content': doc.page_content,
                    'has_enhanced_procedure': has_enhanced,
                    'procedure_complexity': complexity
                })
                
                # Track statistics
                if has_enhanced:
                    enhanced_count += 1
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
                
                # Log detailed retrieval information
                procedure_num = metadata.get('procedure_number', 'N/A')
                doc_type = metadata.get('type', 'unknown')
                section_num = metadata.get('section_number', 'N/A')
                step_count = metadata.get('implementation_step_count', 0)
                
                self.logger.info(f"Document {i+1}: Procedure {procedure_num} ({doc_type}), "
                               f"Section {section_num}, Enhanced: {'✓' if has_enhanced else '✗'}, "
                               f"Complexity: {complexity}, Steps: {step_count}, "
                               f"Content: {len(doc.page_content)} chars")
                
                # Build enhanced reference for context display
                reference = f"Internal Security Procedures - Procedure {procedure_num}"
                if metadata.get('procedure_title'):
                    reference += f": {metadata['procedure_title']}"
                if metadata.get('section_title'):
                    reference += f" (Section {section_num}: {metadata['section_title']})"
                reference += f" - {doc_type}"
                if has_enhanced:
                    reference += f" [Enhanced: {complexity}, {step_count} steps]"
                
                # Create formatted context piece
                context_piece = f"[Document {i+1} - {reference}]\n{doc.page_content}"
                context_pieces.append(context_piece)
                
                # Log content preview for verification
                content_preview = doc.page_content[:150].replace('\n', ' ')
                self.logger.debug(f"Document {i+1} preview: {content_preview}...")
            
            # Create full context string for LLM
            retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
            
            # Log comprehensive retrieval statistics
            self.logger.info("=" * 60)
            self.logger.info("SECURITY PROCEDURE RETRIEVAL STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Enhanced procedural metadata: {enhanced_count}/{len(docs)} documents")
            self.logger.info(f"Complexity distribution: {dict(complexity_distribution)}")
            
            # Procedure and type distribution
            procedure_counts = {}
            type_counts = {}
            for doc in docs:
                procedure = doc.metadata.get('procedure_number', 'unknown')
                doc_type = doc.metadata.get('type', 'unknown')
                procedure_counts[procedure] = procedure_counts.get(procedure, 0) + 1
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            self.logger.info(f"Procedure distribution: {dict(procedure_counts)}")
            self.logger.info(f"Document type distribution: {dict(type_counts)}")
            
            enhancement_rate = (enhanced_count / len(docs) * 100) if docs else 0
            self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
            
            return docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.logger.error(f"Error during enhanced security document retrieval: {e}")
            return [], f"ERROR: Failed to retrieve documents - {str(e)}", []
    
    def _parse_llm_response_to_security_citations(self, llm_response: str, document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse LLM response and create precisely formatted security procedure citations using flattened metadata.
        
        This method combines LLM analysis with our sophisticated flattened procedural metadata
        approach to create the most precise procedure citations possible. It demonstrates how
        the complete security procedure solution works end-to-end.
        """
        self.logger.info("Starting enhanced security citation parsing with flattened procedural metadata support")
        citations = []
        
        try:
            # Split response into citation blocks
            citation_blocks = llm_response.split("CITATION ")[1:]
            self.logger.info(f"Found {len(citation_blocks)} citation blocks for enhanced procedural processing")
            
            for block_index, block in enumerate(citation_blocks):
                try:
                    # Extract citation components from LLM response
                    lines = block.strip().split('\n')
                    procedure_info = ""
                    quote = ""
                    explanation = ""
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith("- Procedure:"):
                            procedure_info = line.replace("- Procedure:", "").strip()
                        elif line.startswith("- Quote:"):
                            quote = line.replace("- Quote:", "").strip()
                            if quote.startswith('"') and quote.endswith('"'):
                                quote = quote[1:-1]
                        elif line.startswith("- Explanation:"):
                            explanation = line.replace("- Explanation:", "").strip()
                    
                    # Find best matching document using enhanced procedural metadata
                    best_match_doc = None
                    best_match_score = 0
                    
                    if quote and document_metadata:
                        for doc_meta in document_metadata:
                            # Score based on content match and procedural metadata quality
                            content_match = quote[:50].lower() in doc_meta['content'].lower()
                            has_enhanced = doc_meta.get('has_enhanced_procedure', False)
                            
                            score = 0
                            if content_match:
                                score += 10
                            if has_enhanced:
                                score += 5
                            
                            if score > best_match_score:
                                best_match_score = score
                                best_match_doc = doc_meta
                        
                        if best_match_doc:
                            self.logger.info(f"Best match for citation {block_index + 1}: "
                                           f"Procedure {best_match_doc['metadata'].get('procedure_number', 'N/A')}, "
                                           f"Enhanced: {best_match_doc.get('has_enhanced_procedure', False)}, "
                                           f"Score: {best_match_score}")
                    
                    # Create precise citation using best match
                    if best_match_doc:
                        precise_procedure = self._create_precise_procedure_citation_reference(
                            best_match_doc['metadata'], 
                            best_match_doc['content'], 
                            quote
                        )
                        self.logger.info(f"Created enhanced citation {block_index + 1}: {precise_procedure}")
                    else:
                        # Fallback citation
                        precise_procedure = f"Internal Security Procedures - {procedure_info}" if procedure_info else "Internal Security Procedures - Procedure Unknown"
                        self.logger.warning(f"Using fallback citation for block {block_index + 1}")
                    
                    # Validate and add citation
                    if precise_procedure and quote and explanation:
                        citation = {
                            "procedure": precise_procedure,
                            "quote": quote,
                            "explanation": explanation
                        }
                        citations.append(citation)
                        self.logger.info(f"Successfully created enhanced security citation {len(citations)}: {precise_procedure}")
                    else:
                        self.logger.warning(f"Incomplete citation {block_index + 1}: missing required fields")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing enhanced citation block {block_index + 1}: {e}")
                    continue
            
            self.logger.info(f"Enhanced security citation parsing completed: {len(citations)} precise citations created")
            return citations
            
        except Exception as e:
            self.logger.error(f"Error during enhanced security citation parsing: {e}")
            return []
    
    def _create_fallback_security_citations(self, retrieved_docs: List) -> List[Dict[str, Any]]:
        """
        Create fallback security procedure citations when enhanced parsing fails, using available metadata.
        
        Even when LLM parsing encounters issues, we can still create useful procedure citations
        by leveraging whatever flattened procedural metadata is available, ensuring the system
        continues to function gracefully for security procedures.
        """
        self.logger.warning("Creating enhanced fallback citations from retrieved security documents")
        fallback_citations = []
        
        try:
            # Group documents by procedure and prioritize enhanced ones
            procedure_groups = {}
            for doc in retrieved_docs[:3]:
                procedure_num = doc.metadata.get('procedure_number', 'Unknown')
                if procedure_num not in procedure_groups:
                    procedure_groups[procedure_num] = []
                procedure_groups[procedure_num].append(doc)
            
            # Create fallback citations with enhanced procedural metadata when available
            for procedure_num, docs in procedure_groups.items():
                # Prioritize documents with enhanced procedural metadata
                primary_doc = None
                for doc in docs:
                    if doc.metadata.get('has_enhanced_procedure', False):
                        primary_doc = doc
                        break
                
                if not primary_doc:
                    primary_doc = docs[0]
                
                # Create citation using available metadata
                if primary_doc.metadata.get('has_enhanced_procedure', False):
                    try:
                        precise_reference = self._create_precise_procedure_citation_reference(
                            primary_doc.metadata, primary_doc.page_content, ""
                        )
                        self.logger.info(f"Created enhanced fallback citation: {precise_reference}")
                    except Exception as e:
                        self.logger.warning(f"Enhanced fallback failed: {e}")
                        precise_reference = self._create_basic_procedure_citation_reference(primary_doc.metadata)
                else:
                    precise_reference = self._create_basic_procedure_citation_reference(primary_doc.metadata)
                
                # Create fallback citation
                citation = {
                    "procedure": precise_reference,
                    "quote": primary_doc.page_content[:200] + "..." if len(primary_doc.page_content) > 200 else primary_doc.page_content,
                    "explanation": "Retrieved relevant content from this internal security procedure"
                }
                
                fallback_citations.append(citation)
            
            # Final fallback if no procedures found
            if not fallback_citations:
                fallback_citations.append({
                    "procedure": "Internal Security Procedures",
                    "quote": "Multiple relevant procedures found in internal security documentation",
                    "explanation": "Retrieved content related to internal security procedure requirements"
                })
                self.logger.info("Created generic enhanced fallback citation")
            
            return fallback_citations
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced fallback security citations: {e}")
            return [{
                "procedure": "Internal Security Procedures (Enhanced System Error)",
                "quote": "Could not retrieve internal security procedure information",
                "explanation": f"Error occurred during enhanced processing: {str(e)}"
            }]
    
    def _create_basic_procedure_citation_reference(self, metadata: Dict[str, Any]) -> str:
        """Create a basic procedure citation reference when enhanced metadata is not available."""
        procedure_num = metadata.get('procedure_number', 'Unknown')
        procedure_title = metadata.get('procedure_title', '')
        section_num = metadata.get('section_number', '')
        section_title = metadata.get('section_title', '')
        
        reference = f"Procedure {procedure_num}"
        if procedure_title:
            reference += f": {procedure_title}"
        if section_num and section_title:
            reference += f" (Section {section_num}: {section_title})"
        
        return reference
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with enhanced flattened procedural metadata support.
        
        This method orchestrates the complete enhanced procedure citation creation process,
        demonstrating how sophisticated security procedure analysis can be achieved while working
        within vector database constraints through intelligent design adapted for internal procedures.
        """
        session_start = time.time()
        user_query = state["user_query"]
        gdpr_citations = state.get("gdpr_citations", [])
        polish_law_citations = state.get("polish_law_citations", [])
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED INTERNAL SECURITY AGENT SESSION WITH FLATTENED PROCEDURAL METADATA")
        self.logger.info(f"User query: {user_query}")
        self.logger.info(f"GDPR citations received: {len(gdpr_citations)}")
        self.logger.info(f"Polish law citations received: {len(polish_law_citations)}")
        self.logger.info("Flattened procedural metadata approach enabled for vector database compatibility")
        self.logger.info("=" * 80)
        
        print("\n🔒 [STEP 3/4] ENHANCED INTERNAL SECURITY AGENT: Analyzing with flattened procedural metadata precision...")
        
        try:
            # Step 1: Retrieve documents with flattened procedural metadata validation
            self.logger.info("STEP 1: Enhanced Security Document Retrieval with Procedural Metadata Validation")
            retrieved_docs, retrieved_context, document_metadata = self._retrieve_relevant_security_documents(user_query)
            
            if not retrieved_docs:
                self.logger.error("No security documents retrieved - cannot proceed with analysis")
                state["internal_policy_citations"] = self._create_fallback_security_citations([])
                return state
            
            # Step 2: Analyze content with LLM
            self.logger.info("STEP 2: Enhanced Security Content Analysis")
            rag_response = self._analyze_retrieved_security_content(user_query, retrieved_context)
            
            # Step 3: Parse response into precise citations using flattened procedural metadata
            self.logger.info("STEP 3: Enhanced Security Citation Creation with Flattened Procedural Metadata")
            security_citations = self._parse_llm_response_to_security_citations(rag_response, document_metadata)
            
            # Step 4: Handle parsing failures with enhanced fallback
            if not security_citations:
                self.logger.warning("Enhanced security citation parsing failed - using enhanced fallback")
                security_citations = self._create_fallback_security_citations(retrieved_docs)
            
            # Update state with enhanced results
            state["internal_policy_citations"] = security_citations
            
            # Comprehensive session completion logging
            session_time = time.time() - session_start
            enhanced_count = sum(1 for doc_meta in document_metadata if doc_meta.get('has_enhanced_procedure', False))
            
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED INTERNAL SECURITY AGENT SESSION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total session time: {session_time:.3f} seconds")
            self.logger.info(f"Documents retrieved: {len(retrieved_docs)}")
            self.logger.info(f"Enhanced procedures processed: {enhanced_count}")
            self.logger.info(f"Precise citations created: {len(security_citations)}")
            self.logger.info("Flattened procedural metadata approach successful!")
            self.logger.info("=" * 80)
            
            print(f"✅ Completed: {len(security_citations)} precise security procedure citations created")
            print(f"📊 Enhanced procedures: {enhanced_count}/{len(retrieved_docs)} documents")
            
        except Exception as e:
            session_time = time.time() - session_start
            self.logger.error("=" * 80)
            self.logger.error("ENHANCED INTERNAL SECURITY AGENT SESSION FAILED")
            self.logger.error(f"Error after {session_time:.3f} seconds: {e}")
            self.logger.error("=" * 80)
            
            # Ensure workflow continues with error information
            state["internal_policy_citations"] = [{
                "procedure": "Internal Security Procedures (Enhanced System Error)",
                "quote": "Could not retrieve internal security procedure information",
                "explanation": f"Error occurred during enhanced processing: {str(e)}"
            }]
            
            print(f"❌ Error in enhanced internal security analysis: {e}")
        
        return state
    
    def _analyze_retrieved_security_content(self, user_query: str, retrieved_context: str) -> str:
        """Analyze retrieved security content using LLM with enhanced procedural guidance."""
        self.logger.info("Starting enhanced LLM analysis of retrieved security content")
        
        try:
            rag_chain = self.rag_prompt | self.model
            start_time = time.time()
            
            response = rag_chain.invoke({
                "user_query": user_query,
                "retrieved_context": retrieved_context
            })
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Enhanced LLM analysis completed in {analysis_time:.3f} seconds")
            self.logger.info(f"Analysis response length: {len(response.content)} characters")
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error during enhanced LLM analysis: {e}")
            return f"ERROR: Failed to analyze retrieved content - {str(e)}"