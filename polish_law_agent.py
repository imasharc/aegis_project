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

class PolishLawAgent:
    """
    Enhanced Polish Law Agent with intelligent flattened metadata support.
    
    This agent represents the complete solution to working with sophisticated legal
    document structure while respecting vector database constraints. It can work with
    flattened metadata for efficiency while accessing complete structural information
    when needed for precise citation creation.
    
    The agent demonstrates how sophisticated functionality can be preserved even when
    adapting to technical limitations through intelligent design patterns.
    """
    
    def __init__(self):
        # Set up comprehensive logging for the enhanced system
        self._setup_logging()
        self.logger.info("Initializing Enhanced Polish Law Agent with flattened metadata support...")
        
        # Initialize language model for sophisticated legal analysis
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.logger.info("Language model initialized: gpt-4o-mini")
        
        # Load vector store with enhanced validation
        self._initialize_vector_store()
        
        # Configure prompts for enhanced analysis
        self._setup_prompts()
        
        self.logger.info("Enhanced Polish Law Agent initialization completed successfully")
        self.logger.info("System ready for precise citation creation with flattened metadata")
    
    def _setup_logging(self):
        """
        Initialize comprehensive logging system for enhanced agent operations.
        
        This logging system provides complete visibility into how the agent processes
        flattened metadata and reconstructs sophisticated structural information for
        precise citation creation. Understanding this flow is crucial for system
        optimization and debugging.
        """
        # Create logs directory structure
        DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        log_base_dir = os.path.join(DATA_DIR, "logs")
        self.log_dir = os.path.join(log_base_dir, "polish_law_agent")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file for this enhanced session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"enhanced_agent_session_{timestamp}.log")
        
        # Configure detailed logger with enhanced formatting for debugging
        self.logger = logging.getLogger(f"EnhancedPolishLawAgent_{timestamp}")
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
        
        self.logger.info(f"Enhanced agent logging initialized. Log file: {log_file}")
    
    def _initialize_vector_store(self):
        """
        Initialize connection to vector store with flattened metadata validation.
        
        This method not only connects to the vector store but also validates that
        documents contain the expected flattened metadata structure. This validation
        helps ensure that our metadata flattening approach is working correctly.
        """
        try:
            # Initialize embeddings using the same model as processing
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.logger.info("Embeddings model initialized: text-embedding-3-large")
            
            # Load Polish law vector store
            polish_db_path = os.path.join(os.path.dirname(__file__), "data/polish_law_db")
            if os.path.exists(polish_db_path):
                self.polish_law_db = Chroma(
                    persist_directory=polish_db_path,
                    embedding_function=embeddings,
                    collection_name="polish_data_protection_law"
                )
                
                # Validate flattened metadata structure with enhanced testing
                try:
                    collection_count = self.polish_law_db._collection.count()
                    self.logger.info(f"Polish law vector store loaded: {collection_count} documents")
                    
                    # Test flattened metadata structure
                    if collection_count > 0:
                        test_docs = self.polish_law_db.similarity_search("Article 1", k=1)
                        if test_docs:
                            test_metadata = test_docs[0].metadata
                            
                            # Validate presence of expected flattened fields
                            flattened_fields = [
                                'has_enhanced_structure', 'paragraph_count', 'has_sub_paragraphs',
                                'complexity_level', 'article_structure_json'
                            ]
                            
                            missing_fields = []
                            present_fields = []
                            
                            for field in flattened_fields:
                                if field in test_metadata:
                                    present_fields.append(field)
                                else:
                                    missing_fields.append(field)
                            
                            if present_fields:
                                self.logger.info(f"Flattened metadata validation: {len(present_fields)}/{len(flattened_fields)} fields present")
                                self.logger.info(f"Present fields: {present_fields}")
                                
                                # Test JSON deserialization if available
                                if 'article_structure_json' in test_metadata:
                                    json_str = test_metadata['article_structure_json']
                                    if json_str:
                                        try:
                                            deserialized = json.loads(json_str)
                                            self.logger.info("✅ JSON deserialization test successful")
                                        except Exception as e:
                                            self.logger.warning(f"⚠️  JSON deserialization test failed: {e}")
                                    else:
                                        self.logger.info("Empty JSON structure found (document has no enhanced metadata)")
                            
                            if missing_fields:
                                self.logger.warning(f"Missing flattened metadata fields: {missing_fields}")
                                self.logger.warning("System will work but may fall back to basic citation mode")
                            else:
                                self.logger.info("✅ All expected flattened metadata fields present and validated")
                        
                except Exception as e:
                    self.logger.warning(f"Could not validate flattened metadata structure: {e}")
                
            else:
                self.polish_law_db = None
                self.logger.error(f"Polish law vector store not found at: {polish_db_path}")
                raise FileNotFoundError(f"Required Polish law vector store not found")
                
        except Exception as e:
            self.logger.error(f"Error initializing enhanced vector store: {e}")
            raise
    
    def _setup_prompts(self):
        """
        Configure prompt templates for enhanced legal analysis with structural awareness.
        
        These prompts are designed to work with our flattened metadata approach while
        encouraging the LLM to identify quotes that can benefit from precise structural
        citation formatting.
        """
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a specialized Polish data protection law expert analyzing retrieved legal content with enhanced structural understanding.
            
            User Query: {user_query}
            
            Based on the following retrieved Polish law content, identify the most relevant provisions:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. Basic article information (precise formatting will be handled automatically using structural metadata)
            2. A direct, specific quote of the relevant text from the retrieved context
            3. A brief explanation of its relevance to the query and how it relates to GDPR
            
            ENHANCED CITATION GUIDANCE:
            - Choose quotes that represent complete legal concepts or requirements
            - Prefer quotes that include structural indicators like "1)" or "a)" when present
            - The system will automatically determine precise paragraph and sub-paragraph references
            - Focus on the legal substance rather than structural formatting in your explanations
            
            Format your response as a structured list of citations in this exact format:
            
            CITATION 1:
            - Article: [Basic article info - precise structure will be determined automatically]
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including GDPR relationship]
            
            CITATION 2:
            - Article: [Basic article info - precise structure will be determined automatically]
            - Quote: "[Direct, specific quote from retrieved context]"
            - Explanation: [Brief explanation including GDPR relationship]
            """
        )
        
        self.logger.info("Enhanced prompt templates configured for flattened metadata processing")
    
    def _extract_flattened_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate flattened metadata from vector store documents.
        
        This method processes the flattened metadata created by our processing script
        and prepares it for use in sophisticated citation creation. It demonstrates
        how flattened metadata can be efficiently processed while preserving access
        to complete structural information when needed.
        """
        self.logger.debug("Extracting flattened metadata from vector store document")
        
        # Initialize metadata structure with safe defaults
        extracted_info = {
            'article_number': metadata.get('article_number', ''),
            'chapter_number': metadata.get('chapter_number', ''),
            'chapter_title': metadata.get('chapter_title', ''),
            'has_enhanced_structure': metadata.get('has_enhanced_structure', False),
            'quick_indicators': {},
            'full_structure': None
        }
        
        # Extract quick structural indicators for efficient processing
        if extracted_info['has_enhanced_structure']:
            extracted_info['quick_indicators'] = {
                'paragraph_count': metadata.get('paragraph_count', 0),
                'has_sub_paragraphs': metadata.get('has_sub_paragraphs', False),
                'numbering_style': metadata.get('numbering_style', ''),
                'complexity_level': metadata.get('complexity_level', 'simple')
            }
            
            # Deserialize complete structure when available
            json_str = metadata.get('article_structure_json', '')
            if json_str:
                try:
                    extracted_info['full_structure'] = json.loads(json_str)
                    self.logger.debug(f"Successfully deserialized complete structure: "
                                    f"{extracted_info['quick_indicators']['paragraph_count']} paragraphs, "
                                    f"complexity: {extracted_info['quick_indicators']['complexity_level']}")
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize complete structure: {e}")
                    # Continue with quick indicators only
            
            self.logger.debug(f"Enhanced metadata extracted: Article {extracted_info['article_number']}, "
                            f"{extracted_info['quick_indicators']['paragraph_count']} paragraphs, "
                            f"sub-paragraphs: {extracted_info['quick_indicators']['has_sub_paragraphs']}")
        else:
            self.logger.debug(f"Basic metadata extracted: Article {extracted_info['article_number']} "
                            f"(no enhanced structure)")
        
        return extracted_info
    
    def _parse_content_structure_with_hints(self, content: str, structural_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse legal content structure using flattened metadata hints for guidance.
        
        This method demonstrates the power of our flattened metadata approach. Instead
        of parsing content blindly, we use the structural hints to guide our analysis,
        making it much more accurate and efficient than generic text parsing.
        
        Think of this like having a map while exploring a complex building - the hints
        tell us what to expect, making navigation much more reliable.
        """
        self.logger.debug("Parsing content structure using flattened metadata hints")
        
        content_map = {
            'paragraphs': {},
            'parsing_successful': False,
            'used_hints': True
        }
        
        # If we don't have structural hints, fall back to basic parsing
        if not structural_hints.get('has_sub_paragraphs', False):
            self.logger.debug("No sub-paragraphs indicated by hints - using simplified parsing")
            return self._parse_simple_content_structure(content)
        
        try:
            # Use hints to guide parsing strategy
            expected_numbering = structural_hints.get('numbering_style', 'number_closing_paren')
            paragraph_count = structural_hints.get('paragraph_count', 0)
            
            self.logger.debug(f"Using hints: expecting {paragraph_count} paragraphs "
                            f"with {expected_numbering} numbering style")
            
            # Parse with guided expectations
            lines = content.split('\n')
            current_paragraph = None
            current_sub_paragraph = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for paragraph markers guided by expected count
                paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
                if paragraph_match:
                    para_num = paragraph_match.group(1)
                    para_text_start = paragraph_match.group(2)
                    
                    # Validate against expected count
                    if int(para_num) <= paragraph_count:
                        content_map['paragraphs'][para_num] = {
                            'start_line': line_idx,
                            'start_text': para_text_start,
                            'full_text': line,
                            'sub_paragraphs': {}
                        }
                        current_paragraph = para_num
                        current_sub_paragraph = None
                        
                        self.logger.debug(f"Found expected paragraph {para_num} at line {line_idx}")
                        continue
                
                # Look for sub-paragraph markers using numbering style hints
                if current_paragraph and expected_numbering == 'number_closing_paren':
                    # Look for "1)", "2)" pattern
                    sub_para_match = re.match(r'^(\d+)\)\s+(.+)', line)
                    if sub_para_match:
                        sub_para_num = sub_para_match.group(1)
                        sub_para_text = sub_para_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_num] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line
                        }
                        current_sub_paragraph = sub_para_num
                        
                        self.logger.debug(f"Found sub-paragraph {current_paragraph}({sub_para_num}) at line {line_idx}")
                        continue
                
                # Handle continuation text
                if current_paragraph:
                    if current_sub_paragraph:
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][current_sub_paragraph]['text'] += ' ' + line
                    else:
                        content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
            
            content_map['parsing_successful'] = len(content_map['paragraphs']) > 0
            self.logger.debug(f"Guided parsing completed: found {len(content_map['paragraphs'])} paragraphs")
            
        except Exception as e:
            self.logger.warning(f"Error in guided content parsing: {e}")
            # Fall back to basic parsing
            return self._parse_simple_content_structure(content)
        
        return content_map
    
    def _parse_simple_content_structure(self, content: str) -> Dict[str, Any]:
        """
        Parse content structure without hints - fallback for documents without enhanced metadata.
        
        This method provides reliable citation creation even for documents that don't
        have enhanced structural metadata, ensuring the system works gracefully
        across all document types in your collection.
        """
        self.logger.debug("Using simple content parsing (no structural hints available)")
        
        content_map = {
            'paragraphs': {},
            'parsing_successful': False,
            'used_hints': False
        }
        
        try:
            # Basic paragraph detection without guidance
            lines = content.split('\n')
            current_paragraph = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for basic paragraph patterns
                paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
                if paragraph_match:
                    para_num = paragraph_match.group(1)
                    para_text = paragraph_match.group(2)
                    
                    content_map['paragraphs'][para_num] = {
                        'start_line': line_idx,
                        'full_text': line,
                        'sub_paragraphs': {}
                    }
                    current_paragraph = para_num
                    continue
                
                # Simple sub-paragraph detection
                if current_paragraph:
                    sub_para_match = re.match(r'^(\d+)\)\s+(.+)', line)
                    if sub_para_match:
                        sub_para_num = sub_para_match.group(1)
                        sub_para_text = sub_para_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_num] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line
                        }
            
            content_map['parsing_successful'] = len(content_map['paragraphs']) > 0
            self.logger.debug(f"Simple parsing completed: found {len(content_map['paragraphs'])} paragraphs")
            
        except Exception as e:
            self.logger.warning(f"Error in simple content parsing: {e}")
        
        return content_map
    
    def _locate_quote_in_structure(self, quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Determine exactly where a specific quote appears within the legal document structure.
        
        This method represents the core intelligence of the enhanced citation system,
        working with both hint-guided and simple parsing results to create the most
        precise citations possible from the available information.
        """
        self.logger.debug(f"Locating quote in document structure: '{quote[:50]}...'")
        
        if not content_map.get('parsing_successful', False):
            self.logger.warning("Content parsing was not successful, cannot locate quote precisely")
            return None
        
        # Clean and normalize quote for better matching
        clean_quote = ' '.join(quote.split()).lower()
        if len(clean_quote) < 10:
            self.logger.warning(f"Quote too short for reliable matching: '{quote}'")
            return None
        
        # Search through parsed structure with enhanced logging
        for para_num, para_data in content_map['paragraphs'].items():
            # Check main paragraph text
            para_text = ' '.join(para_data['full_text'].split()).lower()
            if clean_quote in para_text:
                # Found in paragraph - check for sub-paragraph specificity
                for sub_para_num, sub_para_data in para_data.get('sub_paragraphs', {}).items():
                    sub_para_text = ' '.join(sub_para_data['text'].split()).lower()
                    if clean_quote in sub_para_text:
                        self.logger.info(f"Quote precisely located: paragraph {para_num}, sub-paragraph {sub_para_num}")
                        return {
                            'paragraph': para_num,
                            'sub_paragraph': sub_para_num,
                            'location_type': 'sub_paragraph',
                            'confidence': 'high'
                        }
                
                # Quote found in main paragraph but not in specific sub-paragraph
                self.logger.info(f"Quote located in main paragraph {para_num}")
                return {
                    'paragraph': para_num,
                    'sub_paragraph': None,
                    'location_type': 'main_paragraph',
                    'confidence': 'medium'
                }
        
        self.logger.warning("Could not locate quote in document structure")
        return None
    
    def _create_precise_citation_reference(self, metadata: Dict[str, Any], content: str, quote: str) -> str:
        """
        Create a precise legal citation using flattened metadata and intelligent content analysis.
        
        This method represents the culmination of our enhanced approach, demonstrating how
        sophisticated citation creation can be achieved even when working within vector
        database constraints. It produces citations like "Article 1, paragraph 2(1) (Chapter 1: General provisions)".
        """
        self.logger.info("Creating precise citation using flattened metadata and content analysis")
        
        # Step 1: Extract flattened metadata with enhanced validation
        extracted_info = self._extract_flattened_metadata(metadata)
        
        # Step 2: Use structural hints to guide content parsing
        if extracted_info['has_enhanced_structure']:
            content_map = self._parse_content_structure_with_hints(content, extracted_info['quick_indicators'])
            self.logger.debug("Using hint-guided parsing for enhanced precision")
        else:
            content_map = self._parse_simple_content_structure(content)
            self.logger.debug("Using simple parsing due to limited metadata")
        
        # Step 3: Locate quote within parsed structure
        quote_location = self._locate_quote_in_structure(quote, content_map)
        
        # Step 4: Build precise citation reference
        reference_parts = []
        
        # Add article information (always available)
        article_num = extracted_info['article_number']
        if article_num:
            if quote_location:
                # Create detailed reference based on location analysis
                para_num = quote_location['paragraph']
                sub_para_num = quote_location.get('sub_paragraph')
                
                if sub_para_num:
                    # Most precise: article, paragraph, and sub-paragraph
                    reference_parts.append(f"Article {article_num}, paragraph {para_num}({sub_para_num})")
                    self.logger.info(f"Created maximum precision citation: Article {article_num}, paragraph {para_num}({sub_para_num})")
                else:
                    # Medium precision: article and paragraph
                    reference_parts.append(f"Article {article_num}, paragraph {para_num}")
                    self.logger.info(f"Created paragraph-level citation: Article {article_num}, paragraph {para_num}")
            else:
                # Basic precision: article only
                reference_parts.append(f"Article {article_num}")
                self.logger.info(f"Created article-level citation: Article {article_num}")
        
        # Add chapter information for complete context
        chapter_num = extracted_info['chapter_number']
        chapter_title = extracted_info['chapter_title']
        if chapter_num and chapter_title:
            chapter_info = f"Chapter {chapter_num}: {chapter_title}"
            reference_parts.append(f"({chapter_info})")
            self.logger.debug(f"Added chapter context: {chapter_info}")
        
        # Combine all parts into final citation
        final_citation = " ".join(reference_parts)
        
        # Log citation creation success with metadata about precision level
        precision_level = "maximum" if quote_location and quote_location.get('sub_paragraph') else \
                         "paragraph" if quote_location else "article"
        
        self.logger.info(f"Final citation created with {precision_level} precision: {final_citation}")
        
        return final_citation if reference_parts else f"Polish Data Protection Act - Article {article_num or 'Unknown'}"
    
    def _retrieve_relevant_documents(self, query: str, k: int = 8) -> Tuple[List, str, List[Dict]]:
        """
        Retrieve relevant documents with flattened metadata and prepare for enhanced analysis.
        
        This method handles document retrieval while validating that our flattened
        metadata approach is working correctly. It prepares the retrieved information
        for sophisticated citation creation while providing comprehensive logging.
        """
        self.logger.info(f"Starting enhanced document retrieval for query: '{query[:100]}...'")
        self.logger.info(f"Retrieving top {k} documents with flattened metadata validation")
        
        if not self.polish_law_db:
            self.logger.error("Polish law vector store is not available")
            return [], "ERROR: Polish law vector store is not available", []
        
        try:
            # Perform similarity search with metadata filtering
            start_time = time.time()
            docs = self.polish_law_db.similarity_search(
                query, 
                k=k,
                filter={"law": "polish_data_protection"}
            )
            retrieval_time = time.time() - start_time
            
            self.logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.3f} seconds")
            
            # Prepare enhanced context and validate flattened metadata
            context_pieces = []
            document_metadata = []
            
            # Statistics for flattened metadata validation
            enhanced_count = 0
            complexity_distribution = {}
            
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                
                # Validate flattened metadata structure
                has_enhanced = metadata.get('has_enhanced_structure', False)
                complexity = metadata.get('complexity_level', 'unknown')
                
                document_metadata.append({
                    'index': i,
                    'metadata': metadata,
                    'content': doc.page_content,
                    'has_enhanced_structure': has_enhanced,
                    'complexity_level': complexity
                })
                
                # Track statistics
                if has_enhanced:
                    enhanced_count += 1
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
                
                # Log detailed retrieval information
                article_num = metadata.get('article_number', 'N/A')
                doc_type = metadata.get('type', 'unknown')
                chapter_num = metadata.get('chapter_number', 'N/A')
                paragraph_count = metadata.get('paragraph_count', 0)
                
                self.logger.info(f"Document {i+1}: Article {article_num} ({doc_type}), "
                               f"Chapter {chapter_num}, Enhanced: {'✓' if has_enhanced else '✗'}, "
                               f"Complexity: {complexity}, Paragraphs: {paragraph_count}, "
                               f"Content: {len(doc.page_content)} chars")
                
                # Build enhanced reference for context display
                reference = f"Polish Data Protection Act - Article {article_num}"
                if metadata.get('chapter_title'):
                    reference += f" (Chapter {chapter_num}: {metadata['chapter_title']})"
                reference += f" - {doc_type}"
                if has_enhanced:
                    reference += f" [Enhanced: {complexity}, {paragraph_count}p]"
                
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
            self.logger.info("RETRIEVAL STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Enhanced metadata: {enhanced_count}/{len(docs)} documents")
            self.logger.info(f"Complexity distribution: {dict(complexity_distribution)}")
            
            # Article and type distribution
            article_counts = {}
            type_counts = {}
            for doc in docs:
                article = doc.metadata.get('article_number', 'unknown')
                doc_type = doc.metadata.get('type', 'unknown')
                article_counts[article] = article_counts.get(article, 0) + 1
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            self.logger.info(f"Article distribution: {dict(article_counts)}")
            self.logger.info(f"Document type distribution: {dict(type_counts)}")
            
            enhancement_rate = (enhanced_count / len(docs) * 100) if docs else 0
            self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
            
            return docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.logger.error(f"Error during enhanced document retrieval: {e}")
            return [], f"ERROR: Failed to retrieve documents - {str(e)}", []
    
    def _parse_llm_response_to_citations(self, llm_response: str, document_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse LLM response and create precisely formatted citations using flattened metadata.
        
        This method combines LLM analysis with our sophisticated flattened metadata
        approach to create the most precise citations possible. It demonstrates how
        the complete solution works end-to-end.
        """
        self.logger.info("Starting enhanced citation parsing with flattened metadata support")
        citations = []
        
        try:
            # Split response into citation blocks
            citation_blocks = llm_response.split("CITATION ")[1:]
            self.logger.info(f"Found {len(citation_blocks)} citation blocks for enhanced processing")
            
            for block_index, block in enumerate(citation_blocks):
                try:
                    # Extract citation components from LLM response
                    lines = block.strip().split('\n')
                    article_info = ""
                    quote = ""
                    explanation = ""
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith("- Article:"):
                            article_info = line.replace("- Article:", "").strip()
                        elif line.startswith("- Quote:"):
                            quote = line.replace("- Quote:", "").strip()
                            if quote.startswith('"') and quote.endswith('"'):
                                quote = quote[1:-1]
                        elif line.startswith("- Explanation:"):
                            explanation = line.replace("- Explanation:", "").strip()
                    
                    # Find best matching document using enhanced metadata
                    best_match_doc = None
                    best_match_score = 0
                    
                    if quote and document_metadata:
                        for doc_meta in document_metadata:
                            # Score based on content match and metadata quality
                            content_match = quote[:50].lower() in doc_meta['content'].lower()
                            has_enhanced = doc_meta.get('has_enhanced_structure', False)
                            
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
                                           f"Article {best_match_doc['metadata'].get('article_number', 'N/A')}, "
                                           f"Enhanced: {best_match_doc.get('has_enhanced_structure', False)}, "
                                           f"Score: {best_match_score}")
                    
                    # Create precise citation using best match
                    if best_match_doc:
                        precise_article = self._create_precise_citation_reference(
                            best_match_doc['metadata'], 
                            best_match_doc['content'], 
                            quote
                        )
                        self.logger.info(f"Created enhanced citation {block_index + 1}: {precise_article}")
                    else:
                        # Fallback citation
                        precise_article = f"Polish Data Protection Act - {article_info}" if article_info else "Polish Data Protection Act - Article Unknown"
                        self.logger.warning(f"Using fallback citation for block {block_index + 1}")
                    
                    # Validate and add citation
                    if precise_article and quote and explanation:
                        citation = {
                            "article": precise_article,
                            "quote": quote,
                            "explanation": explanation
                        }
                        citations.append(citation)
                        self.logger.info(f"Successfully created enhanced citation {len(citations)}: {precise_article}")
                    else:
                        self.logger.warning(f"Incomplete citation {block_index + 1}: missing required fields")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing enhanced citation block {block_index + 1}: {e}")
                    continue
            
            self.logger.info(f"Enhanced citation parsing completed: {len(citations)} precise citations created")
            return citations
            
        except Exception as e:
            self.logger.error(f"Error during enhanced citation parsing: {e}")
            return []
    
    def _create_fallback_citations(self, retrieved_docs: List) -> List[Dict[str, Any]]:
        """
        Create fallback citations when enhanced parsing fails, using available metadata.
        
        Even when LLM parsing encounters issues, we can still create useful citations
        by leveraging whatever flattened metadata is available, ensuring the system
        continues to function gracefully.
        """
        self.logger.warning("Creating enhanced fallback citations from retrieved documents")
        fallback_citations = []
        
        try:
            # Group documents by article and prioritize enhanced ones
            article_groups = {}
            for doc in retrieved_docs[:3]:
                article_num = doc.metadata.get('article_number', 'Unknown')
                if article_num not in article_groups:
                    article_groups[article_num] = []
                article_groups[article_num].append(doc)
            
            # Create fallback citations with enhanced metadata when available
            for article_num, docs in article_groups.items():
                # Prioritize documents with enhanced metadata
                primary_doc = None
                for doc in docs:
                    if doc.metadata.get('has_enhanced_structure', False):
                        primary_doc = doc
                        break
                
                if not primary_doc:
                    primary_doc = docs[0]
                
                # Create citation using available metadata
                if primary_doc.metadata.get('has_enhanced_structure', False):
                    try:
                        precise_reference = self._create_precise_citation_reference(
                            primary_doc.metadata, primary_doc.page_content, ""
                        )
                        self.logger.info(f"Created enhanced fallback citation: {precise_reference}")
                    except Exception as e:
                        self.logger.warning(f"Enhanced fallback failed: {e}")
                        precise_reference = self._create_basic_citation_reference(primary_doc.metadata)
                else:
                    precise_reference = self._create_basic_citation_reference(primary_doc.metadata)
                
                # Create fallback citation
                citation = {
                    "article": precise_reference,
                    "quote": primary_doc.page_content[:200] + "..." if len(primary_doc.page_content) > 200 else primary_doc.page_content,
                    "explanation": "Retrieved relevant content from this provision of Polish data protection law"
                }
                
                fallback_citations.append(citation)
            
            # Final fallback if no articles found
            if not fallback_citations:
                fallback_citations.append({
                    "article": "Polish Data Protection Act",
                    "quote": "Multiple relevant provisions found in Polish data protection law",
                    "explanation": "Retrieved content related to Polish implementation of GDPR requirements"
                })
                self.logger.info("Created generic enhanced fallback citation")
            
            return fallback_citations
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced fallback citations: {e}")
            return [{
                "article": "Polish Data Protection Act (Enhanced System Error)",
                "quote": "Could not retrieve Polish law information",
                "explanation": f"Error occurred during enhanced processing: {str(e)}"
            }]
    
    def _create_basic_citation_reference(self, metadata: Dict[str, Any]) -> str:
        """Create a basic citation reference when enhanced metadata is not available."""
        article_num = metadata.get('article_number', 'Unknown')
        chapter_num = metadata.get('chapter_number', '')
        chapter_title = metadata.get('chapter_title', '')
        
        reference = f"Article {article_num}"
        if chapter_num and chapter_title:
            reference += f" (Chapter {chapter_num}: {chapter_title})"
        
        return reference
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with enhanced flattened metadata support.
        
        This method orchestrates the complete enhanced citation creation process,
        demonstrating how sophisticated legal analysis can be achieved while working
        within vector database constraints through intelligent design.
        """
        session_start = time.time()
        user_query = state["user_query"]
        gdpr_citations = state.get("gdpr_citations", [])
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED POLISH LAW AGENT SESSION WITH FLATTENED METADATA")
        self.logger.info(f"User query: {user_query}")
        self.logger.info(f"GDPR citations received: {len(gdpr_citations)}")
        self.logger.info("Flattened metadata approach enabled for vector database compatibility")
        self.logger.info("=" * 80)
        
        print("\n📜 [STEP 2/4] ENHANCED POLISH LAW AGENT: Analyzing with flattened metadata precision...")
        
        try:
            # Step 1: Retrieve documents with flattened metadata validation
            self.logger.info("STEP 1: Enhanced Document Retrieval with Metadata Validation")
            retrieved_docs, retrieved_context, document_metadata = self._retrieve_relevant_documents(user_query)
            
            if not retrieved_docs:
                self.logger.error("No documents retrieved - cannot proceed with analysis")
                state["polish_law_citations"] = self._create_fallback_citations([])
                return state
            
            # Step 2: Analyze content with LLM
            self.logger.info("STEP 2: Enhanced Content Analysis")
            rag_response = self._analyze_retrieved_content(user_query, retrieved_context)
            
            # Step 3: Parse response into precise citations using flattened metadata
            self.logger.info("STEP 3: Enhanced Citation Creation with Flattened Metadata")
            polish_law_citations = self._parse_llm_response_to_citations(rag_response, document_metadata)
            
            # Step 4: Handle parsing failures with enhanced fallback
            if not polish_law_citations:
                self.logger.warning("Enhanced citation parsing failed - using enhanced fallback")
                polish_law_citations = self._create_fallback_citations(retrieved_docs)
            
            # Update state with enhanced results
            state["polish_law_citations"] = polish_law_citations
            
            # Comprehensive session completion logging
            session_time = time.time() - session_start
            enhanced_count = sum(1 for doc_meta in document_metadata if doc_meta.get('has_enhanced_structure', False))
            
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED POLISH LAW AGENT SESSION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total session time: {session_time:.3f} seconds")
            self.logger.info(f"Documents retrieved: {len(retrieved_docs)}")
            self.logger.info(f"Enhanced documents processed: {enhanced_count}")
            self.logger.info(f"Precise citations created: {len(polish_law_citations)}")
            self.logger.info("Flattened metadata approach successful!")
            self.logger.info("=" * 80)
            
            print(f"✅ Completed: {len(polish_law_citations)} precise citations created")
            print(f"📊 Enhanced metadata: {enhanced_count}/{len(retrieved_docs)} documents")
            
        except Exception as e:
            session_time = time.time() - session_start
            self.logger.error("=" * 80)
            self.logger.error("ENHANCED POLISH LAW AGENT SESSION FAILED")
            self.logger.error(f"Error after {session_time:.3f} seconds: {e}")
            self.logger.error("=" * 80)
            
            # Ensure workflow continues with error information
            state["polish_law_citations"] = [{
                "article": "Polish Data Protection Act (Enhanced System Error)",
                "quote": "Could not retrieve Polish law information",
                "explanation": f"Error occurred during enhanced processing: {str(e)}"
            }]
            
            print(f"❌ Error in enhanced Polish law analysis: {e}")
        
        return state
    
    def _analyze_retrieved_content(self, user_query: str, retrieved_context: str) -> str:
        """Analyze retrieved content using LLM with enhanced guidance."""
        self.logger.info("Starting enhanced LLM analysis of retrieved content")
        
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