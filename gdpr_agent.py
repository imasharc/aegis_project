import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class GDPRAgent:
    def __init__(self, vector_db_path: str = None):
        # Initialize the OpenAI model
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize the vector store for RAG
        if vector_db_path is None:
            vector_db_path = os.path.join(os.path.dirname(__file__), "data", "gdpr_db")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings,
            collection_name="gdpr_regulation"
        )
        
        # Prompt template optimized for the exact GDPR JSON structure
        self.prompt = ChatPromptTemplate.from_template(
            """You are a specialized GDPR legal expert with access to the full GDPR text and related documents.
            
            User Query: {user_query}
            
            Based on the following retrieved GDPR content, identify the most relevant provisions:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. The specific article number and title
            2. The chapter number it belongs to (in Roman numerals)
            3. A direct quote of the relevant text from the retrieved context
            4. A brief explanation of its relevance to the query
            
            Important: Only cite information that appears in the retrieved context above.
            If the context doesn't contain sufficient information, say so clearly.
            
            Format your response as a structured list of citations in this exact format:
            
            CITATION 1:
            - Article: [Number and title]
            - Chapter: [Chapter number in Roman numerals]
            - Quote: "[Direct quote from retrieved context]"
            - Relevance: [Brief explanation]
            
            CITATION 2:
            - Article: [Number and title]
            - Chapter: [Chapter number in Roman numerals]
            - Quote: "[Direct quote from retrieved context]"
            - Relevance: [Brief explanation]
            """
        )
    
    def _retrieve_relevant_documents(self, query: str, k: int = 6) -> tuple:
        """Retrieve relevant documents and format them with precise metadata handling."""
        # Search for relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Create formatted context with precise metadata handling
        context_pieces = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            doc_type = metadata.get('type', 'content')
            
            # Create a metadata display based on the exact JSON structure
            metadata_display = ""
            if doc_type == "chapter":
                chapter_num = metadata.get('chapter_number', 'Unknown')
                metadata_display = f"CHAPTER {chapter_num}"
            elif doc_type == "article_full":
                article_num = metadata.get('article_number', 'Unknown')
                article_title = metadata.get('article_title', '')
                chapter = metadata.get('chapter', 'Unknown')
                metadata_display = f"Article {article_num}: {article_title} (Chapter {chapter})"
            elif doc_type == "article_paragraph":
                article_num = metadata.get('article_number', 'Unknown')
                article_title = metadata.get('article_title', '')
                paragraph_num = metadata.get('paragraph_number', 'Unknown')
                chapter = metadata.get('chapter', 'Unknown')
                metadata_display = f"Article {article_num}: {article_title}, Paragraph {paragraph_num} (Chapter {chapter})"
            
            # Format the context piece with structured metadata
            context_piece = f"[Document {i+1} - {doc_type}]\n{metadata_display}\n{doc.page_content}"
            context_pieces.append(context_piece)
        
        retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
        
        return docs, retrieved_context
    
    def _parse_llm_response_to_citations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into structured citations matching the JSON format."""
        citations = []
        
        # Split by "CITATION" to find individual citations
        citation_blocks = llm_response.split("CITATION ")[1:]  # Skip the first empty split
        
        for block in citation_blocks:
            try:
                lines = block.strip().split('\n')
                citation = {
                    "source": "GDPR",
                    "article": "",
                    "chapter": "",
                    "text": "",
                    "quote": "",
                    "explanation": ""
                }
                
                # Parse each line to extract structured information
                for line in lines:
                    line = line.strip()
                    if line.startswith("- Article:"):
                        citation["article"] = line.replace("- Article:", "").strip()
                        citation["text"] = citation["article"]  # For compatibility
                    elif line.startswith("- Chapter:"):
                        citation["chapter"] = line.replace("- Chapter:", "").strip()
                    elif line.startswith("- Quote:"):
                        quote = line.replace("- Quote:", "").strip()
                        # Remove surrounding quotes if present
                        if quote.startswith('"') and quote.endswith('"'):
                            quote = quote[1:-1]
                        citation["quote"] = quote
                    elif line.startswith("- Relevance:"):
                        citation["explanation"] = line.replace("- Relevance:", "").strip()
                
                # Only add citation if we have the essential information
                if citation["article"] and citation["quote"] and citation["explanation"]:
                    citations.append(citation)
                    
            except Exception as e:
                print(f"Warning: Could not parse citation block: {e}")
                continue
        
        return citations
    
    def format_citations_for_display(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations in a human-readable way for display."""
        if not citations:
            return "No relevant GDPR provisions found."
            
        formatted_output = "## Relevant GDPR Provisions\n\n"
        
        for i, citation in enumerate(citations, 1):
            article = citation.get("article", "Unknown Article")
            chapter = citation.get("chapter", "")
            quote = citation.get("quote", "")
            explanation = citation.get("explanation", "")
            
            formatted_output += f"### {i}. {article}\n"
            if chapter:
                formatted_output += f"*Chapter {chapter}*\n\n"
            if quote:
                formatted_output += f"> {quote}\n\n"
            if explanation:
                formatted_output += f"{explanation}\n\n"
            formatted_output += "---\n\n"
            
        return formatted_output
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query using RAG to extract GDPR citations."""
        print("\n🔍 [STEP 1/4] GDPR AGENT: Analyzing query for relevant GDPR provisions using RAG...")
        user_query = state["user_query"]
        
        try:
            # Step 1: Retrieve relevant documents with precise metadata handling
            retrieved_docs, retrieved_context = self._retrieve_relevant_documents(user_query)
            
            print(f"   Retrieved {len(retrieved_docs)} relevant documents from vector store")
            
            # Step 2: Use LLM to analyze retrieved content and extract citations
            chain = self.prompt | self.model
            response = chain.invoke({
                "user_query": user_query,
                "retrieved_context": retrieved_context
            })
            
            # Step 3: Parse the LLM response into structured citations
            citations = self._parse_llm_response_to_citations(response.content)
            
            # Fallback: if parsing fails, create a basic citation structure
            if not citations:
                print("   Warning: Could not parse structured citations, creating fallback")
                citations = [{
                    "source": "GDPR",
                    "article": "Retrieved from RAG system",
                    "chapter": "",
                    "text": "Multiple relevant provisions found",
                    "quote": response.content[:200] + "...",
                    "explanation": "RAG system found relevant GDPR content for your query"
                }]
            
            # Add formatted display text
            state["formatted_gdpr_citations"] = self.format_citations_for_display(citations)
            
            # Update the state with GDPR citations
            state["gdpr_citations"] = citations
            print(f"✅ Completed: Found {len(citations)} GDPR citations using RAG")
            
            # Optional: Store retrieval details for debugging
            state["_debug_info"] = {
                "retrieved_docs": len(retrieved_docs),
                "llm_response": response.content
            }
            
        except Exception as e:
            print(f"❌ Error in GDPR RAG processing: {e}")
            # Fallback to ensure the workflow continues
            state["gdpr_citations"] = [{
                "source": "GDPR",
                "article": "System Error",
                "chapter": "",
                "text": "Could not retrieve information",
                "quote": f"Error occurred during RAG retrieval: {str(e)}",
                "explanation": "Please check system configuration and try again"
            }]
            state["formatted_gdpr_citations"] = "Error retrieving GDPR information."
        
        return state