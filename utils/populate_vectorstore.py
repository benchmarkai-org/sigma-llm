import yaml
from pathlib import Path
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import os
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
import time
from functools import wraps
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

class VectorStorePopulator:
    def __init__(self):
        logger.info("Initializing VectorStorePopulator")
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "sigma-rules")  # Default if not set
        
        if not self.pinecone_api_key:
            raise ValueError("Missing required Pinecone API key")
            
        # Create Pinecone instance and get index
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        index = self.pc.Index(self.index_name)
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Simpler splitter that keeps rules intact but splits very large rules if needed
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Larger chunk size to keep rules intact
            chunk_overlap=0,   # No overlap needed since we want to keep rules as units
            length_function=len,
            separators=["\n---\n"]  # Only split on YAML document separators if needed
        )
        
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",
            namespace="sigma-rules"
        )
        
        self.cutoff_date = self._parse_cutoff_date()

    def _parse_cutoff_date(self) -> datetime:
        """Parse the cutoff date from environment variable"""
        date_str = os.getenv('SIGMA_RULES_CUTOFF_DATE')
        if not date_str:
            logger.warning("No SIGMA_RULES_CUTOFF_DATE set, using all rules")
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format in SIGMA_RULES_CUTOFF_DATE: {e}")
            raise

    @log_execution_time
    def load_sigma_rules(self, rules_directory: str) -> List[Dict]:
        """Load Sigma rules from a directory with date filtering"""
        logger.info(f"Loading Sigma rules from directory: {rules_directory}")
        rules = []
        
        yaml_files = glob.glob(f"{rules_directory}/**/*.yml", recursive=True)
        yaml_files.extend(glob.glob(f"{rules_directory}/**/*.yaml", recursive=True))
        
        total_files = len(yaml_files)
        logger.info(f"Found {total_files} YAML files")
        
        for i, file_path in enumerate(yaml_files, 1):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    rule = yaml.safe_load(file)
                    if rule and isinstance(rule, dict):
                        # Apply date filter if cutoff date is set
                        if self.cutoff_date and 'date' in rule:
                            try:
                                rule_date = datetime.strptime(str(rule['date']), '%Y-%m-%d')
                                if rule_date > self.cutoff_date:
                                    logger.debug(f"Skipping rule {rule.get('title', 'unknown')} - after cutoff date")
                                    continue
                            except ValueError:
                                logger.warning(f"Invalid date format in rule {file_path}")
                                continue
                        
                        rule['source_file'] = file_path
                        rules.append(rule)
                        logger.debug(f"Successfully loaded rule from {file_path}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                
        logger.info(f"Successfully loaded {len(rules)} valid Sigma rules before {self.cutoff_date}")
        return rules

    @log_execution_time
    def prepare_documents(self, rules: List[Dict]) -> List[str]:
        """Convert rules to text documents for embedding"""
        logger.info("Preparing documents for embedding")
        documents = []
        
        # Maximum size for metadata (leaving some buffer)
        MAX_METADATA_SIZE = 35000  # 35KB to be safe
        truncated_count = 0
        
        for rule in rules:
            try:
                # Extract key metadata
                title = rule.get('title', 'N/A')
                description = rule.get('description', 'N/A')
                author = rule.get('author', 'N/A')
                tags = ', '.join(rule.get('tags', []))
                
                # Extract detection-related fields which are crucial for semantic search
                detection = rule.get('detection', {})
                condition = detection.get('condition', 'N/A')
                selection = str(detection.get('selection', 'N/A'))
                
                # Construct header with strategic ordering
                header = f"""
                TITLE: {title}
                DESCRIPTION: {description}
                DETECTION CONDITION: {condition}
                DETECTION SELECTION: {selection}
                TAGS: {tags}
                AUTHOR: {author}
                """
                
                # Check header size first
                header_size = len(header.encode('utf-8'))
                if header_size > MAX_METADATA_SIZE:
                    logger.warning(f"Header alone for rule '{title}' exceeds size limit!")
                    # Truncate the longest fields
                    description = description[:1000] + "... (truncated)"
                    tags = tags[:500] + "... (truncated)"
                    selection = str(selection)[:500] + "... (truncated)"
                    
                    # Reconstruct header with truncated fields
                    header = f"""
                    TITLE: {title}
                    DESCRIPTION: {description}
                    DETECTION CONDITION: {condition}
                    DETECTION SELECTION: {selection}
                    TAGS: {tags}
                    AUTHOR: {author}
                    """
                
                # Convert remaining rule content
                rule_text = yaml.dump(rule, sort_keys=False, allow_unicode=True)
                
                # Calculate available space for rule content
                header_size = len(header.encode('utf-8'))
                available_size = MAX_METADATA_SIZE - header_size - 100  # Buffer for formatting
                
                # Truncate rule_text if needed
                if len(rule_text.encode('utf-8')) > available_size:
                    logger.warning(f"Rule '{title}' content requires truncation")
                    rule_text = rule_text.encode('utf-8')[:available_size].decode('utf-8', errors='ignore')
                    truncated_count += 1
                
                # Combine with clear separator
                full_text = f"{header.strip()}\n\nRULE:\n{rule_text}"
                
                # Final size check
                final_size = len(full_text.encode('utf-8'))
                if final_size > MAX_METADATA_SIZE:
                    logger.error(f"Rule '{title}' still exceeds limit after truncation!")
                    logger.error(f"Size: {final_size} bytes, Limit: {MAX_METADATA_SIZE}")
                    continue  # Skip this rule rather than cause an error
                
                documents.append(full_text)
                logger.debug(f"Successfully processed rule '{title}' ({final_size} bytes)")
                
            except Exception as e:
                logger.error(f"Error preparing document for rule '{title}': {str(e)}")
                
        logger.info(f"Processed {len(documents)} rules")
        if truncated_count > 0:
            logger.warning(f"Had to truncate {truncated_count} rules. This may impact RAG quality.")
            
        return documents

    @log_execution_time
    def populate_vectorstore(self, rules_directory: str):
        """Main method to populate the vector store"""
        logger.info(f"Starting vector store population from {rules_directory}")
        try:
            # Load rules
            rules = self.load_sigma_rules(rules_directory)
            if not rules:
                raise ValueError("No valid rules loaded")
                
            # Prepare documents
            documents = self.prepare_documents(rules)
            
            # Split documents
            logger.info("Splitting documents into chunks")
            texts = self.text_splitter.create_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")
            
            # Create and persist vector store
            logger.info("Creating vector embeddings and storing in Pinecone")
            self.vectorstore.add_documents(texts)
            
            # Get index statistics using new SDK pattern
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            logger.info(f"Vector store population completed. Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error populating vector store: {str(e)}", exc_info=True)
            raise

    @log_execution_time
    def test_vectorstore(self, query: str = "Detect PowerShell execution with encoded commands"):
        """Test the vector store with a sample query"""
        logger.info(f"Testing vector store with query: {query}")
        try:
            results = self.vectorstore.similarity_search(query, k=2)
            logger.info(f"Successfully retrieved {len(results)} results")
            
            for i, doc in enumerate(results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info("-" * 80)
                
        except Exception as e:
            logger.error(f"Error testing vector store: {str(e)}")
            raise

def main():
    load_dotenv()
    
    # Configuration
    RULES_DIR = os.getenv('SIGMA_RULES_DIR', './sigma_all_rules')
    logger.info("Starting vector store population script")
    
    try:
        # Ensure rules directory exists
        if not os.path.exists(RULES_DIR):
            raise ValueError(f"Rules directory not found: {RULES_DIR}")
        
        populator = VectorStorePopulator()
        
        # Populate vector store
        populator.populate_vectorstore(RULES_DIR)
        
        # Test the population
        populator.test_vectorstore()
        
        logger.info("Vector store population and testing completed successfully")
        
    except Exception as e:
        logger.error(f"Vector store population failed: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()