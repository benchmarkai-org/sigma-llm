import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
import yaml
import requests
from bs4 import BeautifulSoup
import re
import urllib3
from dotenv import load_dotenv
import io
import PyPDF2
import tiktoken

# Suppress SSL warnings
urllib3.disable_warnings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_pdf_content(pdf_data: bytes) -> str:
    """
    Parse content from a PDF file.
    """
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_data)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
        
        # Join all pages and clean the text
        text = ' '.join(text_content)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF content: {e}")
        return None

def parse_html_content(html_content: str) -> str:
    """
    Parse content from HTML.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        if not soup.body:
            logger.warning("No HTML body found")
            return None

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
            element.decompose()

        # Find main content
        main_content = soup.find(['main', 'article', 'div.content', 'div.main-content'])
        text = main_content.get_text(separator=' ', strip=True) if main_content else soup.body.get_text(separator=' ', strip=True)

        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        if not text.strip():
            logger.warning("No text content found in HTML")
            return None
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}")
        return None

def parse_webpage_content(url: str) -> str:
    """
    Fetches and parses content from a webpage, handling both HTML and PDF content.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, timeout=10, verify=False, headers=headers)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF content
        if 'application/pdf' in content_type:
            logger.info(f"Processing PDF content from {url}")
            return parse_pdf_content(response.content)
            
        # Handle HTML content
        elif 'text/html' in content_type:
            logger.info(f"Processing HTML content from {url}")
            return parse_html_content(response.text)
            
        else:
            logger.warning(f"Unsupported content type ({content_type}) from {url}")
            return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning(f"Access forbidden (403) for {url} - website may be blocking automated access")
        else:
            logger.error(f"HTTP error accessing {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching/parsing content from {url}: {e}")
        return None

def _assess_reference_quality(reference_content: str) -> float:
    """
    Assess the quality of reference content.
    Returns a score between 0 and 1.
    """
    if not reference_content:
        return 0.0
        
    # Check for minimum content length (at least 100 characters)
    if len(reference_content) < 100:
        return 0.0
        
    # Check for technical depth indicators
    technical_indicators = [
        'command', 'process', 'registry', 'file', 'network',
        'payload', 'exploit', 'attack', 'technique', 'CVE',
        'vulnerability', 'malware', 'threat'
    ]
    
    technical_score = sum(1 for ind in technical_indicators if ind.lower() in reference_content.lower()) / len(technical_indicators)
    
    # Check for structured content indicators
    structure_indicators = [
        'steps', 'procedure', 'method', 'example', 'implementation',
        'detection', 'mitigation', 'analysis'
    ]
    
    structure_score = sum(1 for ind in structure_indicators if ind.lower() in reference_content.lower()) / len(structure_indicators)
    
    # Weighted average of scores
    return (technical_score * 0.6) + (structure_score * 0.4)

def _assess_rule_quality(rule_content: str, rule_data: Dict) -> float:
    """
    Assess the quality of a Sigma rule.
    Returns a score between 0 and 1.
    """
    if not rule_content or not rule_data:
        return 0.0
        
    score = 0.0
    
    # Check for essential fields
    required_fields = ['title', 'description', 'status', 'level', 'logsource', 'detection']
    has_required = all(field in rule_data for field in required_fields)
    if not has_required:
        return 0.0
    
    # Score based on detection logic complexity
    detection = rule_data.get('detection', {})
    if detection:
        # Check for condition complexity (prefer rules with meaningful logic)
        condition = detection.get('condition', '')
        if isinstance(condition, str):
            if 'and' in condition.lower() or 'or' in condition.lower():
                score += 0.3
            if '|' in condition or 'not' in condition.lower():
                score += 0.2
                
        # Check for multiple selection criteria
        selection_count = sum(1 for k in detection.keys() if k != 'condition')
        if selection_count > 1:
            score += min(0.2, selection_count * 0.05)  # Cap at 0.2
            
    # Score based on documentation quality
    description = rule_data.get('description', '')
    if len(description) > 100:  # Prefer well-documented rules
        score += 0.1
        
    # Score based on false positive handling
    if rule_data.get('falsepositives'):
        score += 0.1
        
    # Score based on additional metadata
    if rule_data.get('tags'):
        score += 0.05
    if rule_data.get('author'):
        score += 0.05
        
    return min(1.0, score)

def _count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

def _truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within max_tokens while maintaining coherent sentences."""
    if not text:
        return ""
        
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
        
    # Decode only the allowed tokens, then find the last complete sentence
    truncated = encoding.decode(tokens[:max_tokens])
    sentences = re.split(r'(?<=[.!?])\s+', truncated)
    
    if not sentences:
        return truncated
        
    # Remove the last (potentially incomplete) sentence
    return ' '.join(sentences[:-1])

def _create_training_example(references: List[str], rule_content: str, max_tokens: int = 65000) -> Optional[Dict]:
    """
    Create a training example ensuring it fits within token limits.
    Returns None if example cannot be created within limits.
    """
    system_prompt = "You are an expert cybersecurity analyst specializing in Sigma rules. Your task is to create precise detection rules that follow official Sigma syntax and best practices. Focus on maximizing detection logic effectiveness while minimizing false positives. Consider appropriate log sources, field mappings, and performance impact on SIEM systems. Create rules with clear titles, descriptions, and relevant tags. Return only the YAML rule without any additional explanation."
    
    # Calculate tokens for fixed content
    system_tokens = _count_tokens(system_prompt)
    rule_tokens = _count_tokens(rule_content)
    
    # Reserve tokens for JSON structure and message formatting
    structure_tokens = 100  # Conservative estimate for JSON structure
    
    # Calculate remaining tokens for references
    available_tokens = max_tokens - (system_tokens + rule_tokens + structure_tokens)
    
    if available_tokens <= 0:
        logger.warning("Rule content too large for context window")
        return None
        
    # Combine and truncate references to fit
    combined_refs = "\n\n".join(references)
    truncated_refs = _truncate_text(combined_refs, available_tokens)
    
    if not truncated_refs:
        logger.warning("No space for reference content after truncation")
        return None
        
    # Create the training example
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": truncated_refs
            },
            {
                "role": "assistant",
                "content": rule_content
            }
        ]
    }

def process_rules(rules_directory: str, quality_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Process Sigma rules and their references to create training data.
    Returns a list of dictionaries in OpenAI fine-tuning format with
    reference content as input and Sigma rule as output.
    
    Args:
        rules_directory: Directory containing Sigma rules
        quality_threshold: Minimum quality score (0-1) for including an example
    """
    results = []
    
    # Get cutoff date from environment
    date_str = os.getenv('SIGMA_RULES_CUTOFF_DATE')
    if not date_str:
        logger.warning("No SIGMA_RULES_CUTOFF_DATE set")
        return results
        
    try:
        cutoff_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format in SIGMA_RULES_CUTOFF_DATE: {e}")
        raise
    
    logger.info(f"Processing rules modified or created before {cutoff_date}")
    
    # Walk through all yaml files in directory
    for root, _, files in os.walk(rules_directory):
        for file in files:
            if not file.endswith(('.yml', '.yaml')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                # Read and parse YAML file
                with open(file_path, 'r', encoding='utf-8') as f:
                    rule_content = f.read()
                    rule_data = yaml.safe_load(rule_content)
                
                # Skip if no description
                if not rule_data.get('description'):
                    continue
                    
                # Get the date (modified or creation date)
                rule_date = rule_data.get('modified') or rule_data.get('date')
                if rule_date:
                    if isinstance(rule_date, datetime):
                        rule_date = rule_date.strftime('%Y-%m-%d')
                    try:
                        rule_date = datetime.strptime(str(rule_date), '%Y-%m-%d')
                        if rule_date > cutoff_date:
                            logger.debug(f"⏭️ Skipping rule (after cutoff {cutoff_date}): {rule_data.get('title')}")
                            continue
                    except ValueError as e:
                        logger.warning(f"Could not parse date {rule_date} for rule {rule_data.get('title')}: {e}")
                        continue
                else:
                    logger.debug(f"No date found for rule: {rule_data.get('title')}")
                    continue
                
                # Assess rule quality first
                rule_quality = _assess_rule_quality(rule_content, rule_data)
                if rule_quality < quality_threshold:
                    logger.debug(f"⏭️ Skipping low quality rule (score {rule_quality:.2f}): {rule_data.get('title')}")
                    continue
                
                # Get references and their content
                references = rule_data.get('references', [])
                reference_contents = []
                reference_quality_sum = 0
                
                for ref in references:
                    if not isinstance(ref, str) or not (ref.startswith('http://') or ref.startswith('https://')):
                        continue
                        
                    ref_content = parse_webpage_content(ref)
                    if ref_content:
                        ref_quality = _assess_reference_quality(ref_content)
                        if ref_quality >= quality_threshold:
                            reference_contents.append(ref_content)
                            reference_quality_sum += ref_quality
                
                # Create training example only if we have high-quality references
                if reference_contents and (reference_quality_sum / len(reference_contents)) >= quality_threshold:
                    training_example = _create_training_example(reference_contents, rule_content)
                    if training_example:
                        results.append(training_example)
                        logger.info(f"✅ Processed high-quality rule (score {rule_quality:.2f}): {rule_data.get('title')}")
                    else:
                        logger.warning(f"⚠️ Skipping oversized rule: {rule_data.get('title')}")
                else:
                    logger.info(f"⚠️ Skipping rule with low quality references: {rule_data.get('title')}")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                
    return results

def save_results(results: List[Dict[str, Any]], output_file: str = None, max_examples: int = 100):
    """
    Save processed rules to a JSONL file with timestamp.
    Optionally limit the number of examples.
    Validates token counts before saving.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"finetuning_data_{timestamp}.jsonl"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter out examples that exceed token limits
    valid_results = []
    for result in results:
        total_tokens = sum(_count_tokens(msg["content"]) for msg in result["messages"])
        if total_tokens <= 65000:  # Leave some buffer from the 65,536 limit
            valid_results.append(result)
        else:
            logger.warning(f"Skipping example with {total_tokens} tokens (exceeds limit)")
    
    # If we have more examples than max_examples, take the first max_examples
    if max_examples and len(valid_results) > max_examples:
        logger.info(f"Limiting output to {max_examples} examples (from {len(valid_results)} total)")
        valid_results = valid_results[:max_examples]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in valid_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total valid examples saved: {len(valid_results)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Set rules directory
    rules_directory = "../sigma_all_rules/rules/"
    
    # Check if directory exists
    if not os.path.exists(rules_directory):
        logger.error(f"Rules directory not found: {rules_directory}")
        logger.error("Please make sure the sigma_core repository is cloned in the correct location")
        return
    
    # Process rules with quality filtering
    logger.info(f"Processing rules from directory: {rules_directory}")
    results = process_rules(rules_directory, quality_threshold=0.7)
    
    # Save results with limit
    save_results(results, max_examples=100)
    
    logger.info(f"Processed {len(results)} high-quality rules")

if __name__ == "__main__":
    main() 