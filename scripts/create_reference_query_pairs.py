import json
import os
from typing import List, Dict, Any, Optional
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

def process_rules(rules_directory: str) -> List[Dict[str, Any]]:
    """
    Process Sigma rules and their references to create training data.
    Returns a list of dictionaries with reference content as query and Sigma rule as expected output.
    
    Args:
        rules_directory: Directory containing Sigma rules
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
    
    logger.info(f"Processing rules modified or created after {cutoff_date}")
    
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
                
                # Skip if no description or references
                if not rule_data.get('description') or not rule_data.get('references'):
                    continue
                    
                # Get the date (modified or creation date)
                rule_date = rule_data.get('modified') or rule_data.get('date')
                if rule_date:
                    if isinstance(rule_date, datetime):
                        rule_date = rule_date.strftime('%Y-%m-%d')
                    try:
                        rule_date = datetime.strptime(str(rule_date), '%Y-%m-%d')
                        date_type = "modified" if rule_data.get('modified') else "creation"
                        logger.debug(f"Processing rule with {date_type} date {rule_date}: {rule_data.get('title')}")
                        if rule_date <= cutoff_date:
                            logger.debug(f"⏭️ Skipping rule (before cutoff {cutoff_date}): {rule_data.get('title')} with {date_type} date {rule_date}")
                            continue
                    except ValueError as e:
                        logger.warning(f"Could not parse date {rule_date} for rule {rule_data.get('title')}: {e}")
                        continue
                else:
                    logger.debug(f"No date found for rule: {rule_data.get('title')}")
                    continue
                
                # Get references and their content
                references = rule_data.get('references', [])
                reference_contents = []
                
                for ref in references:
                    if not isinstance(ref, str) or not (ref.startswith('http://') or ref.startswith('https://')):
                        continue
                        
                    ref_content = parse_webpage_content(ref)
                    if ref_content:
                        reference_contents.append(ref_content)
                
                # Create example if we have any valid references
                if reference_contents:
                    # Combine all reference contents
                    combined_refs = "\n\n".join(reference_contents)
                    
                    # Create the example
                    example = {
                        "query": combined_refs,
                        "expected_rule": rule_content
                    }
                    
                    results.append(example)
                    logger.info(f"✅ Processed rule: {rule_data.get('title')}")
                else:
                    logger.debug(f"⚠️ No valid reference content for rule: {rule_data.get('title')}")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                
    logger.info(f"\nProcessed {len(results)} rules that were modified or created after {cutoff_date}")
    return results

def save_results(results: List[Dict[str, Any]], output_file: str = "query_rule_pairs_reference.json"):
    """Save processed rules to a JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total examples saved: {len(results)}")

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
    
    # Process rules
    logger.info(f"Processing rules from directory: {rules_directory}")
    results = process_rules(rules_directory)
    
    # Save results
    save_results(results)
    
    logger.info(f"Processed {len(results)} rules")

if __name__ == "__main__":
    main() 