import requests
import json
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re
import yaml
from datetime import datetime
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_webpage_content(url: str) -> str:
    """
    Fetches and parses content from a webpage.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
            element.decompose()

        # Find main content
        main_content = soup.find(['main', 'article', 'div.content', 'div.main-content'])
        text = main_content.get_text(separator=' ', strip=True) if main_content else soup.body.get_text(separator=' ', strip=True)

        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error parsing webpage {url}: {e}")
        return None

def get_summary_from_service(content: str, service_url: str, service_api_key: str, assistant_id: str) -> str:
    """
    Get content summary from the microservice
    """
    try:
        response = requests.post(
            f"{service_url}/api/v1/summarize-references",
            json={
                "reference_content": content,
                "assistant_id": assistant_id
            },
            headers={
                "Authorization": f"Bearer {service_api_key}",
                "Content-Type": "application/json"
            },
            verify=False  # For development only
        )
        
        response.raise_for_status()
        return response.json()["summary"]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling summarization service: {e}")
        raise

def process_references(rules_directory: str, service_url: str, service_api_key: str, assistant_id: str) -> List[Dict[str, Any]]:
    """
    Process Sigma rules and their references
    """
    results = []
    
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
                
                # Get references from the rule
                references = rule_data.get('references', [])
                if not references:
                    continue
                
                # Get rule title for logging
                rule_title = rule_data.get('title', file)
                
                for ref in references:
                    if not isinstance(ref, str) or not (ref.startswith('http://') or ref.startswith('https://')):
                        continue
                        
                    try:
                        # Parse webpage
                        ref_content = parse_webpage_content(ref)
                        if not ref_content:
                            continue
                            
                        # Get summary from service
                        summary = get_summary_from_service(
                            ref_content,
                            service_url,
                            service_api_key,
                            assistant_id
                        )
                        
                        results.append({
                            "rule": rule_content,
                            "reference_url": ref,
                            "reference_content": ref_content,
                            "summarized_content": summary
                        })
                        
                        logger.info(f"✅ Processed reference for rule: {rule_title}")
                        
                    except Exception as e:
                        logger.error(f"Error processing reference {ref}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                
    return results

def save_results(results: List[Dict[str, Any]], output_file: str = None):
    """Save processed references to a JSON file with timestamp"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reference_results_{timestamp}.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    service_url = os.getenv('RULE_GENERATOR_URL', 'http://localhost:5000')
    service_api_key = os.getenv('SERVICE_API_KEY')
    assistant_id = os.getenv('REFERENCE_SUMMARIZER_ASSISTANT_ID')
    rules_directory = "./sigma_core/rules/linux/"
    
    # Validate configuration
    if not all([service_api_key, assistant_id, rules_directory]):
        raise ValueError("Missing required environment variables")
    
    # Process rules
    logger.info(f"Processing rules from directory: {rules_directory}")
    results = process_references(rules_directory, service_url, service_api_key, assistant_id)
    
    # Save results
    output_dir = Path("evaluation/results")
    save_results(results)  # Remove the output_file parameter to use timestamped name
    
    logger.info(f"Processed {len(results)} references")

if __name__ == "__main__":
    main()