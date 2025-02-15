import os
import yaml
import json
import requests
import urllib3
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Suppress SSL warnings
urllib3.disable_warnings()

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_yaml_rule(yaml_content: str) -> Dict:
    """Parse YAML content string to dictionary"""
    try:
        return yaml.safe_load(yaml_content)
    except Exception as e:
        logger.error(f"Error parsing YAML content: {e}")
        raise

def assess_rule(rule: str, config: Dict) -> Dict:
    """Send a rule to the summarize-detection endpoint and get the results."""
    headers = {
        'Authorization': f"Bearer {config['SERVICE_API_KEY']}",
        'Content-Type': 'application/json'
    }
    
    data = {
        'rule': rule,
        'model': config.get('MODEL_NAME', 'claude-3-5-sonnet-latest')  # Add model to request
    }
    
    try:
        response = requests.post(
            f"{config['RULE_GENERATOR_URL']}/api/v1/summarize-detection",
            headers=headers,
            json=data,
            verify=False,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting detection summary: {e}")
        return None

def process_rules_to_query_pairs(rules_dir: str, config: Dict) -> List[Dict]:
    """
    Process Sigma rules and convert them to query-rule pairs where:
    - query is the rule's detection logic summary from LLM
    - expected_rule is the complete rule in YAML format
    """
    rules_path = Path(rules_dir)
    query_rule_pairs = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Get cutoff date from environment
    date_str = os.getenv('SIGMA_RULES_CUTOFF_DATE')
    if not date_str:
        logger.warning("No SIGMA_RULES_CUTOFF_DATE set")
        return query_rule_pairs
        
    try:
        cutoff_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format in SIGMA_RULES_CUTOFF_DATE: {e}")
        raise
    
    logger.info(f"Processing rules modified or created after {cutoff_date}")
    
    # Walk through all yaml files in directory and subdirectories
    for yaml_file in rules_path.rglob("*.yml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            
            # Basic validation of YAML content
            if not yaml_content.strip():
                logger.warning(f"Empty YAML file: {yaml_file}")
                skipped_count += 1
                continue
                
            rule_dict = parse_yaml_rule(yaml_content)
            if not rule_dict:
                logger.warning(f"Invalid YAML content in {yaml_file}")
                skipped_count += 1
                continue
                
            # Validate required Sigma rule fields
            if not all(field in rule_dict for field in ['title', 'detection']):
                logger.warning(f"Missing required fields in {yaml_file}")
                skipped_count += 1
                continue
            
            # Get the most recent date (modified or creation date)
            rule_date = rule_dict.get('modified') or rule_dict.get('date')
            if rule_date:
                try:
                    # Handle different date formats
                    if isinstance(rule_date, datetime):
                        rule_date = rule_date.strftime('%Y-%m-%d')
                    elif isinstance(rule_date, str):
                        # Try to parse various date formats
                        try:
                            # Try ISO format first
                            rule_date = datetime.fromisoformat(rule_date).strftime('%Y-%m-%d')
                        except ValueError:
                            # Fallback to basic YYYY-MM-DD
                            rule_date = datetime.strptime(rule_date, '%Y-%m-%d').strftime('%Y-%m-%d')
                    else:
                        logger.warning(f"Unexpected date format in {yaml_file}: {type(rule_date)}")
                        skipped_count += 1
                        continue
                        
                    rule_date = datetime.strptime(rule_date, '%Y-%m-%d')
                    date_type = "modified" if rule_dict.get('modified') else "creation"
                    logger.debug(f"Processing rule with {date_type} date {rule_date}: {rule_dict.get('title')}")
                    
                    if rule_date <= cutoff_date:
                        logger.debug(f"⏭️ Skipping rule (before cutoff {cutoff_date}): {rule_dict.get('title')} with {date_type} date {rule_date}")
                        skipped_count += 1
                        continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse date {rule_date} for rule {rule_dict.get('title')}: {e}")
                    skipped_count += 1
                    continue
            else:
                logger.debug(f"No date found for rule: {rule_dict.get('title')}")
                skipped_count += 1
                continue
            
            # Get rule detection summary from LLM
            summary_result = assess_rule(yaml_content, config)
            if not summary_result:
                logger.warning(f"Skipping rule due to summarization failure: {rule_dict.get('title')}")
                error_count += 1
                continue
                
            # Validate summary result
            if not summary_result.get('summary'):
                logger.warning(f"Empty summary received for rule: {rule_dict.get('title')}")
                error_count += 1
                continue
                
            pair = {
                "query": summary_result['summary'],
                "expected_rule": yaml_content
            }
            
            query_rule_pairs.append(pair)
            processed_count += 1
            logger.info(f"✅ Processed rule: {rule_dict.get('title')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing rule {yaml_file}: {e}")
            error_count += 1
    
    # Log final statistics
    logger.info(f"\nProcessing Summary:")
    logger.info(f"- Successfully processed: {processed_count} rules")
    logger.info(f"- Skipped: {skipped_count} rules")
    logger.info(f"- Errors: {error_count} rules")
    logger.info(f"- Total rules found: {processed_count + skipped_count + error_count}")
    
    return query_rule_pairs

def save_query_rule_pairs(pairs: List[Dict], output_file: str = "query_rule_pairs_long.json"):
    """Save query-rule pairs to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=4)
        logger.info(f"\nResults saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving query-rule pairs: {e}")
        raise

def main():
    """
    Process Sigma rules and create query-rule pairs JSON file using LLM assessment
    """
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Configuration
        config = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "sigma-rules"),
            "SERVICE_API_KEY": os.getenv("SERVICE_API_KEY"),
            "RULE_GENERATOR_URL": os.getenv("RULE_GENERATOR_URL", "https://my-microservice-680275457059.us-central1.run.app"),
            "MODEL_NAME": os.getenv("MODEL_NAME", "claude-3-5-sonnet-latest")  # Add model configuration
        }
        
        logger.debug(f"Environment variables present: {list(os.environ.keys())}")
        logger.info(f"Using LLM model: {config['MODEL_NAME']}")
        
        # Validate required config
        required_config = ['RULE_GENERATOR_URL', 'SERVICE_API_KEY']
        if not all(key in config for key in required_config):
            raise ValueError(f"Missing required configuration. Need: {required_config}")
        
        # Set default rules directory (can be overridden via command line)
        rules_dir = "../sigma_all_rules/rules/"
        
        # Check if directory was provided as command line argument
        import sys
        if len(sys.argv) > 1:
            rules_dir = sys.argv[1]
        
        # Validate that the rules directory exists
        if not os.path.exists(rules_dir):
            logger.error(f"Error: Rules directory '{rules_dir}' not found")
            sys.exit(1)
        
        # Process all rules and save results to JSON
        pairs = process_rules_to_query_pairs(rules_dir, config)
        save_query_rule_pairs(pairs)
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()