import json
import os
from typing import List, Dict
from sigma.rule import SigmaRule
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sigma_rule_obj(yaml_content: str) -> SigmaRule:
    """Convert YAML content string to SigmaRule object"""
    try:
        return SigmaRule.from_yaml(yaml_content)
    except Exception as e:
        logger.error(f"Error creating SigmaRule object: {e}")
        raise

def process_rules_to_query_pairs(rules_dir: str) -> List[Dict]:
    """
    Process Sigma rules and convert them to query-rule pairs where:
    - query is the rule's description
    - expected_rule is the complete rule in YAML format
    """
    rules_path = Path(rules_dir)
    query_rule_pairs = []
    
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
    
    logger.info(f"Processing rules after {cutoff_date}")
    
    # Walk through all yaml files in directory and subdirectories
    for yaml_file in rules_path.rglob("*.yml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            
            sigma_rule = create_sigma_rule_obj(yaml_content)
            
            if not hasattr(sigma_rule, 'description') or not sigma_rule.description:
                continue
                
            # Only include rules after the cutoff date
            if hasattr(sigma_rule, 'date'):
                rule_date = datetime.strptime(str(sigma_rule.date), '%Y-%m-%d')
                if rule_date <= cutoff_date:
                    logger.debug(f"⏭️ Skipping rule (before cutoff): {sigma_rule.title}")
                    continue
                
            pair = {
                "query": sigma_rule.description,
                "expected_rule": yaml_content
            }
            
            query_rule_pairs.append(pair)
            logger.info(f"✅ Processed rule: {sigma_rule.title}")
                
        except Exception as e:
            logger.error(f"❌ Error processing rule {yaml_file}: {e}")
    
    logger.info(f"\nProcessed {len(query_rule_pairs)} rules after {cutoff_date}")
    return query_rule_pairs

def save_query_rule_pairs(pairs: List[Dict], output_file: str = "query_rule_pairs.json"):
    """Save query-rule pairs to a JSON file"""
    # Write the pairs to a JSON file with pretty printing (indent=4)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=4)
        logger.info(f"\nResults saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving query-rule pairs: {e}")
        raise

def main():
    """
    Process Sigma rules and create query-rule pairs JSON file
    """
    # Set default rules directory (can be overridden via command line)
    rules_dir = "./sigma_all_rules/rules/"
    
    # Check if directory was provided as command line argument
    import sys
    if len(sys.argv) > 1:
        rules_dir = sys.argv[1]
    
    # Validate that the rules directory exists
    if not os.path.exists(rules_dir):
        logger.error(f"Error: Rules directory '{rules_dir}' not found")
        sys.exit(1)
    
    # Process all rules and save results to JSON
    pairs = process_rules_to_query_pairs(rules_dir)
    save_query_rule_pairs(pairs)

# Only run the main function if this script is being run directly
if __name__ == "__main__":
    main()
