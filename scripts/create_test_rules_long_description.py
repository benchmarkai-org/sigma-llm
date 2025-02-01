import os
import yaml
import json
import requests
import urllib3
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict
import logging

# Suppress SSL warnings
urllib3.disable_warnings()

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def load_yaml_file(file_path: str) -> dict:
    """Load and parse a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def find_sigma_rules(directory: str) -> List[str]:
    """Recursively find all .yml and .yaml files in the directory."""
    sigma_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.yml', '.yaml')):
                sigma_files.append(os.path.join(root, file))
    return sigma_files

def assess_rule(rule: str, config: Dict) -> Dict:
    """Send a rule to the assessment endpoint and get the results."""
    headers = {
        'Authorization': f"Bearer {config['SERVICE_API_KEY']}",
        'Content-Type': 'application/json'
    }
    
    data = {
        'rule': rule,
        'assistant_id': config['OPENAI_ASSISTANT_ID']
    }
    
    try:
        response = requests.post(
            f"{config['RULE_GENERATOR_URL']}/api/v1/assess",
            headers=headers,
            json=data,
            verify=False,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error assessing rule: {e}")
        return None

def main():
    # Configuration
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_ASSISTANT_ID": os.getenv("ASSESSMENT_ASSISTANT_ID"),
        "RULE_GENERATOR_URL": os.getenv("RULE_GENERATOR_URL", "http://127.0.0.1:5001"),
        "SERVICE_API_KEY": os.getenv("SERVICE_API_KEY")
    }
    
    # Validate required config
    required_config = ['RULE_GENERATOR_URL', 'SERVICE_API_KEY', 'OPENAI_ASSISTANT_ID']
    if not all(key in config for key in required_config):
        raise ValueError(f"Missing required configuration. Need: {required_config}")
    
    # Find all Sigma rules
    rules_dir = "./sigma_core/rules/linux/network_connection/"
    sigma_files = find_sigma_rules(rules_dir)
    logger.info(f"Found {len(sigma_files)} Sigma rules to process")
    
    # Process each rule and store results
    results = []
    for file_path in sigma_files:
        logger.info(f"Processing {file_path}")
        
        # Load the rule
        rule_content = load_yaml_file(file_path)
        if not rule_content:
            continue
            
        # Convert rule to YAML string for assessment
        rule_yaml = yaml.dump(rule_content)
        
        # Get assessment
        assessment = assess_rule(rule_yaml, config)
        if not assessment:
            continue
            
        # Store result
        results.append({
            'file_path': file_path,
            'original_rule': rule_content,
            'assessment': assessment['assessment']
        })
    
    # Create output directory if it doesn't exist
    output_dir = Path('assessment_results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'sigma_assessments_{timestamp}.json'
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
    
    logger.info(f"\nAssessment complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()