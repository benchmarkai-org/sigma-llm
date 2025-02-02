import json
import yaml
import time
from typing import Dict, List
import logging
from dotenv import load_dotenv
from datetime import datetime
from langchain.evaluation import load_evaluator
from pathlib import Path
import os
import sys
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_cases(test_file: str) -> List[Dict]:
    """
    Load test cases from a JSON file containing query-rule pairs.
    
    Expected format:
    [
        {
            "query": "Write a rule to detect...",
            "expected_rule": "title: Expected Rule\ndetection:..."
        },
        ...
    ]
    """
    with open(test_file, 'r') as f:
        return json.load(f)

def generate_rule(query: str, config: Dict) -> str:
    """
    Generate a rule by calling the microservice endpoint.
    """
    try:
        service_url = config.get("SERVICE_URL", "https://my-microservice-680275457059.us-central1.run.app")
        url = service_url.rstrip("/") + "/api/v1/rules"
        headers = {"Authorization": f"Bearer {config.get('SERVICE_API_KEY', '')}"}
        payload = {"query": query}

        # Set up a session with a retry strategy to handle 503 errors
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[503],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        rule = data.get("rule")
        if not rule:
            raise Exception("No rule returned from service")
        return rule
    except Exception as e:
        logger.error(f"Rule generation failed: {str(e)}")
        raise Exception(f"Failed to generate rule: {str(e)}")

def get_judge_comparison(rule1: str, rule2: str, config: Dict) -> Dict:
    """
    Get a judgment comparison between two rules by calling the microservice endpoint.
    """
    try:
        service_url = config.get("SERVICE_URL", "https://my-microservice-680275457059.us-central1.run.app")
        url = service_url.rstrip("/") + "/api/v1/judge"
        headers = {"Authorization": f"Bearer {config.get('SERVICE_API_KEY', '')}"}
        payload = {"rule1": rule1, "rule2": rule2}

        # Set up a session with a retry strategy to handle 503 errors
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[503],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        judgment = data.get("judgment")
        return judgment
    except Exception as e:
        logger.error(f"Judge comparison failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get judgment: {str(e)}")

def evaluate_rule(generated_rule: str, expected_rule: str, config: Dict) -> tuple:
    """
    Evaluate a generated rule against the expected rule.
    Combines Langchain-based metrics with LLM judgment.
    Returns (metrics_dict, overall_score)
    """
    try:
        # Get Langchain-based metrics
        langchain_metrics = calculate_langchain_metrics(generated_rule, expected_rule)
        
        # Get LLM judgment
        llm_judgment = get_judge_comparison(generated_rule, expected_rule, config)
        
        # Combine all metrics into one dictionary
        combined_metrics = {
            **langchain_metrics,  # Include all Langchain metrics
            "llm_judgment": llm_judgment  # Add LLM judgment
        }
        
        # Calculate final score using all metrics
        overall_score = calculate_combined_score(combined_metrics)
        
        return combined_metrics, overall_score
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def calculate_langchain_metrics(generated_rule: str, expected_rule: str) -> dict:
    """
    Evaluate a generated rule against the expected rule using multiple criteria.
    """
    try:
        generated = yaml.safe_load(generated_rule)
        expected = yaml.safe_load(expected_rule)
        
        # Initialize metrics
        metrics = {
            "valid_yaml": 1.0,
            "has_required_fields": 0.0,
            "detection_logic_similarity": 0.0,
            "metadata_completeness": 0.0
        }
        
        # Check required fields
        required_fields = ["title", "detection", "logsource"]
        fields_present = sum(1 for field in required_fields if field in generated) / len(required_fields)
        metrics["has_required_fields"] = fields_present
        
        # Use LangChain evaluator for detection logic similarity
        try:
            evaluator = load_evaluator("string_distance")
            evaluation_result = evaluator.evaluate_strings(
                prediction=str(generated.get("detection", {})),
                reference=str(expected.get("detection", {}))
            )
            metrics["detection_logic_similarity"] = evaluation_result["score"]
        except Exception as e:
            logger.error(f"Error during string distance evaluation: {str(e)}")
            metrics["detection_logic_similarity"] = 0.0
        
        # Calculate metadata completeness
        metadata_fields = ["description", "author", "date", "level", "tags"]
        metadata_score = sum(1 for field in metadata_fields if field in generated) / len(metadata_fields)
        metrics["metadata_completeness"] = metadata_score

        return metrics
    
    except yaml.YAMLError:
        return {
            "valid_yaml": 0.0,
            "has_required_fields": 0.0,
            "detection_logic_similarity": 0.0,
            "metadata_completeness": 0.0
        }

def calculate_combined_score(metrics: dict) -> float:
    """
    Calculate overall score incorporating both Langchain metrics and LLM judgment.
    Returns a weighted score between 0 and 1.
    """
    weights = {
        "valid_yaml": 0.15,
        "has_required_fields": 0.20,
        "detection_logic_similarity": 0.25,
        "metadata_completeness": 0.10,
        "llm_judgment": 0.30
    }
    
    metrics_for_calculation = metrics.copy()
    
    # Extract numerical score from llm_judgment if it exists
    if "llm_judgment" in metrics:
        try:
            judgment = metrics["llm_judgment"]
            if isinstance(judgment, dict):
                llm_score = float(judgment.get("score", 0.5))
            else:
                logger.warning(f"Unexpected llm_judgment format: {type(judgment)}. Using default score.")
                llm_score = 0.5
            
            metrics_for_calculation["llm_judgment"] = llm_score
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Could not parse LLM judgment score: {e}. Using default value of 0.5")
            metrics_for_calculation["llm_judgment"] = 0.5
    
    overall_score = sum(
        metrics_for_calculation[k] * weights[k] 
        for k in weights 
        if k in metrics_for_calculation
    )
    
    return min(max(overall_score, 0.0), 1.0)

def run_evaluation(config: Dict, test_cases: List[Dict]) -> List[Dict]:
    """
    Run evaluation on all test cases and return results.
    """
    results = []
    total_cases = len(test_cases)
    
    logger.info(f"Starting evaluation of {total_cases} test cases...")
    
    for idx, case in enumerate(test_cases, 1):
        try:
            logger.info(f"Processing case {idx}/{total_cases}: {case['query'][:50]}...")
            
            yaml_block = generate_rule(case["query"], config)
            
            # Evaluate using both Langchain metrics and LLM judgment
            metrics, score = evaluate_rule(yaml_block, case["expected_rule"], config)
            
            results.append({
                "query": case["query"],
                "generated_rule": yaml_block,
                "expected_rule": case["expected_rule"],
                "metrics": metrics,
                "overall_score": score,
            })
            
            logger.info(f"Completed case {idx}/{total_cases} with score: {score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed case {idx}/{total_cases}: {str(e)}")
            results.append({
                "query": case["query"],
                "error": str(e),
                "metrics": None,
                "overall_score": 0.0,
            })
    
    logger.info(f"Evaluation complete. Processed {total_cases} cases.")
    return results 

def save_results(results: List[Dict], output_dir: str):
    """
    Save evaluation results to a JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_file

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration for using the microservice endpoint
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "gpt-4o"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "sigma-rules"),
        "SERVICE_URL": "https://my-microservice-680275457059.us-central1.run.app",
        "SERVICE_API_KEY": os.getenv("SERVICE_API_KEY")
    }
    
    # Load test cases
    test_cases = load_test_cases("query_rule_pairs_small.json")
    
    # Run evaluation
    results = run_evaluation(config, test_cases)
    
    # Save results
    output_file = save_results(results, "evaluation/results")
    
    # Print summary
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get("metrics") is not None)
    avg_score = sum(r.get("overall_score", 0) for r in results) / total_cases
    
    print(f"\nEvaluation Summary:")
    print(f"Total test cases: {total_cases}")
    print(f"Successful generations: {successful_cases}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()