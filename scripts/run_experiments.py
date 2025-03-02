import json
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
import sys
from evaluate import load_test_cases, run_evaluation, save_results

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_experiment_config(config_file: str) -> dict:
    """
    Load experiment configurations from a YAML file.
    """
    try:
        logger.info(f"Attempting to load config from: {config_file}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Directory contents: {os.listdir('.')}")
        logger.debug(f"Parent directory contents: {os.listdir('..')}")
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Successfully loaded config file")
            return config
    except Exception as e:
        logger.error(f"Failed to load experiment config: {str(e)}", exc_info=True)
        raise

def run_experiments(experiment_config: dict, test_cases: list, single_test: bool = False):
    """
    Run multiple experiments with different model configurations.
    Args:
        experiment_config: Configuration dictionary
        test_cases: List of test cases
        single_test: If True, only run the first test case
    """
    try:
        results = []
        
        for experiment in experiment_config['experiments']:
            experiment_name = experiment['name']
            logger.info(f"\nStarting experiment: {experiment_name}")
            
            # Create config for this experiment
            config = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
                "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "sigma-rules"),
                "SERVICE_API_KEY": os.getenv("SERVICE_API_KEY"),
                **experiment['config']  # Override with experiment-specific config
            }
            
            # Enhanced logging for model configuration
            logger.info(f"Using model: {config.get('MODEL_NAME', 'default')}")
            logger.info(f"Service URL: {config.get('SERVICE_URL')}")
            logger.debug(f"Environment variables present: {list(os.environ.keys())}")
            logger.debug(f"Full experiment config: {experiment}")
            
            # If single_test is True, only use the first test case
            current_test_cases = test_cases[:1] if single_test else test_cases
            if single_test:
                logger.info("Running in single test mode with first test case")
                logger.info(f"Test case query: {current_test_cases[0]['query']}")
            
            # Run evaluation for this experiment
            experiment_results = run_evaluation(config, current_test_cases, experiment_name)
            results.extend(experiment_results)
            
            # Save intermediate results
            save_results(experiment_results, "evaluation/results", experiment_name)
            
            # Print experiment summary
            total_cases = len(experiment_results)
            successful_cases = sum(1 for r in experiment_results if r.get("metrics") is not None)
            avg_score = sum(r.get("overall_score", 0) for r in experiment_results) / total_cases
            
            logger.info(f"\nExperiment Summary - {experiment_name}:")
            logger.info(f"Total test cases: {total_cases}")
            logger.info(f"Successful generations: {successful_cases}")
            logger.info(f"Average score: {avg_score:.2f}")
            
            # Log individual case results
            for idx, result in enumerate(experiment_results):
                logger.info(f"\nCase {idx + 1} Results:")
                logger.info(f"Query: {result['query'][:100]}...")
                logger.info(f"Score: {result.get('overall_score', 0):.2f}")
                if result.get('metrics'):
                    logger.info(f"Metrics: {result['metrics']}")
                if result.get('error'):
                    logger.error(f"Error: {result['error']}")
        
        return results
    except Exception as e:
        logger.error(f"Error in run_experiments: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Add logging for key environment variables
        logger.info(f"Using LLM_MODEL_NAME: {os.getenv('LLM_MODEL_NAME', 'not set')}")
        logger.info(f"Using MODEL_NAME: {os.getenv('MODEL_NAME', 'not set')}")
        logger.info(f"Service URL: {os.getenv('SERVICE_URL', 'not set')}")
        
        # Load experiment configurations
        logger.info("Attempting to load config...")
        experiment_config = load_experiment_config("config/experiments.yaml")
        
        # Load test cases
        logger.info("Attempting to load test cases...")
        test_cases = load_test_cases("query_rule_pairs_reference.json")
        
        # Check if we want to run a single test case
        single_test = os.getenv('SINGLE_TEST', 'false').lower() == 'true'
        if single_test:
            logger.info("Running in single test mode")
        
        # Run all experiments
        logger.info("Starting experiments...")
        results = run_experiments(experiment_config, test_cases, single_test)
        
        # Save combined results
        logger.info("Saving combined results...")
        output_file = save_results(results, "evaluation/results", "combined")
        
        logger.info(f"\nAll experiments completed. Combined results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 