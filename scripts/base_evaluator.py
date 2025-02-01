import json
import yaml
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
from langchain.evaluation import load_evaluator
from pathlib import Path
from abc import ABC, abstractmethod
from sigma_llm.base import LLMBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    def __init__(self, config: Dict):
        self.config = config
        
    @abstractmethod
    def generate_rule(self, query: str) -> str:
        """Generate a rule using the specific implementation."""
        pass
        
    @abstractmethod
    def get_judge_comparison(self, rule1: str, rule2: str) -> Dict:
        """Get a judgment comparison between two rules."""
        pass

    def load_test_cases(self, test_file: str) -> List[Dict]:
        """
        Load test cases from a JSON file containing query-rule pairs.
        """
        try:
            with open(test_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in test file: {test_file}")
            raise

    def evaluate_rule(self, generated_rule: str, expected_rule: str) -> Tuple[Dict, float]:
        """
        Evaluate a generated rule against the expected rule.
        Returns (metrics_dict, overall_score)
        """
        try:
            # Get Langchain-based metrics
            langchain_metrics = self.calculate_langchain_metrics(generated_rule, expected_rule)
            
            # Get LLM judgment
            llm_judgment = self.get_judge_comparison(generated_rule, expected_rule)
            
            # Combine metrics
            combined_metrics = {
                **langchain_metrics,
                "llm_judgment": llm_judgment
            }
            
            # Calculate final score
            overall_score = self.calculate_combined_score(combined_metrics)
            
            return combined_metrics, overall_score
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def calculate_langchain_metrics(self, generated_rule: str, expected_rule: str) -> Dict[str, float]:
        """
        Calculate metrics using Langchain evaluators.
        """
        try:
            generated = yaml.safe_load(generated_rule)
            expected = yaml.safe_load(expected_rule)
            
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
            
            # Calculate detection logic similarity
            try:
                evaluator = load_evaluator("string_distance")
                evaluation_result = evaluator.invoke(
                    prediction=str(generated.get("detection", {})),
                    reference=str(expected.get("detection", {}))
                )
                metrics["detection_logic_similarity"] = evaluation_result["score"]
            except Exception as e:
                logger.error(f"Error during string distance evaluation: {e}")
                metrics["detection_logic_similarity"] = 0.0
            
            # Calculate metadata completeness
            metadata_fields = ["description", "author", "date", "level", "tags"]
            metadata_score = sum(1 for field in metadata_fields if field in generated) / len(metadata_fields)
            metrics["metadata_completeness"] = metadata_score

            return metrics
        
        except yaml.YAMLError:
            logger.error("YAML parsing error")
            return {
                "valid_yaml": 0.0,
                "has_required_fields": 0.0,
                "detection_logic_similarity": 0.0,
                "metadata_completeness": 0.0
            }

    def calculate_combined_score(self, metrics: Dict) -> float:
        """
        Calculate overall score from all metrics.
        """
        weights = {
            "valid_yaml": 0.15,
            "has_required_fields": 0.20,
            "detection_logic_similarity": 0.25,
            "metadata_completeness": 0.10,
            "llm_judgment": 0.30
        }
        
        metrics_for_calculation = metrics.copy()
        
        if "llm_judgment" in metrics:
            try:
                if isinstance(metrics["llm_judgment"], str):
                    judgment_dict = json.loads(metrics["llm_judgment"])
                    if isinstance(judgment_dict, dict):
                        llm_score = float(judgment_dict.get("score", 0.5))
                    else:
                        score_mapping = {
                            "excellent": 1.0,
                            "good": 0.75,
                            "fair": 0.5,
                            "poor": 0.25,
                            "bad": 0.0
                        }
                        llm_score = score_mapping.get(metrics["llm_judgment"].lower(), 0.5)
                else:
                    llm_score = float(metrics["llm_judgment"])
                
                metrics_for_calculation["llm_judgment"] = llm_score
            except (ValueError, AttributeError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Could not parse LLM judgment score: {e}. Using default value of 0.5")
                metrics_for_calculation["llm_judgment"] = 0.5
        
        overall_score = sum(
            metrics_for_calculation[k] * weights[k] 
            for k in weights 
            if k in metrics_for_calculation
        )
        
        return min(max(overall_score, 0.0), 1.0)

    def run_evaluation(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Run evaluation on all test cases.
        """
        results = []
        total_cases = len(test_cases)
        
        logger.info(f"Starting evaluation of {total_cases} test cases...")
        
        for idx, case in enumerate(test_cases, 1):
            try:
                logger.info(f"Processing case {idx}/{total_cases}: {case['query'][:50]}...")
                
                yaml_block = self.generate_rule(case["query"])
                
                metrics, score = self.evaluate_rule(yaml_block, case["expected_rule"])
                
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

    def save_results(self, results: List[Dict], output_dir: str) -> Path:
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