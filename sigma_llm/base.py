from typing import TypedDict
from abc import ABC, abstractmethod

class RuleState(TypedDict):
    query: str
    context: str
    initial_rule: str
    improved_rule: str
    feedback: str
    final_rule: str

class LLMBase(ABC):
    @abstractmethod
    def _improve_rule(self, state: RuleState) -> RuleState:
        """
        Improve a given Sigma rule based on the current state.
        
        Args:
            state: Current state containing the rule and context
            
        Returns:
            Updated state with improved rule
        """
        raise NotImplementedError

    @abstractmethod
    def judge_rules(self, rule1: str, rule2: str) -> str:
        """
        Compare two Sigma rules and determine which is better.
        
        Args:
            rule1: First Sigma rule to compare
            rule2: Second Sigma rule to compare
            
        Returns:
            String explaining which rule is better and why
        """
        raise NotImplementedError

    @abstractmethod 
    def assess_rule(self, rule: str) -> str:
        """
        Assess the quality of a single Sigma rule.
        
        Args:
            rule: Sigma rule to assess
            
        Returns:
            String assessment of the rule's quality
        """
        raise NotImplementedError 