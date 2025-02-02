from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, Sequence
import operator
import logging
import time
from functools import wraps
import pinecone
import os
from pinecone import Pinecone
import weave
from datetime import datetime
from .base import LLMBase
import re

# Configure logging
logger = logging.getLogger(__name__)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

class RuleState(TypedDict):
    query: str
    context: str
    initial_rule: str
    improved_rule: str
    feedback: str
    final_rule: str

class LLMManager(LLMBase):
    def __init__(self, model_name: str = "gpt-4o",
                 wandb_project: str = "sigma-rule-evaluation",
                 use_improvement_loop: bool = False,
                 max_iterations: int = 2):
        logger.info(f"Initializing LLMManager with model: {model_name}")
        try:
            self.vectorstore = self._init_vectorstore()
            self.llm = self._init_llm(model_name)
            self.use_improvement_loop = use_improvement_loop
            self.max_iterations = max_iterations
            self._improvement_counter = 0
            self.rule_generation_graph = self._create_rule_generation_graph()
            self._init_weave(wandb_project) # Initialize weave here
            logger.info("LLMManager initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {str(e)}", exc_info=True)
            raise

    def _init_weave(self, wandb_project: str):
        """Initialize weave, handling potential errors."""
        try:
            logger.info(f"Initializing weave with project: {wandb_project}")
            # Simplified weave initialization
            weave.init(wandb_project)
            logger.info("weave initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize weave: {str(e)}", exc_info=True)
            raise

    @weave.op()
    def _init_vectorstore(self) -> PineconeVectorStore:
        """Initialize connection to Pinecone"""
        logger.info("Connecting to Pinecone")
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "sigma-2024-05-13")  # Default if not set
        
        if not pinecone_api_key:
            raise ValueError("Missing required Pinecone API key")
            
        try:
            # Initialize Pinecone with new SDK pattern
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            return PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text",
                namespace="sigma-rules"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}", exc_info=True)
            raise

    @weave.op()
    def _init_llm(self, model_name: str):
        logger.info(f"Initializing LLM with model: {model_name}")
        
        if model_name.startswith("gpt-4o"):
            return ChatOpenAI(
                model_name="gpt-4o",
                temperature=0
            )
        elif model_name == "gemini-1.5-flash":
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif model_name == "claude-3-5-sonnet-latest":
            return ChatAnthropic(
                model="claude-3-5-sonnet-latest",
                temperature=0,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @weave.op()
    def _retrieve_context(self, state: RuleState) -> RuleState:
        logger.info(f"Retrieving context for query: {state['query'][:100]}...")
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        context = retriever.get_relevant_documents(state["query"])
        state["context"] = "\n".join([doc.page_content for doc in context])
        logger.info(f"Retrieved {len(context)} relevant documents")
        return state

    @weave.op()
    def _generate_initial_rule(self, state: RuleState) -> RuleState:
        logger.info("Generating initial rule")
        prompt = ChatPromptTemplate.from_template("""
        You are an expert cybersecurity analyst specializing in Sigma rules. Your task is to create a Sigma rule based on the provided context and user prompt.

        Reference rules for context (which may or may not be wholly relevant to the rule that is requested - you'll have to judge that):
        {context}
        
        Request for detection: {query}

        Create a precise Sigma rule that:
        - Follows official Sigma syntax and best practices
        - **Maximizes detection logic effectiveness** while minimizing false positives
        - Uses appropriate log sources and field mappings
        - Includes clear title, description, and relevant tags
        - Considers performance impact on SIEM systems

        Return only the YAML rule without any additional explanation.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            state["initial_rule"] = chain.invoke({
                "context": state["context"],
                "query": state["query"]
            })
            logger.info(f"Initial rule generated (length: {len(state['initial_rule'])} characters)")
        except Exception as e:
            logger.error(f"Error generating initial rule: {e}", exc_info=True)
            raise
        return state

    @weave.op()
    def _improve_rule(self, state: RuleState) -> RuleState:
        logger.info("Improving rule")
        if not hasattr(self, '_improvement_counter'):
            self._improvement_counter = 0
        self._improvement_counter += 1  # Increment the counter
        iteration = self._improvement_counter
        
        # Base prompt without feedback
        base_prompt = """
        You are a senior SIEM engineer and Sigma rule expert. Analyze and enhance the following rule for maximum effectiveness.

        Original Rule:
        {rule}

        Original Detection Request:
        {query}

        Similar Reference Rules:
        {context}"""

        # Separate feedback section with proper formatting
        feedback_section = """

        Previous Validation Feedback:
        {feedback}

        Address the feedback above and improve the rule considering:""" if state["feedback"] else "\nImprove the rule considering:"
        
        improvement_instructions = """
        1. Detection Coverage
           - **Ensure comprehensive detection logic effectiveness** of the target behavior
           - Explicitly state how you are addressing any gaps in detection coverage
           - Explain any changes you are making to improve detection logic

        2. Performance Optimization
           - Optimize field selections and conditions
           - Reduce complexity where possible
           - Consider index usage and search efficiency

        3. False Positive Reduction
           - Add precise conditions to filter legitimate behavior
           - Explain how you are reducing the potential for false positives
           - Include appropriate thresholds if needed
           - Consider environmental context

        4. Sigma Compliance
           - Follow latest Sigma specification
           - Use correct field mappings
           - Include all required attributes

        Return only the improved YAML rule without explanation."""
        
        full_prompt = base_prompt + feedback_section + improvement_instructions
        
        # Debug logging to verify prompt construction
        logger.debug(f"Feedback present: {bool(state['feedback'])}")
        logger.debug(f"Full prompt: {full_prompt}")
        
        prompt = ChatPromptTemplate.from_template(full_prompt)
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            state["improved_rule"] = chain.invoke({
                "rule": state["initial_rule"] if not state["improved_rule"] else state["improved_rule"],
                "context": state["context"],
                "query": state["query"],
                "feedback": state["feedback"]
            })
            logger.info(f"Rule improved (iteration {iteration})")
        except Exception as e:
            logger.error(f"Error improving rule: {e}", exc_info=True)
            raise
        return state

    @weave.op()
    def _validate_rule(self, state: RuleState) -> RuleState:
        logger.info("Validating rule")
        prompt = ChatPromptTemplate.from_template("""
        You are a quality assurance specialist for SIEM detections with deep expertise in Sigma rules. Perform a thorough technical review of this rule:

        {improved_rule}

        Conduct a comprehensive analysis covering:

        1. Syntax and Structure
           - Validate YAML syntax
           - Verify required fields (title, id, status, description, etc.)
           - Check field naming conventions
           - Validate condition logic

        2. Detection Logic
           - **Assess detection logic effectiveness** and completeness of detection coverage
           - Verify logical operators and grouping
           - Check for logic gaps or flaws
           - Evaluate condition specificity

        3. Technical Implementation
           - Verify field mappings exist in common SIEM platforms
           - Check for unsupported features or syntax
           - Assess search performance impact
           - Validate value formats and types

        4. Detection Effectiveness
           - Identify potential blind spots
           - Assess susceptibility to evasion
           - Evaluate false positive likelihood
           - Check for common implementation pitfalls

        Provide specific, detailed, and actionable feedback for any issues found. Be direct and thorough in your analysis.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            state["feedback"] = chain.invoke({
                "improved_rule": state["improved_rule"]
            })
            logger.info(f"Validation feedback received (length: {len(state['feedback'])} characters)")
        except Exception as e:
            logger.error(f"Error validating rule: {e}", exc_info=True)
            raise
        return state

    def _should_refine(self, state: RuleState) -> bool:
        """Determine if the rule needs further refinement using the LLM."""
        if not hasattr(self, '_improvement_counter'):
            self._improvement_counter = 0
            
        # First check if we've hit the max iterations
        if self._improvement_counter >= self.max_iterations:
            logger.info(f"Max refinement iterations ({self.max_iterations}) reached")
            return False
            
        # Only proceed with LLM check if we haven't hit the limit
        prompt = ChatPromptTemplate.from_template("""
        You are an expert in Sigma rules and detection engineering. Determine if the following feedback on a Sigma rule indicates a need for further refinement:

        Feedback:
        {feedback}

        Consider the following when making your decision:
        - Are there any critical issues that need to be addressed?
        - Are there any gaps in detection logic or coverage?
        - Is there a high potential for false positives?
        - Are there any technical implementation issues?

        Respond with "true" if the rule requires further refinement, or "false" if the rule is acceptable. Do not provide any additional explanation.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            needs_refinement = chain.invoke({"feedback": state["feedback"]}).lower() == "true"
            logger.info(f"LLM determined rule refinement needed: {needs_refinement}")
            return needs_refinement
        except Exception as e:
            logger.error(f"Error determining if rule needs refinement: {e}", exc_info=True)            
            return False

    @weave.op()
    def _finalize_rule(self, state: RuleState) -> RuleState:
        logger.info("Finalizing rule")
        # Use improved_rule if it exists, otherwise use initial_rule
        state["final_rule"] = state["improved_rule"] if state["improved_rule"] else state["initial_rule"]
        logger.info(f"Rule finalized (length: {len(state['final_rule'])} characters)")
        return state

    def _should_use_improvement_loop(self, state: RuleState) -> bool:
        """Determine whether to use the improvement loop or direct path"""
        return self.use_improvement_loop

    @weave.op()
    def _create_rule_generation_graph(self):
        logger.info("Creating rule generation workflow graph")
        workflow = StateGraph(RuleState)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_initial", self._generate_initial_rule)
        workflow.add_node("improve_rule", self._improve_rule)
        workflow.add_node("validate_rule", self._validate_rule)
        workflow.add_node("finalize_rule", self._finalize_rule)

        # Define edges
        workflow.add_edge("retrieve_context", "generate_initial")
        
        # Add conditional path after initial generation
        workflow.add_conditional_edges(
            "generate_initial",
            self._should_use_improvement_loop,
            {
                True: "improve_rule",  # Use improvement loop
                False: "finalize_rule"  # Direct path
            }
        )
        
        # Rest of the improvement loop remains the same
        workflow.add_edge("improve_rule", "validate_rule")
        workflow.add_conditional_edges(
            "validate_rule",
            self._should_refine,
            {
                True: "improve_rule",
                False: "finalize_rule"
            }
        )
        
        workflow.add_edge("finalize_rule", END)
        workflow.set_entry_point("retrieve_context")
        
        logger.info("Workflow graph created successfully")
        return workflow.compile()

    @weave.op()
    def generate_rule(self, query: str) -> str:
        logger.info(f"Starting rule generation for query: {query[:100]}...")
        # Reset the improvement counter at the start of each rule generation
        self._improvement_counter = 0

        try:
            initial_state = RuleState(
                query=query,
                context="",
                initial_rule="",
                improved_rule="",
                feedback="",
                final_rule=""
            )
            
            final_state = self.rule_generation_graph.invoke(initial_state)
            result = self._extract_yaml(final_state["final_rule"])

            return result
        except ValueError as e:
            logger.error(f"Rule generation failed: {str(e)}", exc_info=True)
            raise

    def _count_improvements(self, state: RuleState) -> int:
        """Count how many improvement iterations were needed"""
        if not hasattr(self, '_improvement_counter'):
            self._improvement_counter = 0
        return self._improvement_counter

    @weave.op()
    def judge_rules(self, rule1: str, rule2: str) -> str:
        logger.info("Starting rule comparison")
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert security analyst specializing in SIGMA rules. You can use the uploaded content to enhance your understanding. Compare the following two SIGMA rules:

        GENERATED RULE:
        ```yaml
        {rule1}
        ```

        EXPECTED RULE:
        ```yaml
        {rule2}
        ```

        Analyze, compare, and score these rules based on these specific weighted criteria:
        1. Detection Logic Effectiveness (50%): How effectively and precisely does the generated rule capture the intended detection logic of the expected rule? Consider:
           - Specific conditions, fields, and operators used
           - Alignment with intended detection logic
           - Critical missing/extra conditions (-0.5 per major issue)
           - Incorrect logical operators (AND vs OR, -0.3)
           - Functional equivalence to expected rule
        2. Coverage Completeness (20%): 
           - All necessary conditions/fields present (-0.2 per missing element)
           - Handling of edge cases
           - Temporal coverage (time windows)
        3. False Positive Potential (20%):
           - Specificity of conditions (-0.4 for broad wildcards)
           - Thresholds for noisy events
           - Environmental considerations
        4. Technical Implementation (10%):
           - Sigma syntax compliance
           - Field mapping correctness
           - Search optimization

        Scoring Guidelines:
        - 1.0 = Perfect functional equivalence
        - 0.9-0.99 = Minor formatting differences
        - 0.8-0.89 = Missing non-critical conditions
        - 0.6-0.79 = Missing important conditions/fields
        - <0.6 = Fundamental logic errors/missing core detection elements
        - 0 = Completely ineffective

        Provide your evaluation in this strict JSON format:
        {{
            "score": <float between 0 and 1>,
            "reasoning": "<concise technical explanation focusing on security impact>",
            "criteria_scores": {{
                "detection_logic": <float 0-1>,
                "completeness": <float 0-1>,
                "false_positive_rate": <float 0-1>,
                "technical_quality": <float 0-1>
            }}
        }}

        Be strict and use the full scoring range:
        - Deduct 0.5 points for missing critical conditions
        - Deduct 0.3 for incorrect logical operators
        - Deduct 0.2 for unnecessary fields increasing FPs
        - Add 0.1 bonus for superior optimizations

        Example scoring:
        - Missing one critical condition: 0.5 (0.5 * 1.0) + ... = total <0.75
        - Incorrect AND/OR logic: 0.7 (0.7 * 1.0) + ... = total <0.8
        - Perfect match but suboptimal fields: 0.95 * 0.5 + ... = ~0.9
        """)
        
        # Instantiate a separate judge LLM using o1 from OpenAI.
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        )

        chain = prompt | judge_llm | StrOutputParser()
        try:
            result = chain.invoke({
                "rule1": rule1,
                "rule2": rule2
            })
            logger.info("Rule comparison completed")
            return result
        except Exception as e:
            logger.error(f"Error during rule comparison: {e}", exc_info=True)
            raise

    @weave.op()
    def assess_rule(self, rule: str) -> str:
        logger.info("Starting rule assessment")
        
        prompt = ChatPromptTemplate.from_template("""
        You are a detection engineering expert with extensive experience in SIEM systems and Sigma rule development. Conduct a comprehensive security assessment of this rule:

        {rule}

        Provide a detailed analysis covering:

        1. Detection Coverage Analysis
           - Identify detected attack patterns and techniques
           - Map coverage to MITRE ATT&CK where applicable
           - **Assess detection logic effectiveness** and completeness
           - Identify potential detection bypass methods
           - Evaluate coverage of attack variants

        2. Technical Implementation Review
           - Validate Sigma syntax compliance
           - Assess field selection efficiency
           - Evaluate condition logic completeness
           - Check for technical limitations
           - Review value formatting and types

        3. Performance Impact Assessment
           - Analyze search pattern efficiency
           - Evaluate resource requirements
           - Assess scaling considerations
           - Identify potential optimization opportunities
           - Consider impact on SIEM performance

        4. False Positive Analysis
           - Identify potential false positive scenarios
           - Assess filter effectiveness
           - Evaluate threshold appropriateness
           - Consider environmental factors
           - Suggest false positive reduction strategies

        5. Maintainability Evaluation
           - Review documentation quality
           - Assess update requirements
           - Evaluate rule complexity
           - Check for dependencies
           - Consider long-term sustainability

        Provide specific, actionable recommendations for improvements in each area. Include technical details and reasoning for all findings.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"rule": rule})
            logger.info("Rule assessment completed")
            return result
        except Exception as e:
            logger.error(f"Error during rule assessment: {e}", exc_info=True)
            raise

    @staticmethod
    def _extract_yaml(text: str) -> str:
        logger.debug("Extracting YAML from response")
        try:
            match = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                logger.warning("YAML extraction failed, returning raw text")
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting YAML: {e}", exc_info=True)
            return text.strip()