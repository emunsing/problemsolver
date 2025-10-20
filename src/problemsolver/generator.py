#!/usr/bin/env python3
"""
Optimizer Generator using Large Language Models

This module generates new optimization algorithms using LLMs, validates them,
benchmarks their performance, and checks if they advance the Pareto frontier.
"""

import time
import os
import sys
import csv
import random
import pathlib
import importlib.util
from typing import List, Dict, Tuple, Optional, Callable

import click
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from problemsolver.utils import check_optimizer_annotations, check_optimizer_function, Interval, Performance, to_camel_case
from problemsolver.evaluator import benchmark_optimizer, generate_test_functions, MAX_ALLOWED_PROBLEM_TIME, MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME
from problemsolver.pareto_metrics import ParetoMetric, StrictDominanceParetoMetric, ConvexHullParetoMetric

import logging

def setup_logging(log_level: str = "INFO"):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=numeric_level,
        force=True  # Override any existing configuration
    )

# Set up default logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


class OptimizerGenerator:
    def __init__(self, api_key: str,
                 api_base: str | None = None,
                 model_name: str = "o4-mini",
                 n_tune_functions: int = 10,
                 n_test_functions: int = 20,
                 n_tuning_trials: int = 100,
                 n_dims: int = 5,
                 output_dir: str = "data/output",
                 pareto_rtol: float = 0.0, n_jobs: int = 1,
                 max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME,
                 max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME,
                 pareto_metric: ParetoMetric | None = None):
        """Initialize the optimizer generator."""
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=model_name, request_timeout=300, max_retries=3,
        )
        self.n_tune_functions = n_tune_functions
        self.n_test_functions = n_test_functions
        self.n_tuning_trials = n_tuning_trials
        self.max_allowed_time_per_function = max_allowed_time_per_function
        self.max_allowed_rolling_average_function_time = max_allowed_rolling_average_function_time
        self.n_dims = n_dims
        self.n_jobs = n_jobs
        self.output_dir = pathlib.Path(output_dir).expanduser().resolve()
        self.pareto_rtol = pareto_rtol
        self.pareto_metric = pareto_metric or StrictDominanceParetoMetric

        self.performance_file = self.output_dir / "optimizers_performant.csv"
        self.load_performance_file(self.performance_file, empty_ok=False)  # Ensure that file exists and is well-formatted
        self.all_performance_file = self.output_dir / "optimizers_all.csv"

        self.code_output_dir_all = self.output_dir / "code" / "other"        
        self.code_output_dir_performant =  self.output_dir / "code" / "performant"
        
        # Ensure directories exist
        os.makedirs(self.code_output_dir_all, exist_ok=True)
        os.makedirs(self.code_output_dir_performant, exist_ok=True)
        
        # Clean up old temp files on initialization
        temp_dir = pathlib.Path(__file__).parent / "data" / "temp"
        if temp_dir.exists():
            self._cleanup_old_temp_files(temp_dir)
        

    @staticmethod
    def load_emergent_ideas(ideas_file: os.PathLike) -> List[str]:
        """Load emergent optimization ideas from the text file."""
        with open(ideas_file, 'r') as f:
            ideas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return ideas

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for the LLM."""
        return """You are an expert researcher in nonconvex/nonlinear mathematical optimization techniques and an expert programmer in Python. 

Your task is to create novel numerical optimization algorithms inspired by emergent behaviors in nature and complex systems. You should:

1. Think critically about how the given emergent behavior could inspire novel minimization techniques
2. Consider the mathematical principles underlying the behavior
3. Translate these principles into algorithmic components
4. Implement a working Python function that follows the specified signature
5. Ensure the code is efficient and accurate; you will be judged on both accuracy and efficiency
6. Your answer should only be code and any docstrings; no preamble or explanation is allowed. Assume that your answer will be directly executed; any non-code non-commented text will cause an error.

You must create a complete, runnable Python function that can be executed immediately. The function should be well-documented and follow Python best practices."""


    def get_generation_prompt(self, inspiration: str) -> str:
        """Generate the prompt for creating a new optimizer."""
        return f"""Create a novel nonconvex optimization algorithm inspired by this emergent behavior:

# INSPIRATION: {inspiration}

{self.get_requirements_prompt()}

CRITICAL THINKING:
Consider how {inspiration} relates to optimization:
- What mathematical principles underlie this behavior?
- How does this behavior achieve its goals efficiently?
- What aspects could be adapted for function minimization?
- How can we model the key mechanisms algorithmically?

Create a complete, runnable Python function that implements your novel algorithm."""

    def get_requirements_prompt(self) -> str:
        return """
# GOALS AND EVALUATION:
- Create a novel algorithm that combines ideas from existing metaheuristics with inspiration from the naturally occuring emergent behavior above.
- Avoid simple exploration/exploitation or canonical first- and second-order methods. Aim to create something new and innovative, fully utilizing the inspiration from naturally occuring systems.
- Focus on both accuracy (finding good minima) and efficiency (total computation time) tested in the following way:
  - Accuracy and efficiency will be evaluated by a downstream test script which takes functions of the standard signature.
  - Randomly generated functions will be used to tune the optimizer's hyperparameters with a standard hyperparameter tuner with a fixed budget.
  - The tuned optimizer will be tested on a set of test functions drawn from the same distribution.
  - Because the test functions are drawn randomly, you cannot attempt to overfit to the test metric.
  - Problems which take more than a time limit (several seconds) will be considered failures; consider this when designing your algorithm and use vectorization and other techniques to make your algorithm efficient.
- Consider how the emergent behavior's principles can be mathematically modeled concisely.
- Think about what makes this behavior effective in nature and how to translate that to optimization
- Choose implementations which are efficient and scalable, avoiding unnecessary complexity or computationally expensive operations

# TEST FUNCTION ASSUMPTIONS:
- Each test function has a global minimum somewhere in the [-10, 10] hypercube
- Functions are smooth, continuous, real-valued, and bounded below
- Functions have a global minimum but also has many local minima and saddle points
- Functions may take arbitrary dimensionality
- Winning designs will achieve a relative error better than 1e-4 and less than 0.1s average compute time when solving a 2-d test function.

REQUIREMENTS:
1. Function signature must include a callable problem function and an initial guess in addition to hyperparameters in kwargs, following the form: `minimize(fun: Callable[[np.ndarray], float], initial_guess: np.ndarray, **kwargs) -> np.ndarray`
2. At least one hyperparameter in kwargs should be annotated for hyperparameter optimization with Annotated[type, Interval(...)] as described below
3. Must handle arbitrary dimensionality (at least 10 dimensions)
4. Must return a numpy array of the same shape as initial_guess
5. Must be pure Python with numpy (no scipy or other advanced libraries)
6. Must include proper error handling and edge cases
7. Include a docstring with a succinct and insightful 1-2 sentence description of the algorithm, its inspiration, and its key features.

DETAILS FOR HYPERPARAMETER OPTIMIZATION:
- Standard hyperparameters in the function signature can be defined with default values
- Hyperparameters which require tuning should be annotated using `Annotated[type, Interval(...)]` where `Interval` defines the range of values for hyperparameter optimization.
- We will run a hyperparameter optimization script which will tune these annotated hyperparameters downstream.
  - Because of the fixed hyperparameter optimization budget, we would recommend being judicious with the number of hyperparameters to tune (generally 2-4 is a good number)
- The `Interval` class for annotation will be accessible in the environment where the function is executed, and is defined as follows:
```
class Interval:
   # Optuna metadata class for use with parameter annotations using typing.Annotated
   # Low and high are required, and must be numeric. 
   # Step is optional, and should be None if log=True.
    def __init__(self, low: int | float, high: int | float, step: int | float | None=None, log: bool=False):
        ...  # Assignment to Interval properties
```

EXAMPLE FUNCTION SIGNATURE:
```python
def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    n_estimators: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 50,
    learning_rate: Annotated[float, Interval(low=0.01, high=1.0, log=True)] = 0.1,
    alpha: Annotated[float, Interval(low=0.1, high=0.9, step=0.05, log=False)] = 0.5,
    beta: float = 0.5,
    rtol: float = 1e-6,
    max_iterations: int = 1000,
    seed: int = None
) -> np.ndarray:
    # INSPIRATION: ...
    # Optimization algorithm implementation here
```
"""

    @staticmethod
    def get_debug_prompt(original_prompt: str, code: str, error: str) -> str:
        """Generate a prompt for debugging the code."""
        return f"""The previous code did not successfully generate a valid optimizer. Please fix it and return the corrected version.

ORIGINAL PROMPT:
{original_prompt}

GENERATED CODE:
```python
{code}
```

ERROR:
{error}

Please fix the error and return the corrected Python function. Ensure it follows all the original requirements."""

    @staticmethod
    def extract_func_and_code_from_response(response: str) -> tuple[Callable, str]:
        """Extract Python code from the LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end == -1:
                response = response[start:].strip()
            else:
                response = response[start:end].strip()

        lines = response.split('\n')
        code_lines = []

        for line in lines:
            if line == "from __future__ import annotations":
                continue
            elif line.strip().startswith("Interval ="):
                continue
            code_lines.append(line)

        initial_code = '\n'.join(code_lines) if code_lines else response

        # Filter out Interval class definitions
        filtered_lines = []
        skip_until_class_end = False
        
        for line in initial_code.split('\n'):
            stripped = line.strip()
            
            # Check if this line starts an Interval class definition
            if stripped.startswith('class Interval'):
                skip_until_class_end = True
                continue
            
            # If we're in an Interval class, skip until we find the end
            if skip_until_class_end:
                # Check if we've reached the end of the class (no indentation or empty line)
                if not stripped or (stripped and not line.startswith(' ') and not line.startswith('\t')):
                    skip_until_class_end = False
                continue
            
            # Include all other lines
            filtered_lines.append(line)
        
        filtered_code = '\n'.join(filtered_lines)

        # Create a pickleable function by writing to a file in the package hierarchy
        optimizer_func = OptimizerGenerator._create_package_function(filtered_code)
        return optimizer_func, filtered_code

    @staticmethod
    def _create_package_function(code: str) -> Callable:
        """Create a pickleable function by writing code to a file in the package hierarchy."""
        import uuid
        import importlib
        
        # Create a unique module name
        module_name = f"optimizer_{uuid.uuid4().hex[:8]}"
        
        # Get the path to the temp directory in the package
        temp_dir = pathlib.Path(__file__).parent / "data" / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Clean up old optimizer files (keep only __init__.py)
        OptimizerGenerator._cleanup_old_temp_files(temp_dir)
        
        # Create the file path
        file_path = temp_dir / f"{module_name}.py"
        
        # Write the code to the file with proper imports
        full_code = f"""import numpy as np
from typing import Annotated
from problemsolver.utils import Interval

{code}
"""
        
        with open(file_path, 'w') as f:
            f.write(full_code)
        
        try:
            # Import the module using the full package path
            full_module_name = f"problemsolver.data.temp.{module_name}"
            spec = importlib.util.spec_from_file_location(full_module_name, file_path)
            temp_module = importlib.util.module_from_spec(spec)
            
            # Register the module in sys.modules
            sys.modules[full_module_name] = temp_module
            
            # Execute the module
            spec.loader.exec_module(temp_module)
            
            # Get the minimize function from the module
            optimizer_func = temp_module.minimize
            
            # Store the file path and module name for cleanup
            optimizer_func._temp_file_path = str(file_path)
            optimizer_func._temp_module_name = full_module_name
            
            return optimizer_func
            
        except Exception as e:
            # Clean up the file if there was an error
            try:
                file_path.unlink()
            except:
                pass
            raise e

    def validate_optimizer_code(self, optimizer_func: Callable, raw_code: str, original_prompt: str, max_iterations: int = 5) -> Tuple[bool, Optional[Callable], str, str]:
        """Validate the optimizer function through multiple iterations of debugging."""
        try:
            logger.info(f"Validating optimizer annotations")
            check_optimizer_annotations(optimizer_func)
        except ValueError as ve:
            if "No Annotated parameters with Interval" in str(ve):
                logger.warning(f"Annotation error; continuing: {str(ve)}")
            else:
                raise ve
        logger.info(f"Validating optimizer function")
        check_optimizer_function(optimizer_func)
        logger.info("✓ Optimizer function is valid")
        return True, optimizer_func, raw_code, ""   # If we get here, the function is valid


    def generate_optimizer(self, messages, original_prompt, max_iterations: int = 5) -> Tuple[bool, Optional[Callable], str, str]:
        """Generate a new optimizer based on the inspiration."""
        success = False
        final_func = None
        final_code = ""
        raw_code = "No code generated yet; please retry."
        error_msg = ""

        for iteration in range(max_iterations):
            try:
                response = self.llm.invoke(messages)
                logger.info(f"Iteration {iteration + 1} optimizer generation: Response received")
                optimizer_func, raw_code = self.extract_func_and_code_from_response(response.content)
                # with open("/Users/eric/src/problemsolver/src/problemsolver/optimizers/bioinspired/firefly.py", "r") as f:
                #     loaded_text = f.read()
                # optimizer_func, raw_code = self.extract_func_and_code_from_response(loaded_text)

                # Validate and debug
                logger.info(f"Iteration {iteration + 1} optimizer generation: Validating code")
                success, final_func, final_code, error_msg = self.validate_optimizer_code(optimizer_func,
                                                                          raw_code=raw_code,
                                                                          original_prompt=original_prompt)
                if success:
                    break  # Exit loop if successful
            except Exception as e:
                error_msg = str(e)
                logger.info(f"Iteration {iteration + 1} optimizer generation: Error - {error_msg}")
                debug_prompt = self.get_debug_prompt(original_prompt, raw_code, error_msg)
                messages = [
                    SystemMessage(content=self.get_system_prompt()),
                    HumanMessage(content=debug_prompt)
                ]
                continue

        if success:
            logger.info("✓ Optimizer code generated successfully")
        else:
            logger.info(f"✗ Failed to generate valid optimizer: {error_msg}")

        return success, final_func, final_code, error_msg

    def benchmark_new_optimizer(self, optimizer_func: Callable, optimizer_name: str) -> Optional[Performance]:
        """Benchmark the new optimizer and return performance metrics."""
        try:
            # Generate test functions
            tune_functions = generate_test_functions(n_samples=self.n_tune_functions, n_dims=self.n_dims)
            test_functions = generate_test_functions(n_samples=self.n_test_functions, n_dims=self.n_dims)

            # Run benchmark with the function
            log_rel_error, time_elapsed, best_params = benchmark_optimizer(
                optimizer=optimizer_func,
                test_functions=test_functions,
                tune_functions=tune_functions,
                n_tuning_trials=self.n_tuning_trials,
                n_jobs=self.n_jobs,
                max_allowed_time_per_function=self.max_allowed_time_per_function,
                max_allowed_rolling_average_function_time=self.max_allowed_rolling_average_function_time
            )

            return Performance(
                name=optimizer_name,
                log_rel_error=log_rel_error,
                time_elapsed=time_elapsed,
                best_params=best_params
            )
        except Exception as e:
            logger.warning(f"Benchmarking failed: {e}")
            return Performance(
                name=optimizer_name,
                failure_mode='benchmark_exception',
                error=str(e)
            )
        finally:
            # Clean up temporary file if it exists
            self._cleanup_package_file(optimizer_func)

    @staticmethod
    def _cleanup_old_temp_files(temp_dir: pathlib.Path) -> None:
        """Clean up old optimizer files from the temp directory, keeping only __init__.py."""
        try:
            for file_path in temp_dir.glob("optimizer_*.py"):
                try:
                    file_path.unlink()
                except (OSError, FileNotFoundError):
                    # Ignore errors if file is already deleted or doesn't exist
                    pass
        except Exception:
            # Ignore any errors during cleanup
            pass

    @staticmethod
    def _cleanup_package_file(optimizer_func: Callable) -> None:
        """Clean up temporary file and module associated with the optimizer function."""
        if hasattr(optimizer_func, '_temp_file_path'):
            try:
                os.unlink(optimizer_func._temp_file_path)
            except (OSError, FileNotFoundError):
                # Ignore errors if file is already deleted or doesn't exist
                pass
        
        # Remove the module from sys.modules
        if hasattr(optimizer_func, '_temp_module_name'):
            module_name = optimizer_func._temp_module_name
            if module_name in sys.modules:
                del sys.modules[module_name]


    @staticmethod
    def load_performance_file(performance_file: os.PathLike, empty_ok: bool = False) -> List[Performance]:
        """Load existing performance data from CSV."""
        if not os.path.exists(performance_file):
            if empty_ok:
                return []
            else:
                raise FileNotFoundError(f"Performance file {performance_file} does not exist")
        
        results = []
        with open(performance_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(Performance.from_csv_row(row))
        return results

    def is_pareto_improvement(self, new_result: Performance, existing_results: List[Performance], rtol=0.0) -> bool:
        """Check if new result represents a Pareto improvement using the configured metric."""
        return self.pareto_metric.is_improvement(new_result, existing_results, rtol)

    def analyze_pareto_gaps(self, new_result: Performance, existing_frontier: List[Performance]) -> Dict:
        """Analyze how close the new result is to breaking through the Pareto frontier using the configured metric."""
        return self.pareto_metric.analyze_gaps(new_result, existing_frontier)

    def get_pareto_frontier(self, results: List[Performance]) -> List[Performance]:
        """Compute the Pareto frontier from a list of performance results using the configured metric."""
        return self.pareto_metric.get_frontier(results)

    @staticmethod
    def get_improvement_prompt(original_prompt: str, previous_code: str, failure_analysis: Dict) -> str:
        """Generate a prompt for improving the optimizer based on failure analysis."""
        error_breakthrough_gap = failure_analysis.get('error_breakthrough_gap', 0)
        time_breakthrough_gap = failure_analysis.get('time_breakthrough_gap', 0)
        closest_breakthrough_dimension = failure_analysis.get('closest_breakthrough_dimension', 'none')
        closest_breakthrough_distance = failure_analysis.get('closest_breakthrough_distance', float('inf'))
        best_error = failure_analysis.get('best_error', float('inf'))
        best_time = failure_analysis.get('best_time', float('inf'))
        failure_mode = failure_analysis.get('failure_mode', 'no_pareto_improvement')
        
        if failure_mode == 'timeout':
            feedback = f"""The previous optimizer was too slow and timed out during benchmarking. 
This suggests the algorithm is computationally expensive or has convergence issues.

IMPROVEMENT FOCUS:
- Optimize for speed and efficiency
- Reduce computational complexity
- Consider early termination conditions
- Simplify the algorithm while maintaining effectiveness"""
        
        elif failure_mode == 'no_pareto_improvement':
            if closest_breakthrough_dimension == 'error':
                feedback = f"""The previous optimizer is closest to breaking through the Pareto frontier by improving accuracy:
- Error breakthrough gap: {error_breakthrough_gap:.3f} (need to improve by this amount)
- Time breakthrough gap: {time_breakthrough_gap:.3f}s
- Closest breakthrough distance: {closest_breakthrough_distance:.3f}

IMPROVEMENT FOCUS:
- Prioritize accuracy improvements - you're very close to a major breakthrough!
- Focus on better convergence criteria or more sophisticated optimization strategies
- Maintain current speed while improving error reduction
- Consider adaptive parameters that improve final accuracy"""
            
            elif closest_breakthrough_dimension == 'time':
                feedback = f"""The previous optimizer is closest to breaking through the Pareto frontier by improving speed:
- Time breakthrough gap: {time_breakthrough_gap:.3f}s (need to improve by this amount)
- Log-Relative error breakthrough gap: {error_breakthrough_gap:.3f}
- Closest breakthrough distance: {closest_breakthrough_distance:.3f}

IMPROVEMENT FOCUS:
- Prioritize speed improvements - you're very close to a major breakthrough!
- Look for computational optimizations and early stopping
- Consider adaptive parameters that reduce computation time
- Maintain current accuracy while improving efficiency"""
            
            elif closest_breakthrough_dimension == 'tie':
                feedback = f"""The previous optimizer is equally close to breaking through in both dimensions:
- Log-Relative error breakthrough gap: {error_breakthrough_gap:.3f}
- Time breakthrough gap: {time_breakthrough_gap:.3f}s
- Closest breakthrough distance: {closest_breakthrough_distance:.3f}

IMPROVEMENT FOCUS:
- Choose one dimension to focus on (either accuracy or speed)
- Consider which improvement would be easier to achieve
- Look for algorithmic changes that improve both dimensions simultaneously
- Focus on the dimension where you have the most room for improvement"""
            
            else:
                feedback = f"""The previous optimizer needs significant improvements in both dimensions:
- Log-Relative Error breakthrough gap: {error_breakthrough_gap:.3f}
- Time breakthrough gap: {time_breakthrough_gap:.3f}s

IMPROVEMENT FOCUS:
- Need fundamental algorithmic improvements
- Consider a different approach inspired by the emergent behavior
- Focus on both accuracy and efficiency simultaneously
- Look for novel optimization strategies"""
        
        else:
            feedback = f"""The previous optimizer failed with error: {failure_mode}

IMPROVEMENT FOCUS:
- Fix the specific error
- Ensure robust implementation
- Add proper error handling"""

        return f"""The previous optimizer did not meet the requirements. Please improve it based on this feedback:

ORIGINAL PROMPT:
{original_prompt}

PREVIOUS CODE:
```python
{previous_code}
```

FAILURE ANALYSIS:
{failure_analysis}

FEEDBACK:
{feedback}

Please create an improved version that addresses these specific issues. Focus on the improvement areas identified above."""

    @staticmethod
    def save_optimizer_code(dir: os.PathLike, raw_code: str, performance: Performance) -> None:
        code_path = os.path.join(dir, f"{performance.name}.py")
        with open(code_path, 'w') as f:
            f.write(raw_code)
        logger.info(f"✓ Optimizer saved as {code_path}")

    @staticmethod
    def save_optimizer_performance(fpath: os.PathLike, performance: Performance) -> None:
        """Save the optimizer performance data to CSV."""
        # Append to performance CSV
        with open(fpath, 'a', newline='') as csvfile:
            fieldnames = Performance.get_csv_fieldnames()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is empty
            if os.path.getsize(fpath) == 0:
                writer.writeheader()
            
            writer.writerow(performance.to_csv_row())
        logger.info(f"✓ Performance appended to {fpath}")

    def run_generation_cycle(self, inspiration:str, max_attempts: int = 5) -> bool:
        """Run a complete generation cycle, where success is defined as generating an optimizer which advances the Pareto frontier."""
        # Load emergent ideas

        # Load existing performance and compute Pareto frontier
        performant_results = self.load_performance_file(self.performance_file)
        existing_frontier = self.get_pareto_frontier(performant_results)
        logger.info(f"Testing against Pareto frontier with {len(existing_frontier)} points")

        # Store the original generation prompt for reference
        original_prompt = self.get_generation_prompt(inspiration)
        previous_code = ""
        failure_analysis: Dict = {}

        for attempt in range(max_attempts):
            logger.info(f"=== Attempt {attempt + 1}/{max_attempts} at pareto improvement ===")
            
            # Generate optimizer with feedback from previous attempts
            if attempt == 0:
                generation_prompt = original_prompt
            else:
                generation_prompt = self.get_improvement_prompt(original_prompt, previous_code, failure_analysis)

            messages = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=generation_prompt)
            ]

            # Validate the optimizer
            generate_start = time.time()
            success, final_func, final_code, error = self.generate_optimizer(messages=messages,
                                                                             original_prompt=generation_prompt,
                                                                             )
            generate_end = time.time()
            logger.info(f"Profiling generation {generate_end - generate_start:.3f}")

            if not success or final_func is None:
                logger.warning(f"Generation failed: {error}")
                failure_analysis = {
                    'failure_mode': 'validation_error',
                    'error': error
                }
                previous_code = final_code
                continue
            
            # Create a unique name for the optimizer
            optimizer_name = f"minimize_{to_camel_case(inspiration.split(':')[0])}_{attempt + 1}"
            
            # Benchmark the optimizer
            try:
                benchmark_start = time.time()
                performance = self.benchmark_new_optimizer(final_func, optimizer_name)
                logger.info(f"Profiling benchmark {time.time() - benchmark_start:.3f}")
                if not performance:
                    logger.warning("Benchmarking failed")
                    performance = Performance(
                        name=optimizer_name,
                        failure_mode='benchmark_failure',
                        error='Benchmarking returned no results'
                    )
                    self.save_optimizer_performance(self.all_performance_file, performance)
                    failure_analysis = {
                        'failure_mode': 'benchmark_failure',
                        'error': 'Benchmarking returned no results'
                    }
                    previous_code = final_code
                    continue
            except TimeoutError as e:
                logger.warning(f"Benchmarking timed out; skipping this optimizer: {str(e)}")
                performance = Performance(
                    name=optimizer_name,
                    failure_mode='timeout',
                    error=str(e)
                )
                self.save_optimizer_performance(self.all_performance_file, performance)
                failure_analysis = {
                    'failure_mode': 'timeout',
                    'error': str(e)
                }
                previous_code = final_code
                continue
            except Exception as e:
                logger.warning(f"Benchmarking failed with exception: {str(e)}")
                performance = Performance(
                    name=optimizer_name,
                    failure_mode='benchmark_exception',
                    error=str(e)
                )
                self.save_optimizer_performance(self.all_performance_file, performance)
                failure_analysis = {
                    'failure_mode': 'benchmark_exception',
                    'error': str(e)
                }
                previous_code = final_code
                continue
            
            if performance.is_successful():
                logger.info(f"Performance: log_rel_error={performance.log_rel_error:.3f}, time={performance.time_elapsed:.4f}s")
            else:
                logger.warning(f"Performance failed: {performance.failure_mode} - {performance.error}")

            # Check if it advances the Pareto frontier
            if self.is_pareto_improvement(performance, existing_frontier, rtol=self.pareto_rtol):
                logger.info("✓ Pareto frontier advancement detected!")
                self.save_optimizer_code(self.code_output_dir_performant, final_code, performance)
                self.save_optimizer_performance(self.performance_file, performance)
                self.save_optimizer_performance(self.all_performance_file, performance)
                return True
            else:
                self.save_optimizer_code(self.code_output_dir_all, final_code, performance)
                logger.info("✗ No Pareto frontier advancement")
                
                # Analyze gaps for next iteration
                gap_analysis = self.analyze_pareto_gaps(performance, existing_frontier)
                
                # Update performance object with failure analysis data
                performance.failure_mode = 'no_pareto_improvement'
                performance.closest_breakthrough_distance = gap_analysis.get('closest_breakthrough_distance')
                performance.error_breakthrough_gap = gap_analysis.get('error_breakthrough_gap')
                performance.time_breakthrough_gap = gap_analysis.get('time_breakthrough_gap')
                performance.best_error = gap_analysis.get('best_error')
                performance.best_time = gap_analysis.get('best_time')
                performance.needs_error_improvement = gap_analysis.get('needs_error_improvement')
                performance.needs_time_improvement = gap_analysis.get('needs_time_improvement')
                performance.closest_breakthrough_dimension = gap_analysis.get('closest_breakthrough_dimension')
                self.save_optimizer_performance(self.all_performance_file, performance)

                failure_analysis = {
                    'failure_mode': 'no_pareto_improvement',
                    'performance': performance,
                    **gap_analysis
                }
                previous_code = final_code
        
        logger.info(f"Failed to generate Pareto-improving optimizer after {max_attempts} attempts")
        return False


class BlendedOptimizerGenerator(OptimizerGenerator):
    def __init__(self, *args, n_blend_examples: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_blend_examples = n_blend_examples
        assert len(list(self.code_output_dir_performant.glob('*.py'))) > 0, f"No performant optimizers found in {self.code_output_dir_performant}"

    def get_generation_prompt(self, inspiration: str) -> str:
        """
        Generate a prompt for creating a new optimizer, blending inspiration from both the given emergent behavior and n existing performant optimizer code examples.
        """
        # Find up to n_blend_examples .py files in self.code_output_dir_performant
        code_files = list(self.code_output_dir_performant.glob('*.py'))
        if len(code_files) <= self.n_blend_examples:
            selected_files = code_files
        else:
            selected_files = random.sample(code_files, self.n_blend_examples)

        # Build the code examples section
        if selected_files:
            code_examples_section = "\n".join([
                f"# EXISTING OPTIMIZER EXAMPLE: {fname}\n" +
                f"""```python\n{code}\n```\n\n""" for fname, code in [(f.name, code) for f, code in zip(selected_files, [open(f, 'r').read() for f in selected_files])]
            ])
        else:
            code_examples_section = ""

        # Prompt for blending inspirations
        prompt = f"""
Create a novel nonconvex optimization algorithm inspired by combining ideas from two different types of inspiration:
- CONCEPTUAL INSPIRATION: {inspiration}
- CODE INSPIRATION: Each of the below code examples is a modern nonconvex optimization algorithm inspired by a different emergent system, and can be used for inspiration.

{code_examples_section}

For each code example, consider:
- What emergent system or principle does it represent?
- What are the key algorithmic dynamics and strengths?
- What are the limitations or areas for improvement?
- What key insights from your CONCEPTUAL INSPIRATION can be blended with methodological techniques from the code to create a novel optimizer inspired by a new form of emergent behavior?

There are an infinite number of ways that emergent behavior can solve complex tasks.  Your task is to critically analyze the emergent principles and algorithmic mechanisms in both the CONCEPTUAL INSPIRATION and the CODE INSPIRATION, and synthesize a single, conceptually coherent nonconvex optimizer that combines the best ideas, novel contributions, and shortcuts or synergies of all the inspiration you've been given.

{self.get_requirements_prompt()}

CRITICAL THINKING:
- For each source of inspiration, what key dynamics create emergent behavior which maximizes net benefit or minimizes net energy?
- How can the new inspiration and these existing approaches be blended or improved upon?
- What novel ideas or shortcuts can be introduced by combining these inspirations?
- How can the resulting algorithm be made efficient, robust, and effective for challenging optimization problems?

Create a complete, runnable Python function that implements your blended, novel algorithm.
"""
        return prompt


@click.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--api-base', default=None, help='OpenAI API base URL')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def main(api_key: str, api_base: str, model: str, n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int, log_level: str):
    """Generate new optimizers using LLMs."""
    setup_logging(log_level)
    generator = OptimizerGenerator(
        api_key=api_key,
        api_base=api_base,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims
    )
    ideas_file = "data/emergent_optimization_ideas.txt"
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        logger.warning("No emergent ideas found!")
        return False
    # Select random inspiration
    inspiration = random.choice(ideas)
    logger.info(f"Inspiration: {inspiration}")

    success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)
    
    if success:
        logger.info("🎉 Successfully generated a Pareto-improving optimizer!")
    else:
        logger.info("😞 Failed to generate a Pareto-improving optimizer")
    return success



@click.group()
def cli():
    pass

@cli.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--api-base', default=None, help='OpenAI API base URL')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def inspire(api_key: str, api_base: str, model: str, n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int, log_level: str):
    """Generate new optimizers using LLMs."""
    setup_logging(log_level)
    generator = OptimizerGenerator(
        api_key=api_key,
        api_base=api_base,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims
    )
    ideas_file = "data/emergent_optimization_ideas.txt"
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        logger.warning("No emergent ideas found!")
        return False
    # Select random inspiration
    inspiration = random.choice(ideas)
    logger.info(f"Inspiration: {inspiration}")

    success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)

    if success:
        logger.info("🎉 Successfully generated a Pareto-improving optimizer!")
    else:
        logger.info("😞 Failed to generate a Pareto-improving optimizer")
    return success


@cli.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--api-base', default=None, help='OpenAI API base URL')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--start-index', default=0, type=int, help='Index to start sweeping from')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--pareto-rtol', default=0.0, type=float, help='Relative tolerance for Pareto frontier')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
@click.option('--n-jobs', default=1, type=int, help='Number of jobs to use for tuning')
@click.option('--max-allowed-time-per-function', default=MAX_ALLOWED_PROBLEM_TIME, type=float, help='Maximum allowed time per function')
@click.option('--max-allowed-rolling-average-function-time', default=MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME, type=float, help='Maximum allowed rolling average function time')
@click.option('--output-dir', default="data/output", help='Output directory')
@click.option('--ideas-file', default="data/emergent_optimization_ideas.txt", help='File containing emergent optimization ideas')
@click.option('--pareto-metric', default='strict', type=click.Choice(['strict', 'convex_hull']), help='Pareto metric to use: strict dominance or convex hull')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def sweep(api_key: str, api_base: str, model: str, start_index: int , n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int, output_dir: str, ideas_file: str, pareto_rtol: float = 0.0, n_jobs: int = 1, max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME, max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME, pareto_metric: str = 'strict', log_level: str = 'INFO'):
    """Generate new optimizers using LLMs, sweeping through all inspirations."""
    setup_logging(log_level)

    # Select the appropriate Pareto metric
    if pareto_metric == 'convex_hull':
        metric = ConvexHullParetoMetric
    elif pareto_metric == 'strict':
        metric = StrictDominanceParetoMetric
    else:
        raise ValueError(f"Invalid Pareto metric: {pareto_metric}")
    
    generator = OptimizerGenerator(
        api_key=api_key,
        api_base=api_base,
        model_name=model,
        n_jobs=n_jobs,
        max_allowed_time_per_function=max_allowed_time_per_function,
        max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims,
        output_dir=output_dir,
        pareto_rtol=pareto_rtol,
        pareto_metric=metric
    )
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        logger.warning("No emergent ideas found!")
        return False
    for inspiration in ideas[start_index:]:
        logger.info(f"=== Sweeping with inspiration: {inspiration} ===")
        success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)

        if success:
            logger.info("🎉 Successfully generated a Pareto-improving optimizer!")
        else:
            logger.info("😞 Failed to generate a Pareto-improving optimizer")


@cli.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--api-base', default=None, help='OpenAI API base URL')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--start-index', default=0, type=int, help='Index to start sweeping from')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--pareto-rtol', default=0.0, type=float, help='Relative tolerance for Pareto frontier')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
@click.option('--n-jobs', default=1, type=int, help='Number of jobs to use for tuning')
@click.option('--max-allowed-time-per-function', default=MAX_ALLOWED_PROBLEM_TIME, type=float, help='Maximum allowed time per function')
@click.option('--max-allowed-rolling-average-function-time', default=MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME, type=float, help='Maximum allowed rolling average function time')
@click.option('--output-dir', default="data/output", help='Output directory')
@click.option('--ideas-file', default="data/emergent_optimization_ideas.txt", help='File containing emergent optimization ideas')
@click.option('--n-blend-examples', default=3, type=int, help='Number of code examples to blend for inspiration')
@click.option('--pareto-metric', default='strict', type=click.Choice(['strict', 'convex_hull']), help='Pareto metric to use: strict dominance or convex hull')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def blend_sweep(api_key: str, api_base: str, model: str, start_index: int , n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int, output_dir: str, ideas_file: str, pareto_rtol: float = 0.0, n_blend_examples: int = 3, n_jobs: int = 1, max_allowed_time_per_function: float = MAX_ALLOWED_PROBLEM_TIME, max_allowed_rolling_average_function_time: float = MAX_ALLOWED_ROLLING_AVERAGE_FUNCTION_TIME, pareto_metric: str = 'strict', log_level: str = 'INFO'):
    """Generate new optimizers using LLMs, sweeping through all inspirations and blending with existing performant optimizers."""
    setup_logging(log_level)
    
    # Select the appropriate Pareto metric
    if pareto_metric == 'convex_hull':
        metric = ConvexHullParetoMetric()
    else:
        metric = StrictDominanceParetoMetric()
    
    generator = BlendedOptimizerGenerator(
        api_key=api_key,
        api_base=api_base,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_jobs=n_jobs,
        max_allowed_time_per_function=max_allowed_time_per_function,
        max_allowed_rolling_average_function_time=max_allowed_rolling_average_function_time,
        n_dims=n_dims,
        output_dir=output_dir,
        pareto_rtol=pareto_rtol,
        pareto_metric=metric,
        n_blend_examples=n_blend_examples
    )
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        logger.warning("No emergent ideas found!")
        return False
    for inspiration in ideas[start_index:]:
        logger.info(f"=== Sweeping with inspiration: {inspiration} ===")
        success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)

        if success:
            logger.info("🎉 Successfully generated a Pareto-improving optimizer!")
        else:
            logger.info("😞 Failed to generate a Pareto-improving optimizer")



def reassess_pareto_candidates_from_files(
    input_fname: os.PathLike,
    benchmark_fname: os.PathLike,
    output_fname: os.PathLike,
    pareto_metric_class: type[ParetoMetric],
    recompute_pareto_frontier: bool = True,
    rtol: float = 0.3
) -> Tuple[int, int]:
    """
    Reassess Pareto candidates from all results and save performant ones to output file.
    
    Args:
        input_fname: Path to CSV file with all results
        benchmark_fname: Path to CSV file with existing performant results
        output_fname: Path to output CSV file for reassessed performant results
        pareto_metric_class: ParetoMetric class to use for assessment
        recompute_pareto_frontier: Re-compute the pareto frontier after each improvement
        rtol: Relative tolerance for Pareto frontier
        
    Returns:
        Tuple of (improvements_count, total_performant_count)
    """
    logger.info(f"Loading all results from {input_fname}")
    all_results = OptimizerGenerator.load_performance_file(input_fname)
    logger.info(f"Loaded {len(all_results)} total results")
    
    logger.info(f"Loading benchmark results from {benchmark_fname}")
    performant_results = OptimizerGenerator.load_performance_file(benchmark_fname)
    logger.info(f"Loaded {len(performant_results)} benchmark results")
    
    logger.info(f"Computing existing Pareto frontier")
    existing_frontier = pareto_metric_class.get_frontier(performant_results)
    logger.info(f"Existing frontier has {len(existing_frontier)} points")
    
    improvements_count = 0
    for idx, new_result in enumerate(all_results):
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(all_results)} results...")
        
        is_pareto_improvement = pareto_metric_class.is_improvement(new_result, existing_frontier, rtol)
        if is_pareto_improvement:
            performant_results.append(new_result)
            if recompute_pareto_frontier:
                existing_frontier = pareto_metric_class.get_frontier(performant_results)
            improvements_count += 1
            logger.info(f"✓ Found Pareto improvement: {new_result.name}")
    
    logger.info(f"Found {improvements_count} new Pareto improvements")
    logger.info(f"Total performant results: {len(performant_results)}")
    
    # Save performant_results to output_fname as CSV
    logger.info(f"Saving {len(performant_results)} performant results to {output_fname}")
    with open(output_fname, 'w', newline='') as csvfile:
        fieldnames = Performance.get_csv_fieldnames()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in performant_results:
            writer.writerow(result.to_csv_row())
    
    logger.info(f"✓ Successfully saved performant results to {output_fname}")
    
    return improvements_count, len(performant_results)


@cli.command()
@click.option('--input-fname', required=True, type=click.Path(exists=True), help='Input CSV file with all results')
@click.option('--benchmark-fname', required=True, type=click.Path(exists=True), help='Benchmark CSV file with existing performant results')
@click.option('--output-fname', required=True, type=click.Path(), help='Output CSV file for reassessed performant results')
@click.option('--pareto-metric', default='strict', type=click.Choice(['strict', 'convex_hull']), help='Pareto metric to use: strict dominance or convex hull')
@click.option('--rtol', default=0.3, type=float, help='Relative tolerance for Pareto frontier (default: 0.3)')
@click.option('--recompute-pareto-frontier', default=True, type=bool, help='Recompute the pareto frontier after each improvement (default: True)')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def reassess(input_fname: str, benchmark_fname: str, output_fname: str, pareto_metric: str, rtol: float, recompute_pareto_frontier: bool, log_level: str):
    """Reassess Pareto candidates from a single file and save performant ones to output file."""
    setup_logging(log_level)
    
    # Select the appropriate Pareto metric
    if pareto_metric == 'convex_hull':
        metric_class = ConvexHullParetoMetric
    elif pareto_metric == 'strict':
        metric_class = StrictDominanceParetoMetric
    else:
        raise ValueError(f"Invalid Pareto metric: {pareto_metric}")
    
    logger.info(f"Using {pareto_metric} Pareto metric with rtol={rtol}")
    
    improvements_count, total_count = reassess_pareto_candidates_from_files(
        input_fname=input_fname,
        benchmark_fname=benchmark_fname,
        output_fname=output_fname,
        pareto_metric_class=metric_class,
        recompute_pareto_frontier=recompute_pareto_frontier,
        rtol=rtol
    )
    
    logger.info(f"=== Reassessment Complete ===")
    logger.info(f"New improvements: {improvements_count}")
    logger.info(f"Total performant: {total_count}")


@cli.command(name='reassess-dir')
@click.option('--input-dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Input directory containing CSV files with all results')
@click.option('--benchmark-fname', required=True, type=click.Path(exists=True), help='Benchmark CSV file with existing performant results')
@click.option('--output-dir', required=True, type=click.Path(file_okay=False, dir_okay=True), help='Output directory for reassessed performant results')
@click.option('--pareto-metric', default='strict', type=click.Choice(['strict', 'convex_hull']), help='Pareto metric to use: strict dominance or convex hull')
@click.option('--rtol', default=0.3, type=float, help='Relative tolerance for Pareto frontier (default: 0.3)')
@click.option('--recompute-pareto-frontier', default=True, type=bool, help='Recompute the pareto frontier after each improvement (default: True)')
@click.option('--pattern', default='*.csv', help='Glob pattern for CSV files to process (default: *.csv)')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Logging level')
def reassess_dir_cli(input_dir: str, benchmark_fname: str, output_dir: str, pareto_metric: str, rtol: float, recompute_pareto_frontier: bool, pattern: str, log_level: str):
    """Reassess Pareto candidates from all CSV files in a directory and save results to output directory."""
    build_pareto_for_dir(input_dir=input_dir, benchmark_fname=benchmark_fname, output_dir=output_dir, pareto_metric=pareto_metric, rtol=rtol, recompute_pareto_frontier=recompute_pareto_frontier, pattern=pattern, log_level=log_level)


def build_pareto_for_dir(input_dir: str, benchmark_fname: str, output_dir: str, pareto_metric: str, rtol: float, recompute_pareto_frontier: bool, pattern: str, log_level: str):
    """Reassess Pareto candidates from all CSV files in a directory and save results to output directory."""
    setup_logging(log_level)
    
    # Select the appropriate Pareto metric
    if pareto_metric == 'convex_hull':
        metric_class = ConvexHullParetoMetric
    elif pareto_metric == 'strict':
        metric_class = StrictDominanceParetoMetric
    else:
        raise ValueError(f"Invalid Pareto metric: {pareto_metric}")
    
    logger.info(f"Using {pareto_metric} Pareto metric with rtol={rtol}")
    
    # Create output directory if it doesn't exist
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Find all CSV files in input directory
    input_path = pathlib.Path(input_dir)
    csv_files = list(input_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    total_improvements = 0
    processed_files = 0
    failed_files = []
    
    for csv_file in csv_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {csv_file.name}")
        logger.info(f"{'='*60}")
        
        output_fname = output_path / csv_file.name
        
        try:
            improvements_count, total_count = reassess_pareto_candidates_from_files(
                input_fname=csv_file,
                benchmark_fname=benchmark_fname,
                output_fname=output_fname,
                pareto_metric_class=metric_class,
                rtol=rtol,
                recompute_pareto_frontier=recompute_pareto_frontier
            )
            
            total_improvements += improvements_count
            processed_files += 1
            logger.info(f"✓ Completed {csv_file.name}: {improvements_count} improvements, {total_count} total performant")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {csv_file.name}: {str(e)}")
            failed_files.append(csv_file.name)
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Batch Reassessment Complete ===")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed successfully: {processed_files}/{len(csv_files)}")
    logger.info(f"Total new improvements: {total_improvements}")
    
    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)}): {', '.join(failed_files)}")
    else:
        logger.info("✓ All files processed successfully!")


if __name__ == "__main__":
    cli()