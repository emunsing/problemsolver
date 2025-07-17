#!/usr/bin/env python3
"""
Optimizer Generator using Large Language Models

This module generates new optimization algorithms using LLMs, validates them,
benchmarks their performance, and checks if they advance the Pareto frontier.
"""

import os
import sys
import csv
import random
import pathlib
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

import click
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Annotated
from problemsolver.utils import check_optimizer_annotations, check_optimizer_function, Interval
from problemsolver.evaluator import benchmark_optimizer, generate_test_functions
import re
import unicodedata

def to_camel_case(text: str) -> str:
    # Convert text like "convert THIS_to–camelCASE!" to "ConvertThisToCamelCase"
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 2. Replace any sequence of non-alphanumeric characters with a single space
    text = re.sub(r'[^0-9A-Za-z]+', ' ', text)
    # 3. Split on whitespace, capitalize each word, and join
    parts = text.strip().split()
    return ''.join(word.capitalize() for word in parts)

class OptimizerGenerator:
    def __init__(self, openai_api_key: str, model_name: str = "o4-mini",
                 n_tune_functions: int = 10, n_test_functions: int = 20, 
                 n_tuning_trials: int = 100, n_dims: int = 5):
        """Initialize the optimizer generator."""
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
        )
        self.n_tune_functions = n_tune_functions
        self.n_test_functions = n_test_functions
        self.n_tuning_trials = n_tuning_trials
        self.n_dims = n_dims
        
        self.performance_file = "data/output/optimizer_performance.csv"
        self.all_performance_file = "data/output/all_optimizer_performance.csv"

        self.code_output_dir_all = pathlib.Path("data/output/code/other")
        self.code_output_dir_performant = pathlib.Path("data/output/code/performant")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.all_performance_file), exist_ok=True)
        os.makedirs(self.code_output_dir_all, exist_ok=True)
        os.makedirs(self.code_output_dir_performant, exist_ok=True)

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

    @staticmethod
    def get_generation_prompt(inspiration: str) -> str:
        """Generate the prompt for creating a new optimizer."""
        return f"""Create a novel optimization algorithm inspired by this emergent behavior:

# INSPIRATION: {inspiration}

# GOALS AND EVALUATION:
- Create a novel algorithm that combines ideas from existing metaheuristics with inspiration from the naturally occuring emergent behavior above.
- Avoid simple exploration/exploitation or canonical first- and second-order methods. Aim to create something new and innovative, fully utilizing the inspiration from naturally occuring systems.
- Focus on both accuracy (finding good minima) and efficiency (total computation time) tested in the following way:
  - Accuracy and efficiency will be evaluated by a downstream test script which takes functions of the standard signature.
  - Randomly generated functions will be used to tune the optimizer's hyperparameters with a standard hyperparameter tuner with a fixed budget.
  - The tuned optimizer will be tested on a set of test functions drawn from the same distribution.
  - Because the test functions are drawn randomly, you cannot attempt to overfit to the test metric.
  - Problems which take more than a time limit (several seconds) will be considered failures; consider this when designing your algorithm.
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
    # Optimization algorithm implementation here
```

CRITICAL THINKING:
Consider how {inspiration} relates to optimization:
- What mathematical principles underlie this behavior?
- How does this behavior achieve its goals efficiently?
- What aspects could be adapted for function minimization?
- How can we model the key mechanisms algorithmically?

Create a complete, runnable Python function that implements your novel algorithm."""

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
            if end != -1:
                initial_code = response[start:end].strip()
        else:
            # If no code block, try to find function definition
            lines = response.split('\n')
            code_lines = []
            in_function = False

            for line in lines:
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    code_lines.append(line)
                    continue
                if line.strip().startswith('def minimize('):
                    in_function = True
                if in_function:
                    code_lines.append(line)
                    if line.strip().endswith('return') or (line.strip().startswith('return') and 'return' in line):
                        break

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

        # Create a function from the code
        # Create a namespace with necessary imports
        namespace = {
            'np': np,
            'Annotated': Annotated,
            'Interval': Interval
        }
        exec(filtered_code, namespace)
        optimizer_func = namespace['minimize']
        return optimizer_func, filtered_code

    def validate_optimizer_code(self, optimizer_func: Callable, raw_code: str, original_prompt: str, max_iterations: int = 5) -> Tuple[bool, Optional[Callable], str, str]:
        """Validate the optimizer function through multiple iterations of debugging."""
        try:
            check_optimizer_annotations(optimizer_func)
        except ValueError as ve:
            if "No Annotated parameters with Interval" in str(ve):
                print(f"Annotation error; continuing: {str(ve)}")
            else:
                raise ve
        check_optimizer_function(optimizer_func)
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
                optimizer_func, raw_code = self.extract_func_and_code_from_response(response.content)

                # Validate and debug
                success, final_func, final_code, error_msg = self.validate_optimizer_code(optimizer_func,
                                                                          raw_code=raw_code,
                                                                          original_prompt=original_prompt)
            except Exception as e:
                error_msg = str(e)
                print(f"Iteration {iteration + 1} optimizer generation: Error - {error_msg}")
                debug_prompt = self.get_debug_prompt(original_prompt, raw_code, error_msg)
                messages = [
                    SystemMessage(content=self.get_system_prompt()),
                    HumanMessage(content=debug_prompt)
                ]
                continue

        if success:
            print("✓ Optimizer code generated successfully")
        else:
            print(f"✗ Failed to generate valid optimizer: {error_msg}")

        return success, final_func, final_code, error_msg

    def benchmark_new_optimizer(self, optimizer_func: Callable, optimizer_name: str) -> Optional[Dict]:
        """Benchmark the new optimizer and return performance metrics."""
        # Generate test functions
        tune_functions = generate_test_functions(n_samples=self.n_tune_functions, n_dims=self.n_dims)
        test_functions = generate_test_functions(n_samples=self.n_test_functions, n_dims=self.n_dims)

        # Run benchmark with the function
        log_rel_error, time_elapsed, best_params = benchmark_optimizer(
            optimizer=optimizer_func,
            test_functions=test_functions,
            tune_functions=tune_functions,
            n_tuning_trials=self.n_tuning_trials
        )

        return {
            'name': optimizer_name,
            'log_rel_error': log_rel_error,
            'time_elapsed': time_elapsed,
            'best_params': best_params
        }

    def load_existing_performance(self) -> List[Dict]:
        """Load existing performance data from CSV."""
        if not os.path.exists(self.performance_file):
            return []
        
        results = []
        with open(self.performance_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append({
                    'method_name': row['method_name'],
                    'log_rel_error': float(row['log_rel_error']),
                    'time_elapsed': float(row['time_elapsed'])
                })
        return results

    @staticmethod
    def is_pareto_improvement(new_result: Dict, existing_results: List[Dict], rtol=0.0) -> bool:
        # Pareto improvement: At least as good as existing in all metrics, and strictly better in at least one metric.

        metric_fields = ['log_rel_error', 'time_elapsed']
        for existing in existing_results:
            existing_bound = {k: v + rtol * np.abs(v) for k, v in existing.items() if k in metric_fields}
            if all(new_result[k] <= existing_bound[k] for k in metric_fields) and any(new_result[k] < existing_bound[k] for k in metric_fields):
                return True
        return False

    @staticmethod
    def analyze_pareto_gaps(new_result: Dict, existing_frontier: List[Dict]) -> Dict:
        """Analyze how close the new result is to breaking through the Pareto frontier.
        
        Instead of measuring gaps to the best points in each dimension, we identify
        which dimension is closest to allowing the point to advance the frontier.
        """
        if not existing_frontier:
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': float('inf'),
                'best_time': float('inf'),
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # Find the best values in each dimension for reference
        best_error = min(point['log_rel_error'] for point in existing_frontier)
        best_time = min(point['time_elapsed'] for point in existing_frontier)
        
        # For each frontier point, calculate how much improvement is needed in each dimension
        # to dominate that point
        error_improvements_needed = []
        time_improvements_needed = []
        
        for frontier_point in existing_frontier:
            # To dominate this frontier point, we need to be at least as good in both dimensions
            # and strictly better in at least one
            
            # Calculate how much we need to improve in each dimension to dominate this point
            error_improvement = max(0, new_result['log_rel_error'] - frontier_point['log_rel_error'])
            time_improvement = max(0, new_result['time_elapsed'] - frontier_point['time_elapsed'])
            
            # If we're already better in one dimension, we only need to improve the other
            if new_result['log_rel_error'] < frontier_point['log_rel_error']:
                # We're already better in error, so we only need to improve time
                time_improvements_needed.append(time_improvement)
            elif new_result['time_elapsed'] < frontier_point['time_elapsed']:
                # We're already better in time, so we only need to improve error
                error_improvements_needed.append(error_improvement)
            else:
                # We need to improve in at least one dimension to dominate
                error_improvements_needed.append(error_improvement)
                time_improvements_needed.append(time_improvement)
        
        # Find the minimum improvement needed in each dimension to break through
        min_error_improvement = min(error_improvements_needed) if error_improvements_needed else float('inf')
        min_time_improvement = min(time_improvements_needed) if time_improvements_needed else float('inf')
        
        # Determine which dimension is closest to breakthrough
        if min_error_improvement < min_time_improvement:
            closest_dimension = 'error'
            closest_breakthrough_distance = min_error_improvement
        elif min_time_improvement < min_error_improvement:
            closest_dimension = 'time'
            closest_breakthrough_distance = min_time_improvement
        else:
            closest_dimension = 'tie'
            closest_breakthrough_distance = min_error_improvement
        
        return {
            'closest_breakthrough_distance': closest_breakthrough_distance,
            'error_breakthrough_gap': min_error_improvement,
            'time_breakthrough_gap': min_time_improvement,
            'best_error': best_error,
            'best_time': best_time,
            'needs_error_improvement': min_error_improvement > 0,
            'needs_time_improvement': min_time_improvement > 0,
            'closest_breakthrough_dimension': closest_dimension
        }

    @staticmethod
    def get_pareto_frontier(results: List[Dict]) -> List[Dict]:
        """Compute the Pareto frontier from a list of performance results.
        
        A point is on the Pareto frontier if it is not dominated by any other point.
        A point dominates another if it is at least as good in all metrics and strictly better in at least one.
        """
        if not results:
            return []
        
        metric_fields = ['log_rel_error', 'time_elapsed']
        frontier = []
        
        for candidate in results:
            is_dominated = False
            
            # Check if this candidate is dominated by any existing frontier point
            for frontier_point in frontier:
                # Check if frontier_point dominates candidate
                if all(frontier_point[k] <= candidate[k] for k in metric_fields) and \
                   any(frontier_point[k] < candidate[k] for k in metric_fields):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove any existing frontier points that are dominated by this candidate
                frontier = [point for point in frontier if not (
                    all(candidate[k] <= point[k] for k in metric_fields) and 
                    any(candidate[k] < point[k] for k in metric_fields)
                )]
                frontier.append(candidate)
        
        return frontier

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
    def save_optimizer_code(dir: os.PathLike, raw_code: str, performance: Dict) -> None:
        code_path = os.path.join(dir, f"{performance['name']}.py")
        with open(code_path, 'w') as f:
            f.write(raw_code)
        print(f"✓ Optimizer saved as {code_path}")

    @staticmethod
    def save_optimizer_performance(fpath, performance: Dict, ):
        """Save the optimizer code and performance."""
        # Append to performance CSV
        with open(fpath, 'a', newline='') as csvfile:
            fieldnames = ['method_name', 'log_rel_error', 'time_elapsed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is empty
            if os.path.getsize(fpath) == 0:
                writer.writeheader()
            
            writer.writerow({
                'method_name': performance['name'],
                'log_rel_error': performance['log_rel_error'],
                'time_elapsed': performance['time_elapsed'],
            })
        print(f"✓ Performance appended to {fpath}")

    def run_generation_cycle(self, inspiration:str, max_attempts: int = 5) -> bool:
        """Run a complete generation cycle, where success is defined as generating an optimizer which advances the Pareto frontier."""
        # Load emergent ideas

        # Load existing performance and compute Pareto frontier
        all_existing_performance = self.load_existing_performance()
        existing_frontier = self.get_pareto_frontier(all_existing_performance)
        print(f"Testing against Pareto frontier with {len(existing_frontier)} points")

        # Store the original generation prompt for reference
        original_prompt = self.get_generation_prompt(inspiration)
        previous_code = ""
        failure_analysis: Dict = {}

        for attempt in range(max_attempts):
            print(f"\n=== Attempt {attempt + 1}/{max_attempts} at pareto improvement ===")
            
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
            success, final_func, final_code, error = self.generate_optimizer(messages=messages,
                                                                             original_prompt=generation_prompt,
                                                                             )

            if not success or final_func is None:
                print(f"Generation failed: {error}")
                failure_analysis = {
                    'failure_mode': 'validation_error',
                    'error': error
                }
                previous_code = final_code
                continue
            
            # Create a unique name for the optimizer
            optimizer_name = f"minimize_{to_camel_case(inspiration.split(":")[0])}_{attempt + 1}"
            
            # Benchmark the optimizer
            try:
                performance = self.benchmark_new_optimizer(final_func, optimizer_name)
                if not performance:
                    print("Benchmarking failed")
                    failure_analysis = {
                        'failure_mode': 'benchmark_failure',
                        'error': 'Benchmarking returned no results'
                    }
                    previous_code = final_code
                    continue
            except TimeoutError as e:
                print(f"Benchmarking timed out; skipping this optimizer: {str(e)}")
                failure_analysis = {
                    'failure_mode': 'timeout',
                    'error': str(e)
                }
                previous_code = final_code
                continue
            except Exception as e:
                print(f"Benchmarking failed with exception: {str(e)}")
                failure_analysis = {
                    'failure_mode': 'benchmark_exception',
                    'error': str(e)
                }
                previous_code = final_code
                continue
            
            print(f"Performance: log_rel_error={performance['log_rel_error']:.3f}, time={performance['time_elapsed']:.4f}s")
            self.save_optimizer_performance(self.all_performance_file, performance)

            # Check if it advances the Pareto frontier
            if self.is_pareto_improvement(performance, existing_frontier):
                print("✓ Pareto frontier advancement detected!")
                self.save_optimizer_code(self.code_output_dir_performant, final_code, performance)
                self.save_optimizer_performance(self.performance_file, performance)
                return True
            else:
                self.save_optimizer_code(self.code_output_dir_all, final_code, performance)
                print("✗ No Pareto frontier advancement")
                
                # Analyze gaps for next iteration
                gap_analysis = self.analyze_pareto_gaps(performance, existing_frontier)
                failure_analysis = {
                    'failure_mode': 'no_pareto_improvement',
                    'performance': performance,
                    **gap_analysis
                }
                previous_code = final_code
        
        print(f"Failed to generate Pareto-improving optimizer after {max_attempts} attempts")
        return False


@click.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
def main(api_key: str, model: str, n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int):
    """Generate new optimizers using LLMs."""
    generator = OptimizerGenerator(
        openai_api_key=api_key,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims
    )
    ideas_file = "data/emergent_optimization_ideas.txt"
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        print("No emergent ideas found!")
        return False
    # Select random inspiration
    inspiration = random.choice(ideas)
    print(f"Inspiration: {inspiration}")

    success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)
    
    if success:
        print("\n🎉 Successfully generated a Pareto-improving optimizer!")
    else:
        print("\n😞 Failed to generate a Pareto-improving optimizer")
    return success



@click.group()
def cli():
    pass

@cli.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
def inspire(api_key: str, model: str, n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int):
    """Generate new optimizers using LLMs."""
    generator = OptimizerGenerator(
        openai_api_key=api_key,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims
    )
    ideas_file = "data/emergent_optimization_ideas.txt"
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        print("No emergent ideas found!")
        return False
    # Select random inspiration
    inspiration = random.choice(ideas)
    print(f"Inspiration: {inspiration}")

    success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)

    if success:
        print("\n🎉 Successfully generated a Pareto-improving optimizer!")
    else:
        print("\n😞 Failed to generate a Pareto-improving optimizer")
    return success


@cli.command()
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--model', default='o4-mini', help='OpenAI model to use')
@click.option('--start-index', default=0, type=int, help='Index to start sweeping from')
@click.option('--n-pareto-attempts', default=5, type=int, help='Number of attempts at pareto improvement')
@click.option('--n-tune-functions', default=10, type=int, help='Number of functions for tuning')
@click.option('--n-test-functions', default=20, type=int, help='Number of functions for testing')
@click.option('--n-tuning-trials', default=100, type=int, help='Number of tuning trials')
@click.option('--n-dims', default=5, type=int, help='Number of dimensions for test functions')
def sweep(api_key: str, model: str, start_index: int , n_pareto_attempts: int, n_tune_functions: int,
         n_test_functions: int, n_tuning_trials: int, n_dims: int):
    """Generate new optimizers using LLMs, sweeping through all inspirations."""
    generator = OptimizerGenerator(
        openai_api_key=api_key,
        model_name=model,
        n_tune_functions=n_tune_functions,
        n_test_functions=n_test_functions,
        n_tuning_trials=n_tuning_trials,
        n_dims=n_dims
    )
    ideas_file = "data/emergent_optimization_ideas.txt"
    ideas = generator.load_emergent_ideas(ideas_file)
    if not ideas:
        print("No emergent ideas found!")
        return False
    for inspiration in ideas[start_index:]:
        print(f"\n=== Sweeping with inspiration: {inspiration} ===")
        success = generator.run_generation_cycle(inspiration=inspiration, max_attempts=n_pareto_attempts)

        if success:
            print("\n🎉 Successfully generated a Pareto-improving optimizer!")
        else:
            print("\n😞 Failed to generate a Pareto-improving optimizer")

if __name__ == "__main__":
    cli()