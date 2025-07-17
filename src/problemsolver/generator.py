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
    
    def load_emergent_ideas(self) -> List[str]:
        """Load emergent optimization ideas from the text file."""
        ideas_file = "data/emergent_optimization_ideas.txt"
        try:
            with open(ideas_file, 'r') as f:
                ideas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return ideas
        except FileNotFoundError:
            print(f"Warning: {ideas_file} not found. Using default ideas.")
            return [
                "Swarm intelligence in bird flocking",
                "Ant colony pheromone trails",
                "Bee colony foraging behavior",
                "Firefly synchronization",
                "Wolf pack hunting strategies",
                "Fish school movement patterns",
                "Slime mold network formation",
                "Neural network plasticity",
                "Evolutionary adaptation",
                "Quantum tunneling effects"
            ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are an expert researcher in nonconvex/nonlinear mathematical optimization techniques and an expert programmer in Python. 

Your task is to create novel optimization algorithms inspired by emergent behaviors in nature and complex systems. You should:

1. Think critically about how the given emergent behavior could inspire novel minimization techniques
2. Consider the mathematical principles underlying the behavior
3. Translate these principles into algorithmic components
4. Implement a working Python function that follows the specified signature
5. Ensure the code is robust, efficient, and handles edge cases
6. Your answer should only be code and any docstrings; no preamble or explanation is allowed. Assume that your answer will be directly executed; any non-code non-commented text will cause an error.

You must create a complete, runnable Python function that can be executed immediately. The function should be well-documented and follow Python best practices."""

    def get_generation_prompt(self, inspiration: str) -> str:
        """Generate the prompt for creating a new optimizer."""
        return f"""Create a novel optimization algorithm inspired by this emergent behavior:

INSPIRATION: {inspiration}

REQUIREMENTS:
1. Function signature must be: minimize(fun: Callable[[np.ndarray], float], initial_guess: np.ndarray, **kwargs) -> np.ndarray
2. Any hyperparameters which require tuning should be indicated using Annotated[type, Interval(...)] for hyperparameter optimization which will be handled by our downstream test function.
3. Must handle arbitrary dimensionality (at least 10 dimensions)
4. Must return a numpy array of the same shape as initial_guess
5. Must be pure Python with numpy (no scipy or other advanced libraries)
6. Must include proper error handling and edge cases

GOALS:
- Create a novel algorithm that combines ideas from existing metaheuristics with the inspiration
- Avoid simple exploration/exploitation or canonical first- and second-order methods. Aim to create something new and innovative, fully utilizing the inspiration from naturally occuring systems.
- Focus on both accuracy (finding good minima) and efficiency (reasonable computation time) tested in the following way:
  - Accuracy and efficiency will be evaluated by a downstream test script which takes functions of the standard signature.
  - Randomly generated functions will be used to tune the optimizer's hyperparameters with a standard hyperparameter tuner with a fixed budget.
  - Because of the fixed budget, we would recommend being judicious with the number of hyperparameters to tune.
  - The tuned optimizer will be tested on a set of test functions drawn from the same distribution.
  - Because the test functions are drawn randomly, you cannot attempt to overfit to the test metric.
- Consider how the emergent behavior's principles can be mathematically modeled
- Think about what makes this behavior effective in nature and how to translate that to optimization

EXAMPLE FUNCTION SIGNATURE:
```python
def minimize(
    fun: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    population_size: Annotated[int, Interval(low=20, high=200, step=10, log=False)] = 50,
    learning_rate: Annotated[float, Interval(low=0.01, high=1.0, step=0.01, log=True)] = 0.1,
    exploration_rate: Annotated[float, Interval(low=0.1, high=0.9, step=0.05, log=False)] = 0.5,
    max_iterations: int = 1000,
    seed: int = None
) -> np.ndarray:
    # Optimization algorithm implementation here
    return np.array([...])
```

CRITICAL THINKING:
Consider how {inspiration} relates to optimization:
- What mathematical principles underlie this behavior?
- How does this behavior achieve its goals efficiently?
- What aspects could be adapted for function minimization?
- How can we model the key mechanisms algorithmically?

Create a complete, runnable Python function that implements your novel algorithm."""

    def get_debug_prompt(self, original_prompt: str, code: str, error: str) -> str:
        """Generate a prompt for debugging the code."""
        return f"""The previous code had an error. Please fix it and return the corrected version.

ORIGINAL PROMPT:
{original_prompt}

GENERATED CODE:
```python
{code}
```

ERROR:
{error}

Please fix the error and return the corrected Python function. Ensure it follows all the original requirements."""

    def extract_func_and_code_from_response(self, response: str) -> tuple[Callable, str]:
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

        # Create a function from the code
        # Create a namespace with necessary imports
        namespace = {
            'np': np,
            'Annotated': Annotated,
            'Interval': Interval
        }
        exec(initial_code, namespace)
        optimizer_func = namespace['minimize']
        return optimizer_func, initial_code

    def validate_optimizer_code(self, optimizer_func: Callable, raw_code: str, original_prompt: str, max_iterations: int = 5) -> Tuple[bool, Optional[Callable], str, str]:
        """Validate the optimizer function through multiple iterations of debugging."""
        
        for iteration in range(max_iterations):
            try:
                check_optimizer_annotations(optimizer_func)
                check_optimizer_function(optimizer_func)
                return True, optimizer_func, raw_code, ""   # If we get here, the function is valid
                
            except Exception as e:
                error_msg = str(e)
                print(f"Iteration {iteration + 1}: Error - {error_msg}")
                
                if iteration < max_iterations - 1:
                    # Get debug prompt and regenerate
                    debug_prompt = self.get_debug_prompt(original_prompt, raw_code, error_msg)
                    messages = [
                        SystemMessage(content=self.get_system_prompt()),
                        HumanMessage(content=debug_prompt)
                    ]
                    
                    response = self.llm(messages)
                    optimizer_func, raw_code = self.extract_func_and_code_from_response(response.content)
        
        return False, None, raw_code, "Max iterations reached"

    def generate_optimizer(self, inspiration: str) -> Tuple[bool, Optional[Callable], str, str]:
        """Generate a new optimizer based on the inspiration."""
        print(f"Generating optimizer inspired by: {inspiration}")
        
        # Generate initial code
        generation_prompt = self.get_generation_prompt(inspiration)
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=generation_prompt)
        ]
        
        response = self.llm.invoke(messages)
        optimizer_func, raw_code = self.extract_func_and_code_from_response(response.content)

        # Validate and debug
        success, final_func, final_code, error = self.validate_optimizer_code(optimizer_func,
                                                                  raw_code=raw_code,
                                                                  original_prompt=generation_prompt)
        
        if success:
            print("✓ Optimizer code generated successfully")
        else:
            print(f"✗ Failed to generate valid optimizer: {error}")
        
        return success, final_func, final_code, error

    def benchmark_new_optimizer(self, optimizer_func: Callable, optimizer_name: str) -> Optional[Dict]:
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
                n_tuning_trials=self.n_tuning_trials
            )
            
            return {
                'name': optimizer_name,
                'log_rel_error': log_rel_error,
                'time_elapsed': time_elapsed,
                'best_params': best_params
            }
                
        except Exception as e:
            print(f"Benchmarking failed: {str(e)}")
            return None

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

    def is_pareto_improvement(self, new_result: Dict, existing_results: List[Dict]) -> bool:
        """Check if the new result advances the Pareto frontier."""
        new_error = new_result['log_rel_error']
        new_time = new_result['time_elapsed']
        
        # Check if this point dominates any existing point
        for existing in existing_results:
            if new_error <= existing['log_rel_error'] and new_time <= existing['time_elapsed']:
                if new_error < existing['log_rel_error'] or new_time < existing['time_elapsed']:
                    return True
        
        # Check if this point is not dominated by any existing point
        for existing in existing_results:
            if existing['log_rel_error'] <= new_error and existing['time_elapsed'] <= new_time:
                if existing['log_rel_error'] < new_error or existing['time_elapsed'] < new_time:
                    return False
        
        return True

    def save_optimizer_code(self, dir: os.PathLike, raw_code: str, performance: Dict) -> None:
        code_path = os.path.join(dir, f"{performance['name']}.py")
        with open(code_path, 'w') as f:
            f.write(raw_code)
        print(f"✓ Optimizer saved as {code_path}")


    def save_optimizer_performance(self, fpath, performance: Dict, ):
        """Save the optimizer code and performance."""
        # Append to performance CSV
        with open(fpath, 'a', newline='') as csvfile:
            fieldnames = ['method_name', 'log_rel_error', 'time_elapsed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is empty
            if os.path.getsize(self.all_performance_file) == 0:
                writer.writeheader()
            
            writer.writerow({
                'method_name': performance['name'],
                'log_rel_error': performance['log_rel_error'],
                'time_elapsed': performance['time_elapsed'],
            })
        print(f"✓ Performance appended to {self.all_performance_file}")

    def run_generation_cycle(self, inspiration:str, max_attempts: int = 5) -> bool:
        """Run a complete generation cycle, where success is defined as generating an optimizer which advances the Pareto frontier."""
        # Load emergent ideas

        # Load existing performance
        existing_performance = self.load_existing_performance()

        for attempt in range(max_attempts):
            print(f"\n=== Attempt {attempt + 1}/{max_attempts} at pareto improvement ===")
            
            # Generate optimizer
            success, optimizer_func, raw_code, error = self.generate_optimizer(inspiration)
            if not success or optimizer_func is None:
                print(f"Generation failed: {error}")
                continue
            
            # Create a unique name for the optimizer
            optimizer_name = f"minimize_{to_camel_case(inspiration.split(":")[0])}_{attempt + 1}"
            
            # Benchmark the optimizer
            try:
                performance = self.benchmark_new_optimizer(optimizer_func, optimizer_name)
                if not performance:
                    print("Benchmarking failed")
                    continue
            except TimeoutError as e:
                print(f"Benchmarking timed out; skipping this optimizer: {str(e)}")
                continue
            
            print(f"Performance: log_rel_error={performance['log_rel_error']:.3f}, time={performance['time_elapsed']:.2f}s")
            self.save_optimizer_performance(self.all_performance_file, performance)

            # Check if it advances the Pareto frontier
            if self.is_pareto_improvement(performance, existing_performance):
                print("✓ Pareto frontier advancement detected!")
                self.save_optimizer_code(self.code_output_dir_performant, raw_code, performance)
                self.save_optimizer_performance(self.performance_file, performance)
                return True
            else:
                self.save_optimizer_code(self.code_output_dir_all, raw_code, performance)
                print("✗ No Pareto frontier advancement")
        
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
    ideas = generator.load_emergent_ideas()
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

if __name__ == "__main__":
    main() 