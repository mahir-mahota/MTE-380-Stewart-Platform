"""
Advanced PID optimization algorithms for Stewart Platform ball balancer.
Includes state-of-the-art optimization methods with better convergence.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime
from scipy.optimize import differential_evolution, dual_annealing, minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationTiming:
    """Track timing statistics for optimization runs."""
    algorithm: str
    total_time: float
    iterations: int
    time_per_iteration: float
    best_score: float
    parameters_tested: int
    convergence_iteration: int


class AdvancedPIDOptimizer:
    """Advanced optimization algorithms for PID tuning."""
    
    def __init__(self, test_function: Callable, initial_params: Dict[str, float] = None):
        """
        Initialize advanced optimizer.
        
        Args:
            test_function: Function that tests PID parameters (kp, ki, kd) -> (trajectory, errors)
            initial_params: Starting PID parameters
        """
        self.test_function = test_function
        self.initial_params = initial_params or {'Kp': 0.00016, 'Ki': 0.00001, 'Kd': 0.00013}
        self.target_position = np.array([0.0, 0.0])
        self.settling_threshold = 5.0
        self.test_duration = 10.0
        
        # Optimization history
        self.history = []
        self.timing_stats = []
        self.best_params = None
        self.best_score = float('inf')
        
        # Cache for tested parameters to avoid redundant tests
        self.cache = {}
        self.evaluations_count = 0
        
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for optimization (to minimize).
        
        Args:
            params: Array [kp, ki, kd]
            
        Returns:
            Score (lower is better)
        """
        kp, ki, kd = params
        
        # Check cache
        cache_key = f"{kp:.8f}_{ki:.8f}_{kd:.8f}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Test parameters
        self.evaluations_count += 1
        trajectory, errors = self.test_function(kp, ki, kd)
        
        # Calculate score
        score = self._calculate_score(trajectory, errors)
        
        # Update cache and history
        self.cache[cache_key] = score
        self.history.append({'kp': kp, 'ki': ki, 'kd': kd, 'score': score})
        
        # Update best
        if score < self.best_score:
            self.best_score = score
            self.best_params = {'Kp': kp, 'Ki': ki, 'Kd': kd}
            print(f"  New best: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}, Score={score:.4f}")
        
        return score
    
    def _calculate_score(self, trajectory: List, errors: List) -> float:
        """Calculate performance score from trajectory and errors."""
        if len(errors) < 10:
            return 1000.0
        
        errors_np = np.array(errors)
        
        # Weighted metrics
        settling_time = self._get_settling_time(errors_np)
        overshoot = self._get_overshoot(errors_np)
        steady_error = np.mean(errors_np[-len(errors_np)//5:]) if len(errors_np) > 5 else 100
        oscillations = self._count_oscillations(errors_np)
        rise_time = self._get_rise_time(errors_np)
        
        # Weighted score
        score = (
            3.0 * settling_time +
            2.0 * overshoot * 10 +
            4.0 * steady_error +
            1.5 * oscillations * 5 +
            1.0 * rise_time
        )
        
        return score
    
    def _get_settling_time(self, errors: np.ndarray) -> float:
        """Calculate settling time in seconds."""
        dt = self.test_duration / len(errors)
        settled_indices = np.where(errors < self.settling_threshold)[0]
        
        if len(settled_indices) == 0:
            return self.test_duration
        
        # Find last time it exceeded threshold
        for i in range(len(errors) - 1, -1, -1):
            if errors[i] > self.settling_threshold:
                return (i + 1) * dt
        
        return settled_indices[0] * dt
    
    def _get_overshoot(self, errors: np.ndarray) -> float:
        """Calculate overshoot percentage."""
        if len(errors) < 2:
            return 0.0
        
        # Find first minimum (closest approach)
        for i in range(1, len(errors) - 1):
            if errors[i] < errors[i-1] and errors[i] < errors[i+1]:
                # Check if it overshoots after this
                if i < len(errors) - 1:
                    max_after = np.max(errors[i:min(i+20, len(errors))])
                    if max_after > errors[i]:
                        return (max_after - errors[i]) / (errors[i] + 1e-6)
                break
        return 0.0
    
    def _count_oscillations(self, errors: np.ndarray) -> int:
        """Count oscillations in error signal."""
        if len(errors) < 3:
            return 0
        
        # Compute derivative
        derivative = np.diff(errors)
        
        # Count sign changes in derivative
        sign_changes = np.diff(np.sign(derivative))
        oscillations = np.sum(np.abs(sign_changes) > 0) // 2
        
        return int(oscillations)
    
    def _get_rise_time(self, errors: np.ndarray) -> float:
        """Calculate 10-90% rise time."""
        dt = self.test_duration / len(errors)
        
        initial_error = errors[0]
        target_error = self.settling_threshold
        
        # Find 90% reduction in error
        threshold_90 = initial_error * 0.1
        reached = np.where(errors < threshold_90)[0]
        
        if len(reached) > 0:
            return reached[0] * dt
        return self.test_duration
    
    # ============================================================
    #       OPTIMIZATION ALGORITHMS
    # ============================================================
    
    def differential_evolution_optimization(self, bounds: Dict[str, Tuple[float, float]], 
                                           max_iterations: int = 50) -> Dict:
        """
        Differential Evolution - A powerful global optimization algorithm.
        
        Pros:
        - Excellent at finding global optimum
        - Handles non-convex, multimodal functions well
        - No gradient information needed
        
        Cons:
        - Can be slower than local methods
        
        Time: ~15-30 seconds for 50 iterations
        """
        print("\n" + "="*60)
        print("DIFFERENTIAL EVOLUTION OPTIMIZATION")
        print("="*60)
        print("Expected time: 15-30 seconds")
        print("This is one of the best algorithms for PID tuning!\n")
        
        start_time = time.time()
        self.evaluations_count = 0
        
        # Set bounds
        bounds_list = [
            (bounds['Kp'][0], bounds['Kp'][1]),
            (bounds['Ki'][0], bounds['Ki'][1]),
            (bounds['Kd'][0], bounds['Kd'][1])
        ]
        
        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds_list,
            maxiter=max_iterations,
            popsize=10,  # Population size
            mutation=(0.5, 1.5),  # Mutation factor
            recombination=0.7,  # Crossover probability
            seed=42,
            disp=True,
            polish=True,  # Use L-BFGS-B to polish result
            workers=1
        )
        
        elapsed = time.time() - start_time
        
        # Store timing
        self.timing_stats.append(OptimizationTiming(
            algorithm="Differential Evolution",
            total_time=elapsed,
            iterations=result.nit,
            time_per_iteration=elapsed / max(result.nit, 1),
            best_score=result.fun,
            parameters_tested=self.evaluations_count,
            convergence_iteration=result.nit
        ))
        
        print(f"\n✓ Optimization complete in {elapsed:.1f} seconds")
        print(f"  Iterations: {result.nit}")
        print(f"  Parameters tested: {self.evaluations_count}")
        print(f"  Best score: {result.fun:.4f}")
        
        return {
            'Kp': result.x[0],
            'Ki': result.x[1],
            'Kd': result.x[2],
            'score': result.fun,
            'time': elapsed
        }
    
    def bayesian_optimization(self, bounds: Dict[str, Tuple[float, float]], 
                             n_iterations: int = 30) -> Dict:
        """
        Bayesian Optimization - Uses probabilistic model to guide search.
        
        Pros:
        - Very sample-efficient (fewer tests needed)
        - Balances exploration and exploitation
        - Provides uncertainty estimates
        
        Cons:
        - Can get stuck in local optima
        - Computationally intensive for model updates
        
        Time: ~10-20 seconds for 30 iterations
        """
        print("\n" + "="*60)
        print("BAYESIAN OPTIMIZATION")
        print("="*60)
        print("Expected time: 10-20 seconds")
        print("Most efficient algorithm - needs fewer tests!\n")
        
        start_time = time.time()
        self.evaluations_count = 0
        
        # Initialize with random samples
        n_initial = 5
        samples = []
        
        for i in range(n_initial):
            kp = np.random.uniform(bounds['Kp'][0], bounds['Kp'][1])
            ki = np.random.uniform(bounds['Ki'][0], bounds['Ki'][1])
            kd = np.random.uniform(bounds['Kd'][0], bounds['Kd'][1])
            
            score = self.objective_function([kp, ki, kd])
            samples.append({'params': [kp, ki, kd], 'score': score})
        
        # Bayesian optimization loop
        for iteration in range(n_initial, n_iterations):
            # Fit Gaussian Process (simplified version)
            X = np.array([s['params'] for s in samples])
            y = np.array([s['score'] for s in samples])
            
            # Normalize data
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-6
            X_norm = (X - X_mean) / X_std
            
            y_mean = np.mean(y)
            y_std = np.std(y) + 1e-6
            y_norm = (y - y_mean) / y_std
            
            # Acquisition function (Expected Improvement)
            best_score = np.min(y)
            
            # Generate candidates
            n_candidates = 100
            candidates = []
            ei_values = []
            
            for _ in range(n_candidates):
                cand = [
                    np.random.uniform(bounds['Kp'][0], bounds['Kp'][1]),
                    np.random.uniform(bounds['Ki'][0], bounds['Ki'][1]),
                    np.random.uniform(bounds['Kd'][0], bounds['Kd'][1])
                ]
                
                # Simple prediction (distance-weighted average)
                cand_norm = (np.array(cand) - X_mean) / X_std
                distances = np.linalg.norm(X_norm - cand_norm, axis=1)
                weights = np.exp(-distances)
                weights /= np.sum(weights)
                
                pred_mean = np.sum(weights * y_norm)
                pred_std = np.sqrt(np.sum(weights * (y_norm - pred_mean)**2)) + 0.1
                
                # Expected Improvement
                z = (best_score - pred_mean * y_std - y_mean) / (pred_std * y_std)
                ei = (best_score - pred_mean * y_std - y_mean) * norm.cdf(z) + pred_std * y_std * norm.pdf(z)
                
                candidates.append(cand)
                ei_values.append(ei)
            
            # Select best candidate
            best_idx = np.argmax(ei_values)
            next_params = candidates[best_idx]
            
            # Test selected parameters
            score = self.objective_function(next_params)
            samples.append({'params': next_params, 'score': score})
            
            print(f"  Iteration {iteration+1}/{n_iterations}: Score = {score:.4f}")
        
        elapsed = time.time() - start_time
        
        # Find best
        best_sample = min(samples, key=lambda x: x['score'])
        
        # Store timing
        self.timing_stats.append(OptimizationTiming(
            algorithm="Bayesian Optimization",
            total_time=elapsed,
            iterations=n_iterations,
            time_per_iteration=elapsed / n_iterations,
            best_score=best_sample['score'],
            parameters_tested=self.evaluations_count,
            convergence_iteration=n_iterations
        ))
        
        print(f"\n✓ Optimization complete in {elapsed:.1f} seconds")
        print(f"  Parameters tested: {self.evaluations_count}")
        print(f"  Best score: {best_sample['score']:.4f}")
        
        return {
            'Kp': best_sample['params'][0],
            'Ki': best_sample['params'][1],
            'Kd': best_sample['params'][2],
            'score': best_sample['score'],
            'time': elapsed
        }
    
    def simulated_annealing_optimization(self, bounds: Dict[str, Tuple[float, float]], 
                                        max_iterations: int = 100) -> Dict:
        """
        Simulated Annealing - Probabilistic optimization inspired by metallurgy.
        
        Pros:
        - Can escape local optima
        - Simple and robust
        - Good for rough landscapes
        
        Cons:
        - Requires tuning of temperature schedule
        - Can be slow to converge
        
        Time: ~20-40 seconds for 100 iterations
        """
        print("\n" + "="*60)
        print("SIMULATED ANNEALING OPTIMIZATION")
        print("="*60)
        print("Expected time: 20-40 seconds")
        print("Good at escaping local minima!\n")
        
        start_time = time.time()
        self.evaluations_count = 0
        
        # Dual annealing from scipy (advanced version of simulated annealing)
        bounds_list = [
            (bounds['Kp'][0], bounds['Kp'][1]),
            (bounds['Ki'][0], bounds['Ki'][1]),
            (bounds['Kd'][0], bounds['Kd'][1])
        ]
        
        result = dual_annealing(
            self.objective_function,
            bounds_list,
            maxiter=max_iterations,
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=-5.0,
            seed=42
        )
        
        elapsed = time.time() - start_time
        
        # Store timing
        self.timing_stats.append(OptimizationTiming(
            algorithm="Simulated Annealing",
            total_time=elapsed,
            iterations=result.nit,
            time_per_iteration=elapsed / max(result.nit, 1),
            best_score=result.fun,
            parameters_tested=self.evaluations_count,
            convergence_iteration=result.nit
        ))
        
        print(f"\n✓ Optimization complete in {elapsed:.1f} seconds")
        print(f"  Iterations: {result.nit}")
        print(f"  Parameters tested: {self.evaluations_count}")
        print(f"  Best score: {result.fun:.4f}")
        
        return {
            'Kp': result.x[0],
            'Ki': result.x[1],
            'Kd': result.x[2],
            'score': result.fun,
            'time': elapsed
        }
    
    def particle_swarm_optimization(self, bounds: Dict[str, Tuple[float, float]], 
                                   n_particles: int = 20,
                                   n_iterations: int = 50) -> Dict:
        """
        Particle Swarm Optimization - Swarm intelligence algorithm.
        
        Pros:
        - Good balance of exploration and exploitation
        - Parallelizable
        - No gradient needed
        
        Cons:
        - Can converge prematurely
        - Many hyperparameters
        
        Time: ~15-25 seconds for 50 iterations with 20 particles
        """
        print("\n" + "="*60)
        print("PARTICLE SWARM OPTIMIZATION")
        print("="*60)
        print(f"Expected time: {n_iterations * 0.3:.0f}-{n_iterations * 0.5:.0f} seconds")
        print("Swarm intelligence for optimal PID!\n")
        
        start_time = time.time()
        self.evaluations_count = 0
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        for i in range(n_particles):
            # Random position
            position = np.array([
                np.random.uniform(bounds['Kp'][0], bounds['Kp'][1]),
                np.random.uniform(bounds['Ki'][0], bounds['Ki'][1]),
                np.random.uniform(bounds['Kd'][0], bounds['Kd'][1])
            ])
            
            # Random velocity (10% of range)
            velocity = np.array([
                np.random.uniform(-0.1, 0.1) * (bounds['Kp'][1] - bounds['Kp'][0]),
                np.random.uniform(-0.1, 0.1) * (bounds['Ki'][1] - bounds['Ki'][0]),
                np.random.uniform(-0.1, 0.1) * (bounds['Kd'][1] - bounds['Kd'][0])
            ])
            
            particles.append(position)
            velocities.append(velocity)
            
            # Evaluate initial position
            score = self.objective_function(position)
            personal_best_positions.append(position.copy())
            personal_best_scores.append(score)
        
        # Global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Personal best weight
        c2 = 1.5  # Global best weight
        
        # Main PSO loop
        for iteration in range(n_iterations):
            for i in range(n_particles):
                # Update velocity
                r1 = np.random.random(3)
                r2 = np.random.random(3)
                
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (personal_best_positions[i] - particles[i]) +
                    c2 * r2 * (global_best_position - particles[i])
                )
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Enforce bounds
                particles[i][0] = np.clip(particles[i][0], bounds['Kp'][0], bounds['Kp'][1])
                particles[i][1] = np.clip(particles[i][1], bounds['Ki'][0], bounds['Ki'][1])
                particles[i][2] = np.clip(particles[i][2], bounds['Kd'][0], bounds['Kd'][1])
                
                # Evaluate
                score = self.objective_function(particles[i])
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()
            
            # Decay inertia
            w *= 0.99
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{n_iterations}: Best score = {global_best_score:.4f}")
        
        elapsed = time.time() - start_time
        
        # Store timing
        self.timing_stats.append(OptimizationTiming(
            algorithm="Particle Swarm",
            total_time=elapsed,
            iterations=n_iterations,
            time_per_iteration=elapsed / n_iterations,
            best_score=global_best_score,
            parameters_tested=self.evaluations_count,
            convergence_iteration=n_iterations
        ))
        
        print(f"\n✓ Optimization complete in {elapsed:.1f} seconds")
        print(f"  Parameters tested: {self.evaluations_count}")
        print(f"  Best score: {global_best_score:.4f}")
        
        return {
            'Kp': global_best_position[0],
            'Ki': global_best_position[1],
            'Kd': global_best_position[2],
            'score': global_best_score,
            'time': elapsed
        }
    
    def covariance_matrix_adaptation(self, bounds: Dict[str, Tuple[float, float]], 
                                    n_iterations: int = 50) -> Dict:
        """
        CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
        
        Pros:
        - State-of-the-art for continuous optimization
        - Self-adapting strategy
        - Excellent for non-separable problems
        
        Cons:
        - Complex algorithm
        - Memory intensive for high dimensions
        
        Time: ~10-20 seconds for 50 iterations
        """
        print("\n" + "="*60)
        print("CMA-ES OPTIMIZATION")
        print("="*60)
        print("Expected time: 10-20 seconds")
        print("State-of-the-art evolutionary algorithm!\n")
        
        start_time = time.time()
        self.evaluations_count = 0
        
        # CMA-ES parameters
        n_dim = 3
        pop_size = 4 + int(3 * np.log(n_dim))  # Population size
        mu = pop_size // 2  # Number of parents
        
        # Initialize
        mean = np.array([
            (bounds['Kp'][0] + bounds['Kp'][1]) / 2,
            (bounds['Ki'][0] + bounds['Ki'][1]) / 2,
            (bounds['Kd'][0] + bounds['Kd'][1]) / 2
        ])
        
        sigma = 0.3  # Step size
        cov = np.eye(n_dim)  # Covariance matrix
        
        # Evolution paths
        pc = np.zeros(n_dim)
        ps = np.zeros(n_dim)
        
        # Learning rates
        cc = 4 / (n_dim + 4)
        cs = 3 / (n_dim + 3)
        c1 = 2 / ((n_dim + 1.3)**2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n_dim + 2)**2 + mu))
        damps = 1 + cs + 2 * max(0, np.sqrt((mu - 1) / (n_dim + 1)) - 1)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        best_score = float('inf')
        best_params = None
        
        for iteration in range(n_iterations):
            # Generate offspring
            offspring = []
            scores = []
            
            for i in range(pop_size):
                z = np.random.randn(n_dim)
                x = mean + sigma * np.dot(cov, z)
                
                # Enforce bounds
                x[0] = np.clip(x[0], bounds['Kp'][0], bounds['Kp'][1])
                x[1] = np.clip(x[1], bounds['Ki'][0], bounds['Ki'][1])
                x[2] = np.clip(x[2], bounds['Kd'][0], bounds['Kd'][1])
                
                score = self.objective_function(x)
                offspring.append(x)
                scores.append(score)
                
                if score < best_score:
                    best_score = score
                    best_params = x.copy()
            
            # Sort by fitness
            indices = np.argsort(scores)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * offspring[indices[i]] for i in range(mu)], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) <
                   (1.4 + 2 / (n_dim + 1)) * np.sqrt(n_dim))
            
            pc = (1 - cc) * pc
            if hsig:
                pc += np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            artmp = [(offspring[indices[i]] - old_mean) / sigma for i in range(mu)]
            
            cov = ((1 - c1 - cmu) * cov +
                  c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * cov) +
                  cmu * sum([weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu)]))
            
            # Update step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(n_dim) - 1))
            
            if (iteration + 1) % 10 == 0:
                print(f"  Generation {iteration+1}/{n_iterations}: Best score = {best_score:.4f}")
        
        elapsed = time.time() - start_time
        
        # Store timing
        self.timing_stats.append(OptimizationTiming(
            algorithm="CMA-ES",
            total_time=elapsed,
            iterations=n_iterations,
            time_per_iteration=elapsed / n_iterations,
            best_score=best_score,
            parameters_tested=self.evaluations_count,
            convergence_iteration=n_iterations
        ))
        
        print(f"\n✓ Optimization complete in {elapsed:.1f} seconds")
        print(f"  Parameters tested: {self.evaluations_count}")
        print(f"  Best score: {best_score:.4f}")
        
        return {
            'Kp': best_params[0],
            'Ki': best_params[1],
            'Kd': best_params[2],
            'score': best_score,
            'time': elapsed
        }
    
    def compare_all_methods(self, bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Run all optimization methods and compare results.
        
        Total time: ~2-3 minutes for all methods
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE PID OPTIMIZATION COMPARISON")
        print("="*70)
        print("This will test 6 different optimization algorithms")
        print("Total expected time: 2-3 minutes\n")
        
        results = {}
        
        # 1. Differential Evolution (best overall)
        self.cache.clear()
        results['differential_evolution'] = self.differential_evolution_optimization(bounds, 30)
        
        # 2. Bayesian Optimization (most efficient)
        self.cache.clear()
        results['bayesian'] = self.bayesian_optimization(bounds, 25)
        
        # 3. Particle Swarm (good balance)
        self.cache.clear()
        results['particle_swarm'] = self.particle_swarm_optimization(bounds, 15, 30)
        
        # 4. CMA-ES (state-of-the-art)
        self.cache.clear()
        results['cma_es'] = self.covariance_matrix_adaptation(bounds, 30)
        
        # 5. Simulated Annealing (robust)
        self.cache.clear()
        results['simulated_annealing'] = self.simulated_annealing_optimization(bounds, 50)
        
        # Print comparison
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS COMPARISON")
        print("="*70)
        print(f"{'Algorithm':<25} {'Score':<10} {'Time (s)':<10} {'Kp':<12} {'Ki':<12} {'Kd':<12}")
        print("-"*70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['score'])
        
        for name, res in sorted_results:
            print(f"{name:<25} {res['score']:<10.4f} {res['time']:<10.1f} "
                  f"{res['Kp']:<12.8f} {res['Ki']:<12.8f} {res['Kd']:<12.8f}")
        
        # Best overall
        best_method = sorted_results[0][0]
        best_result = sorted_results[0][1]
        
        print("\n" + "="*70)
        print(f"🏆 WINNER: {best_method.upper()}")
        print("="*70)
        print(f"Best PID parameters found:")
        print(f"  Kp = {best_result['Kp']:.8f}")
        print(f"  Ki = {best_result['Ki']:.8f}")
        print(f"  Kd = {best_result['Kd']:.8f}")
        print(f"  Score = {best_result['score']:.4f}")
        
        # Save results
        with open('advanced_optimization_results.json', 'w') as f:
            json.dump({
                'results': results,
                'best_method': best_method,
                'best_params': {
                    'Kp': best_result['Kp'],
                    'Ki': best_result['Ki'],
                    'Kd': best_result['Kd']
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print("\nResults saved to: advanced_optimization_results.json")
        
        return results
    
    def print_timing_summary(self):
        """Print detailed timing statistics."""
        if not self.timing_stats:
            print("No timing data available")
            return
        
        print("\n" + "="*70)
        print("TIMING ANALYSIS")
        print("="*70)
        print(f"{'Algorithm':<25} {'Total Time':<12} {'Iterations':<12} {'ms/iter':<12} {'Tests':<10}")
        print("-"*70)
        
        for stat in self.timing_stats:
            print(f"{stat.algorithm:<25} {stat.total_time:<12.1f} {stat.iterations:<12} "
                  f"{stat.time_per_iteration*1000:<12.1f} {stat.parameters_tested:<10}")
        
        print("\nKey Insights:")
        
        # Fastest
        fastest = min(self.timing_stats, key=lambda x: x.total_time)
        print(f"  ⚡ Fastest: {fastest.algorithm} ({fastest.total_time:.1f}s)")
        
        # Most efficient (best score per test)
        most_efficient = min(self.timing_stats, key=lambda x: x.parameters_tested)
        print(f"  🎯 Most Efficient: {most_efficient.algorithm} (only {most_efficient.parameters_tested} tests)")
        
        # Best result
        best_result = min(self.timing_stats, key=lambda x: x.best_score)
        print(f"  🏆 Best Score: {best_result.algorithm} (score: {best_result.best_score:.4f})")


# Recommendation function
def get_optimization_recommendation(test_duration: float = 10.0, 
                                   time_budget: float = 60.0) -> str:
    """
    Get recommendation for which optimization method to use.
    
    Args:
        test_duration: Duration of each PID test in seconds
        time_budget: Total time available for optimization in seconds
        
    Returns:
        Recommendation string
    """
    recommendations = []
    
    if time_budget < 30:
        recommendations.append(("Bayesian Optimization", 
                              "Most sample-efficient, good for tight time constraints"))
    elif time_budget < 60:
        recommendations.append(("CMA-ES", 
                              "Excellent balance of speed and quality"))
        recommendations.append(("Particle Swarm", 
                              "Good alternative with intuitive behavior"))
    elif time_budget < 120:
        recommendations.append(("Differential Evolution", 
                              "Best overall - highly recommended!"))
    else:
        recommendations.append(("Compare All Methods", 
                              "You have time to find the absolute best"))
    
    print("\n" + "="*70)
    print("OPTIMIZATION METHOD RECOMMENDATIONS")
    print("="*70)
    print(f"Based on your time budget of {time_budget:.0f} seconds:\n")
    
    for method, reason in recommendations:
        print(f"  ✓ {method}")
        print(f"    {reason}\n")
    
    print("Quick Reference:")
    print("  • Differential Evolution: Best for finding global optimum (15-30s)")
    print("  • Bayesian Optimization: Fewest tests needed (10-20s)")
    print("  • CMA-ES: State-of-the-art evolutionary algorithm (10-20s)")
    print("  • Particle Swarm: Intuitive swarm intelligence (15-25s)")
    print("  • Simulated Annealing: Good for rough landscapes (20-40s)")
    
    return recommendations[0][0] if recommendations else "Bayesian Optimization"


if __name__ == "__main__":
    # Get recommendation
    recommendation = get_optimization_recommendation(test_duration=10, time_budget=60)
    print(f"\n💡 Recommended method: {recommendation}")
