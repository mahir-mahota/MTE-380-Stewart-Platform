import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import threading
from queue import Queue
import itertools


@dataclass
class OptimizationResult:
    """Store results from a single PID parameter test."""
    Kp: float
    Ki: float
    Kd: float
    settling_time: float
    overshoot: float
    steady_state_error: float
    rise_time: float
    total_error: float
    oscillation_count: int
    score: float
    trajectory: List[Tuple[float, float]]
    errors: List[float]
    timestamp: str


class PIDOptimizer:
    """Optimize PID parameters for the ball balancing system."""
    
    def __init__(self, initial_params: Dict[str, float] = None):
        """
        Initialize the PID optimizer.
        
        Args:
            initial_params: Starting PID parameters {'Kp': float, 'Ki': float, 'Kd': float}
        """
        self.initial_params = initial_params or {'Kp': 0.00016, 'Ki': 0.00001, 'Kd': 0.00013}
        self.results_history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        
        # Performance tracking
        self.current_trajectory = []
        self.current_errors = []
        self.test_start_time = None
        self.target_position = np.array([0.0, 0.0])
        
        # Test parameters
        self.test_duration = 10.0  # seconds per test
        self.settling_threshold = 5.0  # pixels
        self.overshoot_threshold = 1.2  # 20% overshoot
        
    def evaluate_performance(self, trajectory: List[Tuple[float, float]], 
                            errors: List[float], 
                            target: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the performance of a PID configuration.
        
        Args:
            trajectory: List of (x, y) positions over time
            errors: List of error magnitudes over time
            target: Target position
            
        Returns:
            Dictionary of performance metrics
        """
        if len(trajectory) < 10 or len(errors) < 10:
            return {
                'settling_time': float('inf'),
                'overshoot': 0.0,
                'steady_state_error': float('inf'),
                'rise_time': float('inf'),
                'oscillation_count': 0,
                'total_error': float('inf')
            }
        
        trajectory_np = np.array(trajectory)
        errors_np = np.array(errors)
        
        # Calculate settling time (time to reach and stay within threshold)
        settling_time = self._calculate_settling_time(errors_np, self.settling_threshold)
        
        # Calculate overshoot
        overshoot = self._calculate_overshoot(trajectory_np, target)
        
        # Calculate steady-state error (average error in last 20% of trajectory)
        steady_state_error = self._calculate_steady_state_error(errors_np)
        
        # Calculate rise time (time to reach 90% of target for first time)
        rise_time = self._calculate_rise_time(errors_np, target)
        
        # Count oscillations
        oscillation_count = self._count_oscillations(trajectory_np, target)
        
        # Total accumulated error
        total_error = np.sum(errors_np) / len(errors_np)
        
        return {
            'settling_time': settling_time,
            'overshoot': overshoot,
            'steady_state_error': steady_state_error,
            'rise_time': rise_time,
            'oscillation_count': oscillation_count,
            'total_error': total_error
        }
    
    def _calculate_settling_time(self, errors: np.ndarray, threshold: float) -> float:
        """Calculate time to settle within threshold."""
        dt = self.test_duration / len(errors)
        
        # Find last time error exceeded threshold
        settled_indices = np.where(errors < threshold)[0]
        if len(settled_indices) == 0:
            return float('inf')
        
        # Check if it stays settled
        first_settled = settled_indices[0]
        for i in range(first_settled, len(errors)):
            if errors[i] > threshold:
                first_settled = i + 1
        
        if first_settled >= len(errors):
            return float('inf')
            
        return first_settled * dt
    
    def _calculate_overshoot(self, trajectory: np.ndarray, target: np.ndarray) -> float:
        """Calculate maximum overshoot as percentage."""
        if len(trajectory) == 0:
            return 0.0
            
        target_dist = np.linalg.norm(target)
        if target_dist < 1e-6:  # Target is at origin
            max_dist = np.max(np.linalg.norm(trajectory, axis=1))
            return max_dist
        
        # Calculate overshoot along the direction to target
        direction = target / target_dist
        projections = np.dot(trajectory, direction)
        max_projection = np.max(projections)
        
        if max_projection > target_dist:
            return (max_projection - target_dist) / target_dist
        return 0.0
    
    def _calculate_steady_state_error(self, errors: np.ndarray) -> float:
        """Calculate average error in steady state (last 20% of data)."""
        if len(errors) < 5:
            return float('inf')
            
        steady_state_portion = int(0.2 * len(errors))
        if steady_state_portion < 1:
            steady_state_portion = 1
            
        return np.mean(errors[-steady_state_portion:])
    
    def _calculate_rise_time(self, errors: np.ndarray, target: np.ndarray) -> float:
        """Calculate time to first reach 90% of target."""
        dt = self.test_duration / len(errors)
        target_dist = np.linalg.norm(target)
        
        if target_dist < 1e-6:  # Target at origin
            # For origin target, rise time is time to get within 10% of settling threshold
            threshold = 0.1 * self.settling_threshold
            reached = np.where(errors < threshold)[0]
            if len(reached) > 0:
                return reached[0] * dt
            return float('inf')
        
        # Find when error first drops below 10% of target distance
        threshold = 0.1 * target_dist
        reached = np.where(errors < threshold)[0]
        if len(reached) > 0:
            return reached[0] * dt
        return float('inf')
    
    def _count_oscillations(self, trajectory: np.ndarray, target: np.ndarray) -> int:
        """Count number of oscillations around target."""
        if len(trajectory) < 3:
            return 0
            
        # Calculate signed distance from target along primary axis
        if np.linalg.norm(target) < 1e-6:
            # Use x-axis for origin target
            signed_distances = trajectory[:, 0]
        else:
            direction = target / np.linalg.norm(target)
            center_trajectory = trajectory - target
            signed_distances = np.dot(center_trajectory, direction)
        
        # Count zero crossings
        sign_changes = np.diff(np.sign(signed_distances))
        oscillations = np.sum(np.abs(sign_changes) > 0) // 2
        
        return int(oscillations)
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from performance metrics.
        Lower score is better.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Overall score
        """
        # Weighted scoring (lower is better)
        weights = {
            'settling_time': 2.0,
            'overshoot': 1.5,
            'steady_state_error': 3.0,
            'rise_time': 1.0,
            'oscillation_count': 1.0,
            'total_error': 2.0
        }
        
        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                # Handle infinite values
                if value == float('inf'):
                    score += weights[metric] * 1000
                else:
                    score += weights[metric] * value
        
        return score
    
    def grid_search(self, param_ranges: Dict[str, Tuple[float, float, int]], 
                   test_function) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            param_ranges: Dictionary with parameter ranges 
                         {'Kp': (min, max, steps), 'Ki': ..., 'Kd': ...}
            test_function: Function to test PID parameters, should accept (Kp, Ki, Kd)
                          and return (trajectory, errors)
        
        Returns:
            Best optimization result
        """
        # Generate grid
        kp_range = np.linspace(*param_ranges['Kp'])
        ki_range = np.linspace(*param_ranges['Ki'])
        kd_range = np.linspace(*param_ranges['Kd'])
        
        total_tests = len(kp_range) * len(ki_range) * len(kd_range)
        print(f"[OPTIMIZER] Starting grid search with {total_tests} combinations...")
        
        best_score = float('inf')
        best_params = None
        
        for i, (kp, ki, kd) in enumerate(itertools.product(kp_range, ki_range, kd_range)):
            print(f"[OPTIMIZER] Testing {i+1}/{total_tests}: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")
            
            # Test these parameters
            trajectory, errors = test_function(kp, ki, kd)
            
            # Evaluate performance
            metrics = self.evaluate_performance(trajectory, errors, self.target_position)
            score = self.calculate_score(metrics)
            
            # Create result
            result = OptimizationResult(
                Kp=kp, Ki=ki, Kd=kd,
                settling_time=metrics['settling_time'],
                overshoot=metrics['overshoot'],
                steady_state_error=metrics['steady_state_error'],
                rise_time=metrics['rise_time'],
                total_error=metrics['total_error'],
                oscillation_count=metrics['oscillation_count'],
                score=score,
                trajectory=trajectory,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
            
            self.results_history.append(result)
            
            if score < best_score:
                best_score = score
                best_params = (kp, ki, kd)
                self.best_result = result
                print(f"[OPTIMIZER] New best score: {score:.4f}")
        
        print(f"[OPTIMIZER] Best parameters: Kp={best_params[0]:.6f}, Ki={best_params[1]:.6f}, Kd={best_params[2]:.6f}")
        return self.best_result
    
    def random_search(self, param_ranges: Dict[str, Tuple[float, float]], 
                     n_iterations: int, test_function) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            param_ranges: Dictionary with parameter ranges {'Kp': (min, max), ...}
            n_iterations: Number of random combinations to test
            test_function: Function to test PID parameters
            
        Returns:
            Best optimization result
        """
        print(f"[OPTIMIZER] Starting random search with {n_iterations} iterations...")
        
        best_score = float('inf')
        best_params = None
        
        for i in range(n_iterations):
            # Generate random parameters
            kp = np.random.uniform(*param_ranges['Kp'])
            ki = np.random.uniform(*param_ranges['Ki'])
            kd = np.random.uniform(*param_ranges['Kd'])
            
            print(f"[OPTIMIZER] Testing {i+1}/{n_iterations}: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")
            
            # Test these parameters
            trajectory, errors = test_function(kp, ki, kd)
            
            # Evaluate performance
            metrics = self.evaluate_performance(trajectory, errors, self.target_position)
            score = self.calculate_score(metrics)
            
            # Create result
            result = OptimizationResult(
                Kp=kp, Ki=ki, Kd=kd,
                settling_time=metrics['settling_time'],
                overshoot=metrics['overshoot'],
                steady_state_error=metrics['steady_state_error'],
                rise_time=metrics['rise_time'],
                total_error=metrics['total_error'],
                oscillation_count=metrics['oscillation_count'],
                score=score,
                trajectory=trajectory,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
            
            self.results_history.append(result)
            
            if score < best_score:
                best_score = score
                best_params = (kp, ki, kd)
                self.best_result = result
                print(f"[OPTIMIZER] New best score: {score:.4f}")
        
        print(f"[OPTIMIZER] Best parameters: Kp={best_params[0]:.6f}, Ki={best_params[1]:.6f}, Kd={best_params[2]:.6f}")
        return self.best_result
    
    def adaptive_search(self, initial_ranges: Dict[str, Tuple[float, float]], 
                       test_function, max_iterations: int = 50) -> OptimizationResult:
        """
        Perform adaptive search that narrows the search space based on results.
        
        Args:
            initial_ranges: Initial parameter ranges
            test_function: Function to test PID parameters
            max_iterations: Maximum number of iterations
            
        Returns:
            Best optimization result
        """
        print(f"[OPTIMIZER] Starting adaptive search...")
        
        current_ranges = initial_ranges.copy()
        best_score = float('inf')
        best_params = self.initial_params.copy()
        
        for iteration in range(max_iterations):
            # Test 5 random points in current range
            candidates = []
            for _ in range(5):
                kp = np.random.uniform(*current_ranges['Kp'])
                ki = np.random.uniform(*current_ranges['Ki'])
                kd = np.random.uniform(*current_ranges['Kd'])
                
                trajectory, errors = test_function(kp, ki, kd)
                metrics = self.evaluate_performance(trajectory, errors, self.target_position)
                score = self.calculate_score(metrics)
                
                candidates.append((score, kp, ki, kd, trajectory, errors, metrics))
            
            # Sort by score
            candidates.sort(key=lambda x: x[0])
            
            # Update best if improved
            if candidates[0][0] < best_score:
                best_score = candidates[0][0]
                best_params = {'Kp': candidates[0][1], 'Ki': candidates[0][2], 'Kd': candidates[0][3]}
                
                # Create result
                self.best_result = OptimizationResult(
                    Kp=candidates[0][1], Ki=candidates[0][2], Kd=candidates[0][3],
                    settling_time=candidates[0][6]['settling_time'],
                    overshoot=candidates[0][6]['overshoot'],
                    steady_state_error=candidates[0][6]['steady_state_error'],
                    rise_time=candidates[0][6]['rise_time'],
                    total_error=candidates[0][6]['total_error'],
                    oscillation_count=candidates[0][6]['oscillation_count'],
                    score=candidates[0][0],
                    trajectory=candidates[0][4],
                    errors=candidates[0][5],
                    timestamp=datetime.now().isoformat()
                )
                
                self.results_history.append(self.best_result)
                print(f"[OPTIMIZER] Iteration {iteration+1}: New best score: {best_score:.4f}")
                print(f"           Kp={best_params['Kp']:.6f}, Ki={best_params['Ki']:.6f}, Kd={best_params['Kd']:.6f}")
            
            # Narrow search space around best candidates
            kp_values = [c[1] for c in candidates[:2]]
            ki_values = [c[2] for c in candidates[:2]]
            kd_values = [c[3] for c in candidates[:2]]
            
            kp_center = np.mean(kp_values)
            ki_center = np.mean(ki_values)
            kd_center = np.mean(kd_values)
            
            # Reduce range by 20% each iteration
            range_factor = 0.8 ** (iteration / 10)
            
            current_ranges['Kp'] = (
                max(initial_ranges['Kp'][0], kp_center - (initial_ranges['Kp'][1] - initial_ranges['Kp'][0]) * range_factor / 2),
                min(initial_ranges['Kp'][1], kp_center + (initial_ranges['Kp'][1] - initial_ranges['Kp'][0]) * range_factor / 2)
            )
            current_ranges['Ki'] = (
                max(initial_ranges['Ki'][0], ki_center - (initial_ranges['Ki'][1] - initial_ranges['Ki'][0]) * range_factor / 2),
                min(initial_ranges['Ki'][1], ki_center + (initial_ranges['Ki'][1] - initial_ranges['Ki'][0]) * range_factor / 2)
            )
            current_ranges['Kd'] = (
                max(initial_ranges['Kd'][0], kd_center - (initial_ranges['Kd'][1] - initial_ranges['Kd'][0]) * range_factor / 2),
                min(initial_ranges['Kd'][1], kd_center + (initial_ranges['Kd'][1] - initial_ranges['Kd'][0]) * range_factor / 2)
            )
        
        return self.best_result
    
    def save_results(self, filename: str = "pid_optimization_results.json"):
        """Save optimization results to file."""
        results_dict = {
            'best_result': {
                'Kp': self.best_result.Kp,
                'Ki': self.best_result.Ki,
                'Kd': self.best_result.Kd,
                'score': self.best_result.score,
                'settling_time': self.best_result.settling_time,
                'overshoot': self.best_result.overshoot,
                'steady_state_error': self.best_result.steady_state_error,
                'rise_time': self.best_result.rise_time,
                'oscillation_count': self.best_result.oscillation_count,
                'total_error': self.best_result.total_error
            } if self.best_result else None,
            'history': [
                {
                    'Kp': r.Kp, 'Ki': r.Ki, 'Kd': r.Kd,
                    'score': r.score,
                    'settling_time': r.settling_time,
                    'overshoot': r.overshoot,
                    'steady_state_error': r.steady_state_error,
                    'timestamp': r.timestamp
                }
                for r in self.results_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"[OPTIMIZER] Results saved to {filename}")
    
    def plot_results(self):
        """Visualize optimization results."""
        if not self.results_history:
            print("[OPTIMIZER] No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PID Optimization Results', fontsize=16)
        
        # Extract data
        scores = [r.score for r in self.results_history]
        kp_values = [r.Kp for r in self.results_history]
        ki_values = [r.Ki for r in self.results_history]
        kd_values = [r.Kd for r in self.results_history]
        settling_times = [r.settling_time for r in self.results_history]
        overshoots = [r.overshoot for r in self.results_history]
        
        # Score over iterations
        axes[0, 0].plot(scores, 'b-', alpha=0.6)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter evolution
        axes[0, 1].plot(kp_values, 'r-', label='Kp', alpha=0.6)
        axes[0, 1].plot(ki_values, 'g-', label='Ki', alpha=0.6)
        axes[0, 1].plot(kd_values, 'b-', label='Kd', alpha=0.6)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].set_title('Parameter Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Score vs Kp
        axes[0, 2].scatter(kp_values, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
        axes[0, 2].set_xlabel('Kp')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Score vs Kp')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Score vs Ki
        axes[1, 0].scatter(ki_values, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
        axes[1, 0].set_xlabel('Ki')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Score vs Ki')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Score vs Kd
        axes[1, 1].scatter(kd_values, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
        axes[1, 1].set_xlabel('Kd')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Score vs Kd')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance metrics
        axes[1, 2].scatter(settling_times, overshoots, alpha=0.6, c=scores, cmap='coolwarm')
        axes[1, 2].set_xlabel('Settling Time (s)')
        axes[1, 2].set_ylabel('Overshoot')
        axes[1, 2].set_title('Settling Time vs Overshoot')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add colorbar for the last plot
        cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        cbar.set_label('Score')
        
        # Highlight best result
        if self.best_result:
            best_idx = scores.index(self.best_result.score)
            axes[0, 0].plot(best_idx, self.best_result.score, 'r*', markersize=15, label='Best')
            axes[0, 0].legend()
        
        plt.tight_layout()
        plt.savefig('pid_optimization_plot.png', dpi=100)
        plt.show()
        print("[OPTIMIZER] Plot saved as pid_optimization_plot.png")
