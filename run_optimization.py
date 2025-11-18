"""
Standalone PID optimization script for the Stewart Platform ball balancer.
This script can run automated optimization without the GUI.
"""

import numpy as np
import json
import time
from datetime import datetime
from balancer_controller.Balancer import PositionBalancer
from PID import PID
from pid_optimizer import PIDOptimizer
import matplotlib.pyplot as plt


class SimulatedBallSystem:
    """
    Simulated ball dynamics for testing PID parameters without hardware.
    This provides a mathematical model of the ball on the platform.
    """
    
    def __init__(self, platform_radius=200, damping=0.95, noise_level=2.0):
        """
        Initialize simulated ball system.
        
        Args:
            platform_radius: Radius of the platform in pixels
            damping: Velocity damping factor (0-1)
            noise_level: Measurement noise standard deviation
        """
        self.platform_radius = platform_radius
        self.damping = damping
        self.noise_level = noise_level
        
        # Ball state
        self.position = np.array([50.0, 50.0])  # Start off-center
        self.velocity = np.array([0.0, 0.0])
        
        # Physics parameters
        self.gravity_constant = 500.0  # pixels/s^2 per radian of tilt
        
    def reset(self, initial_pos=None):
        """Reset ball to initial position."""
        if initial_pos is not None:
            self.position = np.array(initial_pos)
        else:
            # Random initial position
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, self.platform_radius * 0.5)
            self.position = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
        self.velocity = np.array([0.0, 0.0])
    
    def step(self, tilt_angles: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one time step of ball dynamics.
        
        Args:
            tilt_angles: Platform tilt angles [tilt_x, tilt_y] in radians
            dt: Time step in seconds
            
        Returns:
            Measured ball position with noise
        """
        # Calculate acceleration from tilt
        # Note: tilt_x affects y-acceleration, tilt_y affects x-acceleration
        acceleration = np.array([
            -self.gravity_constant * tilt_angles[1],  # x-acceleration from tilt_y
            self.gravity_constant * tilt_angles[0]     # y-acceleration from tilt_x
        ])
        
        # Update velocity with damping
        self.velocity = self.damping * self.velocity + acceleration * dt
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Boundary constraints (ball stays on platform)
        distance = np.linalg.norm(self.position)
        if distance > self.platform_radius:
            # Reflect off edge
            self.position = self.position * (self.platform_radius / distance)
            self.velocity = -0.5 * self.velocity  # Lose energy on collision
        
        # Add measurement noise
        measured_position = self.position + np.random.normal(0, self.noise_level, 2)
        
        return measured_position


def test_pid_with_simulation(kp: float, ki: float, kd: float, 
                            duration: float = 10.0,
                            target_pos: np.ndarray = np.array([0.0, 0.0]),
                            visualize: bool = False) -> tuple:
    """
    Test PID parameters using simulated ball dynamics.
    
    Args:
        kp, ki, kd: PID parameters
        duration: Test duration in seconds
        target_pos: Target position for the ball
        visualize: Whether to plot the trajectory
        
    Returns:
        Tuple of (trajectory, errors)
    """
    # Create simulated system
    sim = SimulatedBallSystem()
    sim.reset([80, 60])  # Start at specific position
    
    # Create balancer with test platform points
    platform_points = [
        np.array([100, 0, 0]),
        np.array([-50, 86.6, 0]),
        np.array([-50, -86.6, 0])
    ]
    balancer = PositionBalancer(platform_points)
    balancer.pid.Kp = kp
    balancer.pid.Ki = ki
    balancer.pid.Kd = kd
    balancer.desired_pos = target_pos
    
    # Run simulation
    trajectory = []
    errors = []
    dt = 0.05  # 20 Hz update rate
    steps = int(duration / dt)
    
    for i in range(steps):
        # Get current ball position
        measured_pos = sim.position.copy()
        
        # Calculate control output
        ball_3d = np.array([measured_pos[0], measured_pos[1], 0.0])
        servo_deltas = balancer.step(ball_3d, dt)
        
        # Convert servo heights to tilt angles (simplified)
        # This is a rough approximation - actual conversion depends on platform geometry
        tilt_angles = np.array([
            np.sum(servo_deltas) * 0.001,  # tilt_x from combined servo movement
            (servo_deltas[0] - servo_deltas[1]) * 0.001  # tilt_y from differential
        ])
        
        # Simulate physics
        sim.step(tilt_angles, dt)
        
        # Record data
        trajectory.append((sim.position[0], sim.position[1]))
        error = np.linalg.norm(target_pos - sim.position)
        errors.append(error)
    
    if visualize:
        plt.figure(figsize=(12, 5))
        
        # Plot trajectory
        plt.subplot(1, 2, 1)
        traj_array = np.array(trajectory)
        plt.plot(traj_array[:, 0], traj_array[:, 1], 'b-', alpha=0.6, label='Ball path')
        plt.plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=10, label='Start')
        plt.plot(traj_array[-1, 0], traj_array[-1, 1], 'ro', markersize=10, label='End')
        plt.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title(f'Ball Trajectory (Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Plot error over time
        plt.subplot(1, 2, 2)
        time_array = np.arange(len(errors)) * dt
        plt.plot(time_array, errors, 'r-', alpha=0.7)
        plt.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Settling threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (pixels)')
        plt.title('Position Error Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'simulation_kp{kp:.6f}_ki{ki:.6f}_kd{kd:.6f}.png')
        plt.show()
    
    return trajectory, errors


def run_optimization_suite():
    """Run comprehensive PID optimization using different methods."""
    
    print("="*60)
    print("PID OPTIMIZATION FOR STEWART PLATFORM BALL BALANCER")
    print("="*60)
    
    # Initialize optimizer with your current best values
    initial_params = {'Kp': 0.00016, 'Ki': 0.00001, 'Kd': 0.00013}
    optimizer = PIDOptimizer(initial_params)
    
    # Test current parameters first
    print("\n[1] Testing current parameters...")
    trajectory, errors = test_pid_with_simulation(
        initial_params['Kp'], 
        initial_params['Ki'], 
        initial_params['Kd'],
        visualize=True
    )
    
    metrics = optimizer.evaluate_performance(trajectory, errors, np.array([0, 0]))
    score = optimizer.calculate_score(metrics)
    print(f"Current parameters score: {score:.4f}")
    print(f"  Settling time: {metrics['settling_time']:.2f}s")
    print(f"  Overshoot: {metrics['overshoot']*100:.1f}%")
    print(f"  Steady-state error: {metrics['steady_state_error']:.2f} pixels")
    
    # Define search ranges
    param_ranges_grid = {
        'Kp': (0.00008, 0.00032, 5),  # 5 steps
        'Ki': (0.000005, 0.00002, 3),  # 3 steps  
        'Kd': (0.00006, 0.00026, 4)    # 4 steps
    }
    
    param_ranges_random = {
        'Kp': (0.00005, 0.0005),
        'Ki': (0.000001, 0.00005),
        'Kd': (0.00003, 0.0004)
    }
    
    # Choose optimization method
    print("\n" + "="*60)
    print("Choose optimization method:")
    print("1. Grid Search (systematic, slower)")
    print("2. Random Search (faster, may miss optimum)")
    print("3. Adaptive Search (intelligent, balanced)")
    print("4. Quick Test (5 random samples)")
    print("5. Skip optimization")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("\n[2] Running Grid Search Optimization...")
        best_result = optimizer.grid_search(param_ranges_grid, test_pid_with_simulation)
        
    elif choice == "2":
        print("\n[2] Running Random Search Optimization...")
        best_result = optimizer.random_search(param_ranges_random, 30, test_pid_with_simulation)
        
    elif choice == "3":
        print("\n[2] Running Adaptive Search Optimization...")
        best_result = optimizer.adaptive_search(param_ranges_random, test_pid_with_simulation, 25)
        
    elif choice == "4":
        print("\n[2] Running Quick Test...")
        best_result = optimizer.random_search(param_ranges_random, 5, test_pid_with_simulation)
        
    else:
        print("\n[INFO] Skipping optimization.")
        return
    
    # Display results
    if optimizer.best_result:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\nBest parameters found:")
        print(f"  Kp = {optimizer.best_result.Kp:.8f}")
        print(f"  Ki = {optimizer.best_result.Ki:.8f}")
        print(f"  Kd = {optimizer.best_result.Kd:.8f}")
        print(f"\nPerformance metrics:")
        print(f"  Overall Score: {optimizer.best_result.score:.4f}")
        print(f"  Settling Time: {optimizer.best_result.settling_time:.2f} seconds")
        print(f"  Overshoot: {optimizer.best_result.overshoot*100:.1f}%")
        print(f"  Steady-State Error: {optimizer.best_result.steady_state_error:.2f} pixels")
        print(f"  Rise Time: {optimizer.best_result.rise_time:.2f} seconds")
        print(f"  Oscillations: {optimizer.best_result.oscillation_count}")
        
        # Compare with initial
        print(f"\nImprovement over initial parameters:")
        initial_score = score
        improvement = ((initial_score - optimizer.best_result.score) / initial_score) * 100
        print(f"  Score improvement: {improvement:.1f}%")
        
        # Test and visualize best parameters
        print("\n[3] Visualizing best parameters...")
        test_pid_with_simulation(
            optimizer.best_result.Kp,
            optimizer.best_result.Ki,
            optimizer.best_result.Kd,
            visualize=True
        )
        
        # Save results
        optimizer.save_results("optimization_results.json")
        
        # Plot optimization history
        if len(optimizer.results_history) > 1:
            optimizer.plot_results()
        
        # Generate config update
        print("\n" + "="*60)
        print("RECOMMENDED CONFIG UPDATE")
        print("="*60)
        print("\nUpdate your Balancer.py with these values:")
        print(f"self.pid = PID(Kp={optimizer.best_result.Kp:.8f}, "
              f"Ki={optimizer.best_result.Ki:.8f}, "
              f"Kd={optimizer.best_result.Kd:.8f}, integral_limit=10000)")
        
        # Save to config file
        config_update = {
            "pid_parameters": {
                "Kp": optimizer.best_result.Kp,
                "Ki": optimizer.best_result.Ki,
                "Kd": optimizer.best_result.Kd,
                "integral_limit": 10000
            },
            "optimization_timestamp": datetime.now().isoformat(),
            "optimization_score": optimizer.best_result.score
        }
        
        with open("optimized_pid_config.json", "w") as f:
            json.dump(config_update, f, indent=2)
        
        print("\nOptimized parameters saved to: optimized_pid_config.json")


if __name__ == "__main__":
    try:
        run_optimization_suite()
    except KeyboardInterrupt:
        print("\n\n[INFO] Optimization interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
