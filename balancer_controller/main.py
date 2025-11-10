from Balancer import Balancer

import numpy as np

# Testing the class
def test_class():
    balancer = Balancer(platform_points=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)])
    dt = 0.1
    positions = [np.array([0.0, 0.0, 0.0]),
                 np.array([0.1, 0.0, 0.0]),
                 np.array([0.2, 0.1, 0.0]),
                 np.array([0.3, 0.1, 0.1])]
    for pos in positions:
        deltas = balancer.step(pos, dt)
        print(f"Position: {pos}, Platform Adjustments: {deltas}")
        print(balancer.pos, balancer.vel, balancer.accel)

def test_class_missing_meas():
    balancer = Balancer(platform_points=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)])
    dt = 0.1   
    positions = [np.array([0.0, 0.0, 0.0]),
                 np.array([0.2, 0.1, 0.0]),
                 np.array([0.3, 0.1, 0.1]),
                 None, None]
    for pos in positions:
        deltas = balancer.step(pos, dt)
        print(f"Position: {pos}, Platform Adjustments: {deltas}")
        print(balancer.pos, balancer.vel, balancer.accel)

if __name__ == "__main__":
    test_class()
    print("\nTesting with missing measurements:")
    test_class_missing_meas()