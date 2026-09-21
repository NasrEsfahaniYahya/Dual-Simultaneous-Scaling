"""
Dual-Simultaneous Scaling (DSS) Algorithm for Origin-Destination Matrix Balancing
Author: Yahya Nasr Esfahani (nasr.y20@gmail.com)
Affiliation: Department of Civil Engineering, Daneshpajouhan Pishro Higher Education Institute, Isfahan, Iran
License: MIT

Description:
This module provides a fully vectorized, single-pass implementation of the
Dual-Simultaneous Scaling (DSS) algorithm for doubly constrained matrix balancing.
It scales a non-negative base matrix T_0 to satisfy target row sums (O) and column sums (D).
"""

import numpy as np
import time

def dual_simultaneous_scaling(T_base, O_target, D_target, tol=1e-5, max_iter=1000):
    """
    Solves the doubly constrained matrix balancing problem using Dual-Simultaneous Scaling.

    Parameters:
    -----------
    T_base : numpy.ndarray (I x J)
        Non-negative base matrix.
    O_target : numpy.ndarray (I,)
        Target row totals (Trip productions).
    D_target : numpy.ndarray (J,)
        Target column totals (Trip attractions).
    tol : float, optional
        Convergence tolerance on maximum relative marginal error (default: 1e-5).
    max_iter : int, optional
        Maximum number of iterations allowed (default: 1000).

    Returns:
    --------
    T_balanced : numpy.ndarray (I x J)
        Balanced matrix satisfying row and column sum constraints.
    iterations : int
        Number of iterations executed until convergence.
    elapsed_time_ms : float
        Execution time in milliseconds.
    """
    start_time = time.perf_counter()
    
    T = np.array(T_base, dtype=np.float64, copy=True)
    O = np.array(O_target, dtype=np.float64)
    D = np.array(D_target, dtype=np.float64)
    
    target_sum = np.sum(O)
    
    for k in range(1, max_iter + 1):
        O_curr = np.sum(T, axis=1)
        D_curr = np.sum(T, axis=0)
        
        # Prevent division by zero
        O_curr[O_curr == 0] = 1e-12
        D_curr[D_curr == 0] = 1e-12
        
        F_i = O / O_curr
        F_j = D / D_curr
        
        curr_total = np.sum(T)
        T_factor = curr_total / target_sum if curr_total > 0 else 1.0
        
        # Unified outer-product vector scaling
        scale_matrix = np.outer(F_i, F_j) / T_factor
        T *= scale_matrix
        
        # Relative errors
        err_o = np.max(np.abs(np.sum(T, axis=1) - O) / O)
        err_d = np.max(np.abs(np.sum(T, axis=0) - D) / D)
        
        if max(err_o, err_d) < tol:
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
            return T, k, elapsed_time_ms

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    return T, max_iter, elapsed_time_ms

if __name__ == "__main__":
    print("=== Testing DSS Algorithm on 55trafic.csv Benchmark ===")
    T_base = np.array([[0, 5, 10], [10, 0, 20], [15, 20, 0]], dtype=float)
    O_target = np.array([20, 40, 50], dtype=float)
    D_target = np.array([45, 20, 45], dtype=float)

    T_balanced, iters, execution_time = dual_simultaneous_scaling(T_base, O_target, D_target, tol=1e-4)
    
    print(f"Iterations: {iters}")
    print(f"Time: {execution_time:.3f} ms")
    print("Balanced Matrix:")
    print(np.round(T_balanced, 2))
    print(f"Row Sums: {np.round(T_balanced.sum(axis=1), 2)} (Target: {O_target})")
    print(f"Col Sums: {np.round(T_balanced.sum(axis=0), 2)} (Target: {D_target})")
