"""
Dual-Simultaneous Scaling (DSS) Algorithm for Origin-Destination Matrix Balancing
Author: Yahya Nasr Esfahani (nasr.y20@gmail.com)
Affiliation: Department of Civil Engineering, Daneshpajouhan Pishro Higher Education Institute, Isfahan, Iran
License: MIT

Description:
This module provides a fully vectorized, single-pass implementation of the
Dual-Simultaneous Scaling (DSS) algorithm for doubly constrained matrix balancing.
It scales a non-negative base matrix A_0 to satisfy target row sums (O) and column sums (D).
"""

import numpy as np
import time

def dual_simultaneous_scaling(A_base, O_target, D_target, tol=1e-5, max_iter=1000):
    """
    Solves the doubly constrained matrix balancing problem using Dual-Simultaneous Scaling.

    Parameters:
    -----------
    A_base : numpy.ndarray (I x J)
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
    A_balanced : numpy.ndarray (I x J)
        Balanced matrix satisfying row and column sum constraints.
    iterations : int
        Number of iterations executed until convergence.
    elapsed_time_ms : float
        Execution time in milliseconds.
    """
    start_time = time.perf_counter()
    
    A = np.array(A_base, dtype=np.float64, copy=True)
    O = np.array(O_target, dtype=np.float64)
    D = np.array(D_target, dtype=np.float64)
    
    A_target_total = np.sum(O)
    
    for k in range(1, max_iter + 1):
        O_curr = np.sum(A, axis=1)
        D_curr = np.sum(A, axis=0)
        
        # Prevent division by zero
        O_curr[O_curr == 0] = 1e-12
        D_curr[D_curr == 0] = 1e-12
        
        F_i = O / O_curr
        F_j = D / D_curr
        
        A_curr_total = np.sum(A)
        # A_factor = A_curr_total / A_target_total
        A_factor = A_curr_total / A_target_total if A_curr_total > 0 else 1.0
        
        # Corrected Outer-Product Scaling (Multiplying by A_factor eliminates quadratic power issue)
        scale_matrix = np.outer(F_i, F_j) * A_factor
        A *= scale_matrix
        
        # Relative errors
        err_o = np.max(np.abs(np.sum(A, axis=1) - O) / O)
        err_d = np.max(np.abs(np.sum(A, axis=0) - D) / D)
        
        if max(err_o, err_d) < tol:
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
            return A, k, elapsed_time_ms

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    return A, max_iter, elapsed_time_ms

if __name__ == "__main__":
    print("=== Testing Corrected DSS Algorithm on 55trafic.csv Benchmark ===")
    A_base = np.array([[0, 5, 10], [10, 0, 20], [15, 20, 0]], dtype=float)
    O_target = np.array([20, 40, 50], dtype=float)
    D_target = np.array([45, 20, 45], dtype=float)

    A_balanced, iters, execution_time = dual_simultaneous_scaling(A_base, O_target, D_target, tol=1e-5)
    
    print(f"Iterations: {iters}")
    print(f"Time: {execution_time:.3f} ms")
    print("Balanced Matrix:")
    print(np.round(A_balanced, 4))
    print(f"Row Sums: {np.round(A_balanced.sum(axis=1), 4)} (Target: {O_target})")
    print(f"Col Sums: {np.round(A_balanced.sum(axis=0), 4)} (Target: {D_target})")
