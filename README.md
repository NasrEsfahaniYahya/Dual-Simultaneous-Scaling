# Dual-Simultaneous Scaling (DSS) Algorithm for Large-Scale OD Matrix Balancing

Official repository for **"Dual-Simultaneous Scaling Algorithm for Large-Scale Origin-Destination Matrix Balancing: Mathematical Foundations, Vectorization, Sensitivity Analysis, and Cross-Disciplinary Applications"**.

---

## 📌 Abstract & Overview

The **Dual-Simultaneous Scaling (DSS)** algorithm is a fully vectorized, single-pass matrix equilibration solver designed for doubly constrained Origin-Destination (OD) matrix balancing in urban travel demand forecasting, RAS input-output economics, survey raking, and optimal transport (Sinkhorn distance).

Unlike traditional sequential implementations of Iterative Proportional Fitting (IPF / Furness) that require dual memory sweeps per iteration, the DSS algorithm updates row and column scale factors simultaneously using a dynamic global system-volume invariant (\\(T^{(k)}\\)).

---

## 📐 Mathematical Foundations & Pendulum Dynamics

Given an observed base matrix \\(\mathbf{A}^{(0)} = [A_{ij}^{(0)}] \in \mathbb{R}_+^{I \times J}\\), target production totals \\(O_i\\), and attraction totals \\(D_j\\) satisfying system conservation \\(\sum_{i} O_i = \sum_{j} D_j = A_{\text{target}}\\):

1. **Row Scaling Factor:**
   \\[F_i^{(k)} = \frac{O_i}{\sum_{j} A_{ij}^{(k)}}\\]

2. **Column Scaling Factor:**
   \\[G_j^{(k)} = \frac{D_j}{\sum_{i} A_{ij}^{(k)}}\\]

3. **Dynamic System Volume Ratio (\\(T^{(k)}\\)):**
   \\[T^{(k)} = \frac{A_{\text{target}}}{\sum_{i,j} A_{ij}^{(k)}}\\]

4. **Simultaneous Single-Pass Update:**
   \\[A_{ij}^{(k+1)} = A_{ij}^{(k)} \times \left[ \frac{F_i^{(k)} \cdot G_j^{(k)}}{T^{(k)}} \right]\\]

### Vectorized Outer-Product SIMD Formulation
\\[\mathbf{A}^{(k+1)} = \mathbf{A}^{(k)} \odot \left[ \frac{\mathbf{F}^{(k)} \otimes \mathbf{G}^{(k)}}{T^{(k)}} \right]\\]

### 🔄 The Self-Correcting Pendulum Principle
The dynamic factor \\(T^{(k)}\\) acts as a critically damped negative feedback control loop:
* **Volume Expansion (\\(A_{\text{current}} > A_{\text{target}}\\))**: \\(T^{(k)} > 1\\) exerts a contraction force, pulling total volume down.
* **Volume Contraction (\\(A_{\text{current}} < A_{\text{target}}\\))**: \\(T^{(k)} < 1\\) exerts an expansion force, boosting matrix elements.
* **Fixed-Point Equilibrium (\\(A_{\text{current}} = A_{\text{target}}\\))**: \\(T^* = 1.0\\), satisfying Karush-Kuhn-Tucker (KKT) optimality conditions and guaranteeing 1-step exact convergence on \\(1 \times 1\\) contraction boundary cases.

---

## 💻 Quick Start

### 🐍 Python 3 Implementation
```python
import numpy as np

def dual_simultaneous_scaling(A_base, O_target, D_target, tol=1e-5, max_iter=1000):
    """
    Vectorized Dual-Simultaneous Scaling (DSS) Algorithm.
    """
    A_curr = A_base.astype(float).copy()
    target_sum = np.sum(O_target)
    
    for k in range(max_iter):
        row_sums = np.sum(A_curr, axis=1)
        col_sums = np.sum(A_curr, axis=0)
        
        # Compute row and column scale factors
        F_i = np.where(row_sums > 0, O_target / row_sums, 1.0)
        G_j = np.where(col_sums > 0, D_target / col_sums, 1.0)
        
        # Dynamic volume invariant T_k
        curr_total = np.sum(A_curr)
        T_k = target_sum / curr_total if curr_total > 0 else 1.0
        
        # Simultaneous Outer-Product Update
        scale_matrix = np.outer(F_i, G_j) / T_k
        A_next = A_curr * scale_matrix
        
        # Check convergence criteria
        max_row_err = np.max(np.abs(np.sum(A_next, axis=1) - O_target) / O_target)
        max_col_err = np.max(np.abs(np.sum(A_next, axis=0) - D_target) / D_target)
        
        if max(max_row_err, max_col_err) < tol:
            return A_next, k + 1
            
        A_curr = A_next
        
    return A_curr, max_iter

# Example Run
A_base = np.array([[0, 20, 30, 60],
                   [10, 0, 40, 20],
                   [50, 15, 0, 70],
                   [60, 30, 10, 0]], dtype=float)

O_target = np.array([600, 140, 200, 230], dtype=float)
D_target = np.array([230, 550, 90, 300], dtype=float)

A_balanced, iters = dual_simultaneous_scaling(A_base, O_target, D_target)
print(f"Converged in {iters} iterations.")
🔢 MATLAB Implementationfunction [A_balanced, iterations] = dss_algorithm(A_base, O_target, D_target, tol, max_iter)
    if nargin < 4, tol = 1e-5; end
    if nargin < 5, max_iter = 1000; end
    
    A_curr = double(A_base);
    target_sum = sum(O_target);
    
    for k = 1:max_iter
        row_sums = sum(A_curr, 2);
        col_sums = sum(A_curr, 1)';
        
        F_i = O_target ./ max(row_sums, 1e-12);
        G_j = D_target ./ max(col_sums, 1e-12);
        
        curr_total = sum(A_curr, 'all');
        T_k = target_sum / curr_total;
        
        scale_matrix = (F_i * G_j') / T_k;
        A_next = A_curr .* scale_matrix;
        
        row_err = max(abs(sum(A_next, 2) - O_target) ./ O_target);
        col_err = max(abs(sum(A_next, 1)' - D_target) ./ D_target);
        
        if max(row_err, col_err) < tol
            A_balanced = A_next;
            iterations = k;
            return;
        end
        A_curr = A_next;
    end
    A_balanced = A_curr;
    iterations = max_iter;
end
📊 Comprehensive Benchmarks & Sensitivity Analysis1. Matrix Scalability ($N=50$ to $N=3000$)Dimensions ($N \times N$)Total CellsFratar Iter.Furness Iter.DSS Iter.Execution TimeFratar Speedup$50 \times 50$2,5003350.67 ms$14\times$$200 \times 200$40,0003345.92 ms$48\times$$1000 \times 1000$1,000,00022345.40 ms$312\times$$3000 \times 3000$9,000,000223287.61 ms$992\times$2. Sensitivity & Robustness Stress TestsMatrix Sparsity (0% to 90% Zeroes): Stable 4–5 iteration convergence on $500 \times 500$ sparse networks without division-by-zero errors.Severe Contraction (-99% Collapse): 1-step exact convergence on uniform scaling and 4 iterations on localized 20-zone collapse.Extreme Growth (+5000% Shock): Stabilized in 8 iterations under $50\times$ zonal demand shocks.Machine Precision ($\epsilon = 10^{-12}$): Achieved in 7 iterations.🚀 Future Research Directions (Targeting TRC / TRE)Zero-Inflated Noise Modeling: Probabilistic ZINB frameworks for sparse Cellular / LBS mobility data.Elastic Constraints: Lagrangian penalty functions for noisy traffic sensor counts (AVI / Wi-Fi).GPU CUDA / PyTorch Kernels: Parallel tensor implementations for ultra-large matrices ($N > 10,000$).Dynamic 3D OD Tensors: $O \times D \times T$ time-varying tensor balancing.✉️ Citation & ContactAuthor: Yahya Nasr Esfahani
Department of Civil Engineering, Daneshpajouhan Pishro Higher Education Institute, Isfahan, Iran
