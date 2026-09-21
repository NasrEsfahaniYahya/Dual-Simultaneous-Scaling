function [T_balanced, iterations, elapsed_time_ms] = dss_algorithm(T_base, O_target, D_target, tol, max_iter)
% DUAL_SIMULTANEOUS_SCALING Vectorized Matrix Equilibration Algorithm
%
% Author: Yahya Nasr Esfahani (nasr.y20@gmail.com)
% Affiliation: Department of Civil Engineering, Daneshpajouhan Pishro Higher Education Institute
%
% Inputs:
%   T_base   : Non-negative base matrix (I x J)
%   O_target : Target row totals (I x 1)
%   D_target : Target column totals (1 x J or J x 1)
%   tol      : Convergence tolerance (default: 1e-5)
%   max_iter : Maximum iterations allowed (default: 1000)
%
% Outputs:
%   T_balanced      : Balanced matrix satisfying marginal constraints
%   iterations      : Number of iterations executed until convergence
%   elapsed_time_ms : Execution time in milliseconds

if nargin < 4 || isempty(tol)
    tol = 1e-5;
end
if nargin < 5 || isempty(max_iter)
    max_iter = 1000;
end

O_target = O_target(:); % Ensure column vector
D_target = D_target(:)'; % Ensure row vector

T = double(T_base);
target_sum = sum(O_target);

tic;

for k = 1:max_iter
    O_curr = sum(T, 2); % Row sums (I x 1)
    D_curr = sum(T, 1); % Col sums (1 x J)
    
    % Prevent division by zero
    O_curr(O_curr == 0) = 1e-12;
    D_curr(D_curr == 0) = 1e-12;
    
    F_i = O_target ./ O_curr; % (I x 1)
    F_j = D_target ./ D_curr; % (1 x J)
    
    curr_total = sum(T, 'all');
    if curr_total > 0
        T_factor = curr_total / target_sum;
    else
        T_factor = 1.0;
    end
    
    % Vectorized outer-product update (Single-Pass SIMD)
    scale_matrix = (F_i * F_j) ./ T_factor;
    T = T .* scale_matrix;
    
    % Convergence check
    err_o = max(abs(sum(T, 2) - O_target) ./ O_target);
    err_d = max(abs(sum(T, 1) - D_target) ./ D_target);
    
    if max(err_o, err_d) < tol
        iterations = k;
        elapsed_time_ms = toc * 1000;
        T_balanced = T;
        return;
    end
end

iterations = max_iter;
elapsed_time_ms = toc * 1000;
T_balanced = T;

end
