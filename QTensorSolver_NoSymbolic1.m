% QTensorSolver_NoSymbolic.m
%=======================================================================%
%  3D Q-Tensor Calculation — No Symbolic, Single-File, HPC-Ready        %
%  Exact L1..L6 elastic model as in your older code                      %
%  MATLAB R2024b                                                         %
%=======================================================================%
%  Output files (structure matches your old script):
%    - Q_matrix.mat: Q11,Q12,Q13,Q22,Q23,Q33
%    - QTensor_director_solution.mat: u_final, v_final, w_final, theta_matrix, phi_matrix
%=======================================================================%

clc; clear; close all;

%% ----------------------- User / Physics Parameters ----------------------
Simulation_name = 'Sin_2pi_HalfInitialCondition_inXY';

% Elastic constants (SI)
K11 = 1.1e-11;  % splay
K22 = 0.65e-11; % twist
K33 = 1.7e-11;  % bend
K24 = 0.4e-11;  % saddle-splay

% Initial order parameter and measurement S for L_i mapping
S_exp = 0.1;      % initial bulk S1
S_mes = 0.9;      % S used to compute L_i
q0    = 0;        % cholesteric pitch (0 for nematic)

% Landau-de Gennes bulk (thermotropic)
a_thermo = -1740;
b_thermo = -21200;
c_thermo = 17400;

% Domain and grid
L_X = 13e-6;   L_Y = 13e-6;   Lz = 6e-6;

% !!! Start with a smaller grid to validate (HPC-size needs tiling) !!!
N_X = 50;      N_Y = 50;      N_Z = 20;
% After validation, revert to your large sizes and we’ll add z-tiling.

% Iteration
max_iter = 1000000;  % practical limit for test (your older code had huge value)
tol      = 1e-9;

% Pattern parameter
Lambda = 6.5e-6;

% Visualization flags (off for performance/HPC)
plot_result = 0;
wait_bar    = 0;

% Optional: enforce MAT-file version (uncomment if required)
% save_version = '-v7';

%% -------------------------- Derived constants ---------------------------
dx = L_X / (N_X - 1);
dy = L_Y / (N_Y - 1);
dz = Lz  / (N_Z - 1);

% L1..L6 from K's (your exact formulas)
L1 = (1/(6*S_mes^2)) * (K33 - K11 + 3*K22);
L2 = (1/S_mes^2)    * (K11 - K22 - K24);
L3 = (1/S_mes^2)    * K24;
L4 = (2/S_mes^2)    * q0 * K22;
L6 = (1/(2*S_mes^3)) * (K33 - K11);

%% ---------------------------- Grid & pattern ----------------------------
x_vals = linspace(0, L_X, N_X);
y_vals = linspace(0, L_Y, N_Y);
[X, Y] = meshgrid(x_vals, y_vals);

% phi(x) = mod( atan2(cos(pi x / Lambda), sin(pi x / Lambda)), pi )
phi_bottom = mod( atan2(cos(pi * X / Lambda), sin(pi * X / Lambda)), pi );
phi_top    = phi_bottom.'; % top is transposed in your code

%% ------------------------- Allocate Q fields (double) -------------------
Q11 = zeros(N_Y, N_X, N_Z, 'double');
Q12 = zeros(N_Y, N_X, N_Z, 'double');
Q13 = zeros(N_Y, N_X, N_Z, 'double');
Q22 = zeros(N_Y, N_X, N_Z, 'double');
Q23 = zeros(N_Y, N_X, N_Z, 'double');
% Q33 is derived: Q33 = -(Q11 + Q22)

%% --------------------- Vectorized BC + bulk initialization --------------
% Bottom z=1 (S1=1, theta=0, psi=0)
[Q11(:,:,1), Q12(:,:,1), Q13(:,:,1), Q22(:,:,1), Q23(:,:,1)] = ...
    init_bc_from_phi(phi_bottom, 1, 0, 0);

% Top z=Nz (same pattern transposed)
[Q11(:,:,N_Z), Q12(:,:,N_Z), Q13(:,:,N_Z), Q22(:,:,N_Z), Q23(:,:,N_Z)] = ...
    init_bc_from_phi(phi_top, 1, 0, 0);

% Bulk initial guess (S1=S_exp; lower half uses bottom phi, upper half uses top phi)
midZ = N_Z/2;
for k = 2:N_Z-1
    if k < midZ
        [Q11(:,:,k), Q12(:,:,k), Q13(:,:,k), Q22(:,:,k), Q23(:,:,k)] = ...
            init_bc_from_phi(phi_bottom, S_exp, 0, 0);
    else
        [Q11(:,:,k), Q12(:,:,k), Q13(:,:,k), Q22(:,:,k), Q23(:,:,k)] = ...
            init_bc_from_phi(phi_top,    S_exp, 0, 0);
    end
end

%% ---------------------------- FD operators ------------------------------
fd = fd_ops(N_Y, N_X, N_Z, dx, dy, dz);  % periodic x/y, centered z

%% ---------------------------- Iteration loop ----------------------------
fprintf('Starting iterations (no symbolic)…\n');
alpha = [];  % will be set on iter==1 (as in your older code)

if wait_bar==1
    hWaitbar = waitbar(0, 'Iteration 1', 'Name', 'Solving Q-tensor problem','CreateCancelBtn','delete(gcbf)');
end

tic;
for iter = 1:max_iter

    % Keep to restore BC at z-boundaries
    Q11_old = Q11; Q12_old = Q12; Q13_old = Q13; Q22_old = Q22; Q23_old = Q23;

    % ---------- Build full symmetric Q and first derivatives ----------
    Q33 = -(Q11 + Q22);

    % First derivatives for independent comps
    dQ11 = {fd.Dx(Q11), fd.Dy(Q11), fd.Dz(Q11)};
    dQ12 = {fd.Dx(Q12), fd.Dy(Q12), fd.Dz(Q12)};
    dQ13 = {fd.Dx(Q13), fd.Dy(Q13), fd.Dz(Q13)};
    dQ22 = {fd.Dx(Q22), fd.Dy(Q22), fd.Dz(Q22)};
    dQ23 = {fd.Dx(Q23), fd.Dy(Q23), fd.Dz(Q23)};
    % Q33 from traceless: dQ33 = -(dQ11 + dQ22)
    dQ33 = {-(dQ11{1}+dQ22{1}), -(dQ11{2}+dQ22{2}), -(dQ11{3}+dQ22{3})};

    % Package Q and dQ into 3x3 cells for compact index ops
    Q = cell(3,3);
    Q{1,1}=Q11; Q{1,2}=Q12; Q{1,3}=Q13;
    Q{2,1}=Q12; Q{2,2}=Q22; Q{2,3}=Q23;
    Q{3,1}=Q13; Q{3,2}=Q23; Q{3,3}=Q33;

    dQx = cell(3,3); dQy = cell(3,3); dQz = cell(3,3);
    % x-derivs
    dQx{1,1}=dQ11{1}; dQx{1,2}=dQ12{1}; dQx{1,3}=dQ13{1};
    dQx{2,1}=dQ12{1}; dQx{2,2}=dQ22{1}; dQx{2,3}=dQ23{1};
    dQx{3,1}=dQ13{1}; dQx{3,2}=dQ23{1}; dQx{3,3}=dQ33{1};
    % y-derivs
    dQy{1,1}=dQ11{2}; dQy{1,2}=dQ12{2}; dQy{1,3}=dQ13{2};
    dQy{2,1}=dQ12{2}; dQy{2,2}=dQ22{2}; dQy{2,3}=dQ23{2};
    dQy{3,1}=dQ13{2}; dQy{3,2}=dQ23{2}; dQy{3,3}=dQ33{2};
    % z-derivs
    dQz{1,1}=dQ11{3}; dQz{1,2}=dQ12{3}; dQz{1,3}=dQ13{3};
    dQz{2,1}=dQ12{3}; dQz{2,2}=dQ22{3}; dQz{2,3}=dQ23{3};
    dQz{3,1}=dQ13{3}; dQz{3,2}=dQ23{3}; dQz{3,3}=dQ33{3};

    % Divergence-like vectors A_i = ∂_j Q_{ij}
    A = cell(3,1);
    A{1} = dQx{1,1} + dQy{1,2} + dQz{1,3};
    A{2} = dQx{2,1} + dQy{2,2} + dQz{2,3};
    A{3} = dQx{3,1} + dQy{3,2} + dQz{3,3};

    % ---------- H = ∂f/∂Q  and  G^α = ∂f/∂(∂_α Q)  (α ∈ {x,y,z}) ----------
    H   = cell(3,3);
    Gx  = cell(3,3); Gy = cell(3,3); Gz = cell(3,3);
    for i=1:3, for j=1:3
        H{i,j}  = zeros(N_Y, N_X, N_Z,'double');
        Gx{i,j} = zeros(N_Y, N_X, N_Z,'double');
        Gy{i,j} = zeros(N_Y, N_X, N_Z,'double');
        Gz{i,j} = zeros(N_Y, N_X, N_Z,'double');
    end, end

    % ----- Bulk term: 2a Q + 2b Q^2 + 2c tr(Q^2) Q  -----
    [Q2, trQ2] = Qsquare_and_trace(Q); % Q2 is 3x3 cell; trQ2 is array
    for i=1:3, for j=1:3
        H{i,j} = H{i,j} + 2*a_thermo*Q{i,j} + 2*b_thermo*Q2{i,j} + 2*c_thermo*trQ2.*Q{i,j};
    end, end

    % ----- L1: f = (L1/2)*(∂_k Q_ij)^2 -> G^α = L1 * ∂_α Q_ij -----
    for i=1:3, for j=1:3
        Gx{i,j} = Gx{i,j} + L1 * dQx{i,j};
        Gy{i,j} = Gy{i,j} + L1 * dQy{i,j};
        Gz{i,j} = Gz{i,j} + L1 * dQz{i,j};
    end, end

    % ----- L2: f = (L2/2) * (∂_j Q_ij)(∂_k Q_ik) = (L2/2) * sum_i A_i^2
    % ∂f/∂(∂_α Q_mn) = L2 * A_m * δ_{nα}
% QTensorSolver_NoSymbolic.m
%=======================================================================%
%  3D Q-Tensor Calculation — No Symbolic, Single-File, HPC-Ready        %
%  Exact L1..L6 elastic model as in your older code                      %
%  MATLAB R2024b                                                         %
%=======================================================================%
%  Output files (structure matches your old script):
%    - Q_matrix.mat: Q11,Q12,Q13,Q22,Q23,Q33
%    - QTensor_director_solution.mat: u_final, v_final, w_final, theta_matrix, phi_matrix
%=======================================================================%

clc; clear; close all;

%% ----------------------- User / Physics Parameters ----------------------
Simulation_name = 'Sin_2pi_HalfInitialCondition_inXY';

% Elastic constants (SI)
K11 = 1.1e-11;  % splay
K22 = 0.65e-11; % twist
K33 = 1.7e-11;  % bend
K24 = 0.4e-11;  % saddle-splay

% Initial order parameter and measurement S for L_i mapping
S_exp = 0.1;      % initial bulk S1
S_mes = 0.9;      % S used to compute L_i
q0    = 0;        % cholesteric pitch (0 for nematic)

% Landau-de Gennes bulk (thermotropic)
a_thermo = -1740;
b_thermo = -21200;
c_thermo = 17400;

% Domain and grid
L_X = 13e-6;   L_Y = 13e-6;   Lz = 6e-6;

% !!! Start with a smaller grid to validate (HPC-size needs tiling) !!!
N_X = 50;      N_Y = 50;      N_Z = 20;
% After validation, revert to your large sizes and we’ll add z-tiling.

% Iteration
max_iter = 1000000;  % practical limit for test (your older code had huge value)
tol      = 1e-9;

% Pattern parameter
Lambda = 6.5e-6;

% Visualization flags (off for performance/HPC)
plot_result = 0;
wait_bar    = 0;

% Optional: enforce MAT-file version (uncomment if required)
% save_version = '-v7';

%% -------------------------- Derived constants ---------------------------
dx = L_X / (N_X - 1);
dy = L_Y / (N_Y - 1);
dz = Lz  / (N_Z - 1);

% L1..L6 from K's (your exact formulas)
L1 = (1/(6*S_mes^2)) * (K33 - K11 + 3*K22);
L2 = (1/S_mes^2)    * (K11 - K22 - K24);
L3 = (1/S_mes^2)    * K24;
L4 = (2/S_mes^2)    * q0 * K22;
L6 = (1/(2*S_mes^3)) * (K33 - K11);

%% ---------------------------- Grid & pattern ----------------------------
x_vals = linspace(0, L_X, N_X);
y_vals = linspace(0, L_Y, N_Y);
[X, Y] = meshgrid(x_vals, y_vals);

% phi(x) = mod( atan2(cos(pi x / Lambda), sin(pi x / Lambda)), pi )
phi_bottom = mod( atan2(cos(pi * X / Lambda), sin(pi * X / Lambda)), pi );
phi_top    = phi_bottom.'; % top is transposed in your code

%% ------------------------- Allocate Q fields (double) -------------------
Q11 = zeros(N_Y, N_X, N_Z, 'double');
Q12 = zeros(N_Y, N_X, N_Z, 'double');
Q13 = zeros(N_Y, N_X, N_Z, 'double');
Q22 = zeros(N_Y, N_X, N_Z, 'double');
Q23 = zeros(N_Y, N_X, N_Z, 'double');
% Q33 is derived: Q33 = -(Q11 + Q22)

%% --------------------- Vectorized BC + bulk initialization --------------
% Bottom z=1 (S1=1, theta=0, psi=0)
[Q11(:,:,1), Q12(:,:,1), Q13(:,:,1), Q22(:,:,1), Q23(:,:,1)] = ...
    init_bc_from_phi(phi_bottom, 1, 0, 0);

% Top z=Nz (same pattern transposed)
[Q11(:,:,N_Z), Q12(:,:,N_Z), Q13(:,:,N_Z), Q22(:,:,N_Z), Q23(:,:,N_Z)] = ...
    init_bc_from_phi(phi_top, 1, 0, 0);

% Bulk initial guess (S1=S_exp; lower half uses bottom phi, upper half uses top phi)
midZ = N_Z/2;
for k = 2:N_Z-1
    if k < midZ
        [Q11(:,:,k), Q12(:,:,k), Q13(:,:,k), Q22(:,:,k), Q23(:,:,k)] = ...
            init_bc_from_phi(phi_bottom, S_exp, 0, 0);
    else
        [Q11(:,:,k), Q12(:,:,k), Q13(:,:,k), Q22(:,:,k), Q23(:,:,k)] = ...
            init_bc_from_phi(phi_top,    S_exp, 0, 0);
    end
end

%% ---------------------------- FD operators ------------------------------
fd = fd_ops(N_Y, N_X, N_Z, dx, dy, dz);  % periodic x/y, centered z

%% ---------------------------- Iteration loop ----------------------------
fprintf('Starting iterations (no symbolic)…\n');
alpha = [];  % will be set on iter==1 (as in your older code)

if wait_bar==1
    hWaitbar = waitbar(0, 'Iteration 1', 'Name', 'Solving Q-tensor problem','CreateCancelBtn','delete(gcbf)');
end

tic;
for iter = 1:max_iter

    % Keep to restore BC at z-boundaries
    Q11_old = Q11; Q12_old = Q12; Q13_old = Q13; Q22_old = Q22; Q23_old = Q23;

    % ---------- Build full symmetric Q and first derivatives ----------
    Q33 = -(Q11 + Q22);

    % First derivatives for independent comps
    dQ11 = {fd.Dx(Q11), fd.Dy(Q11), fd.Dz(Q11)};
    dQ12 = {fd.Dx(Q12), fd.Dy(Q12), fd.Dz(Q12)};
    dQ13 = {fd.Dx(Q13), fd.Dy(Q13), fd.Dz(Q13)};
    dQ22 = {fd.Dx(Q22), fd.Dy(Q22), fd.Dz(Q22)};
    dQ23 = {fd.Dx(Q23), fd.Dy(Q23), fd.Dz(Q23)};
    % Q33 from traceless: dQ33 = -(dQ11 + dQ22)
    dQ33 = {-(dQ11{1}+dQ22{1}), -(dQ11{2}+dQ22{2}), -(dQ11{3}+dQ22{3})};

    % Package Q and dQ into 3x3 cells for compact index ops
    Q = cell(3,3);
    Q{1,1}=Q11; Q{1,2}=Q12; Q{1,3}=Q13;
    Q{2,1}=Q12; Q{2,2}=Q22; Q{2,3}=Q23;
    Q{3,1}=Q13; Q{3,2}=Q23; Q{3,3}=Q33;

    dQx = cell(3,3); dQy = cell(3,3); dQz = cell(3,3);
    % x-derivs
    dQx{1,1}=dQ11{1}; dQx{1,2}=dQ12{1}; dQx{1,3}=dQ13{1};
    dQx{2,1}=dQ12{1}; dQx{2,2}=dQ22{1}; dQx{2,3}=dQ23{1};
    dQx{3,1}=dQ13{1}; dQx{3,2}=dQ23{1}; dQx{3,3}=dQ33{1};
    % y-derivs
    dQy{1,1}=dQ11{2}; dQy{1,2}=dQ12{2}; dQy{1,3}=dQ13{2};
    dQy{2,1}=dQ12{2}; dQy{2,2}=dQ22{2}; dQy{2,3}=dQ23{2};
    dQy{3,1}=dQ13{2}; dQy{3,2}=dQ23{2}; dQy{3,3}=dQ33{2};
    % z-derivs
    dQz{1,1}=dQ11{3}; dQz{1,2}=dQ12{3}; dQz{1,3}=dQ13{3};
    dQz{2,1}=dQ12{3}; dQz{2,2}=dQ22{3}; dQz{2,3}=dQ23{3};
    dQz{3,1}=dQ13{3}; dQz{3,2}=dQ23{3}; dQz{3,3}=dQ33{3};

    % Divergence-like vectors A_i = ∂_j Q_{ij}
    A = cell(3,1);
    A{1} = dQx{1,1} + dQy{1,2} + dQz{1,3};
    A{2} = dQx{2,1} + dQy{2,2} + dQz{2,3};
    A{3} = dQx{3,1} + dQy{3,2} + dQz{3,3};

    % ---------- H = ∂f/∂Q  and  G^α = ∂f/∂(∂_α Q)  (α ∈ {x,y,z}) ----------
    H   = cell(3,3);
    Gx  = cell(3,3); Gy = cell(3,3); Gz = cell(3,3);
    for i=1:3, for j=1:3
        H{i,j}  = zeros(N_Y, N_X, N_Z,'double');
        Gx{i,j} = zeros(N_Y, N_X, N_Z,'double');
        Gy{i,j} = zeros(N_Y, N_X, N_Z,'double');
        Gz{i,j} = zeros(N_Y, N_X, N_Z,'double');
    end, end

    % ----- Bulk term: 2a Q + 2b Q^2 + 2c tr(Q^2) Q  -----
    [Q2, trQ2] = Qsquare_and_trace(Q); % Q2 is 3x3 cell; trQ2 is array
    for i=1:3, for j=1:3
        H{i,j} = H{i,j} + 2*a_thermo*Q{i,j} + 2*b_thermo*Q2{i,j} + 2*c_thermo*trQ2.*Q{i,j};
    end, end

    % ----- L1: f = (L1/2)*(∂_k Q_ij)^2 -> G^α = L1 * ∂_α Q_ij -----
    for i=1:3, for j=1:3
        Gx{i,j} = Gx{i,j} + L1 * dQx{i,j};
        Gy{i,j} = Gy{i,j} + L1 * dQy{i,j};
        Gz{i,j} = Gz{i,j} + L1 * dQz{i,j};
    end, end

    % ----- L2: f = (L2/2) * (∂_j Q_ij)(∂_k Q_ik) = (L2/2) * sum_i A_i^2
    % ∂f/∂(∂_α Q_mn) = L2 * A_m * δ_{nα}
    for m=1:3
        for n=1:3
            if n==1, Gx{m,n} = Gx{m,n} + L2 * A{m}; end
            if n==2, Gy{m,n} = Gy{m,n} + L2 * A{m}; end
            if n==3, Gz{m,n} = Gz{m,n} + L2 * A{m}; end
        end
    end

    % ----- L3: f = (L3/2) * (∂_j Q_ik)(∂_k Q_ij)
    % ∂f/∂(∂_α Q_mn) = L3 * ∂_n Q_{mα}
    for m=1:3, for n=1:3
        Gx{m,n} = Gx{m,n} + L3 * Dn_of_Q(n, Q{m,1}, dQx{m,1}, dQy{m,1}, dQz{m,1});
        Gy{m,n} = Gy{m,n} + L3 * Dn_of_Q(n, Q{m,2}, dQx{m,2}, dQy{m,2}, dQz{m,2});
        Gz{m,n} = Gz{m,n} + L3 * Dn_of_Q(n, Q{m,3}, dQx{m,3}, dQy{m,3}, dQz{m,3});
    end, end

    % ----- L4: f = (L4/2) * ε_{lik} Q_{lj} ∂_k Q_{ij}
    % G^α_{mn} = (L4/2) * ε_{l m α} Q_{l n}
    % H_{mn}   = (L4/2) * ε_{m i k} ∂_k Q_{i n}
    for m=1:3, for n=1:3
        % H part
        tmpH = zeros(N_Y,N_X,N_Z);
        for i_=1:3, for k_=1:3
            eps_val = levi(m,i_,k_);
            if eps_val~=0
                if     k_==1, tmpH = tmpH + eps_val * dQx{i_,n};
                elseif k_==2, tmpH = tmpH + eps_val * dQy{i_,n};
                else          tmpH = tmpH + eps_val * dQz{i_,n};
                end
            end
        end, end
        H{m,n} = H{m,n} + (L4/2) * tmpH;

        % G part (for each alpha)
        % alpha = 1 (x)
        tmp = zeros(N_Y,N_X,N_Z);
        for l=1:3
            eps_val = levi(l,m,1);
            if eps_val~=0, tmp = tmp + eps_val * Q{l,n}; end
        end
        Gx{m,n} = Gx{m,n} + (L4/2) * tmp;

        % alpha = 2 (y)
        tmp = zeros(N_Y,N_X,N_Z);
        for l=1:3
            eps_val = levi(l,m,2);
            if eps_val~=0, tmp = tmp + eps_val * Q{l,n}; end
        end
        Gy{m,n} = Gy{m,n} + (L4/2) * tmp;

        % alpha = 3 (z)
        tmp = zeros(N_Y,N_X,N_Z);
        for l=1:3
            eps_val = levi(l,m,3);
            if eps_val~=0, tmp = tmp + eps_val * Q{l,n}; end
        end
        Gz{m,n} = Gz{m,n} + (L4/2) * tmp;
    end, end

    % ----- L6: f = (L6/2) * Q_{lk} (∂_l Q_{ij})(∂_k Q_{ij})
    % G^α_{mn} = L6 * Q_{αβ} ∂_β Q_{mn}    (sum over β)
    % H_{mn}   = (L6/2) * (∂_m Q_{ij})(∂_n Q_{ij})  (sum over i,j)
    for m=1:3, for n=1:3
        % Gx: α=1
        tmp = zeros(N_Y,N_X,N_Z);
        for beta=1:3
            Qab = Q{1,beta};  % Q_{α(=1),beta}
            dQmn_beta = pickD(beta, dQx{m,n}, dQy{m,n}, dQz{m,n});
            tmp = tmp + Qab .* dQmn_beta;
        end
        Gx{m,n} = Gx{m,n} + L6 * tmp;

        % Gy: α=2
        tmp = zeros(N_Y,N_X,N_Z);
        for beta=1:3
            Qab = Q{2,beta};
            dQmn_beta = pickD(beta, dQx{m,n}, dQy{m,n}, dQz{m,n});
            tmp = tmp + Qab .* dQmn_beta;
        end
        Gy{m,n} = Gy{m,n} + L6 * tmp;

        % Gz: α=3
        tmp = zeros(N_Y,N_X,N_Z);
        for beta=1:3
            Qab = Q{3,beta};
            dQmn_beta = pickD(beta, dQx{m,n}, dQy{m,n}, dQz{m,n});
            tmp = tmp + Qab .* dQmn_beta;
        end
        Gz{m,n} = Gz{m,n} + L6 * tmp;
    end, end

    % H part for L6
    for m=1:3, for n=1:3
        tmp = zeros(N_Y,N_X,N_Z);
        % sum over i,j of (∂_m Q_{ij})(∂_n Q_{ij})
        for i_=1:3, for j_=1:3
            dQij_m = pickD(m, dQx{i_,j_}, dQy{i_,j_}, dQz{i_,j_});
            dQij_n = pickD(n, dQx{i_,j_}, dQy{i_,j_}, dQz{i_,j_});
            tmp = tmp + dQij_m .* dQij_n;
        end, end
        H{m,n} = H{m,n} + (L6/2) * tmp;
    end, end

    % ---------- Residuals: R = H - div(G) ----------
    R = cell(3,3);
    for i=1:3, for j=1:3
        divG = fd.Dx(Gx{i,j}) + fd.Dy(Gy{i,j}) + fd.Dz(Gz{i,j});
        R{i,j} = H{i,j} - divG;
        % Zero residual on fixed z-boundaries
        R{i,j}(:,:,1)   = 0;
        R{i,j}(:,:,N_Z) = 0;
    end, end

    % ---------- Alpha (first iter) like your code ----------
    if iter == 1
        max_error = 0;
        for i=1:3, for j=1:3
            max_error = max(max_error, max(abs(R{i,j}(:))));
        end, end
        max_old = max([max(abs(Q11_old(:))),max(abs(Q12_old(:))),max(abs(Q13_old(:))), ...
                       max(abs(Q22_old(:))),max(abs(Q23_old(:)))]);
        alpha = (max_old/max_error)/50;
        alpha = min(max(alpha, 1e-12), 1e-2);
        fprintf('Alpha initialized to %.3e\n', alpha);
    end

    % ---------- Update only independent components ----------
    Q11 = Q11 - alpha * R{1,1};
    Q12 = Q12 - alpha * R{1,2};
    Q13 = Q13 - alpha * R{1,3};
    Q22 = Q22 - alpha * R{2,2};
    Q23 = Q23 - alpha * R{2,3};

    % Restore fixed z-boundaries
    Q11(:,:,1)=Q11_old(:,:,1); Q11(:,:,N_Z)=Q11_old(:,:,N_Z);
    Q12(:,:,1)=Q12_old(:,:,1); Q12(:,:,N_Z)=Q12_old(:,:,N_Z);
    Q13(:,:,1)=Q13_old(:,:,1); Q13(:,:,N_Z)=Q13_old(:,:,N_Z);
    Q22(:,:,1)=Q22_old(:,:,1); Q22(:,:,N_Z)=Q22_old(:,:,N_Z);
    Q23(:,:,1)=Q23_old(:,:,1); Q23(:,:,N_Z)=Q23_old(:,:,N_Z);

    % ---------- Convergence ----------
    max_change = max([ ...
        max(abs(Q11(:)-Q11_old(:))), ...
        max(abs(Q12(:)-Q12_old(:))), ...
        max(abs(Q13(:)-Q13_old(:))), ...
        max(abs(Q22(:)-Q22_old(:))), ...
        max(abs(Q23(:)-Q23_old(:))) ]);

    if mod(iter,200)==0
        fprintf('Iter %d: max change = %.3e\n', iter, max_change);
    end
    if wait_bar==1
        if ~ishandle(hWaitbar), disp('Stopped by user'); break; end
        if mod(iter, 100) == 0
            waitbar(iter/max_iter, hWaitbar, ...
                sprintf('Iteration %d, Max change: %.2e', iter, max_change));
        end
    end
    if max_change < tol
        fprintf('Converged after %d iterations. max change = %.2e\n', iter, max_change);
        break;
    end
end
toc;

if wait_bar==1 && ishandle(hWaitbar), close(hWaitbar); end

%% ----------------------------- Save Q matrices --------------------------
Q33 = -(Q11 + Q22);
save("Q_matrix.mat","Q11","Q12","Q13","Q22","Q23","Q33");
fnameQ = sprintf('Q_matrix-%s-K11_%g-K22_%g-K33_%g-K24_%g-Lx_%g-Ly_%g-Lz_%g-Nx_%d-Ny_%d-Nz_%d.mat', ...
                 Simulation_name,K11,K22,K33,K24,L_X,L_Y,Lz,N_X,N_Y,N_Z);
save(fullfile('.', fnameQ), "Q11","Q12","Q13","Q22","Q23","Q33");

%% --------------------- Extract director + save director -----------------
u_final = zeros(N_Y, N_X, N_Z); v_final = u_final; w_final = u_final;
theta_matrix = u_final; phi_matrix = u_final;

parfor iy = 1:N_Y
    for ix = 1:N_X
        for iz = 1:N_Z
            Qloc = [Q11(iy,ix,iz), Q12(iy,ix,iz), Q13(iy,ix,iz);
                    Q12(iy,ix,iz), Q22(iy,ix,iz), Q23(iy,ix,iz);
                    Q13(iy,ix,iz), Q23(iy,ix,iz), -(Q11(iy,ix,iz)+Q22(iy,ix,iz))];
            [V,D] = eig(Qloc);
            ev = diag(D); [~,idx] = max(ev);
            n = V(:,idx); n = n / max(norm(n),eps);
            if n(3) < 0
                n = -n;
            elseif n(3) == 0
                if n(2) < 0 || (n(2)==0 && n(1) < 0), n = -n; end
            end
            u_final(iy,ix,iz) = n(1);
            v_final(iy,ix,iz) = n(2);
            w_final(iy,ix,iz) = n(3);
            theta_matrix(iy,ix,iz) = acos(max(min(n(3),1),-1));
            phi_matrix(iy,ix,iz)   = atan2(n(2), n(1));
        end
    end
end

save('QTensor_director_solution.mat', 'u_final','v_final','w_final','theta_matrix','phi_matrix');
fndir = sprintf('QTensor_director_solution-%s-K11_%g-K22_%g-K33_%g-K24_%g-Lx_%g-Ly_%g-Lz_%g-Nx_%d-Ny_%d-Nz_%d.mat', ...
                Simulation_name,K11,K22,K33,K24,L_X,L_Y,Lz,N_X,N_Y,N_Z);
save(fullfile('.', fndir), 'u_final','v_final','w_final','theta_matrix','phi_matrix');

fprintf('All done. Outputs saved.\n');

%% ============================== Local helpers ===========================
function [Q11,Q12,Q13,Q22,Q23] = init_bc_from_phi(phi, S1, theta, psi)
    % Vectorized conversion of (theta,phi,psi,S1,S2=0) -> Q components
    S2 = 0;
    c = cos(phi); s = sin(phi);
    ct = cos(theta); st = sin(theta);
    cp = cos(psi); sp = sin(psi);
    Q11 = S1*(ct.^2 .* c.^2) + S2*(s.*cp - c.*sp.*st).^2 - (1/3)*(S1 + S2);
    Q12 = S1*(ct.^2 .* s.*c) - S2*(c.*cp + s.*sp.*st).*(s.*cp - c.*sp.*st);
    Q13 = S1*(st.*ct.*c)     + S2*(sp.*ct).*(s.*cp - c.*sp.*st);
    Q22 = S1*(ct.^2 .* s.^2) + S2*(c.*cp + s.*sp.*st).^2 - (1/3)*(S1 + S2);
    Q23 = S1*(ct.*st.*s)     - S2*(sp.*ct).*(c.*cp + s.*sp.*st);
end

function fd = fd_ops(Ny, Nx, Nz, dx, dy, dz)
    % periodic x/y, centered z
    ip1 = @(n) [2:n,1];
    im1 = @(n) [n,1:n-1];
    ixp = ip1(Nx); ixm = im1(Nx);
    iyp = ip1(Ny); iym = im1(Ny);
    kint = 2:Nz-1;

    Dx = @(Q) (Q(:,ixp,:) - Q(:,ixm,:)) / (2*dx);
    Dy = @(Q) (Q(iyp,:,:) - Q(iym,:,:)) / (2*dy);
    Dz = @(Q) cat(3, zeros(Ny,Nx,1), (Q(:,:,kint+1) - Q(:,:,kint-1))/(2*dz), zeros(Ny,Nx,1));

    fd = struct('Dx',Dx,'Dy',Dy,'Dz',Dz);
end

function e = levi(i,j,k)
    if i==j || j==k || i==k
        e = 0;
    elseif (i==1 && j==2 && k==3) || (i==2 && j==3 && k==1) || (i==3 && j==1 && k==2)
        e = 1;
    elseif (i==3 && j==2 && k==1) || (i==2 && j==1 && k==3) || (i==1 && j==3 && k==2)
        e = -1;
    else
        e = 0;
    end
end

function DnQ = Dn_of_Q(n, Qij, dQx_ij, dQy_ij, dQz_ij)
    % Return derivative along coordinate "n" (1=x,2=y,3=z)
    if n==1, DnQ = dQx_ij;
    elseif n==2, DnQ = dQy_ij;
    else,       DnQ = dQz_ij;
    end
end

function D = pickD(n, DxQ, DyQ, DzQ)
    if n==1, D = DxQ;
    elseif n==2, D = DyQ;
    else,        D = DzQ;
    end
end

function [Q2, trQ2] = Qsquare_and_trace(Q)
    % Q2 = Q*Q (componentwise 3x3), trQ2 scalar array
    Q2 = cell(3,3);
    for i=1:3, for j=1:3
        tmp = zeros(size(Q{1,1}), 'like', Q{1,1});
        for k=1:3
            tmp = tmp + Q{i,k} .* Q{k,j};
        end
        Q2{i,j} = tmp;
    end, end
    trQ2 = Q2{1,1} + Q2{2,2} + Q2{3,3};
end