clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Define material parameters for Q-tensor formulation (numerical values)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Simulation_name = 'Sin_2pi_HalfInitialCondition_inXY';
K11 = 1.1e-11;      % Splay elastic constant (in SI units)
K22 = 0.65e-11;     % Twist elastic constant
K33 = 1.7e-11;      % Bend elastic constant
K24 = 0.4e-11;      % Saddle-splay constant
first_run = 1;
S_exp = 0.1;        % Equilibrium order parameter for bulk as an initial condition
S_mes = 0.9;        % elastic constant measurement condition for S parameter
q0 = 0;             % Pitch for cholesteric LC (set to 0 for nematic)

% For Thermotropic energy
a_thermo = -1740;
b_thermo = -21200;
c_thermo = 17400;

% Define the grid (REDUCED FOR TESTING - increase later)
L_X = 13e-6;
L_Y = 13e-6;
Lz = 3e-6;
N_X = 50;      % Reduced for testing
N_Y = 50;      % Reduced for testing
N_Z = 25;      % Reduced for testing

% Plot figures and show waitbar
plot_result = 0;
wait_bar = 0;

% Parameters for numerical calculation
max_iter = 10000;    % Reduced for testing
tol = 1e-3;
dx = L_X / (N_X - 1);
dy = L_Y / (N_Y - 1);
dz = Lz / (N_Z - 1);

% Pre-compute inverse spacings
dx_inv = 1/(2*dx); dy_inv = 1/(2*dy); dz_inv = 1/(2*dz);
dx2_inv = 1/dx^2; dy2_inv = 1/dy^2; dz2_inv = 1/dz^2;
dxdy_inv = 1/(4*dx*dy); dxdz_inv = 1/(4*dx*dz); dydz_inv = 1/(4*dy*dz);

% Calculate L1 to L6 elastic parameters
L1 = (1/(6*S_mes^2)) * (K33 - K11 + 3*K22);
L2 = (1/S_mes^2) * (K11 - K22 - K24);
L3 = (1/S_mes^2) * K24;
L4 = (2/S_mes^2) * q0 * K22;
L6 = (1/(2*S_mes^3)) * (K33 - K11);

% Check for GPU availability
GPU_computation = 0;
if canUseGPU && gpuDeviceCount > 0
    disp('GPU found, running on GPU');
    GPU_computation = 1;
    gpuDevice(1); % Select first GPU
else
    disp('No GPU found, running on CPU');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SYMBOLIC PRE-COMPUTATION - DONE ONCE BEFORE LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Pre-computing symbolic expressions...');

% Define symbolic variables
syms x y z real
syms Q11 Q12 Q13 Q22 Q23 Q33 real
syms dQ11_dx dQ11_dy dQ11_dz real
syms dQ12_dx dQ12_dy dQ12_dz real
syms dQ13_dx dQ13_dy dQ13_dz real
syms dQ22_dx dQ22_dy dQ22_dz real
syms dQ23_dx dQ23_dy dQ23_dz real
syms dQ33_dx dQ33_dy dQ33_dz real

syms d2Q11_dx2 d2Q11_dy2 d2Q11_dz2 real
syms d2Q12_dx2 d2Q12_dy2 d2Q12_dz2 real
syms d2Q13_dx2 d2Q13_dy2 d2Q13_dz2 real
syms d2Q22_dx2 d2Q22_dy2 d2Q22_dz2 real
syms d2Q23_dx2 d2Q23_dy2 d2Q23_dz2 real
syms d2Q33_dx2 d2Q33_dy2 d2Q33_dz2 real

% Mixed derivatives
syms d2Q11_dxdy d2Q11_dxdz d2Q11_dydz real
syms d2Q12_dxdy d2Q12_dxdz d2Q12_dydz real
syms d2Q13_dxdy d2Q13_dxdz d2Q13_dydz real
syms d2Q22_dxdy d2Q22_dxdz d2Q22_dydz real
syms d2Q23_dxdy d2Q23_dxdz d2Q23_dydz real
syms d2Q33_dxdy d2Q33_dxdz d2Q33_dydz real

% Construct Q-tensor
Q = [Q11, Q12, Q13; Q12, Q22, Q23; Q13, Q23, Q33];

% Elastic constants (as symbolic parameters)
syms L1_sym L2_sym L3_sym L4_sym L6_sym real
syms a_thermo_sym b_thermo_sym c_thermo_sym real

% Define all derivatives as arrays for vectorized operations
dQ_dx = [dQ11_dx, dQ12_dx, dQ13_dx; dQ12_dx, dQ22_dx, dQ23_dx; dQ13_dx, dQ23_dx, dQ33_dx];
dQ_dy = [dQ11_dy, dQ12_dy, dQ13_dy; dQ12_dy, dQ22_dy, dQ23_dy; dQ13_dy, dQ23_dy, dQ33_dy];
dQ_dz = [dQ11_dz, dQ12_dz, dQ13_dz; dQ12_dz, dQ22_dz, dQ23_dz; dQ13_dz, dQ23_dz, dQ33_dz];

% Elastic energy density (simplified version for testing)
% L1 term
F_elastic = (L1_sym/2) * (...
    dQ11_dx^2 + dQ12_dx^2 + dQ13_dx^2 + ...
    dQ11_dy^2 + dQ12_dy^2 + dQ13_dy^2 + ...
    dQ11_dz^2 + dQ12_dz^2 + dQ13_dz^2 + ...
    dQ22_dx^2 + dQ23_dx^2 + dQ33_dx^2 + ...
    dQ22_dy^2 + dQ23_dy^2 + dQ33_dy^2 + ...
    dQ22_dz^2 + dQ23_dz^2 + dQ33_dz^2);

% Thermotropic energy (Landau-de Gennes)
F_thermo = a_thermo_sym * (Q11^2 + 2*Q12^2 + 2*Q13^2 + Q22^2 + 2*Q23^2 + Q33^2) + ...
           (2*b_thermo_sym/3) * (Q11^3 + 3*Q11*(Q12^2 + Q13^2) + 3*Q12^2*Q22 + 6*Q12*Q13*Q23 + ...
           3*Q13^2*Q33 + Q22^3 + 3*Q22*(Q23^2 - Q11*Q22) + 3*Q23^2*Q33 + Q33^3) + ...
           (c_thermo_sym/2) * (Q11^2 + 2*Q12^2 + 2*Q13^2 + Q22^2 + 2*Q23^2 + Q33^2)^2;

% Total free energy density
F_total = F_elastic + F_thermo;

% Compute Euler-Lagrange equations
disp('Computing Euler-Lagrange equations...');
eqn1 = diff(F_total, Q11);
eqn2 = diff(F_total, Q12);
eqn3 = diff(F_total, Q13);
eqn4 = diff(F_total, Q22);
eqn5 = diff(F_total, Q23);

% Convert to MATLAB functions
disp('Converting to function handles...');
f_update_Q11 = matlabFunction(eqn1, 'Vars', {...
    Q11, Q12, Q13, Q22, Q23, Q33, ...
    dQ11_dx, dQ11_dy, dQ11_dz, ...
    dQ12_dx, dQ12_dy, dQ12_dz, ...
    dQ13_dx, dQ13_dy, dQ13_dz, ...
    dQ22_dx, dQ22_dy, dQ22_dz, ...
    dQ23_dx, dQ23_dy, dQ23_dz, ...
    dQ33_dx, dQ33_dy, dQ33_dz, ...
    d2Q11_dx2, d2Q11_dy2, d2Q11_dz2, ...
    d2Q12_dx2, d2Q12_dy2, d2Q12_dz2, ...
    d2Q13_dx2, d2Q13_dy2, d2Q13_dz2, ...
    d2Q22_dx2, d2Q22_dy2, d2Q22_dz2, ...
    d2Q23_dx2, d2Q23_dy2, d2Q23_dz2, ...
    d2Q33_dx2, d2Q33_dy2, d2Q33_dz2, ...
    L1_sym, a_thermo_sym, b_thermo_sym, c_thermo_sym});

f_update_Q12 = matlabFunction(eqn2, 'Vars', {...
    Q11, Q12, Q13, Q22, Q23, Q33, ...
    dQ11_dx, dQ11_dy, dQ11_dz, ...
    dQ12_dx, dQ12_dy, dQ12_dz, ...
    dQ13_dx, dQ13_dy, dQ13_dz, ...
    dQ22_dx, dQ22_dy, dQ22_dz, ...
    dQ23_dx, dQ23_dy, dQ23_dz, ...
    dQ33_dx, dQ33_dy, dQ33_dz, ...
    d2Q11_dx2, d2Q11_dy2, d2Q11_dz2, ...
    d2Q12_dx2, d2Q12_dy2, d2Q12_dz2, ...
    d2Q13_dx2, d2Q13_dy2, d2Q13_dz2, ...
    d2Q22_dx2, d2Q22_dy2, d2Q22_dz2, ...
    d2Q23_dx2, d2Q23_dy2, d2Q23_dz2, ...
    d2Q33_dx2, d2Q33_dy2, d2Q33_dz2, ...
    L1_sym, a_thermo_sym, b_thermo_sym, c_thermo_sym});

f_update_Q13 = matlabFunction(eqn3, 'Vars', {...
    Q11, Q12, Q13, Q22, Q23, Q33, ...
    dQ11_dx, dQ11_dy, dQ11_dz, ...
    dQ12_dx, dQ12_dy, dQ12_dz, ...
    dQ13_dx, dQ13_dy, dQ13_dz, ...
    dQ22_dx, dQ22_dy, dQ22_dz, ...
    dQ23_dx, dQ23_dy, dQ23_dz, ...
    dQ33_dx, dQ33_dy, dQ33_dz, ...
    d2Q11_dx2, d2Q11_dy2, d2Q11_dz2, ...
    d2Q12_dx2, d2Q12_dy2, d2Q12_dz2, ...
    d2Q13_dx2, d2Q13_dy2, d2Q13_dz2, ...
    d2Q22_dx2, d2Q22_dy2, d2Q22_dz2, ...
    d2Q23_dx2, d2Q23_dy2, d2Q23_dz2, ...
    d2Q33_dx2, d2Q33_dy2, d2Q33_dz2, ...
    L1_sym, a_thermo_sym, b_thermo_sym, c_thermo_sym});

f_update_Q22 = matlabFunction(eqn4, 'Vars', {...
    Q11, Q12, Q13, Q22, Q23, Q33, ...
    dQ11_dx, dQ11_dy, dQ11_dz, ...
    dQ12_dx, dQ12_dy, dQ12_dz, ...
    dQ13_dx, dQ13_dy, dQ13_dz, ...
    dQ22_dx, dQ22_dy, dQ22_dz, ...
    dQ23_dx, dQ23_dy, dQ23_dz, ...
    dQ33_dx, dQ33_dy, dQ33_dz, ...
    d2Q11_dx2, d2Q11_dy2, d2Q11_dz2, ...
    d2Q12_dx2, d2Q12_dy2, d2Q12_dz2, ...
    d2Q13_dx2, d2Q13_dy2, d2Q13_dz2, ...
    d2Q22_dx2, d2Q22_dy2, d2Q22_dz2, ...
    d2Q23_dx2, d2Q23_dy2, d2Q23_dz2, ...
    d2Q33_dx2, d2Q33_dy2, d2Q33_dz2, ...
    L1_sym, a_thermo_sym, b_thermo_sym, c_thermo_sym});

f_update_Q23 = matlabFunction(eqn5, 'Vars', {...
    Q11, Q12, Q13, Q22, Q23, Q33, ...
    dQ11_dx, dQ11_dy, dQ11_dz, ...
    dQ12_dx, dQ12_dy, dQ12_dz, ...
    dQ13_dx, dQ13_dy, dQ13_dz, ...
    dQ22_dx, dQ22_dy, dQ22_dz, ...
    dQ23_dx, dQ23_dy, dQ23_dz, ...
    dQ33_dx, dQ33_dy, dQ33_dz, ...
    d2Q11_dx2, d2Q11_dy2, d2Q11_dz2, ...
    d2Q12_dx2, d2Q12_dy2, d2Q12_dz2, ...
    d2Q13_dx2, d2Q13_dy2, d2Q13_dz2, ...
    d2Q22_dx2, d2Q22_dy2, d2Q22_dz2, ...
    d2Q23_dx2, d2Q23_dy2, d2Q23_dz2, ...
    d2Q33_dx2, d2Q33_dy2, d2Q33_dz2, ...
    L1_sym, a_thermo_sym, b_thermo_sym, c_thermo_sym});

disp('Function handles created successfully.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize Q-tensor matrices
if GPU_computation
    Q11_matrix = gpuArray.zeros(N_Y, N_X, N_Z);
    Q12_matrix = gpuArray.zeros(N_Y, N_X, N_Z);
    Q13_matrix = gpuArray.zeros(N_Y, N_X, N_Z);
    Q22_matrix = gpuArray.zeros(N_Y, N_X, N_Z);
    Q23_matrix = gpuArray.zeros(N_Y, N_X, N_Z);
else
    Q11_matrix = zeros(N_Y, N_X, N_Z);
    Q12_matrix = zeros(N_Y, N_X, N_Z);
    Q13_matrix = zeros(N_Y, N_X, N_Z);
    Q22_matrix = zeros(N_Y, N_X, N_Z);
    Q23_matrix = zeros(N_Y, N_X, N_Z);
end

% Create coordinate grids
x_vals = linspace(0, L_X, N_X);
y_vals = linspace(0, L_Y, N_Y);
[X, Y] = meshgrid(x_vals, y_vals);

% Compute the director angle pattern
Lambda = 6.5e-6;
phi_pattern = atan2(cos(pi * X / Lambda), sin(pi * X / Lambda));
phi_pattern = mod(phi_pattern, pi);

% Function to calculate Q from angles
calculate_Q_from_angles = @(theta, phi, psi, S1, S2) deal(...
    S1*cos(theta)^2*cos(phi)^2 + S2*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2), ...
    S1*cos(theta)^2*sin(phi)*cos(phi) - S2*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta)), ...
    S1*sin(theta)*cos(theta)*cos(phi) + S2*sin(psi)*cos(theta)*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta)), ...
    S1*cos(theta)^2*sin(phi)^2 + S2*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2), ...
    S1*cos(theta)*sin(theta)*sin(phi) - S2*sin(psi)*cos(theta)*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SET BOUNDARY AND INITIAL CONDITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Setting boundary conditions...');

% Bottom surface (z = 1)
for i = 1:N_Y
    for j = 1:N_X
        theta_bc = 0;
        phi_bc = phi_pattern(i, j);
        psi_bc = 0;
        S1_bc = 1;
        S2_bc = 0;
        
        [Q11_matrix(i,j,1), Q12_matrix(i,j,1), Q13_matrix(i,j,1), ...
         Q22_matrix(i,j,1), Q23_matrix(i,j,1)] = ...
         calculate_Q_from_angles(theta_bc, phi_bc, psi_bc, S1_bc, S2_bc);
    end
end

% Top surface (z = N_Z)
for i = 1:N_Y
    for j = 1:N_X
        theta_bc = 0;
        phi_bc = phi_pattern(j, i);  % Transposed pattern
        psi_bc = 0;
        S1_bc = 1;
        S2_bc = 0;
        
        [Q11_matrix(i,j,N_Z), Q12_matrix(i,j,N_Z), Q13_matrix(i,j,N_Z), ...
         Q22_matrix(i,j,N_Z), Q23_matrix(i,j,N_Z)] = ...
         calculate_Q_from_angles(theta_bc, phi_bc, psi_bc, S1_bc, S2_bc);
    end
end

% Initialize bulk
for k = 2:N_Z-1
    for i = 1:N_Y
        for j = 1:N_X
            theta_init = 0;
            if k < N_Z/2
                phi_init = phi_pattern(i, j);
            else
                phi_init = phi_pattern(j, i);
            end
            psi_init = 0;
            S1_init = S_exp;
            S2_init = 0;
            
            [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
             Q22_matrix(i,j,k), Q23_matrix(i,j,k)] = ...
             calculate_Q_from_angles(theta_init, phi_init, psi_init, S1_init, S2_init);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PRE-COMPUTE INDICES FOR FAST DERIVATIVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Ny, Nx, Nz] = size(Q11_matrix);

% Create shift indices for finite differences
ip1 = circshift(1:Nx, -1);
im1 = circshift(1:Nx, 1);
jp1 = circshift(1:Ny, -1);
jm1 = circshift(1:Ny, 1);

% For interior points in z
k_interior = 2:Nz-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MAIN ITERATION LOOP - OPTIMIZED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Starting iteration...');
tic;

% Initialize step size
alpha = 1e-8;  % Small step size for stability

% Convergence tracking
convergence_history = zeros(min(max_iter, 1000), 1);
energy_history = zeros(min(max_iter, 1000), 1);

for iter = 1:max_iter
    % Store old values for convergence check
    Q11_old = Q11_matrix;
    Q12_old = Q12_matrix;
    Q13_old = Q13_matrix;
    Q22_old = Q22_matrix;
    Q23_old = Q23_matrix;
    
    % Compute Q33 from traceless condition
    Q33_matrix = -(Q11_matrix + Q22_matrix);
    
    % Compute all derivatives using the function (defined at the end)
    derivs = compute_derivatives(Q11_matrix, Q12_matrix, Q13_matrix, ...
                                Q22_matrix, Q23_matrix, Q33_matrix, ...
                                ip1, im1, jp1, jm1, k_interior, ...
                                dx_inv, dy_inv, dz_inv, ...
                                dx2_inv, dy2_inv, dz2_inv, ...
                                dxdy_inv, dxdz_inv, dydz_inv);
    
    % Initialize update arrays
    update_Q11 = zeros(Ny, Nx, Nz);
    update_Q12 = zeros(Ny, Nx, Nz);
    update_Q13 = zeros(Ny, Nx, Nz);
    update_Q22 = zeros(Ny, Nx, Nz);
    update_Q23 = zeros(Ny, Nx, Nz);
    
    % Compute updates for all interior points
    for k = 2:Nz-1
        for j = 1:Nx
            for i = 1:Ny
                % Q11 update
                update_Q11(i,j,k) = f_update_Q11(...
                    Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
                    Q22_matrix(i,j,k), Q23_matrix(i,j,k), Q33_matrix(i,j,k), ...
                    derivs.dQ11_dx(i,j,k), derivs.dQ11_dy(i,j,k), derivs.dQ11_dz(i,j,k), ...
                    derivs.dQ12_dx(i,j,k), derivs.dQ12_dy(i,j,k), derivs.dQ12_dz(i,j,k), ...
                    derivs.dQ13_dx(i,j,k), derivs.dQ13_dy(i,j,k), derivs.dQ13_dz(i,j,k), ...
                    derivs.dQ22_dx(i,j,k), derivs.dQ22_dy(i,j,k), derivs.dQ22_dz(i,j,k), ...
                    derivs.dQ23_dx(i,j,k), derivs.dQ23_dy(i,j,k), derivs.dQ23_dz(i,j,k), ...
                    derivs.dQ33_dx(i,j,k), derivs.dQ33_dy(i,j,k), derivs.dQ33_dz(i,j,k), ...
                    derivs.d2Q11_dx2(i,j,k), derivs.d2Q11_dy2(i,j,k), derivs.d2Q11_dz2(i,j,k), ...
                    derivs.d2Q12_dx2(i,j,k), derivs.d2Q12_dy2(i,j,k), derivs.d2Q12_dz2(i,j,k), ...
                    derivs.d2Q13_dx2(i,j,k), derivs.d2Q13_dy2(i,j,k), derivs.d2Q13_dz2(i,j,k), ...
                    derivs.d2Q22_dx2(i,j,k), derivs.d2Q22_dy2(i,j,k), derivs.d2Q22_dz2(i,j,k), ...
                    derivs.d2Q23_dx2(i,j,k), derivs.d2Q23_dy2(i,j,k), derivs.d2Q23_dz2(i,j,k), ...
                    derivs.d2Q33_dx2(i,j,k), derivs.d2Q33_dy2(i,j,k), derivs.d2Q33_dz2(i,j,k), ...
                    L1, a_thermo, b_thermo, c_thermo);
                
                % Q12 update
                update_Q12(i,j,k) = f_update_Q12(...
                    Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
                    Q22_matrix(i,j,k), Q23_matrix(i,j,k), Q33_matrix(i,j,k), ...
                    derivs.dQ11_dx(i,j,k), derivs.dQ11_dy(i,j,k), derivs.dQ11_dz(i,j,k), ...
                    derivs.dQ12_dx(i,j,k), derivs.dQ12_dy(i,j,k), derivs.dQ12_dz(i,j,k), ...
                    derivs.dQ13_dx(i,j,k), derivs.dQ13_dy(i,j,k), derivs.dQ13_dz(i,j,k), ...
                    derivs.dQ22_dx(i,j,k), derivs.dQ22_dy(i,j,k), derivs.dQ22_dz(i,j,k), ...
                    derivs.dQ23_dx(i,j,k), derivs.dQ23_dy(i,j,k), derivs.dQ23_dz(i,j,k), ...
                    derivs.dQ33_dx(i,j,k), derivs.dQ33_dy(i,j,k), derivs.dQ33_dz(i,j,k), ...
                    derivs.d2Q11_dx2(i,j,k), derivs.d2Q11_dy2(i,j,k), derivs.d2Q11_dz2(i,j,k), ...
                    derivs.d2Q12_dx2(i,j,k), derivs.d2Q12_dy2(i,j,k), derivs.d2Q12_dz2(i,j,k), ...
                    derivs.d2Q13_dx2(i,j,k), derivs.d2Q13_dy2(i,j,k), derivs.d2Q13_dz2(i,j,k), ...
                    derivs.d2Q22_dx2(i,j,k), derivs.d2Q22_dy2(i,j,k), derivs.d2Q22_dz2(i,j,k), ...
                    derivs.d2Q23_dx2(i,j,k), derivs.d2Q23_dy2(i,j,k), derivs.d2Q23_dz2(i,j,k), ...
                    derivs.d2Q33_dx2(i,j,k), derivs.d2Q33_dy2(i,j,k), derivs.d2Q33_dz2(i,j,k), ...
                    L1, a_thermo, b_thermo, c_thermo);
                
                % Q13 update
                update_Q13(i,j,k) = f_update_Q13(...
                    Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
                    Q22_matrix(i,j,k), Q23_matrix(i,j,k), Q33_matrix(i,j,k), ...
                    derivs.dQ11_dx(i,j,k), derivs.dQ11_dy(i,j,k), derivs.dQ11_dz(i,j,k), ...
                    derivs.dQ12_dx(i,j,k), derivs.dQ12_dy(i,j,k), derivs.dQ12_dz(i,j,k), ...
                    derivs.dQ13_dx(i,j,k), derivs.dQ13_dy(i,j,k), derivs.dQ13_dz(i,j,k), ...
                    derivs.dQ22_dx(i,j,k), derivs.dQ22_dy(i,j,k), derivs.dQ22_dz(i,j,k), ...
                    derivs.dQ23_dx(i,j,k), derivs.dQ23_dy(i,j,k), derivs.dQ23_dz(i,j,k), ...
                    derivs.dQ33_dx(i,j,k), derivs.dQ33_dy(i,j,k), derivs.dQ33_dz(i,j,k), ...
                    derivs.d2Q11_dx2(i,j,k), derivs.d2Q11_dy2(i,j,k), derivs.d2Q11_dz2(i,j,k), ...
                    derivs.d2Q12_dx2(i,j,k), derivs.d2Q12_dy2(i,j,k), derivs.d2Q12_dz2(i,j,k), ...
                    derivs.d2Q13_dx2(i,j,k), derivs.d2Q13_dy2(i,j,k), derivs.d2Q13_dz2(i,j,k), ...
                    derivs.d2Q22_dx2(i,j,k), derivs.d2Q22_dy2(i,j,k), derivs.d2Q22_dz2(i,j,k), ...
                    derivs.d2Q23_dx2(i,j,k), derivs.d2Q23_dy2(i,j,k), derivs.d2Q23_dz2(i,j,k), ...
                    derivs.d2Q33_dx2(i,j,k), derivs.d2Q33_dy2(i,j,k), derivs.d2Q33_dz2(i,j,k), ...
                    L1, a_thermo, b_thermo, c_thermo);
                
                % Q22 update
                update_Q22(i,j,k) = f_update_Q22(...
                    Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
                    Q22_matrix(i,j,k), Q23_matrix(i,j,k), Q33_matrix(i,j,k), ...
                    derivs.dQ11_dx(i,j,k), derivs.dQ11_dy(i,j,k), derivs.dQ11_dz(i,j,k), ...
                    derivs.dQ12_dx(i,j,k), derivs.dQ12_dy(i,j,k), derivs.dQ12_dz(i,j,k), ...
                    derivs.dQ13_dx(i,j,k), derivs.dQ13_dy(i,j,k), derivs.dQ13_dz(i,j,k), ...
                    derivs.dQ22_dx(i,j,k), derivs.dQ22_dy(i,j,k), derivs.dQ22_dz(i,j,k), ...
                    derivs.dQ23_dx(i,j,k), derivs.dQ23_dy(i,j,k), derivs.dQ23_dz(i,j,k), ...
                    derivs.dQ33_dx(i,j,k), derivs.dQ33_dy(i,j,k), derivs.dQ33_dz(i,j,k), ...
                    derivs.d2Q11_dx2(i,j,k), derivs.d2Q11_dy2(i,j,k), derivs.d2Q11_dz2(i,j,k), ...
                    derivs.d2Q12_dx2(i,j,k), derivs.d2Q12_dy2(i,j,k), derivs.d2Q12_dz2(i,j,k), ...
                    derivs.d2Q13_dx2(i,j,k), derivs.d2Q13_dy2(i,j,k), derivs.d2Q13_dz2(i,j,k), ...
                    derivs.d2Q22_dx2(i,j,k), derivs.d2Q22_dy2(i,j,k), derivs.d2Q22_dz2(i,j,k), ...
                    derivs.d2Q23_dx2(i,j,k), derivs.d2Q23_dy2(i,j,k), derivs.d2Q23_dz2(i,j,k), ...
                    derivs.d2Q33_dx2(i,j,k), derivs.d2Q33_dy2(i,j,k), derivs.d2Q33_dz2(i,j,k), ...
                    L1, a_thermo, b_thermo, c_thermo);
                
                % Q23 update
                update_Q23(i,j,k) = f_update_Q23(...
                    Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k), ...
                    Q22_matrix(i,j,k), Q23_matrix(i,j,k), Q33_matrix(i,j,k), ...
                    derivs.dQ11_dx(i,j,k), derivs.dQ11_dy(i,j,k), derivs.dQ11_dz(i,j,k), ...
                    derivs.dQ12_dx(i,j,k), derivs.dQ12_dy(i,j,k), derivs.dQ12_dz(i,j,k), ...
                    derivs.dQ13_dx(i,j,k), derivs.dQ13_dy(i,j,k), derivs.dQ13_dz(i,j,k), ...
                    derivs.dQ22_dx(i,j,k), derivs.dQ22_dy(i,j,k), derivs.dQ22_dz(i,j,k), ...
                    derivs.dQ23_dx(i,j,k), derivs.dQ23_dy(i,j,k), derivs.dQ23_dz(i,j,k), ...
                    derivs.dQ33_dx(i,j,k), derivs.dQ33_dy(i,j,k), derivs.dQ33_dz(i,j,k), ...
                    derivs.d2Q11_dx2(i,j,k), derivs.d2Q11_dy2(i,j,k), derivs.d2Q11_dz2(i,j,k), ...
                    derivs.d2Q12_dx2(i,j,k), derivs.d2Q12_dy2(i,j,k), derivs.d2Q12_dz2(i,j,k), ...
                    derivs.d2Q13_dx2(i,j,k), derivs.d2Q13_dy2(i,j,k), derivs.d2Q13_dz2(i,j,k), ...
                    derivs.d2Q22_dx2(i,j,k), derivs.d2Q22_dy2(i,j,k), derivs.d2Q22_dz2(i,j,k), ...
                    derivs.d2Q23_dx2(i,j,k), derivs.d2Q23_dy2(i,j,k), derivs.d2Q23_dz2(i,j,k), ...
                    derivs.d2Q33_dx2(i,j,k), derivs.d2Q33_dy2(i,j,k), derivs.d2Q33_dz2(i,j,k), ...
                    L1, a_thermo, b_thermo, c_thermo);
            end
        end
    end
    
    % Apply updates
    Q11_matrix = Q11_matrix - alpha * update_Q11;
    Q12_matrix = Q12_matrix - alpha * update_Q12;
    Q13_matrix = Q13_matrix - alpha * update_Q13;
    Q22_matrix = Q22_matrix - alpha * update_Q22;
    Q23_matrix = Q23_matrix - alpha * update_Q23;
    
    % Apply boundary conditions (keep boundaries fixed)
    Q11_matrix(:,:,[1 Nz]) = Q11_old(:,:,[1 Nz]);
    Q12_matrix(:,:,[1 Nz]) = Q12_old(:,:,[1 Nz]);
    Q13_matrix(:,:,[1 Nz]) = Q13_old(:,:,[1 Nz]);
    Q22_matrix(:,:,[1 Nz]) = Q22_old(:,:,[1 Nz]);
    Q23_matrix(:,:,[1 Nz]) = Q23_old(:,:,[1 Nz]);
    
    % Check convergence
    max_change = max([...
        max(abs(Q11_matrix(:) - Q11_old(:))), ...
        max(abs(Q12_matrix(:) - Q12_old(:))), ...
        max(abs(Q13_matrix(:) - Q13_old(:))), ...
        max(abs(Q22_matrix(:) - Q22_old(:))), ...
        max(abs(Q23_matrix(:) - Q23_old(:)))]);
    
    convergence_history(min(iter, end)) = max_change;
    
    % Adaptive step size
    if iter > 10
        if max_change < tol * 10
            alpha = min(alpha * 1.05, 1e-5);  % Increase step if converging well
        elseif max_change > tol * 100
            alpha = max(alpha * 0.95, 1e-10);  % Decrease if oscillating
        end
    end
    
    % Display progress
    if mod(iter, 100) == 0
        elapsed = toc;
        fprintf('Iter %d: max change = %.2e, alpha = %.2e, time = %.1f s\n', ...
                iter, max_change, alpha, elapsed);
        tic;
    end
    
    % Check convergence
    if max_change < tol
        fprintf('Converged after %d iterations with max change = %.2e\n', ...
                iter, max_change);
        break;
    end
    
    % Save checkpoint
    if mod(iter, 1000) == 0
        save(sprintf('checkpoint_%s_iter%d.mat', Simulation_name, iter), ...
             'Q11_matrix', 'Q12_matrix', 'Q13_matrix', ...
             'Q22_matrix', 'Q23_matrix', 'alpha', 'iter');
    end
end

total_time = toc;
fprintf('Total simulation time: %.1f seconds\n', total_time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   POST-PROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save final results
save(sprintf('final_%s.mat', Simulation_name), ...
     'Q11_matrix', 'Q12_matrix', 'Q13_matrix', 'Q22_matrix', 'Q23_matrix', ...
     'L1', 'L2', 'L3', 'L4', 'L6', 'convergence_history', ...
     'N_X', 'N_Y', 'N_Z', 'dx', 'dy', 'dz');

% Extract director field
disp('Extracting director field...');
u_final = zeros(Ny, Nx, Nz);
v_final = zeros(Ny, Nx, Nz);
w_final = zeros(Ny, Nx, Nz);
theta_matrix = zeros(Ny, Nx, Nz);
phi_matrix = zeros(Ny, Nx, Nz);
S_final = zeros(Ny, Nx, Nz);

for k = 1:Nz
    for j = 1:Nx
        for i = 1:Ny
            Q_local = [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k);
                       Q12_matrix(i,j,k), Q22_matrix(i,j,k), Q23_matrix(i,j,k);
                       Q13_matrix(i,j,k), Q23_matrix(i,j,k), -(Q11_matrix(i,j,k) + Q22_matrix(i,j,k))];
            
            [V, D] = eig(Q_local);
            eigenvalues = diag(D);
            [~, idx] = max(eigenvalues);
            n = V(:, idx);
            n = n / norm(n);
            
            if n(3) < 0
                n = -n;
            end
            
            u_final(i,j,k) = n(1);
            v_final(i,j,k) = n(2);
            w_final(i,j,k) = n(3);
            theta_matrix(i,j,k) = acos(n(3));
            phi_matrix(i,j,k) = atan2(n(2), n(1));
            S_final(i,j,k) = (3/2) * max(eigenvalues);
        end
    end
end

% Save director field
save(sprintf('director_%s.mat', Simulation_name), ...
     'u_final', 'v_final', 'w_final', 'theta_matrix', 'phi_matrix', 'S_final');

% Create simple visualization
figure('Position', [100, 100, 1200, 500]);

% Plot director field at middle slice
mid_z = round(Nz/2);
subplot(1,3,1);
quiver(X, Y, u_final(:,:,mid_z), v_final(:,:,mid_z), 0.5);
axis equal tight;
title(sprintf('Director Field at z=%d', mid_z));
xlabel('x (μm)'); ylabel('y (μm)');

% Plot scalar order parameter
subplot(1,3,2);
imagesc(x_vals*1e6, y_vals*1e6, S_final(:,:,mid_z));
axis equal tight;
colorbar;
title('Scalar Order Parameter S');
xlabel('x (μm)'); ylabel('y (μm)');

% Plot convergence history
subplot(1,3,3);
semilogy(convergence_history(convergence_history > 0), 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Max Change');
title('Convergence History');
grid on;

disp('Simulation complete!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   FUNCTION DEFINITIONS - MUST BE AT THE END OF THE FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function derivs = compute_derivatives(Q11, Q12, Q13, Q22, Q23, Q33, ...
                                     ip1, im1, jp1, jm1, k_interior, ...
                                     dx_inv, dy_inv, dz_inv, ...
                                     dx2_inv, dy2_inv, dz2_inv, ...
                                     dxdy_inv, dxdz_inv, dydz_inv)
    % Compute all derivatives for all Q-tensor components
    
    [Ny, Nx, Nz] = size(Q11);
    derivs = struct();
    
    % Q11 Derivatives
    derivs.dQ11_dx = (Q11(:, ip1, :) - Q11(:, im1, :)) * dx_inv;
    derivs.dQ11_dy = (Q11(jp1, :, :) - Q11(jm1, :, :)) * dy_inv;
    derivs.dQ11_dz = zeros(Ny, Nx, Nz);
    derivs.dQ11_dz(:, :, k_interior) = (Q11(:, :, k_interior+1) - Q11(:, :, k_interior-1)) * dz_inv;
    
    derivs.d2Q11_dx2 = (Q11(:, ip1, :) - 2*Q11 + Q11(:, im1, :)) * dx2_inv;
    derivs.d2Q11_dy2 = (Q11(jp1, :, :) - 2*Q11 + Q11(jm1, :, :)) * dy2_inv;
    derivs.d2Q11_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q11_dz2(:, :, k_interior) = (Q11(:, :, k_interior+1) - 2*Q11(:, :, k_interior) + Q11(:, :, k_interior-1)) * dz2_inv;
    
    derivs.d2Q11_dxdy = (Q11(jp1, ip1, :) - Q11(jp1, im1, :) - Q11(jm1, ip1, :) + Q11(jm1, im1, :)) * dxdy_inv;
    derivs.d2Q11_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q11_dxdz(:, :, k_interior) = (Q11(:, ip1, k_interior+1) - Q11(:, ip1, k_interior-1) - ...
                                          Q11(:, im1, k_interior+1) + Q11(:, im1, k_interior-1)) * dxdz_inv;
    derivs.d2Q11_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q11_dydz(:, :, k_interior) = (Q11(jp1, :, k_interior+1) - Q11(jp1, :, k_interior-1) - ...
                                          Q11(jm1, :, k_interior+1) + Q11(jm1, :, k_interior-1)) * dydz_inv;
    
    % Q12 Derivatives
    derivs.dQ12_dx = (Q12(:, ip1, :) - Q12(:, im1, :)) * dx_inv;
    derivs.dQ12_dy = (Q12(jp1, :, :) - Q12(jm1, :, :)) * dy_inv;
    derivs.dQ12_dz = zeros(Ny, Nx, Nz);
    derivs.dQ12_dz(:, :, k_interior) = (Q12(:, :, k_interior+1) - Q12(:, :, k_interior-1)) * dz_inv;
    
    derivs.d2Q12_dx2 = (Q12(:, ip1, :) - 2*Q12 + Q12(:, im1, :)) * dx2_inv;
    derivs.d2Q12_dy2 = (Q12(jp1, :, :) - 2*Q12 + Q12(jm1, :, :)) * dy2_inv;
    derivs.d2Q12_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q12_dz2(:, :, k_interior) = (Q12(:, :, k_interior+1) - 2*Q12(:, :, k_interior) + Q12(:, :, k_interior-1)) * dz2_inv;
    
    derivs.d2Q12_dxdy = (Q12(jp1, ip1, :) - Q12(jp1, im1, :) - Q12(jm1, ip1, :) + Q12(jm1, im1, :)) * dxdy_inv;
    derivs.d2Q12_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q12_dxdz(:, :, k_interior) = (Q12(:, ip1, k_interior+1) - Q12(:, ip1, k_interior-1) - ...
                                          Q12(:, im1, k_interior+1) + Q12(:, im1, k_interior-1)) * dxdz_inv;
    derivs.d2Q12_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q12_dydz(:, :, k_interior) = (Q12(jp1, :, k_interior+1) - Q12(jp1, :, k_interior-1) - ...
                                          Q12(jm1, :, k_interior+1) + Q12(jm1, :, k_interior-1)) * dydz_inv;
    
    % Q13 Derivatives
    derivs.dQ13_dx = (Q13(:, ip1, :) - Q13(:, im1, :)) * dx_inv;
    derivs.dQ13_dy = (Q13(jp1, :, :) - Q13(jm1, :, :)) * dy_inv;
    derivs.dQ13_dz = zeros(Ny, Nx, Nz);
    derivs.dQ13_dz(:, :, k_interior) = (Q13(:, :, k_interior+1) - Q13(:, :, k_interior-1)) * dz_inv;
    
    derivs.d2Q13_dx2 = (Q13(:, ip1, :) - 2*Q13 + Q13(:, im1, :)) * dx2_inv;
    derivs.d2Q13_dy2 = (Q13(jp1, :, :) - 2*Q13 + Q13(jm1, :, :)) * dy2_inv;
    derivs.d2Q13_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q13_dz2(:, :, k_interior) = (Q13(:, :, k_interior+1) - 2*Q13(:, :, k_interior) + Q13(:, :, k_interior-1)) * dz2_inv;
    
    derivs.d2Q13_dxdy = (Q13(jp1, ip1, :) - Q13(jp1, im1, :) - Q13(jm1, ip1, :) + Q13(jm1, im1, :)) * dxdy_inv;
    derivs.d2Q13_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q13_dxdz(:, :, k_interior) = (Q13(:, ip1, k_interior+1) - Q13(:, ip1, k_interior-1) - ...
                                          Q13(:, im1, k_interior+1) + Q13(:, im1, k_interior-1)) * dxdz_inv;
    derivs.d2Q13_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q13_dydz(:, :, k_interior) = (Q13(jp1, :, k_interior+1) - Q13(jp1, :, k_interior-1) - ...
                                          Q13(jm1, :, k_interior+1) + Q13(jm1, :, k_interior-1)) * dydz_inv;
    
    % Q22 Derivatives
    derivs.dQ22_dx = (Q22(:, ip1, :) - Q22(:, im1, :)) * dx_inv;
    derivs.dQ22_dy = (Q22(jp1, :, :) - Q22(jm1, :, :)) * dy_inv;
    derivs.dQ22_dz = zeros(Ny, Nx, Nz);
    derivs.dQ22_dz(:, :, k_interior) = (Q22(:, :, k_interior+1) - Q22(:, :, k_interior-1)) * dz_inv;
    
    derivs.d2Q22_dx2 = (Q22(:, ip1, :) - 2*Q22 + Q22(:, im1, :)) * dx2_inv;
    derivs.d2Q22_dy2 = (Q22(jp1, :, :) - 2*Q22 + Q22(jm1, :, :)) * dy2_inv;
    derivs.d2Q22_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q22_dz2(:, :, k_interior) = (Q22(:, :, k_interior+1) - 2*Q22(:, :, k_interior) + Q22(:, :, k_interior-1)) * dz2_inv;
    
    derivs.d2Q22_dxdy = (Q22(jp1, ip1, :) - Q22(jp1, im1, :) - Q22(jm1, ip1, :) + Q22(jm1, im1, :)) * dxdy_inv;
    derivs.d2Q22_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q22_dxdz(:, :, k_interior) = (Q22(:, ip1, k_interior+1) - Q22(:, ip1, k_interior-1) - ...
                                          Q22(:, im1, k_interior+1) + Q22(:, im1, k_interior-1)) * dxdz_inv;
    derivs.d2Q22_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q22_dydz(:, :, k_interior) = (Q22(jp1, :, k_interior+1) - Q22(jp1, :, k_interior-1) - ...
                                          Q22(jm1, :, k_interior+1) + Q22(jm1, :, k_interior-1)) * dydz_inv;
    
    % Q23 Derivatives
    derivs.dQ23_dx = (Q23(:, ip1, :) - Q23(:, im1, :)) * dx_inv;
    derivs.dQ23_dy = (Q23(jp1, :, :) - Q23(jm1, :, :)) * dy_inv;
    derivs.dQ23_dz = zeros(Ny, Nx, Nz);
    derivs.dQ23_dz(:, :, k_interior) = (Q23(:, :, k_interior+1) - Q23(:, :, k_interior-1)) * dz_inv;
    
    derivs.d2Q23_dx2 = (Q23(:, ip1, :) - 2*Q23 + Q23(:, im1, :)) * dx2_inv;
    derivs.d2Q23_dy2 = (Q23(jp1, :, :) - 2*Q23 + Q23(jm1, :, :)) * dy2_inv;
    derivs.d2Q23_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q23_dz2(:, :, k_interior) = (Q23(:, :, k_interior+1) - 2*Q23(:, :, k_interior) + Q23(:, :, k_interior-1)) * dz2_inv;
    
    derivs.d2Q23_dxdy = (Q23(jp1, ip1, :) - Q23(jp1, im1, :) - Q23(jm1, ip1, :) + Q23(jm1, im1, :)) * dxdy_inv;
    derivs.d2Q23_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q23_dxdz(:, :, k_interior) = (Q23(:, ip1, k_interior+1) - Q23(:, ip1, k_interior-1) - ...
                                          Q23(:, im1, k_interior+1) + Q23(:, im1, k_interior-1)) * dxdz_inv;
    derivs.d2Q23_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q23_dydz(:, :, k_interior) = (Q23(jp1, :, k_interior+1) - Q23(jp1, :, k_interior-1) - ...
                                          Q23(jm1, :, k_interior+1) + Q23(jm1, :, k_interior-1)) * dydz_inv;
    
    % Q33 Derivatives (computed from Q11 and Q22)
    derivs.dQ33_dx = -(derivs.dQ11_dx + derivs.dQ22_dx);
    derivs.dQ33_dy = -(derivs.dQ11_dy + derivs.dQ22_dy);
    derivs.dQ33_dz = -(derivs.dQ11_dz + derivs.dQ22_dz);
    
    derivs.d2Q33_dx2 = -(derivs.d2Q11_dx2 + derivs.d2Q22_dx2);
    derivs.d2Q33_dy2 = -(derivs.d2Q11_dy2 + derivs.d2Q22_dy2);
    derivs.d2Q33_dz2 = zeros(Ny, Nx, Nz);
    derivs.d2Q33_dz2(:, :, k_interior) = -(derivs.d2Q11_dz2(:, :, k_interior) + derivs.d2Q22_dz2(:, :, k_interior));
    
    derivs.d2Q33_dxdy = -(derivs.d2Q11_dxdy + derivs.d2Q22_dxdy);
    derivs.d2Q33_dxdz = zeros(Ny, Nx, Nz);
    derivs.d2Q33_dxdz(:, :, k_interior) = -(derivs.d2Q11_dxdz(:, :, k_interior) + derivs.d2Q22_dxdz(:, :, k_interior));
    derivs.d2Q33_dydz = zeros(Ny, Nx, Nz);
    derivs.d2Q33_dydz(:, :, k_interior) = -(derivs.d2Q11_dydz(:, :, k_interior) + derivs.d2Q22_dydz(:, :, k_interior));
end