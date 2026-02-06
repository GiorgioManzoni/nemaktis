%% efficientcode.m
clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATERIAL PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation_name = 'Sin_2pi_HalfInitialCondition_iN_XY';
Simulation_name = 'cazzo';
K11 = 1.1e-11; K22 = 0.65e-11; K33 = 1.7e-11; K24 = 0.4e-11;
S_exp = 0.1; S_mes = 0.9; q0 = 0;

a_thermo = -1740; b_thermo = -21200; c_thermo = 17400;

% Elastic constants
L1 = (1/(6*S_mes^2))*(K33-K11+3*K22);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GRID DEFINITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L_X = 13e-6; L_Y = 13e-6; Lz = 3e-6;
N_X = 50; N_Y = 50; N_Z = 25;

dx = L_X/(N_X-1); dy = L_Y/(N_Y-1); dz = Lz/(N_Z-1);
dx2_inv = 1/dx^2; dy2_inv = 1/dy^2; dz2_inv = 1/dz^2;

x_vals = linspace(0,L_X,N_X);
y_vals = linspace(0,L_Y,N_Y);
[X,Y] = meshgrid(x_vals,y_vals);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIAL CONDITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lambda = 6.5e-6;
phi_pattern = mod(atan2(cos(pi*X/Lambda), sin(pi*X/Lambda)), pi);

% Initialize Q-tensors
Q11 = zeros(N_Y,N_X,N_Z); Q12 = zeros(N_Y,N_X,N_Z);
Q13 = zeros(N_Y,N_X,N_Z); Q22 = zeros(N_Y,N_X,N_Z);
Q23 = zeros(N_Y,N_X,N_Z);

% Function to calculate Q from angles
calculate_Q_from_angles = @(theta, phi, psi, S1, S2) deal(...
    S1*cos(theta)^2*cos(phi)^2 + S2*(sin(phi)*cos(psi)-cos(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1+S2), ...
    S1*cos(theta)^2*sin(phi)*cos(phi) - S2*(cos(phi)*cos(psi)+sin(phi)*sin(psi)*sin(theta))*(sin(phi)*cos(psi)-cos(phi)*sin(psi)*sin(theta)), ...
    S1*sin(theta)*cos(theta)*cos(phi) + S2*sin(psi)*cos(theta)*(sin(phi)*cos(psi)-cos(phi)*sin(psi)*sin(theta)), ...
    S1*cos(theta)^2*sin(phi)^2 + S2*(cos(phi)*cos(psi)+sin(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1+S2), ...
    S1*cos(theta)*sin(theta)*sin(phi) - S2*sin(psi)*cos(theta)*(cos(phi)*cos(psi)+sin(phi)*sin(psi)*sin(theta)));

% Bottom boundary
for i=1:N_Y
    for j=1:N_X
        [Q11(i,j,1), Q12(i,j,1), Q13(i,j,1), Q22(i,j,1), Q23(i,j,1)] = ...
            calculate_Q_from_angles(0, phi_pattern(i,j), 0, 1, 0);
    end
end

% Top boundary
for i=1:N_Y
    for j=1:N_X
        [Q11(i,j,N_Z), Q12(i,j,N_Z), Q13(i,j,N_Z), Q22(i,j,N_Z), Q23(i,j,N_Z)] = ...
            calculate_Q_from_angles(0, phi_pattern(j,i), 0, 1, 0);
    end
end

% Initialize bulk
for k=2:N_Z-1
    for i=1:N_Y
        for j=1:N_X
            phi_init = (k<N_Z/2)*phi_pattern(i,j) + (k>=N_Z/2)*phi_pattern(j,i);
            [Q11(i,j,k), Q12(i,j,k), Q13(i,j,k), Q22(i,j,k), Q23(i,j,k)] = ...
                calculate_Q_from_angles(0, phi_init, 0, S_exp, 0);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRE-COMPUTE INDICES FOR FINITE DIFFERENCES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip1 = [2:N_X 1]; im1 = [N_X 1:N_X-1];
jp1 = [2:N_Y 1]; jm1 = [N_Y 1:N_Y-1];
k_interior = 2:N_Z-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN ITERATION LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = 10000; tol = 1e-3; alpha = 1e-8;

disp('Starting iteration...');
tic;

for iter=1:max_iter
    Q33 = -(Q11+Q22);
    
    % Compute Laplacians
    % Compute 3D Laplacian using circshift (handles edges automatically if needed)
LapQ11 = (circshift(Q11,[1 0 0]) + circshift(Q11,[-1 0 0]) + ...
          circshift(Q11,[0 1 0]) + circshift(Q11,[0 -1 0]) + ...
          circshift(Q11,[0 0 1]) + circshift(Q11,[0 0 -1]) - 6*Q11) / dx^2;

LapQ12 = (circshift(Q12,[1 0 0]) + circshift(Q12,[-1 0 0]) + ...
          circshift(Q12,[0 1 0]) + circshift(Q12,[0 -1 0]) + ...
          circshift(Q12,[0 0 1]) + circshift(Q12,[0 0 -1]) - 6*Q12) / dx^2;

LapQ13 = (circshift(Q13,[1 0 0]) + circshift(Q13,[-1 0 0]) + ...
          circshift(Q13,[0 1 0]) + circshift(Q13,[0 -1 0]) + ...
          circshift(Q13,[0 0 1]) + circshift(Q13,[0 0 -1]) - 6*Q13) / dx^2;

LapQ22 = (circshift(Q22,[1 0 0]) + circshift(Q22,[-1 0 0]) + ...
          circshift(Q22,[0 1 0]) + circshift(Q22,[0 -1 0]) + ...
          circshift(Q22,[0 0 1]) + circshift(Q22,[0 0 -1]) - 6*Q22) / dx^2;

LapQ23 = (circshift(Q23,[1 0 0]) + circshift(Q23,[-1 0 0]) + ...
          circshift(Q23,[0 1 0]) + circshift(Q23,[0 -1 0]) + ...
          circshift(Q23,[0 0 1]) + circshift(Q23,[0 0 -1]) - 6*Q23) / dx^2;


    % Thermotropic derivatives
    dF11 = 2*a_thermo*Q11 + 2*c_thermo*Q11.*(Q11.^2+Q12.^2+Q13.^2+Q22.^2+Q23.^2+Q33.^2);
    dF12 = 2*a_thermo*Q12 + 2*c_thermo*Q12.*(Q11.^2+Q12.^2+Q13.^2+Q22.^2+Q23.^2+Q33.^2);
    dF13 = 2*a_thermo*Q13 + 2*c_thermo*Q13.*(Q11.^2+Q12.^2+Q13.^2+Q22.^2+Q23.^2+Q33.^2);
    dF22 = 2*a_thermo*Q22 + 2*c_thermo*Q22.*(Q11.^2+Q12.^2+Q13.^2+Q22.^2+Q23.^2+Q33.^2);
    dF23 = 2*a_thermo*Q23 + 2*c_thermo*Q23.*(Q11.^2+Q12.^2+Q13.^2+Q22.^2+Q23.^2+Q33.^2);
    
    % Update only interior points
    Q11(:,:,k_interior) = Q11(:,:,k_interior) - alpha*(L1*LapQ11(:,:,k_interior) + dF11(:,:,k_interior));
    Q12(:,:,k_interior) = Q12(:,:,k_interior) - alpha*(L1*LapQ12(:,:,k_interior) + dF12(:,:,k_interior));
    Q13(:,:,k_interior) = Q13(:,:,k_interior) - alpha*(L1*LapQ13(:,:,k_interior) + dF13(:,:,k_interior));
    Q22(:,:,k_interior) = Q22(:,:,k_interior) - alpha*(L1*LapQ22(:,:,k_interior) + dF22(:,:,k_interior));
    Q23(:,:,k_interior) = Q23(:,:,k_interior) - alpha*(L1*LapQ23(:,:,k_interior) + dF23(:,:,k_interior));
    
    % Check convergence
    max_change = max(abs([dF11(:);dF12(:);dF13(:);dF22(:);dF23(:)]));
    if mod(iter,100)==0
        fprintf('Iter %d: max change = %.2e\n', iter, max_change);
    end
    if max_change<tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

toc;
disp('Simulation finished.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   POST-PROCESSING (COPIED FROM PREVIOUS NOT SURE IF SAME NOTATION)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save final results
save(sprintf('final_%s.mat', Simulation_name), ...
     'Q11', 'Q12', 'Q13', 'Q22', 'Q23');

% Extract director field
disp('Extracting director field...');
u_final = zeros(N_Y, N_X, N_Z);
v_final = zeros(N_Y, N_X, N_Z);
w_final = zeros(N_Y, N_X, N_Z);
theta_matrix = zeros(N_Y, N_X, N_Z);
phi_matrix = zeros(N_Y, N_X, N_Z);
S_final = zeros(N_Y, N_X, N_Z);

for k = 1:N_Z
    for j = 1:N_X
        for i = 1:N_Y
            Q_local = [Q11(i,j,k), Q12(i,j,k), Q13(i,j,k);
                       Q12(i,j,k), Q22(i,j,k), Q23(i,j,k);
                       Q13(i,j,k), Q23(i,j,k), -(Q11(i,j,k) + Q22(i,j,k))];
            
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

disp('saving director');
disp(Simulation_name)
% Save director field
save(sprintf('director_%s.mat', Simulation_name), ...
     'u_final', 'v_final', 'w_final', 'theta_matrix', 'phi_matrix', 'S_final');

disp('finish everything'); 