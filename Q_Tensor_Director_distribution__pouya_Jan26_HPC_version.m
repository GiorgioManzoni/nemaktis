%=======================================================================%
%                                                                       %
%  3D Q-Tensor Calculation (Jan 2026 update) HPC version                %
%                                                                       %
%=======================================================================%
%                                                                       %
%                   ███████╗██╗  ██╗██╗                                 %
%                   ██╔════╝██║ ██╔╝██║                                 %
%                   ███████╗█████╔╝ ██║                                 %
%                   ╚════██║██╔═██╗ ██║                                 %
%                   ███████║██║  ██╗███████╗                            %
%                   ╚══════╝╚═╝  ╚═╝╚══════╝                            %
%                                                                       %
%=======================================================================%
%                                                                       %
%  State Key Laboratory of Displays and Opto-Electronics                %
%                                                                       %
%  Developed by: Pouya Nosratkhah               Date: Dec, 2025         %
%                                                                       %
%=======================================================================%

clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Load boundary condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('qplate_mat_40x40.mat','img_s');
% mat = double(img_s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Define material parameters for Q-tensor formulation (numerical values)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Simulation_name='Sin_2pi_HalfInitialCondition_inXY';
%Normal values:
    % K11 = 1.1e-11;  % Splay elastic constant (in SI units)
    % K22 = 0.65e-11;  % Twist elastic constant
    % K33 = 1.7e-11;  % Bend elastic constant
    % K24 = 0.4e-11;  % Saddle-splay constant
K11 = 1.1e-11;  % Splay elastic constant (in SI units)
K22 = 0.65e-11;  % Twist elastic constant
K33 = 1.7e-11;  % Bend elastic constant
K24 = 0.4e-11;  % Saddle-splay constant
first_run=1;
% for K11=0.6e-11:0.01e-11:1.6e-11
S_exp = 0.1; % Equilibrium order parameter for bulk as an initial condition
S_mes = 0.9; %elastic constant measurment condition for S parameter
q0 = 0;  % Pitch for cholesteric LC (set to 0 for nematic)
% For Thermotropic energy
a_thermo = -1740; 
b_thermo = -21200; 
c_thermo = 17400;
% Define the grid
% L = 40e-6; %SI unit
L_X=13e-6;
L_Y=13e-6;
Lz = 6e-6;
% N = 100;
N_X=1000;
N_Y=1000;
N_Z = 500;

% Plot figures and show waitbar
plot_result = 0;
wait_bar = 0;

% Parameters for numerical calculation
max_iter = 100000000;
tol = 1e-9;
dx = L_X / (N_X - 1);
dy=L_Y / (N_Y - 1);
dz = Lz / (N_Z - 1);

[a, b, c] = meshgrid(linspace(0, L_X, N_X), linspace(0, L_Y, N_Y), linspace(0, Lz, N_Z));

% Calculate L1 to L6 elastic parameters using numerical values
L1 = (1/(6*S_mes^2)) * (K33 - K11 + 3*K22);
L2 = (1/S_mes^2) * (K11 - K22 - K24);
L3 = (1/S_mes^2) * K24;
L4 = (2/S_mes^2) * q0 * K22;
L6 = (1/(2*S_mes^3)) * (K33 - K11);

% Define symbolic Q-tensor components directly as functions to minimize
syms Q11(x,y,z) Q12(x,y,z) Q13(x,y,z) Q22(x,y,z) Q23(x,y,z) x y z

% Q33 from traceless condition
Q33 = -(Q11(x,y,z) + Q22(x,y,z));

% Construct the full Q-tensor
Q = [Q11(x,y,z), Q12(x,y,z), Q13(x,y,z);
     Q12(x,y,z), Q22(x,y,z), Q23(x,y,z);
     Q13(x,y,z), Q23(x,y,z), Q33];

% Define all Q-tensor components in a cell array
Q_components = {Q11(x,y,z), Q12(x,y,z), Q13(x,y,z);
                Q12(x,y,z), Q22(x,y,z), Q23(x,y,z);
                Q13(x,y,z), Q23(x,y,z), Q33};

% Calculate derivatives
coords = [x, y, z];

% Initialize elastic energy density
F_d = 0;

% First part of equation (48) - quadratic terms
for i = 1:3
    for j = 1:3
        for k = 1:3
            % Calculate derivatives
            dQij_dxk = diff(Q_components{i,j}, coords(k));
            
            % L1 term
            F_d = F_d + (L1/2) * dQij_dxk^2;
            
            % L2 term
            if k <= 3 && j <= 3
                dQij_dxj = diff(Q_components{i,j}, coords(j));
                dQik_dxk = diff(Q_components{i,k}, coords(k));
                F_d = F_d + (L2/2) * dQij_dxj * dQik_dxk;
            end
            
            % L3 term
            if k <= 3 && j <= 3
                dQik_dxj = diff(Q_components{i,k}, coords(j));
                dQij_dxk = diff(Q_components{i,j}, coords(k));
                F_d = F_d + (L3/2) * dQik_dxj * dQij_dxk;
            end
        end
    end
end

% Second part - cubic terms with L4 and L6
for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                % L4 term
                eps_val = get_levi_civita(l, i, k);
                if eps_val ~= 0
                    dQij_dxk = diff(Q_components{i,j}, coords(k));
                    F_d = F_d + (L4/2) * eps_val * Q_components{l,j} * dQij_dxk;
                end
                
                % L6 term
                if l <= 3 && k <= 3
                    dQij_dxl = diff(Q_components{i,j}, coords(l));
                    dQij_dxk = diff(Q_components{i,j}, coords(k));
                    F_d = F_d + (L6/2) * Q_components{l,k} * dQij_dxl * dQij_dxk;
                end
            end
        end
    end
end

F_thermo = a_thermo * trace(Q*Q) + (2*b_thermo/3) * trace(Q*Q*Q) + (c_thermo/2) * trace(Q*Q)^2;

% Total elastic energy
F_total = F_d+F_thermo;

% Derive Euler-Lagrange equations with respect to Q-tensor components
Q_vars = [Q11(x,y,z), Q12(x,y,z), Q13(x,y,z), Q22(x,y,z), Q23(x,y,z)];
eqn = functionalDerivative(F_total, Q_vars) == 0;



% Initialize Q-tensor component matrices
Q11_matrix = zeros(N_Y, N_X, N_Z);
Q12_matrix = zeros(N_Y, N_X, N_Z);
Q13_matrix = zeros(N_Y, N_X, N_Z);
Q22_matrix = zeros(N_Y, N_X, N_Z);
Q23_matrix = zeros(N_Y, N_X, N_Z);

% Function to convert angles to Q-tensor components
calculate_Q_from_angles = @(theta, phi, psi, S1, S2) deal(...
    S1*cos(theta)^2*cos(phi)^2 + S2*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2), ... % Q11
    S1*cos(theta)^2*sin(phi)*cos(phi) - S2*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta)), ... % Q12
    S1*sin(theta)*cos(theta)*cos(phi) + S2*sin(psi)*cos(theta)*(sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta)), ... % Q13
    S1*cos(theta)^2*sin(phi)^2 + S2*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2), ... % Q22
    S1*cos(theta)*sin(theta)*sin(phi) - S2*sin(psi)*cos(theta)*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)) ... % Q23
);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Set boundary conditions by converting from angles to Q-tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %for sin pattern
% % Control the number of periods/cycles of the sin pattern
% num_periods_x = 1; % Change this to control frequency in x direction
% num_periods_y = 0; % Set to 0 for horizontal gratings only, or any value for gratings in both directions
% 
% % Generate coordinate grids for the top layer
% [x, y] = meshgrid(1:N_X, 1:N_Y);
% x_norm = (x - 1) / (N_X - 1);
% % Create sinusoidal pattern with exact number of periods (same phase at start and end)
% if num_periods_y == 0
%     % Only vary along x direction (horizontal gratings)
%     % sin_pattern = (sin(2*2*pi*num_periods_x * x_norm)+1)/2 * pi;
%         % phi_deg =360 * (1 - cos(pi*num_periods_x * x_norm)) / 2;
% % Total target phase = number of full sinusoidal cycles × 360°
% phi_deg = 360 * num_periods_x * x_norm;  % This is a linear ramp in degrees
% 
% % Now pass this linear ramp through a sin-like curve shape (nonlinearization)
% % Create a sinusoidally varying ramp: the derivative of the angle is sinusoidal
% 
% phi_deg =  cos( 2*pi * phi_deg / 360);  % Now in range [0,1] and smooth
% phi_deg = phi_deg * 90;         % Scale back to full range
% 
% % Convert to radians if needed
% phi_rad = deg2rad(phi_deg);
% 
%     % Convert to radians for director field use
%     sin_pattern = deg2rad(phi_deg);
% elseif num_periods_x == 0
%     % Only vary along y direction (vertical gratings)
%     sin_pattern = 2*pi*num_periods_y*(y-1)/(N_Y-1);
% else
%     % Vary along both directions (grid pattern)
%     sin_pattern = sin(2*pi*num_periods_x*(x-1)/(N_X-1)) .* sin(2*pi*num_periods_y*(y-1)/(N_Y-1));
% end
% Parameters
Lambda = 6.5e-6;       % Period in x-direction (meters)

% Create coordinate grid
x_vals = linspace(0, L_X, N_X);   % x coordinates (0 to L_X)
y_vals = linspace(0, L_Y, N_Y);   % y coordinates (0 to L_Y)
[X, Y] = meshgrid(x_vals, y_vals); % Grid for vectorized computation

% Compute the director angle phi in radians
% phi = atan2(cos(pi*x/Lambda), sin(pi*x/Lambda)) modulo pi
sin_pattern = mod(atan2(cos(pi * X / Lambda), sin(pi * X / Lambda)), pi);
% sin_patterny = 2*pi*num_periods_x*(y-1)/(N-1);
% Bottom surface (z = 1)
phi_check=zeros(N_Y,N_X);
% sin_pattern= repelem(sin_pattern1, 4, 4);

parfor i = 1:N_Y
    for j = 1:N_X
        theta_bc = 0;  % Planar alignment
        phi_bc =sin_pattern(i,j);
        phi_check(j,i)=sin_pattern(i,j);
        % phi_bc = deg2rad(mat(i,j) - 45);
        psi_bc = 0;  % Can be set differently for biaxial BC
        S1_bc = 1;
        S2_bc = 0;  % Set non-zero for biaxial boundary
        
        [Q11_matrix(i,j,1), Q12_matrix(i,j,1), Q13_matrix(i,j,1), ...
         Q22_matrix(i,j,1), Q23_matrix(i,j,1)] = ...
         calculate_Q_from_angles(theta_bc, phi_bc, psi_bc, S1_bc, S2_bc);
    end
end

% Top surface (z = N_Z) - same as bottom
% Q11_matrix(:,:,N_Z) = Q11_matrix(:,:,1);
% Q12_matrix(:,:,N_Z) = Q12_matrix(:,:,1);
% Q13_matrix(:,:,N_Z) = Q13_matrix(:,:,1);
% Q22_matrix(:,:,N_Z) = Q22_matrix(:,:,1);
% Q23_matrix(:,:,N_Z) = Q23_matrix(:,:,1);
parfor i = 1:N_Y
    for j = 1:N_X
        theta_bc = 0;  % Planar alignment
        phi_bc =sin_pattern(j,i);
        % phi_check(j,i)=pi* sin_pattern(j,i)+pi;
        % phi_bc = deg2rad(mat(i,j) - 45);
        psi_bc = 0;  % Can be set differently for biaxial BC
        S1_bc = 1;
        S2_bc = 0;  % Set non-zero for biaxial boundary
        
        [Q11_matrix(i,j,N_Z), Q12_matrix(i,j,N_Z), Q13_matrix(i,j,N_Z), ...
         Q22_matrix(i,j,N_Z), Q23_matrix(i,j,N_Z)] = ...
         calculate_Q_from_angles(theta_bc, phi_bc, psi_bc, S1_bc, S2_bc);
    end
end
if plot_result==1
    figure(25)
    imagesc(phi_check);
end
% Initialize bulk with uniform director (initial guess)
parfor k = 2:N_Z-1
    % Linear interpolation of phi from bottom to top
    % phi_init = pi/100 + (pi/2 - pi/100) * (k-1)/(N_Z-1);
    for i = 1:N_Y
        for j = 1:N_X
            % Initial guess: uniform director along x
            theta_init = 0;
            if k <N_Z/2
                phi_init = sin_pattern(i,j);
            else
                phi_init = sin_pattern(j,i);   
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
%   Parameters for iteration and numerical calculation of PDE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter = 1;

disp('Euler-Lagrange equations derived for Q-tensor components.');
disp('System ready for numerical solution.');
disp(['Number of equations: ', num2str(length(eqn))]);
disp('Boundary conditions converted from angles to Q-tensor values.');

% Visualization of Q-tensor field by extracting director information
% Extract director from Q-tensor by eigenvalue decomposition

% Initialize director components
u = zeros(N_Y, N_X, N_Z);
v = zeros(N_Y, N_X, N_Z);
w = zeros(N_Y, N_X, N_Z);

% Extract director at each point
parfor i = 1:N_Y
    for j = 1:N_X
        for k = 1:N_Z
            % Construct Q-tensor at this point
            Q_local = [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k);
                       Q12_matrix(i,j,k), Q22_matrix(i,j,k), Q23_matrix(i,j,k);
                       Q13_matrix(i,j,k), Q23_matrix(i,j,k), -(Q11_matrix(i,j,k) + Q22_matrix(i,j,k))];
            
            % Find eigenvalues and eigenvectors
            [eigvecs, eigvals] = eig(Q_local);
            eigvals = diag(eigvals);
            
            % Find the largest eigenvalue
            [~, max_idx] = max(eigvals);
            
            % The corresponding eigenvector is the director
            director = eigvecs(:, max_idx);
            
            % Ensure consistent orientation (e.g., positive z-component)
            if director(3) < 0
                director = -director;
            end
            
            u(i,j,k) = director(1);
            v(i,j,k) = director(2);
            w(i,j,k) = director(3);
        end
    end
end

% Choose a fixed vector for cross product
fixed_vector = [0, 0, 1]; % z-axis

% % Initialize new vector components for perpendicular vectors
% u_perp = zeros(size(u));
% v_perp = zeros(size(v));
% w_perp = zeros(size(w));
% 
% % Calculate perpendicular vector components using cross product
% parfor i = 1:N
%     for j = 1:N
%         for k = 1:N_Z
%             original_vector = [u(i, j, k), v(i, j, k), w(i, j, k)];
%             perp_vector = cross(original_vector, fixed_vector);
%             % Normalize the perpendicular vector if it's not zero
%             if norm(perp_vector) > 1e-10
%                 perp_vector = perp_vector / norm(perp_vector);
%             end
%             u_perp(i, j, k) = perp_vector(1);
%             v_perp(i, j, k) = perp_vector(2);
%             w_perp(i, j, k) = perp_vector(3);
%         end
%     end
% end
if plot_result==1
    % Plot the bottom surface boundary condition
    figure('Name', 'Bottom Surface Boundary Condition');
    hold on;
    line_thickness = 1; % Set line thickness to 15
    scale_factor = 3*(L_X+L_Y)/400; % Scale factor for vector length
    for i = 1:1:N_Y % Step by 2 to reduce data points
        for j = 1:1:N_X % Step by 2 to reduce data points
            k = 1; % Bottom surface
            % Calculate the end points of the perpendicular vectors
            x_end = a(i, j, k) + scale_factor * u(i, j, k);
            y_end = b(i, j, k) + scale_factor * v(i, j, k);
            z_end = c(i, j, k) + scale_factor * w(i, j, k);
    
            % Plot the line representing the perpendicular vector
            plot3([a(i, j, k), x_end], [b(i, j, k), y_end], [c(i, j, k), z_end], 'k', 'LineWidth', line_thickness);
        end
    end
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Bottom Surface Pattern');
    grid on;
    % axis equal;
    view(0, 90); % Top view
    hold off;
end
% % Plot the 3D director field
% figure('Name', '3D Director Field');
% hold on;
% line_thickness = 3; % Thinner lines for 3D view
% scale_factor = 5*L/100; % Smaller scale for 3D
% 
% % Plot bottom and top surfaces with different colors
% for i = 1:floor(N/40):N % Step by 3 for clearer visualization
%     for j = 1:floor(N/40):N
%         % Bottom surface (red)
%         k = 1;
%         x_end = a(i, j, k) + scale_factor * u(i, j, k);
%         y_end = b(i, j, k) + scale_factor * v(i, j, k);
%         z_end = c(i, j, k) + scale_factor * w(i, j, k);
%         plot3([a(i, j, k), x_end], [b(i, j, k), y_end], [c(i, j, k), z_end], 'r', 'LineWidth', line_thickness);
% 
%         % Top surface (blue)
%         k = N_Z;
%         x_end = a(i, j, k) + scale_factor * u(i, j, k);
%         y_end = b(i, j, k) + scale_factor * v(i, j, k);
%         z_end = c(i, j, k) + scale_factor * w(i, j, k);
%         plot3([a(i, j, k), x_end], [b(i, j, k), y_end], [c(i, j, k), z_end], 'b', 'LineWidth', line_thickness);
%     end
% end
% 
% % Plot bulk (green)
% for i = 1:floor(N/40):N % Sparse sampling for bulk
%     for j = 1:floor(N/40):N
%         for k = 2:floor((N_Z-2)/5):N_Z-1 % Middle layers
%             x_end = a(i, j, k) + scale_factor * u(i, j, k);
%             y_end = b(i, j, k) + scale_factor * v(i, j, k);
%             z_end = c(i, j, k) + scale_factor * w(i, j, k);
%             plot3([a(i, j, k), x_end], [b(i, j, k), y_end], [c(i, j, k), z_end], 'g', 'LineWidth', line_thickness);
%         end
%     end
% end
% 
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('3D Director Field from Q-tensor (Starting condition)');
% legend('Bottom BC', 'Top BC', 'Bulk', 'Location', 'best');
% grid on;
% % axis equal;
% view(45, 30);
% hold off;

S_matrix = zeros(N_Y, N_X, N_Z);
for i = 1:N_Y
    for j = 1:N_X
        for k = 1:N_Z
            % Construct Q-tensor at this point
            Q_local = [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k);
                       Q12_matrix(i,j,k), Q22_matrix(i,j,k), Q23_matrix(i,j,k);
                       Q13_matrix(i,j,k), Q23_matrix(i,j,k), -(Q11_matrix(i,j,k) + Q22_matrix(i,j,k))];
            
            % Find eigenvalues
            eigvals = eig(Q_local);
            
            % Order parameter is related to the largest eigenvalue
            S_matrix(i,j,k) = (3/2) * max(eigvals);
        end
    end
end

if plot_result==1
    % Plot Q-tensor eigenvalues to check order parameter
    figure('Name', 'Order Parameter Distribution');
    % Plot order parameter at mid-plane
    subplot(1,2,1);
    imagesc(S_matrix(:,:,round(N_Z/2)));
    colorbar;
    title('Order Parameter at Mid-plane');
    xlabel('X index');
    ylabel('Y index');
    % axis equal tight;
    
    % Plot order parameter profile along z
    subplot(1,2,2);
    plot(squeeze(S_matrix(round(N_Y/2), round(N_X/2), :)), 'b-o');
    xlabel('Z index');
    ylabel('Order Parameter S');
    title('Order Parameter Profile along Z');
    grid on;
    
    disp('Visualization complete.');
end
disp(['Initial order parameter at boundaries: ', num2str(S_exp)]);

% The system is now ready to minimize with respect to Q-tensor components
% Periodic boundary conditions for the side walls will be implemented in the iteration
% Iterative solver loop
tic
if wait_bar==1
    hWaitbar = waitbar(0, 'Iteration 1', 'Name', 'Solving Q-tensor problem','CreateCancelBtn','delete(gcbf)');
end
    prev_max_change=10;

% Pre-compute grid parameters
    [Nx, Ny, Nz] = size(Q11_matrix);
    dx2 = dx^2;
    dy2 = dy^2;
    dz2 = dz^2;


      % Vectorize equation strings
    equationStr1 = vectorize(lhs(eqn(1))); % For Q11
    equationStr2 = vectorize(lhs(eqn(2))); % For Q12
    equationStr3 = vectorize(lhs(eqn(3))); % For Q13
    equationStr4 = vectorize(lhs(eqn(4))); % For Q22
    equationStr5 = vectorize(lhs(eqn(5))); % For Q23


  % Define the variable mapping for Q-tensor components
% IMPORTANT: Order matters! Put longer strings (derivatives) before shorter ones
varMap = containers.Map(...
    {'diff(Q11(x, y, z), x, x)', 'diff(Q11(x, y, z), x, y)', 'diff(Q11(x, y, z), x, z)', ...
     'diff(Q11(x, y, z), y, y)', 'diff(Q11(x, y, z), y, z)', 'diff(Q11(x, y, z), z, z)', ...
     'diff(Q11(x, y, z), x)', 'diff(Q11(x, y, z), y)', 'diff(Q11(x, y, z), z)', ...
     'diff(Q12(x, y, z), x, x)', 'diff(Q12(x, y, z), x, y)', 'diff(Q12(x, y, z), x, z)', ...
     'diff(Q12(x, y, z), y, y)', 'diff(Q12(x, y, z), y, z)', 'diff(Q12(x, y, z), z, z)', ...
     'diff(Q12(x, y, z), x)', 'diff(Q12(x, y, z), y)', 'diff(Q12(x, y, z), z)', ...
     'diff(Q13(x, y, z), x, x)', 'diff(Q13(x, y, z), x, y)', 'diff(Q13(x, y, z), x, z)', ...
     'diff(Q13(x, y, z), y, y)', 'diff(Q13(x, y, z), y, z)', 'diff(Q13(x, y, z), z, z)', ...
     'diff(Q13(x, y, z), x)', 'diff(Q13(x, y, z), y)', 'diff(Q13(x, y, z), z)', ...
     'diff(Q22(x, y, z), x, x)', 'diff(Q22(x, y, z), x, y)', 'diff(Q22(x, y, z), x, z)', ...
     'diff(Q22(x, y, z), y, y)', 'diff(Q22(x, y, z), y, z)', 'diff(Q22(x, y, z), z, z)', ...
     'diff(Q22(x, y, z), x)', 'diff(Q22(x, y, z), y)', 'diff(Q22(x, y, z), z)', ...
     'diff(Q23(x, y, z), x, x)', 'diff(Q23(x, y, z), x, y)', 'diff(Q23(x, y, z), x, z)', ...
     'diff(Q23(x, y, z), y, y)', 'diff(Q23(x, y, z), y, z)', 'diff(Q23(x, y, z), z, z)', ...
     'diff(Q23(x, y, z), x)', 'diff(Q23(x, y, z), y)', 'diff(Q23(x, y, z), z)', ...
     'diff(Q33(x, y, z), x, x)', 'diff(Q33(x, y, z), x, y)', 'diff(Q33(x, y, z), x, z)', ...
     'diff(Q33(x, y, z), y, y)', 'diff(Q33(x, y, z), y, z)', 'diff(Q33(x, y, z), z, z)', ...
     'diff(Q33(x, y, z), x)', 'diff(Q33(x, y, z), y)', 'diff(Q33(x, y, z), z)', ...
     'Q11(x, y, z)', 'Q12(x, y, z)', 'Q13(x, y, z)', 'Q22(x, y, z)', 'Q23(x, y, z)', 'Q33(x, y, z)', ...
     'L1', 'L2', 'L3', 'L4', 'L6'}, ...
    {'d2Q11_dx2', 'd2Q11_dxdy', 'd2Q11_dxdz', 'd2Q11_dy2', 'd2Q11_dydz', 'd2Q11_dz2', ...
     'dQ11_dx', 'dQ11_dy', 'dQ11_dz', ...
     'd2Q12_dx2', 'd2Q12_dxdy', 'd2Q12_dxdz', 'd2Q12_dy2', 'd2Q12_dydz', 'd2Q12_dz2', ...
     'dQ12_dx', 'dQ12_dy', 'dQ12_dz', ...
     'd2Q13_dx2', 'd2Q13_dxdy', 'd2Q13_dxdz', 'd2Q13_dy2', 'd2Q13_dydz', 'd2Q13_dz2', ...
     'dQ13_dx', 'dQ13_dy', 'dQ13_dz', ...
     'd2Q22_dx2', 'd2Q22_dxdy', 'd2Q22_dxdz', 'd2Q22_dy2', 'd2Q22_dydz', 'd2Q22_dz2', ...
     'dQ22_dx', 'dQ22_dy', 'dQ22_dz', ...
     'd2Q23_dx2', 'd2Q23_dxdy', 'd2Q23_dxdz', 'd2Q23_dy2', 'd2Q23_dydz', 'd2Q23_dz2', ...
     'dQ23_dx', 'dQ23_dy', 'dQ23_dz', ...
     'd2Q33_dx2', 'd2Q33_dxdy', 'd2Q33_dxdz', 'd2Q33_dy2', 'd2Q33_dydz', 'd2Q33_dz2', ...
     'dQ33_dx', 'dQ33_dy', 'dQ33_dz', ...
     'Q11_matrix', 'Q12_matrix', 'Q13_matrix', 'Q22_matrix', 'Q23_matrix', 'Q33_matrix', ...
     'L1', 'L2', 'L3', 'L4', 'L6'});

% Create ordered list - derivatives first, then base functions
% This ensures longer strings are replaced before shorter ones
ordered_keys = {
    'diff(Q11(x, y, z), x, x)', 'diff(Q11(x, y, z), x, y)', 'diff(Q11(x, y, z), x, z)', ...
    'diff(Q11(x, y, z), y, y)', 'diff(Q11(x, y, z), y, z)', 'diff(Q11(x, y, z), z, z)', ...
    'diff(Q12(x, y, z), x, x)', 'diff(Q12(x, y, z), x, y)', 'diff(Q12(x, y, z), x, z)', ...
    'diff(Q12(x, y, z), y, y)', 'diff(Q12(x, y, z), y, z)', 'diff(Q12(x, y, z), z, z)', ...
    'diff(Q13(x, y, z), x, x)', 'diff(Q13(x, y, z), x, y)', 'diff(Q13(x, y, z), x, z)', ...
    'diff(Q13(x, y, z), y, y)', 'diff(Q13(x, y, z), y, z)', 'diff(Q13(x, y, z), z, z)', ...
    'diff(Q22(x, y, z), x, x)', 'diff(Q22(x, y, z), x, y)', 'diff(Q22(x, y, z), x, z)', ...
    'diff(Q22(x, y, z), y, y)', 'diff(Q22(x, y, z), y, z)', 'diff(Q22(x, y, z), z, z)', ...
    'diff(Q23(x, y, z), x, x)', 'diff(Q23(x, y, z), x, y)', 'diff(Q23(x, y, z), x, z)', ...
    'diff(Q23(x, y, z), y, y)', 'diff(Q23(x, y, z), y, z)', 'diff(Q23(x, y, z), z, z)', ...
    'diff(Q33(x, y, z), x, x)', 'diff(Q33(x, y, z), x, y)', 'diff(Q33(x, y, z), x, z)', ...
    'diff(Q33(x, y, z), y, y)', 'diff(Q33(x, y, z), y, z)', 'diff(Q33(x, y, z), z, z)', ...
    'diff(Q11(x, y, z), x)', 'diff(Q11(x, y, z), y)', 'diff(Q11(x, y, z), z)', ...
    'diff(Q12(x, y, z), x)', 'diff(Q12(x, y, z), y)', 'diff(Q12(x, y, z), z)', ...
    'diff(Q13(x, y, z), x)', 'diff(Q13(x, y, z), y)', 'diff(Q13(x, y, z), z)', ...
    'diff(Q22(x, y, z), x)', 'diff(Q22(x, y, z), y)', 'diff(Q22(x, y, z), z)', ...
    'diff(Q23(x, y, z), x)', 'diff(Q23(x, y, z), y)', 'diff(Q23(x, y, z), z)', ...
    'diff(Q33(x, y, z), x)', 'diff(Q33(x, y, z), y)', 'diff(Q33(x, y, z), z)', ...
    'Q11(x, y, z)', 'Q12(x, y, z)', 'Q13(x, y, z)', ...
    'Q22(x, y, z)', 'Q23(x, y, z)', 'Q33(x, y, z)', ...
    'L1', 'L2', 'L3', 'L4', 'L6'
};
if first_run==0
    load("Q_matrix","Q11_matrix","Q12_matrix","Q22_matrix","Q13_matrix","Q23_matrix","Q33_matrix");
end
first_run=0;
    ip1 = circshift(1:Nx, -1);
    im1 = circshift(1:Nx, 1);
    jp1 = circshift(1:Ny, -1);
    jm1 = circshift(1:Ny, 1);
    k_internal = 2:Nz-1;
% Arbitrary x-axis counter for plot updates (independent of iter)
%% ========= Part A) Put this BEFORE your iter loop =========
Nwin   = 20;          % show last 20 updates on the right
plot_x = 0;           % independent counter (only increments when you compute energy)

maxPts = 20000;       % just make it big enough
E_hist  = nan(maxPts,1);
MC_hist = nan(maxPts,1);

if plot_result==1
  
    figure('Name', 'Progress curve'); clf
    
    % Full history (left)
    ax1 = subplot(2,2,1);
    hE_full = animatedline(ax1,'LineWidth',1.5);
    grid(ax1,'on'); xlabel(ax1,'Update #'); ylabel(ax1,'Energy');
    title(ax1,'Energy (full)');
    
    ax3 = subplot(2,2,3);
    set(ax3,'YScale','log');  % max_change spans decades
    hMC_full = animatedline(ax3,'LineWidth',1.5);
    grid(ax3,'on'); xlabel(ax3,'Update #'); ylabel(ax3,'Max change');
    title(ax3,'Max\_change (full, log-y)');
    
    % Zoom last N (right)
    ax2 = subplot(2,2,2);
    hE_zoom = plot(ax2,nan,nan,'-','LineWidth',1.5);
    grid(ax2,'on'); xlabel(ax2,'Update #'); ylabel(ax2,'Energy');
    title(ax2,sprintf('Energy (last %d)',Nwin));
    
    ax4 = subplot(2,2,4);
    hMC_zoom = plot(ax4,nan,nan,'-','LineWidth',1.5);
    grid(ax4,'on'); xlabel(ax4,'Update #'); ylabel(ax4,'Max change');
    title(ax4,sprintf('Max\\_change (last %d)',Nwin));
end
for iter = 1:max_iter
    
    % Store current values for convergence check
    Q11_old = Q11_matrix;
    Q12_old = Q12_matrix;
    Q13_old = Q13_matrix;
    Q22_old = Q22_matrix;
    Q23_old = Q23_matrix;

    % Calculate derivatives for all Q-tensor components
    % First derivatives - Q11
    dQ11_dx = (Q11_matrix(ip1, :, :) - Q11_matrix(im1, :, :)) / (2 * dx);
    dQ11_dy = (Q11_matrix(:, jp1, :) - Q11_matrix(:, jm1, :)) / (2 * dy);
    dQ11_dz = zeros(Nx, Ny, Nz);
    dQ11_dz(:, :, k_internal) = (Q11_matrix(:, :, k_internal+1) - Q11_matrix(:, :, k_internal-1)) / (2 * dz);

    % First derivatives - Q12
    dQ12_dx = (Q12_matrix(ip1, :, :) - Q12_matrix(im1, :, :)) / (2 * dx);
    dQ12_dy = (Q12_matrix(:, jp1, :) - Q12_matrix(:, jm1, :)) / (2 * dy);
    dQ12_dz = zeros(Nx, Ny, Nz);
    dQ12_dz(:, :, k_internal) = (Q12_matrix(:, :, k_internal+1) - Q12_matrix(:, :, k_internal-1)) / (2 * dz);

    % First derivatives - Q13
    dQ13_dx = (Q13_matrix(ip1, :, :) - Q13_matrix(im1, :, :)) / (2 * dx);
    dQ13_dy = (Q13_matrix(:, jp1, :) - Q13_matrix(:, jm1, :)) / (2 * dy);
    dQ13_dz = zeros(Nx, Ny, Nz);
    dQ13_dz(:, :, k_internal) = (Q13_matrix(:, :, k_internal+1) - Q13_matrix(:, :, k_internal-1)) / (2 * dz);

    % First derivatives - Q22
    dQ22_dx = (Q22_matrix(ip1, :, :) - Q22_matrix(im1, :, :)) / (2 * dx);
    dQ22_dy = (Q22_matrix(:, jp1, :) - Q22_matrix(:, jm1, :)) / (2 * dy);
    dQ22_dz = zeros(Nx, Ny, Nz);
    dQ22_dz(:, :, k_internal) = (Q22_matrix(:, :, k_internal+1) - Q22_matrix(:, :, k_internal-1)) / (2 * dz);

    % First derivatives - Q23
    dQ23_dx = (Q23_matrix(ip1, :, :) - Q23_matrix(im1, :, :)) / (2 * dx);
    dQ23_dy = (Q23_matrix(:, jp1, :) - Q23_matrix(:, jm1, :)) / (2 * dy);
    dQ23_dz = zeros(Nx, Ny, Nz);
    dQ23_dz(:, :, k_internal) = (Q23_matrix(:, :, k_internal+1) - Q23_matrix(:, :, k_internal-1)) / (2 * dz);

    % Second derivatives - Q11
    d2Q11_dx2 = (Q11_matrix(ip1, :, :) - 2 * Q11_matrix + Q11_matrix(im1, :, :)) / dx2;
    d2Q11_dy2 = (Q11_matrix(:, jp1, :) - 2 * Q11_matrix + Q11_matrix(:, jm1, :)) / dy2;
    d2Q11_dz2 = zeros(Nx, Ny, Nz);
    d2Q11_dz2(:, :, k_internal) = (Q11_matrix(:, :, k_internal+1) - 2 * Q11_matrix(:, :, k_internal) + Q11_matrix(:, :, k_internal-1)) / dz2;

    % Second derivatives - Q12
    d2Q12_dx2 = (Q12_matrix(ip1, :, :) - 2 * Q12_matrix + Q12_matrix(im1, :, :)) / dx2;
    d2Q12_dy2 = (Q12_matrix(:, jp1, :) - 2 * Q12_matrix + Q12_matrix(:, jm1, :)) / dy2;
    d2Q12_dz2 = zeros(Nx, Ny, Nz);
    d2Q12_dz2(:, :, k_internal) = (Q12_matrix(:, :, k_internal+1) - 2 * Q12_matrix(:, :, k_internal) + Q12_matrix(:, :, k_internal-1)) / dz2;

    % Second derivatives - Q13
    d2Q13_dx2 = (Q13_matrix(ip1, :, :) - 2 * Q13_matrix + Q13_matrix(im1, :, :)) / dx2;
    d2Q13_dy2 = (Q13_matrix(:, jp1, :) - 2 * Q13_matrix + Q13_matrix(:, jm1, :)) / dy2;
    d2Q13_dz2 = zeros(Nx, Ny, Nz);
    d2Q13_dz2(:, :, k_internal) = (Q13_matrix(:, :, k_internal+1) - 2 * Q13_matrix(:, :, k_internal) + Q13_matrix(:, :, k_internal-1)) / dz2;

    % Second derivatives - Q22
    d2Q22_dx2 = (Q22_matrix(ip1, :, :) - 2 * Q22_matrix + Q22_matrix(im1, :, :)) / dx2;
    d2Q22_dy2 = (Q22_matrix(:, jp1, :) - 2 * Q22_matrix + Q22_matrix(:, jm1, :)) / dy2;
    d2Q22_dz2 = zeros(Nx, Ny, Nz);
    d2Q22_dz2(:, :, k_internal) = (Q22_matrix(:, :, k_internal+1) - 2 * Q22_matrix(:, :, k_internal) + Q22_matrix(:, :, k_internal-1)) / dz2;

    % Second derivatives - Q23
    d2Q23_dx2 = (Q23_matrix(ip1, :, :) - 2 * Q23_matrix + Q23_matrix(im1, :, :)) / dx2;
    d2Q23_dy2 = (Q23_matrix(:, jp1, :) - 2 * Q23_matrix + Q23_matrix(:, jm1, :)) / dy2;
    d2Q23_dz2 = zeros(Nx, Ny, Nz);
    d2Q23_dz2(:, :, k_internal) = (Q23_matrix(:, :, k_internal+1) - 2 * Q23_matrix(:, :, k_internal) + Q23_matrix(:, :, k_internal-1)) / dz2;

    % Mixed second derivatives - Q11
    d2Q11_dxdy = (Q11_matrix(ip1, jp1, :) - Q11_matrix(ip1, jm1, :) - Q11_matrix(im1, jp1, :) + Q11_matrix(im1, jm1, :)) / (4 * dx * dy);
    d2Q11_dxdz = zeros(Nx, Ny, Nz);
    d2Q11_dxdz(:, :, k_internal) = (Q11_matrix(ip1, :, k_internal+1) - Q11_matrix(ip1, :, k_internal-1) - Q11_matrix(im1, :, k_internal+1) + Q11_matrix(im1, :, k_internal-1)) / (4 * dx * dz);
    d2Q11_dydz = zeros(Nx, Ny, Nz);
    d2Q11_dydz(:, :, k_internal) = (Q11_matrix(:, jp1, k_internal+1) - Q11_matrix(:, jp1, k_internal-1) - Q11_matrix(:, jm1, k_internal+1) + Q11_matrix(:, jm1, k_internal-1)) / (4 * dy * dz);

    % Mixed second derivatives - Q12
    d2Q12_dxdy = (Q12_matrix(ip1, jp1, :) - Q12_matrix(ip1, jm1, :) - Q12_matrix(im1, jp1, :) + Q12_matrix(im1, jm1, :)) / (4 * dx * dy);
    d2Q12_dxdz = zeros(Nx, Ny, Nz);
    d2Q12_dxdz(:, :, k_internal) = (Q12_matrix(ip1, :, k_internal+1) - Q12_matrix(ip1, :, k_internal-1) - Q12_matrix(im1, :, k_internal+1) + Q12_matrix(im1, :, k_internal-1)) / (4 * dx * dz);
    d2Q12_dydz = zeros(Nx, Ny, Nz);
    d2Q12_dydz(:, :, k_internal) = (Q12_matrix(:, jp1, k_internal+1) - Q12_matrix(:, jp1, k_internal-1) - Q12_matrix(:, jm1, k_internal+1) + Q12_matrix(:, jm1, k_internal-1)) / (4 * dy * dz);

    % Mixed second derivatives - Q13
    d2Q13_dxdy = (Q13_matrix(ip1, jp1, :) - Q13_matrix(ip1, jm1, :) - Q13_matrix(im1, jp1, :) + Q13_matrix(im1, jm1, :)) / (4 * dx * dy);
    d2Q13_dxdz = zeros(Nx, Ny, Nz);
    d2Q13_dxdz(:, :, k_internal) = (Q13_matrix(ip1, :, k_internal+1) - Q13_matrix(ip1, :, k_internal-1) - Q13_matrix(im1, :, k_internal+1) + Q13_matrix(im1, :, k_internal-1)) / (4 * dx * dz);
    d2Q13_dydz = zeros(Nx, Ny, Nz);
    d2Q13_dydz(:, :, k_internal) = (Q13_matrix(:, jp1, k_internal+1) - Q13_matrix(:, jp1, k_internal-1) - Q13_matrix(:, jm1, k_internal+1) + Q13_matrix(:, jm1, k_internal-1)) / (4 * dy * dz);

    % Mixed second derivatives - Q22
    d2Q22_dxdy = (Q22_matrix(ip1, jp1, :) - Q22_matrix(ip1, jm1, :) - Q22_matrix(im1, jp1, :) + Q22_matrix(im1, jm1, :)) / (4 * dx * dy);
    d2Q22_dxdz = zeros(Nx, Ny, Nz);
    d2Q22_dxdz(:, :, k_internal) = (Q22_matrix(ip1, :, k_internal+1) - Q22_matrix(ip1, :, k_internal-1) - Q22_matrix(im1, :, k_internal+1) + Q22_matrix(im1, :, k_internal-1)) / (4 * dx * dz);
    d2Q22_dydz = zeros(Nx, Ny, Nz);
    d2Q22_dydz(:, :, k_internal) = (Q22_matrix(:, jp1, k_internal+1) - Q22_matrix(:, jp1, k_internal-1) - Q22_matrix(:, jm1, k_internal+1) + Q22_matrix(:, jm1, k_internal-1)) / (4 * dy * dz);

    % Mixed second derivatives - Q23
    d2Q23_dxdy = (Q23_matrix(ip1, jp1, :) - Q23_matrix(ip1, jm1, :) - Q23_matrix(im1, jp1, :) + Q23_matrix(im1, jm1, :)) / (4 * dx * dy);
    d2Q23_dxdz = zeros(Nx, Ny, Nz);
    d2Q23_dxdz(:, :, k_internal) = (Q23_matrix(ip1, :, k_internal+1) - Q23_matrix(ip1, :, k_internal-1) - Q23_matrix(im1, :, k_internal+1) + Q23_matrix(im1, :, k_internal-1)) / (4 * dx * dz);
    d2Q23_dydz = zeros(Nx, Ny, Nz);
    d2Q23_dydz(:, :, k_internal) = (Q23_matrix(:, jp1, k_internal+1) - Q23_matrix(:, jp1, k_internal-1) - Q23_matrix(:, jm1, k_internal+1) + Q23_matrix(:, jm1, k_internal-1)) / (4 * dy * dz);
% The results are stored in the preallocated arrays.
% The results are stored in cell arrays dQ_dx, dQ_dy, etc.
    % Calculate Q33 and its derivatives from traceless condition: Q33 = -(Q11 + Q22)
    Q33_matrix = -(Q11_matrix + Q22_matrix);
    dQ33_dx = -(dQ11_dx + dQ22_dx);
    dQ33_dy = -(dQ11_dy + dQ22_dy);
    dQ33_dz = -(dQ11_dz + dQ22_dz);
    d2Q33_dx2 = -(d2Q11_dx2 + d2Q22_dx2);
    d2Q33_dy2 = -(d2Q11_dy2 + d2Q22_dy2);
    d2Q33_dz2 = -(d2Q11_dz2 + d2Q22_dz2);
    d2Q33_dxdy = -(d2Q11_dxdy + d2Q22_dxdy);
    d2Q33_dxdz = -(d2Q11_dxdz + d2Q22_dxdz);
    d2Q33_dydz = -(d2Q11_dydz + d2Q22_dydz);



% Replace variables in equation strings using ordered list
for idx = 1:length(ordered_keys)
    eqVar = ordered_keys{idx};
    wsVar = varMap(eqVar);
    equationStr1 = strrep(equationStr1, eqVar, wsVar);
    equationStr2 = strrep(equationStr2, eqVar, wsVar);
    equationStr3 = strrep(equationStr3, eqVar, wsVar);
    equationStr4 = strrep(equationStr4, eqVar, wsVar);
    equationStr5 = strrep(equationStr5, eqVar, wsVar);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Calculation rate (alpha parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    if iter == 1
    max_error = max(max(max(abs(eval(equationStr1)))));
        max_old = max(max(max(abs(Q11_old))));
        alpha = max_old/max_error/50;
    end
    % alpha = 1; % Step size for gradient descent
   
    eqn_fd_Q11 = alpha * eval(equationStr1);
    eqn_fd_Q12 = alpha * eval(equationStr2);
    eqn_fd_Q13 = alpha * eval(equationStr3);
    eqn_fd_Q22 = alpha * eval(equationStr4);
    eqn_fd_Q23 = alpha * eval(equationStr5);
    
    % Set boundary condition enforcement (top and bottom fixed)
    eqn_fd_Q11(:,:,1) = 0;
    eqn_fd_Q11(:,:,Nz) = 0;
    eqn_fd_Q12(:,:,1) = 0;
    eqn_fd_Q12(:,:,Nz) = 0;
    eqn_fd_Q13(:,:,1) = 0;
    eqn_fd_Q13(:,:,Nz) = 0;
    eqn_fd_Q22(:,:,1) = 0;
    eqn_fd_Q22(:,:,Nz) = 0;
    eqn_fd_Q23(:,:,1) = 0;
    eqn_fd_Q23(:,:,Nz) = 0;

    % Update Q-tensor components
    Q11_matrix = Q11_matrix - eqn_fd_Q11;
    Q12_matrix = Q12_matrix - eqn_fd_Q12;
    Q13_matrix = Q13_matrix - eqn_fd_Q13;
    Q22_matrix = Q22_matrix - eqn_fd_Q22;
    Q23_matrix = Q23_matrix - eqn_fd_Q23;

    % Calculate total elastic energy (optional, for monitoring convergence)
    if mod(iter, 10000) == 0 
        % total_energy = calculateQTensorElasticEnergy(Q11_matrix, Q12_matrix, Q13_matrix, Q22_matrix, Q23_matrix, ...
                                                      % ip1, im1, jp1, jm1, dx, dz, L1, L2, L3, L4, L6, N, N_Z);
        % disp(['Iteration: ', num2str(iter), ', Total elastic energy: ', num2str(total_energy)]);
        energy=calculate_elastic_energy(Q11_matrix, Q12_matrix, Q13_matrix, Q22_matrix, Q23_matrix, L1, L2, L3, dx, dy, dz)*-1;
        if plot_result==1

            plot_x = plot_x + 1;
            
            energy_s     = double(energy);     energy_s     = energy_s(1);
            max_change_s = double(max_change); max_change_s = max_change_s(1);
            
            E_hist(plot_x)  = energy_s;
            MC_hist(plot_x) = max_change_s;
            
            % Full history (left)
            addpoints(hE_full, plot_x, E_hist(plot_x));
            mc = MC_hist(plot_x); if ~(mc > 0), mc = eps; end
            addpoints(hMC_full, plot_x, mc);
            
            % Zoom window indices (right)
            k0 = max(1, plot_x - Nwin + 1);
            ks = k0:plot_x;
            
            set(hE_zoom,  'XData', ks, 'YData', E_hist(ks));
            set(hMC_zoom, 'XData', ks, 'YData', MC_hist(ks));
            
            % ---- Safe axis limits (avoid [a a]) ----
            if numel(ks) >= 2
                xlim(ax2, [ks(1) ks(end)]);
                xlim(ax4, [ks(1) ks(end)]);
            else
                xlim(ax2, [ks(1)-0.5 ks(1)+0.5]);
                xlim(ax4, [ks(1)-0.5 ks(1)+0.5]);
            end
            
            % Safe y-lims with padding
            yE = E_hist(ks);
            ymin = min(yE); ymax = max(yE);
            if ymax <= ymin
                d = max(1e-12, 0.05*max(1,abs(ymin)));
                ylim(ax2, [ymin-d ymax+d]);
            else
                pad = 0.05*(ymax-ymin);
                ylim(ax2, [ymin-pad ymax+pad]);
            end
            
            yMC = MC_hist(ks);
            ymin = min(yMC); ymax = max(yMC);
            if ymax <= ymin
                d = max(1e-12, 0.05*max(1,abs(ymin)));
                ylim(ax4, [ymin-d ymax+d]);
            else
                pad = 0.05*(ymax-ymin);
                ylim(ax4, [ymin-pad ymax+pad]);
            end
            
            drawnow limitrate nocallbacks
        end

    end


    

    % Check for convergence
    max_change = max([max(max(max(abs(Q11_matrix(:,:,:) - Q11_old(:,:,:))))), ...
                      max(max(max(abs(Q12_matrix(:,:,:) - Q12_old(:,:,:))))), ...
                      max(max(max(abs(Q13_matrix(:,:,:) - Q13_old(:,:,:))))), ...
                      max(max(max(abs(Q22_matrix(:,:,:) - Q22_old(:,:,:))))), ...
                      max(max(max(abs(Q23_matrix(:,:,:) - Q23_old(:,:,:)))))]);

    if max_change < tol
        disp(['Converged after ', num2str(iter), ' iterations.']);
        break;
    end
    if wait_bar==1
        % Update waitbar
        if ~ishandle(hWaitbar)
            disp('Stopped by user');
            break;
        else
            if mod(iter, 100) == 0
                waitbar(iter/max_iter, hWaitbar, ['Iteration ' num2str(iter) ', Max change: ' num2str(max_change, '%.2e')]);
            end
        end
    end
    if mod(iter, 2000) == 0
        toc
        disp(['Iteration: ', num2str(iter), ', Max change: ', num2str(max_change)]);
        tic
    end

   
  
end
save("Q_matrix","Q11_matrix","Q12_matrix","Q22_matrix","Q13_matrix","Q23_matrix","Q33_matrix");
save(strcat('\Q_matrix-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.mat'),"Q11_matrix","Q12_matrix","Q22_matrix","Q13_matrix","Q23_matrix","Q33_matrix");
toc
% Post-iteration visualization

% Extract director from converged Q-tensor
% u_final = zeros(N, N, N_Z);
% v_final = zeros(N, N, N_Z);
% w_final = zeros(N, N, N_Z);
% theta_matrix = zeros(N, N, N_Z);
% phi_matrix = zeros(N, N, N_Z);
% S_final = zeros(N, N, N_Z);
% 
% parfor i = 1:N
%     for j = 1:N
%         for k = 1:N_Z
%             % Construct Q-tensor at this point
%             Q_local = [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k);
%                        Q12_matrix(i,j,k), Q22_matrix(i,j,k), Q23_matrix(i,j,k);
%                        Q13_matrix(i,j,k), Q23_matrix(i,j,k), -(Q11_matrix(i,j,k) + Q22_matrix(i,j,k))];
% 
%             % Find eigenvalues and eigenvectors
%             [eigvecs, eigvals] = eig(Q_local);
%             eigvals = diag(eigvals);
% 
%             % Find the largest eigenvalue
%             [max_eigval, max_idx] = max(eigvals);
% 
%             % The corresponding eigenvector is the director
%             director = eigvecs(:, max_idx);
% 
%             % Ensure consistent orientation
%             if director(3) < 0
%                 director = -director;
%             end
% 
%             u_final(i,j,k) = director(1);
%             v_final(i,j,k) = director(2);
%             w_final(i,j,k) = director(3);
% 
%             % Calculate theta and phi from director components
%             theta_matrix(i,j,k) = acos(w_final(i,j,k));
%             phi_matrix(i,j,k) = atan2(v_final(i,j,k), u_final(i,j,k));
% 
%             % Order parameter
%             S_final(i,j,k) = (3/2) * max_eigval;
%         end
%     end
% end


% Post-iteration visualization

% Extract director from converged Q-tensor
u_final = zeros(N_Y, N_X, N_Z);
v_final = zeros(N_Y, N_X, N_Z);
w_final = zeros(N_Y, N_X, N_Z);
theta_matrix = zeros(N_Y, N_X, N_Z);
phi_matrix = zeros(N_Y, N_X, N_Z);
S_final = zeros(N_Y, N_X, N_Z);
% psi_matrix = zeros(N, N, N_Z);
% S1_matrix = zeros(N, N, N_Z);
% S2_matrix = zeros(N, N, N_Z);
parfor i = 1:N_Y
    for j = 1:N_X
        for k = 1:N_Z
            % Construct Q-tensor at this point
            Q_local = [Q11_matrix(i,j,k), Q12_matrix(i,j,k), Q13_matrix(i,j,k);
                       Q12_matrix(i,j,k), Q22_matrix(i,j,k), Q23_matrix(i,j,k);
                       Q13_matrix(i,j,k), Q23_matrix(i,j,k), -(Q11_matrix(i,j,k) + Q22_matrix(i,j,k))];
            
            %  % Extract parameters using the equation-solving approach
            % [theta_matrix(i,j,k), phi_matrix(i,j,k), psi_matrix(i,j,k), ...
            %  S1_matrix(i,j,k), S2_matrix(i,j,k)] = extract_parameters_from_Q(Q_local);
             
            % Calculate Cartesian components if needed
            % u_final(i,j,k) = sin(theta_matrix(i,j,k)) * cos(phi_matrix(i,j,k));
            % v_final(i,j,k) = sin(theta_matrix(i,j,k)) * sin(phi_matrix(i,j,k));
            % w_final(i,j,k) = cos(theta_matrix(i,j,k));
            % S_final(i,j,k) = S1_matrix(i,j,k);
            % Find eigenvalues and eigenvectors
              % Diagonalize Q-tensor
            [V, D] = eig(Q_local);
            eigenvalues = diag(D);
            
            % Find eigenvector with largest eigenvalue
            [~, idx] = max(eigenvalues);
            n = V(:, idx);
            
            % Normalize and handle sign ambiguity
            n = n / norm(n);
            if n(3) < 0
                n = -n;
            elseif n(3) == 0
                if n(2) < 0
                    n = -n;
                elseif n(2) == 0 && n(1) < 0
                    n = -n;
                end
            end
            
            % Store director components
            u_final(i,j,k) = n(1);
            v_final(i,j,k) = n(2);
            w_final(i,j,k) = n(3);
            
            % Calculate theta and phi
            theta_matrix(i,j,k) = acos(n(3));
            phi_matrix(i,j,k) = atan2(n(2), n(1));
            % phi_matrix(i,j,k) = mod(phi + pi, 2*pi);  % Ensure 0 ≤ φ < 2π
            
            % Calculate scalar order parameter
            S_final(i,j,k) = (3/2) * max(eigenvalues);
        end
    end
end

% Convert phi to headless representation (0 to pi)
phi_matrix_headless = phi_matrix;
phi_matrix_headless(phi_matrix < 0) = phi_matrix(phi_matrix < 0) + pi;
% theta_matrix2(theta_matrix < 0) = theta_matrix(theta_matrix < 0) + pi;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Save director field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
save('QTensor_director_solution.mat', 'u_final', 'v_final', 'w_final', 'theta_matrix', 'phi_matrix');
save(strcat('\QTensor_director_solution-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.mat'), 'u_final', 'v_final', 'w_final', 'theta_matrix', 'phi_matrix');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if plot_result==1

    % Parameters
    target_points = 40;     % Target resolution for 3D visualization
    num_z_slices = 6;       % Number of z slices to show
    
    % Get original dimensions
    [Nx, Ny, Nz] = size(phi_matrix_headless);
    
    % Implement adaptive downsampling for 3D visualization
    max_dim = max([Nx, Ny, Nz]);
    if max_dim > target_points
        ds_factor = ceil(max_dim / target_points);
        
        % Create downsampled indices
        x_idx = 1:ds_factor:Nx;
        y_idx = 1:ds_factor:Ny;
        z_idx = 1:ds_factor:Nz;
        
        % Downsample matrices and grid for visualization
        phi_viz = phi_matrix_headless(x_idx, y_idx, z_idx);
        theta_viz = theta_matrix(x_idx, y_idx, z_idx);
        S_viz=S_final(x_idx, y_idx, z_idx);
        a_viz = a(x_idx, y_idx, z_idx);
        b_viz = b(x_idx, y_idx, z_idx);
        c_viz = c(x_idx, y_idx, z_idx);
    else
        % No downsampling needed
        phi_viz = phi_matrix_headless;
        theta_viz = theta_matrix;
        S_viz=S_final;
        a_viz = a;
        b_viz = b;
        c_viz = c;
    end
    
    % Calculate evenly spaced z slices
    z_min = min(c_viz(:));
    z_max = max(c_viz(:));
    z_slices = linspace(z_min, z_max, num_z_slices);
    
    % Plot theta distribution with downsampled data
    figure('Name', 'Theta Distribution');
    slice(a_viz, b_viz, c_viz, theta_viz, [], [], z_slices);
    title('Theta Distribution');
    colorbar;
    xlabel('x')
    ylabel('y')
    zlabel('z')
    % caxis([-1*pi pi]);
    colormap(jet);
    
    % Plot phi distribution with downsampled data
    FIG = figure('Name', 'Phi Distribution');
    slice(a_viz, b_viz, c_viz, phi_viz, [], [], z_slices);
    title('Phi Distribution');
    colorbar;
    xlabel('x')
    ylabel('y')
    zlabel('z')
    caxis([0 pi]);
    colormap(hsv);
    saveas(FIG,strcat('C:\Users\pnosratkhah\OneDrive - HKUST Connect\Liquid Crystal Displays\Initial Proposal\Q tensor\Q_tensor_simulation_Pouya\result\Phi_stack\Phi_stack-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.jpg'))

    % Plot order parameter distribution
    figure('Name', 'Order Parameter Distribution');
    slice(a_viz, b_viz, c_viz, S_viz, [], [], z_slices);
    title('Order Parameter S Distribution');
    xlabel('x')
    ylabel('y')
    zlabel('z')
    colorbar;
    caxis([0 max(S_final(:))]);
    
    % Create 1D plots at middle x,y point (with full z resolution)
    mid_x = round(Nx/2);
    mid_y = round(Ny/2);
    
    % Extract z-profiles (full resolution)
    phi_z_profile = squeeze(phi_matrix_headless(mid_x, mid_y, :));
    theta_z_profile = squeeze(theta_matrix(mid_x, mid_y, :));
    z_values = squeeze(c(mid_x, mid_y, :));
    
    % Plot phi z-profile
    FIG = figure('Name', 'Phi Z-Profile at Middle Point');
    plot(z_values, phi_z_profile);
    title(['Phi Z-Profile at x=', num2str(a(mid_x,mid_y,1)), ', y=', num2str(b(mid_x,mid_y,1))]);
    xlabel('z');
    ylabel('Phi Value');
    caxis([0 pi]);
    grid on;
        saveas(FIG,strcat('C:\Users\pnosratkhah\OneDrive - HKUST Connect\Liquid Crystal Displays\Initial Proposal\Q tensor\Q_tensor_simulation_Pouya\result\Phi_value_inCenter\Phi_value_inCenter-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.jpg'))
    % Plot theta z-profile
    figure('Name', 'Theta Z-Profile at Middle Point');
    plot(z_values, theta_z_profile);
    title(['Theta Z-Profile at x=', num2str(a(mid_x,mid_y,1)), ', y=', num2str(b(mid_x,mid_y,1))]);
    xlabel('z');
    ylabel('Theta Value');
    caxis([0 pi]);
    grid on;
    
    
    % Plot headless director field at different heights as lines
    figure('Name', 'LC Director Field at Different Heights');
    heights_to_plot = [1, round(N_Z/5),round(2*N_Z/5), round(3*N_Z/5), round(4*N_Z/5), N_Z];
    height_labels = {'Bottom Surface', '20%', '40%','60%','80%', 'Top Surface'};
    
    for idx = 1:6
        subplot(2,3,idx);
        k = heights_to_plot(idx);
        
        hold on;
        % Plot headless directors as lines centered at grid points
        line_length = 3*(L_X+L_Y)/400; % Adjust this value to change line length
        
        for i = 1:2:N_Y % Step by 2 for clearer visualization
            for j = 1:2:N_X
                % Get director components in x-y plane
                dx = u_final(i,j,k);
                dy = v_final(i,j,k);
                
                % Normalize to ensure consistent line length
                norm_factor = sqrt(dx^2 + dy^2);
                if norm_factor > 1e-40
                    dx = dx / norm_factor;
                    dy = dy / norm_factor;
                end
                
                % Plot line centered at grid point (headless - no arrows)
                x_center = a(i,j,k);
                y_center = b(i,j,k);
                
                x_line = [x_center - line_length*dx/2, x_center + line_length*dx/2];
                y_line = [y_center - line_length*dy/2, y_center + line_length*dy/2];
                
                plot(x_line, y_line, 'k-', 'LineWidth', 1);
            end
        end
        
        hold off;
        % axis equal tight;
        xlabel('x');
        ylabel('y');
        title([height_labels{idx}, ' (z = ', num2str(c(1,1,k)), ')']);
        grid on;
    end
    
    % Plot phi component at different heights
    FIG = figure('Name', 'Phi Component at Different Heights');
    for idx = 1:6
        subplot(2,3,idx);
        k = heights_to_plot(idx);
        
        % Use headless phi (0 to pi)
        imagesc(squeeze(phi_matrix_headless(:,:,k)));
        colorbar;
        caxis([0 pi]);
        colormap(hsv);
        axis equal;
        xlabel('x index');
        ylabel('y index');
        title([height_labels{idx}]);
    end
    saveas(FIG,strcat('C:\Users\pnosratkhah\OneDrive - HKUST Connect\Liquid Crystal Displays\Initial Proposal\Q tensor\Q_tensor_simulation_Pouya\result\Phi_value_layer\Phi_value_layer-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.jpg'))

    
    % Plot theta component at different heights
    FIG = figure('Name', 'Theta Component at Different Heights');
    for idx = 1:6
        subplot(2,3,idx);
        k = heights_to_plot(idx);
        
        imagesc(squeeze(theta_matrix(:,:,k)));
        colorbar;
        caxis([0 pi]);
        colormap(jet);
        axis equal;
        xlabel('x index');
        ylabel('y index');
        title([height_labels{idx}]);
    end
        saveas(FIG,strcat('C:\Users\pnosratkhah\OneDrive - HKUST Connect\Liquid Crystal Displays\Initial Proposal\Q tensor\Q_tensor_simulation_Pouya\result\Theta_layer\Theta_layer-',Simulation_name,'-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),'-Lx_',num2str(L_X),'-Ly_',num2str(L_Y),'-Lz_',num2str(Lz),'-Nx_',num2str(N_X),'-Ny_',num2str(N_Y),'-Nz_',num2str(N_Z),'.jpg'))
    % Plot the converged 3D director field with headless lines
    figure('Name', 'Converged 3D LC Director Field');
    hold on;
    line_length = (L_X+L_Y)/(target_points*3); % Scale factor for 3D visualization
    
    % Plot bottom and top surfaces with different colors
    for i = 1:floor(N_Y/target_points):N_Y % Step by 3 for clearer visualization
        for j = 1:floor(N_X/target_points):N_X
            % Bottom surface (red)
            k = 1;
            x_center = a(i,j,k);
            y_center = b(i,j,k);
            z_center = c(i,j,k);
            
            % Create headless line
            x_line = [x_center - line_length*u_final(i,j,k)/2, x_center + line_length*u_final(i,j,k)/2];
            y_line = [y_center - line_length*v_final(i,j,k)/2, y_center + line_length*v_final(i,j,k)/2];
            z_line = [z_center - line_length*w_final(i,j,k)/2, z_center + line_length*w_final(i,j,k)/2];
            plot3(x_line, y_line, z_line, 'r-', 'LineWidth', 2);
            
            % Top surface (blue)
            k = N_Z;
            x_center = a(i,j,k);
            y_center = b(i,j,k);
            z_center = c(i,j,k);
            
            x_line = [x_center - line_length*u_final(i,j,k)/2, x_center + line_length*u_final(i,j,k)/2];
            y_line = [y_center - line_length*v_final(i,j,k)/2, y_center + line_length*v_final(i,j,k)/2];
            z_line = [z_center - line_length*w_final(i,j,k)/2, z_center + line_length*w_final(i,j,k)/2];
            plot3(x_line, y_line, z_line, 'b-', 'LineWidth', 2);
        end
    end
    
    % Plot bulk (green)
    for i = 1:floor(N_Y/target_points):N_Y % Sparse sampling for bulk
        for j = 1:floor(N_X/target_points):N_X
            for k = 2:floor((N_Z-2)/5)+1:N_Z-1 % Middle layers
                x_center = a(i,j,k);
                y_center = b(i,j,k);
                z_center = c(i,j,k);
                
                x_line = [x_center - line_length*u_final(i,j,k)/2, x_center + line_length*u_final(i,j,k)/2];
                y_line = [y_center - line_length*v_final(i,j,k)/2, y_center + line_length*v_final(i,j,k)/2];
                z_line = [z_center - line_length*w_final(i,j,k)/2, z_center + line_length*w_final(i,j,k)/2];
                plot3(x_line, y_line, z_line, 'g-', 'LineWidth', 2);
            end
        end
    end
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Converged 3D LC Director Field');
    legend('Bottom BC', 'Top BC', 'Bulk', 'Location', 'best');
    grid on;
    % axis equal;
    view(45, 30);
    hold off;

end

disp(strcat('Q_tensor-K11_',num2str(K11),'-K22_',num2str(K22),'-K33_',num2str(K33),'-K24_',num2str(K24),' visualization complete.'));
disp(['Final order parameter range: [', num2str(min(S_final(:))), ', ', num2str(max(S_final(:))), ']']);
disp(['Phi range (original): [', num2str(min(phi_matrix(:))), ', ', num2str(max(phi_matrix(:))), '] rad']);
disp(['Phi range (headless): [', num2str(min(phi_matrix_headless(:))), ', ', num2str(max(phi_matrix_headless(:))), '] rad']);
disp(['Theta range: [', num2str(min(theta_matrix(:))), ', ', num2str(max(theta_matrix(:))), '] rad']);
% close all;
% end
% Levi-Civita function
function eps = get_levi_civita(i, j, k)
    if i == j || j == k || i == k
        eps = 0;
    elseif (i == 1 && j == 2 && k == 3) || (i == 2 && j == 3 && k == 1) || (i == 3 && j == 1 && k == 2)
        eps = 1;
    elseif (i == 3 && j == 2 && k == 1) || (i == 2 && j == 1 && k == 3) || (i == 1 && j == 3 && k == 2)
        eps = -1;
    else
        eps = 0;
    end
end
function energy = calculate_elastic_energy(q1, q2, q3, q4, q5, L1, L2, L3, dx, dy, dz)
    % Calculate gradients using finite differences
    % For each Q tensor component
    [dq1dx, dq1dy, dq1dz] = calculate_gradients(q1, dx, dy, dz);
    [dq2dx, dq2dy, dq2dz] = calculate_gradients(q2, dx, dy, dz);
    [dq3dx, dq3dy, dq3dz] = calculate_gradients(q3, dx, dy, dz);
    [dq4dx, dq4dy, dq4dz] = calculate_gradients(q4, dx, dy, dz);
    [dq5dx, dq5dy, dq5dz] = calculate_gradients(q5, dx, dy, dz);
    
    % Calculate q6 = -q1-q4 and its derivatives
    q6 = -q1 - q4;
    dq6dx = -dq1dx - dq4dx;
    dq6dy = -dq1dy - dq4dy;
    dq6dz = -dq1dz - dq4dz;
    
    % First term: L₁(∂ₖQᵢⱼ)(∂ₖQᵢⱼ)
    term1 = dq1dx.^2 + dq1dy.^2 + dq1dz.^2 + dq2dx.^2 + dq2dy.^2 + dq2dz.^2 + dq3dx.^2 + dq3dy.^2 + dq3dz.^2 + dq4dx.^2 + dq4dy.^2 + dq4dz.^2 + dq5dx.^2 + dq5dy.^2 + dq5dz.^2 + dq6dx.^2 + dq6dy.^2 + dq6dz.^2;
    
    % For full three-constant elastic energy, you'll need the other terms
    % (These are more complex to implement and depend on how you index your Q tensor)
    
    % For one-constant approximation, just use:
    energy_density = L1 * term1;
    % If using all three constants, would be:
    % energy_density = L1*term1 + L2*term2 + L3*term3;
    
    % Total energy (sum over all grid points)
    energy = sum(energy_density(:)) * dx * dy * dz; % Cell volume
end

function [dfdx, dfdy, dfdz] = calculate_gradients(f, dx, dy, dz)
    % Central difference for interior points
    dfdx = zeros(size(f));
    dfdy = zeros(size(f));
    dfdz = zeros(size(f));
    
    % x-derivatives
    dfdx(2:end-1,:,:) = (f(3:end,:,:) - f(1:end-2,:,:)) / (2*dx);
    % y-derivatives
    dfdy(:,2:end-1,:) = (f(:,3:end,:) - f(:,1:end-2,:)) / (2*dy);
    % z-derivatives
    dfdz(:,:,2:end-1) = (f(:,:,3:end) - f(:,:,1:end-2)) / (2*dz);
    
    % Forward/backward differences for boundaries
    % x-boundaries
    dfdx(1,:,:) = (f(2,:,:) - f(1,:,:)) / dx;
    dfdx(end,:,:) = (f(end,:,:) - f(end-1,:,:)) / dx;
    
    % y-boundaries
    dfdy(:,1,:) = (f(:,2,:) - f(:,1,:)) / dy;
    dfdy(:,end,:) = (f(:,end,:) - f(:,end-1,:)) / dy;
    
    % z-boundaries
    dfdz(:,:,1) = (f(:,:,2) - f(:,:,1)) / dz;
    dfdz(:,:,end) = (f(:,:,end) - f(:,:,end-1)) / dz;
end
function [theta, phi, psi, S1, S2] = extract_parameters_from_Q(Q_tensor)
    % This function extracts theta, phi, psi, S1, S2 from the Q tensor
    % by solving the system of 5 equations directly
    
    % Extract the 5 independent components from Q tensor
    q1 = Q_tensor(1,1);
    q2 = Q_tensor(1,2);
    q3 = Q_tensor(1,3);
    q4 = Q_tensor(2,2);
    q5 = Q_tensor(2,3);
    
    % Set optimization options
    options = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', 1000);
    
    % Parameter bounds
    lb = [0, 0, 0, -1, -1];      % Lower bounds for theta, phi, psi, S1, S2
    ub = [pi, 2*pi, 2*pi, 1, 1]; % Upper bounds
    
    % Define residual function (the difference between calculated and actual q values)
    residual_func = @(x) calculate_residuals(x, q1, q2, q3, q4, q5);
    
    % Try multiple starting points to avoid local minima
    num_starts = 10;
    best_error = Inf;
    best_params = [];
    
    for i = 1:num_starts
        % Generate random starting point
        x0 = lb + (ub - lb) .* rand(5, 1);
        
        % Run optimization
        [x_opt, resnorm] = lsqnonlin(residual_func, x0, lb, ub, options);
        
        % Keep best solution
        if resnorm < best_error
            best_error = resnorm;
            best_params = x_opt;
        end
    end
    
    % Extract final parameters
    theta = best_params(1);
    phi = best_params(2);
    psi = best_params(3);
    S1 = best_params(4);
    S2 = best_params(5);
    
    % Print error level for debugging
    fprintf('Parameter extraction error: %e\n', best_error);
    
    % Calculate director in Cartesian coordinates
    u = sin(theta) * cos(phi);
    v = sin(theta) * sin(phi);
    w = cos(theta);
    
    fprintf('Director (u,v,w): (%f, %f, %f)\n', u, v, w);
end

function residuals = calculate_residuals(x, q1_actual, q2_actual, q3_actual, q4_actual, q5_actual)
    % Extract parameters from optimization vector
    theta = x(1);
    phi = x(2);
    psi = x(3);
    S1 = x(4);
    S2 = x(5);
    
    % Calculate q values using the formulas from the paper
    q1 = S1 * cos(theta)^2 * cos(phi)^2 + S2 * (sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2);
    
    q2 = S1 * cos(theta)^2 * sin(phi) * cos(phi) - S2 * (cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)) * (sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta));
    
    q3 = S1 * sin(theta) * cos(theta) * cos(phi) + S2 * sin(psi) * cos(theta) * (sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta));
    
    q4 = S1 * cos(theta)^2 * sin(phi)^2 + S2 * (cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))^2 - (1/3)*(S1 + S2);
    
    q5 = S1 * cos(theta) * sin(theta) * sin(phi) - S2 * sin(psi) * cos(theta) * (cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta));
    
    % Return the residuals (differences between calculated and actual values)
    residuals = [q1 - q1_actual; q2 - q2_actual; q3 - q3_actual; q4 - q4_actual; q5 - q5_actual];
end

function [dQ_dx, dQ_dy, dQ_dz, d2Q_dx2, d2Q_dy2, d2Q_dz2, d2Q_dxdy, d2Q_dxdz, d2Q_dydz] = compute_derivatives(Q, ip1, im1, jp1, jm1, k_internal, dx_inv, dy_inv, dz_inv, dx2_inv, dy2_inv, dz2_inv, dxdy_inv, dxdz_inv, dydz_inv, Nx, Ny, Nz)
    % First derivatives
    dQ_dx = (Q(ip1, :, :) - Q(im1, :, :)) * dx_inv;
    dQ_dy = (Q(:, jp1, :) - Q(:, jm1, :)) * dy_inv;
    dQ_dz = zeros(Nx, Ny, Nz);
    dQ_dz(:, :, k_internal) = (Q(:, :, k_internal+1) - Q(:, :, k_internal-1)) * dz_inv;
    
    % Second derivatives
    d2Q_dx2 = (Q(ip1, :, :) - 2*Q + Q(im1, :, :)) * dx2_inv;
    d2Q_dy2 = (Q(:, jp1, :) - 2*Q + Q(:, jm1, :)) * dy2_inv;
    d2Q_dz2 = zeros(Nx, Ny, Nz);
    d2Q_dz2(:, :, k_internal) = (Q(:, :, k_internal+1) - 2*Q(:, :, k_internal) + Q(:, :, k_internal-1)) * dz2_inv;
    
    % Mixed derivatives
    d2Q_dxdy = (Q(ip1, jp1, :) - Q(ip1, jm1, :) - Q(im1, jp1, :) + Q(im1, jm1, :)) * dxdy_inv;
    d2Q_dxdz = zeros(Nx, Ny, Nz);
    d2Q_dxdz(:, :, k_internal) = (Q(ip1, :, k_internal+1) - Q(ip1, :, k_internal-1) - Q(im1, :, k_internal+1) + Q(im1, :, k_internal-1)) * dxdz_inv;
    d2Q_dydz = zeros(Nx, Ny, Nz);
    d2Q_dydz(:, :, k_internal) = (Q(:, jp1, k_internal+1) - Q(:, jp1, k_internal-1) - Q(:, jm1, k_internal+1) + Q(:, jm1, k_internal-1)) * dydz_inv;
end

function yl = local_makeYL(y, center)
% Returns safe [ymin ymax] with ymax>ymin even if y is flat/NaN/Inf.

y = y(:);
y = y(isfinite(y) & ~isnan(y));

c = double(center);
if ~isfinite(c), c = 0; end

if isempty(y)
    yl = [c-1, c+1];
    return
end

ymin = min(y); ymax = max(y);

if ~(isfinite(ymin) && isfinite(ymax))
    yl = [c-1, c+1];
    return
end

if ymax > ymin
    pad = 0.05*(ymax - ymin);
    yl = [ymin-pad, ymax+pad];
else
    d = max(1e-12, 0.05*max(1,abs(c)));
    yl = [c-d, c+d];
end

% final guard
if ~(isfinite(yl(1)) && isfinite(yl(2))) || yl(2) <= yl(1)
    yl = [c-1, c+1];
end
end
