clc;
clear;
close all;

%Load boundry condition
load('qplate_mat_40x40.mat','img_s');
mat = double(img_s);

%equatios for analysing the elastic energy
syms k1 k2 k3 theta(x,y,z) phi(x,y,z) x y z
assume(k1,'positive')
assume(k2,'positive')
assume(k3,'positive')
L=[cos(phi(x,y,z))*sin(theta(x,y,z));sin(phi(x,y,z))*sin(theta(x,y,z));cos(theta(x,y,z))];
F_elastic = 0.5*(k1*(divergence(L))^2+k2*(dot(L,curl(L)))^2+k3*(cross(L,curl(L))).^2);
F_Body = sum(sum(F_elastic));
eqn = functionalDerivative(F_Body,[theta(x,y,z) phi(x,y,z)]) == 0

% Define the grid size and number of points
L = 800; % Length of the box in each dimension
Lz = 100;
s=1;
N = 40/s; % Number of grid points in each dimension
N_Z = 10;
dx = L / (N - 1); % Grid spacing
dz = Lz / (N_Z - 1);
% Create the 3D grid
[a, b, c] = meshgrid(linspace(0, L, N), linspace(0, L, N), linspace(0, Lz, N_Z));
% [a, b, c] = meshgrid(1:20,1:20,1:20);

% Initialize theta and phi
theta_matrix = zeros(N, N, N_Z);
phi_matrix = zeros(N, N, N_Z);
theta_matrix = theta_matrix +pi/2;
% phi_matrix(:, :,1)=  deg2rad(mat(:,:)-45);
phi_matrix(:, :,1)= 0;
% phi_matrix(1:s:20, :,1)=  deg2rad(mat(1:s:20,1:s:end)-45);
% phi_matrix(21:s:end, :,1)=  deg2rad((mat(21:s:end,1:s:end)-45)+180);
% phi_matrix= phi_matrix;
% phi_matrix(:, :, N_Z)=pi/2;
% theta_matrix(:, :, N_Z)=0;
% phi_matrix(:, :, N-1)=pi/2;
phi_matrix(:, :, N_Z)=pi/2;

u = cos(phi_matrix) .* sin(theta_matrix);
v = sin(phi_matrix) .* sin(theta_matrix);
w = cos(theta_matrix);
% Choose a fixed vector for cross product
fixed_vector = [0, 0, 1]; % Example: z-axis

% Initialize new vector components for perpendicular vectors
u_perp = zeros(size(u));
v_perp = zeros(size(v));
w_perp = zeros(size(w));

% Calculate perpendicular vector components using cross product
for i = 1:N
    for j = 1:N
        for k = 1:N_Z
            original_vector = [u(i, j, k), v(i, j, k), w(i, j, k)];
            perp_vector = cross(original_vector, fixed_vector);
            % Normalize the perpendicular vector
            perp_vector = perp_vector / norm(perp_vector);
            u_perp(i, j, k) = perp_vector(1);
            v_perp(i, j, k) = perp_vector(2);
            w_perp(i, j, k) = perp_vector(3);
        end
    end
end

% Plot the perpendicular vectors as lines with reduced data points
figure;
hold on;
line_thickness = 15; % Set line thickness to 15
for i = 1:2:N % Step by 2 to reduce data points by a factor of 2
    for j = 1:2:N % Step by 2 to reduce data points by a factor of 2
        k = 1;
        % Calculate the end points of the perpendicular vectors
        x_end = a(i, j, k) + u_perp(i, j, k);
        y_end = b(i, j, k) + v_perp(i, j, k);
        z_end = c(i, j, k) + w_perp(i, j, k);

        % Plot the line representing the perpendicular vector with increased thickness and black color
        plot3([a(i, j, k), x_end], [b(i, j, k), y_end], [c(i, j, k), z_end], 'k', 'LineWidth', line_thickness);
    end
end

% Customize the plot
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Reduced and Thicker Perpendicular 3D Vectors');
grid on;
axis equal;
hold off;
% Set initial conditions (if any)
% theta(:) = initial value; % Example: theta = 1
% phi(:) = initial value; % Example: phi = 0

% Define maximum iterations and tolerance for convergence
max_iter = 10000000;
tol = 1e-7;
% k11=11.1e-12;
% k22=6.5e-12;
% k33=17.1e-12;
k11=1.1;
k22=0.6;
k33=1.7;
iter = 1;
i=1;
% tic
% Iterative solver loop
tic
hWaitbar = waitbar(0, 'Iteration 1', 'Name', 'Solving problem','CreateCancelBtn','delete(gcbf)');

for iter = 1:max_iter
    theta_new = theta_matrix;
    phi_new = phi_matrix;



    [Nx, Ny, Nz] = size(theta_matrix);
    dx2 = dx^2; % Precomputing dx^2 for efficiency
    dz2 = dz^2;
    % Periodic boundary conditions for x and y directions
    ip1 = circshift(1:Nx, -1); % i+1 with wrap-around
    im1 = circshift(1:Nx, 1);  % i-1 with wrap-around
    jp1 = circshift(1:Ny, -1); % j+1 with wrap-around
    jm1 = circshift(1:Ny, 1);  % j-1 with wrap-around

    % Note: k is handled separately because the boundaries (k=1 and k=Nz) are fixed.

    % Indices for the internal k loop (2:Nz-1)
    k_internal = 2:Nz-1;

    % First derivatives
    dtheta_dx = (theta_matrix(ip1, :, :) - theta_matrix(im1, :, :)) / (2 * dx);
    dtheta_dy = (theta_matrix(:, jp1, :) - theta_matrix(:, jm1, :)) / (2 * dx);
    dtheta_dz = (theta_matrix(:, :, k_internal+1) - theta_matrix(:, :, k_internal-1)) / (2 * dz);

    dphi_dx = (phi_matrix(ip1, :, :) - phi_matrix(im1, :, :)) / (2 * dx);
    dphi_dy = (phi_matrix(:, jp1, :) - phi_matrix(:, jm1, :)) / (2 * dx);
    dphi_dz = (phi_matrix(:, :, k_internal+1) - phi_matrix(:, :, k_internal-1)) / (2 * dz);

    % Second derivatives
    d2theta_dx2 = (theta_matrix(ip1, :, :) - 2 * theta_matrix + theta_matrix(im1, :, :)) / dx2;
    d2theta_dy2 = (theta_matrix(:, jp1, :) - 2 * theta_matrix + theta_matrix(:, jm1, :)) / dx2;
    d2theta_dz2 = (theta_matrix(:, :, k_internal+1) - 2 * theta_matrix(:, :, k_internal) + theta_matrix(:, :, k_internal-1)) / dz2;

    d2phi_dx2 = (phi_matrix(ip1, :, :) - 2 * phi_matrix + phi_matrix(im1, :, :)) / dx2;
    d2phi_dy2 = (phi_matrix(:, jp1, :) - 2 * phi_matrix + phi_matrix(:, jm1, :)) / dx2;
    d2phi_dz2 = (phi_matrix(:, :, k_internal+1) - 2 * phi_matrix(:, :, k_internal) + phi_matrix(:, :, k_internal-1)) / dz2;

    % Mixed second derivatives
    d2theta_dxdy = (theta_matrix(ip1, jp1, :) - theta_matrix(ip1, jm1, :) - theta_matrix(im1, jp1, :) + theta_matrix(im1, jm1, :)) / (4 * dx2);
    d2theta_dxdz = (theta_matrix(ip1, :, k_internal+1) - theta_matrix(ip1, :, k_internal-1) - theta_matrix(im1, :, k_internal+1) + theta_matrix(im1, :, k_internal-1)) / (4 * dx*dz);
    d2theta_dydz = (theta_matrix(:, jp1, k_internal+1) - theta_matrix(:, jp1, k_internal-1) - theta_matrix(:, jm1, k_internal+1) + theta_matrix(:, jm1, k_internal-1)) / (4 * dx*dz);

    d2phi_dxdy = (phi_matrix(ip1, jp1, :) - phi_matrix(ip1, jm1, :) - phi_matrix(im1, jp1, :) + phi_matrix(im1, jm1, :)) / (4 * dx2);
    d2phi_dxdz = (phi_matrix(ip1, :, k_internal+1) - phi_matrix(ip1, :, k_internal-1) - phi_matrix(im1, :, k_internal+1) + phi_matrix(im1, :, k_internal-1)) / (4 * dx*dz);
    d2phi_dydz = (phi_matrix(:, jp1, k_internal+1) - phi_matrix(:, jp1, k_internal-1) - phi_matrix(:, jm1, k_internal+1) + phi_matrix(:, jm1, k_internal-1)) / (4 * dx*dz);

    d2phi_dxdz_new = zeros(Nx, Ny, Nz);
    d2phi_dydz_new = zeros(Nx, Ny, Nz);
    d2theta_dxdz_new = zeros(Nx, Ny, Nz);
    d2theta_dydz_new = zeros(Nx, Ny, Nz);
    d2phi_dz2_new = zeros(Nx, Ny, Nz);
    d2theta_dz2_new = zeros(Nx, Ny, Nz);
    dphi_dz_new = zeros(Nx, Ny, Nz);
    dtheta_dz_new = zeros(Nx, Ny, Nz);

    % Copy the values from the original matrices to the new matrices
    d2phi_dxdz_new(:, :, 2:Nz-1) = d2phi_dxdz;
    d2phi_dydz_new(:, :, 2:Nz-1) = d2phi_dydz;
    d2theta_dxdz_new(:, :, 2:Nz-1) = d2theta_dxdz;
    d2theta_dydz_new(:, :, 2:Nz-1) = d2theta_dydz;
    d2phi_dz2_new(:, :, 2:Nz-1) = d2phi_dz2;
    d2theta_dz2_new(:, :, 2:Nz-1) = d2theta_dz2;
    dphi_dz_new(:, :, 2:Nz-1) = dphi_dz;
    dtheta_dz_new(:, :, 2:Nz-1) = dtheta_dz;

    d2phi_dxdz=   d2phi_dxdz_new;
    d2phi_dydz=d2phi_dydz_new;
    d2theta_dxdz=d2theta_dxdz_new;
    d2theta_dydz=d2theta_dydz_new;
    d2phi_dz2=d2phi_dz2_new ;
    d2theta_dz2=d2theta_dz2_new;
    dphi_dz=dphi_dz_new;
    dtheta_dz=dtheta_dz_new ;

    equationStr1=vectorize(lhs(eqn(1)));
    equationStr2=vectorize(lhs(eqn(2)));

    % Define the variable mapping
    varMap = containers.Map(...
            {'diff(phi(x, y, z), x, x)', 'diff(phi(x, y, z), x, y)', 'diff(phi(x, y, z), x, z)', ...
             'diff(phi(x, y, z), y, y)', 'diff(phi(x, y, z), y, z)', 'diff(phi(x, y, z), z, z)', ...
             'diff(phi(x, y, z), x)', 'diff(phi(x, y, z), y)', 'diff(phi(x, y, z), z)', ...
             'phi(x, y, z)', 'diff(theta(x, y, z), x, x)', 'diff(theta(x, y, z), x, y)', ...
             'diff(theta(x, y, z), x, z)', 'diff(theta(x, y, z), y, y)', 'diff(theta(x, y, z), y, z)', ...
             'diff(theta(x, y, z), z, z)', 'diff(theta(x, y, z), x)', 'diff(theta(x, y, z), y)', ...
             'diff(theta(x, y, z), z)', 'theta(x, y, z)', 'k1', 'k2', 'k3'}, ...
            {'d2phi_dx2', 'd2phi_dxdy', 'd2phi_dxdz', 'd2phi_dy2', 'd2phi_dydz', 'd2phi_dz2', ...
             'dphi_dx', 'dphi_dy', 'dphi_dz', 'phi_matrix', 'd2theta_dx2', 'd2theta_dxdy', ...
             'd2theta_dxdz', 'd2theta_dy2', 'd2theta_dydz', 'd2theta_dz2', 'dtheta_dx', 'dtheta_dy', ...
             'dtheta_dz', 'theta_matrix', 'k11', 'k22', 'k33'});

    % Precompute replacements for equationStr1 and equationStr2
    keys = varMap.keys;
    for idx = 1:length(keys)
        eqVar = keys{idx};               % Variable in the equation
        wsVar = varMap(eqVar);           % Corresponding workspace variable name
        varValue = evalin('base', wsVar); % Get the value of the workspace variable
            equationStr1 = strrep(equationStr1, eqVar, wsVar);
            equationStr2 = strrep(equationStr2, eqVar, wsVar);
        % end
    end

    % Evaluate the final equations directly in matrix form
    % Assuming equationStr1 and equationStr2 are valid MATLAB expressions
    eqn1_fd_theta = eval(equationStr1);
    eqn2_fd_phi = eval(equationStr2);
    eqn1_fd_theta(:,:,1) = 0;
    eqn2_fd_phi(:,:,1) = 0;
    eqn1_fd_theta(:,:,Nz) = 0;
    eqn2_fd_phi(:,:,Nz) = 0;
    % Update theta and phi matrices (for the internal k layers only)
    theta_new(:, :, :) = theta_matrix(:, :, :) - eqn1_fd_theta;
    phi_new(:, :, :) = phi_matrix(:, :, :) - eqn2_fd_phi;


    % Check for convergence
    if max(abs(theta_new(:) - theta_matrix(:))) < tol && max(abs(phi_new(:) - phi_matrix(:))) < tol
        disp(['Converged after ', num2str(iter), ' iterations.']);
        theta_matrix = theta_new;
        phi_matrix = phi_new;
        break;
    end
if ~ishandle(hWaitbar)
        % Stop the if cancel button was pressed
        disp('Stopped by user');
        break;
    else
        % Update the wait bar
        waitbar(iter/1000,hWaitbar, ['Iteration ' num2str(iter)]);
end    
disp("itteration: ")
    disp(iter)
    disp("phi error: ")
    disp(max(abs(phi_new(:) - phi_matrix(:))))
    disp("theta error: ")
    disp(max(abs(theta_new(:) - theta_matrix(:))))
    % Update theta and phi for the next iteration
    % disp(theta_matrix)
    % disp(theta_new)
    % disp(phi_matrix)
    % disp(phi_new)
    theta_matrix = theta_new;
    phi_matrix = phi_new;

end
toc
% Visualize the results
u = cos(phi_matrix) .* sin(theta_matrix);
v = sin(phi_matrix) .* sin(theta_matrix);
w = cos(theta_matrix);
% save('uniform_alignment_matrices.mat', 'u', 'v', 'w')
figure;
quiver3(a,b,c,u,v,w);
title('Converged Director Distribution');
% axis equal
colorbar;
xlabel('x')
ylabel('y')
zlabel('z')
figure;
slice(a, b, c, theta_matrix, [], [], 0:5:Lz);
title('Theta Distribution');
colorbar;
xlabel('x')
ylabel('y')
zlabel('z')
figure;
slice(a, b, c, phi_matrix, [], [], 0:5:Lz);
title('Phi Distribution');
colorbar;
xlabel('x')
ylabel('y')
zlabel('z')
figure;
quiver3(a(:,:,1:2),b(:,:,1:2),c(:,:,1:2),u(:,:,1:2),v(:,:,1:2),w(:,:,1:2));
title('Converged Director Distribution (Bottom Slices)');
% axis equal
xlabel('x')
ylabel('y')
zlabel('z')
figure;
slice(a, b, c, theta_matrix, [], [], 0:5:25);
title('Theta Distribution (Bottom Slices)');
colorbar;
xlabel('x')
ylabel('y')
zlabel('z')
figure;
slice(a, b, c, phi_matrix, [], [], 0:5:25);
title('Phi Distribution (Bottom Slices)');
colorbar;
xlabel('x')
ylabel('y')
zlabel('z')