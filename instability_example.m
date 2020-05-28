%%
clear

% A box of 40 km * 40 km * 40 km
% uniform hexahedron with edge length 8 km
% total 5 * 5 * 5 = 125 cells

load('mesh.mat') % use the same symbol as you did
G = 3e10;
nu = 0.25;

%%
num_element = length(x1);

% initalize green's function matrix K_{from strain}_{to stress}
% components index: (1) 11 (2) 12 (3) 13 (4) 22 (5) 23 (6) 33, the same
K = cell(6, 6);
for i = 1: 6
    for j = 1: 6
        K{i,j} = zeros(num_element, num_element);
    end
end

unit_strains = [ % each column denote a unit strain
    1 0 0 0 0 0;
    0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
];

for i = 1: 6
    % for each strain compoent
    unit_strain = unit_strains(:, i);
    for j = 1: num_element % index of source
        [s11, s12, s13, s22, s23, s33] = computeStressVerticalShearZone( ...
            x1, x2, x3, ...
            q1(j), q2(j), q3(j), L(j), T(j), W(j), theta, ...
            unit_strain(1), unit_strain(2), unit_strain(3), unit_strain(4), unit_strain(5), unit_strain(6),...
            G, nu);
        % src j must be in jth column
        % K_{from strain}_{to stress}
        K{i, 1}(:, j) = s11;
        K{i, 2}(:, j) = s12;
        K{i, 3}(:, j) = s13;
        K{i, 4}(:, j) = s22;
        K{i, 5}(:, j) = s23;
        K{i, 6}(:, j) = s33;
    end
end

%%
% construct the big Green's function matrix, K_{from strain}_{to stress}
GK = vertcat(...
    horzcat(K{1, 1}, K{2, 1}, K{3, 1}, K{4, 1}, K{5, 1}, K{6, 1}),...
    horzcat(K{1, 2}, K{2, 2}, K{3, 2}, K{4, 2}, K{5, 2}, K{6, 2}),...
    horzcat(K{1, 3}, K{2, 3}, K{3, 3}, K{4, 3}, K{5, 3}, K{6, 3}),...
    horzcat(K{1, 4}, K{2, 4}, K{3, 4}, K{4, 4}, K{5, 4}, K{6, 4}),...
    horzcat(K{1, 5}, K{2, 5}, K{3, 5}, K{4, 5}, K{5, 5}, K{6, 5}),...
    horzcat(K{1, 6}, K{2, 6}, K{3, 6}, K{4, 6}, K{5, 6}, K{6, 6})...
);

%%
% construct initial condition
% u0(:,:,1) denotes strain, u0(:,:,2) denotes stress
% both strain and stress have <num_element> * <6 components> 
u0 = zeros(num_element, 6, 2);
du0_dt = zeros(size(u0)); % to avoid repeated allocation

% perturbe stress from steady state
u0(:, :, 2) = 1e7;

% effective viscosity, change me!!!
% < 5e19 will be instable within 1000 year !!! 
eta = 1e17;

[t, u] = ode45(@(t, u) myode(t, u, num_element, eta, GK, du0_dt, ...
    true ... % change to `true` if acting on deviatoric stress
    ), [0 1000.0*365*86400], u0);

function dudt = myode(t, u, num_element, eta, GK, du0_dt, isdeviatoric)
    
    % matlab changes the input into a vector, so do some reshape
    uu = reshape(u, num_element, 6, 2);
    
    if ~isdeviatoric
        % strain rate = stress / viscosity
        du0_dt(:,:,1) = uu(:,:,2) / eta;
    else
        % consider deviatoric stress
        sigma_kk = (uu(:, 1, 2) + uu(:, 4, 2) + uu(:, 6, 2)) / 3; % 1, 4, 6 are the diagonal compoenents
        du0_dt(:, 1, 1) = (uu(:, 1, 2) - sigma_kk) / eta;
        du0_dt(:, 2, 1) =  uu(:, 2, 2) / eta;
        du0_dt(:, 3, 1) =  uu(:, 3, 2) / eta;
        du0_dt(:, 4, 1) = (uu(:, 4, 2) - sigma_kk) / eta;
        du0_dt(:, 5, 1) =  uu(:, 5, 2) / eta;
        du0_dt(:, 6, 1) = (uu(:, 6, 2) - sigma_kk) / eta;
    end
    
    % stress rate = Green's function * strain rate, Matrix-Vector product
    du0_dt(:,:,2) = reshape(GK * reshape(du0_dt(:,:,1), [], 1), [], 6);
    
    % again, return a vector of derivatives for matlab
    dudt = du0_dt(:);
    
    max_strain_rate = max(max(abs(du0_dt(:,:,1))));
    
%     You may setup a threhold after which the program exit
%     for eta = 1e19, it took 420 year

    if max_strain_rate > 1e-9 
%         return
    end
    
    fprintf('Year: %f, max strain rate %f \r\n', t/365/86400, max_strain_rate);
end