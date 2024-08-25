close all
clear
clc

m = 0.15;

%% Exercise 0

disp('%%%%% Exercise 0 %%%%%')
% Adiacence matrix 
Adiacence0 = [
    0 1 1 1;
    0 0 1 1;
    1 0 0 0;
    1 0 1 0;
];

A = zeros(size(Adiacence0,1), size(Adiacence0,2));
for j=1:size(Adiacence0,1)
    indices = find(Adiacence0(:,j))';             % selecting pages that are ingoing to page j
    norm = sum(Adiacence0(indices,:), 2)';        % number of outgoing links from pages selected
    A(j,indices) = 1./norm;   % formula 2.1
end
disp('Link Matrix:')
disp(A)
[eigenvectors, eigenvalues] = eig(A, 'vector');

[~, ind_one] = min(abs(eigenvalues-1));
norm_eig = sum(eigenvectors(:,ind_one));
rank_pages = (A * eigenvectors(:,ind_one))./norm_eig;

disp('Eigenvector corresponding to eigenvalue 1:')
disp(rank_pages)

%% Exercise 1 
% Suppose the people who own page 3 in the web of Figure 1 are infuriated  
% by the fact that its importance score, computed using formula (2.1), is 
% lower than the score of page 1. In an attempt to boost page 3’s score, 
% they create a page 5 that links to page 3; page 3 also links to page 5. 
% Does this 0boost page 3’s score above that of page 1?

disp('%%%%% Exercise 1 %%%%%')
% Adiacence matrix 
Adiacence1 = [
    0 1 1 1 0;
    0 0 1 1 0;
    1 0 0 0 1;
    1 0 1 0 0;
    0 0 1 0 0
];

A1 = zeros(size(Adiacence1,1), size(Adiacence1,2));
for j=1:size(Adiacence1,1)
    indices = find(Adiacence1(:,j))';             % selecting pages that are ingoing to page j
    norm = sum(Adiacence1(indices,:), 2)';        % number of outgoing links from pages selected
    A1(j,indices) = 1./norm;   % formula 2.1
end
disp('Link Matrix:')
disp(A1)
[eigenvectors_1, eigenvalues_1] = eig(A1, 'vector');

[~, ind_one] = min(abs(eigenvalues_1-1));
norm_eig_1 = sum(eigenvectors_1(:,ind_one));
rank_pages_1 = (A1 * eigenvectors_1(:,ind_one))./norm_eig_1;

disp('Eigenvector corresponding to eigenvalue 1:')
disp(rank_pages_1)
disp('The 3rd page has an higher rank than before')

%% Exercise 2
% Construct a web consisting of three or more subwebs and verify that
% dim(V1(A)) equals (or exceeds) the number of the components in the web.
disp('%%%%% Exercise 2 %%%%%')
Adiacence2 = [
    0 1 1 1 0 0 0 0 0;
    0 0 1 1 0 0 0 0 0;
    1 0 0 0 0 0 0 0 0;
    1 0 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 1 0 0;
    0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 0 1;
    0 0 0 0 0 0 0 1 0
];

A2 = zeros(size(Adiacence2,1), size(Adiacence2,2));
for j=1:size(Adiacence2,1)
    indices = find(Adiacence2(:,j))';             % selecting pages that are ingoing to page j
    norm = sum(Adiacence2(indices,:), 2)';        % number of outgoing links from pages selected
    A2(j,indices) = 1./norm;   % formula 2.1
end
disp('Link Matrix:')
disp(A2)
[eigenvectors_2, eigenvalues_2] = eig(A2, 'vector');
disp('Eigenvalues: ')
disp(eigenvalues_2) % Presence of more than one eigenvalue equal to 1 => dimension of V(A) > 1
%% Exercise 3
% Add a link from page 5 to page 1 in the web of Figure 2. The resulting 
% web, considered as an undirected graph, is connected. 
% What is the dimension of V1(A)?
disp('%%%%% Exercise 3 %%%%%')
Adiacence3 = [
    0 1 1 1 0;
    0 0 1 1 0;
    1 0 0 0 0;
    1 0 1 0 0;
    1 0 0 0 0 
];

A3 = zeros(size(Adiacence3,1), size(Adiacence3,2));
for j=1:size(Adiacence3,1)
    indices = find(Adiacence3(:,j))';             % selecting pages that are ingoing to page j
    norm = sum(Adiacence3(indices,:), 2)';        % number of outgoing links from pages selected
    A3(j,indices) = 1./norm;   % formula 2.1
end
disp('Link Matrix:')
disp(A3)
[eigenvectors_3, eigenvalues_3] = eig(A3, 'vector');
disp('Eigenvalues: ')
disp(eigenvalues_3) 

%% Exercise 4
% In the web of Figure 2.1, remove the link from page 3 to page 1. In the 
% resulting web page 3 is now a dangling node. Set up the corresponding 
% substochastic matrix and find its largest positive (Perron) eigenvalue. 
% Find a non-negative Perron eigenvector for this eigenvalue, and scale
% the vector so that components sum to one. Does the resulting ranking seem 
% reasonable?
disp('%%%%% Exercise 4 %%%%%')
Adiacence4 = [
    0 1 1 1;
    0 0 1 1;
    0 0 0 0;
    1 0 1 0;
];

A4 = zeros(size(Adiacence4,2), size(Adiacence4,2));
for j=1:size(Adiacence4,1)
    indices = find(Adiacence4(:,j))';            % selecting pages that are ingoing to page j
    norm = sum(Adiacence4(indices,:), 2)';       % number of outgoing links from pages selected
    A4(j,indices) = 1./norm;   % formula 2.1
end

disp('Link Matrix:')
disp(A4)
[eigenvectors_4, eigenvalues_4] = eig(A4, 'vector');
[perron_eigenvalue, ind_perron] = max(eigenvalues_4);
disp('Perron eigenvalue: ')
disp(perron_eigenvalue)
perron_eigenvector = abs(eigenvectors_4(:,ind_perron))/sum(abs(eigenvectors_4(:,ind_perron)));
disp('Perron eigenvector: ')
disp(perron_eigenvector)

%% Exercise 5
% Prove that in any web the importance score of a page with no backlinks is zero

% Generating a random web of a random dimension from 3 to 9 and forcing
% a random column of zeros (-> correspond to a node with no
% backlinks)
disp('%%%%% Exercise 5 %%%%%')
n_11 = randi([3 9], 1);
Adiacence5 = randi([0 1], n_11);
fprintf('Random web dimension: %d\n', n_11)
Adiacence5 = Adiacence5 - diag(diag(Adiacence5)); %forcing the diagonal to 0 (Eliminates the connection with itselfs)
no_back_page_ind = randi([1 n_11],1);
fprintf('Page number %d forced to have no backlinks\n', no_back_page_ind)
Adiacence5(:, no_back_page_ind) = zeros(n_11,1);

A5 = zeros(size(Adiacence5,2), size(Adiacence5,2));
for j=1:size(Adiacence5,1)
    indices = find(Adiacence5(:,j))';            % selecting pages that are ingoing to page j
    norm = sum(Adiacence5(indices,:), 2)';       % number of outgoing links from pages selected
    A5(j,indices) = 1./norm;   % formula 2.1
end

[eigenvectors_5, eigenvalues_5] = eig(A5, 'vector');
[~, ind_one] = min(abs(eigenvalues_5-1));
norm_eig_5 = sum(eigenvectors_5(:,ind_one));
rank_pages_5 = (A5 * eigenvectors_5(:,ind_one))/norm_eig_5;
disp('Page score:')
disp(eigenvectors_5(:,ind_one))
fprintf('Page number %d has importance score equal to %.3f\n', no_back_page_ind, eigenvectors_5(no_back_page_ind,ind_one))
%% Exercise 11
disp('%%%%% Exercise 11 %%%%%')

Adiacence11 = Adiacence1;
A11 = A1;
n_11 = size(A11,1);
S_11 = 1/n_11 * ones(n_11,n_11);
M11 = (1-m)*A11 + m*S_11;

disp("Matrix M11:")
disp(M11)

[eigenvectors_11, eigenvalues_11] = eig(M11, 'vector');

[~, ind_one] = min(abs(eigenvalues_11-1));
norm_eig_11 = sum(eigenvectors_11(:,ind_one));
rank_pages_11 = (M11 * eigenvectors_11(:,ind_one))/norm_eig_11;

disp('Eigenvector of matrix M corresponding to eigenvalue 1:')
disp(rank_pages_11) % change of direction of the eigenvectors

%% Exercise 12
disp('%%%%% Exercise 12 %%%%%')

% Adiacence matrix 
Adiacence12 = [
    0 1 1 1 0 0;
    0 0 1 1 0 0;
    1 0 0 0 1 0;
    1 0 1 0 0 0;
    0 0 1 0 0 0;
    1 1 1 1 1 0
];

A12 = zeros(size(Adiacence12,1), size(Adiacence12,2));
for j=1:size(Adiacence12,1)
    indices = find(Adiacence12(:,j))';             % selecting pages that are ingoing to page j
    norm = sum(Adiacence12(indices,:), 2)';        % number of outgoing links from pages selected
    A12(j,indices) = 1./norm;   % formula 2.1
end

disp("Matrix A12:")
disp(A12)
m = 0.15;
n_12 = size(A12,1);
S_12 = 1/n_12 * ones(n_12,n_12);
M12 = (1-m)*A12 + m*S_12;
disp("Matrix M12:")
disp(M12)

[eigenvectors_A12, eigenvalues_A12] = eig(A12, 'vector');
[~, ind_one] = min(abs(eigenvalues_A12-1));
norm_eig_A12 = sum(eigenvectors_A12(:,ind_one));
rank_pages_A12 = (A12 * eigenvectors_A12(:,ind_one))/norm_eig_A12;
disp('Eigenvector of matrix A corresponding to eigenvalue 1:')
disp(rank_pages_A12)


[eigenvectors_M12, eigenvalues_M12] = eig(M12, 'vector');
[~, ind_one] = min(abs(eigenvalues_M12-1));
norm_eig_M12 = sum(eigenvectors_M12(:,ind_one));
rank_pages_M12 = (M12 * eigenvectors_M12(:,ind_one))/norm_eig_M12;

disp('Eigenvector of matrix M corresponding to eigenvalue 1:')
disp(rank_pages_M12)
%% Exercise 13
disp('%%%%% Exercise 13 %%%%%')

A13 = A2;
n_13 = size(A13,1);
S_13 = 1/n_13 * ones(n_13,n_13);
M13 = (1-m)*A13 + m*S_13;

disp("Matrix M13:")
disp(M13)

[eigenvectors_M13, eigenvalues_M13] = eig(M13, 'vector');
[~, ind_one] = min(abs(eigenvalues_M13-1));
norm_eig_M13 = sum(eigenvectors_M13(:,ind_one));
rank_pages_M13 = (M13 * eigenvectors_M13(:,ind_one))/norm_eig_M13;

disp('Eigenvector of matrix M corresponding to eigenvalue 1:')
disp(rank_pages_M13)
%% Exercise 14
% for k = 1, 5, 10, 50, using an initial guess x0 not too close to the 
% actual eigenvector q (so that you can watch the convergence). 
% Determine c = max1≤j≤n |1 − 2 min1≤i≤n Mij | and the absolute value
% of the second largest eigenvalue of M

disp('%%%%% Exercise 14 %%%%%')
M14 = M11;
q = rank_pages_11;
fprintf('Vector q:\n')
disp(q)
n_14 = length(q);
x0 = randi([3 9], [n_14 1]);
x0 = x0/sum(x0);
fprintf('Initialized random vector x0:\n')
disp(x0)

k_vec = [1 5 10 50];
eig_approx_matr = zeros(n_14, length(k_vec));
den = 0;

fprintf('k \t| \t||M^k * x0 - q|| \t\t| Eigenvalue\n')
fprintf('---------------------------------------------\n')
fprintf('0 \t| \t\t%.2f \t\t\t\t|\n', sum(abs(x0-q)))


for k=0:k_vec(end)
    num = sum(abs(M14^k * x0 - q));
    if den ~= 0
        ratio = num/den;
        den = num;
        if k == k_vec(1) || k == k_vec(2) || k == k_vec(3) || k == k_vec(4)
            
            fprintf('%d \t| \t\t%e \t\t|\t %.4f\n', k, num, ratio);
        end
    else
        den = num;
    end
    
end
sec_larg = flip(sort(eigenvalues_11));
fprintf('\nSecond largest eigenvalue of M14: %.4f\n', abs(sec_larg(2)));
% calculating c
c_vector = zeros(size(M14,1),1);
for k=1:size(M14,1)
    
    c_vector(k) = abs(1-2*min(M14(:,k)));
    
end
c = max(c_vector);
fprintf("value of c: %.3f\n", c)

%% Exercise 16
% Show that M = (1 − m)A + mS (all Sij = 1/3) is not diagonalizable 
% for 0 ≤ m < 1

disp('%%%%% Exercise 16 %%%%%')
% Define the matrices A16 and S16
A16 = [0 1/2 1/2;
       0 0 1/2;
       1 1/2 0];

syms m16;
S16 = 1/3 * ones(3,3);
M16 = (1-m16)*A16 + m16*S16;
disp("Matrix M16:")
disp(M1S6)
[eigenvectors_M16, eigenvalues_M16] = eig(M16);
eigenvalues_vector = diag(eigenvalues_M16);

% algebraic multiplicities
[unique_eigenvalues, ~, index] = unique(eigenvalues_vector);
algebraic_multiplicities = histc(index, unique(index));

for i = 1:length(unique_eigenvalues)
    
    % Get the current eigenvalue and its algebraic multiplicity
    eigenvalue = unique_eigenvalues(i);
    multiplicity = algebraic_multiplicities(i);
    
    corresponding_indices = find(eigenvalues_vector == eigenvalue, 1);
    corresponding_eigenvectors = eigenvectors_M16(:, corresponding_indices);
    
    % geometric multiplicity
    null_space = null(M16 - eigenvalue * eye(size(M16)));
    geometric_multiplicity = size(null_space, 2);
    
    % Convert eigenvalue to string for display
    eigenvalue_str = char(eigenvalue);
    
    disp(['Eigenvalue: ', eigenvalue_str, ...
          ', Algebraic multiplicity: ', num2str(multiplicity), ...
          ', Geometric multiplicity: ', num2str(geometric_multiplicity), ...
          ', corresponding eigenvector(s):']);
    disp(corresponding_eigenvectors);
end
