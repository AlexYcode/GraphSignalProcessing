% load data & channel labels
load('v1p.mat'); 
channel_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
                  'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'};

% Sampling frequency
fs = 128;

% Preprocessing - Bandpass filter (0.5 Hz to 40 Hz)
low_cutoff = 0.5;
high_cutoff = 40;
[b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');
filtered_eeg = filtfilt(b, a, v1p); 

% Feature Extraction - Average Power in Frequency Bands (PSD)
delta_band = [1 4];
theta_band = [4 8];
alpha_band = [8 13];
beta_band = [13 30];

delta_power = zeros(1, 19);
theta_power = zeros(1, 19);
alpha_power = zeros(1, 19);
beta_power = zeros(1, 19);

for ch = 1:19
    [pxx, f] = pwelch(filtered_eeg(:, ch), [], [], [], fs);
    delta_power(ch) = bandpower(pxx, f, delta_band, 'psd');
    theta_power(ch) = bandpower(pxx, f, theta_band, 'psd');
    alpha_power(ch) = bandpower(pxx, f, alpha_band, 'psd');
    beta_power(ch) = bandpower(pxx, f, beta_band, 'psd');
end

% Construct the feature matrix X for GNN (each row is a channel, each column is a frequency band)
X = [delta_power; theta_power; alpha_power; beta_power]'; 

% Adjacency Matrix A
A = eye(19); 

% GNN Parameters
numNodes = 19;
numFeatures = 4;

% GNN layer
gnnLayer = GraphConvolutionLayer(numNodes, numFeatures, 'gnn_layer');

% Forward pass
Z = predict(gnnLayer, X, A);
disp('Output of GNN Layer:');
disp(Z);

% EEG channel graph based on adjacency matrix
G = graph(A);
figure;
plot(G, 'NodeLabel', channel_labels);
title('EEG Channel Graph');