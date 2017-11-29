function [ data, labels ] = preprocess( filename, svd_dim, unique, to_save, varargin )
% PREPROCESS from .mat file to .csv file
%   Loads .mat file with raw data from ISETBIO, pre-process with svd
%   dimensionality reduction. If unique, select only unique (non-rotated)
%   images.
%
%   Arguments
%       - filename: string with name of file to load
%       - svd_dim: number of dimensions to feed into SVD compression
%       - unique: boolean
%       - to_save: boolean
%       - (optional) num_rotations: number of rotations in dataset
%   
%   Returns
%       - data: pre-processed data, also saved as .csv file
%       - labels: corresponding labels, also saved as .csv file
 
% Add path to data directories
addpath('../Datasets/')
addpath('../Datasets/Single_Cases/')
 
% Load target variables from .mat file
allData = load(filename, 'allDemosaicResponse', 'luminanceLevel');
cone_responses = allData.allDemosaicResponse;
luminance_levels = allData.luminanceLevel;
 
% Number of samples
num_samples = size(cone_responses, 2);
 
% Size of each sample
sample_size = size(cone_responses, 1);
 
% Size of each cone-type channel
channel_size = sqrt(sample_size/3);
 
% If number of rotations is specified, set rotations to that number.
% Otherwise, default value is 11.
if nargin == 5
    rotations = varargin{1};
else
    rotations = 11;
end
 
% if unique is true, only pre-process and save non-rotated images
if unique
    ind = 1:rotations:num_samples;
    cone_responses = cone_responses(:,ind);
    luminance_levels = luminance_levels(:,ind);
    num_samples = size(ind, 2);
end
 
% Downsample data
cone_responses = reshape(cone_responses, channel_size, channel_size, 3, ...
    num_samples);
dInd = 1:3:channel_size;
cone_responses = reshape(cone_responses(dInd, dInd, :, :), [], num_samples);
 
% Compress data through SVD Regression
[data, ~, labels, ~] = svdRegression(cone_responses', luminance_levels', ...
    svd_dim);
 
labels = labels';
 
% Save to .csv file to use in deep learning
if to_save
    % pad data and labels for proper reading in python
    pad = zeros(1, num_samples);
    data = [pad; data];
    labels = [pad; labels];
     
    % filenames for separate data and labels .csv files
    data_filename = sprintf('%s_svd%s.csv', filename(1:end-4), ...
        int2str(svd_dim));
    labels_filename = sprintf('%s_svd_labels.csv', filename(1:end-4));
     
    csvwrite(data_filename, data);
     
    csvwrite(labels_filename, labels)
     
    % Remove pad for return variable
    data = data(2:end, :);
    labels = labels(2, :);
end