% Extracting rhythm information from intermittent data using linear functions
% Note: When using this code to extract rhythm features from intermittent waveforms, depending on the waveform,
% By adjusting the selection criteria for y_scale (synchronous longitudinal and transverse stretching) and maximum values
% Adjust the sensitivity of the algorithm to the boundaries between waveforms and gaps by adjusting the range of neighborhood and lateral stretching
% Load data
filePath = 'data/BFG_4.xlsx';
% Read the first column of data using the xlSread function
data = xlsread(filePath, 1, 'A:A');
% Read the first 1000 pieces of data
len_sample = 1000;
data_sample = data(1:len_sample)';

%% Generate matching functions under different time shifts and stretches
% alpha = [-1, -0.6, -0.3, 0, 0.3, 0.6, 1];
% Using Haar Scale Function
alpha = [0];
beta = 1-alpha*0.5;
% Generate x value range
x = linspace(0, 1, 1000);
y_scale = 1;
pdf_values = zeros(length(alpha), 1000);
% Cycle to draw the Beta distribution curve for each parameter combination
for i = 1:length(alpha)
    % Cycle to draw the Beta distribution curve for each parameter combination
    pdf_values(i, :) = y_scale*(alpha(i)*x+beta(i));
    subplot(length(alpha), 1, i);
    plot(x, pdf_values, 'b-', 'LineWidth', 2);
    title(['Linear (alpha = ', num2str(alpha(i)), ', beta = ', num2str(beta(i)), ')']);
    xlabel('x');
    ylabel('function value');
    grid on;
    hold on;
end

all_coefs = {};
%% Function matching original curve
for i= 1:length(alpha)
    psi_normal = pdf_values(i,:);
    ind = 1;
    size_wav = size(psi_normal);
    lenth_wav = size_wav(2);
    lenth_wav_st = 20;
    nb_SCALES = 35;
    start_SCALES = 5;
    scales = start_SCALES:nb_SCALES;
    lenSIG = len_sample;
    coefs = zeros(nb_SCALES-start_SCALES,lenSIG);
    for k = 1:(nb_SCALES-start_SCALES+1)
        a = scales(k);
        % Pay attention to adjusting the sensitivity of the algorithm to boundaries by adjusting the score index after y_scale!!!
        y_scale = (lenth_wav_st/(a+1))^(4/5);
        indices = round(linspace(1, lenth_wav, a+1));    
        f = y_scale*fliplr(psi_normal(indices));
        coefs(ind,:) = wkeep1(wconv1(data_sample,f),lenSIG);
        % coefs(ind,:) = y_scale*wkeep1(diff(wconv1(LDG_data_sample,f)),lenSIG);
        ind = ind+1;
        if ind==300
            stop = 1;
        end
    end
    all_coefs{end + 1} = coefs;
end

% Draw 3D images of coefs; Draw the matching degree result
for i= 1:length(alpha)
    coefs = all_coefs{i};
    figure;
    surf(1:lenSIG, scales, coefs);
    xlabel('time');
    ylabel('scale');
    zlabel('The value of coefs');
    title('3D images of coefs');
end
% Save wavelet coefficients
% Save data to an Excel file
% outputFileName = 'coefs.xlsx'; %Set the file name to be saved
% xlswrite(outputFileName, coefs, 'Sheet1'); %Write coefs data to Sheet1 worksheet in Excel

%% Obtain time-frequency features (find corresponding feature coordinates based on maximum matching degree)
all_maxValues = {};
all_rowIndices = {};
for i= 1:length(alpha)
    coefs = all_coefs{i};
    [maxValues, rowIndices] = findMaxValues(coefs);
    all_maxValues{end+1} = maxValues;
    all_rowIndices{end+1} = rowIndices;
end

% Combine to obtain the maximum curve
[maxValues, vectorIndices] = find_all_MaxValues(all_maxValues);
len_maxValues = 1:numel(maxValues);
figure;
plot(len_maxValues, maxValues);
xlabel('column index');
ylabel('Maximum element value');
title('Maximum element value per column');
% Add gridlines
grid on;
% Obtain maximum value
[maxima] = findMaxima(maxValues);
% Retrieve the time shift (index) corresponding to the maximum value
max_times = maxima(:,1);
max_types = vectorIndices(max_times);
max_amplitudes = maxima(:,2);
[sum_scales] = type_time2scale(max_times, max_types, all_rowIndices);

%% Combining time-frequency coordinates with convolution and reconstruction of original data, and comparing them
[amplitudes] = get_amplitude(max_amplitudes, sum_scales, scales, lenth_wav_st);
% vector_chonggou = Plot_chonggou_separately(max_times, sum_scales, scales, max_types, amplitudes, pdf_values, len_sample, lenth_wav, BFG_data_sample, maxValues)
[chonggou_data] = Beta_chonggou(max_times, sum_scales, scales, max_types, amplitudes, pdf_values, len_sample, lenth_wav);
len_modifiedVector = 1:numel(chonggou_data);
figure;
plot(len_modifiedVector, chonggou_data/10, 'r');
xlabel('Time/min');
ylabel('Recycling quantity');
grid on;
hold on;
plot(len_modifiedVector, data_sample, 'b');
plot(len_modifiedVector, maxValues/10, 'g');

%% Annotate the rhythm information extracted from the reconstructed image on the horizontal axis
[nonZeroIntervals, zeroIntervals] = analyzeRectangularWave(chonggou_data, len_modifiedVector);
drawRectangles(zeroIntervals, 10, 'cyan');
drawRectangles(nonZeroIntervals, 10, 'red');

outputData1 = extractWaveformSegment(data_sample, nonZeroIntervals);
plotWaveforms(outputData1, 1, 3);

outputData2 = extractWaveformSegment(data_sample, zeroIntervals);
plotWaveforms(outputData2, 1, 3);

%% 函数
function [maxValues, rowIndices] = findMaxValues(coefs)
    % Get the number of rows and columns of the input matrix
    [numRows, numCols] = size(coefs);
    
    % Initialize the vector that stores the maximum value and line number
    maxValues = zeros(1, numCols);
    rowIndices = zeros(1, numCols);
    
    % Traverse each column
    for col = 1:numCols
        % Find the maximum value of the current column and its row number in the column
        [maxValue, rowIndex] = max(coefs(:, col));
        
        % Store the maximum value and line number into the corresponding vector
        maxValues(col) = maxValue;
        rowIndices(col) = rowIndex;
    end
end


function [maxValues, vectorIndices] = find_all_MaxValues(cellArray)
    % Initialization result vector
    maxValues = zeros(1, length(cellArray{1}));
    vectorIndices = zeros(1, length(cellArray{1}));
    
    % Traverse each vector in the cell array
    for i = 1:length(cellArray{1})
        [maxValue, vectorIndex] = findMaxValueAtIndex(cellArray, i);
        maxValues(1, i) = maxValue;
        vectorIndices(1, i) = vectorIndex;
    end
end

function [maxValue, vectorIndex] = findMaxValueAtIndex(cellArray, idx)
    % Initialization result
    maxValue = -inf; % Using negative infinity as the initial maximum value
    vectorIndex = 0; % Initialize vector number
    
    % Traverse each vector in the cell array
    for i = 1:length(cellArray)
        currentVector = cellArray{i}; % Get the current vector
        
        % Check if the vector length is long enough to contain the idx position
        if numel(currentVector) >= idx
            % Retrieve the elements in the vector at the idx position
            elementAtIndex = currentVector(idx);
            
            % If the current element is greater than the maximum value, update the maximum value and vector number
            if elementAtIndex > maxValue
                maxValue = elementAtIndex;
                vectorIndex = i;
            end
        end
    end
    
    % If the maximum value cannot be found (possibly due to idx exceeding the vector range), the vector number is 0
    if vectorIndex == 0
        maxValue = NaN; % When the maximum value is not found, set the maximum value to NaN
    end
end

% Output the maximum value of the time series and its corresponding index
function [maxima] = findMaxima(vector)
    % Obtain the length of the vector
    n = length(vector);
    
    % Initialize the array storing the maximum value points
    maxima = [];
    % Attention: Adjusting the neighborhood size of the maximum value here has a significant impact on the effectiveness of the algorithm!!!
    max_num = 10;
    
    % Traverse each element in the vector
    for i = max_num+1:(n-max_num)
        % Get the value of the current element
        currentValue = vector(i);
        
        % Check if the current element is a maximum value point
        neighbors = vector(i-max_num:i+max_num);
        leftNeighbors = vector(i-max_num:i-1);
        rightNeighbors = vector(i+1:i+max_num);
        
        isMaxima = currentValue > max(leftNeighbors) && currentValue > max(rightNeighbors);
        
        % If it is a maximum value point, add it to the result array
        if isMaxima
            maxima = [maxima; [i, currentValue]];
        end
    end
end


% Estimate the amplitude of the original waveform based on the matching sum and flatness combined with the degree of stretching
function [vector] = get_amplitude(sum_amplitudes, scale_idx, scales, lenth_wav)
    vector = zeros(numel(scale_idx),1);
    for i = 1:numel(scale_idx)
        amlitude = sum_amplitudes(i);
        scale = scales(scale_idx(i))+1;
        y_scale = (lenth_wav/scale)^(4/5);
        vector(i) = amlitude/(y_scale*scale);
    end
end

% Locate the scale stretching dimension based on two dimensions: time shift and category
function [sum_scales] = type_time2scale(times, types, all_rowIndices)
    sum_scales = [];
    for i = 1:numel(types)
        scales = all_rowIndices{types(i)};
        sum_scales(end+1) = scales(times(i));
    end
end

% Reconstruct the original signal based on the extracted waveform type, stretching degree, time shift, and amplitude information
function [vector_chonggou] = Beta_chonggou(times, scales_idx, scales, types, amplitudes, beta_functions, len_data, len_wav)
    vector_chonggou = zeros(1, len_data);
    x = 1:1:len_data;
    for i = 1:numel(times)
        % Perform type selection, stretching, and time shifting operations on the beta distribution function
        beta_function = beta_functions(types(i),:);
        % stretch
        a = scales(scales_idx(i));
        indices = round(linspace(1, len_wav, a+1)); 
        y_scale = (len_wav/(a+1))^(4/5);
        beta_function = y_scale*beta_function(indices);
        % Longitudinal stretching (amplitude)
        beta_function = amplitudes(i)*beta_function;
        % Time shift
        vector_chonggou = sumVectorsAtIndex(vector_chonggou, beta_function, times(i));
    end
end


function resultVector = sumVectorsAtIndex(a, b, index)
    % Check if the index is valid
    if index < 1 || index > length(a)
        error('Index exceeds the range of vector a');
    end
    % Initialization result vector
    resultVector = a;
    b = flip(b);
    % Calculate the starting and ending indices of the parts that need to be added together
    % Note that the wconv1 and wkeep1 functions need to be considered here!!!!!!!!!!
    index = index  - numel(b)/2;
    startIndex = index;
    endIndex = index + numel(b) - 1;
    % If the ending index exceeds the range of the result vector, truncate it
    if endIndex > length(resultVector)
        % Truncate End Index
        endIndex = length(resultVector);
        % Truncate vector b so that its length matches the truncated result vector
        b = b(1:endIndex - startIndex + 1);
    end
    % Sum vector b with vector a starting from the specified index position
    resultVector(startIndex:endIndex) = resultVector(startIndex:endIndex) + b;
end

function drawRectangles(rectangleIntervals, amplitude, fillColor)
    % Draw a rectangular wave and fill the specified interval
    hold on;
    for i = 1:size(rectangleIntervals, 1)
        start_x = rectangleIntervals(i, 1);
        end_x = rectangleIntervals(i, 2);
        rectangle('Position', [start_x, 0, end_x - start_x, amplitude], 'FaceColor', fillColor, 'EdgeColor', 'none');
    end 
    xlabel('X-axis label');
    ylabel('Y-axis label');
    grid on;
    title('Rectangular wave drawing and filling');
    hold off;
end


function [nonZeroIntervals, zeroIntervals] = analyzeRectangularWave(rectangularWave, xValues)
    % Find the interval where the rectangular wave is not 0
    nonZeroIndices = find(rectangularWave ~= 0);
    nonZeroIntervals = [];

    % Find the interval where the rectangular wave is 0
    zeroIndices = find(rectangularWave == 0);
    zeroIntervals = [];

    % Calculate intervals that are not zero
    start_idx = nonZeroIndices(1);
    for i = 2:length(nonZeroIndices)
        if nonZeroIndices(i) ~= nonZeroIndices(i-1) + 1
            end_idx = nonZeroIndices(i-1);
            nonZeroIntervals = [nonZeroIntervals; [xValues(start_idx), xValues(end_idx)]];
            start_idx = nonZeroIndices(i);
        end
    end
    end_idx = nonZeroIndices(end);
    nonZeroIntervals = [nonZeroIntervals; [xValues(start_idx), xValues(end_idx)]];

    % Calculate intervals that are not zero
    if ~isempty(zeroIndices)
        start_idx = zeroIndices(1);
        for i = 2:length(zeroIndices)
            if zeroIndices(i) ~= zeroIndices(i-1) + 1
                end_idx = zeroIndices(i-1);
                zeroIntervals = [zeroIntervals; [xValues(start_idx), xValues(end_idx)]];
                start_idx = zeroIndices(i);
            end
        end
        end_idx = zeroIndices(end);
        zeroIntervals = [zeroIntervals; [xValues(start_idx), xValues(end_idx)]];
    end
end

function outputData = extractWaveformSegment(inputWaveform, intervals)
    %Extract data within a specified horizontal interval from input waveform data
    %OutputData=EXTRACTWAVERMSEGMENT (inputWaveform, intervals) from input
    %Extract data within the specified horizontal axis interval from waveform data. The input parameter inputWaveform is the raw waveform data,
    %Intervals is a matrix containing horizontal intervals, with each row representing an interval [start, end].
    %The function checks the index range of each interval and extracts the data within the corresponding interval from the input waveform data,
    %And store the data of each interval as separate element arrays.

    % Initialize unit array
    outputData = cell(size(intervals, 1), 1);
    
    % Traverse each interval and extract waveform data
    for i = 1:size(intervals, 1)
        start_idx = intervals(i, 1);
        end_idx = intervals(i, 2);
        
        % Check the index range to ensure it does not exceed the length of the waveform data
        if start_idx >= 1 && end_idx <= length(inputWaveform)
            segment = inputWaveform(start_idx:end_idx);
            outputData{i} = segment;
        else
            error('Invalid interval indices.');
        end
    end
end

function plotWaveforms(outputData, separatePlot, numColumns)
    %PLOTWAVEFORMS draws waveform curves
    %PLOTWAVEFORMS (output Data) plots each row of data in the output data in the same window
    %On different canvases.
    %PLOTWAVERMS (output data, separated lot) If separated lot is true,
    %Then each row of data will be drawn separately on a separate canvas, otherwise it will be drawn on the same canvas.
    %PLOTWAVERMS (output data, separated lot, numColumns) specifies the number of columns on the canvas when drawn separately.
        
    % Check input parameters
    if nargin < 2
        separatePlot = false; % By default, all curves are drawn on the same canvas
    end
    
    if nargin < 3
        numColumns = 1;
    end
    
    figure;
    
    for i = 1:length(outputData)
        if separatePlot
            numRows = ceil(length(outputData) / numColumns);
            subplot(numRows, numColumns, i);
        else
            ax = axes;
        end
        
        plot(outputData{i});
        
        xlabel('X');
        ylabel('Y');
        title(['curve', num2str(i)]);
        grid on;
        
        if separatePlot
            % Draw on separate canvases and set the Y-axis label for each subgraph
            ylabel(['Y', num2str(i)]);
        end
    end
end

function saveCellDataToExcel(cellData, excelFileName)
    % Specify the Excel file name and the name of the worksheet to save
    sheet = 'Sheet1';
    % Loop through output Data1 and save the vectors of each cell to different columns in Excel
    for i = 1:length(cellData)
        % Get the vector of the current cell
        currentVector = cellData{i};
        % Calculate the letters in the current column (A, B, C,...)
        columnLetter = getExcelColumnLabel(i);
        % Calculate the coordinates of the starting cell
        startCell = [columnLetter '1']; % The first cell of the current column
        % Use the xlswrite function to save the vector of the current cell to the current column of the Excel file
        writematrix(currentVector', excelFileName, 'Sheet', sheet, 'Range', startCell);
    end
end

% Convert numbers to column indexes for Excel spreadsheets
function columnStr = getExcelColumnLabel(columnNumber)
    dividend = columnNumber;
    columnStr = '';
    modulo = 0;

    while dividend > 0
        modulo = mod((dividend - 1), 26);
        columnStr = [char(65 + modulo), columnStr];
        dividend = floor((dividend - modulo - 1) / 26);
    end
end






