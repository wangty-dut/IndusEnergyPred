% This code extracts the curve waveform features of the segmented intermittent data and reconstructs the curves corresponding to the features
% Feature extraction and reconstruction using waveform matching: adjusting curves through waveform parameters, waveform area, longitudinal stretching, and longitudinal translation
% Define refactoring data and features
chonggou_sign = {};
alphas_feature = [];
betas_feature = [];
translates_feature = [];
areas_feature = [];
stretch_feature = [];
% Read raw data
% Define the path and file name for Excel files
% file = 'outputData_wave.xlsx';
file = 'outputData_inter.xlsx';
curve_bests = {};
% Pay attention to adjusting the range of numbers here, adjust according to the actual data range of the file!!!
sum_number = getExcelColumnCount(file, 'sheet1');
for number_ = 1:sum_number
    % Using the xlSread function to read specified column data from an Excel file
    char_i = getExcelColumnLabel(number_);
    time_series = readmatrix(file, 'Sheet', 'Sheet1', 'Range', sprintf('%s:%s',char_i,char_i)); % Read all row data from column B
    nanIndices = isnan(time_series);
    time_series(nanIndices, :) = [];
    s = size(time_series);
    % Obtain the width of the original curve
    width = s(1);
    % Obtain the original curve area
    area = norm(time_series);
    sum_ = sum(time_series);
    % Construct parameter curve
    % Specify parameter combinations for multiple Beta distributions
    alpha = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
    beta = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1];
    % Ensure that the width of the parameter curve is consistent with the actual curve
    x = linspace(0, 1, width);
    pdf_values = {};
    % Store the 2-norm and sum of curves separately
    pdf_areas = [];
    pdf_sum = [];
    for i = 1:length(alpha)
        % Calculate the probability density function value of Beta distribution
        pdf_value = betapdf(x, alpha(i), beta(i));
        % Save the generated parameter curve
        pdf_values{end+1} = pdf_value;
        pdf_sum = [pdf_sum, sum(pdf_value)];
        pdf_areas = [pdf_areas, norm(pdf_value)];
    end

    % Stretch and translate the parameter curve longitudinally to match the original curve
    % Define longitudinal stretching and longitudinal translation parameters
    stretchs = 1:30;
    pdf_sum_a = stretchs'*pdf_sum;
    pdf_areas_a = stretchs'*pdf_areas;
    % The translation parameter is determined based on the stretching parameter, curve type, curve norm, etc., and is obtained by solving a quadratic equation
    translates = (-2*pdf_sum_a + sqrt(4*pdf_sum_a.^2 - 4*width*(pdf_areas_a.^2 - area^2)))/(2*width);

    % Curve type, number of operations
    types = length(alpha);
    number = length(stretchs);
    degree_matchs = zeros(types, number);
    for i = 1:types
        for j = 1:number
            % Stretch and translate the parameter curve
            curve = pdf_values{i};
            curve = curve*stretchs(j) + translates(j, i);
            degree = dot(time_series, curve);
            degree_matchs(i, j) = degree;
        end
    end
    degree_matchs = degree_matchs/10000;
    % Find the curve type and operation with the highest matching degree
    [type, stretch] = find(degree_matchs == max(max(degree_matchs))); 

    % Save refactoring data
    curve_best = pdf_values{type};
    curve_best = curve_best*stretchs(stretch) + translates(stretch);
    fprintf('stretch:%f, translate:%f\n', stretchs(stretch), translates(stretch, type));
    curve_bests{end + 1} = curve_best;
    translates_feature = [translates_feature;translates(stretch, type)];
    alphas_feature = [alphas_feature;alpha(type)];
    betas_feature = [betas_feature;beta(type)];
    areas_feature = [areas_feature;area];
    stretch_feature = [stretch_feature;stretch];
end
% Retain refactored data
saveCellDataToExcel(curve_bests, 'output_inter_pipei.xlsx');
% Preserving waveform features: curve parameters, amplitude, area (2-norm)
filename = 'interval_feature.xlsx';
waveforms_feature = [areas_feature, translates_feature, alphas_feature, betas_feature, stretch_feature];
writematrix(waveforms_feature, filename);
%Function
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

% Retrieve the number of columns in an Excel spreadsheet
function columnCount = getExcelColumnCount(excelFileName, sheetName)
    [~, ~, raw] = xlsread(excelFileName, sheetName);
    columnCount = size(raw, 2);
end

