% Reconstruct the waveforms of Four furnaces, and based on the gap time and width, the waveforms are given from the actual occurrence data.
warning off;
% Read waveform data
file_path_seg1 = './segmented1.xlsx';
file_path_seg2 = './segmented2.xlsx';
file_path_seg3 = './segmented3.xlsx';
file_path_seg4 = './segmented4.xlsx';

method = 'c_pinn';
str_end = '.xlsx';

file_name1 = ['pred_data1_seg_', method, str_end];
file_name2 = 'true_data1.xlsx';
file_name3 = ['pred_data2_seg_', method, str_end];
file_name4 = 'true_data2.xlsx';
file_name5 = ['pred_data3_seg_', method, str_end];
file_name6 = 'true_data3.xlsx';
file_name7 = ['pred_data4_seg_', method, str_end];
file_name8 = 'true_data4.xlsx';

% Build a complete file path
full_file_path1 = fullfile(file_path, file_name1);
full_file_path2 = fullfile(file_path, file_name2);
full_file_path3 = fullfile(file_path, file_name3);
full_file_path4 = fullfile(file_path, file_name4);
full_file_path5 = fullfile(file_path, file_name5);
full_file_path6 = fullfile(file_path, file_name6);
full_file_path7 = fullfile(file_path, file_name7);
full_file_path8 = fullfile(file_path, file_name8);

% The first output parameter is data, and the second output parameter is text data
day_idx = 8;
[state_pred1, state_true1] = get_data(full_file_path1, full_file_path2);
error_lists1 = get_eve_error(state_pred1, state_true1);
[vector_pred1] = get_day_vector(day_idx, state_pred1);% Obtain the waveform of a day based on its features
[vector_true1] = get_day_vector_true(day_idx, state_true1, file_path_seg1);% Obtain the waveform of a day based on its features
 plotVectorComparison(vector_pred1, vector_true1);% Draw a waveform for a day
 [cumulative_pred1, cumulative_true1] = plot_cumulative_comparison(vector_pred1, vector_true1);
 [mae1, mape1, rmse1] = get_accuracy_metrics(vector_pred1, vector_true1);
% Output result 1
fprintf('MAE1: %.4f\n', mae1);
fprintf('MAPE1: %.4f\n', mape1);
fprintf('RMSE1: %.4f\n', rmse1);
fprintf('error_sum:%.4f\n', abs(cumulative_true1(end)-cumulative_pred1(end)));

[state_pred2, state_true2] = get_data(full_file_path3, full_file_path4);
error_lists2 = get_eve_error(state_pred2, state_true2);
[vector_pred2] = get_day_vector(day_idx, state_pred2);% Obtain the waveform of a day based on its features
[vector_true2] = get_day_vector_true(day_idx, state_true2, file_path_seg2);% Obtain the waveform of a day based on its features
 plotVectorComparison(vector_pred2, vector_true2);% Draw a waveform for a day
 [cumulative_pred2, cumulative_true2] = plot_cumulative_comparison(vector_pred2, vector_true2);
 [mae2, mape2, rmse2] = get_accuracy_metrics(vector_pred2, vector_true2);
% Output result 2
fprintf('MAE2: %.4f\n', mae2);
fprintf('MAPE2: %.4f\n', mape2);
fprintf('RMSE2: %.4f\n', rmse2);
fprintf('error_sum:%.4f\n', abs(cumulative_true2(end)-cumulative_pred2(end)));

[state_pred3, state_true3] = get_data(full_file_path5, full_file_path6);
error_lists3 = get_eve_error(state_pred3, state_true3);
[vector_pred3] = get_day_vector(day_idx, state_pred3);
[vector_true3] = get_day_vector_true(day_idx, state_true3, file_path_seg3);
 plotVectorComparison(vector_pred3, vector_true3);
 [cumulative_pred3, cumulative_true3] = plot_cumulative_comparison(vector_pred3, vector_true3);
 [mae3, mape3, rmse3] = get_accuracy_metrics(vector_pred3, vector_true3);
% Output result 3
fprintf('MAE3: %.4f\n', mae3);
fprintf('MAPE3: %.4f\n', mape3);
fprintf('RMSE3: %.4f\n', rmse3);
fprintf('error_sum:%.4f\n', abs(cumulative_true3(end)-cumulative_pred3(end)));

[state_pred4, state_true4] = get_data(full_file_path7, full_file_path8);
error_lists4 = get_eve_error(state_pred4, state_true4);
[vector_pred4] = get_day_vector(day_idx, state_pred4);
[vector_true4] = get_day_vector_true(day_idx, state_true4, file_path_seg4);
 plotVectorComparison(vector_pred4, vector_true4);
 [cumulative_pred4, cumulative_true4] = plot_cumulative_comparison(vector_pred4, vector_true4);
 [mae4, mape4, rmse4] = get_accuracy_metrics(vector_pred4, vector_true4);
% Output result 4
fprintf('MAE4: %.4f\n', mae4);
fprintf('MAPE4: %.4f\n', mape4);
fprintf('RMSE4: %.4f\n', rmse4);
fprintf('error_sum:%.4f\n', abs(cumulative_true4(end)-cumulative_pred4(end)));


function error_lists = get_eve_error(state_pred, state_true)
    error_lists = {};  % Use a unit array to store each error list
    error_list = [];
    len = length(state_pred);
    
    for i = 1:len
        if state_true(i, 1) ~= 0
            error_list = [error_list; state_true(i, 1) - state_pred(i, 1)];
        else
            if ~isempty(error_list)
                error_lists{end+1} = error_list;  % Add error list to error lists
                error_list = [];
            end
        end
    end
    
    % Check if the final error list is empty
    if ~isempty(error_list)
        error_lists{end+1} = error_list;
    end
end




function [cumulative_pred, cumulative_true] = plot_cumulative_comparison(vector_pred, vector_true)
    % Calculate cumulative value
    cumulative_pred = cumsum(vector_pred);
    cumulative_true = cumsum(vector_true);
   
    figure();
    
    % Comparison of drawing cumulative value curves
    plot(cumulative_pred, 'Marker', '*', 'Markersize', 2);
    hold on;
    plot(cumulative_true, 'Marker', '.', 'Markersize', 2);
    
    title('Cumulative Value Comparison');
    xlabel('time/min');
    ylabel('Cumulative Amount of recycling');
    
    legend('Predicted Cumulative', 'True Cumulative');
    
    grid on;
    hold off;
end

function [state_pred, state_true] = get_data(full_file_path1, full_file_path2)% Retrieve data from files
    state_pred = xlsread(full_file_path1);
    state_pred(1,:)=[];
    state_true = xlsread(full_file_path2);
    state_true(1,:)=[];
end


% Calculate accuracy index parameters, only calculate places where vector_true is not 0
function [mae, mape, rmse] = get_accuracy_metrics(vector_pred, vector_true)
    % Create a mask to filter out elements where vector_true is not zero
    non_zero_mask = vector_true ~= 0;
    % Filter out elements corresponding to 0
    vector_pred = vector_pred(non_zero_mask);
    vector_true = vector_true(non_zero_mask);
    % calculation error
    errors = vector_true - vector_pred;
    % MAE calculate
    mae = mean(abs(errors));
    % MAPE calculate
    mape = mean(abs(errors) ./ abs(vector_true)) * 100; % Multiply by 100 to represent percentage
    % RMSE calculate
    rmse = sqrt(mean(errors.^2));
end

% Obtain the waveform of a day based on its features
function [vector_pred] = get_day_vector(day_idx, state_pred)
    % data partitioning
    day_plot=day_idx;
    segmented_data_pred = segmentArray1(state_pred, day_plot);
    segmented_data_pred(1, :) = [];
    % Draw data
    vector_pred=plotVectorWithsetment(segmented_data_pred);
    vector_pred = replace_ones(vector_pred);
end

function [vector_pred] = get_day_vector_true(day_idx, state_pred, filepath)
    % data partitioning
    day_plot=day_idx;
    segmented_data_pred = segmentArray1(state_pred, day_plot);
    segmented_data_pred(1, :) = [];
    % Draw data
    sheet_name = sprintf('%s%d', 'Segment', day_idx);
    segment_data = readcell(filepath, 'Sheet', sheet_name);
    vector_pred=plotVectorWithsetment_true(segmented_data_pred, segment_data);
end

% Reconstruct the original waveform curve based on features
function [vector_pred, vector_true] = get_vector(state_pred, state_true)
    vector_pred = [];
    vector_true = [];
    for day=1:1:20
        segmented_data_pred = segmentArray1(state_pred, day);
        segmented_data_true = segmentArray1(state_true, day);
        segmented_data_pred(1, :) = [];
        segmented_data_true(1, :) = [];
        vector1=plotVectorWithZeros(segmented_data_pred);
        vector2=plotVectorWithZeros(segmented_data_true);
        vector1 = set_zeros_ones(vector1);
        vector2 = set_zeros_ones(vector2);
        vector_pred = [vector_pred, vector1];
        vector_true = [vector_true, vector2];
    end
end

function segmented_data = segmentArray1(a, index)
    % Find the segmentation position
    now_idx=0;
    segmented_data=[];
    for i=1:length(a)-1
        if now_idx+1==index
            segmented_data=[segmented_data;a(i,:)];
        end
        if a(i, 1)>a(i+1, 1)
            now_idx = now_idx+1;
        end
    end
end

function vector=plotVectorWithsetment(intervals)
    % Create a 1600 length all 1 vector
    vector = ones(1, 1600);

    % Traverse the input interval information
    for i = 1:size(intervals, 1)
        start_time = intervals(i, 1);  % Read features
        dafd = intervals(i, 2);
        if i< size(intervals, 1)
            width = round(intervals(i+1, 1)-intervals(i, 1));  
        else
            width = round(intervals(i, 2))+1;   
        end
        area = intervals(i, 3);
        amplitude = intervals(i, 4);
        alpha = min(max(intervals(i, 5), 1), 5);
        beta = 6-alpha;
        % Reconstruct fragments based on features
        start_index = start_time;
        end_index = start_time+width-1;
        x = linspace(0, 1, width);
        pdf_value = betapdf(x, alpha, beta);
        pdf_area = norm(pdf_value);
        pdf_sum = sum(pdf_value);
        stretch = (-2*amplitude*pdf_sum+sqrt(4*amplitude.^2*pdf_sum.^2-4*pdf_area.^2*(width*amplitude.^2-area.^2)))/(2*pdf_area.^2);
        stretch = real(stretch); % Extract real parts
        stretch(stretch < 0) = 1; % Set values less than 0 to 1
        sub_vector = stretch*pdf_value+amplitude;
        % Replace the elements of the corresponding interval
        vector(start_index:end_index) = sub_vector;
    end
end

function vector=plotVectorWithsetment_true(intervals, segment_datas)
    % Create a 1600 length all 1 vector
    vector = zeros(1, 1600);

    % Traverse the input interval information
    for i = 1:size(intervals, 1)
        start_index = round(intervals(i, 1));  % Starting position of interval
        width = intervals(i, 2);   % Interval width
        % Calculate the starting and ending positions of the interval
        end_index = min(1600, round(start_index + width));
        % Set the elements of the corresponding interval to zero
        segment_data = segment_datas(2:end, i);
        segment_data = remove_missing(segment_data);
        vector(start_index:end_index) = cell2mat(segment_data);
    end
end

function plotVectorComparison(vector1, vector2)
    figure();
    plot(vector1, 'Marker', '*', 'Markersize', 2);
    hold on;
    plot(vector2, 'Marker', '.', 'Markersize', 2);
    title('Vector Comparison');
    xlabel('time/min');
    ylabel('Amount of recycling');
    legend('Pred', 'True');
    grid on; 
    hold off;
end



function cleaned_array = remove_missing(cellArray)
    % Use CellFun to check if each element is missing
    is_not_missing = ~cellfun(@ismissing, cellArray);
    % Keep only non missing values
    cleaned_array = cellArray(is_not_missing);
end

function result = replace_ones(arr)
    % Copy the input array
    result = arr;
    % Get the length of the array
    n = length(arr);
    % Traverse the array
    i = 1;
    while i <= n
        % If the current value is 1
        if arr(i) == 1
            % Record the start position of consecutive 1s
            start_idx = i;        
            % Find the end position of consecutive 1s
            while i <= n && arr(i) == 1
                i = i + 1;
            end
            end_idx = i - 1;         
            % Calculate the length of consecutive 1s
            len = end_idx - start_idx + 1;          
            % Process the segment of consecutive 1s
            if len <= 3
                % Check if it is at the array endpoints
                if start_idx > 1 && end_idx < n
                    % Not at the beginning or end, replace with the average of left and right values
                    for j = start_idx:end_idx
                        left_val = arr(j - 1);
                        right_val = arr(j + 1);
                        result(j) = (left_val + right_val) / 2;
                    end
                else
                    % At the array endpoints, set to zero
                    result(start_idx:end_idx) = 0;
                end
            else
                % Length greater than 3, set to zero
                result(start_idx:end_idx) = 0;
            end
        else
            i = i + 1;
        end
    end
end













