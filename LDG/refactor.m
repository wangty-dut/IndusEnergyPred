%% Read waveform data for reconstructing waveform shape
fileName = './chonggou_fangbo_from_jianxi_data/output_segments.xlsx';
% Read sub table
boxing1 = readcell(fileName, 'Sheet', 'Boxing 1');
boxing2 = readcell(fileName, 'Sheet', 'Boxing 2');
boxing3 = readcell(fileName, 'Sheet', 'Boxing 3');
file_path = 'chonggou_fangbo_from_jianxi_data/segment_data/duibi/';
method = 'c_pinn';
[full_file_path1, full_file_path2, full_file_path3, full_file_path4, full_file_path5, full_file_path6] = get_filepath(file_path, method);
% Refactor data
day_idx = 7;% Obtain the reconstruction data of the three furnaces' actual predictions for that day
[state_pred1, state_true1] = get_data(full_file_path1, full_file_path2);
[vector_pred1, vector_true1] = get_day_vector(day_idx, state_pred1, state_true1);% Obtain the waveform of a day based on its features
[state_pred2, state_true2] = get_data(full_file_path3, full_file_path4);
[vector_pred2, vector_true2] = get_day_vector(day_idx, state_pred2, state_true2);% Obtain the waveform of a day based on its features
[state_pred3, state_true3] = get_data(full_file_path5, full_file_path6);
[vector_pred3, vector_true3] = get_day_vector(day_idx, state_pred3, state_true3);% Obtain the waveform of a day based on its features
% Waveform replacement
vector_pred1_1 = replace_segments(vector_pred1, boxing1);
vector_true1_1 = replace_segments(vector_true1, boxing1);
vector_pred2_1 = replace_segments(vector_pred2, boxing2);
vector_true2_1 = replace_segments(vector_true2, boxing2);
vector_pred3_1 = replace_segments(vector_pred3, boxing3);
vector_true3_1 = replace_segments(vector_true3, boxing3);
% Obtain the best predicted segment and draw it
plot_three_segments(vector_pred1_1, vector_pred2_1, vector_pred3_1, vector_true1_1, vector_true2_1, vector_true3_1, 1600);
% Calculation accuracy index
[mae_1, mse_1, rmse_1] = get_accuracy_metrics([vector_pred1_1, vector_pred2_1, vector_pred3_1], [vector_true1_1, vector_true2_1, vector_true3_1]);
fprintf('mae: %.4f\n', mae_1);
fprintf('mae: %.4f\n', mse_1);
fprintf('mae: %.4f\n', rmse_1);


% Cut sub segments of the predicted true curves of three furnaces using a sliding window and merge them into two matrices
function [pred_combined1, true_combined1, pred_combined2, true_combined2, pred_combined3, true_combined3] = sliding_window_combine(vector_pred1, vector_pred2, vector_pred3, vector_true1, vector_true2, vector_true3, window_size, step_size)
    % Get the length of the input vectors
    vector_length = length(vector_pred1);

    % Calculate the number of windows
    num_windows = floor((vector_length - window_size) / step_size) + 1;

    % Initialize the matrices to store the combined windows
    pred_combined1 = [];
    true_combined1 = [];
    pred_combined2 = [];
    true_combined2 = [];
    pred_combined3 = [];
    true_combined3 = [];

    % Sliding window process
    for i = 1:num_windows
        start_idx = (i-1) * step_size + 1;
        end_idx = start_idx + window_size - 1;

        % Extract the window for each vector
        pred_window1 = vector_pred1(start_idx:end_idx);
        pred_window2 = vector_pred2(start_idx:end_idx);
        pred_window3 = vector_pred3(start_idx:end_idx);
        true_window1 = vector_true1(start_idx:end_idx);
        true_window2 = vector_true2(start_idx:end_idx);
        true_window3 = vector_true3(start_idx:end_idx);

        % Combine the windows
        pred_combined1 = [pred_combined1;pred_window1];
        true_combined1 = [true_combined1;true_window1];
        pred_combined2 = [pred_combined2;pred_window2];
        true_combined2 = [true_combined2;true_window2];
        pred_combined3 = [pred_combined3;pred_window3];
        true_combined3 = [true_combined3;true_window3];
    end
end


function [full_file_path1, full_file_path2, full_file_path3, full_file_path4, full_file_path5, full_file_path6] = get_filepath(file_path, method)
    str_end = '.xlsx';
    file_name1 = ['pred_data1_seg_', method, str_end];
    file_name2 = 'true_data1_seg.xlsx';
    file_name3 = ['pred_data2_seg_', method, str_end];
    file_name4 = 'true_data2_seg.xlsx';
    file_name5 = ['pred_data3_seg_', method, str_end];
    file_name6 = 'true_data3_seg.xlsx';

    full_file_path1 = fullfile(file_path, file_name1);
    full_file_path2 = fullfile(file_path, file_name2);
    full_file_path3 = fullfile(file_path, file_name3);
    full_file_path4 = fullfile(file_path, file_name4);
    full_file_path5 = fullfile(file_path, file_name5);
    full_file_path6 = fullfile(file_path, file_name6);
end

function plot_three_segments(vector_pred1, vector_pred2, vector_pred3, vector_true1, vector_true2, vector_true3, x_size)

    figure();

    subplot(3, 1, 1);
    plot(vector_pred1, 'Marker', '*', 'Markersize', 2);
    hold on;
    plot(vector_true1, 'Marker', 'o', 'Markersize', 2);
    title('Comparison 1: Predicted vs True Values');
    xlabel('time/min');
    ylabel('Amount of recycling');
    legend('Predicted', 'True');
    xlim([-10 x_size]);
    ylim([-10 500]);
    grid on;
    hold off;


    subplot(3, 1, 2);
    plot(vector_pred2, 'Marker', '*', 'Markersize', 2);
    hold on;
    plot(vector_true2, 'Marker', 'o', 'Markersize', 2);
    title('Comparison 2: Predicted vs True Values');
    xlabel('time/min');
    ylabel('Amount of recycling');
    legend('Predicted', 'True');
    xlim([-10 x_size]);
    ylim([-10 500]);
    grid on;
    hold off;

    subplot(3, 1, 3);
    plot(vector_pred3, 'Marker', '*', 'Markersize', 2);
    hold on;
    plot(vector_true3, 'Marker', 'o', 'Markersize', 2);
    title('Comparison 3: Predicted vs True Values');
    xlabel('time/min');
    ylabel('Amount of recycling');
    legend('Predicted', 'True');
    xlim([-10 x_size]);
    ylim([-10 500]);
    grid on;
    hold off;
end

function dp = replace_segments(dp0, cl)
% REPLACE-SEGMENT replaces the segments in the 'dp' array with the corresponding segments in 'cl'.
    dp = dp0;  % Initialize the result array as the input dp array
    % Get the length of the dp array
    n = length(dp);
    % Traverse the dp array and search for segments with consecutive 1s
    i = 1;
    j = 1;
    while i <= n
        if dp(i) == 1
            % Find the starting index of consecutive segments with 1
            start_index = i;      
            % Continue to search for the ending index of segments with consecutive 1s
            while i <= n && dp(i) == 1
                i = i + 1;
            end
            % Find the ending index of consecutive segments with 1
            end_index = i - 1;
            % Obtain the length of segments that are continuously 1
            segment_length = end_index - start_index + 1;
            % Obtain the data segment b corresponding to the position in cl
            sample = filter_numeric_data(cl(:, j));
            if length(sample)>=segment_length
                last_index = min(n, start_index+length(sample)-1);
                dp(start_index:last_index) = sample(1:(last_index - start_index + 1));
            else
                dp(start_index:start_index+length(sample)-1) = sample;
                dp(start_index+length(sample):end_index) = 0;
            end
            j = j+1;
        else
            % If dp (i) is not 1, proceed to the next index
            i = i + 1;
        end
    end
end

function numeric_data = filter_numeric_data(cell_array)
    numeric_data = [];
    for i = 1:length(cell_array)
        % Check if the current element is of numeric type and not NaN or empty
        if isnumeric(cell_array{i}) && ~isnan(cell_array{i}) && ~isempty(cell_array{i})
            numeric_data = [numeric_data, cell_array{i}];
        end
    end
end


function [state_pred, state_true] = get_data(full_file_path1, full_file_path2)% Retrieve data from the file
    state_pred = xlsread(full_file_path1);
    state_pred(1,:)=[];
    state_pred(:,1)=[];
    state_true = xlsread(full_file_path2);
    state_true(1,:)=[];
    state_true(:,1)=[];
end

% Calculate precision index parameters
function [mae, mse, rmse] = get_accuracy_metrics(vector_pred, vector_true)
    errors = vector_true - vector_pred;
    % MAE calculate
    mae = mean(abs(errors));
    % MSE calculate
    mse = mean(errors.^2);
    % RMSE calculate
    rmse = sqrt(mse);

    fprintf('MAE: %.4f\n', mae);
    fprintf('MSE: %.4f\n', mse);
    fprintf('RMSE: %.4f\n', rmse);
end

% Obtain the waveform of a day based on its features
function [vector_pred, vector_true] = get_day_vector(day_idx, state_pred, state_true)
    % data partitioning
    day_plot=day_idx;
    segmented_data_pred = segmentArray1(state_pred, day_plot);
    segmented_data_true = segmentArray1(state_true, day_plot);

    % Draw data
    vector_pred=plotVectorWithZeros(segmented_data_pred);
    vector_true=plotVectorWithZeros(segmented_data_true);
    vector_pred = set_zeros_ones(vector_pred);
    vector_true = set_zeros_ones(vector_true);
end




function segmented_data = segmentArray1(a, index)
    % Find the segmentation position
    now_idx=0;
    segmented_data=[];
    for i=1:length(a)-1
        if now_idx+1==index
            segmented_data=[segmented_data;a(i,:)];
        end
        if a(i+1, 1)==0 && a(i+1, 2)==0 && a(i+1, 3)==0 && a(i+1, 4)==0
            now_idx = now_idx+1;
        end
    end
end

% Convert feature groups into square waves
function vector=plotVectorWithZeros(intervals)
    % Create a 1600 length all 1 vector
    vector = ones(1, 1600);

    % Traverse the input interval information
    for i = 1:size(intervals, 1)
        center = intervals(i, 1);  % Center position of interval
        width = intervals(i, 2);   % Interval width
        
        % Calculate the starting and ending positions of the interval
        start_index = max(1, round(center - width/2));
        end_index = min(1600, round(center + width/2));

        % Set the elements of the corresponding interval to zero
        vector(start_index:end_index) = 0;
    end
end

function data = set_zeros_ones(data)
    idx1=0;
    idx2=0;
    flag1=0;
    flag2=0;
    for i=1:length(data)
        if data(i)==0 && flag1==0
            idx1=i;
            flag1=1;
        end
        if data(end-i)==0 && flag2==0
            idx2=length(data)-i;
            flag2=1;
        end
        if flag1&&flag2
            break;
        end
    end
    data(1:idx1)=0;
    data(idx2:end)=0;
    if idx2+15>length(data)
        data(idx2:end)=1;
    else
        data(idx2:idx2+15)=1;
    end
end





