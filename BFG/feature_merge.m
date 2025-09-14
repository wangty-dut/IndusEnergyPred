% Integrate rhythm information features and waveform features
% Rhythm information feature: rhythm-wave
% Waveform information features: waveform_feature, interval_feature features

rhythm_wave = readmatrix('rhythm_wave', 'Sheet', 'sheet1', 'Range', 'A:B');
rhythm_inter = readmatrix('rhythm_inter', 'Sheet',  'sheet1', 'Range', 'A:B');
waveform_feature = readmatrix('waveform_feature', 'Sheet', 'sheet1', 'Range', 'A:E');
interval_feature = readmatrix('interval_feature', 'Sheet', 'sheet1', 'Range', 'A:E');
%Note: A value of 1 indicates that the gap is at the front, while a value of 0 indicates that the waveform is at the front!!!
flag = 0;

rhythm_merge = [];
length_wave = length(waveform_feature);
length_inter = length(interval_feature);

for i = 1:(length_wave+length_inter)
    if flag == 0
        if mod(i, 2) == 1
            rhythm_merge = [rhythm_merge;rhythm_wave((i+1)/2,:), waveform_feature((i+1)/2,:)];
        else
            rhythm_merge = [rhythm_merge;rhythm_inter(i/2,:), interval_feature(i/2,:)];
        end
    else
        if mod(i, 2) == 1
            rhythm_merge = [rhythm_merge;rhythm_inter((i+1)/2,:),interval_feature((i+1)/2,:)];
        else
            rhythm_merge = [rhythm_merge;rhythm_wave(i/2,:), waveform_feature(i/2,:)];
        end
    end
end
rhythm_merge(:, 2) = rhythm_merge(:, 2) - rhythm_merge(:, 1);

% Save features
xlswrite('feature', rhythm_merge, 'sheet1');
