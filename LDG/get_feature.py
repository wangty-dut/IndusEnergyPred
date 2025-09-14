import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

def get_rhythm(data):
    rhythm = []
    yuzhi = 40
    for i in range(len(data)):
        if data[i]>yuzhi:
            rhythm.append(1)
        else:
            rhythm.append(0)
    return rhythm

def fill_short_zero_intervals(arr):
    # Get the length of the array
    n = len(arr)

    # Mark whether it is within a 0 interval
    in_zero_region = False
    start_idx = -1

    # 遍历数组
    for i in range(n):
        if arr[i] == 0:
            if not in_zero_region:
                # Start a new 0 interval
                start_idx = i
                in_zero_region = True
        else:
            if in_zero_region:
                # End the 0 interval and check if it needs to be filled
                if i - start_idx < 15:
                    # Fill in the 0 interval
                    arr[start_idx:i] = 1
                in_zero_region = False

    # Process the last 0 interval
    if in_zero_region and (n - start_idx < 15):
        arr[start_idx:n] = 1

    return arr


def find_zero_intervals(arr):
    # Get the length of the array
    n = len(arr)
    # List of stored results
    results = []
    # Mark whether it is within a 0 interval
    in_zero_region = False
    start_idx = -1
    # Traversal array
    for i in range(n):
        if arr[i] == 0:
            if not in_zero_region:
                # Start a new 0 interval
                start_idx = i
                in_zero_region = True
        else:
            if in_zero_region:
                # End the 0 interval
                end_idx = i
                length = end_idx - start_idx
                midpoint = start_idx + length // 2
                results.append((midpoint, length))
                in_zero_region = False

    # Process the last 0 interval
    if in_zero_region:
        end_idx = n
        length = end_idx - start_idx
        midpoint = start_idx + length // 2
        results.append((midpoint, length))

    return results

def plot_compare(data1, data2):
    x = np.array(range(len(data1)))
    plt.figure()
    plt.plot(x, data1, label='true_data')
    plt.plot(x, data2, label='pred_data')
    plt.legend()
    plt.show()

def action_diff(feature):
    times = feature[:, 0]
    widths = feature[:, 1]
    liangang_times = times + widths/2 + 15
    return liangang_times

def judgement_chongdie(time1, time2, time3):
    length1 = len(time1)
    length2 = len(time2)
    length3 = len(time3)
    length = min(length3, length2, length1)
    buchongdie_num = 0
    for i in range(length):
        if(abs(time1[i]-time2[i])>=15 and abs(time1[i]-time3[i])>=15 and abs(time3[i]-time2[i])>=15):
            buchongdie_num+=1

    return buchongdie_num/length, buchongdie_num, length

def merge_and_sort_arrays(array1, array2, array3):
    # Merge three arrays
    merged_array = np.concatenate((array1, array2, array3))
    # Sort the merged array
    sorted_array = np.sort(merged_array)
    return sorted_array

def judgement_chongdie1(array, yuzhi=10):
    length = len(array)
    num = 0
    for i in range(length-1):
        if(abs(array[i]-array[i+1])>=yuzhi):
            num+=1

    return num/(length-1), length-1, num


def extract_and_save_segments(lu1_data, rhythm_1, lu2_data, rhythm_2, lu3_data, rhythm_3, file_name):
    '''
    Extract time series fragments and save them to multiple sub tables in an Excel file
    '''


    def extract_segments(d1, d2):
        segments = []
        start = None

        for i in range(len(d2)):
            if d2[i] == 1 and start is None:
                start = i
            elif d2[i] == 0 and start is not None:
                segments.append(d1[start:i])
                start = None

        if start is not None:
            segments.append(d1[start:])

        return segments

    def segments_to_dataframe(segments):
        max_length = max(len(segment) for segment in segments)
        df_data = {f'Segment {i+1}': segment for i, segment in enumerate(segments)}
        return pd.DataFrame(dict([(key, pd.Series(value)) for key, value in df_data.items()]))

    # Extract data fragments
    boxing1 = extract_segments(lu1_data, rhythm_1)
    boxing2 = extract_segments(lu2_data, rhythm_2)
    boxing3 = extract_segments(lu3_data, rhythm_3)

    # Convert to DataFrame
    df_boxing1 = segments_to_dataframe(boxing1)
    df_boxing2 = segments_to_dataframe(boxing2)
    df_boxing3 = segments_to_dataframe(boxing3)

    with pd.ExcelWriter(file_name) as writer:
        df_boxing1.to_excel(writer, sheet_name='Boxing 1', index=False)
        df_boxing2.to_excel(writer, sheet_name='Boxing 2', index=False)
        df_boxing3.to_excel(writer, sheet_name='Boxing 3', index=False)

    print(f"Data has been written to the {file_name} file")

if __name__ == "__main__":
    filepath = "./data/4.6~5.4.xlsx"
    # filepath = "example.xlsx"
    data = pd.read_excel(filepath).to_numpy()
    lu1_data = data[:, 0]
    lu2_data = data[:, 1]
    lu3_data = data[:, 2]
    rhythm1 = get_rhythm(lu1_data)
    rhythm2 = get_rhythm(lu2_data)
    rhythm3 = get_rhythm(lu3_data)

    rhythm_1 = fill_short_zero_intervals(np.array(rhythm1)).reshape(-1, 1)
    rhythm_2 = fill_short_zero_intervals(np.array(rhythm2)).reshape(-1, 1)
    rhythm_3 = fill_short_zero_intervals(np.array(rhythm3)).reshape(-1, 1)

    # Extract fragments: real occurrence waveform
    extract_and_save_segments(lu1_data, rhythm_1, lu2_data, rhythm_2, lu3_data, rhythm_3, 'output_segments.xlsx')

    # plot_compare(data[:4000, 0], 100*rhythm_1[:4000])
    feature1 = find_zero_intervals(rhythm_1)
    feature2 = find_zero_intervals(rhythm_2)
    feature3 = find_zero_intervals(rhythm_3)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    feature3 = np.array(feature3)

    feature1[:, 0] = feature1[:, 0]%1440
    feature2[:, 0] = feature2[:, 0] % 1440
    feature3[:, 0] = feature3[:, 0] % 1440

    df = pd.DataFrame(feature1)
    df.to_excel("./data/three_ldg_data1.xlsx", index=False)

    df = pd.DataFrame(feature2)
    df.to_excel("./data/three_ldg_data2.xlsx", index=False)

    df = pd.DataFrame(feature3)
    df.to_excel("./data/three_ldg_data3.xlsx", index=False)



