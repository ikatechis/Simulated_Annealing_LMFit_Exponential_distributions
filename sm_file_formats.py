import numpy as np

def import_traces_file(traces_filepath, number_of_colours=2):
    
    with open(traces_filepath, 'r') as traces_file:
        number_of_frames = np.fromfile(traces_file, dtype=np.int32, count=1).item()
        number_of_traces = np.fromfile(traces_file, dtype=np.int16, count=1).item()
        number_of_molecules = number_of_traces // number_of_colours
        rawData = np.fromfile(traces_file, dtype=np.int16, count=number_of_frames * number_of_traces)
        traces = np.reshape(rawData.ravel(), (number_of_colours, number_of_molecules, number_of_frames), order='F')
    return traces
                                
def import_log_file(log_file_path):
    exposure_time = np.genfromtxt(log_file_path, max_rows=1)[2]
    log_details = open(log_file_path).readlines()
    log_details = ''.join(log_details)
                                
    return exposure_time