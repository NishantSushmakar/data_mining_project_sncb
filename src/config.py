

DATA_PATH = "../data/sncb_data_challenge.csv"
CLEAN_DATA_PATH = "../data/clean_data.pkl"
ENCODER_PATH = "../misc/"

MODEL_OUTPUT = "../models/"

SEQUENCE_COLUMNS = [
        "vehicles_sequence",
        "events_sequence",
        "seconds_to_incident_sequence",
        "train_kph_sequence",
        "dj_ac_state_sequence",
        "dj_dc_state_sequence",
    ]

COL_TO_DROP = ['Unnamed: 0', 'incident_id', 'vehicles_sequence', 'events_sequence',
       'seconds_to_incident_sequence', 'approx_lat', 'approx_lon',
       'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence',
        'window_vehicles_sequence',
        'window_events_sequence',
        'window_seconds_to_incident_sequence',
        'window_train_kph_sequence',
        'window_dj_ac_state_sequence',
        'window_dj_dc_state_sequence','sequence_dict','index_sequence','window_min_idx','window_max_idx']