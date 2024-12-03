

DATA_PATH = "../data/sncb_data_challenge.csv"


MAIN_DATA_PATH = "../data/main_dataset_overlap_1800.pkl"
TEST_DATA_PATH = "../data/main_dataset_overlap_1800_test.pkl"

R_EVENT_DICT_PATH = "../misc/relevance_dict.pkl"


TRAIN_DATA_CSV_PATH = "../data/training_data.csv"
TEST_DATA_CSV_PATH = "../data/test_data.csv"

ANOMALY_MODEL_PATH = "../misc/anomaly_model.pkl"

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

COL_TO_DROP = ['Unnamed: 0', 'vehicles_sequence', 'events_sequence',
       'seconds_to_incident_sequence', 'approx_lat', 'approx_lon',
       'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence',
        'window_vehicles_sequence',
        'window_events_sequence',
        'window_seconds_to_incident_sequence',
        'window_train_kph_sequence',
        'window_dj_ac_state_sequence',
        'window_dj_dc_state_sequence','sequence_dict','index_sequence','window_min_idx','window_max_idx']