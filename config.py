# BirdNET uses eBird checklist frequency data to determine plausible species
# occurrences for a specific location (lat, lon) and one week. An EBIRD_THRESHOLD
# of 0.02 means that a species must occur on at least 2% of all checklists
# for a location to be considered plausible.
EBIRD_SPECIES_CODES = 'metadata/eBird_taxonomy_codes_2018.json'
EBIRD_MDATA = 'metadata/eBird_grid_data_weekly.gz'
USE_EBIRD_CHECKLIST = True
EBIRD_THRESHOLD = 0.02
DEPLOYMENT_LOCATION = (-1, -1)
DEPLOYMENT_WEEK = -1
GRID_STEP_SIZE = 0.25

# We use 3-second spectrograms to identify avian vocalizations.
# You can specify the overlap of consecutive spectrograms and the minimum
# length of a valid signal chunk (in seconds). You can also combine a number
# of extracted spectrograms for each prediction.
SPEC_OVERLAP = 0
SPEC_MINLEN = 1.0
SPECS_PER_PREDICTION = 1

# Adjusting the sigmoid sensitivity of the output layer can increase the
# number of detections (but will most likely also increase the number of
# false positives). You can set a minimum confidence threshold to suppress
# predictions with low score.

# The adjustment of the sigmoid sensitivity of the output layer can lead to an increase 
# of detections (but will most likely also increase the number of false positives). 
# You can set a minimum confidence threshold to suppress low score predictions.
SENSITIVITY = 1.0
MIN_CONFIDENCE = 0.1

# Loading a snapshot automatically sets the corresponding settings. Do not
# change these settings at runtime!
def setModelSettings(s):

    if 'classes' in s:
        global CLASSES
        CLASSES = s['classes']

    if 'spec_type' in s:
        global SPEC_TYPE
        SPEC_TYPE = s['spec_type']

    if 'magnitude_scale' in s:
        global MAGNITUDE_SCALE
        MAGNITUDE_SCALE = s['magnitude_scale']

    if 'sample_rate' in s:
        global SAMPLE_RATE
        SAMPLE_RATE = s['sample_rate']

    if 'win_len' in s:
        global WIN_LEN
        WIN_LEN = s['win_len']

    if 'spec_length' in s:
        global SPEC_LENGTH
        SPEC_LENGTH = s['spec_length']

    if 'spec_fmin' in s:
        global SPEC_FMIN
        SPEC_FMIN = s['spec_fmin']

    if 'spec_fmax' in s:
        global SPEC_FMAX
        SPEC_FMAX = s['spec_fmax']
        
    if 'im_dim' in s:        
        global IM_DIM
        IM_DIM = s['im_dim']

    if 'im_size' in s:
        global IM_SIZE
        IM_SIZE = s['im_size']