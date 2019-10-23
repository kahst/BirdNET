import sys
sys.path.append("..")

import pickle
import json, gzip
import datetime

import numpy as np

import config as cfg
from utils import log

##################### GLOBAL VARS #######################
GRID = []
CODES = []
STEP = 0.25

###################### LOAD DATA ########################
def load():

    global GRID
    global CODES
    global STEP

    if len(GRID) == 0:

        # Status
        log.p('LOADING eBIRD GRID DATA...', new_line=False)

        # Load pickled or zipped grid data
        if cfg.EBIRD_MDATA.rsplit('.', 1)[-1] == 'gz':
            with gzip.open(cfg.EBIRD_MDATA, 'rt') as pfile:
                GRID = json.load(pfile)
        else:
            with open(cfg.EBIRD_MDATA, 'rb') as pfile:
                GRID = pickle.load(pfile)

        # Load species codes
        with open(cfg.EBIRD_SPECIES_CODES, 'r') as jfile:
            CODES = json.load(jfile)

        STEP = cfg.GRID_STEP_SIZE

        log.p(('DONE!', len(GRID), 'GRID CELLS'))

#################### PROBABILITIES ######################
def getCellData(lat, lon):

    # Find nearest cell
    for cell in GRID:
        if lat > cell['lat'] - STEP and lat < cell['lat'] + STEP and lon > cell['lon'] - STEP and lon < cell['lon'] + STEP:
            return cell

    # No cell
    return None

def getWeek():

    w = datetime.datetime.now().isocalendar()[1]

    return min(48, max(1, int(48.0 * w / 52.0)))

def getWeekFromDate(y, m, d):

    w = datetime.date(int(y), int(m), int(d)).isocalendar()[1]

    return min(48, max(1, int(48.0 * w / 52.0)))
        
def getSpeciesProbabilities(lat=-1, lon=-1, week=-1):

    # Dummy array
    p = np.zeros((len(cfg.CLASSES)), dtype='float32')

    # No coordinates?
    if lat == -1 or lon == -1:
        return p + 1.0
    else:

        # Get checklist data for nearest cell
        cdata = getCellData(lat, lon)

        # No cell data?
        if cdata == None:
            return p + 1.0
        else:

            # Get probabilities from checklist frequencies
            for entry in cdata['data']:
                for species in entry:

                    try:
                        # Get class index from species code
                        for i in range(len(cfg.CLASSES)):
                            if cfg.CLASSES[i].split('_')[0] == CODES[species].split('_')[0]:                                

                                # Do we want a specific week?
                                if week >= 1 and week <= 48:
                                    p[i] = entry[species][week - 1] / 100.0

                                # If not, simply return the max frequency
                                else:
                                    p[i] = max(entry[species]) / 100.0
                                    
                                break
                        
                    except:
                        pass                
        

        return p

def getSpeciesLists(lat=-1, lon=-1, week=-1, threshold=0.02):

    # Get species probabilities from for date and location
    p = getSpeciesProbabilities(lat, lon, week)

    # Parse probabilities and create white list and black list
    white_list, black_list = [], [] 
    for i in range(p.shape[0]):

        if p[i] >= threshold:
            white_list.append(cfg.CLASSES[i])
        else:
            black_list.append(cfg.CLASSES[i])

    return white_list, black_list