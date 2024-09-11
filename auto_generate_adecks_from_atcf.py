# Generates (unofficial) a-deck (.dat) files for active storms from ensemble trackers
# Uses data downloaded (download.py) from NCEP ensemble tracker's model atcf unix files (GEFS, GEPS, FNMOC ensembles)
# This overwites any .dat files completely from scratch (does not do a simple append)
# Unlike official adecks this won't be sorted by model name (only model init time)

# EXPERIMENTAL
# Work in progress (do not use)

last_run_dates = {}

model_data_folders_by_model_name = {
    'GEFS-ATCF': '/home/db/metview/JRPdata/globalmodeldata/gefs-atcf',
    'GEPS-ATCF': '/home/db/metview/JRPdata/globalmodeldata/geps-atcf',
    'FNMOC-ATCF': '/home/db/metview/JRPdata/globalmodeldata/fnmoc-atcf'
}

# where to store the adeck files for the ensembles
adeck_folder = '/home/db/metview/JRPdata/globalmodeldata/adeck-ens-atcf'

# show file paths being processed
print_file_paths = True

import json
import pandas as pd

import traceback
import copy
import time
import os
import re

import pytz

from datetime import datetime, timedelta

total_model_members_by_time_step = {
    'GEFS-ATCF': {
        '00': 32,
        '06': 32,
        '12': 32,
        '18': 32
    },
    'GEPS-ATCF': {
        '00': 22,
        '12': 22
    },
    'FNMOC-ATCF': {
        '00': 22,
        '12': 22
    }
}

# 76 members in this super-ensemble
# Deterministic, control and perturbation members of the ensembles (don't download AEMN)
gefs_atcf_members = ['AVNX', 'AC00', 'AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AP21', 'AP22', 'AP23', 'AP24', 'AP25', 'AP26', 'AP27', 'AP28', 'AP29', 'AP30']
geps_atcf_members = ['CMC', 'CC00', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06', 'CP07', 'CP08', 'CP09', 'CP10', 'CP11', 'CP12', 'CP13', 'CP14', 'CP15', 'CP16', 'CP17', 'CP18', 'CP19', 'CP20']
fnmoc_atcf_members = ['NGX', 'NC00', 'NP01', 'NP02', 'NP03', 'NP04', 'NP05', 'NP06', 'NP07', 'NP08', 'NP09', 'NP10', 'NP11', 'NP12', 'NP13', 'NP14', 'NP15', 'NP16', 'NP17', 'NP18', 'NP19', 'NP20']

# Create a dictionary to map short strings to list names
model_name_to_ensemble_name = {}
for list_name, lst in zip(['GEFS-ATCF', 'GEPS-ATCF', 'FNMOC-ATCF'], [gefs_atcf_members, geps_atcf_members, fnmoc_atcf_members]):
    for model_name in lst:
        model_name_to_ensemble_name[model_name] = list_name

# number of files expected per time-step (or member for tcgen)
expected_num_grib_files_by_model_name = {
    'GEFS-TCGEN': 1,
    'GEPS-TCGEN': 1,
    'FNMOC-TCGEN': 1
}

# in order to select grib files from a folder we need to ignore these file name extensions
ignored_file_extensions = [
    'asc',
    'idx',
    'json',
    'index'
]

model_member_re_str_by_model_name = {
    'GEFS-ATCF': r'^[0-9]+_(?P<member>[a-z0-9]+)\.',
    'GEPS-ATCF': r'[0-9]+_(?P<member>[a-z0-9]+)\.',
    'FNMOC-ATCF': r'[0-9]+_(?P<member>[a-z0-9]+)\.'
}

#########################################################
#########################################################

def datetime_utcnow():
    return datetime.now(pytz.utc).replace(tzinfo=None)

#########################################################
#########################################################

def process_atcf_data_from_trackers():
    # Sort the data from disturbances
    decks_by_atcf_id_sorted = get_all_decks_by_atcf_id_sorted_by_model_init_time()

    if not decks_by_atcf_id_sorted:
        return

    os.makedirs(adeck_folder, exist_ok=True)
    dt_now = datetime_utcnow()
    dt_str = datetime.strftime(dt_now, '%Y-%m-%d %H:%m')
    print(f"{dt_str} UTC : Updating adecks.")
    for atcf_id, deck_str in decks_by_atcf_id_sorted.items():
        try:
            file_path = os.path.join(adeck_folder, f'a{atcf_id.lower()}.dat')
            mod_create_or_update = 'Updated'
            created = False
            if not os.path.exists(file_path):
                mod_create_or_update = 'Created'
                created = True
            with open(file_path, 'w') as f:
                f.write(deck_str)
                dt_now = datetime_utcnow()
                dt_str = datetime.strftime(dt_now, '%Y-%m-%d %H:%m')
                # only print the file when we create a new adeck file (new invest, named storm)
                if created:
                    print(f"{dt_str} UTC : {mod_create_or_update} {file_path}")
        except:
            print(f"Error writing file for {file_path}")
            print(traceback.format_exc())
            # reset this to retry writing all of them
            global last_model_file_paths_by_model_name_and_timestamp
            last_model_file_paths_by_model_name_and_timestamp = None

# returns completed and partial model folders (partial to get EPS-HRES early)
def get_completed_unprocessed_model_folders_by_model_name():
    completed_models_folder_by_model_name = {}
    # EPS only (as we want its HRES member early)
    partial_model_folders_by_model_name = {}
    for model_name, model_base_dir in model_data_folders_by_model_name.items():
        if not os.path.exists(model_base_dir):
            continue
        model_dirs = []
        partial_model_dirs = []
        for root, dirs, files in os.walk(model_base_dir):
            for folder in dirs:
                if os.path.exists(os.path.join(root, folder, '_COMPLETE_')):
                    model_dirs.append(os.path.join(root, folder))
            # walk only the top level
            dirs.clear()
        if model_dirs:
            completed_models_folder_by_model_name[model_name] = sorted(model_dirs)
        if partial_model_dirs:
            partial_model_folders_by_model_name[model_name] = sorted(partial_model_dirs)

    return completed_models_folder_by_model_name, partial_model_folders_by_model_name

# get tcgenesis file paths (.txt for NCEP, bufr .bin files for ECM)
def get_model_file_paths(model_dir, is_complete):
    # if not is_complete (EPS) then only get the HRS bufr bin files
    model_file_paths = []
    if not os.path.exists(model_dir):
        return model_file_paths

    tcgen_file_ext = ['txt']
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            f_ext = file.split('.')[-1]
            if f_ext in tcgen_file_ext:
                if is_complete:
                    model_file_paths.append(os.path.join(root, file))

    return sorted(model_file_paths)

def get_decks_from_atcf_txt_files(ensemble_model_name, model_files_by_stamp):
    decks_by_atcf_id = []

    for model_timestamp, model_file_paths in model_files_by_stamp.items():
        model_member_re = re.compile(model_member_re_str_by_model_name[ensemble_model_name])
        for model_file_path in model_file_paths:
            f = os.path.basename(model_file_path)
            res = re.match(model_member_re, f)
            if res:
                model_member = res['member'].upper()
            else:
                print("Warning: Skipping file! Could not match member for: ", model_file_path)
                continue

            # split txt file into decks by each ATCF ID
            new_decks_by_atcf_id = read_atcf_txt_file_to_decks(model_file_path)

            if new_decks_by_atcf_id is not None:
                decks_by_atcf_id.append(new_decks_by_atcf_id)

    return decks_by_atcf_id

# splits each atcf file into a dict by atcf_id
def read_atcf_txt_file_to_decks(model_file_path):
    atcf_dict = {}
    try:
        with open(model_file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return {}

        for line in lines:
            columns = line.strip().split(', ')
            atcf_id = columns[0].strip() + columns[1].strip() + columns[2].strip()[:4]
            model_init_time = columns[2].strip()
            if atcf_id not in atcf_dict:
                atcf_dict[atcf_id] = (model_init_time, [line.strip()])
            else:
                atcf_dict[atcf_id][1].append(line.strip())

    except:
        print("")
        # reset this to retry writing all of them
        global last_model_file_paths_by_model_name_and_timestamp
        last_model_file_paths_by_model_name_and_timestamp = None
        pass

    return atcf_dict

# combines the atcf dicts into a deck by atcf id
def combine_atcf_dicts_to_deck(atcf_dicts):
    decks_by_atcf_id = {}
    for atcf_dict in atcf_dicts:
        for atcf_id, atcf_time_lines_tuple in atcf_dict.items():
            if atcf_id not in decks_by_atcf_id:
                decks_by_atcf_id[atcf_id] = [atcf_time_lines_tuple]
            else:
                decks_by_atcf_id[atcf_id].append(atcf_time_lines_tuple)

    decks_by_atcf_id_sorted = {}
    for atcf_id, tuples in decks_by_atcf_id.items():
        sorted_tuples = sorted(tuples, key=lambda x: x[0])
        # model_init_times = [t[0] for t in sorted_tuples]
        deck_strings = '\n'.join([line for t in sorted_tuples for line in t[1]])
        decks_by_atcf_id_sorted[atcf_id] = deck_strings

    return decks_by_atcf_id_sorted

# returns all decks by atcf id (sorted by model init time)
def get_all_decks_by_atcf_id_sorted_by_model_init_time():
    # get completed model folders first (not processed yet)
    # ECM HRES should be an exception as it is separate from genesis
    # i.e.  files named *JSXXnnECMF* are HRES and completed earlier than the ensemble members!
    completed_folders_by_model_name, partial_folders_by_model_name = get_completed_unprocessed_model_folders_by_model_name()
    if not completed_folders_by_model_name and not partial_folders_by_model_name:
        return {}

    model_file_paths_by_model_name_and_timestamp = {}
    # as above but excluding partial directories (used to keep track of complete folders)
    complete_file_paths_by_model_name_and_timestamp = {}
    # get all relevant files first
    # we use model name here, but we mean ensemble name
    for is_partial, folders_by_model_name in enumerate([
        completed_folders_by_model_name, partial_folders_by_model_name]):
        for model_name, model_folders in folders_by_model_name.items():
            if model_name in model_file_paths_by_model_name_and_timestamp:
                # handling partial folders (second loop when is_partial is True)
                model_file_paths_by_timestamp = model_file_paths_by_model_name_and_timestamp[model_name]
            else:
                model_file_paths_by_timestamp = {}
            for model_folder in model_folders:
                model_timestamp = os.path.basename(model_folder)
                if is_partial == 1:
                    # EPS case: exclude genesis fields until complete (only take HRES member)
                    model_file_paths = get_model_file_paths(model_folder, False)
                else:
                    model_file_paths = get_model_file_paths(model_folder, True)
                if model_file_paths:
                    # check here whether we should skip (processed same exact number of files already)
                    num_files_processed = 0

                    if num_files_processed != len(model_file_paths):
                        # either process a new directory, or reprocess a directory that has new files (all files reprocessed)
                        # the number of components (tracks) in the candidates table should not decrease so it should be fine
                        model_file_paths_by_timestamp[model_timestamp] = model_file_paths

                        if is_partial == 0:
                            if model_name not in complete_file_paths_by_model_name_and_timestamp:
                                complete_file_paths_by_model_name_and_timestamp[model_name] = {}
                            complete_file_paths_by_model_name_and_timestamp[model_name][model_timestamp] = model_file_paths

            if model_file_paths_by_timestamp:
                model_file_paths_by_model_name_and_timestamp[model_name] = model_file_paths_by_timestamp

    if not model_file_paths_by_model_name_and_timestamp:
        return {}

    global last_model_file_paths_by_model_name_and_timestamp
    if model_file_paths_by_model_name_and_timestamp == last_model_file_paths_by_model_name_and_timestamp:
        # no changes: don't bother updating
        return {}

    last_model_file_paths_by_model_name_and_timestamp = model_file_paths_by_model_name_and_timestamp

    # this creates a deck for each model timestamp
    atcf_dicts = []
    for model_name, model_files_by_timestamp in model_file_paths_by_model_name_and_timestamp.items():
        new_atcf_dicts = get_decks_from_atcf_txt_files(model_name, model_files_by_timestamp)
        if new_atcf_dicts:
            atcf_dicts.extend(new_atcf_dicts)

    decks_by_atcf_id_sorted = combine_atcf_dicts_to_deck(atcf_dicts)

    return decks_by_atcf_id_sorted

# get dict of the completed TC candidate (tracks) by model name
def get_model_timestamp_of_completed_tc_by_model_name():
    completed_tc_dicts = get_completed_tc()
    # create a dict by model_name from the rows
    completed_tc_by_model_name = {}
    if completed_tc_dicts:
        for completed_tc_dict in completed_tc_dicts:
            model_name = completed_tc_dict['model_name']
            model_timestamp = completed_tc_dict['model_timestamp']
            if model_name not in completed_tc_by_model_name:
                completed_tc_by_model_name[model_name] = []

            completed_tc_by_model_name[model_name].append(model_timestamp)

    return completed_tc_by_model_name

#######################################
### CALCULATE TCs FROM DISTURBANCES ###
#######################################

# Only not already computed calculate disturbances from complete model runs

## AUTO UPDATE CODE ##

# polling interval in minutes to calculate (look for disturbances that have completed runs but have not yet computed tc candidates)
polling_interval = 5

last_model_file_paths_by_model_name_and_timestamp = None

if __name__ == "__main__":
    while True:
        process_atcf_data_from_trackers()
        time.sleep(60 * polling_interval)
