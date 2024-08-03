# requires curl, and grib_copy for navgem (ECC grib_tools)
import traceback

import re
import os
import subprocess
import time
import requests

from datetime import datetime, timedelta
import sqlite3
import json
import shutil
# generalize to ecmwf (json), navgem(asc), and cmc
# todo need a space margin (1 gb at least, as 500 mb for one run)
# todo somehow need to be able to optionally download missing runs

import threading
import sys
import signal

# print based on level (lower is more important):
#  0 is None (for no messages printed)
#  1 is important (i.e. disk space, database error)
#  2 is notification level: messages for when download runs complete
#  3 is potentially not important (could not download files (as in an incomplete model run))
#  4 is for verbose debugging (print curl/grib_copy statements)
print_level_importance = 2

# how often to check (seconds)
backoff_time = 600
# wait between downloads (seconds)
sleep_time = 1
# number of last n runs to get (might get one more than this)
num_latest_runs = 4

# only download runs not already calculated
disturbances_db_file_path = '/home/db/Documents/JRPdata/cyclone-genesis/disturbances.db'

# GFS and NAV are in the same thread as they use the same server
model_names_to_download_thread_groupings = [['EPS-TCGEN'], ['GEFS-TCGEN', 'GEPS-TCGEN', 'FNMOC-TCGEN', 'GFS', 'NAV'], ['CMC'], ['ECM']]

# use this to download individual grib files for NAVGEM so we can split them with grib_copy
tmp_download_dir = '/tmp'

# min space default is 100 MB (to make sure to not cause other programs to crash)
# tmp refers to tmp_download_dir, the rest refer to model_data_folders_by_model_name
min_space_bytes = {
    'tmp': '100000000',
    'EPS-TCGEN': '50000000',
    'GEFS-TCGEN': '50000000',
    'GEPS-TCGEN': '50000000',
    'FNMOC-TCGEN': '50000000',
    'GFS': '100000000',
    'ECM': '100000000',
    'CMC': '100000000',
    'NAV': '100000000',
}

model_data_folders_by_model_name = {
    'EPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/eps-tcgen',
    'GEFS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen',
    'GEPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/geps-tcgen',
    'FNMOC-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/fnmoc-tcgen',
    'GFS': '/home/db/metview/JRPdata/globalmodeldata/gfs',
    'ECM': '/home/db/metview/JRPdata/globalmodeldata/ecm',
    'CMC': '/home/db/metview/JRPdata/globalmodeldata/cmc',
    'NAV': '/home/db/metview/JRPdata/globalmodeldata/nav'
}

url_folder_by_model = {
    'EPS-TCGEN': 'ftp://wmo:essential@diss.ecmwf.int',
    'GEFS-TCGEN': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/ens_tracker/prod',
    'GEPS-TCGEN': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/ens_tracker/prod',
    'FNMOC-TCGEN': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/ens_tracker/prod',
    'GFS': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod',
    'ECM': 'https://data.ecmwf.int/forecasts',
    'CMC': 'http://hpfx.collab.science.gc.ca',
    'NAV': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/fnmoc/prod'
}

# the time steps in the file name are not all ### (ECM just uses a single digit)
model_time_step_str_format = {
    'GFS': '03',
    'ECM': '00',
    'CMC': '03',
    'NAV': '03'
}

# not including GFS analysis files (.anl)
#   this is the number of members for TCGEN ensembles (excluding EPS)
total_model_time_steps = {
    'EPS-TCGEN': {
        '00': 9999,
        '12': 9999
    },
    'GEFS-TCGEN': {
        '00': 32,
        '06': 32,
        '12': 32,
        '18': 32
    },
    'GEPS-TCGEN': {
        '00': 22,
        '12': 22
    },
    'FNMOC-TCGEN': {
        '00': 22,
        '12': 22
    },
    'GFS': {
        '00': 65,
        '06': 65,
        '12': 65,
        '18': 65
    },
    'ECM': {
        '00': 41,
        '12': 41
    },
    'CMC': {
        '00': 41,
        '12': 41
    },
    'NAV': {
        '00': 31,
        '06': 31,
        '12': 31,
        '18': 31
    }
}
# only interested in 6 hour time steps
#   for tcgen ensembles these are the member names (excluding EPS)
all_time_steps_by_model = {
    'EPS-TCGEN': {
        '00': ['A_JSXX'],
        '12': ['A_JSXX']
    },
    'GEFS-TCGEN': {
        '00': ['gfso', 'ac00', 'ap01', 'ap02', 'ap03', 'ap04', 'ap05', 'ap06', 'ap07', 'ap08', 'ap09', 'ap10', 'ap11', 'ap12', 'ap13', 'ap14', 'ap15', 'ap16', 'ap17', 'ap18', 'ap19', 'ap20', 'ap21', 'ap22', 'ap23', 'ap24', 'ap25', 'ap26', 'ap27', 'ap28', 'ap29', 'ap30'],
        '06': ['gfso', 'ac00', 'ap01', 'ap02', 'ap03', 'ap04', 'ap05', 'ap06', 'ap07', 'ap08', 'ap09', 'ap10', 'ap11', 'ap12', 'ap13', 'ap14', 'ap15', 'ap16', 'ap17', 'ap18', 'ap19', 'ap20', 'ap21', 'ap22', 'ap23', 'ap24', 'ap25', 'ap26', 'ap27', 'ap28', 'ap29', 'ap30'],
        '12': ['gfso', 'ac00', 'ap01', 'ap02', 'ap03', 'ap04', 'ap05', 'ap06', 'ap07', 'ap08', 'ap09', 'ap10', 'ap11', 'ap12', 'ap13', 'ap14', 'ap15', 'ap16', 'ap17', 'ap18', 'ap19', 'ap20', 'ap21', 'ap22', 'ap23', 'ap24', 'ap25', 'ap26', 'ap27', 'ap28', 'ap29', 'ap30'],
        '18': ['gfso', 'ac00', 'ap01', 'ap02', 'ap03', 'ap04', 'ap05', 'ap06', 'ap07', 'ap08', 'ap09', 'ap10', 'ap11', 'ap12', 'ap13', 'ap14', 'ap15', 'ap16', 'ap17', 'ap18', 'ap19', 'ap20', 'ap21', 'ap22', 'ap23', 'ap24', 'ap25', 'ap26', 'ap27', 'ap28', 'ap29', 'ap30']
    },
    'GEPS-TCGEN': {
        '00': ['cmc', 'cc00', 'cp01', 'cp02', 'cp03', 'cp04', 'cp05', 'cp06', 'cp07', 'cp08', 'cp09', 'cp10', 'cp11', 'cp12', 'cp13', 'cp14', 'cp15', 'cp16', 'cp17', 'cp18', 'cp19', 'cp20', 'cp21', 'cp22', 'cp23', 'cp24', 'cp25', 'cp26', 'cp27', 'cp28', 'cp29', 'cp30'],
        '12': ['cmc', 'cc00', 'cp01', 'cp02', 'cp03', 'cp04', 'cp05', 'cp06', 'cp07', 'cp08', 'cp09', 'cp10', 'cp11', 'cp12', 'cp13', 'cp14', 'cp15', 'cp16', 'cp17', 'cp18', 'cp19', 'cp20', 'cp21', 'cp22', 'cp23', 'cp24', 'cp25', 'cp26', 'cp27', 'cp28', 'cp29', 'cp30']
    },
    'FNMOC-TCGEN': {
        '00': ['ngx', 'nc00', 'np01', 'np02', 'np03', 'np04', 'np05', 'np06', 'np07', 'np08', 'np09', 'np10', 'np11', 'np12', 'np13', 'np14', 'np15', 'np16', 'np17', 'np18', 'np19', 'np20', 'np21', 'np22', 'np23', 'np24', 'np25', 'np26', 'np27', 'np28', 'np29', 'np30'],
        '12': ['ngx', 'nc00', 'np01', 'np02', 'np03', 'np04', 'np05', 'np06', 'np07', 'np08', 'np09', 'np10', 'np11', 'np12', 'np13', 'np14', 'np15', 'np16', 'np17', 'np18', 'np19', 'np20', 'np21', 'np22', 'np23', 'np24', 'np25', 'np26', 'np27', 'np28', 'np29', 'np30']
    },
    'GFS': {
        '00': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384],
        '06': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384],
        '12': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384],
        '18': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384]
        },
    'ECM': {
        '00': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240],
        '12': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240]
    },
    'CMC': {
        '00': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240],
        '12': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240]
    },
    'NAV': {
        '00': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180],
        '06': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180],
        '12': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180],
        '18': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180]
    }
}

# number of files expected per time-step (or member for tcgen)
expected_num_grib_files_by_model_name = {
    'EPS-TCGEN': 9999,
    'GEFS-TCGEN': 1,
    'GEPS-TCGEN': 1,
    'FNMOC-TCGEN': 1,
    'GFS': 9,
    'CMC': 9,
    'ECM': 9,
    'NAV': 9,
    'UKM': 9
}

# in order to select grib files from a folder we need to ignore these file name extensions
ignored_file_extensions = [
    'asc',
    'idx',
    'json',
    'index',
    "gene"
]

# hours between model runs (not the time steps)
model_interval_hours = {
    'EPS-TCGEN': 12,
    'GEFS-TCGEN': 6,
    'GEPS-TCGEN': 12,
    'FNMOC-TCGEN': 12,
    'GFS': 6,
    'ECM': 12,
    'CMC': 12,
    'NAV': 6
}

# this is used to check for already downloaded files
#   for tcgen ensembles these are expected members (TC candidates for EPS)
time_step_re_str_by_model_name = {
    'EPS-TCGEN': r'.*?A_JSXX(?P<time_step>\d+EC..)',
    'GEFS-TCGEN': r'.*?storms\.(?P<time_step>[a-z0-9]+)\.',
    'GEPS-TCGEN': r'.*?storms\.(?P<time_step>[a-z0-9]+)\.',
    'FNMOC-TCGEN': r'.*?storms\.(?P<time_step>[a-z0-9]+)\.',
    'GFS': r'.*?\.f(?P<time_step>\d\d\d)_',
    'ECM': r'.*?-(?P<time_step>\d+)h-oper-fc',
    'CMC': r'.*?\d+_P(?P<time_step>\d+)',
    'NAV': r'.*?navgem_\d{10}f(?P<time_step>\d\d\d)'
}

# Specifies what parameters to gather (and also used in how to name it for split gribs)
# GFS is what is used in the .idx file
# ECM is what is in the .index file (list of jsons)
# CMC is the portion of the file name
# NAV are the keys used for grib_copy (download the entire grib and split it)
params_by_model_name = {
    'GFS': [
        ':PRMSL:mean sea level:',
        ':HGT:250 mb:',
        ':HGT:850 mb:',
        ':UGRD:850 mb:',
        ':VGRD:850 mb:',
        ':UGRD:925 mb:',
        ':VGRD:925 mb:',
        ':UGRD:10 m above ground:',
        ':VGRD:10 m above ground:'
        ],
    'ECM': [
        {"levtype": 'sfc', "param": 'msl'},
        {"levtype": 'sfc', "param": '10u'},
        {"levtype": 'sfc', "param": '10v'},
        {"levtype": 'pl', "levlist": '250', "param": 'gh'},
        {"levtype": 'pl', "levlist": '850', "param": 'gh'},
        {"levtype": 'pl', "levlist": '850', "param": 'u'},
        {"levtype": 'pl', "levlist": '850', "param": 'v'},
        {"levtype": 'pl', "levlist": '925', "param": 'u'},
        {"levtype": 'pl', "levlist": '925', "param": 'v'},
    ],
    'CMC': [
        'PRMSL_MSL_0',
        'HGT_ISBL_250',
        'HGT_ISBL_850',
        'UGRD_ISBL_850',
        'UGRD_ISBL_925',
        'VGRD_ISBL_850',
        'VGRD_ISBL_925',
        'UGRD_TGL_10',
        'VGRD_TGL_10'
    ],
    'NAV': [
        {"levtype": 'sfc', "level": 0, "shortName": 'prmsl'},
        {"levtype": 'sfc', "level": 10, "shortName": '10u'},
        {"levtype": 'sfc', "level": 10, "shortName": '10v'},
        {"levtype": 'pl', "level": 250, "shortName": 'gh'},
        {"levtype": 'pl', "level": 850, "shortName": 'gh'},
        {"levtype": 'pl', "level": 850, "shortName": 'u'},
        {"levtype": 'pl', "level": 850, "shortName": 'v'},
        {"levtype": 'pl', "level": 925, "shortName": 'u'},
        {"levtype": 'pl', "level": 925, "shortName": 'v'}
    ]
}

# handle thread errors
exit_event = threading.Event()

# print lines at or below print_level_importance
def print_level(level, warning_str):
    if level <= print_level_importance:
        print(warning_str)

def have_enough_disk_space():
    paths_to_check = {}
    if tmp_download_dir:
        paths_to_check['tmp'] = tmp_download_dir

    for model_name, path in model_data_folders_by_model_name.items():
        paths_to_check[model_name] = path

    enough_space_for_all_paths = True
    for name, path in paths_to_check.items():
        try:
            os.makedirs(path, exist_ok = True)
            dustats = shutil.disk_usage(path)
            if dustats:
                free_bytes = dustats.free
                if free_bytes < int(min_space_bytes[name]):
                    enough_space_for_all_paths = False
                    break

        except Exception as e:
            print_level(1, e)
            print_level(1, "Warning could not get disk space for {path}")

    return enough_space_for_all_paths

def get_calculated_and_missing_time_steps(model_name, model_timestamp):
    model_hour = f'{model_timestamp.hour:02}'
    # remove from this the calculated timesteps
    #   this is the time steps missing, or the members for tcgen ensembles (just the file name prefix for EPS)
    model_time_steps_missing = set(all_time_steps_by_model[model_name][model_hour])

    if "TCGEN" not in model_name:
        disturbance_candidates = get_disturbances_from_db(model_name, model_timestamp)
        time_steps_calculated = set()
        if disturbance_candidates:
            for time_step_str, candidates in disturbance_candidates.items():
                time_step_int = int(time_step_str)
                time_steps_calculated.add(time_step_int)
                # discard will not throw an error even if it is not there
                model_time_steps_missing.discard(time_step_int)
        return sorted(list(time_steps_calculated)), sorted(list(model_time_steps_missing))
    else:
        model_members_missing = model_time_steps_missing
        # always download recent files for tcgen (don't check disturbances db for this)
        return sorted(list(model_members_missing)), sorted(list(model_members_missing))

def get_disturbances_from_db(model_name, model_timestamp):
    conn = None
    retrieved_data = {}
    try:
        # store disturbance candidates in database
        conn = sqlite3.connect(disturbances_db_file_path)
        cursor = conn.cursor()

        ds = model_timestamp.isoformat()

        cursor.execute('SELECT data FROM disturbances WHERE model_name = ? AND date = ?', (model_name, ds))
        result = cursor.fetchone()
        if result:
            retrieved_data = json.loads(result[0])

    except sqlite3.Error as e:
        print_level(1, f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

    return retrieved_data

# for interval_hours = 6, find 6 hour model timestamp prior to 00,06,12,18Z
def get_latest_possible_model_time(model_name):
    interval_hours = model_interval_hours[model_name]
    utc_now = datetime.utcnow()
    model_hour = utc_now.hour
    latest_hour = (model_hour // interval_hours) * interval_hours
    latest_time = datetime.replace(utc_now, hour = latest_hour, minute = 0, second = 0, microsecond = 0)
    return latest_time

# for interval_hours = 6, find 6 hour model timestamp prior to 00,06,12,18Z
def get_latest_n_model_times(model_name, n=num_latest_runs):
    model_times = []
    interval_hours = model_interval_hours[model_name]
    utc_now = datetime.utcnow()
    model_hour = utc_now.hour
    latest_hour = (model_hour // interval_hours) * interval_hours
    latest_time = datetime.replace(utc_now, hour = latest_hour, minute = 0, second = 0, microsecond = 0)

    model_times.append(latest_time)
    for n in range(n):
        latest_time = get_prior_model_time(model_name, latest_time)
        model_times.append(latest_time)
    return model_times

def get_prior_model_time(model_name, dt):
    interval_hours = model_interval_hours[model_name]
    prior_time = dt - timedelta(hours = interval_hours)
    return prior_time

def load_index_file_ecm(idx_file_path):
    with open(idx_file_path, 'r') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]

def find_matching_lines_gfs(idx_file_path, params):
    matching_lines = []
    all_lines = []
    with open(idx_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if any(param in line for param in params):
                matching_lines.append((i, line))
            if ':' in line:
                all_lines.append((i, line))
    return all_lines, matching_lines

def generate_curl_grib_copy_commands_nav(timestamp_prefix, url_folder, url_base_file_name, output_dir, params):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)

    # have no index file so use requests.head first
    # this allows us to check if the directory exists before trying to get all the split grib files
    try:
        res = requests.head(url_folder)
        if not res:
            print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
            return -1

        if res.status_code != 200:
            print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
            return -1
    except:
        print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
        return -1


    # download to tmp folder first
    output_file = os.path.join(tmp_download_dir, f"{timestamp_prefix}_{url_base_file_name}.grib2")

    # always overwrite temp file in case of corruption, since all the split gribs should be able to be produced if it is OK
    url = f'{url_folder}{url_base_file_name}.grib2'
    curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url} -o {output_file}"
    print_level(4, curl_command)

    # Run the subprocess and capture its output
    result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(sleep_time)
    if result.returncode != 0:
        print_level(3, f"Warning: Command failed with exit code {result.returncode}")
        print_level(3, f"Command: {curl_command}")
        print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
        return -2

    grib_file_path = output_file

    try:
        os.makedirs(output_dir_timestamp, exist_ok=True)
    except:
        print(1, f"Could not make folder: {output_dir_timestamp}")
        return -1

    for param in params:
        levtype = param["levtype"]
        level = param["level"]
        shortName = param["shortName"]

        output_file = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_{url_base_file_name}_{shortName}_{levtype}_{level}.grib2")

        grib_copy_command = f"grib_copy -w levtype='{levtype}',level={level},shortName='{shortName}' '{grib_file_path}' '{output_file}'"
        print_level(4, grib_copy_command)

        # Run the subprocess and capture its output
        result = subprocess.run(grib_copy_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print_level(3, f"Warning: Command failed with exit code {result.returncode}")
            print_level(3, f"Command: {curl_command}")
            print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
            # delete the tmp file now on an error
            try:
                os.remove(grib_file_path)
            except:
                # this is somewhat serious as we could run out of tmp space
                print_level(1, f"Warning could not delete tmp file: {grib_file_path}")

            return -2

    # delete the tmp file now that done
    try:
        os.remove(grib_file_path)
    except:
        print_level(1, f"Warning could not delete tmp file: {grib_file_path}")

    return 0

def generate_curl_commands_cmc(timestamp_prefix, url_folder, url_base_file_name, output_dir, params):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)

    # have no index file so use requests.head first
    # this allows us to to check if the directory exists before trying to get all the split grib files
    try:
        res = requests.head(url_folder)
        if not res:
            print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
            return -1

        if res.status_code != 200:
            print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
            return -1
    except:
        print_level(3, f"Warning: Could not get requests.head() from {url_folder}")
        return -1

    for param in params:
        url_base_file_name_with_param = url_base_file_name.replace(r'{PARAM}', param)

        output_file = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_{url_base_file_name_with_param}.grib2")

        overwrite_or_download = True

        if os.path.exists(output_file):
            overwrite_or_download = False

        if overwrite_or_download:
            url = f'{url_folder}{url_base_file_name_with_param}.grib2'
            curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url} -o {output_file}"
            print_level(4, curl_command)

            # Run the subprocess and capture its output
            result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(sleep_time)
            if result.returncode != 0:
                print_level(3, f"Warning: Command failed with exit code {result.returncode}")
                print_level(3, f"Command: {curl_command}")
                print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
                return -2

    return 0

def generate_curl_commands_ecm(timestamp_prefix, idx_file_name, url_folder, url_base_file_name, output_dir, params):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)
    idx_file_path = os.path.join(output_dir, timestamp_prefix, f'{timestamp_prefix}_{idx_file_name}')
    #os.makedirs(output_dir_timestamp, exist_ok=True)
    if not os.path.exists(idx_file_path):
        url_idx = f'{url_folder}{idx_file_name}'
        curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url_idx} -o {idx_file_path}"
        print_level(4, curl_command)

        # Run the subprocess and capture its output
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(sleep_time)
        if result.returncode != 0:
            print_level(3, f"Warning: Command failed with exit code {result.returncode}")
            print_level(3, f"Command: {curl_command}")
            print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
            try:
                # clear directory if no files ready yet
                os.rmdir(output_dir_timestamp)
            except:
                pass

            return -1

    data = load_index_file_ecm(idx_file_path)

    for param_dict in params:
        levtype = param_dict["levtype"]
        levlist = param_dict.get("levlist", "")
        param = param_dict["param"]

        matching_lines = [entry for entry in data if entry["levtype"] == levtype and entry["param"] == param and entry.get("levelist", "") == levlist]

        for entry in matching_lines:
            start = int(entry["_offset"])
            length = int(entry["_length"])
            size = length
            end = start + length - 1

            param_name = entry["param"].replace(':', '_').replace(' ', '_').lower()
            if levlist == "":
                full_name = f'{param_name}_{levtype}'
            else:
                full_name = f'{param_name}_{levlist}_{levtype}'

            full_name = full_name.replace(':', '_').replace(' ', '_').lower()
            range_str = f"{start}-{end}"

            output_file = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_{url_base_file_name}_{full_name}.grib2")

            overwrite_or_download = True

            if os.path.exists(output_file):
                if os.path.getsize(output_file) == size:
                    overwrite_or_download = False

            if overwrite_or_download:
                url = f'{url_folder}{url_base_file_name}.grib2'
                curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s -r {range_str} {url} -o {output_file}"
                print_level(4, curl_command)

                # Run the subprocess and capture its output
                result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(sleep_time)
                if result.returncode != 0:
                    print_level(3, f"Warning: Command failed with exit code {result.returncode}")
                    print_level(3, f"Command: {curl_command}")
                    print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
                    return -2

    return 0

def generate_curl_commands_gfs(timestamp_prefix, idx_file_name, url_folder, url_base_file_name, output_dir, params):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)
    idx_file_path = os.path.join(output_dir, timestamp_prefix, f'{timestamp_prefix}_{idx_file_name}')
    #os.makedirs(output_dir_timestamp, exist_ok=True)
    if not os.path.exists(idx_file_path):
        url_idx = f'{url_folder}{idx_file_name}'
        curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url_idx} -o {idx_file_path}"
        print_level(4, curl_command)

        # Run the subprocess and capture its output
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(sleep_time)
        if result.returncode != 0:
            print_level(3, f"Warning: Command failed with exit code {result.returncode}")
            print_level(3, f"Command: {curl_command}")
            print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
            try:
                # clear directory if no files ready yet
                os.rmdir(output_dir_timestamp)
            except:
                pass

            return -1

    all_lines, matching_lines = find_matching_lines_gfs(idx_file_path, params)
    total_lines = len(all_lines)

    for i, (line_num, line) in enumerate(matching_lines):
        fields = line.split(':')
        start = int(fields[1])
        if i == total_lines - 1:
            range_str = f"{start}-"
            # TODO, right now always overwrite last file? no, change from getting the http size of the url first and setting that as end-1
            size = -1
        else:
            next_line_fields = all_lines[line_num + 1][1].split(':')
            end = int(next_line_fields[1]) - 1
            range_str = f"{start}-{end}"
            size = end - start + 1

        param_name = fields[3].replace(':', '_').replace(' ', '_').lower()
        param_level = fields[4].replace(':', '_').replace(' ', '_').lower()
        output_file = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_{os.path.basename(url_base_file_name)}_{param_name}_{param_level}.grib2")

        overwrite_or_download = True

        if os.path.exists(output_file):
            if os.path.getsize(output_file) == size:
                overwrite_or_download = False

        if overwrite_or_download:
            url = f'{url_folder}{url_base_file_name}'
            curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s -r {range_str} {url} -o {output_file}"
            print_level(4, curl_command)

            # Run the subprocess and capture its output
            result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(sleep_time)
            if result.returncode != 0:
                print_level(3, f"Warning: Command failed with exit code {result.returncode}")
                print_level(3, f"Command: {curl_command}")
                print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
                try:
                    # clear directory if no files ready yet
                    os.rmdir(output_dir_timestamp)
                except:
                    pass
                return -2

    return 0

#def generate_curl_commands_gfs(timestamp_prefix, idx_file_name, url_folder, url_base_file_name, output_dir, params):
def generate_curl_commands_eps(timestamp_prefix, url_folder, file_name_prefix, finish_url_path, output_dir):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)
    output_file_prefix = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_")

    is_complete = False
    # get the ftp file listing
    try:

        url = f'{url_folder}'
        curl_command = f"curl -y 30 -Y 30 -f -v -s  --list-only {url}"
        print_level(4, curl_command)

        # Run the subprocess and capture its output
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(sleep_time)
        if result.returncode == 9:
            # Likely could not change directory to get directory listing since it doesn't exist yet
            print_level(3, f"Warning: Could not change directory to {url_folder}")
            return -1
        elif result.returncode != 0:
            print_level(3, f"Warning: Command failed with exit code {result.returncode}")
            print_level(3, f"Command: {curl_command}")
            print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
            return -2

        stdout = result.stdout
        if not stdout:
            print_level(3, f"Warning: No stdout or no files from file listing of: {url_folder}")
            return -1

        files_list_str = result.stdout.decode('utf-8')
        if not files_list_str:
            print_level(3, f"Warning: No files in listing of: {url_folder}")
            return -1

        all_files_list = files_list_str.split('\n')

        tc_files = [x.strip() for x in all_files_list if re.match(re.compile(f'{file_name_prefix}'), x.strip())]

        if not tc_files:
            # Possible there are none forecast or they have not yet been created (they are the last to be created)
            # check if the correspoding bufr is complete
            #   example is: 'https://data.ecmwf.int/forecasts/20240801/12z/ifs/0p4-beta/enfo/20240801000000-240h-enfo-tf.bufr'

            # check if finish_url_path exists
            res = requests.head(finish_url_path)
            if not res:
                print_level(3, f"Warning: Could not get requests.head() of finish_url_path {finish_url_path}")
                return -1

            if res.status_code != 200:
                print_level(3, f"Warning: Could not get requests.head() of finish_url_path: {finish_url_path}")
                return -1

            is_complete = True
            # it exists so there must not be any TCs produced in WMO essential
    except:
        print_level(3, f"Warning: Could not get ftp listing of {url_folder}")
        return -1

    if is_complete:
        # create the flag directory 'COMPLETE'
        os.makedirs(os.path.join(output_dir_timestamp, '_COMPLETE_'), exist_ok = True)
        return 0

    for tc_file_name in tc_files:
        overwrite_or_download = True

        output_file = f"{output_file_prefix}{tc_file_name}"

        if os.path.exists(output_file):
            overwrite_or_download = False

        if overwrite_or_download:
            url = f'{url_folder}{tc_file_name}'
            curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url} -o {output_file}"
            print_level(4, curl_command)

            # Run the subprocess and capture its output
            result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(sleep_time)
            if result.returncode != 0:
                print_level(3, f"Warning: Command failed with exit code {result.returncode}")
                print_level(3, f"Command: {curl_command}")
                print_level(3, f"Error output: {result.stderr.decode('utf-8')}")

                try:
                    # clear directory if no files ready yet
                    os.rmdir(output_dir_timestamp)
                except:
                    pass
                return -2

    # assume that all the files that would be created were got, and create the flag directory 'COMPLETE'
    os.makedirs(os.path.join(output_dir_timestamp, '_COMPLETE_'), exist_ok=True)
    return 0

def generate_curl_commands_gfdl(timestamp_prefix, url_folder, url_base_file_name, output_dir):
    output_dir_timestamp = os.path.join(output_dir, timestamp_prefix)

    output_file = os.path.join(output_dir_timestamp, f"{timestamp_prefix}_{os.path.basename(url_base_file_name)}.txt")

    overwrite_or_download = True

    if os.path.exists(output_file):
        overwrite_or_download = False

    if overwrite_or_download:
        url = f'{url_folder}{url_base_file_name}'
        curl_command = f"curl --create-dirs -y 30 -Y 30 -f -v -s {url} -o {output_file}"
        print_level(4, curl_command)

        # Run the subprocess and capture its output
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(sleep_time)
        if result.returncode != 0:
            print_level(3, f"Warning: Command failed with exit code {result.returncode}")
            print_level(3, f"Command: {curl_command}")
            print_level(3, f"Error output: {result.stderr.decode('utf-8')}")
            try:
                # clear directory if no files ready yet
                os.rmdir(output_dir_timestamp)
            except:
                pass
            return -2

    return 0

# the full timestep string as used in the file names we are parsing from the model data
# include any leading zeros up that make sense (only up to what the model covers)
def convert_model_time_step_to_str(model_name, model_time_step_int):
    if "TCGEN" not in model_name:
        str_format = model_time_step_str_format[model_name]
        return f'{model_time_step_int:{str_format}}'
    else:
        return model_time_step_int

def get_model_time_steps(model_name, model_timestamp):
    hour = model_timestamp.hour
    hour_str = f'{hour:02}'
    return all_time_steps_by_model[model_name][hour_str]

def get_completed_and_missing_downloaded_model_steps(model_name, model_timestamp):
    model_time_steps = get_model_time_steps(model_name, model_timestamp)
    model_base_dir = model_data_folders_by_model_name[model_name]
    model_date_str_with_hour = datetime.strftime(model_timestamp, '%Y%m%d%H')
    model_date_str = datetime.strftime(model_timestamp, '%Y%m%d')
    model_dir = os.path.join(model_base_dir, model_date_str_with_hour)
    model_hour = datetime.strftime(model_timestamp, '%H')
    model_time_step_re = re.compile(time_step_re_str_by_model_name[model_name])
    if not os.path.exists(model_dir):
        if "TCGEN" not in model_name:
            model_time_steps_int = [int(x) for x in model_time_steps]
            return [], model_time_steps_int
        else:
            # tc gen member names (or tc candidate id for EPS; for named storms, deterministic is provided also)
            #   for EPS: examples are '01ECEP', '01ECMF'
            #       (matching ensemble permutation candidate, for deterministic forecast)
            model_members = [str(x) for x in model_time_steps]
            return [], model_members
    files = os.listdir(model_dir)
    steps_downloaded = []
    steps_missing = []

    expected_num_grib_files_per_step = expected_num_grib_files_by_model_name[model_name]
    # replace step for member for tcgen ensembles
    for model_time_step in model_time_steps:
        model_time_step_str = convert_model_time_step_to_str(model_name, model_time_step)

        # may not be a trailing slash or not
        if not os.path.isdir(model_dir):
            model_timestamp_str = os.path.basename(os.path.dirname(model_dir))
        else:
            model_timestamp_str = os.path.basename(model_dir)

        model_step_timestamp = datetime.strptime(model_timestamp_str, '%Y%m%d%H')

        grib_files = []

        completed_flag = False
        for f in files:
            f_ext = f.split('.')[-1]
            file_path = os.path.join(model_dir, f)
            is_file = os.path.isfile(file_path)
            # use a completed directory as a flag for EPS tcgen folders
            if f == "_COMPLETED_":
                completed_flag = True
            elif is_file and f_ext not in ignored_file_extensions:
                res = re.match(model_time_step_re, f)
                if res:
                    file_model_time_step_str = res['time_step']
                    if file_model_time_step_str == model_time_step_str:
                        grib_files.append(file_path)

        num_grib_files_in_step = len(grib_files)

        if expected_num_grib_files_per_step != 9999:
            if "TCGEN" not in model_name:
                if num_grib_files_in_step >= expected_num_grib_files_per_step:
                    steps_downloaded.append(int(model_time_step_str))
                else:
                    steps_missing.append(int(model_time_step_str))
            else:
                if num_grib_files_in_step >= expected_num_grib_files_per_step:
                    steps_downloaded.append(model_time_step_str)
                else:
                    steps_missing.append(model_time_step_str)
        else:
            # EPS tcgen case
            if not completed_flag:
                steps_missing.append(model_time_step_str)

    return steps_downloaded, steps_missing

# returns complete (boolean), downloaded steps (int list), missing steps (int list)
def model_run_steps_downloaded_info(model_name, model_timestamp):
    model_hour = datetime.strftime(model_timestamp, '%H')
    # number of expected steps, or members for tcgen ensembles (-1 for EPS)
    expected_num_total_steps = total_model_time_steps[model_name][model_hour]
    steps_downloaded, steps_missing = get_completed_and_missing_downloaded_model_steps(model_name, model_timestamp)

    return steps_downloaded, steps_missing

def download_step(model_name, model_timestamp, time_step_int):
    model_date = datetime.strftime(model_timestamp, '%Y%m%d')
    model_hour = datetime.strftime(model_timestamp, '%H')
    # time_step_int is the member name for tcgen ensembles (a file name prefix for EPS)
    time_step = convert_model_time_step_to_str(model_name, time_step_int)

    output_dir = model_data_folders_by_model_name[model_name]
    # include trailing slash in url_folder (not in url_folder_by_model)
    model_url_folder = url_folder_by_model[model_name]
    # todo generalize:

    timestamp_prefix = f'{model_date}{model_hour}'
    if "TCGEN" not in model_name:
        params = params_by_model_name[model_name]

    if model_name == 'EPS-TCGEN':
        file_name_prefix = time_step
        url_folder = f'{model_url_folder}/{model_date}{model_hour}0000/'
        # prefix is like 'A_JSXX'; for EPS, we don't know the file names of the bufr.bin files in advance
        # finish url is a check to see whether we are actually done if we have no tc files in the listed dir
        finish_hour = convert_model_time_step_to_str('ECM', all_time_steps_by_model['ECM'][model_hour][-1])
        finish_model_url = url_folder_by_model['ECM']
        finish_url_folder = f'{finish_model_url}/{model_date}/{model_hour}z/ifs/0p4-beta/enfo/'
        finish_file_name = f'{model_date}{model_hour}0000-{finish_hour}h-enfo-tf.bufr'
        finish_url_path = f'{finish_url_folder}{finish_file_name}'
        r = generate_curl_commands_eps(timestamp_prefix, url_folder, file_name_prefix, finish_url_path, output_dir)
    elif model_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'FNMOC-TCGEN']:
        member_name = time_step
        prefix = ''
        if model_name == 'GEFS-TCGEN':
            prefix = 'gefs'
        elif model_name == 'GEPS-TCGEN':
            prefix = 'cmce'
        elif model_name == 'FNMOC-TCGEN':
            prefix = 'fens'
        url_folder = f'{model_url_folder}/{prefix}.{model_date}/{model_hour}/genesis/'
        # atcf gfdl file
        url_base_file_name = f'storms.{member_name}.atcf_gen.altg.{model_date}{model_hour}'
        r = generate_curl_commands_gfdl(timestamp_prefix, url_folder, url_base_file_name, output_dir)
    elif model_name == 'GFS':
        url_folder = f'{model_url_folder}/gfs.{model_date}/{model_hour}/atmos/'
        idx_file_name = f'gfs.t{model_hour}z.pgrb2.0p25.f{time_step}.idx'
        url_base_file_name = f'gfs.t{model_hour}z.pgrb2.0p25.f{time_step}'
        r = generate_curl_commands_gfs(timestamp_prefix, idx_file_name, url_folder, url_base_file_name, output_dir, params)
    elif model_name == 'ECM':
        url_folder = f'{model_url_folder}/{model_date}/{model_hour}z/ifs/0p4-beta/oper/'
        idx_file_name = f'{model_date}{model_hour}0000-{time_step}h-oper-fc.index'
        # we will add the .grib2 extension before we download
        url_base_file_name = f'{model_date}{model_hour}0000-{time_step}h-oper-fc'
        r = generate_curl_commands_ecm(timestamp_prefix, idx_file_name, url_folder, url_base_file_name, output_dir, params)
    elif model_name == 'CMC':
        url_folder = f'{model_url_folder}/{model_date}/WXO-DD/model_gem_global/15km/grib2/lat_lon/{model_hour}/{time_step}/'
        # we will add the .grib2 extension before we download and substitute PARAM ourselves
        url_base_file_name = r'CMC_glb_{PARAM}' + f'_latlon.15x.15_{model_date}{model_hour}_P{time_step}'
        r = generate_curl_commands_cmc(timestamp_prefix, url_folder, url_base_file_name, output_dir, params)
    elif model_name == 'NAV':
        url_folder = f'{model_url_folder}/navgem.{model_date}/'
        # we will add the .grib2 extension before we download and substitute PARAM ourselves
        url_base_file_name = f'navgem_{model_date}{model_hour}f{time_step}'
        r = generate_curl_grib_copy_commands_nav(timestamp_prefix, url_folder, url_base_file_name, output_dir, params)

    return r

def download_model_run(model_name, model_timestamp):
    # missing steps (or members) (-1 for EPS)
    already_calculated_time_steps, missing_calculated_time_steps = get_calculated_and_missing_time_steps(model_name, model_timestamp)

    if missing_calculated_time_steps is None:
        return 0

    steps_downloaded, steps_missing = model_run_steps_downloaded_info(model_name, model_timestamp)

    # remove already downloaded, calculated, and deleted steps
    steps_missing = sorted(list(set(steps_missing).intersection(set(missing_calculated_time_steps))))

    none_downloaded = (len(steps_downloaded) == 0)
    none_could_download = False
    r = 0
    if steps_missing:
        # time_step_int is a str for tcgen ensembles
        for time_step_int in steps_missing:
            if not have_enough_disk_space():
                print_level(1, "Fatal error: Exiting since disk space is below the minimum set.")
                exit()
            cur_r = download_step(model_name, model_timestamp, time_step_int)
            if cur_r != 0:
                if r != -1:
                    # OK or could not get one of the gribs
                    r = cur_r
                elif cur_r == -1:
                    # could not get an index
                    r = cur_r

                if none_downloaded:
                    none_could_download = True
                    r = -3
                    break
                # assume all time steps are complete (if there are missing time steps this will fail to download rest)
                break
            else:
                none_could_download = False
                none_downloaded = False
    return r

def download_thread(model_names):
    try:
        model_timestamps = {}
        complete_model_runs = {}
        for model_name in model_names:
            complete_model_runs[model_name] = []

        while not exit_event.is_set():
            exiting = False
            for model_name in model_names:
                model_timestamps[model_name] = get_latest_n_model_times(model_name)
                for model_timestamp in model_timestamps[model_name]:
                    if exit_event.is_set():
                        exiting = True
                        break
                    if model_timestamp in complete_model_runs[model_name]:
                        print_level(3, f'Already have {model_timestamp} for {model_name}.')
                        r = 0
                        continue
                    # download latest run with available timesteps
                    # there may be some cycling but the staleness check should always avoid infinite cycling between old runs
                    r = download_model_run(model_name, model_timestamp)
                    if r == -2 or r == -1:
                        print_level(3, f'Failed to finish downloading latest {model_name} run from {model_timestamp}. Retrying later.')
                    elif r == -3:
                        # could not even download any files, try the prior run
                        print_level(3, f'{model_name} run from {model_timestamp} not available.')
                    elif r == 0:
                        # successfully downloaded the latest run
                        print_level(2, f'Finished downloading latest {model_name} run for {model_name} {model_timestamp} (or already exists).')
                        # get the next one after backing off
                        if model_timestamp not in complete_model_runs[model_name]:
                            complete_model_runs[model_name].append(model_timestamp)
                if exiting:
                    break
            if exiting:
                break

            print_level(3, f'Backing off for {backoff_time} seconds in thread {threading.current_thread().name}.')
            time.sleep(backoff_time)

        if exiting:
            print_level(1, f'Thread {threading.current_thread().name} exiting.')

    except Exception as e:
        print_level(1, f"Thread {threading.current_thread().name} encountered an error: {e}")
        print_level(1, traceback.format_exc())
        exit_event.set()
        sys.exit(1)

# Register a signal handler to catch Ctrl+C and exit gracefully
def signal_handler(sig, frame):
    print_level(1, f"\nCtrl+C received. Exiting once download runs are finished. May take {backoff_time} s to exit. Repeat to forcefully exit (may corrupt downloads)")
    exit_event.set()
    global ctrl_c_count
    ctrl_c_count += 1

    if ctrl_c_count >= 4:
        print("Received multiple Ctrl-C. Exiting forcefully immediately.")
        os._exit(1)

if __name__ == '__main__':
    # run the downloads per thread (default )
    threads = []

    ctrl_c_count = 0

    signal.signal(signal.SIGINT, signal_handler)

    for group in model_names_to_download_thread_groupings:
        thread = threading.Thread(target=download_thread, args=(group,), name='_'.join(group))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print_level(1, "All threads have finished.")
