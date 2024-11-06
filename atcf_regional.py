# Calculate (per-storm) statistics for regional models using NHC advisories (all update advisories)
# EXPERIMENTAL!!
# Interpolates on the advisory time; weights are based on "duration" of advisory (halfway between prior/future advisory)
# To use:
# set start_urls and requested_storm_statistics (get these from the model pages)
# modify do_update to False if don't want to update pickled dataframes (stores both into "atcf_data")

# See example output for Hurricane Milton (2024) in stats_AL142024.txt (python3 atcf_regional.py > stats_AL142024.txt)

# homepages for regional models:
# https://www.emc.ncep.noaa.gov/hurricane/HFSA/tcall.php
# https://www.emc.ncep.noaa.gov/hurricane/HFSB/tcall.php
# https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HWRF_legacy/index.php
# https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HMON_legacy/index.php

import numpy as np
import pickle
import os
import re
import requests
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from datetime import datetime, timedelta
import traceback
import pytz
def datetime_utcnow():
    return datetime.now(pytz.utc).replace(tzinfo=None)

import pandas as pd

import time

from os import listdir
from os.path import isfile, join


import os
import re
import requests
from bs4 import BeautifulSoup

from geographiclib.geodesic import Geodesic

# store all the atcf files here
root_dir = "atcf_regional"
# store the advisories here
nhc_advisory_dir = "nhc_public_adv"

do_update = True

# Define the start URLs for each model (the first model init time of interest -- i.e. after genesis)
# Get from the home page under ATCF data
start_urls = [
    "https://www.emc.ncep.noaa.gov/hurricane/HFSAForecast/RT2024_NATL/EIGHTEEN18L/EIGHTEEN18L.2024110400/18l.2024110400.hfsa.trak.atcfunix",
    "https://www.emc.ncep.noaa.gov/hurricane/HFSBForecast/RT2024_NATL/EIGHTEEN18L/EIGHTEEN18L.2024110400/18l.2024110400.hfsb.trak.atcfunix",
    "https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HWRFForecast/RT2024_NATL/EIGHTEEN18L/EIGHTEEN18L.2024110400/eighteen18l.2024110400.trak.hwrf.atcfunix",
    "https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HMONForecast/RT2024_NATL/EIGHTEEN18L/EIGHTEEN18L.2024110400/eighteen18l.2024110400.trak.hmon.atcfunix",
    "https://www.emc.ncep.noaa.gov/hurricane/HFSAForecast/RT2024_NATL/RAFAEL18L/RAFAEL18L.2024110418/18l.2024110418.hfsa.trak.atcfunix",
    "https://www.emc.ncep.noaa.gov/hurricane/HFSBForecast/RT2024_NATL/RAFAEL18L/RAFAEL18L.2024110418/18l.2024110418.hfsb.trak.atcfunix",
    "https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HWRFForecast/RT2024_NATL/RAFAEL18L/RAFAEL18L.2024110418/rafael18l.2024110418.trak.hwrf.atcfunix",
    "https://www.emc.ncep.noaa.gov/gc_wmb/vxt/HMONForecast/RT2024_NATL/RAFAEL18L/RAFAEL18L.2024110418/rafael18l.2024110418.trak.hmon.atcfunix"
]

# if requested_storm_statistics = [] then do all of them
#requested_storm_statistics = []
requested_storm_statistics = ['AL182024']

# Map timezone abbreviations to their corresponding timezone names
tz_map = {
    'CDT': 'US/Central',
    'EDT': 'US/Eastern',
    'MDT': 'US/Mountain',
    'PDT': 'US/Pacific',
    'CST': 'US/Central',
    'EST': 'US/Eastern',
    'MST': 'US/Mountain',
    'PST': 'US/Pacific',
    'AST': 'America/Puerto_Rico',  # Atlantic Standard Time
    'CVT': 'Atlantic/Cape_Verde'  # Cape Verde Time
}

def ab_deck_line_to_dict(line):
    raw_data = decode_ab_deck_line(line)
    if raw_data:
        init_datetime = datetime.strptime(raw_data['YYYYMMDDHH'], '%Y%m%d%H')
        raw_data['time_step'] = raw_data['tau']
        if 'router' in raw_data.keys():
            if type(raw_data['router']) is int:
                raw_data['roci'] = raw_data['router'] * 1852
        else:
            raw_data['roci'] = None
        if 'rmw' in raw_data.keys():
            if type(raw_data['rmw']) is int:
                raw_data['rmw'] = raw_data['rmw'] * 1852
        else:
            raw_data['rmw'] = None
        if 'vmax' in raw_data.keys():
            if type(raw_data['vmax']) is int:
                if raw_data['vmax'] == 0:
                    # sometimes gets reported as 0
                    raw_data['vmax10m'] = None
                else:
                    raw_data['vmax10m'] = raw_data['vmax']
        else:
            raw_data['vmax10m'] = None

        mslp = None

        if 'mslp' in raw_data.keys():
            if type(raw_data['mslp']) is int:
                mslp = raw_data['mslp']
                if mslp > 0:
                    raw_data['mslp_value'] = mslp

        outer_slp = None
        if 'pouter' in raw_data.keys():
            if type(raw_data['pouter']) is int:
                outer_slp = raw_data['pouter']
                if outer_slp > 0:
                    raw_data['outer_slp'] = outer_slp

        raw_data['closed_isobar_delta'] = None
        if mslp and outer_slp:
            if mslp > 0 and outer_slp > 0:
                raw_data['closed_isobar_delta'] = outer_slp - mslp

        valid_datetime = init_datetime + timedelta(hours=raw_data['time_step'])
        raw_data['init_time'] = init_datetime.isoformat()
        raw_data['valid_time'] = valid_datetime.isoformat()
    return raw_data

def decode_ab_deck_line(line):
    columns = [
        "BASIN", "CY", "YYYYMMDDHH", "TECHNUM", "TECH", "TAU",
        "LatNS", "LonEW", "VMAX", "MSLP", "TY", "RAD", "WINDCODE",
        "RAD1", "RAD2", "RAD3", "RAD4", "POUTER", "ROUTER", "RMW",
        "GUSTS", "EYE", "SUBREGION", "MAXSEAS", "INITIALS", "DIR",
        "SPEED", "STORMNAME", "DEPTH", "SEAS", "SEASCODE", "SEAS1",
        "SEAS2", "SEAS3", "SEAS4", "USERDEFINED", "userdata"
    ]

    # Splitting the line by commas
    values = line.split(',')

    data = None
    if len(values) > 4:
        data = {}
        # Creating the dictionary
        for i in range(len(columns)):
            if i < len(values):
                if columns[i] in ['TAU', 'VMAX', 'MSLP', 'RMW', 'GUSTS', 'ROUTER', 'POUTER', 'SPEED']:
                    try:
                        num = int(values[i].strip())
                    except:
                        num = None
                    data[columns[i].lower()] = num
                elif columns[i] == "LatNS":
                    lat = values[i].strip()
                    if lat != '':
                        latd = lat[-1]
                        latu = lat[0:-1]
                        if latd == 'N':
                            latsigned = float(latu) / 10.0
                        else:
                            latsigned = -1.0 * float(latu) / 10.0
                        data['lat'] = latsigned
                    else:
                        data['lat'] = None

                elif columns[i] == "LonEW":
                    lon = values[i].strip()
                    if lon != '':
                        lond = lon[-1]
                        lonu = lon[0:-1]
                        if lond == 'E':
                            lonsigned = float(lonu) / 10.0
                        else:
                            lonsigned = -1.0 * float(lonu) / 10.0
                        data['lon'] = lonsigned
                    else:
                        data['lon'] = None
                else:
                    data[columns[i]] = values[i].strip()

    return data


def download_atcf_regional_tracks(urls):
    # Define the directory structure
    atcf_storm_ids = set()
    model_names = set()

    # Loop through each start URL
    for start_url in urls:
        # Extract the model name, short name, and datetime from the start URL
        model_name = re.search(r"(HFSA|HFSB|HWRF|HMON)", start_url).group(1)
        short_name = re.search(r"(hfsa|hfsb|hwrf|hmon)", start_url).group(1)
        model_names.add(model_name)
        datetime_str = re.search(r"(\d{8}\d{2})", start_url).group(1)
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H").replace(tzinfo=pytz.utc)
        # Create the directory structure if it doesn't exist
        model_dir = os.path.join(root_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Download the first file and extract the basin and number
        response = requests.get(start_url)
        if response.status_code == 200:
            atcf_file = response.text
            basin = atcf_file[0:2]
            number = atcf_file[4:6]
            #print(f"Basin: {basin}, Number: {number}")
            atcf_storm_id = f'{basin}{number}{datetime_obj.year}'
            atcf_storm_ids.add(atcf_storm_id)

            storm_dir = os.path.join(model_dir, atcf_storm_id)
            file_path = os.path.join(storm_dir, f"{atcf_storm_id}{datetime_str}.atcfunix")
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write(atcf_file)
                print(f"Downloaded file: {file_path}")

            # Loop through each 6-hour interval until a 404 is encountered
            last_datetime_str = None
            next_datetime_str = None
            while True:
                # Construct the URL for the next file
                next_datetime_obj = datetime_obj + timedelta(hours=6)
                last_datetime_str = next_datetime_str
                next_datetime_str = next_datetime_obj.strftime("%Y%m%d%H")
                next_url = start_url.replace(datetime_str, next_datetime_str)
                if last_datetime_str is None:
                    last_datetime_str = next_datetime_str

                # Check if the file already exists
                storm_dir = os.path.join(model_dir, atcf_storm_id)
                if not os.path.exists(storm_dir):
                    os.makedirs(storm_dir)
                next_file_path = os.path.join(storm_dir, f"{atcf_storm_id}{next_datetime_str}.atcfunix")
                if os.path.exists(next_file_path):
                    #print(f"File already exists: {next_file_path}")
                    datetime_obj = next_datetime_obj
                    continue

                # Download the next file
                response = requests.get(next_url)
                if response.status_code == 200:
                    atcf_file = response.text
                    with open(next_file_path, "w") as f:
                        f.write(atcf_file)
                    print(f"Downloaded file: {next_file_path}")
                    datetime_obj = next_datetime_obj
                elif response.status_code == 404:
                    print(f"Latest {model_name} model is from: {last_datetime_str}")
                    break
                else:
                    print(f"Error: {response.status_code}")
                    break
        else:
            #print(f"Error: {response.status_code}")
            pass

    return atcf_storm_ids, model_names

def get_deck_files(urls):
    atcf_storm_ids, model_names = download_atcf_regional_tracks(urls)
    adeck = defaultdict(dict)
    year = datetime_utcnow().year
    most_recent_model_dates = defaultdict(lambda: datetime.min)
    dt_mods_adeck = {}

    for storm_id in sorted(list(atcf_storm_ids)):
        basin_id = storm_id[:2]
        storm_number = storm_id[2:4]
        # Download A-Deck files
        storm_id = f"{basin_id}{storm_number}{year}"
        global root_dir

        rows = []

        for model_name in model_names:
            model_storm_dir = os.path.join(root_dir, model_name, storm_id)
            files = [f for f in listdir(model_storm_dir) if isfile(join(model_storm_dir, f))]
            for file_name in files:
                local_filename = os.path.join(root_dir, model_name, storm_id, file_name)
                try:
                    if not os.path.exists(local_filename):
                        continue

                    dt_mod = os.path.getmtime(local_filename)
                    if not dt_mod:
                        # no mtime (should not happen)
                        print(f"Warning: Could not get modification time for: {local_filename}, skipping")
                        continue

                    lines = []
                    with open(local_filename, 'r') as f:
                        lines = f.readlines()

                    if lines is None or len(lines) == 0:
                        continue

                    latest_date = datetime.min

                    # see if it is latest first before decoding
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) > 4 and parts[3].strip() == '03':  # Model identifier '03'
                            date_str = parts[2].strip()
                            model_date = datetime.strptime(date_str, '%Y%m%d%H')
                            latest_date = model_date

                    most_recent_model_dates[storm_id] = latest_date
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) > 4 and parts[3].strip() == '03':  # TECH '03' are forecast models
                            date_str = parts[2].strip()
                            model_date = datetime.strptime(date_str, '%Y%m%d%H')
                            ab_deck_line_dict = ab_deck_line_to_dict(line)
                            model_id = ab_deck_line_dict['TECH']
                            valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])
                            init_datetime = datetime.fromisoformat(ab_deck_line_dict['init_time'])
                            if storm_id not in adeck.keys():
                                adeck[storm_id] = {}
                            if model_id not in adeck[storm_id].keys():
                                adeck[storm_id][model_id] = {}
                            if init_datetime not in adeck[storm_id][model_id].keys():
                                adeck[storm_id][model_id][init_datetime.isoformat()] = {}
                            ab_deck_line_dict['basin'] = basin_id.upper()
                            adeck[storm_id][model_id][init_datetime.isoformat()][valid_datetime.isoformat()] = ab_deck_line_dict

                            row_dict = {
                                'storm_id': storm_id,
                                'model_id': model_id,
                                'init_datetime': init_datetime,  # keep datetime object
                                'valid_datetime': valid_datetime,  # keep datetime object
                            }

                            # Add the ab_deck_line_dict key-value pairs, excluding init_time and valid_time
                            for key, value in ab_deck_line_dict.items():
                                if key not in ['init_time', 'valid_time']:
                                    row_dict[key] = value

                            # Append the row dictionary to the list
                            rows.append(row_dict)

                    dt_mods_adeck[local_filename] = dt_mod
                except OSError as e:
                    traceback.print_exc()
                    print(f"OSError opening/reading file: {e}")
                except UnicodeDecodeError as e:
                    traceback.print_exc()
                    print(f"UnicodeDecodeError: {e}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"Failed to read file: {e}")

    df = pd.DataFrame(rows)

    return dt_mods_adeck, df, adeck, atcf_storm_ids



def download_nhc_public_advisories(storm_id):
    # Construct the URL for the NHC archive page
    basin = storm_id[:2].lower()  # e.g., "al"
    year = storm_id[4:8]  # e.g., "2024"
    storm_number = storm_id[2:4]  # e.g., "14"
    url = f"https://www.nhc.noaa.gov/archive/{year}/{basin}{storm_number}/index.shtml"
    print(url)

    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}. Status code: {response.status_code}")

        return

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links on the page
    links = soup.find_all('a', href=True)

    # Create the folder structure if it doesn't exist
    folder_path = f"{nhc_advisory_dir}/{storm_id.upper()}/"
    os.makedirs(folder_path, exist_ok=True)

    file_paths = []

    # Iterate over the links and download the matching files
    for link in links:
        href = link['href']
        if re.match(rf"/archive/{year}/{basin}{storm_number}/{basin}{storm_number}{year}\.(public|update)", href):
            file_url = f"https://www.nhc.noaa.gov/{href}"
            file_path = os.path.join(folder_path, os.path.basename(href))
            # strip ? from end
            if file_path[-1] == '?':
                file_path = file_path[:-1]

            # Check if the file already exists
            if os.path.exists(file_path):
                #print(f"Skipping {file_url} (already exists)")
                file_paths.append(file_path)
                continue

            print(f"Downloading {file_url} to {file_path}")

            # Send a GET request to the file URL and save the response to a file
            file_response = requests.get(file_url)
            with open(file_path, 'w') as file:
                file.write(file_response.text)

            file_paths.append(file_path)

            time.sleep(1)

    return file_paths

def df_from_ofcl_advisory(html):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the <pre> tag with the SUMMARY section
    pre_tags = soup.find_all('pre')
    summary_pre = None
    for pre in pre_tags:
        text = pre.get_text()
        lines = text.split('\n')
        for line in lines:
            if re.search(r'SUMMARY OF \d+ (?:AM|PM) (?:CDT|EDT|MDT|PDT|CST|EST|MST|PST)...', line):
                summary_pre = pre
                break
        if summary_pre:
            break

    if not summary_pre:
        print("No SUMMARY section found")
        return None

    # Extract the text from the SUMMARY section
    summary_text = summary_pre.get_text()
    lines = summary_text.split('\n')

    # Initialize variables to store the extracted information
    storm_name = None
    init_storm_type = None
    advisory_date = None
    advisory_time = None
    timezone = None
    atcf_id = None
    vmax_kt = None
    mslp_hpa = None
    storm_direction = None
    storm_heading = None
    lat = None
    lon = None

    # Extract the storm name and type from the heading
    r_name = re.compile(r'(?P<init_storm_type>(Sub|Post\-|Ex\-)?(Tropical)?(Storm|Depression|Hurricane)) +(?P<storm_name>\w+) (Special )?+(Intermediate )?Advisory Number +(?P<advisory_num>\d+[\w]?)')
    r_name2 = re.compile(r'(?P<init_storm_type>(Sub|Post\-|Ex\-)?(Tropical Storm|Tropical Depression|Hurricane)) +(?P<storm_name>\w+) +[\w ]*?(?P<update>Update)')
    advisory = False
    update = False
    for line in lines:
        match = r_name.search(line)
        match2 = r_name2.search(line)
        if match:
            storm_name = match.group('storm_name')
            init_storm_type = match.group('init_storm_type')
            advisory = True
            break
        if match2:
            storm_name = match2.group('storm_name')
            init_storm_type = match2.group('init_storm_type')
            update = True
            break

    # Extract the advisory date and time from the heading
    r_advisory_date = re.compile(r'^(?P<advisory_time>\d+) (?P<am_pm>[AP]M) (?P<timezone>\w+) (?P<day_of_week>\w+) +(?P<month_abbrev>\w+) +(?P<advisory_day_num>\d+) +(?P<advisory_year>\d\d\d\d)')
    for line in lines:
        match = r_advisory_date.search(line)

        if match:
            advisory_time = match.group('advisory_time')
            am_pm = match.group('am_pm')
            timezone = match.group('timezone')
            hour = int(advisory_time[:len(advisory_time) - 2])  # Get the hour part
            minute = advisory_time[len(advisory_time) - 2:]  # Get the minute part
            if am_pm == 'PM' and hour != 12:
                hour += 12
            elif am_pm == 'AM' and hour == 12:
                hour = 0
            advisory_time = f"{hour:02d}{minute}"  # Zero-pad the hour
            advisory_date = f"{match.group('advisory_year')}-{match.group('month_abbrev')}-{match.group('advisory_day_num')} {advisory_time}"
            advisory_date = pd.to_datetime(advisory_date, format='%Y-%b-%d %H%M')  # Specify the format
            tz = pytz.timezone(tz_map[timezone])  # Get the timezone object
            advisory_date = tz.localize(advisory_date).astimezone(pytz.utc)  # Convert to UTC
            break

    # Extract the ATCF ID from the heading
    r_atcf = re.compile(r'^NWS (?:National Hurricane Center Miami FL|Central Pacific Hurricane Center Honolulu HI) +(?P<atcf_id>\w+)')
    for line in lines:
        match = r_atcf.search(line)
        if match:
            atcf_id = match.group('atcf_id')
            break

    # Extract the SUMMARY section information
    summary_section = False
    for line in lines:
        if re.search(r'SUMMARY OF \d+ (?:AM|PM) (?:\w+)...', line):
            summary_section = True
        elif summary_section and line.strip() == '':
            break
        elif summary_section:
            if re.search(r'LOCATION...', line):
                match = re.search(r'LOCATION...(?P<lat>\d+\.\d+)(?P<lat_dir>N|S) +(?P<lon>\d+\.\d+)(?P<lon_dir>W|E)', line)
                if match:
                    lat = float(match.group('lat'))
                    if match.group('lat_dir') == 'S':
                        lat = -lat
                    lon = float(match.group('lon'))
                    if match.group('lon_dir') == 'W':
                        lon = -lon
            elif re.search(r'MAXIMUM SUSTAINED WINDS...', line):
                match = re.search(r'MAXIMUM SUSTAINED WINDS...(?P<vmax_mph>\d+) MPH...(?P<vmax_kmh>\d+) KM/H', line)
                if match:
                    vmax_kt = float(match.group('vmax_kmh')) * 0.539956803
            elif re.search(r'PRESENT MOVEMENT...', line):
                match = re.search(r'PRESENT MOVEMENT...(?P<storm_direction>\w+) OR (?P<storm_heading>\d+) DEGREES AT (?P<storm_speed_mph>\d+) MPH...(?P<storm_speed_kmh>\d+) KM/H', line)
                if match:
                    storm_direction = match.group('storm_direction')
                    storm_heading = float(match.group('storm_heading'))
                    storm_speed_kt = float(match.group('storm_speed_kmh')) * 0.539956803
            elif re.search(r'MINIMUM CENTRAL PRESSURE...', line):
                match = re.search(r'MINIMUM CENTRAL PRESSURE...(?P<mslp_mb>\d+) MB', line)
                if match:
                    mslp_hpa = float(match.group('mslp_mb'))

    # Create a DataFrame with the extracted information
    df = pd.DataFrame({
        'ATCF_ID': [atcf_id],
        'STORM_NAME': [storm_name],
        'INIT_STORM_TYPE': [init_storm_type],
        'ADVISORY_DATE': [advisory_date],
        'ADVISORY': advisory,
        'LAT': [lat],
        'LON': [lon],
        'VMAX_KT': [vmax_kt],
        'MSLP_HPA': [mslp_hpa],
        'STORM_DIRECTION': [storm_direction],
        'STORM_HEADING': [storm_heading],
        'STORM_SPEED_KT': storm_speed_kt
    })

    return df

def parse_storm_advisories(storms_file_paths):
    storms_dfs = {}
    for atcf_storm_id, storm_file_paths in storms_file_paths.items():
        storm_dfs = []
        for file_path in storm_file_paths:
            with open(file_path, 'r') as f:
                html = f.read()
            df = df_from_ofcl_advisory(html)
            if df is not None:
                storm_dfs.append(df)
        if storm_dfs:
            storm_df = pd.concat(storm_dfs, ignore_index=True).reset_index(drop=True)
            storms_dfs[atcf_storm_id] = storm_df
    return storms_dfs

def update_files_for_decks_and_advisories():

    dt_mods_adeck, df_adeck, adecks, atcf_storm_ids = get_deck_files(start_urls)

    storms_file_paths = {}
    for atcf_storm_id in atcf_storm_ids:
        storm_file_paths = download_nhc_public_advisories(atcf_storm_id)
        storms_file_paths[atcf_storm_id] = storm_file_paths

    storms_dfs = parse_storm_advisories(storms_file_paths)

    # Save the DataFrames
    save_data(df_adeck, storms_dfs, 'atcf_data')

def save_data(df_adeck, storms_dfs, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'df_adeck': df_adeck, 'storms_dfs': storms_dfs}, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['df_adeck'], data['storms_dfs']


def analyze_storms(storms_dfs, df_adeck, requested_storm_statistics):
    if len(requested_storm_statistics) > 0:
        for storm_id in requested_storm_statistics:
            adeck_storm_ids = set(df_adeck['storm_id'])
            valid_storm_id = storm_id in storms_dfs and storm_id in adeck_storm_ids
            if not valid_storm_id:
                print("Invalid storm id (skipping):", storm_id)
                continue

            df_adeck_filtered = df_adeck.loc[df_adeck['storm_id'] == storm_id]
            storms_dfs_filtered = {storm_id: storms_dfs[storm_id]}
            print()
            print()
            print()
            print()
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print(f"STATISTICS FOR {storm_id}:")
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print()
            print()
            print()
            print()
            analyze_storm(storms_dfs, df_adeck)
    else:
        requested_storm_statistics = list(set(df_adeck['storm_id']))
        for storm_id in requested_storm_statistics:
            adeck_storm_ids = set(df_adeck['storm_id'])
            valid_storm_id = storm_id in storms_dfs and storm_id in adeck_storm_ids
            if not valid_storm_id:
                continue

            df_adeck_filtered = df_adeck.loc[df_adeck['storm_id'] == storm_id]
            storms_dfs_filtered = {storm_id: storms_dfs[storm_id]}
            print()
            print()
            print()
            print()
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print(f"STATISTICS FOR {storm_id}:")
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print('==========================')
            print()
            print()
            print()
            print()
            analyze_storm(storms_dfs_filtered, df_adeck_filtered)


def analyze_storm(storms_dfs, df_adeck):

    # Define the column mapping
    column_mapping = {
        'valid_time': 'ADVISORY_DATE',
        'lat': 'LAT',
        'lon': 'LON',
        'vmax10m': 'VMAX_KT',
        'mslp_value': 'MSLP_HPA',
        'DIR': 'STORM_HEADING',
        'speed': 'STORM_SPEED_KT',
        'LAT_LON': ('LAT', 'LON'),
    }

    column_mapping_stats = {
        'vmax10m': 'VMAX_KT',
        'mslp_value': 'MSLP_HPA',
        'DIR': 'STORM_HEADING',
        'speed': 'STORM_SPEED_KT',
        'LAT_LON': 'LAT_LON',
    }

    # no direction or speed for HWRF
    exclude_models_per_column = {
        'DIR': ['HWRF'],
        'speed': ['HWRF']
    }

    # Initialize the interpolated DataFrames
    interp_dfs = {}

    for storm_name, storm_df in storms_dfs.items():
        storm_df['LAT_LON'] = storm_df.apply(lambda row: (row['LAT'], row['LON']), axis=1)
        storm_df.drop_duplicates(inplace=True)
        storm_df = storm_df.sort_values(by='ADVISORY_DATE')
        storm_df = storm_df.set_index('ADVISORY_DATE')  # Set ADVISORY_DATE as index

        for (model_name, init_time), run_df in df_adeck.groupby(['model_id', 'init_datetime']):
            run_df.drop_duplicates(inplace=True)
            run_df = run_df.sort_values(by='valid_datetime')
            run_df.reset_index(inplace=True)

            # Interpolate the model data
            df_temp = run_df.copy()
            df_temp['interp_valid_time'] = pd.to_datetime(df_temp['valid_datetime'],
                                                            utc=True)  # Convert to datetime with UTC timezone
            df_temp.drop_duplicates(subset='interp_valid_time', inplace=True)
            valid_times = df_temp['interp_valid_time']

            df_temp.reset_index(inplace=True)

            # Get the ADVISORY_DATE datetimes
            advisory_dates = storm_df.index

            # Get the valid times for this model run
            #valid_times = run_df.index

            # Create a new dataframe to store the interpolated values
            interp_df = pd.DataFrame(index=advisory_dates)

            # Loop through each advisory date
            valid_times_series = pd.Series(valid_times)
            valid_times_dt = pd.to_datetime(valid_times, unit='s')

            interp_df['LAT_LON'] = pd.Series(index=interp_df.index, dtype=object)

            for advisory_date in advisory_dates:
                # Find the nearest two points in the model run
                advisory_date_np = advisory_date.to_numpy()
                advisory_date_np = pd.to_datetime(advisory_date_np, utc=True)

                prior_idx_list = valid_times_dt[valid_times_dt <= advisory_date_np].index
                if len(prior_idx_list) > 0:
                    prior_idx = prior_idx_list[0]
                else:
                    prior_idx = None  # or some other default value

                future_idx_list = valid_times_dt[valid_times_dt > advisory_date_np].index
                if len(future_idx_list) > 0:
                    future_idx = future_idx_list[0]
                else:
                    future_idx = valid_times_dt[-1]

                # Handle edge cases
                if prior_idx == future_idx:
                    # Exact match, use the same values for interp columns
                    prior_weight = 1.0
                    future_weight = 0.0
                elif advisory_date < valid_times.min():
                    # Before the first point, use the first point for interp
                    prior_idx = valid_times.idxmin()
                    future_idx = prior_idx
                    prior_weight = 1.0
                    future_weight = 0.0
                elif advisory_date > valid_times.max():
                    # After the last point, use the last point for interp
                    prior_idx = valid_times.idxmax()
                    future_idx = future_idx
                    prior_weight = 1.0
                    future_weight = 0.0
                else:
                    # Calculate the weights based on the time distance
                    prior_time = valid_times.loc[prior_idx]
                    future_time = valid_times.loc[future_idx]
                    time_diff = future_time - prior_time
                    prior_weight = (future_time - advisory_date) / time_diff
                    future_weight = (advisory_date - prior_time) / time_diff

                # Store the weights in the interp dataframe
                interp_df.loc[advisory_date, 'prior_weight'] = prior_weight
                interp_df.loc[advisory_date, 'future_weight'] = future_weight

                # Interpolate the values using the weights
                for col in ['lat', 'lon', 'vmax10m', 'mslp_value', 'DIR', 'speed']:
                    prior_value = run_df.loc[prior_idx, col]
                    future_value = run_df.loc[future_idx, col]
                    interp_value = prior_weight * float(prior_value) + future_weight * float(future_value)
                    interp_df.loc[advisory_date, col] = interp_value

                # Interpolate the lat_lon values using the geodesic method
                prior_lat_lon = (run_df.loc[prior_idx, 'lat'], run_df.loc[prior_idx, 'lon'])
                future_lat_lon = (run_df.loc[future_idx, 'lat'], run_df.loc[future_idx, 'lon'])
                g = Geodesic.WGS84
                L = g.Inverse(prior_lat_lon[0], prior_lat_lon[1], future_lat_lon[0], future_lat_lon[1])
                az = L['azi1']
                distance = L['s12']
                interp_lat_lon = g.Direct(prior_lat_lon[0], prior_lat_lon[1], az, distance * prior_weight)
                interp_df.at[advisory_date, 'LAT_LON'] = (interp_lat_lon['lat2'], interp_lat_lon['lon2'])


                # Calculate the interp_valid_time
                prior_valid_time = run_df.loc[prior_idx, 'valid_datetime']
                future_valid_time = run_df.loc[future_idx, 'valid_datetime']
                interp_valid_time = prior_valid_time + (future_valid_time - prior_valid_time) * prior_weight
                interp_df.loc[advisory_date, 'interp_valid_time'] = interp_valid_time

                # Calculate the interpolated duration
                interp_df['interp_duration'] = np.nan
                for i in range(1, len(interp_df) - 1):
                    prev_time = interp_df.index[i - 1]
                    curr_time = interp_df.index[i]
                    next_time = interp_df.index[i + 1]
                    duration = (next_time - prev_time).total_seconds() / 2
                    interp_df.at[curr_time, 'interp_duration'] = duration

                # Normalize the interpolated duration
                interp_df['interp_weight'] = interp_df['interp_duration'] / interp_df['interp_duration'].sum()

                # Add the interpolated DataFrame to the dictionary
                interp_dfs[(storm_name, model_name, init_time)] = interp_df

    # Calculate the statistics for each model run
    stats = {}
    g = Geodesic.WGS84
    for (storm_name, model_name, init_time), interp_df in interp_dfs.items():
        storm_df = storms_dfs[storm_name]
        stats[(model_name, init_time)] = {}
        interp_df = interp_df.reset_index()

        # Merge interp_df and storm_df on ADVISORY_DATE
        merged_df = pd.merge(interp_df, storm_df, on='ADVISORY_DATE', suffixes=('_interp', '_storm'))

        for col in column_mapping_stats:
            col_diff = f'{col}_diff'
            col_diff_sq = f'{col}_diff_sq'

            if col == "LAT_LON":
                # Compute geodesic distance
                merged_df[col_diff] = merged_df.apply(
                    lambda row: g.Inverse(row[f'{col}_interp'][0], row[f'{col}_interp'][1],
                                          row[f'{column_mapping_stats[col]}_storm'][0],
                                          row[f'{column_mapping_stats[col]}_storm'][1])['s12'], axis=1)
            else:
                merged_df[col_diff] = merged_df[col] - merged_df[column_mapping[col]]

            merged_df[col_diff_sq] = merged_df[col_diff] ** 2

            # Calculate the unweighted statistics
            abs_error = np.abs(merged_df[col_diff]).mean()
            bias = merged_df[col_diff].mean()
            mse = merged_df[col_diff_sq].mean()
            rmse = np.sqrt(mse)
            stats[(model_name, init_time)][f'{col}_unweighted'] = {
                'abs_error': abs_error,
                'bias': bias,
                'mse': mse,
                'rmse': rmse,
                'cases': len(merged_df),
                'duration': merged_df['interp_duration'].sum()
            }

            # Calculate the weighted statistics
            weighted_abs_error = np.abs(merged_df[col_diff] * merged_df['interp_weight']).sum()
            weighted_bias = (merged_df[col_diff] * merged_df['interp_weight']).sum()
            weighted_mse = (merged_df[col_diff_sq] * merged_df['interp_weight']).sum()
            weighted_rmse = np.sqrt(weighted_mse)
            stats[(model_name, init_time)][f'{col}_weighted'] = {
                'abs_error': weighted_abs_error,
                'bias': weighted_bias,
                'mse': weighted_mse,
                'rmse': weighted_rmse,
                'cases': len(merged_df),
                'duration': merged_df['interp_duration'].sum()
            }

        # Calculate the aggregated statistics for each model
        model_stats = {}
        for model_name, model_runs in stats.items():
            model_stats[model_name] = {}
            for col in column_mapping:
                if 'valid_time' in col:
                    continue
                col_unweighted = f'{col}_unweighted'
                col_weighted = f'{col}_weighted'

                model_stats[model_name][col_unweighted] = {
                    'min': None,
                    'median': None,
                    'mean': None,
                    'max': None
                }

                model_stats[model_name][col_weighted] = {
                    'min': None,
                    'median': None,
                    'mean': None,
                    'max': None
                }

                unweighted_rmse_values = []
                weighted_rmse_values = []

                for run in model_runs.values():
                    if col_unweighted in run and 'rmse' in run[col_unweighted]:
                        unweighted_rmse_values.append(run[col_unweighted]['rmse'])
                    if col_weighted in run and 'rmse' in run[col_weighted]:
                        weighted_rmse_values.append(run[col_weighted]['rmse'])

                if unweighted_rmse_values:
                    model_stats[model_name][col_unweighted]['min'] = np.min(unweighted_rmse_values)
                    model_stats[model_name][col_unweighted]['median'] = np.median(unweighted_rmse_values)
                    model_stats[model_name][col_unweighted]['mean'] = np.mean(unweighted_rmse_values)
                    model_stats[model_name][col_unweighted]['max'] = np.max(unweighted_rmse_values)

                if weighted_rmse_values:
                    model_stats[model_name][col_weighted]['min'] = np.min(weighted_rmse_values)
                    model_stats[model_name][col_weighted]['median'] = np.median(weighted_rmse_values)
                    model_stats[model_name][col_weighted]['mean'] = np.mean(weighted_rmse_values)
                    model_stats[model_name][col_weighted]['max'] = np.max(weighted_rmse_values)



    # Find the best model run for each model, by metric

    metrics = [
        'LAT_LON',
        'vmax10m',
        'mslp_value',
        'DIR',
        'speed'
    ]

    for metric in metrics:
        print('\n\n\n\n')
        print('==========================================')
        print(f'        {metric} METRICS')
        print('==========================================')
        best_runs_per_model = {}
        best_run_overall = {'unweighted': None, 'weighted': None}
        best_rmse_overall = {'unweighted': float('inf'), 'weighted': float('inf')}

        for weight_type in ['unweighted', 'weighted']:
            best_runs_per_model[weight_type] = {}

            for (model_name, init_time), model_runs in stats.items():
                if metric in exclude_models_per_column and model_name in exclude_models_per_column[metric]:
                    continue
                metric_name = f'{metric}_{weight_type}'
                rmse = model_runs[metric_name]['rmse']

                # Update best run for this model
                if model_name not in best_runs_per_model[weight_type] or rmse < \
                        best_runs_per_model[weight_type][model_name]['rmse']:
                    best_runs_per_model[weight_type][model_name] = {'init_time': init_time, 'rmse': rmse}

                # Update best run overall
                if rmse < best_rmse_overall[weight_type]:
                    best_rmse_overall[weight_type] = rmse
                    best_run_overall[weight_type] = (model_name, init_time)

        print(f"Best runs per model ({metric}):")
        for weight_type in ['unweighted', 'weighted']:
            print(f"\n{weight_type.capitalize()}:")
            for model_name, best_run in best_runs_per_model[weight_type].items():
                print(f"  {model_name}: {best_run['init_time']} (RMSE: {best_run['rmse']:.1f})")

        print(f"\nBest run overall ({metric}):")
        for weight_type in ['unweighted', 'weighted']:
            print(f"  {weight_type.capitalize()}: {best_run_overall[weight_type]} (RMSE: {best_rmse_overall[weight_type]:.1f})")

        # Calculate the aggregated statistics for all models
        all_model_stats = {}
        for col in column_mapping:
            if 'valid_time' in col:
                continue
            col_unweighted = f'{col}_unweighted'
            col_weighted = f'{col}_weighted'
            unweighted_mins = [model.get(col_unweighted, {}).get('min') for model in model_stats.values()]
            unweighted_mins = [x for x in unweighted_mins if x is not None]
            unweighted_medians = [model.get(col_unweighted, {}).get('median') for model in model_stats.values()]
            unweighted_medians = [x for x in unweighted_medians if x is not None]
            unweighted_means = [model.get(col_unweighted, {}).get('mean') for model in model_stats.values()]
            unweighted_means = [x for x in unweighted_means if x is not None]
            unweighted_maxes = [model.get(col_unweighted, {}).get('max') for model in model_stats.values()]
            unweighted_maxes = [x for x in unweighted_maxes if x is not None]

            weighted_mins = [model.get(col_weighted, {}).get('min') for model in model_stats.values()]
            weighted_mins = [x for x in weighted_mins if x is not None]
            weighted_medians = [model.get(col_weighted, {}).get('median') for model in model_stats.values()]
            weighted_medians = [x for x in weighted_medians if x is not None]
            weighted_means = [model.get(col_weighted, {}).get('mean') for model in model_stats.values()]
            weighted_means = [x for x in weighted_means if x is not None]
            weighted_maxes = [model.get(col_weighted, {}).get('max') for model in model_stats.values()]
            weighted_maxes = [x for x in weighted_maxes if x is not None]

            all_model_stats[col_unweighted] = {
                'min': np.min(unweighted_mins) if unweighted_mins else float('inf'),
                'median': np.median(unweighted_medians) if unweighted_medians else float('nan'),
                'mean': np.mean(unweighted_means) if unweighted_means else float('nan'),
                'max': np.max(unweighted_maxes) if unweighted_maxes else float('-inf')
            }
            all_model_stats[col_weighted] = {
                'min': np.min(weighted_mins) if weighted_mins else float('inf'),
                'median': np.median(weighted_medians) if weighted_medians else float('nan'),
                'mean': np.mean(weighted_means) if weighted_means else float('nan'),
                'max': np.max(weighted_maxes) if weighted_maxes else float('-inf')
            }

        # Find the best model run overall
        best_model_run_unweighted = min(best_runs_per_model['unweighted'],
                                        key=lambda x: best_runs_per_model['unweighted'][x]['rmse'])
        best_model_run_weighted = min(best_runs_per_model['weighted'],
                                      key=lambda x: best_runs_per_model['weighted'][x]['rmse'])
        print(f'Best model run overall (unweighted) ({metric}): {best_model_run_unweighted}')
        print(f'Best model run overall (weighted) ({metric}): {best_model_run_weighted}')

    return stats, model_stats, all_model_stats

def print_ace(df_adeck):
    print('\n\n\n\n')
    print('==========================================')
    print(f'        ACE values (kt^2 * 10^-4)')
    print('==========================================')
    for (model_name, init_time), run_df in df_adeck.groupby(['model_id', 'init_datetime']):
        run_df.drop_duplicates(subset='tau', inplace=True)
        run_df = run_df.sort_values(by='valid_datetime')
        run_df.reset_index(inplace=True)
        filtered_df = run_df[(run_df['tau'] % 6 == 0) & (run_df['vmax'] >= 34)]
        filtered_df.reset_index(inplace=True)
        model_ace = pow(10, -4) * np.sum(np.power(filtered_df['vmax'], 2))
        print(model_name, init_time, round(model_ace, 1))
    print('\n\n\n\n')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# run only when needed
if do_update:
    update_files_for_decks_and_advisories()

# Load the DataFrames
df_adeck, storms_dfs = load_data('atcf_data')

print_ace(df_adeck)

print(storms_dfs)

# This will analyze metrics per storm, and for all storms in df
analyze_storms(storms_dfs, df_adeck, requested_storm_statistics)

