# an experimental TC plotter (standa-alone full screen viewer) for intensity models and genesis
######
#   uses a-deck,b-deck,tcvitals (from NHC,UCAR) and tc genesis candidates (tc_candidates.db)

#### EXPERIMENTAL: DO NOT USE OR RELY ON THIS!

## NOTES
# ADECK MODE: 'ALL' excludes GEFS members (only GEFS members visible in specific combo selection)

######## GUI Keys/Functions
# ZOOM IN = mouse left click and drag
# ZOOM OUT = mouse right click
# ZOOM OUT MAX = 0 key (0 key)
# ZOOM IN STEP ON CURSOR = = key (equal key)
# ZOOM OUT STEP ON CURSOR = - key (minus key)
# ABORT ZOOM = Escape key  (if started dragging hit escape to abort a zoom)
# MEASURE (GEODESIC) = shift + mouse left click and drag
# ERASE MEASURE = right click
# VIEW NEXT OVERLAPPED HOVER POINT STATUS = n key  (this will not redraw points of overlapped storms, only update hover text)
# CIRCLE & ANNOTATE STORM EXTREMA = x key  (annotates either the selected storm in circle patch, or all storms in current view (zoom))
# CLEAR ALL STORM EXTREMA ANNOTATIONS = c key

####### CONFIG

# how often (in minutes) to check for stale data in three classes: tcvitals, adeck, bdecks
#   for notification purposes only..
#      colors reload button red (a-deck), orange (b-deck), yellow (tcvitals) -without- downloading data automatically
#   checks modification date in any class of the three, and refreshes the entire class
#      timer resets after manual reloads
TIMER_INTERVAL_MINUTES = 30

# On mouse hover, select points to display status within a 0.1 degree bounding box
MOUSE_SELECT_IN_DEGREES = 0.5

# Size of circle patch radius on mouse hover (in pixels)
DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS = 20

# assume any subsequent circle annotations (for extrema) in this bounding box are overlapped in the display, so don't render
ANNOTATE_CIRCLE_OVERLAP_IN_DEGREES = 0.01

# how much to zoom in when using minus key
ZOOM_IN_STEP_FACTOR = 1.05

# remove any from list not wanted visible for extrema nnotations (x key)
#   closed_isobar_delta is only from tc_candidates.db
#      (it is the difference in pressure from the outermost closed isobar to the inner most close isobar) (i.e. how many concentric circles on a chart of pressure)
#displayed_extremum_annotations = ["dt_start", "dt_end", "vmax10m", "mslp_value", "roci", "closed_isobar_delta"]
displayed_extremum_annotations = ["dt_start", "vmax10m", "roci", "closed_isobar_delta"]

DEFAULT_ANNOTATE_MARKER_COLOR = "#FFCCCC"
DEFAULT_ANNOTATE_TEXT_COLOR = "#FFCCCC"
ANNOTATE_DT_START_COLOR = "#00FF00"
ANNOTATE_VMAX_COLOR = "#FF60F0"

from datetime import datetime, timedelta
# URLs
tcvitals_urls = [
    "https://ftp.nhc.noaa.gov/atcf/com/tcvitals",
    "https://hurricanes.ral.ucar.edu/repository/data/tcvitals_open/combined_tcvitals.{year}.dat".format(year=datetime.utcnow().year)
]

adeck_urls = [
    "https://ftp.nhc.noaa.gov/atcf/aid_public/a{basin_id}{storm_number}{year}.dat.gz",
    "https://hurricanes.ral.ucar.edu/repository/data/adecks_open/a{basin_id}{storm_number}{year}.dat"
]

bdeck_urls = [
    "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{basin_id}{storm_number}{year}.dat",
    "https://ftp.nhc.noaa.gov/atcf/btk/b{basin_id}{storm_number}{year}.dat",
    "https://ftp.nhc.noaa.gov/atcf/btk/cphc/b{basin_id}{storm_number}{year}.dat"
]

### END CONFIG ###

# for cycling overlapped points
from collections import OrderedDict

# for mouse hover features
from rtree import index

# for measurements (geodesic)
#   note: measures geodesic distance, not the 'path' distance that might show as a blue line
import cartopy.geodesic as cgeo
from shapely.geometry import LineString
import numpy as np

# for tracking modification date of source files
from dateutil import parser
import copy

# for plotting boundaries
import geopandas as gpd
from shapely.geometry import Polygon

import traceback
import requests
import gzip
import os
from collections import defaultdict

import json
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle, Circle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from mpl_toolkits.axes_grid1 import Divider, Size

import sqlite3

from matplotlib.collections import LineCollection

# performance optimizations
import matplotlib

# presence of importance of  level 1 item overrides 0, and level 2 overrides both levels 0 and 1 for coloring
annotate_color_levels = {0: DEFAULT_ANNOTATE_TEXT_COLOR, 1: ANNOTATE_DT_START_COLOR, 2: ANNOTATE_VMAX_COLOR}

tcvitals_basin_to_atcf_basin = {
    'E': 'EP',
    'L': 'AL',
    'C': 'CP',
    'S': 'SH',
    'B': 'IO',
    'A': 'IO',
    'Q': 'SH',
    'P': 'SH',
    'W': 'WP'
}

### plot greater houston boundary
# Load the shapefile using Geopandas
# from census shapefiles
#shapefile_path = "tl_2023_us_cbsa/tl_2023_us_cbsa.shp"

try:
    gdf = gpd.read_file(shapefile_path)
    # Filter the GeoDataFrame to only include the Houston-The Woodlands-Sugar Land, TX Metro Area
    houston_gdf = gdf[gdf['NAME'] == 'Houston-Pasadena-The Woodlands, TX']
    if houston_gdf.empty:
        raise ValueError("Houston-The Woodlands-Sugar Land, TX Metro Area not found in the shapefile")
    # Ensure the CRS matches that of the Cartopy map (PlateCarree)
    custom_gdf = houston_gdf.to_crs(ccrs.PlateCarree().proj4_init)
except:
    custom_gdf = None

matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = 100

# this is for accessing by model and storm (internal component id)
tc_candidates_db_file_path = 'tc_candidates.db'

model_data_folders_by_model_name = {
    'GFS': '/home/db/metview/JRPdata/globalmodeldata/gfs',
    'ECM': '/home/db/metview/JRPdata/globalmodeldata/ecm',
    'CMC': '/home/db/metview/JRPdata/globalmodeldata/cmc',
    'NAV': '/home/db/metview/JRPdata/globalmodeldata/nav'
}

intensity_models = {
    'AC00',
    'AEMI',
    'AEMN',
    'AP01',
    'AP02',
    'AP03',
    'AP04',
    'AP05',
    'AP06',
    'AP07',
    'AP08',
    'AP09',
    'AP10',
    'AP11',
    'AP12',
    'AP13',
    'AP14',
    'AP15',
    'AP16',
    'AP17',
    'AP18',
    'AP19',
    'AP20',
    'AP21',
    'AP22',
    'AP23',
    'AP24',
    'AP25',
    'AP26',
    'AP27',
    'AP28',
    'AP29',
    'AP30',
    'AVNI',
    'AVNO',
    'CEM2',
    'CEMI',
    'CEMN',
    'CMC',
    'CMC2',
    'CMCI',
    'DRCL',
    'DSHP',
    'ECM2',
    'ECMI',
    'EGR2',
    'EGRI',
    'EGRR',
    'EMX2',
    'EMXI',
    'FSSE',
    'HCCA',
    'HFA2',
    'HFAI',
    'HFB2',
    'HFBI',
    'HFSA',
    'HFSB',
    'HMN2',
    'HMNI',
    'HMON',
    'HWF2',
    'HWFI',
    'HWRF',
    'ICON',
    'IVCN',
    'LGEM',
    'NGX',
    'NGX2',
    'NGXI',
    'NNIB',
    'NNIC',
    'NVG2',
    'NVGI',
    'NVGM',
    'OCD5',
    'OFCI',
    'OFCL',
    'RI25',
    'RI30',
    'RI35',
    'RI40',
    'RVCN',
    'RYOC',
    'SHF5',
    'SHIP',
    'SPC3',
    'TABD',
    'TABM',
    'TABS',
    'TCLP',
    'UKM',
    'UKM2',
    'UKMI',
    'UKX',
    'UKX2',
    'UKXI',
    'XTRP',
    'TCVITALS',
    'BEST'
}
# exclude these models from all lists (except GEFS-MEMBERS)
excluded_models = {
    'TABS',
    'TABM',
    'TABD'
    'AC00',
    'AP01',
    'AP02',
    'AP03',
    'AP04',
    'AP05',
    'AP06',
    'AP07',
    'AP08',
    'AP09',
    'AP10',
    'AP11',
    'AP12',
    'AP13',
    'AP14',
    'AP15',
    'AP16',
    'AP17',
    'AP18',
    'AP19',
    'AP20',
    'AP21',
    'AP22',
    'AP23',
    'AP24',
    'AP25',
    'AP26',
    'AP27',
    'AP28',
    'AP29',
    'AP30',
    'RI25',
    'RI30',
    'RI35',
    'RI40'
}
included_intensity_models = set(intensity_models) - set(excluded_models)
statistical_models = ['OCD5', 'TCLP', 'SHIP', 'DSHP', 'LGEM', 'DRCL', 'SHF5', 'BEST', 'TCVITALS']
gefs_members_models = [
    'AC00',
    'AP01',
    'AP02',
    'AP03',
    'AP04',
    'AP05',
    'AP06',
    'AP07',
    'AP08',
    'AP09',
    'AP10',
    'AP11',
    'AP12',
    'AP13',
    'AP14',
    'AP15',
    'AP16',
    'AP17',
    'AP18',
    'AP19',
    'AP20',
    'AP21',
    'AP22',
    'AP23',
    'AP24',
    'AP25',
    'AP26',
    'AP27',
    'AP28',
    'AP29',
    'AP30',
    'BEST',
    'TCVITALS'
]
global_models = [
    'AVNO',
    'AVNI',
    'AEMN',
    'AEMI',
    'AEM2',
    'AMMN',
    'CMC',
    'CMCI',
    'CMC2',
    'EGR2',
    'EGRI',
    'EGRR',
    'UKM',
    'UKM2',
    'UKMI',
    'UKX',
    'UKX2',
    'UKXI'
    'ECM2',
    'ECMI',
    'EMX2',
    'EMXI',
    'NGX',
    'NGX2',
    'NGXI',
    'NNIB',
    'NNIC',
    'NVG2',
    'NVGI',
    'NVGM',
    'BEST',
    'TCVITALS'
]
regional_models = ['HFSA', 'HFSB', 'HFAI', 'HFA2', 'HFBI', 'HFB2',
                   'HMON',  'HMNI', 'HMN2',
                   'HWRF', 'HWF2', 'HWFI',  'HWFI', 'BEST', 'TCVITALS']
official_models = ['OFCI', 'OFCL', 'BEST', 'TCVITALS']
consensus_models = ['ICON', 'IVCN', 'RVCN', 'NNIC', 'NNIB', 'BEST', 'TCVITALS']

# get model cycles relative to init_date timestamp
def get_tc_model_init_times_relative_to(init_date):
    if type(init_date) == str:
        init_date = datetime.fromisoformat(init_date)

    conn = None
    # 'at' can be strictly before init_date, but previous must be also before 'at'
    model_init_times = {'previous': None, 'next': None, 'at': None}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT init_date FROM completed WHERE init_date <= ? ORDER BY init_date DESC LIMIT 1', (datetime.isoformat(init_date),))
        results = cursor.fetchall()
        if results:
            for row in results:
                init_time = row[0]
                model_init_times['at'] = datetime.fromisoformat(init_time)

        if model_init_times['at']:
            cursor.execute('SELECT DISTINCT init_date FROM completed WHERE init_date > ? ORDER BY init_date ASC LIMIT 1', (datetime.isoformat(model_init_times['at']),))
            results = cursor.fetchall()
            if results:
                for row in results:
                    init_time = row[0]
                    model_init_times['next'] = datetime.fromisoformat(init_time)

        if model_init_times['at']:
            cursor.execute('SELECT DISTINCT init_date FROM completed WHERE init_date < ? ORDER BY init_date DESC LIMIT 1', (datetime.isoformat(model_init_times['at']),))
            results = cursor.fetchall()
            if results:
                for row in results:
                    init_time = row[0]
                    model_init_times['previous'] = datetime.fromisoformat(init_time)



    except sqlite3.Error as e:
        print(f"SQLite error (get_tc_candidates_from_valid_time): {e}")
    finally:
        if conn:
            conn.close()

    return model_init_times

# get list of completed TC candidates
def get_tc_candidates_from_valid_time(interval_start):
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    model_names = list(model_data_folders_by_model_name.keys())
    model_init_times = {}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()

        for model_name in model_names:
            cursor.execute('SELECT DISTINCT init_date FROM completed WHERE model_name = ? ORDER BY init_date DESC LIMIT 1', (model_name, ))
            results = cursor.fetchall()
            recent_model_init_times = []
            if results:
                for row in results:
                    init_time = row[0]
                    recent_model_init_times.append(init_time)
                    if model_name not in model_init_times:
                        model_init_times[model_name] = []

                for init_date in recent_model_init_times:
                    model_init_times[model_name].append(init_date)
                    cursor.execute('SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE model_name = ? AND init_date = ? AND start_valid_date >= ? ORDER BY start_valid_date DESC', (model_name, init_date, interval_start))
                    # Query all rows from the 'disturbances' table and order by 'model_timestamp'
                    results = cursor.fetchall()
                    if results:
                        # Process data for each row
                        for row in results:
                            model_name, init_date, start_valid_date, ws_max_10m, json_data = row
                            retrieved_data = {
                                "model_name": model_name,
                                "model_timestamp": init_date,
                                "start_valid_time": start_valid_date,
                                "ws_max_10m": ws_max_10m,
                                "disturbance_candidates": json.loads(json_data)
                            }
                            all_retrieved_data.append(retrieved_data)

    except sqlite3.Error as e:
        print(f"SQLite error (get_tc_candidates_from_valid_time): {e}")
    finally:
        if conn:
            conn.close()

    return model_init_times, all_retrieved_data

# get list of completed TC candidates
def get_tc_candidates_at_or_before_init_time(interval_end):
    if type(interval_end) == str:
        interval_end = datetime.fromisoformat(interval_end)

    all_retrieved_data = []  # List to store data from all rows
    conn = None
    model_names = list(model_data_folders_by_model_name.keys())
    model_init_times = {}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()

        for model_name in model_names:
            cursor.execute('SELECT DISTINCT init_date FROM completed WHERE model_name = ? AND init_date <= ? ORDER BY init_date DESC LIMIT 1', (model_name, datetime.isoformat(interval_end)))
            results = cursor.fetchall()
            recent_model_init_times = []
            if results:
                for row in results:
                    init_time = row[0]
                    recent_model_init_times.append(init_time)
                    if model_name not in model_init_times:
                        model_init_times[model_name] = []

                for init_date in recent_model_init_times:
                    model_init_times[model_name].append(init_date)
                    cursor.execute('SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE model_name = ? AND init_date = ? ORDER BY start_valid_date DESC', (model_name, init_date))
                    # Query all rows from the 'disturbances' table and order by 'model_timestamp'
                    results = cursor.fetchall()
                    if results:
                        # Process data for each row
                        for row in results:
                            model_name, init_date, start_valid_date, ws_max_10m, json_data = row
                            retrieved_data = {
                                "model_name": model_name,
                                "model_timestamp": init_date,
                                "start_valid_time": start_valid_date,
                                "ws_max_10m": ws_max_10m,
                                "disturbance_candidates": json.loads(json_data)
                            }
                            all_retrieved_data.append(retrieved_data)

    except sqlite3.Error as e:
        print(f"SQLite error (get_tc_candidates_from_valid_time): {e}")
    finally:
        if conn:
            conn.close()

    return model_init_times, all_retrieved_data

def http_get_modification_date(url):
    try:
        response = requests.head(url)
        # Check if the request was successful (status code 200 or 2xx)
        if response.status_code // 100 == 2:
            # Extract the modification date from the 'Last-Modified' header
            modification_date = response.headers.get('Last-Modified')
            return modification_date
        # If the request was not successful, return None
        return None

    except Exception as e:
        print(f"An error occurred getting {url}: {e}")
        return None

# Function to download a file from a URL
def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            dt_mod = get_modification_date_from_header(r.headers)
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to download {url}: {e}")
        return None
    return dt_mod

# Function to get the most recent records for each storm from TCVitals files
def get_recent_storms(tcvitals_urls):
    storms = {}
    dt_mods_tcvitals = {}
    current_time = datetime.utcnow()
    for url in tcvitals_urls:
        response = requests.get(url)
        if response.status_code == 200:
            dt_mod = get_modification_date_from_header(response.headers)
            lines = response.text.splitlines()
            for line in lines:
                parts = line.split()
                timestamp = datetime.strptime(parts[3] + parts[4], '%Y%m%d%H%M')
                basin = tcvitals_basin_to_atcf_basin[parts[1][2]]
                storm_number = parts[1][0:2]
                year = parts[3][0:4]
                storm_id = f"{basin}{storm_number}{year}"
                if timestamp >= current_time - timedelta(hours=24):
                    if storm_id not in storms or storms[storm_id]['timestamp'] < timestamp:
                        storms[storm_id] = {
                            'timestamp': timestamp,
                            'data': line
                        }

            dt_mods_tcvitals[url] = dt_mod

    recent_storms = {}
    for storm_id, val in storms.items():
        data = val['data']
        storm_dict = tcvitals_line_to_dict(data)
        recent_storms[storm_id] = storm_dict
    return dt_mods_tcvitals, recent_storms

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
        valid_datetime = init_datetime + timedelta(hours=raw_data['time_step'])
        raw_data['init_time'] = init_datetime.isoformat()
        raw_data['valid_time'] = valid_datetime.isoformat()
    return raw_data

def get_modification_date_from_header(responseheaders):
    try:
        # Assume already done status code check
            modification_date = responseheaders.get('Last-Modified')
            dt_offset_aware = parser.parse(modification_date)
            dt_offset_native = dt_offset_aware.astimezone().replace(tzinfo=None)
            return dt_offset_native
    except:
        return None

def http_get_modification_date(url):
    try:
        response = requests.head(url)
        if response.status_code // 100 == 2:
            return get_modification_date_from_header(response.headers)
        # If the request was not successful, return None
        return None

    except Exception as e:
        print(f"An error occurred getting {url}: {e}")
        return None

# Function to get the corresponding A-Deck and B-Deck files for the identified storms
def get_deck_files(storms, adeck_urls, bdeck_urls, do_update_adeck, do_update_bdeck):
    adeck = defaultdict(dict)
    bdeck = defaultdict(dict)
    year = datetime.utcnow().year
    most_recent_model_dates = defaultdict(lambda: datetime.min)
    most_recent_bdeck_dates = defaultdict(lambda: datetime.min)
    dt_mods_adeck = {}
    dt_mods_bdeck = {}

    for storm_id in storms.keys():
        basin_id = storm_id[:2]
        storm_number = storm_id[2:4]
        # Download A-Deck files
        if do_update_adeck:
            for url in adeck_urls:
                file_url = url.format(basin_id=basin_id.lower(), year=year, storm_number=storm_number)
                isgz = False
                if file_url[-3:] == ".gz":
                    isgz = True
                    local_filename = f"a{storm_id.lower()}.dat.gz"
                else:
                    local_filename = f"a{storm_id.lower()}.dat"
                try:
                    dt_mod = download_file(file_url, local_filename)
                    if not dt_mod:
                        # download failed
                        continue

                    with open(local_filename, 'rb') as f:
                        file_header = f.read(4)

                    # If the file starts with '1f 8b', it's likely a gzipped file
                    if file_header[:2] == b'\x1f\x8b':
                        with gzip.open(local_filename, 'rt') as z:
                            file_content = z.read()
                    else:
                        with open(local_filename, 'rt') as f:
                            file_content = f.read()

                    os.remove(local_filename)

                    lines = file_content.splitlines()
                    latest_date = datetime.min

                    # see if it is latest first before decoding
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) > 4 and parts[3].strip() == '03':  # Model identifier '03'
                            date_str = parts[2].strip()
                            model_date = datetime.strptime(date_str, '%Y%m%d%H')
                            if model_date > latest_date:
                                latest_date = model_date

                    # now we know if we have data to update from source
                    if latest_date > most_recent_model_dates[storm_id]:
                        most_recent_model_dates[storm_id] = latest_date
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) > 4 and parts[3].strip() == '03':  # TECH '03' are forecast models
                                date_str = parts[2].strip()
                                model_date = datetime.strptime(date_str, '%Y%m%d%H')
                                if model_date == latest_date:
                                    ab_deck_line_dict = ab_deck_line_to_dict(line)
                                    model_id = ab_deck_line_dict['TECH']
                                    valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])
                                    if storm_id not in adeck.keys():
                                        adeck[storm_id] = {}
                                    if model_id not in adeck[storm_id].keys():
                                        adeck[storm_id][model_id] = {}
                                    adeck[storm_id][model_id][valid_datetime.isoformat()] = ab_deck_line_dict
                                elif model_date >= (latest_date - timedelta(hours=6)):
                                    # GEFS members ATCF is reported late by 6 hours...
                                    ab_deck_line_dict = ab_deck_line_to_dict(line)
                                    model_id = ab_deck_line_dict['TECH']
                                    # allow only late GEFS members:
                                    if model_id[0:2] in ['AC', 'AP']:
                                        valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])
                                        if storm_id not in adeck.keys():
                                            adeck[storm_id] = {}
                                        if model_id not in adeck[storm_id].keys():
                                            adeck[storm_id][model_id] = {}
                                        adeck[storm_id][model_id][valid_datetime.isoformat()] = ab_deck_line_dict

                    dt_mods_adeck[file_url] = dt_mod
                except OSError as e:
                    traceback.print_exc()
                    print(f"OSError opening/reading file: {e}")
                except UnicodeDecodeError as e:
                    traceback.print_exc()
                    print(f"UnicodeDecodeError: {e}")
                    with gzip.open(local_filename, 'rb') as z:
                        raw_content = z.read(10)  # Read a few bytes to check the content
                        print(f"First 10 bytes of the file (in hex): {raw_content.hex()}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"Failed to download {file_url}: {e}")
        if do_update_bdeck:
            # Download B-Deck files
            for url in bdeck_urls:
                file_url = url.format(year=year, basin_id=basin_id.lower(), storm_number=storm_number)
                try:
                    response = requests.get(file_url)
                    if response.status_code == 200:
                        dt_mod = get_modification_date_from_header(response.headers)

                        lines = response.text.splitlines()
                        latest_date = datetime.min

                        # see if it is latest first before decoding
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) > 4:
                                date_str = parts[2].strip()
                                bdeck_date = datetime.strptime(date_str, '%Y%m%d%H')
                                if bdeck_date > latest_date:
                                    latest_date = bdeck_date

                        # now we know if we have data to update from source
                        if latest_date >= most_recent_bdeck_dates[storm_id]:
                            most_recent_bdeck_dates[storm_id] = latest_date
                            for line in lines:
                                parts = line.split(',')
                                if len(parts) > 4:
                                    date_str = parts[2].strip()
                                    bdeck_date = datetime.strptime(date_str, '%Y%m%d%H')
                                    ab_deck_line_dict = ab_deck_line_to_dict(line)
                                    # id should be 'BEST'
                                    bdeck_id = ab_deck_line_dict['TECH']
                                    valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])
                                    if storm_id not in bdeck.keys():
                                        bdeck[storm_id] = {}
                                    if bdeck_id not in bdeck[storm_id].keys():
                                        bdeck[storm_id][bdeck_id] = {}
                                    bdeck[storm_id][bdeck_id][valid_datetime.isoformat()] = ab_deck_line_dict

                        dt_mods_bdeck[file_url] = dt_mod
                except OSError as e:
                    traceback.print_exc()
                    print(f"OSError opening/reading file: {e}")
                except UnicodeDecodeError as e:
                    traceback.print_exc()
                    print(f"UnicodeDecodeError: {e}")
                    with gzip.open(local_filename, 'rb') as z:
                        raw_content = z.read(10)  # Read a few bytes to check the content
                        print(f"First 10 bytes of the file (in hex): {raw_content.hex()}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"Failed to download {file_url}: {e}")
    return dt_mods_adeck, dt_mods_bdeck, adeck, bdeck

def parse_tcvitals_line(line):
    if len(line) >= 149:
        return {
            "organization_id": line[0:4].strip(),
            "storm_id": line[5:7].strip(),
            "basin_identifier": line[7].strip(),
            "storm_name": line[9:18].strip(),
            "first_occurrence_indicator": line[18].strip(),
            "report_date": line[19:27].strip(),
            "report_time": line[28:32].strip(),
            "latitude": line[33:36].strip(),
            "latitude_indicator": line[36].strip(),
            "longitude": line[38:42].strip(),
            "longitude_indicator": line[42].strip(),
            "storm_direction_flag": line[43].strip(),
            "storm_direction": line[44:47].strip(),
            "storm_speed_flag": line[47].strip(),
            "storm_speed": line[48:51].strip(),
            "central_pressure_flag": line[51].strip(),
            "central_pressure": line[52:56].strip(),
            "environmental_pressure_flag": line[56].strip(),
            "environmental_pressure": line[57:61].strip(),
            "radius_isobar_flag": line[61].strip(),
            "radius_isobar": line[62:66].strip(),
            "max_wind_flag": line[66].strip(),
            "max_wind_speed": line[67:69].strip(),
            "radius_max_wind": line[70:73].strip(),
            "radius_34_ne": line[74:78].strip(),
            "radius_34_se": line[79:83].strip(),
            "radius_34_sw": line[84:88].strip(),
            "radius_34_nw": line[89:93].strip(),
            "storm_depth_flag": line[93].strip(),
            "storm_depth": line[94].strip(),
            "radius_50_ne": line[96:100].strip(),
            "radius_50_se": line[101:105].strip(),
            "radius_50_sw": line[106:110].strip(),
            "radius_50_nw": line[111:115].strip(),
            "max_forecast_time": line[116:118].strip(),
            "forecast_latitude": line[119:122].strip(),
            "forecast_latitude_indicator": line[122].strip(),
            "forecast_longitude": line[124:128].strip(),
            "forecast_longitude_indicator": line[128].strip(),
            "radius_64_ne": line[130:134].strip(),
            "radius_64_se": line[135:139].strip(),
            "radius_64_sw": line[140:144].strip(),
            "radius_64_nw": line[145:149].strip()
        }
    elif len(line) >= 95 :
        return {
            "organization_id": line[0:4].strip(),
            "storm_id": line[5:7].strip(),
            "basin_identifier": line[7].strip(),
            "storm_name": line[9:18].strip(),
            "first_occurrence_indicator": line[18].strip(),
            "report_date": line[19:27].strip(),
            "report_time": line[28:32].strip(),
            "latitude": line[33:36].strip(),
            "latitude_indicator": line[36].strip(),
            "longitude": line[38:42].strip(),
            "longitude_indicator": line[42].strip(),
            "storm_direction_flag": line[43].strip(),
            "storm_direction": line[44:47].strip(),
            "storm_speed_flag": line[47].strip(),
            "storm_speed": line[48:51].strip(),
            "central_pressure_flag": line[51].strip(),
            "central_pressure": line[52:56].strip(),
            "environmental_pressure_flag": line[56].strip(),
            "environmental_pressure": line[57:61].strip(),
            "radius_isobar_flag": line[61].strip(),
            "radius_isobar": line[62:66].strip(),
            "max_wind_flag": line[66].strip(),
            "max_wind_speed": line[67:69].strip(),
            "radius_max_wind": line[70:73].strip(),
            "radius_34_ne": line[74:78].strip(),
            "radius_34_se": line[79:83].strip(),
            "radius_34_sw": line[84:88].strip(),
            "radius_34_nw": line[89:93].strip(),
            "storm_depth_flag": line[93].strip(),
            "storm_depth": line[94].strip()
        }

def parse_tcvitals_line(line):
    if len(line) >= 149:
        return {
            "organization_id": line[0:4].strip(),
            "storm_id": line[5:7].strip(),
            "basin_identifier": line[7].strip(),
            "storm_name": line[9:18].strip(),
            "first_occurrence_indicator": line[18].strip(),
            "report_date": line[19:27].strip(),
            "report_time": line[28:32].strip(),
            "latitude": line[33:36].strip(),
            "latitude_indicator": line[36].strip(),
            "longitude": line[38:42].strip(),
            "longitude_indicator": line[42].strip(),
            "storm_direction_flag": line[43].strip(),
            "storm_direction": line[44:47].strip(),
            "storm_speed_flag": line[47].strip(),
            "storm_speed": line[48:51].strip(),
            "central_pressure_flag": line[51].strip(),
            "central_pressure": line[52:56].strip(),
            "environmental_pressure_flag": line[56].strip(),
            "environmental_pressure": line[57:61].strip(),
            "radius_isobar_flag": line[61].strip(),
            "radius_isobar": line[62:66].strip(),
            "max_wind_flag": line[66].strip(),
            "max_wind_speed": line[67:69].strip(),
            "radius_max_wind": line[70:73].strip(),
            "radius_34_ne": line[74:78].strip(),
            "radius_34_se": line[79:83].strip(),
            "radius_34_sw": line[84:88].strip(),
            "radius_34_nw": line[89:93].strip(),
            "storm_depth_flag": line[93].strip(),
            "storm_depth": line[94].strip(),
            "radius_50_ne": line[96:100].strip(),
            "radius_50_se": line[101:105].strip(),
            "radius_50_sw": line[106:110].strip(),
            "radius_50_nw": line[111:115].strip(),
            "max_forecast_time": line[116:118].strip(),
            "forecast_latitude": line[119:122].strip(),
            "forecast_latitude_indicator": line[122].strip(),
            "forecast_longitude": line[124:128].strip(),
            "forecast_longitude_indicator": line[128].strip(),
            "radius_64_ne": line[130:134].strip(),
            "radius_64_se": line[135:139].strip(),
            "radius_64_sw": line[140:144].strip(),
            "radius_64_nw": line[145:149].strip()
        }
    elif len(line) >= 95 :
        return {
            "organization_id": line[0:4].strip(),
            "storm_id": line[5:7].strip(),
            "basin_identifier": line[7].strip(),
            "storm_name": line[9:18].strip(),
            "first_occurrence_indicator": line[18].strip(),
            "report_date": line[19:27].strip(),
            "report_time": line[28:32].strip(),
            "latitude": line[33:36].strip(),
            "latitude_indicator": line[36].strip(),
            "longitude": line[38:42].strip(),
            "longitude_indicator": line[42].strip(),
            "storm_direction_flag": line[43].strip(),
            "storm_direction": line[44:47].strip(),
            "storm_speed_flag": line[47].strip(),
            "storm_speed": line[48:51].strip(),
            "central_pressure_flag": line[51].strip(),
            "central_pressure": line[52:56].strip(),
            "environmental_pressure_flag": line[56].strip(),
            "environmental_pressure": line[57:61].strip(),
            "radius_isobar_flag": line[61].strip(),
            "radius_isobar": line[62:66].strip(),
            "max_wind_flag": line[66].strip(),
            "max_wind_speed": line[67:69].strip(),
            "radius_max_wind": line[70:73].strip(),
            "radius_34_ne": line[74:78].strip(),
            "radius_34_se": line[79:83].strip(),
            "radius_34_sw": line[84:88].strip(),
            "radius_34_nw": line[89:93].strip(),
            "storm_depth_flag": line[93].strip(),
            "storm_depth": line[94].strip()
        }
    else:
        return None

def tcvitals_line_to_dict(line):
    storm_vitals = parse_tcvitals_line(line)
    if storm_vitals:
        date_report_str = f"{storm_vitals['report_date']}{storm_vitals['report_time']}"
        try:
            valid_datetime = datetime.strptime(date_report_str, '%Y%m%d%H%M')
            storm_vitals['valid_time'] = valid_datetime.isoformat()
        except:
            return None

        latu = storm_vitals['latitude']
        latd = storm_vitals['latitude_indicator']
        lonu = storm_vitals['longitude']
        lond = storm_vitals['longitude_indicator']

        if latd == 'N':
            latsigned = float(latu) / 10.0
        else:
            latsigned = -1.0 * float(latu) / 10.0
        storm_vitals['lat'] = latsigned

        if lond == 'E':
            lonsigned = float(lonu) / 10.0
        else:
            lonsigned = -1.0 * float(lonu) / 10.0
        storm_vitals['lon'] = lonsigned

        try:
            vmax = float(storm_vitals['max_wind_speed']) * 1.9438452
            storm_vitals['vmax10m'] = vmax
        except:
            storm_vitals['vmax10m'] = None

        try:
            mslp = int(storm_vitals['central_pressure'])
            storm_vitals['mslp'] = mslp
        except:
            storm_vitals['mslp'] = None

    return storm_vitals

# returns a list of modified/created or keys in new_dict
def diff_dicts(old_dict, new_dict):
    new_keys = new_dict.keys() - old_dict.keys()
    modified_keys = {key for key in old_dict.keys() & new_dict.keys() if new_dict[key] != old_dict[key]}
    all_changed_keys = new_keys | modified_keys
    return all_changed_keys

# a sorted cyclic dict that has the item number enumerated (sorted by value)
# has a get() the next enumerated number and the key of the item in a cycle
class SortedCyclicEnumDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_value = lambda x: x[1]  # default sort by value
        self._index = 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._reorder()

    def _reorder(self):
        sorted_items = sorted(self.items(), key=self._sort_value)
        temp_dict = OrderedDict(sorted_items)
        self.clear()
        for key, value in temp_dict.items():
            super().__setitem__(key, value)
        self._index = 0

    def get_first_key(self):
        if not self:
            return None
        return next(iter(self.items()))[0]

    def next_enum_key_tuple(self):
        if not self:
            return None
        sorted_items = list(self.items())
        index = self._index % len(self)
        key, value = sorted_items[index]
        self._index = (self._index + 1) % len(self)
        return (index + 1, key)

    # get previous key without moving backwards (assumes have gotten at least one tuple with next)
    def get_prev_enum_key_tuple(self):
        if not self:
            return None
        sorted_items = list(self.items())
        prev_index = (self._index - 1) % len(self)
        key, value = sorted_items[prev_index]
        return (prev_index + 1, key)

class DraggableAnnotation:
    def __init__(self, annotation, original_point, ax, bbox_props):
        self.original_annotation = annotation
        self.ax = ax
        self.zorder = self.original_annotation.get_zorder()

        self.original_point = original_point
        self.press = None
        self.background = None
        self.line = None
        self.bbox_props = bbox_props
        self.radius_degrees = self.calculate_radius_pixels()

        self.dragging = False

        # Extract essential properties from the original annotation
        text = self.original_annotation.get_text()
        xy = self.original_annotation.get_position()
        xytext = getattr(self.original_annotation, '_xytext', xy)

        # Define essential properties to keep
        essential_props = [
            'color', 'fontsize', 'fontweight', 'fontstyle',
            'horizontalalignment', 'verticalalignment',
            'rotation', 'wrap'
        ]

        props = {prop: getattr(self.original_annotation, f'get_{prop}')()
                 for prop in essential_props if hasattr(self.original_annotation, f'get_{prop}')}

        # Ensure we have valid coordinates
        xycoords = getattr(self.original_annotation, 'xycoords', 'data')
        textcoords = getattr(self.original_annotation, 'textcoords', xycoords)

        # Create a copy of the annotation for dragging
        self.dragging_annotation = self.ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            xycoords=xycoords,
            textcoords=textcoords,
            bbox=self.bbox_props,  # Use the bbox_props passed to the constructor
            **props
        )
        self.dragging_annotation.set_visible(False)

        #self.bring_to_front()

        # Connect to the event handlers
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def is_dragging(self):
        return self.dragging

    @classmethod
    def get_topmost_annotation(cls, annotations, event):
        # need a separate is dragging flag as bbox is unstable when we are blitting
        #  (as we it can become invisible during the setup for blitting (in an async event with the other handlers)
        #  the bbox will result erroneously in a 1 pixel box causing the wrong annotation to drag
        valid_annotations = [ann for ann in annotations if ann.is_dragging() or ann.contains_point(event)]
        if not valid_annotations:
            return None
        max_zorder = max(valid_annotations, key=lambda ann: ann.zorder).zorder
        return max(valid_annotations, key=lambda ann: ann.zorder)

    def calculate_radius_pixels(self):
        # Get current extent of the map in degrees and pixels
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = self.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # Convert pixels to degrees
        radius_degrees = max(lon_deg_per_pixel, lat_deg_per_pixel) * DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS

        return radius_degrees

    def contains_point(self, event):
        bbox = self.original_annotation.get_window_extent()
        return bbox.contains(event.x, event.y)

    def on_press(self, event):
        if self != self.get_topmost_annotation(self.ax.draggable_annotations, event):
            return

        contains, attrd = self.original_annotation.contains(event)
        if not contains:
            return

        xy_orig = self.original_annotation.xy
        if self.dragging_annotation and self.dragging_annotation.xy:
            xy_cur = self.dragging_annotation.xy
        else:
            xy_cur = xy_orig
        self.press = (xy_orig, self.original_annotation.get_position(), event.xdata, event.ydata)
        self.original_annotation.set_visible(False)
        if self.line:
            self.line.set_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.dragging_annotation.set_position(self.original_annotation.get_position())
        self.dragging_annotation.set_visible(True)
        if self.line:
            self.line.set_visible(True)
        self.ax.figure.canvas.draw()
        self.dragging = True

    def bring_to_front(self):
        if self.ax.draggable_annotations:
            top_zorder = max(ann.zorder for ann in self.ax.draggable_annotations) + 1
            self.zorder = top_zorder
            self.original_annotation.set_zorder(top_zorder)
            self.dragging_annotation.set_zorder(top_zorder)
        else:
            # first
            top_zorder = 10
            self.zorder = top_zorder
            self.original_annotation.set_zorder(top_zorder)
            self.dragging_annotation.set_zorder(top_zorder)

    def on_release(self, event):
        if self.press is None:
            return

        # Update original annotation position
        self.original_annotation.set_position(self.dragging_annotation.get_position())

        # Show original annotation and hide dragging annotation
        self.dragging_annotation.set_visible(False)
        self.original_annotation.set_visible(True)

        self.press = None
        self.background = None
        self.dragging = False

        self.bring_to_front()

        # Redraw the figure to reflect the changes
        self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if self.press is None:
            return

        (x0, y0), (x0_cur, y0_cur), xpress, ypress = self.press

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        inbound = False
        if event.inaxes == self.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

        if not(inbound) or not(xpress) or not(ypress):
            return

        dx = event.xdata - xpress
        dy = event.ydata - ypress

        cur_x = x0_cur + dx
        cur_y = y0_cur + dy
        # Convert the offset from display coordinates to data coordinates
        new_x = x0_cur + dx
        new_y = y0_cur + dy

        # Calculate the position on the circle's edge
        angle = np.arctan2(new_y - y0, new_x - x0)
        edge_x = x0 + self.radius_degrees * np.cos(angle)
        edge_y = y0 + self.radius_degrees * np.sin(angle)

        self.dragging_annotation.set_position((cur_x, cur_y))

        # Remove the old line and draw the new line
        if self.line:
            self.line.remove()
        self.line = self.ax.plot(
            [edge_x, new_x],
            [edge_y, new_y],
            linestyle='--',
            color=DEFAULT_ANNOTATE_TEXT_COLOR,
            transform=ccrs.PlateCarree(),
        )[0]

        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.dragging_annotation)
        self.ax.draw_artist(self.line)
        self.ax.figure.canvas.blit(self.ax.bbox)

    def remove(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        if self.line:
            self.line.remove()
        self.original_annotation.remove()
        self.dragging_annotation.remove()

class AnnotatedCircles():
    def __init__(self, ax):
        self.ax = ax
        self.circle_handles = None
        #self.annotation_handles = None
        self.ax.draggable_annotations = None
        self.rtree_p = index.Property()
        self.rtree_idx = index.Index(properties=self.rtree_p)

    def get_annotations(self):
        return self.ax.draggable_annotations

    # calculate radius of pixels in degrees
    def calculate_radius_pixels(self):
        # Get current extent of the map in degrees and pixels
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = self.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # Convert pixels to degrees
        radius_degrees = max(lon_deg_per_pixel, lat_deg_per_pixel) * DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS

        return radius_degrees

    # calculate annotation offset of pixels in degrees (of where to place the annotation next to the circle)
    def calculate_offset_pixels(self):
        # Get current extent of the map in degrees and pixels
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = self.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # how far outside the annotated circle to place the annotation
        offset_pixels = DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS + 2

        # Convert pixels to degrees
        lon_offset = max(lon_deg_per_pixel, lat_deg_per_pixel) * (offset_pixels + 2)
        lat_offset = max(lon_deg_per_pixel, lat_deg_per_pixel) * (offset_pixels + 2)

        return lon_offset, lat_offset

    def add(self, lat=None, lon=None, label=None, label_color=DEFAULT_ANNOTATE_TEXT_COLOR):
        if lat is None or lon is None or label is None or self.ax is None:
            return
        if self.circle_handles is None:
            self.circle_handles = []
        #if self.annotation_handles is None:
        #    self.annotation_handles = []
        if self.ax.draggable_annotations is None:
            self.ax.draggable_annotations = []

        if self.has_overlap(lat=lat, lon=lon):
            return

        lon_offset, lat_offset = self.calculate_offset_pixels()
        # calculate radius of pixels in degrees
        radius_pixels_degrees = self.calculate_radius_pixels()
        circle_handle = Circle((lon, lat), radius=radius_pixels_degrees, color=DEFAULT_ANNOTATE_MARKER_COLOR, fill=False, linestyle='dotted', linewidth=2, alpha=0.8,
                                    transform=ccrs.PlateCarree())
        self.ax.add_patch(circle_handle)
        self.rtree_idx.insert(len(self.rtree_idx), (lon, lat, lon, lat))
        self.circle_handles.append(circle_handle)

        bbox_props = {
            'boxstyle': 'round,pad=0.3',
            'edgecolor': '#FFFFFF',
            'facecolor': '#000000',
            'alpha': 1.0
        }

        # Original annotation creation with DraggableAnnotation integration
        annotation_handle = self.ax.annotate(label, xy=(lon, lat), xytext=(lon + lon_offset, lat + lat_offset), textcoords='data', color=label_color,
                            fontsize=12, ha='left', va='bottom', bbox=bbox_props)

        # Create DraggableAnnotation instance
        draggable_annotation = DraggableAnnotation(annotation_handle, (lon, lat), self.ax, bbox_props)

        # Store the DraggableAnnotation instance
        #self.annotations.append(draggable_annotation)
        self.ax.draggable_annotations.append(draggable_annotation)

        # draw later as we will likely add multiple circles
        #self.canvas.draw()

    def has_overlap(self, lat=None, lon=None):
        if lat is None or lon is None or len(self.rtree_idx) == 0:
            return False
        # Define a bounding box around the annotated circle for initial query (in degrees)
        buffer = ANNOTATE_CIRCLE_OVERLAP_IN_DEGREES  # Adjust this value based on desired precision
        bounding_box = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        # Query the R-tree for points within the bounding box
        possible_matches = list(self.rtree_idx.intersection(bounding_box, objects=True))

        if possible_matches:
            return True
        else:
            return False

    def clear(self):
        cleared = False
        #if self.annotation_handles:
        if self.ax.draggable_annotations:
            cleared = True
            try:
                #for annotation_handle in self.annotation_handles:
                #    annotation_handle.remove()
                for annotation in self.ax.draggable_annotations:
                    annotation.remove()
            except:
                traceback.print_exc()
                pass
            #self.annotation_handles = None
            self.ax.draggable_annotations = None
        if self.circle_handles:
            cleared = True
            try:
                for circle_handle in self.circle_handles:
                    circle_handle.remove()
            except:
                traceback.print_exc()
                pass
            self.circle_handles = None
        if self.rtree_p:
            cleared = True
            self.rtree_p = index.Property()
        if self.rtree_idx:
            cleared = True
            self.rtree_idx = index.Index(properties=self.rtree_p)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cartopy and Tkinter App")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")

        self.mode = "ADECK"
        self.recent_storms = None
        self.adeck = None
        self.bdeck = None
        self.adeck_selected = tk.StringVar()
        self.adeck_previous_selected = None
        self.adeck_storm = None

        self.genesis_model_cycle_time = None
        self.zoom_rect = None
        self.rect_patch = None
        self.last_cursor_lon_lat = (0.0, 0.0)

        self.lastgl = None

        self.have_deck_data = False
        # track whether there is new tcvitals,adecks,bdecks data
        self.timer_id = None
        self.stale_urls = {}
        self.stale_urls['tcvitals'] = set()
        self.stale_urls['adeck'] = set()
        self.stale_urls['bdeck'] = set()
        # keys are the urls we will check for if it is stale, values are datetime objects
        self.dt_mods_tcvitals = {}
        self.dt_mods_adeck = {}
        self.dt_mods_bdeck = {}

        self.measure_mode = False
        self.start_point = None
        self.end_point = None
        self.line = None
        self.distance_text = None

        self.cid_press = None
        self.cid_release = None
        self.cid_motion = None
        self.cid_key_press = None
        self.cid_key_release = None

        # keep list of all lat_lon_with_time_step_list (plural) used to create points in last drawn map
        #   note, each item is also a list of dicts (not enumerated)
        self.plotted_tc_candidates = []
        # r-tree index
        self.rtree_p = index.Property()
        self.rtree_idx = index.Index(properties=self.rtree_p)
        # Mapping from rtree point index to (tc_index, tc_candidate_point_index)
        self.rtree_tuple_point_id = 0
        self.rtree_tuple_index_mapping = {}

        # circle patch for selected marker
        self.circle_handle = None
        self.last_circle_lon = None
        self.last_circle_lat = None

        # track overlapped points (by index, pointing to the plotted_tc_candidates)
        #   this will hold information on the marker where the cursor previously pointed to (current circle patch),
        #   and which one of the possible matches was (is currently) viewed
        self.nearest_point_indices_overlapped = SortedCyclicEnumDict()

        # annotated circles (for storm extrema)
        self.annotated_circles = None

        self.create_widgets()
        self.display_map()

    # calculate radius of pixels in degrees
    def calculate_radius_pixels(self):
        # Get current extent of the map in degrees and pixels
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = self.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # Convert pixels to degrees
        radius_degrees = max(lon_deg_per_pixel, lat_deg_per_pixel) * DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS

        return radius_degrees

    def clear_plotted_list(self):
        self.plotted_tc_candidates = []
        self.rtree_p = index.Property()
        self.rtree_idx = index.Index(properties=self.rtree_p)
        self.rtree_tuple_point_id = 0
        self.rtree_tuple_index_mapping = {}
        # reset all labels
        self.update_tc_status_labels()
        self.clear_circle_patch()

    def update_plotted_list(self, tc_candidate):
        # zero indexed
        tc_index = len(self.plotted_tc_candidates)
        for point_index, point in enumerate(tc_candidate):  # Iterate over each point in the track
            lat, lon = point['lat'], point['lon']
            # Can't use a tuple (tc_index, point_index) as the index so use a mapped index
            self.rtree_idx.insert(self.rtree_tuple_point_id, (lon, lat, lon, lat))
            self.rtree_tuple_index_mapping[self.rtree_tuple_point_id] = (tc_index, point_index)
            self.rtree_tuple_point_id += 1

        self.plotted_tc_candidates.append(tc_candidate)

    def update_tc_status_labels(self, tc_index = None, tc_point_index = None, overlapped_point_num = 0, total_num_overlapped_points = 0):
        # may not have init interface yet
        try:
            if tc_index is None or tc_point_index is None or len(self.plotted_tc_candidates) == 0:
                self.label_mouse_hover_matches.config(text="0  ", style="FixedWidthWhite.TLabel")
                self.label_mouse_hover_info_coords.config(text="(-tt.tttt, -nnn.nnnn)")
                self.label_mouse_hover_info_valid_time.config(text="YYYY-MM-DD hhZ")
                self.label_mouse_hover_info_model_init.config(text="YYYY-MM-DD hhZ")
                self.label_mouse_hover_info_vmax10m.config(text="---.- kt")
                self.label_mouse_hover_info_mslp.config(text="----.- hPa")
                self.label_mouse_hover_info_roci.config(text="---- km")
                self.label_mouse_hover_info_isobar_delta.config(text="--- hPa")
            else:
                # list of dicts (points in time) for tc candidate
                tc_candidate = self.plotted_tc_candidates[tc_index]
                # dict at a point in time for tc candidate
                tc_candidate_point = tc_candidate[tc_point_index]
                model_name = tc_candidate_point['model_name']
                lat = tc_candidate_point['lat']
                lon = tc_candidate_point['lon']
                valid_time = tc_candidate_point['valid_time'].strftime('%Y-%m-%d %HZ')
                init_time = tc_candidate_point['init_time'].strftime('%Y-%m-%d %HZ')
                if tc_candidate_point['vmax10m_in_roci']:
                    vmax10m = tc_candidate_point['vmax10m_in_roci']
                else:
                    vmax10m = tc_candidate_point['vmax10m']
                mslp = tc_candidate_point['mslp_value']
                roci = tc_candidate_point['roci']
                isobar_delta = tc_candidate_point['closed_isobar_delta']

                self.label_mouse_hover_matches.config(text=f"{overlapped_point_num}/{total_num_overlapped_points}")
                if total_num_overlapped_points > 1:
                    self.label_mouse_hover_matches.config(style="FixedWidthRed.TLabel")
                else:
                    self.label_mouse_hover_matches.config(style="FixedWidthWhite.TLabel")
                self.label_mouse_hover_info_coords.config(text=f"({lat:>8.4f}, {lon:>9.4f})")
                if valid_time:
                    self.label_mouse_hover_info_valid_time.config(text=valid_time)
                else:
                    self.label_mouse_hover_info_valid_time.config(text="YYYY-MM-DD hhZ")
                if init_time:
                    self.label_mouse_hover_info_model_init.config(text=f"{model_name:>4} {init_time}")
                else:
                    self.label_mouse_hover_info_model_init.config(text="YYYY-MM-DD hhZ")
                if vmax10m:
                    self.label_mouse_hover_info_vmax10m.config(text=f"{vmax10m:>5.1f} kt")
                else:
                    self.label_mouse_hover_info_vmax10m.config(text="---.- kt")
                if mslp:
                    self.label_mouse_hover_info_mslp.config(text=f"{mslp:>6.1f} hPa")
                else:
                    self.label_mouse_hover_info_mslp.config(text="----.- hPa")
                if roci:
                    self.label_mouse_hover_info_roci.config(text=f"{roci:>4.0f} km")
                else:
                    self.label_mouse_hover_info_roci.config(text="---- km")
                if isobar_delta:
                    self.label_mouse_hover_info_isobar_delta.config(text=f"{isobar_delta:>3.0f} hPa")
                else:
                    self.label_mouse_hover_info_isobar_delta.config(text="--- hPa")
        except:
            traceback.print_exc()
            pass

    def update_labels_for_mouse_hover(self, lat=None, lon=None):
        if not(lat) or not(lon):
            return

        # Update label for mouse cursor position on map first
        self.label_mouse_coords.config(text=f"({lat:>8.4f}, {lon:>9.4f})")

        # Next, find nearest point (within some bounding box, as we want to be selective)
        # Define a bounding box around the cursor for initial query (in degrees)
        buffer = MOUSE_SELECT_IN_DEGREES  # Adjust this value based on desired precision
        bounding_box = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        # Query the R-tree for points within the bounding box
        possible_matches = list(self.rtree_idx.intersection(bounding_box, objects=True))

        # Calculate the geodesic distance and find the nearest point
        nearest_point_index = None
        min_distance = float('inf')
        # a sorted cyclic dict that has the item number enumerated
        # has a get() the next enumerated number and the key (the point_index tuple) in a cycle (sorted by value, which will be a datetime)
        self.nearest_point_indices_overlapped = SortedCyclicEnumDict()
        for item in possible_matches:
            unmapped_point_index = item.id
            tc_index, point_index = self.rtree_tuple_index_mapping[unmapped_point_index]
            point = self.plotted_tc_candidates[tc_index][point_index]
            item_is_overlapping = False
            if len(self.nearest_point_indices_overlapped):
                overlapping_tc_index, overlapping_point_index = self.nearest_point_indices_overlapped.get_first_key()
                possible_overlapping_point = self.plotted_tc_candidates[overlapping_tc_index][overlapping_point_index]
                lon_diff = round(abs(possible_overlapping_point['lon'] - point['lon']), 3)
                lat_diff = round(abs(possible_overlapping_point['lat'] - point['lat']), 3)
                if lon_diff == 0.0 and lat_diff == 0.0:
                    item_is_overlapping = True

            # check to see if it is an almost exact match (~3 decimals in degrees) to approximate whether it is an overlapped point
            if item_is_overlapping:
                # this will likely be an overlapped point in the grid
                self.nearest_point_indices_overlapped[(tc_index, point_index)] = point['valid_time']
                # min distance should not significantly change (we are using the first point as reference for overlapping)
            else:
                distance = self.calculate_distance((lon, lat), (point['lon'], point['lat']))
                if distance < min_distance:
                    # not an overlapping point but still closer to cursor, so update
                    # first clear any other points since this candidate is closer and does not have an overlapping point
                    self.nearest_point_indices_overlapped = SortedCyclicEnumDict()
                    self.nearest_point_indices_overlapped[(tc_index, point_index)] = point['valid_time']
                    min_distance = distance

        # Update the labels if a nearest point is found within the threshold
        total_num_overlapped_points = len(self.nearest_point_indices_overlapped)
        if total_num_overlapped_points:
            overlapped_point_num, nearest_point_index = self.nearest_point_indices_overlapped.next_enum_key_tuple()
            tc_index, point_index = nearest_point_index
            self.update_tc_status_labels(tc_index,point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = self.plotted_tc_candidates[tc_index][point_index]
            lon = point['lon']
            lat = point['lat']
            self.update_circle_patch(lon=lon, lat=lat)
        else:
            # clear the label if no point is found? No.
            #   Not only will this prevent the constant reconfiguring of labels, it allows the user more flexibility
            #self.update_tc_status_labels()

            # Do clear the circle though as it might be obtrusive
            self.clear_circle_patch()

    def cycle_to_next_overlapped_point(self):
        # called when user hovers on overlapped points and hits the TAB key
        total_num_overlapped_points = len(self.nearest_point_indices_overlapped)
        if total_num_overlapped_points > 1:
            overlapped_point_num, nearest_point_index = self.nearest_point_indices_overlapped.next_enum_key_tuple()
            tc_index, point_index = nearest_point_index
            self.update_tc_status_labels(tc_index,point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = self.plotted_tc_candidates[tc_index][point_index]
            lon = point['lon']
            lat = point['lat']
            self.update_circle_patch(lon=lon, lat=lat)

    def is_extremum(self, extremum_key=None, current_extremum=None, potential=None):
        if extremum_key is None:
            # error
            return False
        if current_extremum is None and potential is None:
            return False
        if potential is None:
            return False
        if current_extremum is None:
            return True
        if extremum_key in ['vmax10m', 'roci', 'closed_isobar_delta', 'dt_end']:
            return potential > current_extremum
        elif extremum_key in ['mslp_value', 'dt_start']:
            return potential < current_extremum
        else:
            # unhandled case
            return False

    # defines how the annotated circle text is formatted
    def format_extremum_label_lines(self, append = False, model_name = '', valid_time = None, extremum_key = None, extremum_val = None):
        if valid_time is None or extremum_key is None or extremum_val is None or not(extremum_key in displayed_extremum_annotations):
            return None, 0
        prefix_line_str = None
        if append == False:
            # For first line of annotation of extremum, include Model name and Valid Time
            model_name_prefix_str = ""
            if model_name:
                model_name_prefix_str = f"{model_name} "

            valid_time_str = valid_time.strftime('%Y-%m-%d %HZ')
            prefix_line_str = f"{model_name_prefix_str}{valid_time_str}\n"

        extremum_label_line = None
        color_level = 0
        if extremum_key == "dt_start":
            extremum_label_line = "START"
            color_level = 1
        elif extremum_key == "dt_end":
            extremum_label_line = "END"
        elif extremum_key == "vmax10m":
            color_level = 2
            extremum_label_line = f"VMAX_10m: {extremum_val:.1f} kt"
        elif extremum_key == "mslp_value":
            extremum_label_line = f"MSLP: {extremum_val:.1f} hPa"
        elif extremum_key == "roci":
            extremum_label_line = f"ROCI: {extremum_val:.0f} km"
        elif extremum_key == "closed_isobar_delta":
            extremum_label_line = f"ISOBAR_DELTA: {extremum_val:.0f} hPa"

        extremum_label_lines = None
        if extremum_label_line:
            if prefix_line_str:
                extremum_label_lines = f"{prefix_line_str}{extremum_label_line}"
            else:
                extremum_label_lines = f"{extremum_label_line}"

        return extremum_label_lines, color_level

    def clear_storm_extrema_annotations(self):
        if self.annotated_circles:
            self.annotated_circles.clear()

    def any_storm_points_in_bounds(self, tc_index):
        if not self.plotted_tc_candidates:
            return False
        if tc_index > len(self.plotted_tc_candidates):
            return False
        if not self.ax:
            return False

        any_in_bound = False

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        try:
            for point in self.plotted_tc_candidates[tc_index]:
                lat = point['lat']
                lon = point['lon']
                any_in_bound = any_in_bound or (xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1])
        except:
            pass

        return any_in_bound

    def annotate_storm_extrema(self):
        if len(self.plotted_tc_candidates) == 0:
            return

        # note: point_index is a tuple of tc_index, tc_point_index
        if len(self.nearest_point_indices_overlapped) == 0:
            # annotate all storm extrema in current view
            for tc_index in range(len(self.plotted_tc_candidates)):
                if len(self.plotted_tc_candidates[tc_index]):
                    if self.any_storm_points_in_bounds(tc_index):
                        point_index = (tc_index, 0)
                        self.annotate_single_storm_extrema(point_index=point_index)
        else:
            # annotate storm extrema of previously selected
            num, cursor_point_index = self.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            self.annotate_single_storm_extrema(point_index=cursor_point_index)

    def annotate_single_storm_extrema(self, point_index = None):
        tc_index, tc_point_index = point_index

        # tuple of (storm) extremum (along whole path) and corresponding tc_point_index
        #   this is the order the extrema will appear if they share the same tc_point_index
        extrema = {
            'dt_start': (None, None),
            'dt_end': (None, None),
            'vmax10m': (None, None),
            'mslp_value': (None, None),
            'roci': (None, None),
            'closed_isobar_delta': (None, None),
        }
        for tc_point_index, point in enumerate(self.plotted_tc_candidates[tc_index]):
            for extremum_key, extremum_val_tuple in extrema.items():
                extremum_val, extremum_index = extremum_val_tuple
                point_keys = point.keys()
                point_extremum_val = None
                if extremum_key == 'vmax10m':
                    if 'vmax10m_in_roci' in point_keys and point['vmax10m_in_roci']:
                        point_extremum_val = point['vmax10m_in_roci']
                    else:
                        if extremum_key in point_keys:
                            point_extremum_val = point[extremum_key]
                else:
                    if extremum_key in ['dt_start', 'dt_end']:
                        if 'valid_time' in point_keys:
                            point_extremum_val = point['valid_time']
                    elif extremum_key in point_keys:
                        point_extremum_val = point[extremum_key]
                is_extremum = self.is_extremum(extremum_key=extremum_key, current_extremum=extremum_val, potential=point_extremum_val)
                if is_extremum:
                    # replace previous extremum with the point at tc_point_index
                    extrema[extremum_key] = (point_extremum_val, tc_point_index)

        # annotate the extrema for the storm
        point_index_labels = {}
        # since some extremum may show up for the same point, we need to combine the extremum labels first (by point_index)
        prev_color_level = 0
        for extremum_key, extremum_val_tuple in extrema.items():
            extremum_val, extremum_index = extremum_val_tuple
            if extremum_index is None or extremum_val is None:
                continue
            extremum_point = self.plotted_tc_candidates[tc_index][extremum_index]
            append = False
            if extremum_index in point_index_labels:
                append = True

            extremum_point_keys = extremum_point.keys()
            model_name = None
            valid_time = None
            if 'model_name' in extremum_point_keys:
                model_name = extremum_point['model_name']
            if 'valid_time' in extremum_point_keys:
                valid_time = extremum_point['valid_time']
            extremum_lines, new_color_level = self.format_extremum_label_lines(append=append, model_name=model_name, valid_time=valid_time, extremum_key=extremum_key, extremum_val=extremum_val)
            if extremum_lines:
                if append:
                    prev_lines, prev_color_level = point_index_labels[extremum_index]
                    if new_color_level > prev_color_level:
                        lines_color_level = new_color_level
                    else:
                        lines_color_level = prev_color_level
                    new_lines = f"{prev_lines}\n{extremum_lines}"
                    point_index_labels[extremum_index] = (new_lines, lines_color_level)
                else:
                    prev_color_level = new_color_level
                    point_index_labels[extremum_index] = (extremum_lines, prev_color_level)

        # finally add the annotated circle for each label
        added = False
        for label_point_index, (point_label, color_level) in point_index_labels.items():
            point = self.plotted_tc_candidates[tc_index][label_point_index]
            added = True
            self.annotated_circles.add(lat=point['lat'], lon=point['lon'], label=point_label, label_color=annotate_color_levels[color_level])
        if added:
            self.fig.canvas.draw()

    def display_custom_boundaries(self):
        if custom_gdf is not None:
            for geometry in custom_gdf.geometry:
                if isinstance(geometry, Polygon):
                    self.ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none', linewidth=2)
                else:
                    for polygon in geometry:
                        self.ax.add_geometries([polygon], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none', linewidth=2)

    def combo_selected_models_event(self, event):
        current_value = self.adeck_selected_combobox.get()
        if current_value == self.adeck_previous_selected:
            # user did not change selection
            return
        else:
            self.adeck_previous_selected = current_value
            self.display_map()
            if not self.have_deck_data:
                self.update_deck_data()
            self.display_deck_data()

    def create_widgets(self):
        self.top_frame = ttk.Frame(self.root, style="TopFrame.TFrame")
        self.top_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        # Middle frame
        self.tools_frame = ttk.Frame(self.root, style="ToolsFrame.TFrame")
        self.tools_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        self.create_adeck_mode_widgets()
        self.create_genesis_mode_widgets()
        self.create_tools_widgets()

        self.update_mode()

        self.canvas_frame = ttk.Frame(self.root, style="CanvasFrame.TFrame")
        self.canvas_frame.pack(fill=tk.X, expand=True)

        self.canvas = None
        self.fig = None

    def create_adeck_mode_widgets(self):
        self.adeck_mode_frame = ttk.Frame(self.top_frame, style="TopFrame.TFrame")

        self.exit_button_adeck = ttk.Button(self.adeck_mode_frame, text="EXIT", command=self.root.quit, style="TButton")
        self.exit_button_adeck.pack(side=tk.LEFT, padx=5, pady=5)

        self.reload_button_adeck = ttk.Button(self.adeck_mode_frame, text="(RE)LOAD", command=self.reload_adeck, style="White.TButton")
        self.reload_button_adeck.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_adeck_mode = ttk.Label(self.adeck_mode_frame, text="ADECK MODE. Models: 0", style="TLabel")
        self.label_adeck_mode.pack(side=tk.LEFT, padx=5, pady=5)

        style = ttk.Style()
        style.map('TCombobox', fieldbackground=[('readonly','black')])
        style.map('TCombobox', foreground=[('readonly','white')])
        style.map('TCombobox', selectbackground=[('readonly', 'black')])
        style.map('TCombobox', selectforeground=[('readonly', 'white')])
        self.adeck_selected_combobox = ttk.Combobox(self.adeck_mode_frame, width = 14, textvariable = self.adeck_selected, state='readonly')
        self.adeck_selected_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.adeck_selected_combobox['state'] = 'readonly' # Set the state according to configure colors
        self.adeck_selected_combobox['values'] = ('ALL', 'STATISTICAL', 'GLOBAL', 'GEFS-MEMBERS', 'REGIONAL', 'CONSENSUS', 'OFFICIAL')
        self.adeck_selected_combobox.current(6)
        self.adeck_previous_selected = self.adeck_selected.get()

        self.adeck_selected_combobox.bind("<<ComboboxSelected>>", self.combo_selected_models_event)

        self.switch_to_genesis_button = ttk.Button(self.adeck_mode_frame, text="SWITCH TO GENESIS MODE", command=self.switch_mode, style="TButton")
        self.switch_to_genesis_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_genesis_mode_widgets(self):
        self.genesis_mode_frame = ttk.Frame(self.top_frame, style="TopFrame.TFrame")

        self.exit_button_genesis = ttk.Button(self.genesis_mode_frame, text="EXIT", command=self.root.quit, style="TButton")
        self.exit_button_genesis.pack(side=tk.LEFT, padx=5, pady=5)

        self.reload_button_genesis = ttk.Button(self.genesis_mode_frame, text="(RE)LOAD", command=self.reload, style="TButton")
        self.reload_button_genesis.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_genesis_mode = ttk.Label(self.genesis_mode_frame, text="GENESIS MODE: Start valid day: YYYY-MM-DD", style="TLabel")
        self.label_genesis_mode.pack(side=tk.LEFT, padx=5, pady=5)

        self.prev_genesis_cycle_button = ttk.Button(self.genesis_mode_frame, text="PREV CYCLE", command=self.prev_genesis_cycle, style="TButton")
        self.prev_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.prev_genesis_cycle_button = ttk.Button(self.genesis_mode_frame, text="NEXT CYCLE", command=self.next_genesis_cycle, style="TButton")
        self.prev_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.latest_genesis_cycle_button = ttk.Button(self.genesis_mode_frame, text="LATEST CYCLE", command=self.latest_genesis_cycle, style="TButton")
        self.latest_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.genesis_models_label = ttk.Label(self.genesis_mode_frame, text="Models: GFS [--/--Z], ECM[--/--Z], NAV[--/--Z], CMC[--/--Z]", style="TLabel")
        self.genesis_models_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.switch_to_adeck_button = ttk.Button(self.genesis_mode_frame, text="SWITCH TO ADECK MODE", command=self.switch_mode, style="TButton")
        self.switch_to_adeck_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_tools_widgets(self):
        #self.tools_frame = ttk.Frame(self.tools_frame, style="Tools.TFrame")

        self.add_marker_button = ttk.Button(self.tools_frame, text="ADD MARKER", command=self.add_marker, style="TButton")
        self.add_marker_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_coords_prefix = ttk.Label(self.tools_frame, text="Cursor position:", style="TLabel")
        self.label_mouse_coords_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_coords = ttk.Label(self.tools_frame, text="(-tt.tttt, -nnn.nnnn)", style="FixedWidthWhite.TLabel")
        self.label_mouse_coords.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_prefix = ttk.Label(self.tools_frame, text="(Hover) Matches:", style="TLabel")
        self.label_mouse_hover_info_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_matches = ttk.Label(self.tools_frame, text="0  ", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_matches.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_coords = ttk.Label(self.tools_frame, text="(-tt.tttt, -nnn.nnnn)", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_coords.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_valid_time_prefix = ttk.Label(self.tools_frame, text="Valid time:", style="TLabel")
        self.label_mouse_hover_info_valid_time_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_valid_time = ttk.Label(self.tools_frame, text="YYYY-MM-DD hhZ", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_valid_time.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_model_init_prefix = ttk.Label(self.tools_frame, text="Model init:", style="TLabel")
        self.label_mouse_hover_info_model_init_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_model_init = ttk.Label(self.tools_frame, text="YYYY-MM-DD hhZ", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_model_init.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_vmax10m_prefix = ttk.Label(self.tools_frame, text="Vmax @ 10m:", style="TLabel")
        self.label_mouse_hover_info_vmax10m_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_vmax10m = ttk.Label(self.tools_frame, text="---.- kt", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_vmax10m.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_mslp_prefix = ttk.Label(self.tools_frame, text="MSLP:", style="TLabel")
        self.label_mouse_hover_info_mslp_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_mslp = ttk.Label(self.tools_frame, text="----.- hPa", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_mslp.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_roci_prefix = ttk.Label(self.tools_frame, text="ROCI:", style="TLabel")
        self.label_mouse_hover_info_roci_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_roci = ttk.Label(self.tools_frame, text="---- km", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_roci.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_isobar_delta_prefix = ttk.Label(self.tools_frame, text="Isobar delta:", style="TLabel")
        self.label_mouse_hover_info_isobar_delta_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_mouse_hover_info_isobar_delta = ttk.Label(self.tools_frame, text="--- hPa", style="FixedWidthWhite.TLabel")
        self.label_mouse_hover_info_isobar_delta.pack(side=tk.LEFT, padx=5, pady=5)

    def add_marker(self):
        pass

    def switch_mode(self):
        self.mode = "GENESIS" if self.mode == "ADECK" else "ADECK"
        self.update_mode()

    def update_mode(self):
        if self.mode == "ADECK":
            self.genesis_mode_frame.pack_forget()
            self.adeck_mode_frame.pack(side=tk.TOP, fill=tk.X)
        else:
            self.adeck_mode_frame.pack_forget()
            self.genesis_mode_frame.pack(side=tk.TOP, fill=tk.X)

    def update_reload_button_color(self):
        if self.stale_urls['adeck']:
            self.reload_button_adeck.configure(style='Red.TButton')
        elif self.stale_urls['bdeck']:
            self.reload_button_adeck.configure(style='Orange.TButton')
        elif self.stale_urls['tcvitals']:
            self.reload_button_adeck.configure(style='Yellow.TButton')
        else:
            self.reload_button_adeck.configure(style='White.TButton')

    def update_deck_data(self):
        # track which data is stale (tc vitals, adeck, bdeck)
        # unfortunately we need to get all of one type (vitals, adeck, bdeck) from both mirrors since it's unknown which mirror actually has most up-to-date data from the modification date alone
        updated_urls_tcvitals = set()
        updated_urls_adeck = set()
        updated_urls_bdeck = set()

        # logic for updating classes
        do_update_tcvitals = do_update_adeck = do_update_bdeck = False
        if not self.have_deck_data:
            # first fetch of data
            do_update_tcvitals = do_update_adeck = do_update_bdeck = True
            if self.timer_id is not None:
                self.root.after_cancel(self.timer_id)
            self.timer_id = self.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, self.check_for_stale_data)
        else:
            # refresh status of stale data one more time since the user has requested a reload
            self.check_for_stale_data()
            if self.dt_mods_tcvitals and self.stale_urls['tcvitals']:
                do_update_tcvitals = True
            if self.dt_mods_adeck and self.stale_urls['adeck']:
                do_update_adeck = True
            if self.dt_mods_bdeck and self.stale_urls['bdeck']:
                do_update_bdeck = True

        # Get recent storms (tcvitals)
        if do_update_tcvitals:
            new_dt_mods_tcvitals, new_recent_storms = get_recent_storms(tcvitals_urls)
            if new_dt_mods_tcvitals:
                old_dt_mods = copy.deepcopy(self.dt_mods_tcvitals)
                self.dt_mods_tcvitals.update(new_dt_mods_tcvitals)
                updated_urls_tcvitals = diff_dicts(old_dt_mods, self.dt_mods_tcvitals)
                if updated_urls_tcvitals:
                    self.recent_storms = new_recent_storms
        # Get A-Deck and B-Deck files
        if do_update_adeck or do_update_bdeck:
            new_dt_mods_adeck, new_dt_mods_bdeck, new_adeck, new_bdeck = get_deck_files(self.recent_storms, adeck_urls, bdeck_urls, do_update_adeck, do_update_bdeck)
            if new_dt_mods_adeck and do_update_adeck:
                old_dt_mods = copy.deepcopy(self.dt_mods_adeck)
                self.dt_mods_adeck.update(new_dt_mods_adeck)
                updated_urls_adeck = diff_dicts(old_dt_mods, self.dt_mods_adeck)
                if updated_urls_adeck:
                    self.adeck = new_adeck
            if new_dt_mods_bdeck and do_update_bdeck:
                old_dt_mods = copy.deepcopy(self.dt_mods_bdeck)
                self.dt_mods_bdeck.update(new_dt_mods_bdeck)
                updated_urls_bdeck = diff_dicts(old_dt_mods, self.dt_mods_bdeck)
                if updated_urls_bdeck:
                    self.bdeck = new_bdeck

        if self.dt_mods_tcvitals or self.dt_mods_adeck or self.dt_mods_bdeck:
            # at least something was downloaded
            self.have_deck_data = True

        self.stale_urls['tcvitals'] = self.stale_urls['tcvitals'] - set(updated_urls_tcvitals)
        self.stale_urls['adeck'] = self.stale_urls['adeck'] - set(updated_urls_adeck)
        self.stale_urls['bdeck'] = self.stale_urls['bdeck'] - set(updated_urls_bdeck)
        self.update_reload_button_color()

    def check_for_stale_data(self):
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, self.check_for_stale_data)
        if self.dt_mods_tcvitals:
            for url, old_dt_mod in self.dt_mods_tcvitals.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        self.stale_urls['tcvitals'] = self.stale_urls['tcvitals'] | set([url])
        if self.dt_mods_adeck:
            for url, old_dt_mod in self.dt_mods_adeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        self.stale_urls['adeck'] = self.stale_urls['adeck'] | set([url])
        if self.dt_mods_bdeck:
            for url, old_dt_mod in self.dt_mods_bdeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        self.stale_urls['bdeck'] = self.stale_urls['bdeck'] | set([url])

        self.update_reload_button_color()

    def reload_adeck(self):
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, self.check_for_stale_data)

        self.reload()

    def reload(self):
        if self.mode == "ADECK":
            self.display_map()
            self.update_deck_data()
            self.display_deck_data()
        elif self.mode == "GENESIS":
            self.display_map()
            model_cycle = self.genesis_model_cycle_time
            if model_cycle is None:
                model_cycles = get_tc_model_init_times_relative_to(datetime.now())
                if model_cycles['next'] is None:
                    model_cycle = model_cycles['at']
                else:
                    model_cycle = model_cycles['next']
            if model_cycle:
                # clear map
                self.display_map()
                self.update_genesis(model_cycle)

    def adeck(self):
        pass  # Placeholder for adeck function

    def get_selected_model_list(self):
        selected_text = self.adeck_selected.get()
        if selected_text == 'ALL':
            return included_intensity_models
        elif selected_text == 'GLOBAL':
            return global_models
        elif selected_text == 'GEFS-MEMBERS':
            return gefs_members_models
        elif selected_text == 'STATISTICAL':
            return statistical_models
        elif selected_text == 'REGIONAL':
            return regional_models
        elif selected_text == 'CONSENSUS':
            return consensus_models
        elif selected_text == 'OFFICIAL':
            return official_models
        else:
            # sanity check
            return official_models

    # get model data from adeck, plus bdeck and tcvitals
    def get_selected_model_candidates_from_decks(self):
        selected_models = self.get_selected_model_list()
        valid_datetime = datetime.max
        earliest_model_valid_datetime = valid_datetime
        selected_model_data = {}
        actual_models = set()
        all_models = set()

        for storm_atcf_id in self.adeck.keys():
            for model_id, models in self.adeck[storm_atcf_id].items():
                for valid_time, data in models.items():
                    dt = datetime.fromisoformat(valid_time)
                    if dt < earliest_model_valid_datetime:
                        earliest_model_valid_datetime = dt
                all_models.add(model_id)
                if model_id in selected_models:
                    if storm_atcf_id not in selected_model_data.keys():
                        selected_model_data[storm_atcf_id] = {}
                    selected_model_data[storm_atcf_id][model_id] = models
                    actual_models.add(model_id)

        for storm_atcf_id in self.bdeck.keys():
            for model_id, models in self.bdeck[storm_atcf_id].items():
                if model_id in selected_models:
                    if storm_atcf_id not in selected_model_data.keys():
                        selected_model_data[storm_atcf_id] = {}
                    selected_model_data[storm_atcf_id][model_id] = models

        # tcvitals
        for storm_atcf_id, data in self.recent_storms.items():
            valid_date_str = data['valid_time']
            if storm_atcf_id not in selected_model_data.keys():
                selected_model_data[storm_atcf_id] = {}
            selected_model_data[storm_atcf_id]['TCVITALS'] = {valid_date_str: data}
            dt = datetime.fromisoformat(valid_date_str)
            if dt < earliest_model_valid_datetime:
                earliest_model_valid_datetime = dt

        return earliest_model_valid_datetime, len(all_models), len(actual_models), selected_model_data

    def display_deck_data(self):
        vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'), (float('inf'), '*')]
        vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2606 5']
        marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

        valid_datetime, num_all_models, num_models, tc_candidates = self.get_selected_model_candidates_from_decks()
        start_of_day = valid_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_day = start_of_day.isoformat()

        most_recent_timestamp = None
        numc = 0

        colors = [
            '#ffff00',
            '#ba0a0a', '#e45913', '#fb886e', '#fdd0a2',
            '#005b1c', '#07a10b', '#9cd648', '#a5ee96',
            '#0d3860', '#2155c4', '#33aaff', '#7acaff',
            '#710173', '#b82cae', '#c171cf', '#ffb9ee',
            '#636363', '#969696', '#bfbfbf', '#e9e9e9'
        ]
        label_fg_colors = [
            '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#000000', '#000000', '#000000'
        ]

        # Define 6 different time_step ranges and their corresponding colors
        time_step_ranges = [
            (float('-inf'),-1),
            (0, 23),
            (24,47),
            (48,71),
            (72,95),
            (96,119),
            (120,143),
            (144,167),
            (168,191),
            (192,215),
            (216,239),
            (240,263),
            (264,287),
            (288,311),
            (312,335),
            (336,359),
            (360,383),
            (384,407),
            (408,431),
            (432,455),
            (456, float('inf'))
        ]

        self.clear_plotted_list()

        for storm_atcf_id, tc in tc_candidates.items():
            numc += 1
            for model_name, disturbance_candidates in tc.items():
                if disturbance_candidates:
                    prev_lat = None
                    prev_lon = None
                    prev_lon_repeat2 = None
                    prev_lon_repeat3 = None
                    numdisturb = len(disturbance_candidates)
                    num = 0

                    lat_lon_with_time_step_list = []
                    for valid_time, candidate in disturbance_candidates.items():
                        num += 1
                        if 'time_step' in candidate.keys():
                            time_step_int = candidate['time_step']
                        else:
                            time_step_int = 0
                        lon = candidate['lon']
                        lat = candidate['lat']
                        candidate_info = {}
                        candidate_info['model_name'] = model_name
                        if 'init_time' in candidate.keys() and candidate['init_time']:
                            candidate_info['init_time'] = datetime.fromisoformat(candidate['init_time'])
                        candidate_info['lat'] = lat
                        candidate_info['lon'] = lon
                        candidate_info['lon_repeat'] = candidate_info['lon']
                        candidate_info['valid_time'] = datetime.fromisoformat(valid_time)
                        candidate_info['time_step'] = time_step_int

                        # calculate the difference in hours
                        hours_diff = (candidate_info['valid_time'] - datetime.fromisoformat(valid_day)).total_seconds() / 3600
                        # round to the nearest hour
                        hours_diff_rounded = round(hours_diff)
                        candidate_info['hours_after_valid_day'] = hours_diff_rounded

                        if 'roci' in candidate and candidate['roci'] and float(candidate['roci']):
                            candidate_info['roci'] = float(candidate['roci'])/1000.0
                        else:
                            candidate_info['roci'] = None
                        if 'vmax10m' in candidate and candidate['vmax10m']:
                            candidate_info['vmax10m_in_roci'] = candidate['vmax10m']
                        else:
                            candidate_info['vmax10m_in_roci'] = None
                        candidate_info['closed_isobar_delta'] = None
                        if 'mslp' in candidate and candidate['mslp']:
                            candidate_info['mslp_value'] = candidate['mslp']
                        else:
                            candidate_info['mslp_value'] = None

                        if prev_lon:
                            prev_lon_f = float(prev_lon)
                            if abs(prev_lon_f - lon) > 270:
                                if prev_lon_f < lon:
                                    prev_lon = prev_lon_f + 360
                                else:
                                    prev_lon = prev_lon_f - 360

                        candidate_info['prev_lat'] = prev_lat
                        candidate_info['prev_lon'] = prev_lon
                        candidate_info['prev_lon_repeat'] = prev_lon

                        prev_lat = candidate_info['lat']
                        prev_lon = lon
                        prev_lon_repeat1 = lon

                        lat_lon_with_time_step_list.append(candidate_info)

                    self.update_plotted_list(lat_lon_with_time_step_list)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
                        opacity = 1.0
                        radius = 6
                        lons = {}
                        lats = {}
                        for point in reversed(lat_lon_with_time_step_list):
                            hours_after = point['hours_after_valid_day']
                            #if start <= time_step <= end:
                            # use hours after valid_day instead
                            if start <= hours_after <= end:
                                if point['vmax10m_in_roci']:
                                    for upper_bound, vmaxmarker in vmax_kt_threshold:
                                        marker = vmaxmarker
                                        if point['vmax10m_in_roci'] < upper_bound:
                                            break
                                    if marker not in lons:
                                        lons[marker] = []
                                        lats[marker] = []
                                    lons[marker].append(point['lon_repeat'])
                                    lats[marker].append(point['lat'])

                                #self.ax.plot([point['lon_repeat']], [point['lat']], marker=marker, color=colors[i], markersize=radius, alpha=opacity)

                        for vmaxmarker in lons.keys():
                            self.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker, facecolors='none', edgecolors=colors[i], s=marker_sizes[vmaxmarker]**2, alpha=opacity, antialiased=False)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
                        line_color = colors[i]
                        opacity = 1.0
                        strokewidth = 0.5

                        line_segments = []
                        for point in reversed(lat_lon_with_time_step_list):
                            hours_after = point['hours_after_valid_day']
                            if start <= hours_after <= end:
                                if point['prev_lon_repeat']:

                                    # Create a list of line segments
                                    line_segments.append([(point['prev_lon_repeat'], point['prev_lat']), (point['lon_repeat'], point['lat'])])
                                    """
                                    plt.plot([point['prev_lon_repeat'], point['lon_repeat']], [point['prev_lat'], point['lat']],
                                             color=color, linewidth=strokewidth, marker='', markersize = 0, alpha=opacity)
                                    """

                        # Create a LineCollection
                        lc = LineCollection(line_segments, color=colors[i], linewidth=strokewidth, alpha=opacity)
                        # Add the LineCollection to the axes
                        self.ax.add_collection(lc)

                        name = 'Tracks'

        labels_positive = [f' D+{str(i): >2} ' for i in range(len(colors)-1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        for i, (color, label) in enumerate(zip(reversed(colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i*20

            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color=list(reversed(label_fg_colors))[i],
                        fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor=color, alpha=1.0))

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i*35
            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                        fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor='#000000', alpha=1.0))
            #self.ax.scatter([x_pos + 200], [y_pos], marker=marker, edgecolors='#BBDDCC', facecolors='#446655', s=200.0, alpha=1.0, antialiased=False, transform=self.ax.transAxes)

        self.fig.canvas.draw()
        self.label_adeck_mode.config(text=f"ADECK MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime('%Y-%m-%d') + f". Models: {num_models}/{num_all_models}")

    def latest_genesis_cycle(self):
        model_cycles = get_tc_model_init_times_relative_to(datetime.now())
        if model_cycles['next'] is None:
            model_cycle = model_cycles['at']
        else:
            model_cycle = model_cycles['next']

        if model_cycle:
            # clear map
            self.display_map()
            self.update_genesis(model_cycle)

    def prev_genesis_cycle(self):
        if self.genesis_model_cycle_time is None:
            self.genesis_model_cycle_time = datetime.now()
            model_cycles = get_tc_model_init_times_relative_to(self.genesis_model_cycle_time)
            if model_cycles['previous'] is None:
                model_cycle = model_cycles['at']
            else:
                model_cycle = model_cycles['previous']
        else:
            model_cycles = get_tc_model_init_times_relative_to(self.genesis_model_cycle_time)
            if model_cycles['previous'] is None:
                # nothing before current cycle
                return
            model_cycle = model_cycles['previous']

        if model_cycle:
            if model_cycle != self.genesis_model_cycle_time:
                self.display_map()
                self.update_genesis(model_cycle)

    def next_genesis_cycle(self):
        if self.genesis_model_cycle_time is None:
            self.genesis_model_cycle_time = datetime.now()
            model_cycles = get_tc_model_init_times_relative_to(self.genesis_model_cycle_time)
            if model_cycles['next'] is None:
                model_cycle = model_cycles['at']
            else:
                model_cycle = model_cycles['next']
        else:
            model_cycles = get_tc_model_init_times_relative_to(self.genesis_model_cycle_time)
            if model_cycles['next'] is None:
                # nothing after current cycle
                return
            model_cycle = model_cycles['next']

        if model_cycle:
            if model_cycle != self.genesis_model_cycle_time:
                self.display_map()
                self.update_genesis(model_cycle)

    def update_genesis(self, model_cycle):
        #vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, 'p'), (113.0, 'o'), (137.0, 'D'), (float('inf'), '+')]
        vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'), (float('inf'), '*')]
        vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2605 5']
        marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}
        #disturbance_candidates = get_disturbances_from_db(model_name, model_timestamp)
        #now = datetime.utcnow()
        #start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        #valid_day = start_of_day.isoformat()
        #model_init_times, tc_candidates = get_tc_candidates_from_valid_time(now.isoformat())
        model_init_times, tc_candidates = get_tc_candidates_at_or_before_init_time(model_cycle)
        #if model_init_times != last_model_init_times:

        model_dates = {
            'GFS': None,
            'ECM': None,
            'NAV': None,
            'CMC': None
        }
        most_recent_model_timestamps = {
            'GFS': None,
            'ECM': None,
            'NAV': None,
            'CMC': None
        }
        for model_name, model_init_times in model_init_times.items():
            for model_timestamp_str in model_init_times:
                model_timestamp = datetime.fromisoformat(model_timestamp_str)
                if not most_recent_model_timestamps[model_name]:
                    most_recent_model_timestamps[model_name] = model_timestamp
                    model_dates[model_name] = model_timestamp.strftime('%d/%HZ')

                if most_recent_model_timestamps[model_name] < model_timestamp:
                    most_recent_model_timestamps[model_name] = model_timestamp
                    model_dates[model_name] = model_timestamp.strftime('%d/%HZ')

        # should match model cycle
        most_recent_model_cycle = max(most_recent_model_timestamps.values())
        oldest_model_cycle = min(most_recent_model_timestamps.values())
        start_of_day = oldest_model_cycle.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_day = start_of_day.isoformat()
        #model_init_times, tc_candidates = get_tc_candidates_from_valid_time(now.isoformat())
        model_init_times, tc_candidates = get_tc_candidates_at_or_before_init_time(most_recent_model_cycle)

        most_recent_timestamp = None
        numc = 0

        colors = [
            '#ffff00',
            '#ba0a0a', '#e45913', '#fb886e', '#fdd0a2',
            '#005b1c', '#07a10b', '#9cd648', '#a5ee96',
            '#0d3860', '#2155c4', '#33aaff', '#7acaff',
            '#710173', '#b82cae', '#c171cf', '#ffb9ee',
            '#636363', '#969696', '#bfbfbf', '#e9e9e9'
        ]
        label_fg_colors = [
            '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#000000', '#000000', '#000000'
        ]

        # Define 6 different time_step ranges and their corresponding colors
        time_step_ranges = [
            (float('-inf'),-1),
            (0, 23),
            (24,47),
            (48,71),
            (72,95),
            (96,119),
            (120,143),
            (144,167),
            (168,191),
            (192,215),
            (216,239),
            (240,263),
            (264,287),
            (288,311),
            (312,335),
            (336,359),
            (360,383),
            (384,407),
            (408,431),
            (432,455),
            (456, float('inf'))
        ]

        self.clear_plotted_list()

        for tc in tc_candidates:
            numc += 1
            numcandidates = len(tc_candidates)

            model_name = tc['model_name']
            model_timestamp = tc['model_timestamp']
            disturbance_candidates = tc['disturbance_candidates']
            if not most_recent_timestamp:
                most_recent_timestamp = model_timestamp
                model_dates[model_name] = datetime.fromisoformat(most_recent_timestamp).strftime('%d/%HZ')

            if datetime.fromisoformat(most_recent_timestamp) < datetime.fromisoformat(model_timestamp):
                most_recent_timestamp = model_timestamp
                model_dates[model_name] = datetime.fromisoformat(most_recent_timestamp).strftime('%d/%HZ')

            if disturbance_candidates:
                prev_lat = None
                prev_lon = None
                prev_lon_repeat2 = None
                prev_lon_repeat3 = None
                numdisturb = len(disturbance_candidates)
                num = 0

                lat_lon_with_time_step_list = []
                for time_step_str, valid_time_str, candidate in disturbance_candidates:
                    num += 1
                    time_step_int = int(time_step_str)
                    lon = candidate['lon']
                    candidate_info = {}
                    candidate_info['model_name'] = model_name
                    candidate_info['init_time'] = datetime.fromisoformat(model_timestamp)
                    candidate_info['lat'] = candidate['lat']
                    candidate_info['lon'] = lon
                    candidate_info['lon_repeat'] = candidate_info['lon']
                    candidate_info['valid_time'] = datetime.fromisoformat(valid_time_str)
                    candidate_info['time_step'] = time_step_int

                    # calculate the difference in hours
                    hours_diff = (candidate_info['valid_time'] - datetime.fromisoformat(valid_day)).total_seconds() / 3600
                    # round to the nearest hour
                    hours_diff_rounded = round(hours_diff)
                    candidate_info['hours_after_valid_day'] = hours_diff_rounded

                    candidate_info['roci'] = candidate['roci']/1000
                    vmaxkt = candidate['vmax10m_in_roci'] * 1.9438452
                    candidate_info['vmax10m_in_roci'] = vmaxkt
                    candidate_info['closed_isobar_delta'] = candidate['closed_isobar_delta']
                    candidate_info['mslp_value'] = candidate['mslp_value']

                    if prev_lon:
                        prev_lon_f = float(prev_lon)
                        if abs(prev_lon_f - lon) > 270:
                            if prev_lon_f < lon:
                                prev_lon = prev_lon_f + 360
                            else:
                                prev_lon = prev_lon_f - 360

                    candidate_info['prev_lat'] = prev_lat
                    candidate_info['prev_lon'] = prev_lon
                    candidate_info['prev_lon_repeat'] = prev_lon

                    prev_lat = candidate_info['lat']
                    prev_lon = lon
                    prev_lon_repeat1 = lon

                    lat_lon_with_time_step_list.append(candidate_info)
                    """
                    # add copies to extend the map left and right
                    ccopy = candidate_info.copy()
                    lon_repeat2 = lon - 360
                    ccopy['lon_repeat'] = f"{lon_repeat2:3.2f}"
                    if prev_lon_repeat2:
                        prev_lon_repeat2_f = float(prev_lon_repeat2)
                        if abs(prev_lon_repeat2_f - lon_repeat2) > 270:
                            if prev_lon_repeat2_f < lon_repeat2:
                                prev_lon_repeat2 = f"{prev_lon_repeat2_f + 360:3.2f}"
                            else:
                                prev_lon_repeat2 = f"{prev_lon_repeat2_f - 360:3.2f}"

                    ccopy['prev_lon_repeat'] = prev_lon_repeat2
                    prev_lon_repeat2 = ccopy['lon_repeat']

                    lat_lon_with_time_step_list.append(ccopy)
                    ccopy = candidate_info.copy()
                    lon_repeat3 = lon + 360
                    ccopy['lon_repeat'] = f"{lon_repeat3:3.2f}"
                    if prev_lon_repeat3:
                        prev_lon_repeat3_f = float(prev_lon_repeat3)
                        if abs(prev_lon_repeat3_f - lon_repeat3) > 270:
                            if prev_lon_repeat3_f < lon_repeat3:
                                prev_lon_repeat3 = f"{prev_lon_repeat3_f + 360:3.2f}"
                            else:
                                prev_lon_repeat3 = f"{prev_lon_repeat3_f - 360:3.2f}"

                    ccopy['prev_lon_repeat'] = prev_lon_repeat3
                    prev_lon_repeat3 = ccopy['lon_repeat']

                    lat_lon_with_time_step_list.append(ccopy)

                    """

                self.update_plotted_list(lat_lon_with_time_step_list)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
                    opacity = 1.0
                    radius = 6
                    lons = {}
                    lats = {}
                    for point in reversed(lat_lon_with_time_step_list):
                        hours_after = point['hours_after_valid_day']
                        #if start <= time_step <= end:
                        # use hours after valid_day instead
                        if start <= hours_after <= end:
                            for upper_bound, vmaxmarker in vmax_kt_threshold:
                                marker = vmaxmarker
                                if point['vmax10m_in_roci'] < upper_bound:
                                    break
                            if marker not in lons:
                                lons[marker] = []
                                lats[marker] = []
                            lons[marker].append(point['lon_repeat'])
                            lats[marker].append(point['lat'])

                            #self.ax.plot([point['lon_repeat']], [point['lat']], marker=marker, color=colors[i], markersize=radius, alpha=opacity)

                    for vmaxmarker in lons.keys():
                        #self.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker, facecolors='none', edgecolors=colors[i], s=radius**2, alpha=opacity, antialiased=False)
                        self.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker, facecolors='none', edgecolors=colors[i], s=marker_sizes[vmaxmarker]**2, alpha=opacity, antialiased=False)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
                    line_color = colors[i]
                    opacity = 1.0
                    strokewidth = 0.5

                    line_segments = []
                    for point in reversed(lat_lon_with_time_step_list):
                        hours_after = point['hours_after_valid_day']
                        if start <= hours_after <= end:
                            if point['prev_lon_repeat']:

                                # Create a list of line segments
                                line_segments.append([(point['prev_lon_repeat'], point['prev_lat']), (point['lon_repeat'], point['lat'])])
                                """
                                plt.plot([point['prev_lon_repeat'], point['lon_repeat']], [point['prev_lat'], point['lat']],
                                         color=color, linewidth=strokewidth, marker='', markersize = 0, alpha=opacity)
                                """

                    # Create a LineCollection
                    lc = LineCollection(line_segments, color=colors[i], linewidth=strokewidth, alpha=opacity)
                    # Add the LineCollection to the axes
                    self.ax.add_collection(lc)

                    name = 'Tracks'

        labels_positive = [f' D+{str(i): >2} ' for i in range(len(colors)-1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        for i, (color, label) in enumerate(zip(reversed(colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i*20

            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color=list(reversed(label_fg_colors))[i],
                        fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor=color, alpha=1.0))

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i*35
            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                        fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor='#000000', alpha=1.0))
            #self.ax.scatter([x_pos + 200], [y_pos], marker=marker, edgecolors='#BBDDCC', facecolors='#446655', s=200.0, alpha=1.0, antialiased=False, transform=self.ax.transAxes)

        self.fig.canvas.draw()
        self.label_genesis_mode.config(text="GENESIS MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime('%Y-%m-%d'))
        self.genesis_model_cycle_time = most_recent_model_cycle
        self.genesis_models_label.config(text=f"Latest models: GFS [{model_dates['GFS']}], ECM[{model_dates['ECM']}], NAV[{model_dates['NAV']}], CMC[{model_dates['CMC']}]")

    def update_axes(self):
            gl = self.lastgl
            if self.zoom_rect and len(self.zoom_rect) == 4:
                x0, y0, x1, y1 = self.zoom_rect
            else:
                x0, y0, x1, y1 = -180, -90, 180, 90
            extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
            if gl:
                gl.top_labels = False
                gl.left_labels = False
                gl.xlines = False
                gl.ylines = False
                for artist in (gl.xline_artists + gl.yline_artists +
                               gl.bottom_label_artists + gl.top_label_artists +
                               gl.left_label_artists + gl.right_label_artists):
                    try:
                        artist.remove()
                    except:
                        pass

                try:
                    self.ax._gridliners.remove(gl)
                except:
                    pass

            gl = self.ax.gridlines(draw_labels=["bottom", "left"], x_inline=False, y_inline=False, auto_inline=False, color='white', alpha=0.5, linestyle='--')
            # Move axis labels inside the subplot
            self.ax.tick_params(axis='both', direction='in', labelsize=16)
            gl.xpadding = -10     # Ideally, this would move labels inside the map, but results in hidden labels
            gl.ypadding = -10     # Ideally, this would move labels inside the map, but results in hidden labels

            gl.xlabel_style = {'color': 'orange'}
            gl.ylabel_style = {'color': 'orange'}

            # adjust spacing of grid lines by zoom
            xdiff = abs(extent[1] - extent[0])
            ydiff = abs(extent[3] - extent[2])

            # preferenced by experimentation with 16:9 monitor
            if xdiff <= 25:
                xdegn = 1
            elif xdiff <= 50:
                xdegn = 5
            else:
                xdegn = 10

            if ydiff <= 15:
                ydegn = 1
            elif ydiff <= 40:
                ydegn = 5
            else:
                ydegn = 10

            gl.xlocator = plt.MultipleLocator(xdegn)
            gl.ylocator = plt.MultipleLocator(ydegn)

            self.lastgl = gl

            #gl.xformatter = LONGITUDE_FORMATTER
            #gl.yformatter = LATITUDE_FORMATTER
            lat_formatter = LatitudeFormatter(direction_label=True)
            lon_formatter = LongitudeFormatter(direction_label=True)
            self.ax.xaxis.set_major_formatter(lon_formatter)
            self.ax.yaxis.set_major_formatter(lat_formatter)

    def display_map(self):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        # Adjust figure size to fill the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        dpi = 100
        if self.fig:
            try:
                plt.close(self.fig)
                self.reset_measurement()
            except:
                pass

        self.fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi, facecolor='black')

        #self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        self.ax = plt.axes(projection=ccrs.PlateCarree())

        #self.ax.autoscale_view(scalex=True,scaley=True)
        self.ax.set_facecolor('black')

        # Draw a rectangle patch around the subplot to visualize its boundaries
        extent = self.ax.get_extent()
        rect = Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2],
                         linewidth=2, edgecolor='white', facecolor='none')
        self.ax.add_patch(rect)

        # Adjust aspect ratio of the subplot
        self.ax.set_aspect('equal')

        # Draw red border around figure
        #self.fig.patch.set_edgecolor('red')
        #self.fig.patch.set_linewidth(2)

        #self.ax.stock_img()
        self.ax.set_extent([-180, 180, -90, 90])
        self.ax.add_feature(cfeature.COASTLINE, edgecolor='yellow', linewidth=0.5)

        self.update_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key_press = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_key_release = self.canvas.mpl_connect('key_release_event', self.on_key_release)

        # we only know how big the canvas frame/plot is after drawing/packing, and we need to wait until drawing to fix the size
        self.canvas_frame.update_idletasks()
        # fix the size to the canvas frame
        frame_hsize = float(self.canvas_frame.winfo_width()) / dpi
        frame_vsize = float(self.canvas_frame.winfo_height()) / dpi
        h = [Size.Fixed(0), Size.Fixed(frame_hsize)]
        v = [Size.Fixed(0), Size.Fixed(frame_vsize)]
        divider = Divider(self.fig, (0, 0, 1, 1), h, v, aspect=False)
        self.ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        self.clear_storm_extrema_annotations()
        self.annotated_circles = AnnotatedCircles(self.ax)
        self.canvas.draw()
        self.display_custom_boundaries()

    def on_key_press(self, event):
        if event.key == 'shift':
            self.measure_mode = True

        if event.key == 'escape':
            # abort a zoom
            # Remove the rectangle patch
            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None
            self.zoom_rect = None
            self.fig.canvas.draw()

        if event.key == '0':
            self.zoom_out(max_zoom=True)

        if event.key == '-':
            self.zoom_out(step_zoom=True)

        if event.key == '=':
            self.zoom_in(step_zoom=True)

        if event.key == 'n':
            self.cycle_to_next_overlapped_point()

        if event.key == 'x':
            # annotate storm extrema
            self.annotate_storm_extrema()

        if event.key == 'c':
            # annotate storm extrema
            self.clear_storm_extrema_annotations()
            self.fig.canvas.draw()

    def on_key_release(self, event):
        if event.key == 'shift':
            self.measure_mode = False

    def on_click(self, event):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if event.inaxes == self.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

            if inbound:
                if self.measure_mode:
                    if event.button == 1:
                        self.start_point = (event.xdata, event.ydata)
                        self.end_point = (event.xdata, event.ydata)
                    elif event.button == 3:
                        self.reset_measurement()
                else:
                    if event.button == 1:  # Left click
                        if self.annotated_circles:
                            annotations = self.annotated_circles.get_annotations()
                            if annotations:
                                for annotation in annotations:
                                    if annotation and annotation.contains_point(event):
                                        return

                        self.zoom_rect = [event.xdata, event.ydata]
                        if self.rect_patch:
                            self.rect_patch.remove()
                            self.rect_patch = None
                    elif event.button == 3:  # Right click
                        self.zoom_out()

    def on_release(self, event):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Check if mouse coordinates are within figure bounds
        try:
            inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
        except:
            inbound = False

        if self.measure_mode:
            if event.button == 1 and self.start_point:
                if inbound:
                    self.end_point = (event.xdata, event.ydata)

                try:
                    distance = self.calculate_distance(self.start_point, self.end_point)
                except:
                    distance = 0
                if distance > 0:
                    if self.line:
                        try:
                            self.line.remove()
                        except:
                            pass
                    if self.distance_text:
                        try:
                            self.distance_text.remove()
                        except:
                            pass

                    self.line = self.ax.plot([self.start_point[0], self.end_point[0]],
                                             [self.start_point[1], self.end_point[1]],
                                             color='cyan', linewidth=2, transform=ccrs.PlateCarree())[0]
                    self.display_distance_text(distance)
                    self.start_point = None
                    self.end_point = None
                    self.fig.canvas.draw()
                else:
                    self.reset_measurement()
                    self.fig.canvas.draw()
        else:
            if event.button == 1 and self.zoom_rect:  # Left click release
                if event.xdata is None or event.ydata is None or not inbound:
                    if self.rect_patch is None:
                        self.zoom_rect = None
                        return
                    x1 = self.zoom_rect[0] + self.rect_patch.get_width()
                    y1 = self.zoom_rect[1] + self.rect_patch.get_height()
                    self.zoom_rect.extend([x1, y1])
                else:
                    self.zoom_rect.extend([event.xdata, event.ydata])

                # Remove the rectangle patch after zoom operation
                if self.rect_patch:
                    self.rect_patch.remove()
                    self.rect_patch = None
                self.zoom_in()
                self.zoom_rect = None

    def update_circle_patch(self, lon=None, lat=None):
        if self.last_circle_lon == lon and self.last_circle_lat == lat:
            return

        if self.circle_handle:
            self.circle_handle.remove()

        self.circle_handle = Circle((lon, lat), radius=self.calculate_radius_pixels(), color='pink', fill=False, linestyle='dotted', linewidth=2, alpha=0.8,
                                    transform=ccrs.PlateCarree())
        self.ax.add_patch(self.circle_handle)
        self.fig.canvas.draw()

        self.last_circle_lon = lon
        self.last_circle_lat = lat

    def clear_circle_patch(self):
        if self.circle_handle:
            self.circle_handle.remove()
            self.circle_handle = None
            self.fig.canvas.draw()

        self.last_circle_lon = None
        self.last_circle_lat = None

    def on_motion(self, event):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if event.inaxes == self.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

            lon = event.xdata
            lat = event.ydata
            self.last_cursor_lon_lat = (lon, lat)
            self.update_labels_for_mouse_hover(lat=lat, lon=lon)

            if self.measure_mode:
                if self.start_point:
                    if inbound:
                        self.end_point = (event.xdata, event.ydata)
                    else:
                        return

                    try:
                        if not self.line:
                            self.line = self.ax.plot([self.start_point[0], self.end_point[0]],
                                                     [self.start_point[1], self.end_point[1]],
                                                     color='cyan', linewidth=2, transform=ccrs.PlateCarree())[0]
                    except:
                        pass

                    try:
                        self.line.set_data([self.start_point[0], self.end_point[0]],
                                           [self.start_point[1], self.end_point[1]])
                    except:
                        pass

                    if self.distance_text:
                        try:
                            self.distance_text.remove()
                        except:
                            pass

                    try:
                        distance = self.calculate_distance(self.start_point, self.end_point)
                        self.display_distance_text(distance)
                    except:
                        distance = 0

                    self.fig.canvas.draw()
            else:
                if self.zoom_rect:
                    x0, y0 = self.zoom_rect
                    if inbound:
                        x1, y1 = event.xdata, event.ydata
                    else:
                        # out of bound motion
                        return
                    if type(x0) == type(x1) == type(y0) == type(y1):
                        if self.rect_patch:
                            try:
                                self.rect_patch.remove()
                            except:
                                pass
                        width = x1 - x0
                        height = y1 - y0
                        self.rect_patch = Rectangle((x0, y0), width, height, fill=False, color='yellow', linestyle='--')
                        self.ax.add_patch(self.rect_patch)
                        self.fig.canvas.draw_idle()

    def reset_measurement(self):
        if self.line:
            try:
                self.line.remove()
            except:
                pass
            self.line = None
        if self.distance_text:
            try:
                self.distance_text.remove()
            except:
                pass
            self.distance_text = None
        self.start_point = None
        self.end_point = None
        self.fig.canvas.draw()

    def calculate_distance(self, start_point, end_point):
        geod = cgeo.Geodesic()
        lats_lons = [start_point, end_point]
        line = LineString([start_point, end_point])
        total_distance = geod.geometry_length(line)
        nautical_miles = total_distance / 1852.0
        return nautical_miles

    def display_distance_text(self, distance):
        if self.start_point and self.end_point:
            mid_point = ((self.start_point[0] + self.end_point[0]) / 2,
                         (self.start_point[1] + self.end_point[1]) / 2)

            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # Fixed offset distance in pixels
            offset_pixels = 20  # Adjust as needed

            # Calculate degrees per pixel in both x and y directions
            extent = self.ax.get_extent()
            lon_diff = extent[1] - extent[0]
            lat_diff = extent[3] - extent[2]

            lon_deg_per_pixel = lon_diff / self.ax.get_window_extent().width
            lat_deg_per_pixel = lat_diff / self.ax.get_window_extent().height

            # Convert offset from pixels to degrees
            offset_degrees = offset_pixels * lon_deg_per_pixel

            # Calculate offset in degrees in x and y directions
            offset_x_deg = offset_degrees * np.cos(np.radians(angle + 90))
            offset_y_deg = offset_degrees * np.sin(np.radians(angle + 90))

            # Adjust angle to ensure text is always right-side up
            if angle > 90 or angle < -90:
                angle += 180
                offset_x_deg = -offset_x_deg
                offset_y_deg = -offset_y_deg

            self.distance_text = self.ax.text(mid_point[0] + offset_x_deg, mid_point[1] + offset_y_deg,
                                              f"{distance:.2f} NM", color='white', fontsize=12,
                                              ha='center', va='center', rotation=angle,
                                              bbox=dict(facecolor='black', alpha=0.5))

    def zoom_in(self, step_zoom=False):
        if step_zoom:
            extent = self.ax.get_extent()
            zoom_factor = ZOOM_IN_STEP_FACTOR
            if zoom_factor <= 1.0:
                zoom_factor = 3.0
            lon_diff = extent[1] - extent[0]
            lat_diff = extent[3] - extent[2]
            lon_center, lat_center = self.last_cursor_lon_lat
            target_width = lon_diff / ZOOM_IN_STEP_FACTOR
            target_height = lat_diff / ZOOM_IN_STEP_FACTOR
            x0 = lon_center - (target_width / 2.0)
            x1 = lon_center + (target_width / 2.0)
            y0 = lat_center - (target_height / 2.0)
            y1 = lat_center + (target_height / 2.0)
            x0_diff = x1_diff = y0_diff = y1_diff = 0.0
            if x0 < -180.0:
                x0_diff = x0 - (-180.0)
                x0 = -180.0
                x1 = x0 + target_width
            if x1 > 180.0:
                x1_diff = 180.0 - x1
                x1 = 180.0
                x0 = x1 - target_width
            if y0 < -90.0:
                y1_diff = y0 - (-90.0)
                y0 = 90.0
                y1 = y0 + target_height
            if y1 > 90.0:
                y1_diff = 90.0 - y1
                y1 = 90.0
                y0 = y1 - target_height

            # Ensure new extent is within bounds
            new_extent = [
                x0,
                x1,
                y0,
                y1,
            ]

            self.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
            self.clear_storm_extrema_annotations()
            self.update_axes()
            self.fig.canvas.draw()

        elif self.zoom_rect and (None not in self.zoom_rect) and len(self.zoom_rect) == 4:
            x0, y0, x1, y1 = self.zoom_rect
            if x0 != x1 and y0 != y1:
                extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
                self.ax.set_extent(extent, crs=ccrs.PlateCarree())

                self.clear_storm_extrema_annotations()
                self.update_axes()
                self.fig.canvas.draw()

    def zoom_out(self, max_zoom=False, step_zoom=False):
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        max_lon_diff = 360.0  # Maximum longitudinal extent
        max_lat_diff = 180.0  # Maximum latitudinal extent

        if step_zoom:
            zoom_factor = ZOOM_IN_STEP_FACTOR
            if zoom_factor <= 1.0:
                zoom_factor = 3.0
            lon_center = (extent[1] + extent[0]) / 2.0
            lat_center = (extent[3] + extent[2]) / 2.0
            target_width = lon_diff * ZOOM_IN_STEP_FACTOR
            target_height = lat_diff * ZOOM_IN_STEP_FACTOR
            x0 = lon_center - (target_width / 2.0)
            x1 = lon_center + (target_width / 2.0)
            y0 = lat_center - (target_height / 2.0)
            y1 = lat_center + (target_height / 2.0)
            x0_diff = x1_diff = y0_diff = y1_diff = 0.0
            if x0 < -180.0:
                x0_diff = x0 - (-180.0)
                x0 = -180.0
                x1 = x0 + target_width
            if x1 > 180.0:
                x1_diff = 180.0 - x1
                x1 = 180.0
                x0 = x1 - target_width
            if y0 < -90.0:
                y1_diff = y0 - (-90.0)
                y0 = 90.0
                y1 = y0 + target_height
            if y1 > 90.0:
                y1_diff = 90.0 - y1
                y1 = 90.0
                y0 = y1 - target_height

            # Ensure new extent is within bounds
            new_extent = [
                x0,
                x1,
                y0,
                y1,
            ]

            if x1 - x0 <= max_lon_diff and y1 - y1 <= max_lat_diff:
                self.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
                self.clear_storm_extrema_annotations()
                self.update_axes()
                self.fig.canvas.draw()

        else:
            # Define zoom factor or step size
            zoom_factor = 0.5

            new_lon_min = max(extent[0] - lon_diff / zoom_factor, -max_lon_diff / 2)
            new_lon_max = min(extent[1] + lon_diff / zoom_factor, max_lon_diff / 2)
            new_lat_min = max(extent[2] - lat_diff / zoom_factor, -max_lat_diff / 2)
            new_lat_max = min(extent[3] + lat_diff / zoom_factor, max_lat_diff / 2)

            if new_lat_max - new_lat_min > 140 or max_zoom:
                new_lat_min = -90
                new_lat_max = 90
                new_lon_min = -180
                new_lon_max = 180

            # Ensure new extent is within bounds
            new_extent = [
                new_lon_min,
                new_lon_max,
                new_lat_min,
                new_lat_max,
            ]

            # Ensure the zoom doesn't exceed maximum extents
            if new_lon_max - new_lon_min <= max_lon_diff and new_lat_max - new_lat_min <= max_lat_diff:
                self.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
                self.clear_storm_extrema_annotations()
                self.update_axes()
                self.fig.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()

    # Style configuration for ttk widgets
    style = ttk.Style()
    style.theme_use('clam')  # Ensure using a theme that supports customization
    default_bg = "black"
    default_fg = "white"
    style.configure("TButton", background=default_bg, foreground=default_fg)
    style.configure("White.TButton", background=default_bg, foreground="white")
    style.configure("Red.TButton", background=default_bg, foreground="red")
    style.configure("Orange.TButton", background=default_bg, foreground="orange")
    style.configure("Yellow.TButton", background=default_bg, foreground="yellow")

    style.configure("TLabel", background=default_bg, foreground=default_fg)
    style.configure("FixedWidthWhite.TLabel", font=("Latin Modern Mono", 12), background=default_bg, foreground="white")
    style.configure("FixedWidthRed.TLabel", font=("Latin Modern Mono", 12), background=default_bg, foreground="red")

    style.configure("TCheckbutton", background=default_bg, foreground=default_fg)
    style.configure("TopFrame.TFrame", background=default_bg)
    style.configure("ToolsFrame.TFrame", background=default_bg)
    style.configure("CanvasFrame.TFrame", background=default_bg)

    app = App(root)
    root.mainloop()