# an experimental TC plotter (stand-alone full screen viewer) for intensity models and genesis
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
# CLEAR (ALL OR SINGLE BOX) STORM EXTREMA ANNOTATIONS = c key
# HIDE (MOUSE HOVERED) STORMS / SHOW ALL HIDDEN = h key
# SAVE SCREENSHOT = p key
# TOGGLE RVOR CONTOURS = v key
# TOGGLE RVOR CONTOUR LABELS = l key
# SHOW RVOR LEVELS CHOOSER = V key

# Click on (colored) legend for valid time days to cycle between what to display
#   (relative to (00Z) start of earliest model init day)
#   clicking on a legend box with a white edge toggles between:
#       1: (default) display all storm track valid times prior to selected (changes successive legend edges bright pink)
#       2: hide all storms with tracks with the first valid day later than selected (changes successive edges dark pink)
#       3: hide all points with tracks later than valid day
#   clicking on a pink/dark pink edge switches it to (1) white edge for all prior and (2) pink edge for days after

## MEASURE TOOL DOC
#   HOLDING SHIFT KEY TOGGLES UPDATING END POINT, ALLOWING ZOOMING IN/OUT:
#   ALLOWS FOR SINGLE MEASUREMENT WITHOUT NEEDING TO DRAG END POINTS
#   ZOOM IN TO PRECISION NEEDED TO SET START POINT, START MEASUREMENT (SHIFT CLICK AND MOTION), RELEASE SHIFT (AND CLICK)
#   ZOOM OUT, THEN ZOOM IN TO WHERE TO PLACE END POINT AND HOLD SHIFT AND MOUSE MOTION TO PLACE END POINT
#   RELEASE SHIFT WHEN END POINT POSITION SET. FREE TO ZOOM IN/OUT AFTERWARD

####### CONFIG

# how often (in minutes) to check for stale data in three classes: tcvitals, adeck, bdeck
#   for notification purposes only.
#      colors reload button red (a-deck), orange (b-deck), yellow (tcvitals) -without- downloading data automatically
#   checks modification date in any class of the three, and refreshes the entire class
#      timer resets after manual reloads
TIMER_INTERVAL_MINUTES = 30

# Plot relative vorticity contours (cyclonic/anti-cyclonic) and labels
RVOR_CYCLONIC_CONTOURS = True
RVOR_CYCLONIC_LABELS = True
RVOR_ANTICYCLONIC_CONTOURS = True
RVOR_ANTICYCLONIC_LABELS = True
# RVOR LEVELS TO DISPLAY SIMULTANEOUSLY
SELECTED_PRESSURE_LEVELS = [925, 850, 700, 500, 200]
MINIMUM_CONTOUR_PX_X = 20
MINIMUM_CONTOUR_PX_Y = 20

# On mouse hover, select points to display status within a 0.1 degree bounding box
MOUSE_SELECT_IN_DEGREES = 0.5

# Size of circle patch radius on mouse hover (in pixels)
DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS = 20

# assume any subsequent circle annotations (for extrema) in this bounding box are overlapped in the display, so don't render
ANNOTATE_CIRCLE_OVERLAP_IN_DEGREES = 0.01

# how much to zoom in when using minus key
ZOOM_IN_STEP_FACTOR = 3

# should not affect the grid spacing (as the grid spacing is based on inches)
CHART_DPI = 100

# Calculate the grid line spacing in inches for each multiple
GRID_LINE_SPACING_DEGREES  = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 90.0]
# Choose the smallest option to meet the minimum grid line spacing requirement (otherwise will use largest of above)
MIN_GRID_LINE_SPACING_INCHES = 0.75

CYCLONIC_PRESSURE_LEVEL_COLORS = {
    '925': '#af2eff',
    '850': '#ff00ec',
    '700': '#ff7dfc',
    '500': '#fdc6ff',
    '200': '#fef1ff'
}

ANTI_CYCLONIC_PRESSURE_LEVEL_COLORS = {
    '925': '#2295b3',
    '850': '#00b6f5',
    '700': '#60e9ff',
    '500': '#78fff7',
    '200': '#b9faff'
}

DEFAULT_ANNOTATE_MARKER_COLOR = "#FFCCCC"
DEFAULT_ANNOTATE_TEXT_COLOR = "#FFCCCC"
ANNOTATE_DT_START_COLOR = "#00FF00"
ANNOTATE_EARLIEST_NAMED_COLOR = "#FFFF00"
ANNOTATE_VMAX_COLOR = "#FF60F0"

# importance assigned to colors
#  higher is more important and will switch to more important color for a point with multiple labels
ANNOTATE_COLOR_LEVELS = {
    0: DEFAULT_ANNOTATE_TEXT_COLOR,
    1: ANNOTATE_DT_START_COLOR,
    2: ANNOTATE_EARLIEST_NAMED_COLOR,
    3: ANNOTATE_VMAX_COLOR
}

# remove any from list not wanted visible for extrema annotations (x key)
#   closed_isobar_delta is only from tc_candidates.db
#      (it is the difference in pressure from the outermost closed isobar to the innermost closed isobar)
#       (i.e. how many concentric circles on a chart of pressure); or POUTER - MSLP in abdecks
#displayed_extremum_annotations = ["dt_start", "dt_end", "vmax10m", "mslp_value", "roci", "closed_isobar_delta"]
#displayed_extremum_annotations = ["dt_start", "vmax10m", "roci", "closed_isobar_delta"]

# rather than a simple annotations list, use a function for interrogation & the parameters using it
#    four parameters for the config:
#       order (an integer from low to high), lower order displayed first
#       label lines, the lambda func to return what is displayed, with {extremum_val} of note
#       the parameter used by the function
#       the name of the function
# duplicate lines are removed (so it is fine to prefix multiple parameters with the valid time
# key name determines order (lower order first)

# what short names (keys) from annotations_result_val are actually annotated (order is as below for multiple lines at a point)
#   displayable options are in annotations_result_val

DISPLAYED_FUNCTIONAL_ANNOTATIONS = [
    "TC Start",
    "Earliest Named"
]

##### END CONFIG

# Maximum number of hours old before not loading rvor contours
MAX_RVOR_HOURS_OLD = 6

##### CODING TWEAKS (Extend what extrema are annotated)

# not a config. This is what is displayed in modal dialog for options. Placed here in case of user extensions.
displayed_functional_annotation_options = [
    "TC Start",
    "Earliest Named",
    "Peak Vmax",
    "Peak Isobar Delta",
    'Peak ROCI',
    "TC End"
]

# custom comparison functions for annotations
#   passing the lst of points (disturbance candidates) for the tc_candidate
#   param_keys is the list of key naems in the point used for comparison
#   should return the index of interest (None if none satisfy)
annotations_comparison_func_dict = {
    'index_of_first_>=_n': lambda lst, param_keys, min_n: next(((i, x[param_keys[0]]) for i, x in enumerate(lst)
        if param_keys[0] in x and x[param_keys[0]] and x[param_keys[0]] >= min_n), None),
    'index_of_first_<=_n': lambda lst, param_keys, max_n: next(((i, x[param_keys[0]]) for i, x in enumerate(lst)
        if param_keys[0] in x and x[param_keys[0]] and x[param_keys[0]] <= max_n), None),
    'index_of_max': lambda lst, param_keys: next(
        ((i, x[param_keys[0]]) for i, x in enumerate(lst)
            if x.get(param_keys[0]) is not None
                and
                max(
                    y.get(param_keys[0]) for y in lst
                    ) is not None
                and
                x.get(param_keys[0]) ==
                    max(
                        y.get(param_keys[0]) for y in lst
                    )
        ), None) if lst and param_keys else None,
    'index_of_min': lambda lst, param_keys: next(
        ((i, x[param_keys[0]]) for i, x in enumerate(lst)
            if x.get(param_keys[0]) is not None
                and
                min(
                    y.get(param_keys[0]) for y in lst
                    ) is not None
                and
                x.get(param_keys[0]) ==
                    min(
                        y.get(param_keys[0]) for y in lst
                    )
        ), None) if lst and param_keys else None
}

# determine which function should be associated with each label line, return (tc_point_index, parameter value)
#   tc_candidate is the list of dicts (points) with all parameters
#   param_keys is always a list of parameter keys used by annotations_comparison_func_dict
annotations_result_val = {
    'TC Start': lambda tc_candidate: annotations_comparison_func_dict['index_of_min'](tc_candidate, ['valid_time']),
    'Earliest Named': lambda tc_candidate: annotations_comparison_func_dict['index_of_first_>=_n'](tc_candidate, ['vmax10m'], 34),
    'Peak Vmax': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['vmax10m']),
    'Min MSLP': lambda tc_candidate: annotations_comparison_func_dict['index_of_min'](tc_candidate, ['mslp_value']),
    'Peak ROCI': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['roci']),
    'Peak Isobar Delta': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['closed_isobar_delta']),
    'TC End': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['valid_time'])
}
# formatting for annotation
#   key = short name, result_val is result from lambda comparison functions above
#   point tc_candidate[tc_point_index) passed to lambda with added parameter 'result_val'
#       {result_val} is val from annotations_comparison_func_dict evaluation
annotations_label_func_dict = {
    'TC Start': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        "\nSTART",
    'Earliest Named': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        "\nEARLIEST 34kt",
    'Peak Vmax': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        f"\nVMAX_10m: {result_val:.1f} kt",
    'Min MSLP': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        f"\nMSLP: {result_val:.1f} hPa",
    'Peak ROCI': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        f"\nROCI: {result_val:.0f} km",
    'Peak Isobar Delta': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        f"\nISOBAR_DELTA: {result_val:.0f} hPa",
    'TC End': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
        "\nEND"
}

annotations_color_level = {
    'TC Start':1,
    'Earliest Named': 2,
    'Peak Vmax': 3,
    'Min MSLP': 0,
    'Peak ROCI': 0,
    'Peak Isobar Delta': 0,
    'TC End': 0
}

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
##################

# for screenshots
from PIL import ImageGrab

# for cycling overlapped points
from collections import OrderedDict

# for mouse hover features
from rtree import index

# for measurements (geodesic)
#   note: measures geodesic distance, not the 'path' distance that might show as a blue line
import cartopy.geodesic as cgeo
from shapely.geometry import LineString
import numpy as np

#zoom box blitting

# config dialog
from tkinter import colorchooser

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from mpl_toolkits.axes_grid1 import Divider, Size

import sqlite3

from matplotlib.collections import LineCollection

# performance optimizations
import matplotlib

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
shapefile_path = None
try:
    tmp_gdf = gpd.read_file(shapefile_path)
    # Filter the GeoDataFrame to only include the Houston-The Woodlands-Sugar Land, TX Metro Area
    houston_gdf = tmp_gdf[tmp_gdf['NAME'] == 'Houston-Pasadena-The Woodlands, TX']
    if houston_gdf.empty:
        raise ValueError("Houston-The Woodlands-Sugar Land, TX Metro Area not found in the shapefile")
    # Ensure the CRS matches that of the Cartopy map (PlateCarree)
    custom_gdf = houston_gdf.to_crs(ccrs.PlateCarree().proj4_init)
except:
    custom_gdf = None


#### CUSTOM OVERLAY
# plot relative vorticity contours

shapefile_paths = {
    'rvor_c_poly': 'rvor_contours/rvor_cyclonic_contours.shp',
    'rvor_c_points': 'rvor_contours/rvor_cyclonic_labels.shp',
    'rvor_ac_poly': 'rvor_contours/rvor_anticyclonic_contours.shp',
    'rvor_ac_points': 'rvor_contours/rvor_anticyclonic_labels.shp'
}

# disable rvor
# noinspection PyRedeclaration
shapefile_paths = {}

overlay_gdfs = {}
try:
    shapefiles_are_recent = all(
        os.path.exists(path) and
        datetime.fromtimestamp(os.path.getmtime(path)) > datetime.now() - timedelta(hours=MAX_RVOR_HOURS_OLD)
        for path in shapefile_paths.values()
    )
    if shapefiles_are_recent:
        for name, shapefile_path in shapefile_paths.items():
            gdf = gpd.read_file(shapefile_path)
            # Filter the GeoDataFrame to only include the Houston-The Woodlands-Sugar Land, TX Metro Area
            if gdf.empty:
                raise ValueError("Empty shapefile")
            # Ensure the CRS matches that of the Cartopy map (PlateCarree)
            custom_gdf2 = gdf.to_crs(ccrs.PlateCarree().proj4_init)
            overlay_gdfs[name] = custom_gdf2
except:
    traceback.print_exc()

#### END CUSTOM OVERLAY
# Debugging # TODO REMOVE
import inspect
def print_line_number():
    frame = inspect.getouterframes(inspect.currentframe())[1]
    print(f"Current line number: {frame.lineno}")
###


matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = CHART_DPI

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
def get_recent_storms(urls):
    storms = {}
    dt_mods_tcvitals = {}
    current_time = datetime.utcnow()
    for url in urls:
        response = None
        try:
            response = requests.get(url)
        except:
            pass

        if response and response.status_code == 200:
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

def get_modification_date_from_header(response_headers):
    try:
        # Assume already done status code check
            modification_date = response_headers.get('Last-Modified')
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
def get_deck_files(storms, urls_a, urls_b, do_update_adeck, do_update_bdeck):
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
            for url in urls_a:
                file_url = url.format(basin_id=basin_id.lower(), year=year, storm_number=storm_number)
                if file_url[-3:] == ".gz":
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
                except Exception as e:
                    traceback.print_exc()
                    print(f"Failed to download {file_url}: {e}")
        if do_update_bdeck:
            # Download B-Deck files
            for url in urls_b:
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

        mslp = None
        try:
            mslp = int(storm_vitals['central_pressure'])
            storm_vitals['mslp'] = int(mslp)
        except:
            storm_vitals['mslp'] = None

        outer_slp = None
        try:
            outer_slp = int(storm_vitals['environmental_pressure'])
            storm_vitals['outer_slp'] = int(outer_slp)
        except:
            storm_vitals['outer_slp'] = None

        storm_vitals['closed_isobar_delta'] = None
        try:
            if mslp and outer_slp:
                if mslp > 0 and outer_slp > 0:
                    storm_vitals['closed_isobar_delta'] = outer_slp - mslp
        except:
            pass

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
        idx = self._index % len(self)
        key, value = sorted_items[idx]
        self._index = (self._index + 1) % len(self)
        return (idx + 1, key)

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
        self.blocking = False
        self.circle_annotation = None
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
        self.blocking = False
        self.visible = True
        self.annotated_circle = None

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

    def set_dragging(self, dragging):
        if dragging and not self.dragging:
            self.block_for_dragging()
        elif self.dragging and not dragging:
            self.unblock_for_dragging()
        self.dragging = dragging

    def unblock_for_dragging(self):
        if self.blocking:
            EventManager.unblock_events()
            self.blocking = False

    def block_for_dragging(self):
        # only start blocking when we have a line
        block_for_dragging = EventManager.block_events('dragging_annotation')
        if not block_for_dragging:
            raise ValueError("Failed to block events for DraggingAnnotation")
        self.blocking = True
        return True

    def is_dragging(self):
        return self.dragging

    def set_circle_annotation(self, circle_annotation):
        self.circle_annotation = circle_annotation

    @classmethod
    def get_topmost_annotation(cls, annotations, event):
        # need a separate is dragging flag as bbox is unstable when we are blitting
        #  (as it can become invisible during the setup for blitting (in an async event with the other handlers)
        #  the bbox will result erroneously in a 1 pixel box causing the wrong annotation to drag
        if not annotations:
            return None
        valid_annotations = [ann for ann in annotations if ann and (ann.is_dragging() or ann.contains_point(event))]
        if not valid_annotations:
            return None
        # max_zorder = max(valid_annotations, key=lambda ann: ann.zorder).zorder
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
        if self.original_annotation and self.original_annotation.get_visible():
            bbox = self.original_annotation.get_window_extent()
            return bbox.contains(event.x, event.y)
        else:
            return False

    def on_press(self, event):
        if not self.visible:
            return

        if not self.ax.draggable_annotations:
            return

        if self != self.get_topmost_annotation(self.ax.draggable_annotations, event):
            return

        contains, attrd = self.original_annotation.contains(event)
        if not contains:
            return

        xy_orig = self.original_annotation.xy
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
        self.set_dragging(True)

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

    # noinspection PyUnusedLocal
    def on_release(self, event):
        if self.press is None:
            return

        # Update original annotation position
        self.original_annotation.set_position(self.dragging_annotation.get_position())

        # Show original annotation and hide dragging annotation
        self.dragging_annotation.set_visible(False)
        self.original_annotation.set_visible(True)
        # handle edge case of hide during drag
        self.set_visible(self.visible)

        self.press = None
        self.background = None
        self.set_dragging(False)

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

    def has_focus(self, event):
        if not self.visible:
            return False

        if self != self.get_topmost_annotation(self.ax.draggable_annotations, event):
            return False

        contains, attrd = self.original_annotation.contains(event)
        if not contains:
            return False

        return True

    #def isVisible(self):
    #    return self.visible

    def set_visible(self, visibility_target):
        self.visible = visibility_target

        # we will call set_visible after mouse release for that edge case of hide while dragging
        if self.dragging:
            return

        try:
            if self.line:
                self.line.set_visible(visibility_target)
        except:
            traceback.print_exc()
            pass

        try:
            if self.original_annotation:
                self.original_annotation.set_visible(visibility_target)
        except:
            traceback.print_exc()

        try:
            if self.dragging_annotation:
                self.dragging_annotation.set_visible(False)
        except:
            traceback.print_exc()
            pass

    def remove(self):
        self.unblock_for_dragging()
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        if self.line:
            self.line.remove()
        self.original_annotation.remove()
        self.dragging_annotation.remove()

class AnnotatedCircles:
    ax = None
    circle_handles = None
    rtree_p = index.Property()
    rtree_idx = index.Index(properties=rtree_p)
    annotated_circles = None
    # only increment counter so we only have unique ids
    counter = 0

    def __init__(self, ax):
        self.__class__.ax = ax

    @classmethod
    def any_annotation_contains_point(cls, event):
        annotations = cls.get_draggable_annotations()
        if annotations:
            for annotation in annotations:
                if annotation and annotation.contains_point(event):
                    return True
        return False

    @classmethod
    def changed_extent(cls, ax):
        cls.ax = ax
        cls.circle_handles = None
        cls.ax.draggable_annotations = None
        cls.rtree_p = index.Property()
        cls.rtree_idx = index.Index(properties=cls.rtree_p)
        cls.annotated_circles = None
        cls.counter = 0

    @classmethod
    def add_point(cls, coords):
        cls.counter += 1
        cls.rtree_idx.insert(cls.counter, coords)
        return cls.counter

    @classmethod
    def delete_point(cls, rtree_id):
        # past the entire map
        coords = (-1000.0, -1000, 1000.0, 1000.0)
        # need the precise bbox to delete from an item from the rtree
        # find the matching item first
        matching_item = None
        for item in cls.rtree_idx.intersection(coords, objects=True):
            if item.id == rtree_id:
                matching_item = item
                break

        if not matching_item:
            return

        # Delete the item using its own coordinates
        cls.rtree_idx.delete(matching_item.id, matching_item.bbox)

    class AnnotatedCircle:
        def __init__(self, draggable_annotation, circle_handle, rtree_id):
            self.draggable_annotation_object = draggable_annotation
            self.circle_handle_object = circle_handle
            self.visible = True
            # id in the index for rtree_idx
            self.rtree_id = rtree_id

        def annotation_has_focus(self, event):
            if self.draggable_annotation_object:
                return self.draggable_annotation_object.has_focus(event)

        def is_visible(self):
            return self.visible

        def set_visible(self, visibility_target):
            if visibility_target == self.visible:
                return

            if self.draggable_annotation_object:
                try:
                    self.draggable_annotation_object.set_visible(visibility_target)
                except:
                    traceback.print_exc()
                    pass
            if self.circle_handle_object:
                try:
                    self.circle_handle_object.set_visible(visibility_target)
                except:
                    traceback.print_exc()
                    pass

            self.visible = visibility_target

        def remove(self):
            removed = False
            if self.draggable_annotation_object:
                try:
                    self.draggable_annotation_object.remove()
                    removed = True
                except:
                    traceback.print_exc()
                    pass
                #self.annotation_handles = None
                draggable_annotations = AnnotatedCircles.get_draggable_annotations()
                if draggable_annotations:
                    draggable_annotations.remove(self.draggable_annotation_object)
                self.draggable_annotation_object = None
            if self.circle_handle_object:
                try:
                    self.circle_handle_object.remove()
                    removed = True
                except:
                    traceback.print_exc()
                    pass
                circle_handles = AnnotatedCircles.get_circle_handles()
                if circle_handles:
                    circle_handles.remove(self.circle_handle_object)
                self.circle_handle_object = None
            if removed:
                AnnotatedCircles.delete_point(self.rtree_id)

                if AnnotatedCircles.annotated_circles:
                    AnnotatedCircles.annotated_circles.remove(self)

    @classmethod
    def get_draggable_annotations(cls):
        if cls.ax and hasattr(cls.ax, 'draggable_annotations'):
            return cls.ax.draggable_annotations
        return None

    @classmethod
    def get_circle_handles(cls):
        if cls.circle_handles:
            return cls.circle_handles
        return None

    # calculate radius of pixels in degrees
    @classmethod
    def calculate_radius_pixels(cls):
        # Get current extent of the map in degrees and pixels
        extent = cls.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = cls.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # Convert pixels to degrees
        radius_degrees = max(lon_deg_per_pixel, lat_deg_per_pixel) * DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS

        return radius_degrees

    # calculate annotation offset of pixels in degrees (of where to place the annotation next to the circle)
    @classmethod
    def calculate_offset_pixels(cls):
        # Get current extent of the map in degrees and pixels
        extent = cls.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = cls.ax.get_window_extent()
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

    @classmethod
    def add(cls, lat=None, lon=None, label=None, label_color=DEFAULT_ANNOTATE_TEXT_COLOR):
        if lat is None or lon is None or label is None or cls.ax is None:
            return None
        if cls.circle_handles is None:
            cls.circle_handles = []
        if not hasattr(cls.ax, 'draggable_annotations') or cls.ax.draggable_annotations is None:
            cls.ax.draggable_annotations = []

        if cls.has_overlap(lat=lat, lon=lon):
            return None

        lon_offset, lat_offset = cls.calculate_offset_pixels()
        # calculate radius of pixels in degrees
        radius_pixels_degrees = cls.calculate_radius_pixels()
        circle_handle = Circle((lon, lat), radius=radius_pixels_degrees, color=DEFAULT_ANNOTATE_MARKER_COLOR, fill=False, linestyle='dotted', linewidth=2, alpha=0.8,
                                    transform=ccrs.PlateCarree())
        cls.ax.add_patch(circle_handle)
        rtree_id = cls.add_point((lon, lat, lon, lat))
        cls.circle_handles.append(circle_handle)

        bbox_props = {
            'boxstyle': 'round,pad=0.3',
            'edgecolor': '#FFFFFF',
            'facecolor': '#000000',
            'alpha': 1.0
        }

        # Original annotation creation with DraggableAnnotation integration
        annotation_handle = cls.ax.annotate(label, xy=(lon, lat),
                            xytext=(lon + lon_offset, lat + lat_offset),
                            textcoords='data', color=label_color,
                            fontsize=12, ha='left', va='bottom', bbox=bbox_props)

        # Create DraggableAnnotation instance
        draggable_annotation = DraggableAnnotation(
            annotation_handle, (lon, lat), cls.ax, bbox_props)

        annotated_circle = cls.AnnotatedCircle(draggable_annotation, circle_handle, rtree_id)
        if not cls.annotated_circles:
            cls.annotated_circles = []
        cls.annotated_circles.append(annotated_circle)
        # create a way access the annotated_circle from the draggable annotation
        draggable_annotation.set_circle_annotation(annotated_circle)
        cls.ax.draggable_annotations.append(draggable_annotation)

        return annotated_circle
        # draw later as we will likely add multiple circles
        #self.canvas.draw()

    @classmethod
    def has_overlap(cls, lat=None, lon=None):
        if lat is None or lon is None or len(cls.rtree_idx) == 0:
            return False
        # Define a bounding box around the annotated circle for initial query (in degrees)
        buffer = ANNOTATE_CIRCLE_OVERLAP_IN_DEGREES  # Adjust this value based on desired precision
        bounding_box = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        # Query the R-tree for points within the bounding box
        possible_matches = list(cls.rtree_idx.intersection(bounding_box, objects=True))

        if possible_matches:
            return True
        else:
            return False

    @classmethod
    def clear(cls):
        #if self.annotation_handles:
        if not cls.ax or not hasattr(cls.ax, 'draggable_annotations'):
            return
        if cls.ax.draggable_annotations:
            try:
                for annotation in cls.ax.draggable_annotations:
                    annotation.remove()
            except:
                traceback.print_exc()
                pass
            cls.ax.draggable_annotations = None
        if cls.circle_handles:
            try:
                for circle_handle in cls.circle_handles:
                    circle_handle.remove()
            except:
                traceback.print_exc()
                pass
            cls.circle_handles = None
        if cls.rtree_p:
            cls.rtree_p = index.Property()
        if cls.rtree_idx:
            cls.rtree_idx = index.Index(properties=cls.rtree_p)
        if cls.annotated_circles:
            cls.annotated_circles = []

class App:

    def __init__(self, root):
        self.level_vars = None
        self.top_frame = None
        self.tools_frame = None
        self.canvas_frame = None
        self.canvas = None
        self.adeck_mode_frame = None
        self.exit_button_adeck = None
        self.reload_button_adeck = None
        self.label_adeck_mode = None
        self.adeck_selected_combobox = None
        self.switch_to_genesis_button = None
        self.adeck_config_button = None
        self.genesis_mode_frame = None
        self.exit_button_genesis = None
        self.reload_button_genesis = None
        self.label_genesis_mode = None
        self.prev_genesis_cycle_button = None
        self.latest_genesis_cycle_button = None
        self.genesis_models_label = None
        self.switch_to_adeck_button = None
        self.genesis_config_button = None
        self.add_marker_button = None
        self.toggle_selection_loop_button = None
        self.label_mouse_coords_prefix = None
        self.label_mouse_coords = None
        self.label_mouse_hover_info_prefix = None
        self.label_mouse_hover_matches = None
        self.label_mouse_hover_info_coords = None
        self.label_mouse_hover_info_valid_time_prefix = None
        self.label_mouse_hover_info_valid_time = None
        self.label_mouse_hover_info_model_init_prefix = None
        self.label_mouse_hover_info_model_init = None
        self.label_mouse_hover_info_vmax10m_prefix = None
        self.label_mouse_hover_info_vmax10m = None
        self.label_mouse_hover_info_mslp_prefix = None
        self.label_mouse_hover_info_mslp = None
        self.label_mouse_hover_info_roci_prefix = None
        self.label_mouse_hover_info_roci = None
        self.label_mouse_hover_info_isobar_delta_prefix = None
        self.label_mouse_hover_info_isobar_delta = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.axes_size = None
        self.root = root
        self.root.title("tcviewer")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")

        self.global_extent = [-180, 180, -90, 90]

        self.mode = "ADECK"
        self.recent_storms = None
        self.adeck = None
        self.bdeck = None
        self.adeck_selected = tk.StringVar()
        self.adeck_previous_selected = None
        self.adeck_storm = None

        self.genesis_model_cycle_time = None

        self.zoom_selection_box = None
        self.last_cursor_lon_lat = (0.0, 0.0)

        self.lastgl = None

        self.have_deck_data = False
        # track whether there is new tcvitals,adecks,bdecks data
        self.timer_id = None
        self.stale_urls = dict()
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
        # Mapping from rtree point index to (internal_id, tc_index, tc_candidate_point_index)
        self.rtree_tuple_point_id = 0
        self.rtree_tuple_index_mapping = {}

        # manually hidden tc candidates and annotations
        self.hidden_tc_candidates = set()
        self.scatter_objects = {}
        self.line_collection_objects = {}
        self.annotated_circle_objects = {}

        # circle patch for selected marker
        self.circle_handle = None
        self.last_circle_lon = None
        self.last_circle_lat = None

        # track overlapped points (by index, pointing to the plotted_tc_candidates)
        #   this will hold information on the marker where the cursor previously pointed to (current circle patch),
        #   and which one of the possible matches was (is currently) viewed
        self.nearest_point_indices_overlapped = SortedCyclicEnumDict()

        # settings for plotting
        self.time_step_marker_colors = [
            '#ffff00',
            '#ba0a0a', '#e45913', '#fb886e', '#fdd0a2',
            '#005b1c', '#07a10b', '#9cd648', '#a5ee96',
            '#0d3860', '#2155c4', '#33aaff', '#7acaff',
            '#710173', '#b82cae', '#c171cf', '#ffb9ee',
            '#636363', '#969696', '#bfbfbf', '#e9e9e9'
        ]
        self.time_step_legend_fg_colors = [
            '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#ffffff', '#000000', '#000000',
            '#ffffff', '#000000', '#000000', '#000000'
        ]
        # Will change when clicked
        self.time_step_opacity = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ]
        # Define 6 different time_step ranges and their corresponding colors
        self.time_step_ranges = [
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
        self.time_step_legend_objects = []

        # a pair of dicts of selected rvor levels' contours, with contour ids as keys
        self.overlay_rvor_contour_objs = []
        self.overlay_rvor_label_objs = []
        self.overlay_rvor_contour_dict = None
        self.init_rvor_contour_dict()
        self.overlay_rvor_contour_visible = True
        self.overlay_rvor_label_visible = True
        self.overlay_rvor_label_last_alpha = 1.0

        self.load_settings()

        self.root.bind("p", self.take_screenshot)

        self.root.bind("l", self.toggle_rvor_labels)
        self.root.bind("v", self.toggle_rvor_contours)
        #self.root.bind("v", self.toggle_rvor_contours)
        self.rvor_dialog_open = False
        self.root.bind("V", self.show_rvor_dialog)

        self.create_widgets()

        self.measure_tool = MeasureTool(self.ax)
        self.selection_loop_mode = False
        self.display_map()

    def updated_rvor_levels(self):
        # Update the global variable with the new values
        global SELECTED_PRESSURE_LEVELS

        new_SELECTED_PRESSURE_LEVELS = [level for level, var in self.level_vars.items() if var.get()]

        if SELECTED_PRESSURE_LEVELS == new_SELECTED_PRESSURE_LEVELS:
            return

        SELECTED_PRESSURE_LEVELS = new_SELECTED_PRESSURE_LEVELS

        contours = self.overlay_rvor_contour_dict['contour_objs']
        labels = self.overlay_rvor_contour_dict['label_objs']
        for objs_dict in [contours, labels]:
            for _, obj_list in objs_dict.items():
                #for obj in obj_list:
                # don't remove bbox as that will automatically get removed
                try:
                    obj_list[0].remove()
                except:
                    traceback.print_exc()
        self.ax.set_yscale('linear')
        self.display_custom_overlay()
        self.ax.figure.canvas.draw()

    def on_rvor_dialog_close(self, dialog):
        self.rvor_dialog_open = False
        dialog.destroy()
        # focus back on app
        self.set_focus_on_map()

    # noinspection PyUnusedLocal
    def show_rvor_dialog(self, event=None):
        if not self.rvor_dialog_open:
            self.rvor_dialog_open = True
        else:
            return
        # Create the toplevel dialog
        global SELECTED_PRESSURE_LEVELS
        dialog = tk.Toplevel(self.root)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.on_rvor_dialog_close(dialog))
        dialog.title("Selected RVOR levels:")
        height = 300
        width = 250
        dialog.geometry(f"{width}x{height}")  # Set the width to 300 and height to 250
        dialog.geometry(f"+{tk_root.winfo_x() - width // 2 + tk_root.winfo_width() // 2}+{tk_root.winfo_y() - height // 2 + tk_root.winfo_height() // 2}")
        # modal
        dialog.grab_set()

        # Create a frame to hold the checkboxes
        frame = tk.Frame(dialog)
        frame.pack(fill="both", expand=True)

        # Create the checkboxes and their corresponding variables
        self.level_vars = {}
        chk_first = None
        for i, level in enumerate([925, 850, 700, 500, 200], start=1):
            val = 1 if level in SELECTED_PRESSURE_LEVELS else 0
            var = tk.IntVar(value=val)
            chk = tk.Checkbutton(frame, text=f"{level} mb", variable=var, pady = 5)
            var.set(val)
            if not chk_first:
                chk_first = chk
            chk.grid(row=i, column=0, sticky="n")

            # Center the labels
            frame.columnconfigure(0, weight=1)
            chk.columnconfigure(0, weight=1)
            self.level_vars[level] = var


        # Focus on the first checkbox
        frame.focus_set()
        frame.focus()  # Set focus on the frame
        chk_first.focus_set()

        # OK button
        ok_btn = tk.Button(dialog, text="OK", command=lambda: [self.updated_rvor_levels(), self.on_rvor_dialog_close(dialog)])
        ok_btn.pack(fill="x", pady=5)

        # Cancel button
        cancel_btn = tk.Button(dialog, text="Cancel", command=dialog.destroy)
        cancel_btn.config(width=ok_btn.cget("width"))  # Set the width of the Cancel button to match the OK button
        cancel_btn.pack(fill="x", pady=5)

        dialog.bind("<Return>", lambda e: [self.updated_rvor_levels(), self.on_rvor_dialog_close(dialog)])
        dialog.bind("<Escape>", lambda e: self.on_rvor_dialog_close(dialog))

    def rvor_labels_new_extent(self):
        self.update_rvor_contour_renderable_after_zoom()
        extent = self.ax.get_extent(ccrs.PlateCarree())
        contour_visible = self.overlay_rvor_contour_visible
        not_at_global_extent = not (
            extent[0] == self.global_extent[0] and
            extent[1] == self.global_extent[1] and
            extent[2] == self.global_extent[2] and
            extent[3] == self.global_extent[3]
        )
        label_visible = contour_visible and self.overlay_rvor_label_visible and not_at_global_extent
        if label_visible:
            alpha_label_visible = 1.0
        else:
            alpha_label_visible = 0.0
        #if self.overlay_rvor_label_last_alpha != alpha_label_visible:
        try:
            renderable_ids = self.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, obj_list in self.overlay_rvor_contour_dict['label_objs'].items():
                # limit detail for labels based on zoom extent
                is_renderable = (contour_id in renderable_ids)
                is_visible = label_visible and is_renderable
                for obj in obj_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass

        self.overlay_rvor_label_last_alpha = alpha_label_visible

    # noinspection PyUnusedLocal
    def toggle_rvor_contours(self, *args):
        new_vis = not self.overlay_rvor_contour_visible
        try:
            renderable_ids = self.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, objs_list in self.overlay_rvor_contour_dict['contour_objs'].items():
                # is_renderable = (contour_id in renderable_ids)
                # is_visible = new_vis and is_renderable
                # don't limit contour detail by extent (only limit labels)
                is_visible = new_vis
                for obj in objs_list:
                    obj.set_visible(is_visible)
            self.overlay_rvor_label_visible = new_vis
            for contour_id, objs_list in self.overlay_rvor_contour_dict['label_objs'].items():
                is_renderable = (contour_id in renderable_ids)
                is_visible = new_vis and is_renderable
                for obj in objs_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass
        self.overlay_rvor_contour_visible = new_vis
        self.ax.set_yscale('linear')
        self.ax.figure.canvas.draw()

    # noinspection PyUnusedLocal
    def toggle_rvor_labels(self, *args):
        new_vis = not self.overlay_rvor_label_visible
        if new_vis:
            new_alpha = 1.0
        else:
            new_alpha = 0.0
        try:
            renderable_ids = self.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, obj_list in self.overlay_rvor_contour_dict['label_objs'].items():
                is_renderable = (contour_id in renderable_ids)
                is_visible = new_vis and is_renderable
                for obj in obj_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass
        self.overlay_rvor_label_last_alpha = new_alpha
        self.overlay_rvor_label_visible = new_vis
        # fixes bug in matplotlib after modifying many artists (especially test boxes)
        self.ax.set_yscale('linear')
        self.ax.figure.canvas.draw()

    def set_focus_on_map(self):
        self.canvas.get_tk_widget().focus_set()

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

    def update_plotted_list(self, internal_id, tc_candidate):
        # zero indexed
        tc_index = len(self.plotted_tc_candidates)
        for point_index, point in enumerate(tc_candidate):  # Iterate over each point in the track
            lat, lon = point['lat'], point['lon']
            # Can't use a tuple (tc_index, point_index) as the index so use a mapped index
            self.rtree_idx.insert(self.rtree_tuple_point_id, (lon, lat, lon, lat))
            self.rtree_tuple_index_mapping[self.rtree_tuple_point_id] = (internal_id, tc_index, point_index)
            self.rtree_tuple_point_id += 1

        self.plotted_tc_candidates.append((internal_id, tc_candidate))

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
                internal_id, tc_candidate = self.plotted_tc_candidates[tc_index]
                # dict at a point in time for tc candidate
                tc_candidate_point = tc_candidate[tc_point_index]
                model_name = tc_candidate_point['model_name']
                lat = tc_candidate_point['lat']
                lon = tc_candidate_point['lon']
                if 'valid_time' in tc_candidate_point:
                    valid_time = tc_candidate_point['valid_time'].strftime('%Y-%m-%d %HZ')
                else:
                    valid_time = None
                if 'init_time' in tc_candidate_point:
                    init_time = tc_candidate_point['init_time'].strftime('%Y-%m-%d %HZ')
                else:
                    init_time = None
                vmax10m = None
                if 'vmax10m_in_roci' in tc_candidate_point and tc_candidate_point['vmax10m_in_roci']:
                    vmax10m = tc_candidate_point['vmax10m_in_roci']
                elif 'vmax10m' in tc_candidate_point and tc_candidate_point['vmax10m']:
                    vmax10m = tc_candidate_point['vmax10m']
                mslp = tc_candidate_point['mslp_value']
                roci = tc_candidate_point['roci']
                isobar_delta = None
                if 'closed_isobar_delta' in tc_candidate_point:
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
                    if model_name == "TCVITALS" and valid_time:
                        self.label_mouse_hover_info_model_init.config(text=f"{model_name:>4} {valid_time}")
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

    # noinspection GrazieInspection
    def update_labels_for_mouse_hover(self, lat=None, lon=None):
        if not(lat) or not(lon):
            return

        # Update label for mouse cursor position on map first
        self.label_mouse_coords.config(text=f"({lat:>8.4f}, {lon:>9.4f})")

        if EventManager.get_blocking_purpose():
            # blocking for zoom, measure
            return

        # Next, find nearest point (within some bounding box, as we want to be selective)
        # Define a bounding box around the cursor for initial query (in degrees)
        buffer = MOUSE_SELECT_IN_DEGREES  # Adjust this value based on desired precision
        bounding_box = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        # Query the R-tree for points within the bounding box
        possible_matches = list(self.rtree_idx.intersection(bounding_box, objects=True))

        # Calculate the geodesic distance and find the nearest point (nearest_point_index)
        min_distance = float('inf')
        # a sorted cyclic dict that has the item number enumerated
        # has a get() the next enumerated number and the key (the point_index tuple) in a cycle (sorted by value, which will be a datetime)
        self.nearest_point_indices_overlapped = SortedCyclicEnumDict()
        for item in possible_matches:
            unmapped_point_index = item.id
            internal_id, tc_index, point_index = self.rtree_tuple_index_mapping[unmapped_point_index]
            point = self.plotted_tc_candidates[tc_index][1][point_index]
            item_is_overlapping = False
            if internal_id in self.hidden_tc_candidates:
                continue
            if len(self.nearest_point_indices_overlapped):
                overlapping_internal_id, overlapping_tc_index, overlapping_point_index = self.nearest_point_indices_overlapped.get_first_key()
                possible_overlapping_point = self.plotted_tc_candidates[overlapping_tc_index][1][overlapping_point_index]
                lon_diff = round(abs(possible_overlapping_point['lon'] - point['lon']), 3)
                lat_diff = round(abs(possible_overlapping_point['lat'] - point['lat']), 3)
                if lon_diff == 0.0 and lat_diff == 0.0:
                    item_is_overlapping = True

            # check to see if it is an almost exact match (~3 decimals in degrees) to approximate whether it is an overlapped point
            if item_is_overlapping:
                # this will likely be an overlapped point in the grid
                self.nearest_point_indices_overlapped[(internal_id, tc_index, point_index)] = point['valid_time']
                # min distance should not significantly change (we are using the first point as reference for overlapping)
            else:
                distance = self.calculate_distance((lon, lat), (point['lon'], point['lat']))
                if distance < min_distance:
                    # not an overlapping point but still closer to cursor, so update
                    # first clear any other points since this candidate is closer and does not have an overlapping point
                    self.nearest_point_indices_overlapped = SortedCyclicEnumDict()
                    self.nearest_point_indices_overlapped[(internal_id, tc_index, point_index)] = point['valid_time']
                    min_distance = distance

        # Update the labels if a nearest point is found within the threshold
        total_num_overlapped_points = len(self.nearest_point_indices_overlapped)
        if total_num_overlapped_points > 0:
            overlapped_point_num, nearest_point_index = self.nearest_point_indices_overlapped.next_enum_key_tuple()
            internal_id, tc_index, point_index = nearest_point_index
            self.update_tc_status_labels(tc_index, point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = self.plotted_tc_candidates[tc_index][1][point_index]
            lon = point['lon']
            lat = point['lat']
            self.update_circle_patch(lon=lon, lat=lat)
        else:
            # clear the label if no point is found? No.
            #   Not only will this prevent the constant reconfiguring of labels, it allows the user more flexibility
            # self.update_tc_status_labels()
            # Do clear the circle though as it might be obtrusive
            self.clear_circle_patch()

    def cycle_to_next_overlapped_point(self):
        # called when user hovers on overlapped points and hits the TAB key
        total_num_overlapped_points = len(self.nearest_point_indices_overlapped)
        if total_num_overlapped_points > 1:
            overlapped_point_num, nearest_point_index = self.nearest_point_indices_overlapped.next_enum_key_tuple()
            internal_id, tc_index, point_index = nearest_point_index
            self.update_tc_status_labels(tc_index, point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = self.plotted_tc_candidates[tc_index][1][point_index]
            lon = point['lon']
            lat = point['lat']
            self.update_circle_patch(lon=lon, lat=lat)

    # hide selected, or show all if none selected
    def hide_tc_candidate(self):
        total_num_overlapped_points = len(self.nearest_point_indices_overlapped)
        if total_num_overlapped_points == 0:
            # unhide all
            if len(self.hidden_tc_candidates) > 0:
                if self.scatter_objects:
                    for internal_id, scatters in self.scatter_objects.items():
                        try:
                            for scatter in scatters:
                                scatter.set_visible(True)
                        except:
                            traceback.print_exc()

                if self.line_collection_objects:
                    for internal_id, line_collections in self.line_collection_objects.items():
                        try:
                            for line_collection in line_collections:
                                line_collection.set_visible(True)
                        except:
                            traceback.print_exc()

                if self.annotated_circle_objects:
                    for internal_id, annotated_circles in self.annotated_circle_objects.items():
                        try:
                            for annotated_circle in annotated_circles:
                                if annotated_circle:
                                    annotated_circle.set_visible(True)
                        except:
                            traceback.print_exc()

                self.hidden_tc_candidates = set()
                (lon,lat) = self.last_cursor_lon_lat
                self.update_labels_for_mouse_hover(lat=lat, lon=lon)
                self.ax.figure.canvas.draw()
        else:
            num, cursor_point_index = self.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            if cursor_point_index:
                cursor_internal_id, tc_index, tc_point_index = cursor_point_index
                self.hidden_tc_candidates.add(cursor_internal_id)

                if self.scatter_objects:
                    for internal_id, scatters in self.scatter_objects.items():
                        if cursor_internal_id == internal_id:
                            try:
                                for scatter in scatters:
                                    scatter.set_visible(False)
                            except:
                                traceback.print_exc()

                if self.line_collection_objects:
                    for internal_id, line_collections in self.line_collection_objects.items():
                        if cursor_internal_id == internal_id:
                            try:
                                for line_collection in line_collections:
                                    line_collection.set_visible(False)
                            except:
                                traceback.print_exc()

                if self.annotated_circle_objects:
                    for internal_id, annotated_circles in self.annotated_circle_objects.items():
                        if cursor_internal_id == internal_id:
                            try:
                                for annotated_circle in annotated_circles:
                                    annotated_circle.set_visible(False)
                            except:
                                traceback.print_exc()

                (lon,lat) = self.last_cursor_lon_lat
                self.update_labels_for_mouse_hover(lat=lat, lon=lon)
                self.ax.figure.canvas.draw()

    @staticmethod
    def clear_storm_extrema_annotations():
        AnnotatedCircles.clear()

    def any_storm_points_in_bounds(self, tc_index):
        if not self.plotted_tc_candidates:
            return False
        if (tc_index + 1) > len(self.plotted_tc_candidates):
            return False
        if not self.ax:
            return False

        any_in_bound = False

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        try:
            internal_id, tc_candidate = self.plotted_tc_candidates[tc_index]
            for point in tc_candidate:
                if len(self.hidden_tc_candidates) == 0 or internal_id not in self.hidden_tc_candidates:
                    lat = point['lat']
                    lon = point['lon']
                    any_in_bound = any_in_bound or (xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1])
        except:
            traceback.print_exc()
            pass

        return any_in_bound

    def annotate_storm_extrema(self):
        if len(self.plotted_tc_candidates) == 0:
            return

        # note: point_index is a tuple of tc_index, tc_point_index
        if len(self.nearest_point_indices_overlapped) == 0:
            # annotate all storm extrema in current view
            for tc_index in range(len(self.plotted_tc_candidates)):
                internal_id, tc_candidate = self.plotted_tc_candidates[tc_index]
                if len(tc_candidate):
                    if self.any_storm_points_in_bounds(tc_index):
                        point_index = (internal_id, tc_index, 0)
                        self.annotate_single_storm_extrema(point_index=point_index)
        else:
            # annotate storm extrema of previously selected
            num, cursor_point_index = self.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            self.annotate_single_storm_extrema(point_index=cursor_point_index)

    def annotate_single_storm_extrema(self, point_index = None):
        global ANNOTATE_COLOR_LEVELS
        if point_index is None or len(point_index) != 3:
            return
        internal_id, tc_index, tc_point_index = point_index
        if not self.plotted_tc_candidates or (tc_index + 1) > len(self.plotted_tc_candidates):
            return

        results = {}
        for short_name in DISPLAYED_FUNCTIONAL_ANNOTATIONS:
            result_tuple = annotations_result_val[short_name](self.plotted_tc_candidates[tc_index][1])
            if not result_tuple:
                continue
            point_idx, result_val = result_tuple
            if point_idx is None:
                continue
            if result_val is None:
                continue
            if point_idx not in results:
                results[point_idx] = []
            results[point_idx].append((short_name, result_val))

        if not results:
            return

        # annotate the extrema for the storm
        point_index_labels = {}
        # since some extremum may show up for the same point, we need to combine the extremum labels first (by point_index)
        for result_idx in sorted(results.keys()):
            if result_idx is None:
                continue
            for results_tuple in results[result_idx]:
                if results_tuple is None:
                    continue
                short_name, result_val = results_tuple
                if result_val is None or short_name is None:
                    continue

                result_point = self.plotted_tc_candidates[tc_index][1][result_idx]
                append = False
                if result_idx in point_index_labels:
                    append = True

                label_str = annotations_label_func_dict[short_name](result_point, result_val)
                new_color_level = annotations_color_level[short_name]

                if label_str:
                    if append:
                        prev_lines, prev_color_level = point_index_labels[result_idx]
                        if new_color_level > prev_color_level:
                            lines_color_level = new_color_level
                        else:
                            lines_color_level = prev_color_level

                        # remove any duplicates in the new string to be appended
                        prev_lines = prev_lines.splitlines()
                        new_lines = label_str.splitlines()
                        new_lines = [line for line in new_lines if line not in prev_lines]
                        prev_lines.extend(new_lines)
                        appended_str = "\n".join(prev_lines)

                        point_index_labels[result_idx] = (appended_str, lines_color_level)
                    else:
                        prev_color_level = new_color_level
                        point_index_labels[result_idx] = (label_str, prev_color_level)

        # finally add the annotated circle for each label
        added = False
        for label_point_index, (point_label, color_level) in point_index_labels.items():
            point = self.plotted_tc_candidates[tc_index][1][label_point_index]
            added = True
            # check if already annotated
            #   there is a question of how we want to use annotations (one or many per (nearby or same) point?)
            #   in AnnotatedCircles, we use has_overlap() to prevent that
            annotated_circle = AnnotatedCircles.add(lat=point['lat'], lon=point['lon'], label=point_label, label_color=ANNOTATE_COLOR_LEVELS[color_level])
            # handle case to make sure we don't add doubles or nearby
            if annotated_circle is None:
                continue
            if internal_id not in self.annotated_circle_objects:
                self.annotated_circle_objects[internal_id] = []
            self.annotated_circle_objects[internal_id].append(annotated_circle)

        if added:
            self.ax.figure.canvas.draw()

    def display_custom_boundaries(self, label_column=None):
        if custom_gdf is not None:
            for geometry in custom_gdf.geometry:
                if isinstance(geometry, Polygon):
                    self.ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none', linewidth=2)
                else:
                    self.ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none', linewidth=2)

        if label_column:
            for idx, row in custom_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                self.ax.text(x, y, row[label_column], transform=ccrs.PlateCarree(), fontsize=8, color='magenta')

    def init_rvor_contour_dict(self):
        self.overlay_rvor_contour_dict = defaultdict(dict)
        # noinspection PyTypeChecker
        self.overlay_rvor_contour_dict['ids'] = set()
        # noinspection PyTypeChecker
        self.overlay_rvor_contour_dict['renderable_ids'] = set()
        self.overlay_rvor_contour_dict['contour_span_lons'] = {}
        self.overlay_rvor_contour_dict['contour_span_lats'] = {}
        self.overlay_rvor_contour_dict['contour_objs'] = defaultdict(dict)
        self.overlay_rvor_contour_dict['label_objs'] = defaultdict(dict)

    # display custom rvor overlay from shape files
    def display_custom_overlay(self):
        if not overlay_gdfs:
            return
        self.init_rvor_contour_dict()
        global RVOR_CYCLONIC_CONTOURS
        global RVOR_CYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_CONTOURS
        global SELECTED_PRESSURE_LEVELS
        if SELECTED_PRESSURE_LEVELS == []:
            return

        extent, min_span_lat_deg, min_span_lon_deg = self.get_contour_min_span_deg()

        do_overlay_shapes = {
            'rvor_c_poly': RVOR_CYCLONIC_CONTOURS,
            'rvor_c_points': RVOR_CYCLONIC_LABELS,
            'rvor_ac_poly': RVOR_ANTICYCLONIC_CONTOURS,
            'rvor_ac_points': RVOR_ANTICYCLONIC_LABELS
        }

        renderable_ids = self.overlay_rvor_contour_dict['renderable_ids']
        for gdf_name, do_overlay in do_overlay_shapes.items():
            if not do_overlay:
                continue
            all_levels_gdf = overlay_gdfs.get(gdf_name, None)
            if all_levels_gdf is None:
                continue

            # noinspection PyUnresolvedReferences
            gdf2 = all_levels_gdf[all_levels_gdf['level'].isin(SELECTED_PRESSURE_LEVELS)]

            # Filter contours based on span
            for _, row in gdf2.iterrows():
                contour_id = row['contour_id']
                span_lon = row['span_lon']
                span_lat = row['span_lat']
                if span_lon >= min_span_lon_deg and span_lat >= min_span_lat_deg:
                    renderable_ids.add(contour_id)
                self.overlay_rvor_contour_dict['ids'].add(contour_id)
                self.overlay_rvor_contour_dict['contour_span_lons'][contour_id] = span_lon
                self.overlay_rvor_contour_dict['contour_span_lats'][contour_id] = span_lat

        # Draw contours and labels based on the filtered IDs
        for gdf_name, do_overlay in do_overlay_shapes.items():
            if not do_overlay:
                continue
            all_levels_gdf = overlay_gdfs.get(gdf_name, None)
            if all_levels_gdf is None:
                continue

            cyclonic = (gdf_name[5] == 'c')

            # requiring that changing pressure levels to remove and re-add all the artists
            # noinspection PyUnresolvedReferences
            gdf3 = all_levels_gdf[all_levels_gdf['level'].isin(SELECTED_PRESSURE_LEVELS)]

            # global visibility based on current display settings (keyboard shortcuts) and if global extent or not
            contour_visible = self.overlay_rvor_contour_visible
            not_at_global_extent = not (
                extent[0] == self.global_extent[0] and
                extent[1] == self.global_extent[1] and
                extent[2] == self.global_extent[2] and
                extent[3] == self.global_extent[3]
            )
            label_visible = contour_visible and self.overlay_rvor_label_visible and not_at_global_extent
            if label_visible:
                alpha_label_visible = 1.0
            else:
                alpha_label_visible = 0.0
            self.overlay_rvor_label_last_alpha = alpha_label_visible

            if cyclonic:
                edge_colors = CYCLONIC_PRESSURE_LEVEL_COLORS
            else:
                edge_colors = ANTI_CYCLONIC_PRESSURE_LEVEL_COLORS

            for _, row in gdf3.iterrows():
                contour_id = row['contour_id']
                # is renderable based on the level of detail (size of contour relative to current extent)
                is_renderable = contour_id in renderable_ids
                geom = row['geometry']
                if isinstance(geom, Polygon):
                    # Draw contours
                    edge_color = edge_colors[str(row['level'])]
                    #is_visible = contour_visible and is_renderable
                    # only limit labels to detail
                    is_visible = contour_visible
                    obj = self.ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor=edge_color, facecolor='none', linewidth=2, visible=is_visible)
                    self.overlay_rvor_contour_dict['contour_objs'][contour_id] = [obj]
                else:
                    # Draw labels
                    edge_color = edge_colors[str(row['level'])]
                    is_visible = label_visible and is_renderable
                    x, y = geom.x, geom.y
                    label = row['label']
                    #obj = self.ax.text(x, y, label, transform=ccrs.PlateCarree(), color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor=edge_color, edgecolor='black', pad=2), alpha=alpha_label_visible)
                    obj = self.ax.text(x, y, label, transform=ccrs.PlateCarree(), color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor=edge_color, edgecolor='black', pad=2), visible=is_visible)
                    obj_bbox = obj.get_bbox_patch()
                    #obj_bbox.set_alpha(alpha_label_visible)
                    obj_bbox.set_visible(is_visible)
                    self.overlay_rvor_contour_dict['label_objs'][contour_id] = [obj, obj_bbox]

        self.ax.set_yscale('linear')

    def update_rvor_contour_renderable_after_zoom(self):
        if self.overlay_rvor_contour_dict and 'ids' in self.overlay_rvor_contour_dict:
            extent, min_span_lat_deg, min_span_lon_deg = self.get_contour_min_span_deg()
            renderable_ids = set()
            span_lons = self.overlay_rvor_contour_dict['contour_span_lons']
            span_lats = self.overlay_rvor_contour_dict['contour_span_lats']
            for contour_id in self.overlay_rvor_contour_dict['ids']:
                span_lon = span_lons[contour_id]
                span_lat = span_lats[contour_id]
                if span_lon >= min_span_lon_deg and span_lat >= min_span_lat_deg:
                    renderable_ids.add(contour_id)
            self.overlay_rvor_contour_dict['renderable_ids'] = renderable_ids

    def get_contour_min_span_deg(self):
        global MINIMUM_CONTOUR_PX_X
        global MINIMUM_CONTOUR_PX_Y
        # Get the current extent in degrees
        extent = self.ax.get_extent(ccrs.PlateCarree())
        min_lon, max_lon, min_lat, max_lat = extent
        span_lon_extent = max_lon - min_lon
        span_lat_extent = max_lat - min_lat
        # Determine the minimum span in degrees for display
        ax_width, ax_height = self.ax.get_figure().get_size_inches() * self.ax.get_figure().dpi
        lon_pixel_ratio = span_lon_extent / ax_width
        lat_pixel_ratio = span_lat_extent / ax_height
        minimum_contour_extent = [MINIMUM_CONTOUR_PX_X, MINIMUM_CONTOUR_PX_Y]  # Minimum in pixels for lon, lat
        min_span_lon_deg = minimum_contour_extent[0] * lon_pixel_ratio
        min_span_lat_deg = minimum_contour_extent[1] * lat_pixel_ratio
        return extent, min_span_lat_deg, min_span_lon_deg

    # noinspection PyUnusedLocal
    def combo_selected_models_event(self, event):
        current_value = self.adeck_selected_combobox.get()
        if current_value == self.adeck_previous_selected:
            # user did not change selection
            self.set_focus_on_map()
            return
        else:
            self.adeck_previous_selected = current_value
            self.display_map()
            if not self.have_deck_data:
                self.update_deck_data()
            self.hidden_tc_candidates = set()
            self.display_deck_data()
            self.set_focus_on_map()

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
        self.canvas_frame.pack(fill=tk.X, expand=True, anchor=tk.NW)

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

        self.adeck_config_button = ttk.Button(self.adeck_mode_frame, text="CONFIG \u2699", command=self.show_config_adeck_dialog, style="TButton")
        self.adeck_config_button.pack(side=tk.RIGHT, padx=5, pady=5)

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

        self.genesis_config_button = ttk.Button(self.genesis_mode_frame, text="CONFIG \u2699", command=self.show_config_genesis_dialog, style="TButton")
        self.genesis_config_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_tools_widgets(self):
        #self.tools_frame = ttk.Frame(self.tools_frame, style="Tools.TFrame")

        self.toggle_selection_loop_button = ttk.Button(self.tools_frame, text="\u27B0 SELECT", command=self.toggle_selection_loop_mode, style="TButton")
        self.toggle_selection_loop_button.pack(side=tk.LEFT, padx=5, pady=5)

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
        self.set_focus_on_map()

    def update_mode(self):
        if self.mode == "ADECK":
            self.genesis_mode_frame.pack_forget()
            self.adeck_mode_frame.pack(side=tk.TOP, fill=tk.X)
        else:
            self.adeck_mode_frame.pack_forget()
            self.genesis_mode_frame.pack(side=tk.TOP, fill=tk.X)

    def update_toggle_selection_loop_button_color(self):
        if self.selection_loop_mode:
            self.toggle_selection_loop_button.configure(style='YellowAndBorder.TButton')
        else:
            self.toggle_selection_loop_button.configure(style='WhiteAndBorder.TButton')

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
        # unfortunately we need to get all each type (vitals, adeck, bdeck) from both mirrors, since
        # it's unknown which mirror actually has most up-to-date data from the modification date alone
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
                        self.stale_urls['tcvitals'] = self.stale_urls['tcvitals'] | {url}
        if self.dt_mods_adeck:
            for url, old_dt_mod in self.dt_mods_adeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        self.stale_urls['adeck'] = self.stale_urls['adeck'] | {url}
        if self.dt_mods_bdeck:
            for url, old_dt_mod in self.dt_mods_bdeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        self.stale_urls['bdeck'] = self.stale_urls['bdeck'] | {url}

        self.update_reload_button_color()

    def reload_adeck(self):
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, self.check_for_stale_data)

        self.reload()
        self.set_focus_on_map()

    def reload(self):
        if self.mode == "ADECK":
            self.update_deck_data()
            self.redraw_map_with_data()
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
                self.redraw_map_with_data(model_cycle=model_cycle)

    def redraw_map_with_data(self, model_cycle = None):
        self.hidden_tc_candidates = set()
        self.display_map()
        if self.mode == "GENESIS" and model_cycle:
            self.update_genesis(model_cycle)
        elif self.mode == "ADECK":
            self.display_deck_data()

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
        if self.adeck and self.adeck.keys():
            for storm_atcf_id in self.adeck.keys():
                if storm_atcf_id in self.adeck and self.adeck[storm_atcf_id]:
                    for model_id, models in self.adeck[storm_atcf_id].items():
                        if models:
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

        if self.bdeck:
            for storm_atcf_id in self.bdeck.keys():
                if storm_atcf_id in self.bdeck[storm_atcf_id] and self.bdeck[storm_atcf_id]:
                    for model_id, models in self.bdeck[storm_atcf_id].items():
                        if models:
                            if model_id in selected_models:
                                if storm_atcf_id not in selected_model_data.keys():
                                    selected_model_data[storm_atcf_id] = {}
                                selected_model_data[storm_atcf_id][model_id] = models

        # tcvitals
        if self.recent_storms:
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

        self.clear_plotted_list()
        numc = 0
        for storm_atcf_id, tc in tc_candidates.items():
            numc += 1
            numd = 0
            for model_name, disturbance_candidates in tc.items():
                numd += 1
                if disturbance_candidates:
                    prev_lat = None
                    prev_lon = None
                    numdisturb = len(disturbance_candidates)
                    # check if it should be hidden
                    internal_id = (numc, numd)
                    if numdisturb > 0:
                        #TODO conditionally hidden by interface (first forecast valid time > user selected horizon (latest first valid time))
                        #valid_time_str = disturbance_candidates[0][1]['valid_time']
                        #valid_time_datetime.fromisoformat(valid_time_str)

                        # check for manually hidden
                        if len(self.hidden_tc_candidates) != 0 and internal_id in self.hidden_tc_candidates:
                            continue

                    lat_lon_with_time_step_list = []
                    # handle when we are hiding certain time steps or whether the storm itself should be hidden based on time step
                    have_displayed_points = False
                    for valid_time, candidate in disturbance_candidates.items():
                        if 'time_step' in candidate.keys():
                            time_step_int = candidate['time_step']
                        else:
                            time_step_int = 0
                        lon = candidate['lon']
                        lat = candidate['lat']
                        candidate_info = dict()
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
                            candidate_info['vmax10m'] = candidate['vmax10m']
                        else:
                            candidate_info['vmax10m_in_roci'] = None
                            candidate_info['vmax10m'] = None
                        if 'closed_isobar_delta' in candidate and candidate['closed_isobar_delta']:
                            candidate_info['closed_isobar_delta'] = candidate['closed_isobar_delta']
                        else:
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

                        # check whether we want to display it (first valid time limit)
                        for i, (start, end) in list(enumerate(self.time_step_ranges)):
                            hours_after = candidate_info['hours_after_valid_day']
                            if start <= hours_after <= end:
                                if self.time_step_opacity[i] == 1.0:
                                    lat_lon_with_time_step_list.append(candidate_info)
                                    have_displayed_points = True
                                elif have_displayed_points and self.time_step_opacity[i] == 0.6:
                                    lat_lon_with_time_step_list.append(candidate_info)
                                else:
                                    # opacity == 0.3 case (hide all points beyond legend valid time)
                                    break

                    if lat_lon_with_time_step_list:
                        self.update_plotted_list(internal_id, lat_lon_with_time_step_list)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(self.time_step_ranges))):
                        opacity = 1.0
                        lons = {}
                        lats = {}
                        for point in reversed(lat_lon_with_time_step_list):
                            hours_after = point['hours_after_valid_day']
                            #if start <= time_step <= end:
                            # use hours after valid_day instead
                            if start <= hours_after <= end:
                                if point['vmax10m_in_roci']:
                                    marker = '*'
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
                            scatter = self.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker, facecolors='none', edgecolors=self.time_step_marker_colors[i], s=marker_sizes[vmaxmarker]**2, alpha=opacity, antialiased=False)
                            if internal_id not in self.scatter_objects:
                                self.scatter_objects[internal_id] = []
                            self.scatter_objects[internal_id].append(scatter)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(self.time_step_ranges))):
                        line_color = self.time_step_marker_colors[i]
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
                        lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                        # Add the LineCollection to the axes
                        line_collection = self.ax.add_collection(lc)
                        if internal_id not in self.line_collection_objects:
                            self.line_collection_objects[internal_id] = []
                        self.line_collection_objects[internal_id].append(line_collection)

        labels_positive = [f' D+{str(i): >2} ' for i in range(len(self.time_step_marker_colors)-1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        self.time_step_legend_objects = []

        for i, (color, label) in enumerate(zip(reversed(self.time_step_marker_colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i*20
            time_step_opacity = list(reversed(self.time_step_opacity))[i]
            if time_step_opacity == 1.0:
                edgecolor = "#FFFFFF"
            elif time_step_opacity == 0.6:
                edgecolor = "#FF77B0"
            else:
                edgecolor = "#A63579"
            legend_object = self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color=list(reversed(self.time_step_legend_fg_colors))[i],
                        fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor=edgecolor, facecolor=color, alpha=1.0))
            self.time_step_legend_objects.append(legend_object)

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i*35
            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                        fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor='#000000', alpha=1.0))

        self.ax.figure.canvas.draw()
        self.label_adeck_mode.config(text=f"ADECK MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime('%Y-%m-%d') + f". Models: {num_models}/{num_all_models}")

    def latest_genesis_cycle(self):
        model_cycles = get_tc_model_init_times_relative_to(datetime.now())
        if model_cycles['next'] is None:
            model_cycle = model_cycles['at']
        else:
            model_cycle = model_cycles['next']

        if model_cycle:
            # clear map
            self.redraw_map_with_data(model_cycle=model_cycle)

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
                self.redraw_map_with_data(model_cycle=model_cycle)

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
                self.redraw_map_with_data(model_cycle=model_cycle)

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

        self.clear_plotted_list()
        numc = 0
        for tc in tc_candidates:
            numc += 1

            model_name = tc['model_name']
            model_timestamp = tc['model_timestamp']
            disturbance_candidates = tc['disturbance_candidates']
            if not most_recent_timestamp:
                most_recent_timestamp = model_timestamp
                model_dates[model_name] = datetime.fromisoformat(most_recent_timestamp).strftime('%d/%HZ')

            if datetime.fromisoformat(most_recent_timestamp) < datetime.fromisoformat(model_timestamp):
                most_recent_timestamp = model_timestamp
                model_dates[model_name] = datetime.fromisoformat(most_recent_timestamp).strftime('%d/%HZ')

            numdisturb = 0
            prev_lat = None
            prev_lon = None
            if disturbance_candidates:
                numdisturb = len(disturbance_candidates)

            # check if it should be hidden
            if numdisturb > 0:
                #TODO conditionally hidden by interface (first forecast valid time > user selected horizon (latest first valid time))
                #valid_time_str = disturbance_candidates[0][1]['valid_time']
                #valid_time_datetime.fromisoformat(valid_time_str)

                # check for manually hidden
                internal_id = numc
                if len(self.hidden_tc_candidates) != 0 and internal_id in self.hidden_tc_candidates:
                    continue

                lat_lon_with_time_step_list = []
                # handle when we are hiding certain time steps or whether the storm itself should be hidden based on time step
                have_displayed_points = False
                for time_step_str, valid_time_str, candidate in disturbance_candidates:
                    time_step_int = int(time_step_str)
                    lon = candidate['lon']
                    candidate_info = dict()
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
                    candidate_info['vmax10m'] = vmaxkt
                    candidate_info['mslp_value'] = candidate['mslp_value']
                    candidate_info['closed_isobar_delta'] = candidate['closed_isobar_delta']

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

                    # check whether we want to display it (first valid time limit)
                    for i, (start, end) in list(enumerate(self.time_step_ranges)):
                        hours_after = candidate_info['hours_after_valid_day']
                        if start <= hours_after <= end:
                            if self.time_step_opacity[i] == 1.0:
                                lat_lon_with_time_step_list.append(candidate_info)
                                have_displayed_points = True
                            elif have_displayed_points and self.time_step_opacity[i] == 0.6:
                                lat_lon_with_time_step_list.append(candidate_info)
                            else:
                                # opacity == 0.3 case (hide all points beyond legend valid time)
                                break

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

                if lat_lon_with_time_step_list:
                    self.update_plotted_list(internal_id, lat_lon_with_time_step_list)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(self.time_step_ranges))):
                    opacity = 1.0
                    lons = {}
                    lats = {}
                    for point in reversed(lat_lon_with_time_step_list):
                        hours_after = point['hours_after_valid_day']
                        #if start <= time_step <= end:
                        # use hours after valid_day instead
                        if start <= hours_after <= end:
                            marker = "*"
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
                        scatter = self.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker, facecolors='none', edgecolors=self.time_step_marker_colors[i], s=marker_sizes[vmaxmarker]**2, alpha=opacity, antialiased=False)
                        if internal_id not in self.scatter_objects:
                            self.scatter_objects[internal_id] = []
                        self.scatter_objects[internal_id].append(scatter)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(self.time_step_ranges))):
                    line_color = self.time_step_marker_colors[i]
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
                    lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                    # Add the LineCollection to the axes
                    line_collection = self.ax.add_collection(lc)
                    if internal_id not in self.line_collection_objects:
                        self.line_collection_objects[internal_id] = []
                    self.line_collection_objects[internal_id].append(line_collection)

        labels_positive = [f' D+{str(i): >2} ' for i in range(len(self.time_step_marker_colors)-1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        self.time_step_legend_objects = []

        for i, (color, label) in enumerate(zip(reversed(self.time_step_marker_colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i*20

            time_step_opacity = list(reversed(self.time_step_opacity))[i]
            if time_step_opacity == 1.0:
                edgecolor = "#FFFFFF"
            elif time_step_opacity == 0.6:
                edgecolor = "#FF77B0"
            else:
                edgecolor = "#A63579"
            legend_object = self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color=list(reversed(self.time_step_legend_fg_colors))[i],
                        fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor=edgecolor, facecolor=color, alpha=1.0))
            self.time_step_legend_objects.append(legend_object)

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i*35
            self.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                        fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                      edgecolor='#FFFFFF', facecolor='#000000', alpha=1.0))

        self.ax.figure.canvas.draw()
        self.label_genesis_mode.config(text="GENESIS MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime('%Y-%m-%d'))
        self.genesis_model_cycle_time = most_recent_model_cycle
        self.genesis_models_label.config(text=f"Latest models: GFS [{model_dates['GFS']}], ECM[{model_dates['ECM']}], NAV[{model_dates['NAV']}], CMC[{model_dates['CMC']}]")

    def update_axes(self):
        gl = self.lastgl

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
        self.ax.set_yscale('linear')

        gl = self.ax.gridlines(draw_labels=["bottom", "left"], x_inline=False, y_inline=False, auto_inline=False, color='white', alpha=0.5, linestyle='--')
        # Move axis labels inside the subplot
        self.ax.tick_params(axis='both', direction='in', labelsize=16)
        # https://github.com/SciTools/cartopy/issues/1642
        gl.xpadding = -10     # Ideally, this would move labels inside the map, but results in hidden labels
        gl.ypadding = -10     # Ideally, this would move labels inside the map, but results in hidden labels

        gl.xlabel_style = {'color': 'orange'}
        gl.ylabel_style = {'color': 'orange'}

        # in pixels
        window_extent = self.ax.get_window_extent()
        # in degrees
        extent = self.ax.get_extent()

        lon_pixels = window_extent.width
        lat_pixels = window_extent.height
        chart_hsize = lon_pixels / CHART_DPI
        chart_vsize = lat_pixels / CHART_DPI
        lon_degrees = (extent[1] - extent[0])
        lat_degrees = (extent[3] - extent[2])

        # Calculate the pixel-to-degree ratio
        lon_inches_per_degree = chart_hsize / lon_degrees
        lat_inches_per_degree = chart_vsize / lat_degrees


        multiples = GRID_LINE_SPACING_DEGREES

        grid_line_spacing_inches = []
        for multiple in multiples:
            lon_grid_spacing = lon_inches_per_degree * multiple
            lat_grid_spacing = lat_inches_per_degree * multiple
            grid_line_spacing_inches.append(min(lon_grid_spacing, lat_grid_spacing))

        #fitted_grid_line_degrees = min(multiples, key=lambda x: (x - min_grid_line_spacing_pixels if x >= min_grid_line_spacing_pixels else float('inf')))
        fitted_grid_line_degrees = min([multiple for multiple, spacing in zip(multiples, grid_line_spacing_inches) if spacing >= MIN_GRID_LINE_SPACING_INCHES], default=float('inf'))
        if fitted_grid_line_degrees == float('inf'):
            # must pick a reasonable number
            fitted_grid_line_degrees = multiples[-1]

        gl.xlocator = plt.MultipleLocator(fitted_grid_line_degrees)
        gl.ylocator = plt.MultipleLocator(fitted_grid_line_degrees)

        self.lastgl = gl

        #gl.xformatter = LONGITUDE_FORMATTER
        #gl.yformatter = LATITUDE_FORMATTER
        lat_formatter = LatitudeFormatter(direction_label=True)
        lon_formatter = LongitudeFormatter(direction_label=True)
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)

    def display_map(self):
        self.scatter_objects = {}
        self.line_collection_objects = {}

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        # Adjust figure size to fill the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        if self.fig:
            try:
                plt.close(self.fig)
                self.measure_tool.reset_measurement()
            except:
                pass

        self.fig = plt.figure(figsize=(screen_width / CHART_DPI, screen_height / CHART_DPI), dpi=CHART_DPI, facecolor='black')

        # self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.measure_tool.changed_extent(self.ax)
        self.ax.set_anchor("NW")

        self.ax.autoscale_view(scalex=False,scaley=False)
        self.ax.set_yscale('linear')
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
        self.ax.set_extent(self.global_extent)

        # Add state boundary lines
        states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces',
            scale='50m',
            facecolor='none'
        )

        self.ax.add_feature(states, edgecolor='gray')

        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none'
        )
        self.ax.add_feature(country_borders, edgecolor='white', linewidth=0.5)
        self.ax.add_feature(cfeature.COASTLINE, edgecolor='yellow', linewidth=0.5)

        self.update_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key_press = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_key_release = self.canvas.mpl_connect('key_release_event', self.on_key_release)

        # we only know how big the canvas frame/plot is after drawing/packing, and we need to wait until drawing to fix the size
        self.canvas_frame.update_idletasks()
        # fix the size to the canvas frame
        frame_hsize = float(self.canvas_frame.winfo_width()) / CHART_DPI
        frame_vsize = float(self.canvas_frame.winfo_height()) / CHART_DPI
        h = [Size.Fixed(0), Size.Fixed(frame_hsize)]
        v = [Size.Fixed(0), Size.Fixed(frame_vsize)]
        divider = Divider(self.fig, (0, 0, 1, 1), h, v, aspect=False)
        self.ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        self.clear_storm_extrema_annotations()
        AnnotatedCircles.changed_extent(self.ax)
        SelectionLoops.changed_extent(self.ax)
        self.measure_tool.changed_extent(self.ax)
        self.canvas.draw()
        self.display_custom_boundaries()
        self.display_custom_overlay()
        self.axes_size = self.ax.get_figure().get_size_inches() * self.ax.get_figure().dpi

    def on_key_press(self, event):
        if not self.zoom_selection_box:
            self.measure_tool.on_key_press(event)

        if event.key == 'escape':
            # abort a zoom
            if self.zoom_selection_box is not None:
                self.zoom_selection_box.destroy()
                self.zoom_selection_box = None
            # self.fig.canvas.draw()

        if event.key == '0':
            self.zoom_out(max_zoom=True)

        if event.key == '-':
            self.zoom_out(step_zoom=True)

        if event.key == '=':
            self.zoom_in(step_zoom=True)

        if event.key == 'n':
            self.cycle_to_next_overlapped_point()

        if event.key == 'g':
            #TODO DEBUGGING ONLY
            print(SelectionLoops.get_polygons())

        if event.key == 'h':
            if self.selection_loop_mode:
                SelectionLoops.toggle_visible()
            else:
                self.hide_tc_candidate()

        if event.key == 'x':
            # annotate storm extrema
            self.annotate_storm_extrema()

        if event.key == 'c':
            if self.selection_loop_mode:
                SelectionLoops.clear()
            # clear storm extrema annotation(s)
            # check if any annotations has focus
            elif self.annotated_circle_objects:
                any_has_focus = False
                for internal_id, annotated_circles in self.annotated_circle_objects.items():
                    removed_circle = None
                    if not annotated_circles:
                        continue
                    try:
                        for annotated_circle in annotated_circles:
                            if annotated_circle:
                                if annotated_circle.annotation_has_focus(event):
                                    any_has_focus = True
                                    # we can't let annotation object pick it up,
                                    #  since this causes a race condition since we are removing it
                                    removed_circle = annotated_circle
                                    any_has_focus = True
                                    annotated_circle.remove()
                                    break
                    except:
                        traceback.print_exc()
                    self.ax.set_yscale('linear')
                    if any_has_focus:
                        annotated_circles.remove(removed_circle)
                        if len(annotated_circles) == 0:
                            del(self.annotated_circle_objects[internal_id])
                        else:
                            self.annotated_circle_objects[internal_id] = annotated_circles
                        break
                    self.ax.set_yscale('linear')

                if not any_has_focus:
                    self.clear_storm_extrema_annotations()
            else:
                self.clear_storm_extrema_annotations()
            self.ax.figure.canvas.draw()

    def on_key_release(self, event):
        if not self.zoom_selection_box:
            self.measure_tool.on_key_release(event)

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
                if self.selection_loop_mode:
                    SelectionLoops.on_click(event)
                    return
                try:
                    do_measure = self.measure_tool.on_click(event)
                except:
                    # start measurement
                    do_measure = False
                    traceback.print_exc()
                if do_measure:
                    pass
                else:
                    # check for drag annotation,
                    # then a legend click (change time viewing),
                    # then finally zoom
                    if event.button == 1:  # Left click
                        if AnnotatedCircles.any_annotation_contains_point(event):
                            # pass through for the draggable annotation click handler
                            return

                        changed_opacity = False
                        changed_opacity_index = 0
                        # the time_step legend is ordered bottom up (furthest valid time first)
                        new_opacity = 1.0
                        if self.time_step_legend_objects:

                            for i, time_step_legend_object in list(enumerate(reversed(self.time_step_legend_objects))):
                                bbox = time_step_legend_object.get_window_extent()
                                if bbox.contains(event.x, event.y):
                                    changed_opacity = True
                                    # clicked on legend
                                    changed_opacity_index = i
                                    # cycle between:
                                    #   1.0:    all visible
                                    #   0.6:    hide all storms tracks with start valid_time later than selected
                                    #   0.3:    hide all points beyond valid_time later than selected
                                    next_opacity_index = min(changed_opacity_index + 1, len(self.time_step_opacity) - 1)
                                    if self.time_step_opacity[i] != 1.0:
                                        # not cycling
                                        new_opacity = 0.6
                                    else:
                                        if self.time_step_opacity[next_opacity_index] == 1.0:
                                            new_opacity = 0.6
                                        elif self.time_step_opacity[next_opacity_index] == 0.6:
                                            new_opacity = 0.3
                                        else:
                                            new_opacity = 1.0
                                    break

                            if changed_opacity:
                                for i, time_step_legend_object in list(enumerate(reversed(self.time_step_legend_objects))):
                                    if i > changed_opacity_index:
                                        self.time_step_opacity[i] = new_opacity
                                    else:
                                        self.time_step_opacity[i] = 1.0

                        if changed_opacity:
                            # update map as we have changed what is visible
                            model_cycle = None
                            if self.mode == "GENESIS":
                                model_cycle = self.genesis_model_cycle_time

                            self.redraw_map_with_data(model_cycle=model_cycle)

                        else:
                            # zooming
                            try:
                                self.zoom_selection_box = SelectionBox(self.ax)
                                # handle case we are not blocking (something other action is blocking)
                                self.zoom_selection_box.update_box(event.xdata, event.ydata, event.xdata, event.ydata)
                            except:
                                # some other action is blocking
                                pass
                                traceback.print_exc()

                    elif event.button == 3:  # Right click
                        self.zoom_out(step_zoom=True)

    def on_release(self, event):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Check if mouse coordinates are within figure bounds
        try:
            inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
        except:
            inbound = False

        if self.selection_loop_mode:
            SelectionLoops.on_release(event)
            return

        doing_measurement = self.measure_tool.in_measure_mode()
        if doing_measurement:
            pass
        else:
            if event.button == 1 and self.zoom_selection_box:  # Left click release
                if event.xdata is None or event.ydata is None or not inbound:
                    if not self.zoom_selection_box.is_2d():
                        self.zoom_selection_box.destroy()
                        self.zoom_selection_box = None
                        return

                self.zoom_in()

    def update_circle_patch(self, lon=None, lat=None):
        if self.last_circle_lon == lon and self.last_circle_lat == lat:
            return

        if self.circle_handle:
            self.circle_handle.remove()

        self.ax.set_yscale('linear')

        self.circle_handle = Circle((lon, lat), radius=self.calculate_radius_pixels(), color=DEFAULT_ANNOTATE_MARKER_COLOR, fill=False, linestyle='dotted', linewidth=2, alpha=0.8,
                                    transform=ccrs.PlateCarree())
        self.ax.add_patch(self.circle_handle)
        self.ax.figure.canvas.draw()

        self.last_circle_lon = lon
        self.last_circle_lat = lat

    def clear_circle_patch(self):
        if self.circle_handle:
            self.circle_handle.remove()
            self.circle_handle = None
            self.ax.set_yscale('linear')
            self.ax.figure.canvas.draw()

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

            if self.selection_loop_mode:
                if inbound:
                    SelectionLoops.on_motion(event)
                return
            elif self.zoom_selection_box:
                x0 = self.zoom_selection_box.lon1
                y0 = self.zoom_selection_box.lat1
                if inbound:
                    x1, y1 = event.xdata, event.ydata
                else:
                    # out of bound motion
                    return
                if type(x0) == type(x1) == type(y0) == type(y1):
                    if self.zoom_selection_box is not None:
                        self.zoom_selection_box.update_box(x0, y0, x1, y1)

                    # blitting object will redraw using cached buffer
                    # self.fig.canvas.draw_idle()
            else:
                self.measure_tool.on_motion(event, inbound)

    @staticmethod
    def calculate_distance(start_point, end_point):
        geod = cgeo.Geodesic()
        line = LineString([start_point, end_point])
        total_distance = geod.geometry_length(line)
        nautical_miles = total_distance / 1852.0
        return nautical_miles

    def zoom_in(self, step_zoom=False):
        if step_zoom:
            extent = self.ax.get_extent()
            lon_diff = extent[1] - extent[0]
            lat_diff = extent[3] - extent[2]
            lon_center, lat_center = self.last_cursor_lon_lat
            target_width = lon_diff / ZOOM_IN_STEP_FACTOR
            target_height = lat_diff / ZOOM_IN_STEP_FACTOR
            x0 = lon_center - (target_width / 2.0)
            x1 = lon_center + (target_width / 2.0)
            y0 = lat_center - (target_height / 2.0)
            y1 = lat_center + (target_height / 2.0)
            if x0 < -180.0:
                x0 = -180.0
                x1 = x0 + target_width
            if x1 > 180.0:
                x1 = 180.0
                x0 = x1 - target_width
            if y0 < -90.0:
                y0 = 90.0
                y1 = y0 + target_height
            if y1 > 90.0:
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
            AnnotatedCircles.changed_extent(self.ax)
            SelectionLoops.changed_extent(self.ax)
            self.measure_tool.changed_extent(self.ax)
            self.ax.figure.canvas.draw()
            self.rvor_labels_new_extent()
            self.ax.figure.canvas.draw()

        #elif self.zoom_rect and (None not in self.zoom_rect) and len(self.zoom_rect) == 4:
        # 2d checks if valid rect and that x's aren't close and y's aren't close
        elif self.zoom_selection_box:
            if not self.zoom_selection_box.is_2d():
                self.zoom_selection_box.destroy()
                self.zoom_selection_box = None
                return

            x0 = self.zoom_selection_box.lon1
            y0 = self.zoom_selection_box.lat1
            x1 = self.zoom_selection_box.lon2
            y1 = self.zoom_selection_box.lat2

            self.zoom_selection_box.destroy()
            self.zoom_selection_box = None

            extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]

            # Calculate the aspect ratio of the zoom rectangle
            zoom_aspect_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])

            # Calculate the aspect ratio of the canvas frame
            frame_width_pixels = self.canvas_frame.winfo_width()
            frame_height_pixels = self.canvas_frame.winfo_height()
            frame_aspect_ratio = frame_height_pixels / frame_width_pixels

            # Determine the dimension to set the extent and the dimension to expand
            if zoom_aspect_ratio < frame_aspect_ratio:
                # Set the longitude extent and expand the latitude extent
                lon_extent = extent[0], extent[1]
                lat_center = (extent[2] + extent[3]) / 2
                lat_height = (lon_extent[1] - lon_extent[0]) * frame_aspect_ratio
                if lat_center + (lat_height / 2) > 90.0:
                    lat_max = 90.0
                    lat_min = lat_max - lat_height
                elif lat_center - (lat_height / 2) < -90.0:
                    lat_min = -90.0
                    lat_max = lat_min + lat_height
                else:
                    lat_min = lat_center - (lat_height / 2)
                    lat_max = lat_center + (lat_height / 2)
                lat_extent = [lat_min, lat_max]
            else:
                # Set the latitude extent and expand the longitude extent
                lat_extent = extent[2], extent[3]
                lon_center = (extent[0] + extent[1]) / 2
                lon_width = (lat_extent[1] - lat_extent[0]) / frame_aspect_ratio
                if lon_center + (lon_width / 2) > 180.0:
                    lon_max = 180.0
                    lon_min = lon_max - lon_width
                elif lon_center - (lon_width / 2) < -180.0:
                    lon_min = -180.0
                    lon_max = lon_min + lon_width
                else:
                    lon_min = lon_center - (lon_width / 2)
                    lon_max = lon_center + (lon_width / 2)
                lon_extent = [lon_min, lon_max]

            self.ax.set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree())

            self.clear_storm_extrema_annotations()
            self.update_axes()
            AnnotatedCircles.changed_extent(self.ax)
            SelectionLoops.changed_extent(self.ax)
            self.measure_tool.changed_extent(self.ax)
            self.ax.figure.canvas.draw()
            self.rvor_labels_new_extent()
            self.ax.figure.canvas.draw()

    def zoom_out(self, max_zoom=False, step_zoom=False):
        extent = self.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        max_lon_diff = 360.0  # Maximum longitudinal extent
        max_lat_diff = 180.0  # Maximum latitudinal extent

        if step_zoom:
            lon_center = (extent[1] + extent[0]) / 2.0
            lat_center = (extent[3] + extent[2]) / 2.0
            target_width = lon_diff * ZOOM_IN_STEP_FACTOR
            target_height = lat_diff * ZOOM_IN_STEP_FACTOR
            x0 = lon_center - (target_width / 2.0)
            x1 = lon_center + (target_width / 2.0)
            y0 = lat_center - (target_height / 2.0)
            y1 = lat_center + (target_height / 2.0)
            if x0 < -180.0:
                x0 = -180.0
                x1 = x0 + target_width
            if x1 > 180.0:
                x1 = 180.0
                x0 = x1 - target_width
            if y0 < -90.0:
                y0 = 90.0
                y1 = y0 + target_height
            if y1 > 90.0:
                y1 = 90.0
                y0 = y1 - target_height

            if x0 < -180.0:
                x0 = -180.0
            if x1 > 180.0:
                x1 = 180.0
            if y0 < -90.0:
                y0 = -90.0
            if y1 > 90.0:
                y1 = 90.0

            # Ensure new extent is within bounds
            new_extent = [
                x0,
                x1,
                y0,
                y1,
            ]

            if x1 - x0 <= max_lon_diff and y1 - y0 <= max_lat_diff:
                self.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
                self.clear_storm_extrema_annotations()
                self.update_axes()
                AnnotatedCircles.changed_extent(self.ax)
                SelectionLoops.changed_extent(self.ax)
                self.measure_tool.changed_extent(self.ax)
                self.ax.figure.canvas.draw()
                self.rvor_labels_new_extent()
                self.ax.figure.canvas.draw()

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
                AnnotatedCircles.changed_extent(self.ax)
                SelectionLoops.changed_extent(self.ax)
                # do measure last as we are going to remove and redraw it
                self.measure_tool.changed_extent(self.ax)
                self.ax.figure.canvas.draw()
                self.rvor_labels_new_extent()
                self.ax.figure.canvas.draw()

    def show_config_genesis_dialog(self):
        self.show_config_dialog()
        self.set_focus_on_map()

    def show_config_adeck_dialog(self):
        self.show_config_dialog()
        self.set_focus_on_map()

    def show_config_dialog(self):
        global displayed_functional_annotation_options
        global ANNOTATE_COLOR_LEVELS
        global RVOR_CYCLONIC_CONTOURS
        global RVOR_CYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_CONTOURS
        global MINIMUM_CONTOUR_PX_X
        global MINIMUM_CONTOUR_PX_Y
        global DISPLAYED_FUNCTIONAL_ANNOTATIONS
        global DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS
        global ZOOM_IN_STEP_FACTOR
        global MIN_GRID_LINE_SPACING_INCHES
        global DEFAULT_ANNOTATE_MARKER_COLOR
        global DEFAULT_ANNOTATE_TEXT_COLOR
        global ANNOTATE_DT_START_COLOR
        global ANNOTATE_EARLIEST_NAMED_COLOR
        global ANNOTATE_VMAX_COLOR
        settings = {
            'RVOR_CYCLONIC_CONTOURS': tk.BooleanVar(value=RVOR_CYCLONIC_CONTOURS),
            'RVOR_CYCLONIC_LABELS': tk.BooleanVar(value=RVOR_CYCLONIC_LABELS),
            'RVOR_ANTICYCLONIC_LABELS': tk.BooleanVar(value=RVOR_ANTICYCLONIC_LABELS),
            'RVOR_ANTICYCLONIC_CONTOURS': tk.BooleanVar(value=RVOR_ANTICYCLONIC_CONTOURS),
            'MINIMUM_CONTOUR_PX_X': tk.IntVar(value=MINIMUM_CONTOUR_PX_X),
            'MINIMUM_CONTOUR_PX_Y': tk.IntVar(value=MINIMUM_CONTOUR_PX_Y),
            'DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS': tk.IntVar(value=DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS),
            'ZOOM_IN_STEP_FACTOR': tk.DoubleVar(value=ZOOM_IN_STEP_FACTOR),
            'MIN_GRID_LINE_SPACING_INCHES': tk.DoubleVar(value=MIN_GRID_LINE_SPACING_INCHES),
            'DEFAULT_ANNOTATE_MARKER_COLOR': tk.StringVar(value=DEFAULT_ANNOTATE_MARKER_COLOR),
            'DEFAULT_ANNOTATE_TEXT_COLOR': tk.StringVar(value=DEFAULT_ANNOTATE_TEXT_COLOR),
            'ANNOTATE_DT_START_COLOR': tk.StringVar(value=ANNOTATE_DT_START_COLOR),
            'ANNOTATE_EARLIEST_NAMED_COLOR': tk.StringVar(value=ANNOTATE_EARLIEST_NAMED_COLOR),
            'ANNOTATE_VMAX_COLOR': tk.StringVar(value=ANNOTATE_VMAX_COLOR),
        }
        root_width = self.root.winfo_screenwidth()
        root_height = self.root.winfo_screenheight()

        dialog = ConfigDialog(self.root, displayed_functional_annotation_options, DISPLAYED_FUNCTIONAL_ANNOTATIONS, settings, root_width, root_height)
        if dialog.result:
            result = dialog.result
            updated_annotated_colors = False
            for key, vals in result.items():
                if key == 'annotation_label_options':
                    DISPLAYED_FUNCTIONAL_ANNOTATIONS = [option for option in displayed_functional_annotation_options if option in vals]
                elif key == 'settings':
                    for global_setting_name, val in vals.items():
                        if global_setting_name in settings.keys():
                            if 'ANNOTATE' and 'COLOR' in global_setting_name:
                                updated_annotated_colors = True
                            globals()[global_setting_name] = val
            if updated_annotated_colors:
                # must update the color levels from which we actually pick the colors
                ANNOTATE_COLOR_LEVELS = {
                    0: DEFAULT_ANNOTATE_TEXT_COLOR,
                    1: ANNOTATE_DT_START_COLOR,
                    2: ANNOTATE_EARLIEST_NAMED_COLOR,
                    3: ANNOTATE_VMAX_COLOR
                }

            # Save settings to file
            with open('settings_tcviewer.json', 'w') as f:
                json.dump(result, f, indent=4)

        # fix focus back to map
        self.set_focus_on_map()

    @staticmethod
    def load_settings():
        try:
            global displayed_functional_annotation_options
            global ANNOTATE_COLOR_LEVELS
            global DISPLAYED_FUNCTIONAL_ANNOTATIONS
            global DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS
            global ZOOM_IN_STEP_FACTOR
            global MIN_GRID_LINE_SPACING_INCHES
            global DEFAULT_ANNOTATE_MARKER_COLOR
            global DEFAULT_ANNOTATE_TEXT_COLOR
            global ANNOTATE_DT_START_COLOR
            global ANNOTATE_EARLIEST_NAMED_COLOR
            global ANNOTATE_VMAX_COLOR
            with open('settings_tcviewer.json', 'r') as f:
                settings = json.load(f)
                for key, val in settings.items():
                    if key == 'annotation_label_options':
                        DISPLAYED_FUNCTIONAL_ANNOTATIONS = [option for option in displayed_functional_annotation_options if option in val]
                    elif key == 'settings':
                        for global_setting_name, v in val.items():
                            globals()[global_setting_name] = v
                ANNOTATE_COLOR_LEVELS = {
                    0: DEFAULT_ANNOTATE_TEXT_COLOR,
                    1: ANNOTATE_DT_START_COLOR,
                    2: ANNOTATE_EARLIEST_NAMED_COLOR,
                    3: ANNOTATE_VMAX_COLOR
                }
        except FileNotFoundError:
            pass

    # noinspection PyUnusedLocal
    def take_screenshot(self, *args):
        # Get the current UTC date and time
        current_time = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

        # Create the screenshots folder if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        # Capture the screenshot
        screenshot = ImageGrab.grab()

        # Save the screenshot as a PNG file
        screenshot.save(f"screenshots/{current_time}.png")

        # Create a custom dialog window
        dialog = tk.Toplevel(self.root, bg="#000000")
        dialog.title("tcviewer")
        dialog.geometry("200x100")  # Set the dialog size

        # Center the dialog
        x = (tk_root.winfo_screenwidth() - 200) // 2
        y = (tk_root.winfo_screenheight() - 100) // 2
        dialog.geometry(f"+{x}+{y}")

        label = tk.Label(dialog, text="Screenshot saved", bg="#000000", fg="#FFFFFF")
        label.pack(pady=10)
        button = tk.Button(dialog, text="OK", command=dialog.destroy)
        button.pack(pady=10)

        # Set focus on the OK button
        button.focus_set()

        # Bind the Enter key to the button's command
        dialog.bind("<Return>", lambda event: button.invoke())

        dialog.bind("<Escape>", lambda event: dialog.destroy())

        # Bind the FocusOut event to the dialog's destroy method
        dialog.bind("<FocusOut>", lambda event: dialog.destroy())

    def toggle_selection_loop_mode(self):
        self.selection_loop_mode = not(self.selection_loop_mode)
        #TODO Change color of button
        self.update_toggle_selection_loop_button_color()
        self.set_focus_on_map()

# Blitting measure tool (single instance)
class MeasureTool:
    def __init__(self, ax):
        self.ax = ax
        self.measure_mode = False
        self.start_point = None
        self.end_point = None
        self.line = None
        self.distance_text = None
        self.distance = 0
        # Save background
        self.bg = None
        self.blocking = False

    def in_measure_mode(self):
        return self.measure_mode

    def unblock_for_measure(self):
        if self.blocking:
            self.blocking = False
            if EventManager.get_blocking_purpose():
                EventManager.unblock_events()

    def block_for_measure(self):
        if self.measure_mode:
            if not EventManager.get_blocking_purpose():
                blocking_for_measure = EventManager.block_events('measure')
                if not blocking_for_measure:
                    raise ValueError("Failed to block events for MeasureTool")
            self.blocking = True
            return True
        return False

    def changed_extent(self, ax):
        self.ax = ax
        if self.line or self.distance_text:
            self.remove_artists()
            ax.figure.canvas.draw()
            # update background region
            self.bg = ax.figure.canvas.copy_from_bbox(ax.bbox)
            self.create_artists()

    @staticmethod
    def calculate_distance(start_point, end_point):
        geod = cgeo.Geodesic()
        line = LineString([start_point, end_point])
        total_distance = geod.geometry_length(line)
        nautical_miles = total_distance / 1852.0
        return nautical_miles

    def on_key_press(self, event):
        if event.key == 'shift':
            self.measure_mode = True
            self.block_for_measure()

    def on_key_release(self, event):
        if event.key == 'shift':
            self.measure_mode = False
            self.unblock_for_measure()

    def on_motion(self, event, inbound):
        ax = self.ax
        if not self.measure_mode:
            return False

        if not self.start_point:
            return False

        if not inbound:
            return False

        self.end_point = (event.xdata, event.ydata)

        try:
            if not self.line:
                self.bg = ax.figure.canvas.copy_from_bbox(ax.bbox)
                self.create_line_artist()
        except:
            traceback.print_exc()
            pass

        try:
            self.line.set_data([self.start_point[0], self.end_point[0]],
                               [self.start_point[1], self.end_point[1]])
        except:
            pass

        self.remove_distance_text_artist()

        try:
            self.distance = self.calculate_distance(self.start_point, self.end_point)
            self.create_distance_text_artist(self.distance)
        except:
            self.distance = 0

        self.restore_region()
        if self.line:
            ax.draw_artist(self.line)
        if self.distance_text:
            ax.draw_artist(self.distance_text)
        ax.figure.canvas.blit(ax.bbox)
        return True

    def create_artists(self):
        self.create_line_artist()
        self.create_distance_text_artist(self.distance)

    def restore_region(self):
        if self.bg:
            self.ax.figure.canvas.restore_region(self.bg)

    def remove_line_artist(self):
        if self.line:
            try:
                self.line.remove()
            except:
                pass
            self.line = None

    def remove_distance_text_artist(self):
        if self.distance_text:
            try:
                self.distance_text.remove()
            except:
                pass
            self.distance_text = None

    def remove_artists(self):
        self.remove_line_artist()
        self.remove_distance_text_artist()

    def on_click(self, event):
        if self.measure_mode:
            if event.button == 1:
                self.start_point = (event.xdata, event.ydata)
                self.end_point = (event.xdata, event.ydata)
                return True
            elif event.button == 3:
                self.reset_measurement()
        return False

    def create_line_artist(self):
        if self.ax and self.start_point and self.end_point:
            self.line = self.ax.plot([self.start_point[0], self.end_point[0]],
                                    [self.start_point[1], self.end_point[1]],
                                    color='cyan', linewidth=2, transform=ccrs.PlateCarree())[0]

    def create_distance_text_artist(self, distance):
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
            offset_degrees_x = offset_pixels * lon_deg_per_pixel
            offset_degrees_y = offset_pixels * lat_deg_per_pixel

            # Calculate offset in degrees in x and y directions
            offset_x_deg = offset_degrees_x * np.cos(np.radians(angle + 90))
            offset_y_deg = offset_degrees_y * np.sin(np.radians(angle + 90))

            # Adjust angle to ensure text is always right-side up
            if angle > 90 or angle < -90:
                angle += 180
                offset_x_deg = -offset_x_deg
                offset_y_deg = -offset_y_deg

            self.distance_text = self.ax.text(mid_point[0] + offset_x_deg, mid_point[1] + offset_y_deg,
                                              f"{distance:.2f} NM", color='white', fontsize=12,
                                              ha='center', va='center', rotation=angle,
                                              bbox=dict(facecolor='black', alpha=0.5))

    def reset_measurement(self):
        self.remove_artists()
        self.ax.set_yscale('linear')
        self.start_point = None
        self.end_point = None
        self.line = None
        self.distance = 0
        self.ax.figure.canvas.draw()

# blitting box for map (for zooms / selections)
class SelectionBox:
    def __init__(self, ax):
        block_for_zoom = EventManager.block_events('zoom')
        if not block_for_zoom:
            raise ValueError("Failed to block events for SelectionBox")
        else:
            self.ax = ax
            self.box = None
            self.lon1, self.lat1, self.lon2, self.lat2 = None, None, None, None
            self.has_latlons = False
            self.bg = ax.figure.canvas.copy_from_bbox(ax.bbox)  # Save background

    def restore_region(self):
        self.ax.figure.canvas.restore_region(self.bg)

    def is_2d(self):
        if self.lon1 and self.lon2 and self.lat1 and self.lat2:
            return not(np.isclose(self.lon1, self.lon2) or np.isclose(self.lat1, self.lat2))
        else:
            return False

    def draw_box(self):
        if self.box:
            try:
                self.box.remove()
            except:
                pass
        self.box = self.ax.plot(
            [self.lon1, self.lon2, self.lon2, self.lon1, self.lon1],
            [self.lat1, self.lat1, self.lat2, self.lat2, self.lat1],
            color='yellow', linestyle='--', transform=ccrs.PlateCarree())
        for artist in self.box:
            self.ax.draw_artist(artist)
        self.ax.figure.canvas.blit(self.ax.bbox)

    def update_box(self, lon1, lat1, lon2, lat2):
        self.remove()
        self.restore_region()  # Restore first
        self.lon1, self.lat1, self.lon2, self.lat2 = lon1, lat1, lon2, lat2
        if lon1 and lat1 and lon2 and lat2:
            self.has_latlons = True
            self.draw_box()  # Draw the box
        self.ax.figure.canvas.blit(self.ax.bbox)

    def remove(self):
        if self.box:
            try:
                for artist in self.box:
                    artist.remove()
            except:
                pass

    def destroy(self):
        self.remove()
        self.restore_region()  # Restore regions without drawing
        self.ax.figure.canvas.blit(self.ax.bbox)
        EventManager.unblock_events()

from matplotlib.patches import Polygon as MPLPolygon

class SelectionLoops:
    ax = None
    selection_loop_objects = []
    last_loop = None
    selecting = False
    blocking = False
    visible = True

    @classmethod
    def unblock(cls):
        if cls.blocking:
            try:
                EventManager.unblock_events()
            except:
                traceback.print_exc()
        cls.blocking = False

    @classmethod
    def block(cls):
        try:
            EventManager.block_events('selection_loop')
            cls.blocking = True
        except:
            traceback.print_exc()
            cls.blocking = False

    @classmethod
    def get_polygons(cls):
        all_polygons = []
        for selection_loop_obj in cls.selection_loop_objects:
            all_polygons.extend(selection_loop_obj.get_polygons())
        return all_polygons

    @classmethod
    def clear(cls):
        for selection_loop_obj in cls.selection_loop_objects:
            selection_loop_obj.remove()
        cls.selection_loop_objects = []
        cls.last_loop = None
        cls.selecting = False
        cls.unblock()
        cls.ax.figure.canvas.draw_idle()

    @classmethod
    def toggle_visible(cls):
        if not cls.selecting:
            cls.visible = not(cls.visible)
            for selection_loop_obj in cls.selection_loop_objects:
                selection_loop_obj.set_visible(cls.visible)
            cls.ax.figure.canvas.draw_idle()

    @classmethod
    def on_click(cls, event):
        if event.button == 1:  # left click
            cls.last_loop = cls.SelectionLoop(cls.ax, event)
            cls.selection_loop_objects.append(cls.last_loop)
            cls.selecting = True
            cls.block()
            return True
        else:
            return False

    @classmethod
    def on_motion(cls, event):
        if cls.selecting and event.button == 1:  # left click
            return cls.last_loop.on_motion(event)
        else:
            return False

    @classmethod
    def on_release(cls, event):
        if cls.selecting and event.button == 1:  # left click
            cls.last_loop.on_release()
            cls.unblock()
            cls.selecting = False
            return True
        else:
            cls.selecting = False
            return False

    @classmethod
    def changed_extent(cls, new_ax):
        if cls.ax != new_ax:
            cls.visible = True
        cls.ax = new_ax
        for selection_loop_obj in cls.selection_loop_objects:
            selection_loop_obj.changed_extent(new_ax)

    class SelectionLoop:
        def __init__(self, ax, event):
            self.ax = ax
            self.verts = []
            self.polygons = []
            self.preview = None
            self.preview_artists = []
            self.closed_artists = []
            self.background = None
            self.closed = False
            self.verts.append((event.xdata, event.ydata))
            self.alpha = 0.35
            self.bg = None

        def get_polygons(self):
            return self.polygons

        def changed_extent(self, ax):
            # preserve the polygons on the map across all map & data changes
            if ax != self.ax:
                self.ax = ax
                if self.closed_artists:
                    self.remove()
                    ax.figure.canvas.draw()
                    # update background region
                    self.bg = ax.figure.canvas.copy_from_bbox(ax.bbox)
                    self.update_closed_artists()

        def set_visible(self, visible):
            for patch in self.closed_artists:
                patch.set_visible(visible)

        def on_motion(self, event):
            self.verts.append((event.xdata, event.ydata))
            self.update_preview()
            return True

        def on_release(self):
            polys = self.close_loop()
            self.set_polygons(polys)
            self.update_closed_artists()
            self.closed = True
            # clean up memory
            self.background = None

        def remove_preview_artists(self):
            if self.preview_artists:
                for artist in self.preview_artists:
                    artist.remove()
                self.preview_artists = []
                self.preview = []

        def remove_closed_artists(self):
            if self.closed_artists:
                for artist in self.closed_artists:
                    artist.remove()
                self.closed_artists = []
                self.closed = []

        def remove(self):
            self.remove_preview_artists()
            self.remove_closed_artists()

        def update_preview(self):
            if self.background is None:
                self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
            if self.preview_artists is not None:
                self.ax.figure.canvas.restore_region(self.background)

            self.remove_preview_artists()
            self.preview = self.close_loop()
            if self.preview:
                for poly in self.preview:
                    patch = MPLPolygon(poly.exterior.coords, alpha=self.alpha)
                    artist = self.ax.add_patch(patch)
                    self.preview_artists.append(artist)
                    self.ax.draw_artist(artist)
                self.ax.figure.canvas.blit(self.ax.bbox)
                return
            else:
                self.ax.figure.canvas.restore_region(self.background)
                return

        def close_loop(self):
            polys = []
            if len(self.verts) > 2:
                shapely_poly = Polygon(self.verts)
                if shapely_poly.is_simple:
                    # no self-intersections
                    polys.append(shapely_poly)
                else:
                    # self-intersections, find loops
                    loops = self.find_loops(shapely_poly)
                    polys.extend(loops)
            return polys

        def set_polygons(self, polygons):
            self.polygons = polygons

        @staticmethod
        def find_loops(shapely_poly):
            loops = []
            if shapely_poly.is_simple:
                loops.append(shapely_poly)
            else:
                interiors = shapely_poly.interiors
                for interior in interiors:
                    loop = Polygon(interior)
                    loops.append(loop)
                loops.append(Polygon(shapely_poly.exterior))
            return loops

        def update_closed_artists(self):
            for poly in self.polygons:
                patch = MPLPolygon(poly.exterior.coords, alpha=self.alpha)
                self.closed_artists.append(self.ax.add_patch(patch))
            self.remove_preview_artists()
            self.ax.figure.canvas.draw_idle()

# for mutex blocking of mouse hover events
class EventManager:
    blocked = False
    blocking_purpose = None

    @classmethod
    def reset(cls):
        cls.blocked = False
        cls.blocking_purpose = False

    @classmethod
    def block_events(cls, purpose):
        if cls.blocked:
            return False
        cls.blocked = True
        cls.blocking_purpose = purpose
        return True

    @classmethod
    def unblock_events(cls):
        if not cls.blocked:
            raise ValueError("Events are not blocked")
        cls.blocked = False
        cls.blocking_purpose = None

    @classmethod
    def get_blocking_purpose(cls):
        return cls.blocking_purpose

# use toplevel rather than simple dialog as with this we can center the dialog
class ConfigDialog(tk.Toplevel):
    def __init__(self, parent, annotation_label_options, selected_annotation_label_options, settings, root_width, root_height):
        self.buttonbox = None
        self.notebook = None
        self.default_annotate_marker_color_label = None
        self.default_annotate_text_color_label = None
        self.annotate_dt_start_color_label = None
        self.annotate_earliest_named_color_label = None
        self.annotate_vmax_color_label = None
        self.color_labels = None
        self.annotation_label_checkboxes = None
        self.result = None
        self.annotation_label_options = annotation_label_options
        self.selected_annotation_label_options = selected_annotation_label_options
        self.settings = settings
        self.root_width = root_width
        self.root_height = root_height

        super().__init__(parent)  # Set the title here
        self.title('Settings')
        # this is required to center the dialog on the window
        self.transient(parent)
        self.parent = parent

        self.body_frame = tk.Frame(self)
        self.body_frame.pack(padx=5, pady=5)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(padx=5, pady=5)

        self.body(self.body_frame, )
        self.create_buttonbox(self.button_frame)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.grab_set()
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.wait_visibility()
        self.center_window()
        self.wait_window(self)

    def center_window(self):
        self.update_idletasks()

        # Get the dimensions of the dialog
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()

        # Get the dimensions of the parent window
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate the position to center the dialog on the parent window
        x = self.parent.winfo_x() + (parent_width // 2) - (dialog_width // 2)
        y = self.parent.winfo_y() + (parent_height // 2) - (dialog_height // 2)

        # Set the position of the dialog
        self.geometry(f'{dialog_width}x{dialog_height}+{x}+{y}')

    def restore_defaults(self):
        import os
        os.remove('settings_tcviewer.json')
        self.cancel()

    def create_buttonbox(self, master):
        self.buttonbox = tk.Frame(master)
        w = tk.Button(self.buttonbox, text="Restore All Defaults (requires restart)", command=self.restore_defaults)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        ok_w = tk.Button(self.buttonbox, text="OK", command=self.ok)
        ok_w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(self.buttonbox, text="Cancel", command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        # Set focus on the OK button
        ok_w.focus_set()

        self.buttonbox.pack()
        return self.buttonbox

    def body(self, master):
        frame = ttk.Frame(master)
        frame.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill="both", expand=True)

        # Create a frame for each tab
        map_settings_frame = tk.Frame(self.notebook)
        self.notebook.add(map_settings_frame, text="Map")

        # Create a frame for each tab
        overlay_settings_frame = tk.Frame(self.notebook)
        self.notebook.add(overlay_settings_frame, text="Overlay")

        annotation_colors_frame = tk.Frame(self.notebook)
        self.notebook.add(annotation_colors_frame, text="Annotation Colors")

        extrema_annotations_frame = tk.Frame(self.notebook)
        self.notebook.add(extrema_annotations_frame, text="Annotation Labels")

        tk.Label(map_settings_frame, text="Circle patch radius (pixels):").pack()
        tk.Entry(map_settings_frame, textvariable=self.settings['DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS']).pack()
        tk.Label(map_settings_frame, text="Zoom in step factor:").pack()
        tk.Entry(map_settings_frame, textvariable=self.settings['ZOOM_IN_STEP_FACTOR']).pack()
        tk.Label(map_settings_frame, text="Min grid line spacing (inches):").pack()
        tk.Entry(map_settings_frame, textvariable=self.settings['MIN_GRID_LINE_SPACING_INCHES']).pack()

        overlay_var_text = {
            'RVOR_CYCLONIC_CONTOURS': 'RVOR Cyclonic Contours',
            'RVOR_CYCLONIC_LABELS': 'RVOR Cyclonic Labels',
            'RVOR_ANTICYCLONIC_CONTOURS': 'RVOR Anti-Cyclonic Contours',
            'RVOR_ANTICYCLONIC_LABELS': 'RVOR Anti-cyclonic Labels',
        }

        for varname, option in overlay_var_text.items():
            var = self.settings[varname]
            cb = tk.Checkbutton(overlay_settings_frame, text=option, variable=var)
            cb.pack(anchor=tk.W)

        tk.Label(overlay_settings_frame, text="Min. contour pixels X:").pack()
        tk.Entry(overlay_settings_frame, textvariable=self.settings['MINIMUM_CONTOUR_PX_X']).pack()

        tk.Label(overlay_settings_frame, text="Min. contour pixels Y:").pack()
        tk.Entry(overlay_settings_frame, textvariable=self.settings['MINIMUM_CONTOUR_PX_Y']).pack()

        tk.Label(map_settings_frame, text="Zoom in step factor:").pack()
        tk.Entry(map_settings_frame, textvariable=self.settings['ZOOM_IN_STEP_FACTOR']).pack()
        tk.Label(map_settings_frame, text="Min grid line spacing (inches):").pack()
        tk.Entry(map_settings_frame, textvariable=self.settings['MIN_GRID_LINE_SPACING_INCHES']).pack()

        tk.Label(annotation_colors_frame, text="(Circle hover) Marker color:").pack()
        self.default_annotate_marker_color_label = tk.Label(annotation_colors_frame, text="", width=10, bg=self.settings['DEFAULT_ANNOTATE_MARKER_COLOR'].get())
        self.default_annotate_marker_color_label.pack()
        self.default_annotate_marker_color_label.bind("<Button-1>", lambda event: self.choose_color('DEFAULT_ANNOTATE_MARKER_COLOR'))

        tk.Label(annotation_colors_frame, text="Default annotation text color:").pack()
        self.default_annotate_text_color_label = tk.Label(annotation_colors_frame, text="", width=10, bg=self.settings['DEFAULT_ANNOTATE_TEXT_COLOR'].get())
        self.default_annotate_text_color_label.pack()
        self.default_annotate_text_color_label.bind("<Button-1>", lambda event: self.choose_color('DEFAULT_ANNOTATE_TEXT_COLOR'))

        tk.Label(annotation_colors_frame, text="TC start text color:").pack()
        self.annotate_dt_start_color_label = tk.Label(annotation_colors_frame, text="", width=10, bg=self.settings['ANNOTATE_DT_START_COLOR'].get())
        self.annotate_dt_start_color_label.pack()
        self.annotate_dt_start_color_label.bind("<Button-1>", lambda event: self.choose_color('ANNOTATE_DT_START_COLOR'))

        tk.Label(annotation_colors_frame, text="Earliest named text color:").pack()
        self.annotate_earliest_named_color_label = tk.Label(annotation_colors_frame, text="", width=10, bg=self.settings['ANNOTATE_EARLIEST_NAMED_COLOR'].get())
        self.annotate_earliest_named_color_label.pack()
        self.annotate_earliest_named_color_label.bind("<Button-1>", lambda event: self.choose_color('ANNOTATE_EARLIEST_NAMED_COLOR'))

        tk.Label(annotation_colors_frame, text="Vmax text color:").pack()
        self.annotate_vmax_color_label = tk.Label(annotation_colors_frame, text="", width=10, bg=self.settings['ANNOTATE_VMAX_COLOR'].get())
        self.annotate_vmax_color_label.pack()
        self.annotate_vmax_color_label.bind("<Button-1>", lambda event: self.choose_color('ANNOTATE_VMAX_COLOR'))

        self.color_labels = {
            'DEFAULT_ANNOTATE_MARKER_COLOR': self.default_annotate_marker_color_label,
            'DEFAULT_ANNOTATE_TEXT_COLOR': self.default_annotate_text_color_label,
            'ANNOTATE_DT_START_COLOR': self.annotate_dt_start_color_label,
            'ANNOTATE_EARLIEST_NAMED_COLOR': self.annotate_earliest_named_color_label,
            'ANNOTATE_VMAX_COLOR': self.annotate_vmax_color_label,
        }

        # Add your widgets for the "Extrema Annotations" tab here
        self.annotation_label_checkboxes = {}
        for option in self.annotation_label_options:
            var = tk.IntVar()
            cb = tk.Checkbutton(extrema_annotations_frame, text=option, variable=var)
            cb.pack(anchor=tk.W)
            self.annotation_label_checkboxes[option] = var
            if option in self.selected_annotation_label_options:
                var.set(1)

    def choose_color(self, setting_name):
        color_obj = self.settings[setting_name]
        if color_obj:
            color = colorchooser.askcolor(color=color_obj.get())[1]
            if color:
                color_obj.set(color)
                self.color_labels[setting_name].config(bg=color)

    def apply(self):
        self.result = {
            'annotation_label_options': [option for option, var in self.annotation_label_checkboxes.items() if var.get()],
            'settings': {key: var.get() for key, var in self.settings.items()}
        }

    # noinspection PyUnusedLocal
    def ok(self, event=None):
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    # noinspection PyUnusedLocal
    def cancel(self, event=None):
        self.parent.focus_set()
        self.destroy()

if __name__ == "__main__":
    tk_root = tk.Tk()

    # Style configuration for ttk widgets
    tk_style = ttk.Style()
    tk_style.theme_use('clam')  # Ensure using a theme that supports customization
    default_bg = "black"
    default_fg = "white"
    tk_style.configure("TButton", background=default_bg, foreground=default_fg)
    tk_style.configure("White.TButton", background=default_bg, foreground="white")
    tk_style.configure("Red.TButton", background=default_bg, foreground="red")
    tk_style.configure("Orange.TButton", background=default_bg, foreground="orange")
    tk_style.configure("Yellow.TButton", background=default_bg, foreground="yellow")

    tk_style.configure("YellowAndBorder.TButton", background=default_bg, foreground="yellow", bordercolor="yellow")
    tk_style.configure("WhiteAndBorder.TButton", background=default_bg, foreground="white", bordercolor="white")

    tk_style.configure("TLabel", background=default_bg, foreground=default_fg)
    tk_style.configure("FixedWidthWhite.TLabel", font=("Latin Modern Mono", 12), background=default_bg, foreground="white")
    tk_style.configure("FixedWidthRed.TLabel", font=("Latin Modern Mono", 12), background=default_bg, foreground="red")

    tk_style.configure("TCheckbutton", background=default_bg, foreground=default_fg)
    tk_style.configure("TopFrame.TFrame", background=default_bg)
    tk_style.configure("ToolsFrame.TFrame", background=default_bg)
    tk_style.configure("CanvasFrame.TFrame", background=default_bg)

    tk_style.configure("TMessaging", background=default_bg, foreground=default_fg)

    app = App(tk_root)

    tk_root.mainloop()
