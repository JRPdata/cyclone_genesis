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
# ZOOM TO BASIN = z key (popup dialog to select which basin to zoom in)
# MEASURE (GEODESIC) = shift + mouse left click and drag
# ERASE MEASURE = right click
# VIEW NEXT OVERLAPPED HOVER POINT STATUS = n key  (this will not redraw points of overlapped storms, only update hover text)
# CIRCLE & ANNOTATE STORM EXTREMA = x key  (annotates either the selected storm in circle patch, or all storms in current view (zoom))
# CLEAR (ALL OR SINGLE BOX) STORM EXTREMA ANNOTATIONS = c key
# HIDE (MOUSE HOVERED) STORMS / SHOW ALL HIDDEN = h key (also hides tracks under selection loops)
# SAVE SCREENSHOT = p key
# TOGGLE RVOR CONTOURS = v key
# TOGGLE RVOR CONTOUR LABELS = l key
# SHOW RVOR LEVELS CHOOSER = V key
# TOGGLE SELECTION LOOPS MODE = s key (also use button); used for selecting storm tracks (to group as a storm) for analysis
#   CLEAR SELECTION LOOPS = c key (only in selection loop mode)
#   HIDE SELECTION LOOPS = h key (only in selection loop mode)
# SELECTION LOOP ON BASIN = b key (popup dialog to create a selection loop on a basin) (uses shapes/basin) (alternatively, draw by clicking 'Select')
# TOGGLE ANALYSIS MODE = a key (must have selection loop on tracks); NOTE: TRACK ANALYSIS ONLY PRESENTLY WORKS IN UTC TZ
#   SAVE FIGURE = s key (or p key) (saves current figure in analysis mode)
# PRINT TRACK STATS = i key (must be hovering; prints to terminal)
# DISPLAY SST DATA = 4 key (set DISPLAY_NETCDF to 'sst') (Day/Night SST from CoastWatch/NOAA)
# DISPLAY OHC DATA = 5 key (set DISPLAY_NETCDF to 'ohc') (OHC (26C) from the SOHC from NESDIS/NOAA)
# DISPLAY ISO26C DATA = 6 key (set DISPLAY_NETCDF to 'iso26C') (Depth of the 26 degree isobar from the SOHC from NESDIS/NOAA)
# DISPLAY NO NETCDF DATA = r key (set DISPLAY_NETCDF to None) (Don't display any NETCDF overlay)
# HIDE TRACKS BY DETIAL = k key (hide tracks by fields (vmax, lat/lon, etc) and valid time)
# SAVE HIDDEN TRACKS (SLOT 1) = 1 key (saves the current view - same hidden tracks; will not work after changing ensembles)
# SAVE HIDDEN TRACKS (SLOT 2) = 2 key (saves the current view - same hidden tracks)
# LOAD HIDDEN TRACKS (SLOT 1) = 8 key (re-hides the tracks saved in slot 1 pressing '1')
# LOAD HIDDEN TRACKS (SLOT 2) = 9 key ("" from slot 2 from pressing '2')
# BLIT MEAN TRACK = m key (temporarily, draw the mean track from the selected tracks)


# Click on (colored) legend for valid time days to cycle between what to display
#   (relative to (00Z) start of earliest model init day)
#   clicking on a legend box with a white edge toggles between:
#       1: (default) display all storm track valid times prior to selected (changes successive legend edges bright pink)
#       2: hide all storms with tracks with the first valid day later than selected (changes successive edges dark pink)
#       3: hide all points with tracks later than valid day
#   clicking on a pink/dark pink edge switches it to (1) white edge for all prior and (2) pink edge for days after
#  D- works differently in mode 3 (flips visible/invisible) for ADECK mode (to show/hide old best track data)

## MEASURE TOOL DOC
#   HOLDING SHIFT KEY TOGGLES UPDATING END POINT, ALLOWING ZOOMING IN/OUT:
#   ALLOWS FOR SINGLE MEASUREMENT WITHOUT NEEDING TO DRAG END POINTS
#   ZOOM IN TO PRECISION NEEDED TO SET START POINT, START MEASUREMENT (SHIFT CLICK AND MOTION), RELEASE SHIFT (AND CLICK)
#   ZOOM OUT, THEN ZOOM IN TO WHERE TO PLACE END POINT AND HOLD SHIFT AND MOUSE MOTION TO PLACE END POINT
#   RELEASE SHIFT WHEN END POINT POSITION SET. FREE TO ZOOM IN/OUT AFTERWARD



####### CONFIG
# Process the wind radii data from TCGEN ensembles
PROCESS_TCGEN_WIND_RADII = True
PROCESS_TCGEN_WIND_RADII = False

# how smooth (and also large) the (interpolated) 'cone' is
# how often should the wind field be interpolated (in seconds)
# every 10 minutes is fairly smooth
# (used by WindField class)
# to slow to interpret more than per hour currently
INTERP_PERIOD_SECONDS = 3600

# pressure statistics for NA, EP basins by likelihood of being categorized at that pressure (from cyclone-climatology notebook)
# used by presets for Analysis
output_cat_file_name = 'output_cat_values.json'

NETCDF_FOLDER_PATH = 'netcdf'

# Not sure about this data source as D26 == TCHP (so don't use)
#https://cwcgom.aoml.noaa.gov/erddap/griddap/aomlTCHP.html
TCHP_NC_PATH = 'netcdf/aomlTCHP_32f9_dc68_adee-2024-09-26.nc'

# https://www.ncei.noaa.gov/data/oceans/sohcs/2024/09/
# The North Pacific OHC is not masked properly (around -80W) so plot it before North America
OHC_NC_PATHS = [
    'netcdf/OHC-NPQG3_v1r0_blend_s202409130000000_e202409262359599_c202409260922177.nc',
    'netcdf/OHC-NAQG3_v1r0_blend_s202409130000000_e202409262359599_c202409260922256.nc',
    'netcdf/OHC-SPQG3_v1r0_blend_s202409130000000_e202409262359599_c202409260922501.nc'
]

# https://coastwatch.noaa.gov/pub/socd2/coastwatch/sst_blended/sst5km/daynight/ghrsst_ospo/2024/20241002000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended-GLOB-v02.0-fv01.0.nc
SST_NC_PATH = 'netcdf/20241005000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended-GLOB-v02.0-fv01.0.nc'

# Auto download overwrites the paths given above:
# Download latest OHC on startup
AUTO_DOWNLOAD_LATEST_OHC = True
# Download latest SST on startup
AUTO_DOWNLOAD_LATEST_SST = True

FINE_SST_BINS = False
# load netcdf files (regardless of display)
LOAD_NETCDF = True
# default netcdf display?
DISPLAY_NETCDF = None
#DISPLAY_NETCDF = 'd26'
#DISPLAY_NETCDF = 'tchp'
DISPLAY_NETCDF = 'ohc'
#DISPLAY_NETCDF = 'iso26C'
#DISPLAY_NETCDF = 'sst'
# Set to True or False to bin the NETCDF data
BIN_NETCDF_DATA = True

# how often (in minutes) to check for stale data in three classes: tcvitals, adeck, bdeck
#   for notification purposes only.
#      colors reload button red (a-deck), orange (b-deck), yellow (tcvitals) -without- downloading data automatically
#   checks modification date in any class of the three, and refreshes the entire class
#      timer resets after manual reloads
TIMER_INTERVAL_MINUTES = 30
# similar but only for genesis data: global-det (own tracker), and tcgen (NCEP/NOAA,ECMWF data from auto_update_cyclones_tc_gen.py)
LOCAL_TIMER_INTERVAL_MINUTES = 10

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
GRID_LINE_SPACING_DEGREES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 90.0]
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

# settable as a setting in the analysis window
ANALYSIS_TZ = "UTC"

# remove any from list not wanted visible for extrema annotations (x key)
#   closed_isobar_delta is only from tc_candidates.db
#      (it is the difference in pressure from the outermost closed isobar to the innermost closed isobar)
#       (i.e. how many concentric circles on a chart of pressure); or POUTER - MSLP in abdecks
# displayed_extremum_annotations = ["dt_start", "dt_end", "vmax10m", "mslp_value", "roci", "closed_isobar_delta"]
# displayed_extremum_annotations = ["dt_start", "vmax10m", "roci", "closed_isobar_delta"]

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
                                                                if param_keys[0] in x and x[param_keys[0]] and x[
                                                                    param_keys[0]] >= min_n), None),
    'index_of_first_<=_n': lambda lst, param_keys, max_n: next(((i, x[param_keys[0]]) for i, x in enumerate(lst)
                                                                if param_keys[0] in x and x[param_keys[0]] and x[
                                                                    param_keys[0]] <= max_n), None),
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
    'Earliest Named': lambda tc_candidate: annotations_comparison_func_dict['index_of_first_>=_n'](tc_candidate,
                                                                                                   ['vmax10m'], 34),
    'Peak Vmax': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['vmax10m']),
    'Min MSLP': lambda tc_candidate: annotations_comparison_func_dict['index_of_min'](tc_candidate, ['mslp_value']),
    'Peak ROCI': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['roci']),
    'Peak Isobar Delta': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate,
                                                                                               ['closed_isobar_delta']),
    'TC End': lambda tc_candidate: annotations_comparison_func_dict['index_of_max'](tc_candidate, ['valid_time'])
}
# formatting for annotation
#   key = short name, result_val is result from lambda comparison functions above
#   point tc_candidate[tc_point_index) passed to lambda with added parameter 'result_val'
#       {result_val} is val from annotations_comparison_func_dict evaluation
annotations_label_func_dict = {
    'TC Start': lambda point,
                       result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                   "\nSTART",
    'Earliest Named': lambda point,
                             result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                         "\nEARLIEST 34kt",
    'Peak Vmax': lambda point,
                        result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                    f"\nVMAX_10m: {result_val:.1f} kt",
    'Min MSLP': lambda point,
                       result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                   f"\nMSLP: {result_val:.1f} hPa",
    'Peak ROCI': lambda point,
                        result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                    f"\nROCI: {result_val:.0f} km",
    'Peak Isobar Delta': lambda point,
                                result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                            f"\nISOBAR_DELTA: {result_val:.0f} hPa",
    'TC End': lambda point, result_val: f"{point['model_name']} " + f"{point['valid_time'].strftime('%Y-%m-%d %HZ')}" +
                                        "\nEND"
}

annotations_color_level = {
    'TC Start': 1,
    'Earliest Named': 2,
    'Peak Vmax': 3,
    'Min MSLP': 0,
    'Peak ROCI': 0,
    'Peak Isobar Delta': 0,
    'TC End': 0
}

from datetime import datetime, timedelta
import pytz
def datetime_utcnow():
    return datetime.now(pytz.utc).replace(tzinfo=None)

# URLs
tcvitals_urls = [
    "https://ftp.nhc.noaa.gov/atcf/com/tcvitals",
    "https://hurricanes.ral.ucar.edu/repository/data/tcvitals_open/combined_tcvitals.{year}.dat".format(
        year=datetime_utcnow().year)
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

# SKIPS urls with string (if only want AL/EP/CP data
#SKIP_SOURCE = []
SKIP_SOURCES = ['ucar.edu']

for urls in [tcvitals_urls, adeck_urls, bdeck_urls]:
    # Filter out URLs containing SKIP_SOURCES substrings
    urls[:] = [url for url in urls if not any(skip_source in url for skip_source in SKIP_SOURCES)]

### END CONFIG ###
##################

# IMPORTS

# for screenshots
from PIL import ImageGrab

# for tracking modification date of source files
import copy
from dateutil import parser

# main utility imports
from collections import defaultdict
import gzip
import json
import numpy as np
import os
import requests
import traceback
import re

# for parsing ADT information
from bs4 import BeautifulSoup

# for interp. mean track
from scipy.interpolate import interp1d

# main app imports for map charts and plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import Divider, Size
# for analysis
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import colorspacious as cs
from geopy.distance import geodesic

# for mouse hover features
from rtree import index
# for cycling overlapped points (mouse hover status label)
from collections import OrderedDict

# for measurements (geodesic)
#   note: measures geodesic distance, not the 'path' distance that might show as a blue line
import cartopy.geodesic as cgeo
from shapely.geometry import LineString

# for plotting custom boundaries, rvor contour data
import geopandas as gpd

# for selection loop tool
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
# for selection loop tool
from matplotlib.patches import Polygon as MPLPolygon

# for tcgen/ensemble WindField / cone
from shapely.ops import unary_union
from geographiclib.geodesic import Geodesic
from matplotlib.path import Path
from shapely.geometry import Point
import antimeridian
import pandas as pd
from shapely.validation import make_valid
# TODO: remove
from shapely.errors import GEOSException
from shapely import is_valid_reason


# main app gui imports
import tkinter as tk
from tkinter import ttk, font

# config dialog
from tkinter import colorchooser
from tkinter import simpledialog

# for input databases
import sqlite3

# for performance optimizations
import matplotlib

# for plotting netcdf4 data
import netCDF4 as nc
import matplotlib.colors as mcolors

# Disable all pre-existing keymaps in mpl (conflict causes bugs)
for key in plt.rcParams.keys():
    if key.startswith('keymap'):
        plt.rcParams[key] = []

# Input data variables

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

# Get basin extents and polygons
basins_gdf = gpd.read_file('shapes/basins.shp')
basin_extents = {}
basin_polys = {}

shape_basin_names_to_threshold_names_map = {
    'NATL': 'NA',
    'WPAC': 'WP',
    'IO': 'IO',
    'SH': 'SH',
    'EPAC': 'EP'
}
for basin_name in basins_gdf['basin_name'].unique():
    # Get the extent of the current basin
    geom = basins_gdf.loc[basins_gdf['basin_name'] == basin_name, 'geometry']
    # Store the extent in the dictionary
    basin_short_name = shape_basin_names_to_threshold_names_map[basin_name]
    geom_u_bounds = geom.unary_union.bounds
    min_lon, min_lat, max_lon, max_lat = geom_u_bounds
    if len(geom) > 1:
        for row in list(geom):
            min_lon, min_lat, max_lon, max_lat = row.bounds
            if min_lon <= -180:
                suffix = '-W'
            elif max_lon >= 180:
                suffix = '-E'

            basin_short_name = shape_basin_names_to_threshold_names_map[basin_name] + suffix
            if basin_short_name in basin_extents:
                basin_extents[basin_short_name] = basin_extents[basin_short_name].union(row.bounds)
                basin_polys[basin_short_name] = basin_polys[basin_short_name].union(row.bounds)
            else:
                basin_extents[basin_short_name] = row.bounds
                basin_polys[basin_short_name] = gpd.GeoSeries([row])
    else:
        basin_short_name = shape_basin_names_to_threshold_names_map[basin_name]
        basin_extents[basin_short_name] = geom_u_bounds
        basin_polys[basin_short_name] = geom

### plot greater houston boundary
# Load the shapefile using Geopandas
# from census shapefiles
# shapefile_path = "tl_2023_us_cbsa/tl_2023_us_cbsa.shp"
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

### plot Shanghai boundary
# Load the shapefile using Geopandas
# from gadm.org (china, level 2 shapefile)
# shapefile_path = "tl_2023_us_cbsa/tl_2023_us_cbsa.shp"
#shapefile_path = 'gadm41_CHN_shp/gadm41_CHN_2.shp'

try:
    tmp_gdf = gpd.read_file(shapefile_path)
    # Filter the GeoDataFrame to only include Shanghai
    print(tmp_gdf.keys())
    shanghai_gdf = tmp_gdf[tmp_gdf['NAME_1'] == 'Shanghai']
    if shanghai_gdf.empty:
        raise ValueError("Shanghai area not found in the shapefile")
    # Ensure the CRS matches that of the Cartopy map (PlateCarree)
    custom_gdf = shanghai_gdf.to_crs(ccrs.PlateCarree().proj4_init)
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

# Program Settings

matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = CHART_DPI

# own tracker for genesis data - this accesses data by model and storm (internal component id)
tc_candidates_db_file_path = 'tc_candidates.db'

# tcgen data (processed from NCEP/NOAA, ECMWF) by model and storm
tc_candidates_tcgen_db_file_path = 'tc_candidates_tcgen.db'

# global_det models (keys are models in own tracker), and tcgen models
model_data_folders_by_model_name = {
    'GFS': '/home/db/metview/JRPdata/globalmodeldata/gfs',
    'ECM': '/home/db/metview/JRPdata/globalmodeldata/ecm',
    'CMC': '/home/db/metview/JRPdata/globalmodeldata/cmc',
    'NAV': '/home/db/metview/JRPdata/globalmodeldata/nav',
    'EPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/eps-tcgen',
    'GEFS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen',
    'GEPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/geps-tcgen',
    'FNMOC-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/fnmoc-tcgen',
    'GEFS-ATCF': '/home/db/metview/JRPdata/globalmodeldata/gefs-atcf',
    'GEPS-ATCF': '/home/db/metview/JRPdata/globalmodeldata/geps-atcf',
    'FNMOC-ATCF': '/home/db/metview/JRPdata/globalmodeldata/fnmoc-atcf',
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
    'AVNX',
    "CC00",
    'CEM2',
    'CEMI',
    'CEMN',
    'CMC',
    'CMC2',
    'CMCI',
    "CP01",
    "CP02",
    "CP03",
    "CP04",
    "CP05",
    "CP06",
    "CP07",
    "CP08",
    "CP09",
    "CP10",
    "CP11",
    "CP12",
    "CP13",
    "CP14",
    "CP15",
    "CP16",
    "CP17",
    "CP18",
    "CP19",
    "CP20"
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
    "NC00",
    'NGX',
    'NGX2',
    'NGXI',
    'NNIB',
    'NNIC',
    "NP01",
    "NP02",
    "NP03",
    "NP04",
    "NP05",
    "NP06",
    "NP07",
    "NP08",
    "NP09",
    "NP10",
    "NP11",
    "NP12",
    "NP13",
    "NP14",
    "NP15",
    "NP16",
    "NP17",
    "NP18",
    "NP19",
    "NP20",
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
    'NVG2',
    'NVGI',
    'NVGM',
    'BEST',
    'TCVITALS'
]
regional_models = ['HFSA', 'HFSB', 'HFAI', 'HFA2', 'HFBI', 'HFB2',
                   'HMON', 'HMNI', 'HMN2',
                   'HWRF', 'HWF2', 'HWFI', 'HWFI', 'BEST', 'TCVITALS']
official_models = ['OFCI', 'OFCL', 'BEST', 'TCVITALS']
consensus_models = ['ICON', 'IVCN', 'RVCN', 'NNIC', 'NNIB', 'BEST', 'TCVITALS']

# atcf models (don't include tcvitals, best track as that messes with statistics)
gefs_atcf_members = [
    'AVNX',
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
    'AP30'
]

geps_atcf_members = [
    "CMC",
    "CC00",
    "CP01",
    "CP02",
    "CP03",
    "CP04",
    "CP05",
    "CP06",
    "CP07",
    "CP08",
    "CP09",
    "CP10",
    "CP11",
    "CP12",
    "CP13",
    "CP14",
    "CP15",
    "CP16",
    "CP17",
    "CP18",
    "CP19",
    "CP20"
]

fnmoc_atcf_members = [
    "NGX",
    "NC00",
    "NP01",
    "NP02",
    "NP03",
    "NP04",
    "NP05",
    "NP06",
    "NP07",
    "NP08",
    "NP09",
    "NP10",
    "NP11",
    "NP12",
    "NP13",
    "NP14",
    "NP15",
    "NP16",
    "NP17",
    "NP18",
    "NP19",
    "NP20"
]
all_atcf_ens_members = gefs_atcf_members + geps_atcf_members + fnmoc_atcf_members

atcf_ens_models_by_ensemble = {
    'GEFS-ATCF': gefs_atcf_members,
    'GEPS-ATCF': geps_atcf_members,
    'FNMOC-ATCF': fnmoc_atcf_members,
    'ALL-ATCF': all_atcf_ens_members
}
atcf_ens_num_models_by_ensemble = {
    'GEFS-ATCF': len(gefs_atcf_members),
    'GEPS-ATCF': len(geps_atcf_members),
    'FNMOC-ATCF': len(fnmoc_atcf_members),
    'ALL-ATCF': len(all_atcf_ens_members)
}
atcf_ens_names_in_all = ['GEFS-ATCF', 'GEPS-ATCF', 'FNMOC-ATCF']

# for statistics in case one of the ensembles is under maintenance (not reporting)
# TODO: ADD BACK FNMOC WHEN WORKING AGAIN
atcf_active_ensembles_all = ['GEFS-ATCF', 'GEPS-ATCF']

# Create a dictionary to map model names to ensemble names
atcf_ens_model_name_to_ensemble_name = {}
for list_name, lst in zip(['GEFS-ATCF', 'GEPS-ATCF', 'FNMOC-ATCF'], [gefs_atcf_members, geps_atcf_members, fnmoc_atcf_members]):
    for model_name in lst:
        atcf_ens_model_name_to_ensemble_name[model_name] = list_name

# where the atcf ens (unofficial) adeck files are (from auto_generate_adecks_from_atcf.py)
adeck_folder = '/home/db/metview/JRPdata/globalmodeldata/adeck-ens-atcf'

# tcgen models
# Deterministic, control and perturbation members of the ensembles
gefs_members = ['GFSO', 'AC00', 'AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AP21', 'AP22', 'AP23', 'AP24', 'AP25', 'AP26', 'AP27', 'AP28', 'AP29', 'AP30']
geps_members = ['CMC', 'CC00', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06', 'CP07', 'CP08', 'CP09', 'CP10', 'CP11', 'CP12', 'CP13', 'CP14', 'CP15', 'CP16', 'CP17', 'CP18', 'CP19', 'CP20']
eps_members = ['ECHR', 'ECME', 'EE01', 'EE02', 'EE03', 'EE04', 'EE05', 'EE06', 'EE07', 'EE08', 'EE09', 'EE10', 'EE11', 'EE12', 'EE13', 'EE14', 'EE15', 'EE16', 'EE17', 'EE18', 'EE19', 'EE20', 'EE21', 'EE22', 'EE23', 'EE24', 'EE25', 'EE26', 'EE27', 'EE28', 'EE29', 'EE30', 'EE31', 'EE32', 'EE33', 'EE34', 'EE35', 'EE36', 'EE37', 'EE38', 'EE39', 'EE40', 'EE41', 'EE42', 'EE43', 'EE44', 'EE45', 'EE46', 'EE47', 'EE48', 'EE49', 'EE50']
fnmoc_members = ['NGX', 'NC00', 'NP01', 'NP02', 'NP03', 'NP04', 'NP05', 'NP06', 'NP07', 'NP08', 'NP09', 'NP10', 'NP11', 'NP12', 'NP13', 'NP14', 'NP15', 'NP16', 'NP17', 'NP18', 'NP19', 'NP20']
all_tcgen_members = gefs_members + geps_members + eps_members + fnmoc_members

tcgen_models_by_ensemble = {
    'GEFS-TCGEN': gefs_members,
    'GEPS-TCGEN': geps_members,
    'EPS-TCGEN': eps_members,
    'FNMOC-TCGEN': fnmoc_members,
    'ALL-TCGEN': all_tcgen_members
}
tcgen_num_models_by_ensemble = {
    'GEFS-TCGEN': len(gefs_members),
    'GEPS-TCGEN': len(geps_members),
    'EPS-TCGEN': len(eps_members),
    'FNMOC-TCGEN': len(fnmoc_members),
    'ALL-TCGEN': len(all_tcgen_members)
}
tcgen_names_in_all = ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']

# for statistics in case one of the ensembles is under maintenance (not reporting)
# TODO: ADD BACK FNMOC WHEN WORKING AGAIN
tcgen_active_ensembles_all = ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN']

'''
# Member counts of above
GEFS-TCGEN : 32
GEPS-TCGEN : 22
EPS-TCGEN : 52
FNMOC-TCGEN : 22
ALL-TCGEN : 128
'''
global_det_members = ['GFS', 'CMC', 'ECM', 'NAV']
num_global_det_members = len(global_det_members)
# TODO: modify when NAVGEM WORKS AGAIN
active_global_det_members = ['GFS', 'CMC', 'ECM']
num_active_global_det_members = {
    'GLOBAL-DET': len(active_global_det_members)
}
global_det_active_ensembles = ['GLOBAL-DET']
global_det_lookup = {}
for member in global_det_lookup:
    global_det_lookup[member] = 'GLOBAL-DET'

# Create a dictionary to map model names to ensemble names
model_name_to_ensemble_name = {}
for list_name, lst in zip(['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN'], [gefs_members, geps_members, eps_members, fnmoc_members]):
    for model_name in lst:
        model_name_to_ensemble_name[model_name] = list_name

# add these to group them under the same organization (ensemble) even though they are not part of the ensemble
model_name_to_ensemble_name['GFS'] = 'GEFS-TCGEN'
model_name_to_ensemble_name['ECM'] = 'EPS-TCGEN'
model_name_to_ensemble_name['NAV'] = 'FNMOC-TCGEN'

# all genesis members across both datasets: global-det + all_tcgen (overlap with CMC)
all_genesis_names = list(set(all_tcgen_members + global_det_members))

# Helper functions for input data

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

# for managing tracks (loop selections on them) that cross the antimeridian
def cut_line_string_at_antimeridian(line_string):
    coordinates = list(line_string.coords)
    cut_lines = []
    current_line = [coordinates[0]]

    for i in range(1, len(coordinates)):
        prev_lon, prev_lat = current_line[-1]
        lon, lat = coordinates[i]

        # Check if the line crosses the antimeridian
        if abs(lon - prev_lon) > 180:
            # Calculate the intersection latitude at the antimeridian
            denom = (180.0 - abs(prev_lon) + 180.0 - abs(lon))
            if denom != 0:
                intersection_lat = prev_lat + (lat - prev_lat) * (180.0 - abs(prev_lon)) / (180.0 - abs(prev_lon) + 180.0 - abs(lon))
            else:
                intersection_lat = 0.0
            if prev_lon > 0 and lon < 0:
                # Westward crossing
                current_line.append((180, intersection_lat))
                cut_lines.append(LineString(current_line))
                current_line = [(-180, intersection_lat), (lon, lat)]
            elif prev_lon < 0 and lon > 0:
                # Eastward crossing
                current_line.append((-180, intersection_lat))
                cut_lines.append(LineString(current_line))
                current_line = [(180, intersection_lat), (lon, lat)]
        else:
            # No crossing, simply add the point
            current_line.append((lon, lat))

    # Add the final segment
    cut_lines.append(LineString(current_line))

    return cut_lines

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

def download_latest_ohc_nc_files():
    local_file_paths = []
    # Get current year and month
    dt_now = datetime.now()
    current_year = dt_now.year
    current_month = dt_now.month

    # Try to send request and get HTML response
    for attempt in range(2):
        # Construct URL
        url = f'https://www.ncei.noaa.gov/data/oceans/sohcs/{current_year:04}/{current_month:02}/'
        # Send request and get HTML response
        response = requests.get(url)
        if response.ok:
            break
        else:
            if current_month == 1:
                current_year -= 1
                current_month = 12
            else:
                current_month -= 1
    else:
        # If both attempts fail, return
        print("Failed to get list of latest OHC files...")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links ending with .nc
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]

    # Create a dictionary to store the links
    link_dict = {}

    # Loop through the links and extract datetime and key
    for link in links:
        filename = link.split('/')[-1]
        key = filename.split('_')[0]
        creation = re.search(r'_c(\d{8})(\d{4})(\d{3})\.nc', filename)
        datetime_obj = datetime.strptime(creation.groups()[0] + creation.groups()[1] + creation.groups()[2], '%Y%m%d%H%M%f')
        if key not in link_dict:
            link_dict[key] = {}
        link_dict[key][datetime_obj] = link

    # Process the keys in reverse alphabetical order
    # This is to fix the bug in the North Pacific mask for OHC
    keys = sorted(link_dict.keys(), reverse=True)

    # Get the latest links for each key
    latest_links = []
    for key in keys:
        datetime_objs = sorted(link_dict[key].keys())
        latest_link = link_dict[key][datetime_objs[-1]]
        latest_links.append(latest_link)

    # Download the latest links
    for link in latest_links: # Limit to 3 files
        filename = link.split('/')[-1]
        dest_path = os.path.join(NETCDF_FOLDER_PATH, filename)
        file_ok = False
        if not os.path.exists(dest_path):
            os.makedirs(NETCDF_FOLDER_PATH, exist_ok=True)
            response = requests.get(url + link, stream=True)
            if response.ok:
                print("Downloading latest OHC file: ", link)
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                file_ok = True
        else:
            file_ok = True

        if file_ok:
            local_file_paths.append(dest_path)

    return local_file_paths

def download_latest_sst_nc_file():
    dt_now = datetime.now()
    # Start 1 day forward to account for clock/TZ mismatch
    dt_start = dt_now + timedelta(days=1)

    # Try to send request and get the latest SST NC file
    # Don't try to get files older than ~ 10 days
    for attempt in range(10):
        dt = dt_start - timedelta(days=attempt)
        year = dt.year
        month = dt.month
        day = dt.day
        url = f'https://coastwatch.noaa.gov/pub/socd2/coastwatch/sst_blended/sst5km/daynight/ghrsst_ospo/{year}/{year}{month:02}{day:02}000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended-GLOB-v02.0-fv01.0.nc'
        filename = url.split('/')[-1]
        dest_path = os.path.join(NETCDF_FOLDER_PATH, filename)
        os.makedirs(NETCDF_FOLDER_PATH, exist_ok=True)
        file_ok = False
        if not os.path.exists(dest_path):
            response = requests.get(url)
            if response.ok:
                print("Dowloading latest SST file:", filename)
                filename = url.split('/')[-1]
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return dest_path
        else:
            # Already have latest
            return dest_path

    print("Failed to get list of latest SST file...")
    return None

def get_adt_urls():
    # Define the URLs
    ospo_url = "https://www.ospo.noaa.gov/products/ocean/tropical/adt.html"
    cimss_url = "https://tropic.ssec.wisc.edu/real-time/adt/adt.html"

    # Send HTTP requests and get the HTML responses
    ospo_response = requests.get(ospo_url)
    cimss_response = requests.get(cimss_url)

    # Parse the HTML content using BeautifulSoup
    ospo_soup = BeautifulSoup(ospo_response.content, 'html.parser')
    cimss_soup = BeautifulSoup(cimss_response.content, 'html.parser')

    # Find all ADT URLs on the OSPO page
    ospo_adt_urls = []
    for link in ospo_soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/tropical-data/') and href.endswith('.txt') and '/adt/' in href:
            adt_url = f"https://www.ospo.noaa.gov{href}"
            ospo_adt_urls.append(adt_url)

    # Find all ADT URLs on the WISC page
    cimss_adt_urls = []
    for link in cimss_soup.find_all('a', href=True):
        href = link['href']
        match = re.search(r'odt(\d{2})([A-Z]).html', href)
        if match:
            storm_number = match.group(1)
            storm_letter = match.group(2)
            adt_url = f"https://tropic.ssec.wisc.edu/real-time/adt/{storm_number}{storm_letter}-list.txt"
            cimss_adt_urls.append(adt_url)

    return ospo_adt_urls, cimss_adt_urls

# Function to get the corresponding A-Deck and B-Deck files for the identified storms
# adeck2 is the unofficial adeck from auto_generate_adecks_from_atcf
def get_deck_files(storms, urls_a, urls_b, do_update_adeck, do_update_adeck2, do_update_bdeck):
    adeck = defaultdict(dict)
    bdeck = defaultdict(dict)
    year = datetime_utcnow().year
    most_recent_model_dates = defaultdict(lambda: datetime.min)
    most_recent_bdeck_dates = defaultdict(lambda: datetime.min)
    dt_mods_adeck = {}
    dt_mods_adeck2 = {}
    dt_mods_bdeck = {}
    # keep track of which ensembles we already have/don't have and their model date
    # this allows us to prune members from old ensembles that no longer show genesis/tracks
    atcf_active_ensembles_have = {}

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
                                    init_datetime = datetime.fromisoformat(ab_deck_line_dict['init_time'])
                                    if storm_id not in adeck.keys():
                                        adeck[storm_id] = {}
                                    if model_id not in adeck[storm_id].keys():
                                        adeck[storm_id][model_id] = {}
                                    ab_deck_line_dict['basin'] = basin_id.upper()
                                    if model_id in atcf_ens_model_name_to_ensemble_name:
                                        ens_name = atcf_ens_model_name_to_ensemble_name[model_id]
                                        atcf_active_ensembles_have[ens_name] = init_datetime

                                    adeck[storm_id][model_id][valid_datetime.isoformat()] = ab_deck_line_dict
                                elif model_date >= (latest_date - timedelta(hours=6)):
                                    # GEFS members ATCF is reported late by 6 hours...
                                    ab_deck_line_dict = ab_deck_line_to_dict(line)
                                    model_id = ab_deck_line_dict['TECH']
                                    init_datetime = datetime.fromisoformat(ab_deck_line_dict['init_time'])
                                    # allow only late GEFS members:
                                    if model_id[0:2] in ['AC', 'AP']:
                                        valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])

                                        skip_old = False
                                        if model_id not in adeck[storm_id].keys():
                                            adeck[storm_id][model_id] = {}
                                        else:
                                            prev_valid_time_str = list(adeck[storm_id][model_id].keys())[0]
                                            if prev_valid_time_str and 'init_time' in adeck[storm_id][model_id][prev_valid_time_str]:
                                                prev_init_time = datetime.fromisoformat(adeck[storm_id][model_id][prev_valid_time_str]['init_time'])
                                                if prev_init_time > init_datetime:
                                                    skip_old = True

                                        ab_deck_line_dict['basin'] = basin_id.upper()
                                        if not skip_old:
                                            if model_id in atcf_ens_model_name_to_ensemble_name:
                                                ens_name = atcf_ens_model_name_to_ensemble_name[model_id]
                                                atcf_active_ensembles_have[ens_name] = init_datetime

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
        if do_update_adeck2:
            storm_id = f"{basin_id}{storm_number}{year}"
            global adeck_folder
            local_filename = os.path.join(adeck_folder, f"a{storm_id.lower()}.dat")
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
                        if model_date > latest_date:
                            latest_date = model_date

                if latest_date > most_recent_model_dates[storm_id]:
                    most_recent_model_dates[storm_id] = latest_date

                # ensemble members will be late
                if most_recent_model_dates[storm_id] >= (latest_date - timedelta(hours=6)):
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
                                init_datetime = datetime.fromisoformat(ab_deck_line_dict['init_time'])
                                if storm_id not in adeck.keys():
                                    adeck[storm_id] = {}
                                if model_id not in adeck[storm_id].keys():
                                    adeck[storm_id][model_id] = {}
                                ab_deck_line_dict['basin'] = basin_id.upper()
                                if model_id in atcf_ens_model_name_to_ensemble_name:
                                    ens_name = atcf_ens_model_name_to_ensemble_name[model_id]
                                    atcf_active_ensembles_have[ens_name] = init_datetime
                                adeck[storm_id][model_id][valid_datetime.isoformat()] = ab_deck_line_dict
                            elif model_date >= (latest_date - timedelta(hours=6)):
                                # GEPS/EPS/FNMOC members ATCF are usually later than GEFS (allow up to 6 hours late)
                                ab_deck_line_dict = ab_deck_line_to_dict(line)
                                model_id = ab_deck_line_dict['TECH']
                                # allow late members
                                valid_datetime = datetime.fromisoformat(ab_deck_line_dict['valid_time'])
                                init_datetime = datetime.fromisoformat(ab_deck_line_dict['init_time'])
                                if storm_id not in adeck.keys():
                                    adeck[storm_id] = {}

                                valid_time_str = valid_datetime.isoformat()
                                skip_old = False
                                if model_id not in adeck[storm_id].keys():
                                    adeck[storm_id][model_id] = {}
                                else:
                                    prev_valid_time_str = list(adeck[storm_id][model_id].keys())[0]
                                    if prev_valid_time_str and 'init_time' in adeck[storm_id][model_id][prev_valid_time_str]:
                                        prev_init_time = datetime.fromisoformat(adeck[storm_id][model_id][prev_valid_time_str]['init_time'])
                                        if prev_init_time > init_datetime:
                                            skip_old = True

                                ab_deck_line_dict['basin'] = basin_id.upper()
                                if not skip_old:
                                    if model_id in atcf_ens_model_name_to_ensemble_name:
                                        ens_name = atcf_ens_model_name_to_ensemble_name[model_id]
                                        atcf_active_ensembles_have[ens_name] = init_datetime
                                    adeck[storm_id][model_id][valid_time_str] = ab_deck_line_dict

                dt_mods_adeck2[local_filename] = dt_mod
            except OSError as e:
                traceback.print_exc()
                print(f"OSError opening/reading file: {e}")
            except UnicodeDecodeError as e:
                traceback.print_exc()
                print(f"UnicodeDecodeError: {e}")
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to read from {adeck_folder}: {e}")

        # Prune old ensemble member points that are older than current ensemble members present
        for ens_name, model_members in atcf_ens_models_by_ensemble.items():
            if ens_name not in atcf_active_ensembles_have:
                continue

            ens_init_date = atcf_active_ensembles_have[ens_name]
            for model_name in model_members:
                if model_name not in adeck[storm_id]:
                    continue
                valid_times = list(adeck[storm_id][model_name].keys())
                if len(valid_times) == 0:
                    continue
                for prev_valid_time_str in valid_times:
                    if prev_valid_time_str and 'init_time' in adeck[storm_id][model_name][prev_valid_time_str]:
                        prev_init_time = datetime.fromisoformat(adeck[storm_id][model_name][prev_valid_time_str]['init_time'])
                        if prev_init_time < ens_init_date:
                            # this member point is not present in latest ensemble
                            del adeck[storm_id][model_name][prev_valid_time_str]

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
                                    ab_deck_line_dict['basin'] = basin_id.upper()
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
    return dt_mods_adeck, dt_mods_adeck2, dt_mods_bdeck, adeck, bdeck

def get_modification_date_from_header(response_headers):
    try:
        # Assume already done status code check
        modification_date = response_headers.get('Last-Modified')
        dt_offset_aware = parser.parse(modification_date)
        dt_offset_native = dt_offset_aware.astimezone().replace(tzinfo=None)
        return dt_offset_native
    except:
        return None

# Function to get the most recent records for each storm from TCVitals files
def get_recent_storms(urls):
    storms = {}
    dt_mods_tcvitals = {}
    current_time = datetime_utcnow()
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
                            'data': line,
                            'basin': basin
                        }

            dt_mods_tcvitals[url] = dt_mod

    recent_storms = {}
    for storm_id, val in storms.items():
        data = val['data']
        storm_dict = tcvitals_line_to_dict(data)
        storm_dict['basin'] = val['basin']
        recent_storms[storm_id] = storm_dict
    return dt_mods_tcvitals, recent_storms

# get list of completed TC candidates
# handle own genesis tracker data and tcgen data processed from NCEP, ECMWF
def get_tc_candidates_at_or_before_init_time(genesis_previous_selected, interval_end):
    if type(interval_end) == str:
        interval_end = datetime.fromisoformat(interval_end)

    all_retrieved_data = []  # List to store data from all rows
    conn = None
    model_names = list(model_data_folders_by_model_name.keys())
    model_init_times = {}

    model_completed_times = {}
    # list of completed times per ensemble for member completion for a recent cycle (may spread across more than one cycle)
    ensemble_completed_times = {}
    # get whether the ensemble is complete at a single cycle (only one init_time and ens_status is complete)
    completed_ensembles = {}
    # earliest init time in most recent ensemble
    earliest_recent_ensemble_init_times = {}
    latest_ensemble_init_time = {}

    tcgen_ensemble = False
    ensemble_names = []
    if genesis_previous_selected == 'GLOBAL-DET':
        db_file_path = tc_candidates_db_file_path
        # filter out all model_names from tcgen and keep all models from own tracker
        model_names = [model_name for model_name in model_names if model_name[-5:] != 'TCGEN']
    else:
        tcgen_ensemble = True
        db_file_path = tc_candidates_tcgen_db_file_path
        if genesis_previous_selected == 'ALL-TCGEN':
            # filter out all model_names from own tracker and keep all tcgen models
            ensemble_names = [ensemble_name for ensemble_name in tcgen_models_by_ensemble.keys() if ensemble_name != 'ALL-TCGEN']
            model_names = tcgen_models_by_ensemble['ALL-TCGEN']
        else:
            # pick the selected ensemble
            ensemble_names = [genesis_previous_selected]
            model_names = tcgen_models_by_ensemble[genesis_previous_selected]

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        for model_name in model_names:
            cursor.execute(
                'SELECT DISTINCT init_date, completed_date FROM completed WHERE model_name = ? AND init_date <= ? ORDER BY init_date DESC LIMIT 1',
                (model_name, datetime.isoformat(interval_end)))
            results = cursor.fetchall()
            if results:
                for row in results:
                    init_time = row[0]
                    completed_time = datetime.fromisoformat(row[1])
                    model_init_times[model_name] = init_time
                    model_completed_times[model_name] = completed_time
                    init_date = init_time

                    # retrieve the candidates at init_time and sort by valid date (descending)
                    cursor.execute(
                        'SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE model_name = ? AND init_date = ? ORDER BY start_valid_date DESC',
                        (model_name, init_date))
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

        for ensemble_name in ensemble_names:
            ensemble_init_times = set()
            ensemble_completed_times_by_init_date = {}
            for model_name, model_init_time in model_init_times.items():
                if model_name not in model_name_to_ensemble_name:
                    print("Warning could not find ensemble name for model name (skipping):", model_name)
                    continue
                if ensemble_name == model_name_to_ensemble_name[model_name]:
                    ensemble_init_times.add(model_init_time)
                    if model_init_time not in ensemble_completed_times_by_init_date:
                        ensemble_completed_times_by_init_date[model_init_time] = []
                    ensemble_completed_times_by_init_date[model_init_time].append(model_completed_times[model_name])
            if not ensemble_init_times:
                continue

            # for tcgen case, handle case when no predictions from a member but have a complete ensemble
            if tcgen_ensemble:
                cursor.execute(
                    f'SELECT init_date FROM ens_status WHERE ensemble_name = ? AND completed = 1 AND init_date <= ? ORDER BY init_date DESC LIMIT 1',
                    (ensemble_name, datetime.isoformat(interval_end)))
                results = cursor.fetchall()
                ens_is_completed = 0
                if results:
                    for row in results:
                        ens_init_date = row[0]
                        if ens_init_date:
                            if ens_init_date not in ensemble_init_times:
                                if ens_init_date > max(ensemble_init_times):
                                    # latest complete ensemble has no predictions from a member
                                    # add the latest date so we can prune the old models for it eventually below
                                    ensemble_init_times.add(ens_init_date)

            num_init_time_ens_is_complete = 0
            init_complete_times = set()
            for ensemble_init_time in ensemble_init_times:
                cursor.execute(
                    f'SELECT completed FROM ens_status WHERE init_date = ? AND ensemble_name = ? ORDER BY init_date DESC LIMIT 1',
                    (ensemble_init_time, ensemble_name))
                results = cursor.fetchall()
                ens_is_completed = 0
                if results:
                    for row in results:
                        ens_is_completed = row[0]

                if ens_is_completed:
                    init_complete_times.add(ensemble_init_time)

                if ensemble_name not in ensemble_completed_times:
                    ensemble_completed_times[ensemble_name] = []

                if ensemble_init_time in ensemble_completed_times_by_init_date:
                    ensemble_completed_time = max(ensemble_completed_times_by_init_date[ensemble_init_time])
                    ensemble_completed_times[ensemble_name].append(ensemble_completed_time)

                num_init_time_ens_is_complete += int(ens_is_completed)

            if num_init_time_ens_is_complete == 1 and len(ensemble_init_times) == 1:
                completed_ensembles[ensemble_name] = ensemble_completed_times[ensemble_name][0]
                earliest_recent_ensemble_init_times[ensemble_name] = ensemble_init_time
                latest_ensemble_init_time[ensemble_name] = ensemble_init_time
            elif num_init_time_ens_is_complete > 1 and len(ensemble_init_times) > 1:
                # old members of the ensemble in data, yet complete ensemble (to be pruned)
                completed_ensembles[ensemble_name] = max(ensemble_completed_times[ensemble_name])
                # disregard earlier members as they will be pruned
                earliest_recent_ensemble_init_times[ensemble_name] = max(init_complete_times)
                latest_ensemble_init_time[ensemble_name] = max(init_complete_times)
            else:
                # incomplete ensemble?
                if init_complete_times is not None and len(init_complete_times) != 0:
                    earliest_recent_ensemble_init_times[ensemble_name] = min(init_complete_times)
                    latest_ensemble_init_time[ensemble_name] = max(init_complete_times)
                else:
                    latest_ensemble_init_time[ensemble_name] = None
                    earliest_recent_ensemble_init_times[ensemble_name] = None

            # if a tcgen ensemble, we need to mark any candidates that don't belong to this init time to drop them
            # otherwise we will get models from a previous run if they have no predictions this run
            if tcgen_ensemble:
                # first mark ensemble names for models
                max_ens_init_time = latest_ensemble_init_time[ensemble_name]
                for i, tc_candidate in enumerate(all_retrieved_data):
                    model_name = tc_candidate['model_name']
                    model_ens_name = None
                    if 'ensemble_name' in tc_candidate:
                        model_ens_name = tc_candidate['ensemble_name']
                    elif model_name in model_name_to_ensemble_name:
                        model_ens_name = model_name_to_ensemble_name[model_name]
                        all_retrieved_data[i]['ensemble_name'] = model_ens_name

                    if model_ens_name and model_ens_name == ensemble_name:
                        if model_ens_name in completed_ensembles:
                            all_retrieved_data[i]['in_complete_ensemble'] = True
                        # max ens init time not set right for no prediction ensemble (FNMOC)
                        if tc_candidate['model_timestamp'] != max_ens_init_time:
                            all_retrieved_data[i]['old_ensemble_member'] = True
                        else:
                            all_retrieved_data[i]['old_ensemble_member'] = False

        if tcgen_ensemble:
            # drop old ensemble members in complete ensembles
            pruned_all_retrieved_data = []
            for tc_candidate in all_retrieved_data:
                if 'old_ensemble_member' in tc_candidate and 'in_complete_ensemble' in tc_candidate and tc_candidate['old_ensemble_member']:
                    # prune this old member in a complete ensemble (no predictions for the tcgen member in this complete run)
                    continue
                pruned_all_retrieved_data.append(tc_candidate)

            all_retrieved_data = pruned_all_retrieved_data

    except sqlite3.Error as e:
        print(f"SQLite error (get_tc_candidates_from_valid_time): {e}")
    finally:
        if conn:
            conn.close()

    # completed times are datetimes while init times are strings
    # earliest_recent_ensemble_init_times denotes the earliest member init time among all most recent members' init times (or if its an empty ensemble last complete run time)
    return model_init_times, earliest_recent_ensemble_init_times, model_completed_times, ensemble_completed_times, completed_ensembles, all_retrieved_data

# get list of completed TC candidates
def get_tc_candidates_from_valid_time(genesis_previous_selected, interval_start):
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    # either model names for own tracker, or ensemble names
    model_names = list(model_data_folders_by_model_name.keys())
    model_init_times = {}

    model_completed_times = {}
    # list of completed times per ensemble for member completion for a recent cycle (may spread across more than one cycle)
    ensemble_completed_times = {}
    # get whether the ensemble is complete at a single cycle (only one init_time and ens_status is complete)
    completed_ensembles = {}

    ensemble_names = []
    if genesis_previous_selected == 'GLOBAL-DET':
        db_file_path = tc_candidates_db_file_path
        # filter out all model_names from tcgen and keep all models from own tracker
        model_names = [model_name for model_name in model_names if model_name[-5:] != 'TCGEN']
    else:
        db_file_path = tc_candidates_tcgen_db_file_path
        if genesis_previous_selected == 'ALL-TCGEN':
            # filter out all model_names from own tracker and keep all tcgen models
            ensemble_names = [ensemble_name for ensemble_name in tcgen_models_by_ensemble.keys() if ensemble_name != 'ALL-TCGEN']
            model_names = tcgen_models_by_ensemble['ALL-TCGEN']
        else:
            # pick the selected ensemble
            ensemble_names = [genesis_previous_selected]
            model_names = tcgen_models_by_ensemble[genesis_previous_selected]

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        for model_name in model_names:
            cursor.execute(
                f'SELECT DISTINCT init_date, completed_date FROM completed WHERE model_name = ? ORDER BY init_date DESC LIMIT 1',
                (model_name,))
            results = cursor.fetchall()
            if results:
                for row in results:
                    init_time = row[0]
                    completed_time = datetime.fromisoformat(row[1])
                    model_init_times[model_name] = init_time
                    model_completed_times[model_name] = completed_time

                    init_date = init_time
                    model_init_times[model_name] = init_date
                    cursor.execute(
                        'SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE model_name = ? AND init_date = ? AND start_valid_date >= ? ORDER BY start_valid_date DESC',
                        (model_name, init_date, interval_start))
                    # Query all rows from the 'disturbances' table and order by 'model_timestamp'
                    results = cursor.fetchall()
                    if results:
                        # Process data for each row
                        for row2 in results:
                            model_name, init_date, start_valid_date, ws_max_10m, json_data = row2
                            retrieved_data = {
                                "model_name": model_name,
                                "model_timestamp": init_date,
                                "start_valid_time": start_valid_date,
                                "ws_max_10m": ws_max_10m,
                                "disturbance_candidates": json.loads(json_data)
                            }
                            all_retrieved_data.append(retrieved_data)

        for ensemble_name in ensemble_names:
            ensemble_init_times = set()
            ensemble_completed_times_by_init_date = {}
            for model_name, model_init_time in model_init_times.items():
                if model_name not in model_name_to_ensemble_name:
                    print("Warning could not find ensemble name for model name (skipping):", model_name)
                    continue
                if ensemble_name == model_name_to_ensemble_name[model_name]:
                    ensemble_init_times.add(model_init_time)
                    if model_init_time not in ensemble_completed_times_by_init_date:
                        ensemble_completed_times_by_init_date[model_init_time] = []
                    ensemble_completed_times_by_init_date[model_init_time].append(model_completed_times[model_name])
            if not ensemble_init_times:
                continue

            num_init_time_ens_is_complete = 0
            for ensemble_init_time in ensemble_init_times:
                cursor.execute(
                    f'SELECT completed FROM ens_status WHERE init_date = ? AND ensemble_name = ? ORDER BY init_date DESC LIMIT 1',
                    (ensemble_init_time, ensemble_name))
                results = cursor.fetchall()
                ens_is_completed = 0
                if results:
                    for row in results:
                        ens_is_completed = row[0]

                ensemble_completed_time = max(ensemble_completed_times_by_init_date[ensemble_init_time])
                if ensemble_name not in ensemble_completed_times:
                    ensemble_completed_times[ensemble_name] = []
                ensemble_completed_times[ensemble_name].append(ensemble_completed_time)
                num_init_time_ens_is_complete += int(ens_is_completed)

            if num_init_time_ens_is_complete == 1 and len(ensemble_init_times) == 1:
                completed_ensembles[ensemble_name] = ensemble_completed_times[ensemble_name][0]

    except sqlite3.Error as e:
        print(f"SQLite error (get_tc_candidates_from_valid_time): {e}")
    finally:
        if conn:
            conn.close()

    # completed times are datetimes while init times are strings
    return model_init_times, model_completed_times, ensemble_completed_times, completed_ensembles, all_retrieved_data

# get model cycles relative to init_date timestamp
def get_tc_model_init_times_relative_to(init_date, genesis_previous_selected):
    if type(init_date) == str:
        init_date = datetime.fromisoformat(init_date)

    if genesis_previous_selected == 'GLOBAL-DET':
        db_file_path = tc_candidates_db_file_path
    else:
        db_file_path = tc_candidates_tcgen_db_file_path

    conn = None
    # 'at' can be strictly before init_date, but previous must be also before 'at'
    model_init_times = {'previous': None, 'next': None, 'at': None}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT init_date FROM completed WHERE init_date <= ? ORDER BY init_date DESC LIMIT 1',
                       (datetime.isoformat(init_date),))
        results = cursor.fetchall()
        if results:
            for row in results:
                init_time = row[0]
                model_init_times['at'] = datetime.fromisoformat(init_time)

        if model_init_times['at']:
            cursor.execute(
                'SELECT DISTINCT init_date FROM completed WHERE init_date > ? ORDER BY init_date ASC LIMIT 1',
                (datetime.isoformat(model_init_times['at']),))
            results = cursor.fetchall()
            if results:
                for row in results:
                    init_time = row[0]
                    model_init_times['next'] = datetime.fromisoformat(init_time)

        if model_init_times['at']:
            cursor.execute(
                'SELECT DISTINCT init_date FROM completed WHERE init_date < ? ORDER BY init_date DESC LIMIT 1',
                (datetime.isoformat(model_init_times['at']),))
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

# returns a list of modified/created or keys in new_dict
def diff_dicts(old_dict, new_dict):
    new_keys = new_dict.keys() - old_dict.keys()
    modified_keys = {key for key in old_dict.keys() & new_dict.keys() if new_dict[key] != old_dict[key]}
    all_changed_keys = new_keys | modified_keys
    return all_changed_keys

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

def parse_adt(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data from {url}")

    # Split the response into lines
    lines = response.text.splitlines()

    i = 0
    while i < len(lines):
        if lines[i][:10].strip() == 'Date':
            i += 1
            break
        i += 1

    if i == len(lines):
        return []

    # Skip the header lines
    data_lines = lines[i:]


    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Loop through each line
    for line in data_lines:
        # check if valid line
        if re.match(r'^([0-9]{4})', line[0:4]) is None:
            continue

        # Extract the data using fixed width positions
        date = line[:9].strip()
        time = line[10:16].strip()
        mslp_value = float(line[22:28].strip())
        vmax10m = float(line[29:34].strip())
        lat = float(line[114:120].strip())
        # ADT wackily uses degrees W
        lon = -float(line[121:128].strip())
        fix = line[130:136].strip()
        if fix == 'FCST':
            fixmethod = 'Forecast'
        elif fix == 'ARCHER':
            fixmethod = 'ARCHER'

        # Combine date and time into a datetime object
        valid_time = datetime.strptime(f"{date} {time}", "%Y%b%d %H%M%S")

        # Create a dictionary for this entry
        entry = {
            "valid_time": valid_time,
            "mslp_value": mslp_value,
            "vmax10m": vmax10m,
            "lat": lat,
            "lon": lon,
            "fixmethod": fixmethod
        }

        # Add the entry to the list
        parsed_data.append(entry)

    return parsed_data

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
    elif len(line) >= 95:
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


# Classes used by App class

class AnalysisDialog(tk.Toplevel):
    def __init__(self, parent, plotted_tc_candidates, root_width, root_height, previous_selected_combo):
        # Load the existing JSON data for pressure presets
        with open(output_cat_file_name, 'r') as f:
            pres_stats_by_basin_by_intensity_cat = json.load(f)

        self.selected_internal_storm_ids = App.get_selected_internal_storm_ids()
        if not self.selected_internal_storm_ids:
            return

        self.plotted_tc_candidates = plotted_tc_candidates
        self.buttonbox = None
        self.notebook = None
        self.root_width = root_width
        self.root_height = root_height
        self.previous_selected_combo = previous_selected_combo

        if self.previous_selected_combo == 'GLOBAL-DET':
            self.possible_ensemble = False
            self.total_ensembles = 0
            model_names = model_data_folders_by_model_name.keys()
            model_names = [model_name for model_name in model_names if model_name[-5:] != 'TCGEN']
            self.total_models = len(model_names)
            self.ensemble_type = 'GLOBAL-DET'
            self.lookup_model_name_ensemble_name = global_det_lookup
            # TODO: test; this won't work but is for redundancy, needs fixing
            self.lookup_num_models_by_ensemble_name = num_active_global_det_members
        elif self.previous_selected_combo[-5:] == 'TCGEN':
            self.possible_ensemble = True
            self.total_models = len(tcgen_models_by_ensemble[self.previous_selected_combo])
            if self.previous_selected_combo == 'ALL-TCGEN':
                # Should be 4 (EPS, FNMOC, GEFS, GEPS; don't count ALL)
                self.total_ensembles = len(tcgen_models_by_ensemble.keys()) - 1
                if len(tcgen_active_ensembles_all) != self.total_ensembles:
                    self.total_ensembles = len(tcgen_active_ensembles_all)
            else:
                self.total_ensembles = 1

            self.ensemble_type = 'TCGEN'
            self.lookup_model_name_ensemble_name = model_name_to_ensemble_name
            self.lookup_num_models_by_ensemble_name = tcgen_num_models_by_ensemble
        elif self.previous_selected_combo[-4:] in 'ATCF':
            # unofficial adeck (GEFS, GEPS, FNMOC)
            # not handling "GEFS-MEMBERS" (no official a-track gefs members as no guarantee which members are present?)
            self.possible_ensemble = True
            self.total_models = len(atcf_ens_models_by_ensemble[self.previous_selected_combo])
            if self.previous_selected_combo == 'ALL-ATCF':
                # Should be 3 (FNMOC, GEFS, GEPS; don't count ALL)
                self.total_ensembles = len(atcf_ens_num_models_by_ensemble.keys()) - 1
                if len(atcf_active_ensembles_all) != self.total_ensembles:
                    self.total_ensembles = len(atcf_active_ensembles_all)
            else:
                self.total_ensembles = 1

            self.ensemble_type = 'ATCF'
            self.lookup_model_name_ensemble_name = atcf_ens_model_name_to_ensemble_name
            self.lookup_num_models_by_ensemble_name = atcf_ens_num_models_by_ensemble
        else:
            # TODO: for now don't count ensembles for adeck data as we don't have a complete list of member names
            self.total_ensembles = 0
            self.possible_ensemble = False
            self.ensemble_type = 'UNKNOWN'
            # won't work for now until fixed in code
            self.lookup_model_name_ensemble_name = model_name_to_ensemble_name

        self.basin_presets = ['EP', 'NA']
        self.basin_selected_preset = tk.StringVar()
        self.basin_previous_selected = None

        # TODO: Point probabilities, not by 'Earliest'
        self.cdf_analyses = ['Earliest VMax @ 10m >= min',
                             'min <= Earliest VMax @ 10m <= max',
                             'Earliest MSLP <= max',
                             'min <= Earliest MSLP <= max'
                             ]
        self.pdf_analyses = ['Earliest VMax @ 10m >= min',
                             'min <= Earliest VMax @ 10m <= max',
                             'Earliest MSLP <= max',
                             'min <= Earliest MSLP <= max'
                             ]

        # Construct presets
        self.intensity_preset_names = ['TC', 'SD', 'TD', 'SS', 'TS', 'CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5']

        pres_presets = {}
        for basin, basin_dict in pres_stats_by_basin_by_intensity_cat.items():
            min_value = 9999
            max_value = 0
            basin_presets = {}
            for preset_name, mslp_pct_tuples in basin_dict.items():
                mslp_min = int(round(min(mslp_pct_tuples)[0]))
                mslp_max = int(round(max(mslp_pct_tuples)[0]))
                # the min_value corresponds to the strongest MSLP while not being likely to be classified in the next higher category
                min_value = min(mslp_min, min_value)
                # this is the >= 50% value (>= 50% storms initially classified at this category had a MSLP of this value)
                max_value = max(mslp_max, max_value)
                basin_presets[preset_name] = (mslp_min, mslp_max)

            # don't use the actual max_value as there is no upper bound for TC category
            basin_presets['TC'] = (min_value, 9999)
            pres_presets[basin] = basin_presets

        vmax_presets = {
            'TC': (0, 9999),
            'SD': (0, 33),
            'TD': (0, 33),
            'SS': (34, 63),
            'TS': (34, 63)
        }
        sss = [64, 83, 96, 113, 137]
        # construct preset vals for vmax for CAT1, CAT2, CAT3, CAT4, CAT5
        for i, preset_name in enumerate(self.intensity_preset_names[5:]):
            if i+1 == len(sss):
                range_max = 9999
            else:
                range_max = sss[i+1] - 1
            range_min = sss[i]
            vmax_presets[preset_name] = (range_min, range_max)

        self.min_max_presets_by_preset_and_basin_and_analysis = {}
        for intensity_range_name in self.intensity_preset_names:
            basin_dicts = {}
            for basin_name in self.basin_presets:
                basin_dict = {}
                for analysis_name in self.cdf_analyses:
                    if analysis_name == 'Earliest VMax @ 10m >= min':
                        basin_dict[analysis_name] = (vmax_presets[intensity_range_name][0], 9999)
                    elif analysis_name == 'min <= Earliest VMax @ 10m <= max':
                        basin_dict[analysis_name] = vmax_presets[intensity_range_name]
                    elif analysis_name == 'Earliest MSLP <= max':
                        basin_dict[analysis_name] = (0, pres_presets[basin_name][intensity_range_name][1])
                    elif analysis_name == 'min <= Earliest MSLP <= max':
                        basin_dict[analysis_name] = pres_presets[basin_name][intensity_range_name]

                basin_dicts[basin_name] = basin_dict

            self.min_max_presets_by_preset_and_basin_and_analysis[intensity_range_name] = basin_dicts

        self.warm_core_selected = tk.StringVar()
        self.warm_core_previous_selected = None
        self.tc_phase_selected = tk.StringVar()
        self.tc_phase_previous_selected = None

        self.cdf_selected_analysis = tk.StringVar()
        self.cdf_previous_selected = None
        self.cdf_preset_selected = tk.StringVar()
        self.cdf_preset_previous_selected = None

        self.pdf_selected_analysis = tk.StringVar()
        self.pdf_previous_selected = None
        self.pdf_preset_selected = tk.StringVar()
        self.pdf_preset_previous_selected = None

        self.cdf_min = tk.IntVar(value=0)
        self.cdf_max = tk.IntVar(value=9999)

        self.pdf_min = tk.IntVar(value=0)
        self.pdf_max = tk.IntVar(value=9999)

        self.cdf_min.trace("w", self.on_cdf_range_changed)
        self.cdf_max.trace("w", self.on_cdf_range_changed)

        self.pdf_min.trace("w", self.on_pdf_range_changed)
        self.pdf_max.trace("w", self.on_pdf_range_changed)

        self.do_update_cdf_chart = False
        self.do_update_pdf_chart = False

        self.tz_selected_timezone = tk.StringVar(value="UTC")
        self.tz_previous_selected = None

        self.notebook_tab_names = dict()
        self.notebook_figs = dict()

        self.ax_pres = None
        self.ax_rvor = None
        self.ax_size = None
        self.ax_vmax = None
        self.fig_cdf = None
        self.fig_pdf = None
        self.fig_pres = None
        self.fig_rvor = None
        self.fig_size = None
        self.fig_track_spread = None
        self.fig_vmax = None
        self.canvas_cdf = None
        self.canvas_pdf = None
        self.canvas_pres = None
        self.canvas_rvor = None
        self.canvas_size = None
        self.canvas_track_spread = None
        self.canvas_vmax = None

        super().__init__(parent)  # Set the title here
        self.title('Analysis on Selected Tracks')
        self.wm_protocol("WM_TAKE_FOCUS", self.wm_title)
        # this is required to center the dialog on the window
        self.transient(parent)
        self.parent = parent

        self.body_frame = ttk.Frame(self, style='CanvasFrame.TFrame')
        self.body_frame.pack(padx=5, pady=5, fill="both", expand=True)

        self.button_frame = ttk.Frame(self, style='CanvasFrame.TFrame')
        self.button_frame.pack(padx=5, pady=5)

        self.body(self.body_frame, )
        self.create_buttonbox(self.button_frame)

        self.attributes('-fullscreen', True)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.grab_set()
        self.geometry("500x500+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.wait_visibility()
        self.center_window()
        self.wait_window(self)

    @staticmethod
    def aggregate_tracks_to_model_series(track_series_data, first=True):
        model_series_data = []
        tracks_series_by_model_name = {}
        for model_name, x_vals, y_vals in track_series_data:
            if model_name not in tracks_series_by_model_name:
                tracks_series_by_model_name[model_name] = []

            tracks_series_by_model_name[model_name].append((x_vals, y_vals))

        model_tracks_agg_by_name_and_x = {}
        for model_name, tracks_xy_tuples in tracks_series_by_model_name.items():
            min_x = None

            model_track_agg_by_x = {}
            y_val_max = 0
            x_val_min = None
            for (x_vals, y_vals) in tracks_xy_tuples:
                for i, y_val in enumerate(y_vals):
                    if y_val > 0:
                        x_val = x_vals[i]
                        if x_val_min == None:
                            x_val_min = x_val

                        model_track_agg_by_x[x_val] = max(y_val, y_val_max)

            model_tracks_agg_by_name_and_x[model_name] = model_track_agg_by_x

        if first:
            # As this is an 'EARLIEST' analysis, prune data after the earliest
            for model_name, xy_dict in model_tracks_agg_by_name_and_x.items():
                first_x_val = sorted(xy_dict.keys())[0]
                first_y_val = xy_dict[first_x_val]
                model_series_data.append((model_name, [first_x_val], [first_y_val]))
        else:
            for model_name, xy_dict in model_tracks_agg_by_name_and_x.items():
                x_vals = sorted(xy_dict.keys())
                y_vals = []
                for x_val in x_vals:
                    y_vals.append(xy_dict[x_val])

                model_series_data.append((model_name, x_vals, y_vals))

        return model_series_data

    def body(self, master):
        global ANALYSIS_TZ
        frame = ttk.Frame(master, style='CanvasFrame.TFrame')
        frame.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(frame, style='TNotebook')
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.notebook.pack(fill="both", expand=True)

        tab_index = 0
        # Create a frame for each tab
        self.intensity_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.intensity_frame, text="Vmax")
        self.notebook_tab_names[tab_index] = 'vmax'
        tab_index += 1

        self.pressure_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.pressure_frame, text="Pres")
        self.notebook_tab_names[tab_index] = 'pres'
        tab_index += 1

        self.size_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.size_frame, text="Size")
        self.notebook_tab_names[tab_index] = 'size'
        tab_index += 1

        self.rvor_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.rvor_frame, text="RVor")
        self.notebook_tab_names[tab_index] = 'rvor'
        tab_index += 1

        self.track_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.track_frame, text="Track")
        self.notebook_tab_names[tab_index] = 'track'
        tab_index += 1

        self.pdf_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.pdf_frame, text="PDF")
        self.notebook_tab_names[tab_index] = 'pdf'
        tab_index += 1

        self.cdf_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.cdf_frame, text="CDF")
        self.notebook_tab_names[tab_index] = 'cdf'
        tab_index += 1

        self.model_info_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.model_info_frame, text="Models")
        self.notebook_tab_names[tab_index] = 'models'
        tab_index += 1

        self.opt_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        self.notebook.add(self.opt_frame, text="Options")
        self.notebook_tab_names[tab_index] = 'opt'
        tab_index += 1

        # Get the list of all timezones with offsets
        self.timezones = self.get_timezones_with_offsets()

        r = 0
        self.label_tz = ttk.Label(self.opt_frame, text="Timezone:", style='TLabel')
        self.label_tz.grid(row=r, column=0, sticky='w')
        # Create the timezone dropdown menu
        # Create a combobox widget with the new style

        self.tz_combobox = ttk.Combobox(self.opt_frame, textvariable=self.tz_selected_timezone, values=self.timezones,
                                        width=50, state='readonly', style="Black.TCombobox")
        tz_names = self.get_timezones()
        if ANALYSIS_TZ in tz_names:
            tzindex = tz_names.index(ANALYSIS_TZ)
            self.tz_combobox.current(tzindex)
        else:
            # bad tz?
            if 'UTC' in tz_names:
                tzindex = tz_names.index('UTC')
                self.tz_combobox.current(tzindex)
        self.tz_previous_selected = self.tz_selected_timezone.get()

        self.tz_combobox.bind("<<ComboboxSelected>>", self.combo_selected_tz_event)
        self.tz_combobox.grid(row=r, column=1)

        indent = "     "
        # Flags on which include points to include (based on warm core flag, CPS paramaters)
        # Always include 'Unknown' as EPS does not have these params)
        r += 1
        c = 0
        ttk.Label(self.opt_frame, text="Include Points (All charts):", style='TLabel').grid(row=r, column=c, sticky='w')

        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}300-500 hPa, 1 K warm core contour:", style='TLabel').grid(row=r, column=c, sticky='w')
        c += 1
        self.warm_core_combobox = ttk.Combobox(self.opt_frame, textvariable=self.warm_core_selected,
                                              values=['Any', 'Warm Core', 'Cold Core'],
                                              width=15, state='readonly', style="Black.TCombobox")
        # DEFAULT: Any
        self.warm_core_combobox.current(0)
        self.warm_core_previous_selected = self.warm_core_selected.get()
        self.warm_core_combobox.bind("<<ComboboxSelected>>", self.combo_selected_warm_core_event)
        self.warm_core_combobox.grid(row=r, column=c, sticky='w')
        c += 1

        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}Phase (CPS):", style='TLabel').grid(row=r, column=c, sticky='w')
        c += 1
        self.tc_phase_combobox = ttk.Combobox(self.opt_frame, textvariable=self.tc_phase_selected,
                                           values=['Any', 'Tropical/Subtropical', 'Tropical', 'Subtropical'],
                                           width=25, state='readonly', style="Black.TCombobox")
        # DEFAULT: Any
        self.tc_phase_combobox.current(0)
        self.tc_phase_previous_selected = self.tc_phase_selected.get()
        self.tc_phase_combobox.bind("<<ComboboxSelected>>", self.combo_selected_tc_phase_event)
        self.tc_phase_combobox.grid(row=r, column=c, sticky='w')

        r += 1
        c = 0
        ttk.Label(self.opt_frame, text="Include Points (PDF/CDF charts):", style='TLabel').grid(row=r, column=c, sticky='w')

        # Preset that controls PDF/CDF Min/Min Combobox Preset values
        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}Basin Preset:", style='TLabel').grid(row=r, column=c, sticky='w')
        self.basin_combobox = ttk.Combobox(self.opt_frame, textvariable=self.basin_selected_preset,
                                         values=self.basin_presets,
                                         width=5, state='readonly', style="Black.TCombobox")
        # DEFAULT: NA BASIN
        self.basin_combobox.current(1)
        self.basin_previous_selected = self.basin_selected_preset.get()
        self.basin_combobox.bind("<<ComboboxSelected>>", self.combo_selected_basin_event)
        self.basin_combobox.grid(row=r, column=1, sticky='w')

        # PDF Options
        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}PDF Analysis:", style='TLabel').grid(row=r, column=c, sticky='w')
        self.pdf_combobox = ttk.Combobox(self.opt_frame, textvariable=self.pdf_selected_analysis, values=self.pdf_analyses,
                                        width=50, state='readonly', style="Black.TCombobox")
        c += 1
        self.pdf_combobox.current(0)
        self.pdf_combobox.grid(row=r, column=c, sticky='w')
        self.pdf_combobox.bind("<<ComboboxSelected>>", self.combo_selected_pdf_event)
        self.pdf_previous_selected = self.pdf_selected_analysis.get()

        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}PDF Min/Max Preset:", style='TLabel').grid(row=r, column=c, sticky='w')
        self.pdf_preset_combobox = ttk.Combobox(self.opt_frame, textvariable=self.pdf_preset_selected,
                                         values=self.intensity_preset_names,
                                         width=10, state='readonly', style="Black.TCombobox")
        c += 1
        self.pdf_preset_combobox.current(0)
        self.pdf_preset_combobox.grid(row=r, column=c, sticky='w')
        self.pdf_preset_combobox.bind("<<ComboboxSelected>>", self.combo_selected_pdf_preset_event)
        self.pdf_preset_previous_selected = self.pdf_preset_selected.get()
        c += 1
        ttk.Label(self.opt_frame, text="Min:", style='TLabel').grid(row=r, column=c)
        c += 1
        ttk.Entry(self.opt_frame, textvariable=self.pdf_min, style='TEntry', width=5).grid(row=r, column=c)
        c += 1
        ttk.Label(self.opt_frame, text="Max:", style='TLabel').grid(row=r, column=c)
        c += 1
        ttk.Entry(self.opt_frame, textvariable=self.pdf_max, style='TEntry', width=5).grid(row=r, column=c)

        # CDF Options
        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}CDF Analysis:", style='TLabel').grid(row=r, column=c, sticky='w')
        self.cdf_combobox = ttk.Combobox(self.opt_frame, textvariable=self.cdf_selected_analysis,
                                         values=self.cdf_analyses,
                                         width=50, state='readonly', style="Black.TCombobox")
        c += 1
        self.cdf_combobox.current(0)
        self.cdf_combobox.grid(row=r, column=c, sticky='w')
        self.cdf_combobox.bind("<<ComboboxSelected>>", self.combo_selected_cdf_event)
        self.cdf_previous_selected = self.cdf_selected_analysis.get()

        r += 1
        c = 0
        ttk.Label(self.opt_frame, text=f"{indent}CDF Min/Max Preset:", style='TLabel').grid(row=r, column=c, sticky='w')
        self.cdf_preset_combobox = ttk.Combobox(self.opt_frame, textvariable=self.cdf_preset_selected,
                                                values=self.intensity_preset_names,
                                                width=10, state='readonly', style="Black.TCombobox")
        c += 1
        self.cdf_preset_combobox.current(0)
        self.cdf_preset_combobox.grid(row=r, column=c, sticky='w')
        self.cdf_preset_combobox.bind("<<ComboboxSelected>>", self.combo_selected_cdf_preset_event)
        self.cdf_preset_previous_selected = self.cdf_preset_selected.get()
        c += 1
        ttk.Label(self.opt_frame, text="Min:", style='TLabel').grid(row=r, column=c, sticky='w')
        c += 1
        ttk.Entry(self.opt_frame, textvariable=self.cdf_min, style='TEntry', width=5).grid(row=r, column=c)
        c += 1
        ttk.Label(self.opt_frame, text="Max:", style='TLabel').grid(row=r, column=c, sticky='w')
        c += 1
        ttk.Entry(self.opt_frame, textvariable=self.cdf_max, style='TEntry', width=5).grid(row=r, column=c)

        r += 1
        save_tz_session_btn = ttk.Button(self.opt_frame, text="Save TZ (this session)", command=self.save_tz_session, style='TButton')
        save_tz_session_btn.grid(row=r, column=0, sticky='w')

        save_tz_setting = ttk.Button(self.opt_frame, text="Save TZ (as setting)", command=self.save_tz_setting, style='TButton')
        save_tz_setting.grid(row=r, column=1, sticky='w')

        for i in range(2):
            self.opt_frame.grid_columnconfigure(i, pad=10)
        for i in range(r):
            self.opt_frame.grid_rowconfigure(i, pad=10)

        self.update_all_charts()

    @staticmethod
    def calculate_mean_track(plotted_tc_candidates, all_datetimes, filtered_storm_ids):
        # Filter plotted_tc_candidates based on selected_internal_storm_ids
        filtered_candidates = [(iid, tc) for iid, tc in plotted_tc_candidates if
                               iid in filtered_storm_ids]

        # Initialize a dictionary to store values for each valid time
        track_values = {dt: {'lat': [], 'lon': [], 'vmax10m': []} for dt in all_datetimes}

        # Populate the dictionary with values from filtered candidates
        for internal_id, tc_candidate in filtered_candidates:
            for tc in tc_candidate:
                dt = tc['valid_time']
                if dt in track_values:
                    track_values[dt]['lat'].append(tc['lat'])
                    track_values[dt]['lon'].append(tc['lon'])
                    track_values[dt]['vmax10m'].append(tc['vmax10m'])

        # Calculate the mean for each valid time
        mean_track = []
        for dt, values in track_values.items():
            if values['lat'] and values['lon']:
                mean_lat = np.mean(values['lat'])
                mean_lon = np.mean(values['lon'])
                mean_vmax = np.mean(values['vmax10m'])
                mean_track.append({'valid_time': dt, 'lat': mean_lat, 'lon': mean_lon, 'vmax10m': mean_vmax})

        return mean_track

    @staticmethod
    def calculate_mean_track_interp(plotted_tc_candidates, all_datetimes, filtered_storm_ids):
        # Filter plotted_tc_candidates based on selected_internal_storm_ids
        filtered_candidates = [(iid, tc) for iid, tc in plotted_tc_candidates if
                               iid in filtered_storm_ids]

        # Initialize a dictionary to store interpolated values for each valid time
        track_values = {dt: {'lat': [], 'lon': [], 'vmax10m': []} for dt in all_datetimes}

        # Interpolate each individual track hourly and populate the dictionary
        for internal_id, tc_candidate in filtered_candidates:
            # Extract valid times and corresponding values for interpolation
            valid_times = [tc['valid_time'] for tc in tc_candidate]
            lats = [tc['lat'] for tc in tc_candidate]
            lons = [tc['lon'] for tc in tc_candidate]
            vmaxs = [tc['vmax10m'] for tc in tc_candidate]

            # Convert valid times to datetime objects and then to timestamps
            valid_times = np.array([dt.timestamp() for dt in valid_times])

            # Interpolate values hourly
            f_lat = interp1d(valid_times, lats, kind='linear', fill_value='extrapolate')
            f_lon = interp1d(valid_times, lons, kind='linear', fill_value='extrapolate')
            f_vmax = interp1d(valid_times, vmaxs, kind='linear', fill_value='extrapolate')

            # Evaluate interpolating functions at hourly intervals
            hourly_times = np.array([dt.timestamp() for dt in all_datetimes])
            interp_lats = f_lat(hourly_times)
            interp_lons = f_lon(hourly_times)
            interp_vmaxs = f_vmax(hourly_times)

            # Set values outside the candidate's interpolate range to NaN
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            mask = (hourly_times < min_time) | (hourly_times > max_time)
            interp_lats[mask] = np.nan
            interp_lons[mask] = np.nan
            interp_vmaxs[mask] = np.nan

            # Populate the dictionary with interpolated values
            for i, dt in enumerate(all_datetimes):
                track_values[dt]['lat'].append(interp_lats[i])
                track_values[dt]['lon'].append(interp_lons[i])
                track_values[dt]['vmax10m'].append(interp_vmaxs[i])

        # Calculate the mean for each valid time
        mean_track = []
        for dt, values in track_values.items():
            if values['lat'] and values['lon']:
                mean_lat = np.nanmean(values['lat'])
                mean_lon = np.nanmean(values['lon'])
                mean_vmax = np.nanmean(values['vmax10m'])
                mean_track.append({'valid_time': dt, 'lat': mean_lat, 'lon': mean_lon, 'vmax10m': mean_vmax})

        return mean_track

    def calculate_spread(self, mean_track, filtered_storm_ids):
        # Filter plotted_tc_candidates based on selected_internal_storm_ids
        filtered_candidates = [(iid, tc) for iid, tc in self.plotted_tc_candidates if
                               iid in filtered_storm_ids]

        cross_track_spread = []
        along_track_spread = []

        for mean_point in mean_track:
            dt = mean_point['valid_time']
            mean_lat = mean_point['lat']
            mean_lon = mean_point['lon']

            cross_diffs = []
            along_diffs = []

            for internal_id, tc_candidate in filtered_candidates:
                for tc in tc_candidate:
                    if tc['valid_time'] == dt:
                        model_lat = tc['lat']
                        model_lon = tc['lon']

                        # Calculate cross-track difference
                        if len(mean_track) > 1:
                            before_idx = max(0, mean_track.index(mean_point) - 1)
                            after_idx = min(len(mean_track) - 1, mean_track.index(mean_point) + 1)
                            before_lat = mean_track[before_idx]['lat']
                            before_lon = mean_track[before_idx]['lon']
                            after_lat = mean_track[after_idx]['lat']
                            after_lon = mean_track[after_idx]['lon']

                            # Distance from model point to mean track line segments
                            before_dist = geodesic((model_lat, model_lon), (before_lat, before_lon)).nm
                            after_dist = geodesic((model_lat, model_lon), (after_lat, after_lon)).nm
                            cross_diffs.append(min(before_dist, after_dist))

                        # Calculate along-track difference
                        along_dist = geodesic((mean_lat, mean_lon), (model_lat, model_lon)).nm
                        along_diffs.append(along_dist)

            # Compute spread measures (e.g., standard deviation)
            if cross_diffs:
                cross_track_spread.append(np.std(cross_diffs))
            if along_diffs:
                along_track_spread.append(np.std(along_diffs))

        return cross_track_spread, along_track_spread

    def cancel(self, event=None):
        # close figures to free up memory
        if self.fig_cdf:
            plt.close(self.fig_cdf)
            self.fig_cdf = None
            self.notebook_figs['cdf'] = None
        if self.fig_pdf:
            plt.close(self.fig_pdf)
            self.fig_pdf = None
            self.notebook_figs['pdf'] = None
        if self.fig_pres:
            plt.close(self.fig_pres)
            self.fig_pres = None
            self.notebook_figs['pres'] = None
        if self.fig_rvor:
            plt.close(self.fig_rvor)
            self.fig_rvor = None
            self.notebook_figs['rvor'] = None
        if self.fig_size:
            plt.close(self.fig_size)
            self.fig_size = None
            self.notebook_figs['size'] = None
        if self.fig_track_spread:
            plt.close(self.fig_track_spread)
            self.fig_track_spread = None
            self.notebook_figs['track'] = None
        if self.fig_vmax:
            plt.close(self.fig_vmax)
            self.fig_vmax = None
            self.notebook_figs['vmax'] = None

        self.parent.focus_set()
        self.destroy()

    def center_window(self):
        self.update_idletasks()

        # Get the dimensions of the dialog
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height() + 20

        # Get the dimensions of the parent window
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate the position to center the dialog on the parent window
        x = self.parent.winfo_x() + (parent_width // 2) - (dialog_width // 2)
        y = self.parent.winfo_y() + (parent_height // 2) - (dialog_height // 2)

        # Set the position of the dialog
        self.geometry(f'{dialog_width}x{dialog_height}+{x}+{y}')
        self.update_idletasks()  # Force update after centering
        self.parent.update_idletasks()  # Force update after centering

    def combo_selected_basin_event(self, event):
        current_value = self.basin_combobox.get()
        if current_value != self.basin_previous_selected:
            self.basin_previous_selected = self.basin_combobox.get()
            self.update_opt_preset_min_max(pdf=True)

    def combo_selected_cdf_event(self, event):
        current_value = self.cdf_combobox.get()
        if current_value != self.cdf_previous_selected:
            self.cdf_previous_selected = current_value
            # Update the chart on tab switch only
            self.do_update_cdf_chart = True
            self.update_opt_preset_min_max(cdf=True)

    def combo_selected_cdf_preset_event(self, event):
        current_value = self.cdf_preset_combobox.get()
        if current_value != self.cdf_preset_previous_selected:
            self.cdf_preset_previous_selected = current_value
            self.update_opt_preset_min_max(cdf=True)

    def combo_selected_pdf_event(self, event):
        current_value = self.pdf_combobox.get()
        if current_value != self.pdf_previous_selected:
            self.pdf_previous_selected = current_value
            # Update the chart on tab switch only
            self.do_update_pdf_chart = True
            self.update_opt_preset_min_max(pdf=True)

    def combo_selected_pdf_preset_event(self, event):
        current_value = self.pdf_preset_combobox.get()
        if current_value != self.pdf_preset_previous_selected:
            self.pdf_preset_previous_selected = current_value
            self.update_opt_preset_min_max(pdf=True)

    def combo_selected_tc_phase_event(self, event):
        current_value = self.tc_phase_combobox.get()
        if current_value != self.tc_phase_previous_selected:
            self.tc_phase_previous_selected = current_value
            self.update_all_charts()

    def combo_selected_tz_event(self, event):
        current_value = self.tz_combobox.get()
        if current_value != self.tz_previous_selected:
            self.tz_previous_selected = current_value
            self.update_all_charts()

    def combo_selected_warm_core_event(self, event):
        current_value = self.warm_core_combobox.get()
        if current_value != self.warm_core_previous_selected:
            self.warm_core_previous_selected = current_value
            self.update_all_charts()

    # convert it to native (strip tz) for easier charting
    @staticmethod
    def convert_to_selected_timezone(dt, timezone_str):
        """Convert a datetime object from UTC to the selected timezone."""
        # Get the timezone string without the UTC offset
        timezone_name = timezone_str.split(') ')[-1]
        # Assign UTC timezone to the input datetime
        dt_utc = dt.replace(tzinfo=pytz.utc)
        # Convert to the selected timezone
        target_timezone = pytz.timezone(timezone_name)
        dt_converted = dt_utc.astimezone(target_timezone)
        dt_native = dt_converted.replace(tzinfo=None)
        return dt_native

    def create_buttonbox(self, master):
        self.buttonbox = ttk.Frame(master, style="CanvasFrame.TFrame")
        ok_w = ttk.Button(self.buttonbox, text="OK", command=self.ok, style='TButton')
        ok_w.pack(side=tk.LEFT, padx=5, pady=5)

        save_w = ttk.Button(self.buttonbox, text="Save", command=self.save, style='TButton')
        save_w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("s", self.save)
        self.bind("p", self.save)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        # Set focus on the OK button
        ok_w.focus_set()

        self.buttonbox.pack()
        return self.buttonbox

    @staticmethod
    def filter_series(first=False, min_val=0, max_val=9999, x=None, y=None, pdf=False, cdf=False):
        new_x = new_y = None
        if not (pdf or cdf):
            return new_x, new_y

        x_counts = {}
        if pdf:
            for x_val, y_val in list(zip(x, y)):
                if y_val <= max_val and y_val >= min_val:
                    # aggregate by 24 hours
                    x_day = x_val.replace(hour=0, minute=0, second=0, microsecond=0)
                    if first:
                        x_counts[x_day] = 1
                    else:
                        if x_day not in x_counts:
                            x_counts[x_day] = 0

                        x_counts[x_day] += 1

            if not x_counts:
                return new_x, new_y

            earliest_day = min(x_counts.keys())
            new_x = [earliest_day]
            new_y = [x_counts[earliest_day]]

        elif cdf:
            for x_val, y_val in list(zip(x, y)):
                if y_val <= max_val and y_val >= min_val:
                    # don't aggregate by 24 hours
                    if first:
                        x_counts[x_val] = 1
                    else:
                        if x_day not in x_counts:
                            x_counts[x_val] = 0

                        x_counts[x_val] += 1

            if not x_counts:
                return new_x, new_y

            earliest_time = min(x_counts.keys())
            new_x = [earliest_time]
            new_y = [x_counts[earliest_time]]

        return new_x, new_y

    @staticmethod
    def generate_distinct_colors(n):
        # Generates evenly spaced colors in Lab color space
        colors_lab = np.zeros((n, 3))
        for i in range(n):
            angle = 2 * np.pi * i / n
            lightness = 65
            chroma = 60
            a = chroma * np.cos(angle)
            b = chroma * np.sin(angle)
            colors_lab[i] = [lightness, a, b]

        # Lab to RGB
        colors_rgb = cs.cspace_convert(colors_lab, "CIELab", "sRGB1")
        colors_rgb = np.clip(colors_rgb, 0, 1)

        return colors_rgb

    def generate_freq_analysis_series(self, first=True, model_series_freq=None,
                                      num_ensembles=0, pdf=False, cdf=False):
        # generate the pdf/cdf series for each ensemble
        # for the ALL-TCGEN case, equally weight the ensembles
        series_data = []
        pdf_series_data = []
        cdf_series_data = []
        all_x = []
        all_y = []
        if not model_series_freq:
            return

        if num_ensembles == 0:
            total_ens_name = 'GLOBAL-DET'
            counts = {}
            counts_total_ens = {}
            for model_name, x, y in model_series_freq:
                first_x = x[0]
                if first:
                    if model_name not in counts:
                        counts[model_name] = {}
                    if first_x not in counts:
                        counts[model_name][first_x] = 0
                    if y[0] > 0:
                        counts[model_name][first_x] += y[0]
                        all_x.append(first_x)
                else:
                    # this is freq of the PDF for each synoptic hour which is 0 or 1
                    for x_val, y_val in list(zip(x,y)):
                        if x_val not in counts:
                            counts[model_name][x_val] = 0

                        if y_val > 0:
                            counts[model_name][x_val] += y_val
                            all_x.append(x_val)

            if not all_x:
                # no data
                return [], [], []

            # aggregate model members
            counts_total_ens = {}
            for model_name, xy_dict in counts.items():
                for x_val in sorted(xy_dict.keys()):
                    y_val = xy_dict[x_val]
                    if x_val not in counts_total_ens:
                        counts_total_ens[x_val] = 0

                    counts_total_ens[x_val] += y_val

            counts[total_ens_name] = counts_total_ens

            # first do pdf
            # now we need the pdf at each point (normalize)
            all_x = sorted(all_x)

            model_y_vals = set()
            for model_name, xy_dict in counts.items():
                if model_name == total_ens_name:
                    norm_by_num_models = num_global_det_members
                else:
                    norm_by_num_models = 1
                x_vals = []
                y_vals = []
                for x_val in sorted(xy_dict.keys()):
                    y_val = xy_dict[x_val]
                    x_vals.append(x_val)
                    # normalize, convert to percent

                    if first:
                        norm_yval = 100.0 * y_val / norm_by_num_models
                    else:
                        # For varying number of synoptic hours: will be wrong at start/end of model init
                        # TODO: FIX IF USE OTHER FUNCTION THAT EARLIEST FOR ANALYSIS
                        # Normalize to number of synoptic hours
                        norm_yval = 100.0 * y_val / (4.0 * norm_by_num_models)

                    y_vals.append(norm_yval)
                    model_y_vals.add(norm_yval)

                pdf_series_data.append((model_name, x_vals, y_vals))
                all_y.extend(model_y_vals)

            # cdf
            if cdf:
                #TODO: FIX FOR ANALYSIS CASE OTHER THAN EARLIEST
                all_y = []
                for (model_name, x_vals, y_vals) in pdf_series_data:
                    max_yval = None
                    series_x = []
                    series_y = []

                    for i, y_val in enumerate(y_vals):
                        if max_yval is None:
                            max_yval = y_val
                        else:
                            max_yval += y_val

                        series_x.append(x_vals[i])
                        series_y.append(max_yval)
                        all_y.append(max_yval)

                    cdf_series_data.append((model_name, series_x, series_y))

        elif num_ensembles == 1:
            # to be replaced
            total_ens_name = 'Ensemble Mean'
            ens_names = set()
            counts = {}
            counts_total_ens = {}
            for model_name, x, y in model_series_freq:
                first_x = x[0]
                if first:
                    if model_name not in counts:
                        counts[model_name] = {}
                    if first_x not in counts:
                        counts[model_name][first_x] = 0
                    if y[0] > 0:
                        counts[model_name][first_x] += y[0]
                        all_x.append(first_x)
                else:
                    # this is freq of the PDF for each synoptic hour which is 0 or 1
                    for x_val, y_val in list(zip(x,y)):
                        if x_val not in counts:
                            counts[model_name][x_val] = 0

                        if y_val > 0:
                            counts[model_name][x_val] += y_val
                            all_x.append(x_val)

            if not all_x:
                # no data
                return [], [], []

            # aggregate model members
            counts_total_ens = {}
            for model_name, xy_dict in counts.items():
                if model_name in self.lookup_model_name_ensemble_name:
                    ens_names.add(self.lookup_model_name_ensemble_name[model_name])
                for x_val in sorted(xy_dict.keys()):
                    y_val = xy_dict[x_val]
                    if x_val not in counts_total_ens:
                        counts_total_ens[x_val] = 0

                    counts_total_ens[x_val] += y_val

            ens_name = None
            ens_names = list(ens_names)
            # sanity check
            if ens_names and len(ens_names) == 1:
                ens_name = ens_names[0]
                total_ens_name = ens_names[0]
                total_num_models_in_ensemble = self.lookup_num_models_by_ensemble_name[ens_name]
            else:
                print(ens_names)
                print("Error (bug): Wrong number of ensembles in analysis calculation (expected 1)")
                return [], [], []

            counts[total_ens_name] = counts_total_ens

            # first do pdf
            # now we need the pdf at each point (normalize)
            all_x = sorted(all_x)

            model_y_vals = set()
            for model_name, xy_dict in counts.items():
                if model_name != total_ens_name:
                    # for single (perturbation) ensemble, skip showing each member
                    continue
                if model_name == total_ens_name:
                    norm_by_num_models = total_num_models_in_ensemble
                else:
                    norm_by_num_models = 1
                x_vals = []
                y_vals = []
                for x_val in sorted(xy_dict.keys()):
                    y_val = xy_dict[x_val]
                    x_vals.append(x_val)
                    # normalize, convert to percent

                    if first:
                        norm_yval = 100.0 * y_val / norm_by_num_models
                    else:
                        # For varying number of synoptic hours: will be wrong at start/end of model init
                        # TODO: FIX IF USE OTHER FUNCTION THAT EARLIEST FOR ANALYSIS
                        # Normalize to number of synoptic hours
                        norm_yval = 100.0 * y_val / (4.0 * norm_by_num_models)

                    y_vals.append(norm_yval)
                    model_y_vals.add(norm_yval)

                pdf_series_data.append((model_name, x_vals, y_vals))
                all_y.extend(model_y_vals)

            # cdf
            if cdf:
                #TODO: FIX FOR ANALYSIS CASE OTHER THAN EARLIEST
                all_y = []
                for (model_name, x_vals, y_vals) in pdf_series_data:
                    max_yval = None
                    series_x = []
                    series_y = []

                    for i, y_val in enumerate(y_vals):
                        if max_yval is None:
                            max_yval = y_val
                        else:
                            max_yval += y_val

                        series_x.append(x_vals[i])
                        series_y.append(max_yval)
                        all_y.append(max_yval)

                    cdf_series_data.append((model_name, series_x, series_y))

        elif num_ensembles > 1:
            total_ens_name = 'ALL-TCGEN'
            ens_names = set()
            counts = {}
            counts_total_ens = {}
            for model_name, x, y in model_series_freq:
                first_x = x[0]
                if first:
                    if model_name not in counts:
                        counts[model_name] = {}
                    if first_x not in counts:
                        counts[model_name][first_x] = 0
                    if y[0] > 0:
                        counts[model_name][first_x] += y[0]
                        all_x.append(first_x)
                else:
                    # this is freq of the PDF for each synoptic hour which is 0 or 1
                    for x_val, y_val in list(zip(x,y)):
                        if x_val not in counts:
                            counts[model_name][x_val] = 0

                        if y_val > 0:
                            counts[model_name][x_val] += y_val
                            all_x.append(x_val)

            if not all_x:
                # no data
                return [], [], []

            ens_counts = {}
            # aggregate counts by ensemble
            for model_name, xy_dict in counts.items():
                if model_name in self.lookup_model_name_ensemble_name:
                    ens_name = self.lookup_model_name_ensemble_name[model_name]
                    ens_names.add(ens_name)
                    if ens_name not in ens_counts:
                        ens_counts[ens_name] = {}
                    for x_val in sorted(xy_dict.keys()):
                        y_val = xy_dict[x_val]
                        if x_val not in ens_counts[ens_name]:
                            ens_counts[ens_name][x_val] = 0

                        ens_counts[ens_name][x_val] += y_val

            if ens_names and len(ens_names) > 1:
                # TODO: HARDCODED. FIX IF CHANGING ANALYSIS FROM ENSEMBLES OTHER THAN TCGEN
                if self.ensemble_type == 'ATCF':
                    total_ens_name = 'ALL-ATCF'
                elif self.ensemble_type == 'TCGEN':
                    total_ens_name = 'ALL-TCGEN'
                else:
                    total_ens_name = 'ALL'

            else:
                # case when some ensembles in the super ensemble don't have data fitting the criteria
                if self.ensemble_type == 'ATCF':
                    total_ens_name = 'ALL-ATCF'
                elif self.ensemble_type == 'TCGEN':
                    total_ens_name = 'ALL-TCGEN'
                else:
                    print(ens_names)
                    print("Error (bug): Wrong number of ensembles in analysis calculation (expected >1)")
                    return [], [], []

            # first do pdf
            # now we need the pdf at each point (normalize)
            all_x = sorted(all_x)

            model_y_vals = set()
            for ens_name, xy_dict in ens_counts.items():
                norm_by_num_models = self.lookup_num_models_by_ensemble_name[ens_name]
                x_vals = []
                y_vals = []
                for x_val in sorted(xy_dict.keys()):
                    y_val = xy_dict[x_val]
                    x_vals.append(x_val)
                    # normalize, convert to percent

                    if first:
                        norm_yval = 100.0 * y_val / norm_by_num_models
                    else:
                        # For varying number of synoptic hours: will be wrong at start/end of model init
                        # TODO: FIX IF USE OTHER FUNCTION THAT EARLIEST FOR ANALYSIS
                        # Normalize to number of synoptic hours
                        norm_yval = 100.0 * y_val / (4.0 * norm_by_num_models)

                    y_vals.append(norm_yval)
                    all_y.append(norm_yval)
                    model_y_vals.add(norm_yval)

                pdf_series_data.append((ens_name, x_vals, y_vals))
                all_y.extend(model_y_vals)

            # calculate the mean now for all ensembles
            mean_data = {}
            # normalize by number of ensembles (equally weight each ensemble)
            weight = 1.0 / num_ensembles
            for ens_name, x_vals, y_vals in pdf_series_data:
                for x_val, y_val in list(zip(x_vals, y_vals)):
                    if x_val not in mean_data:
                        mean_data[x_val] = 0

                    mean_data[x_val] += y_val * weight

            mean_x_vals = []
            mean_y_vals = []
            for x_val in sorted(mean_data.keys()):
                y_val = mean_data[x_val]
                mean_x_vals.append(x_val)
                mean_y_vals.append(y_val)
                all_y.append(y_val)

            # ADD ALL-TCGEN to series (equally weighted by ensemble)
            pdf_series_data.append((total_ens_name, mean_x_vals, mean_y_vals))

            # cdf
            if cdf:
                #TODO: FIX FOR ANALYSIS CASE OTHER THAN EARLIEST
                all_y = []
                for (model_name, x_vals, y_vals) in pdf_series_data:
                    max_yval = None
                    series_x = []
                    series_y = []

                    for i, y_val in enumerate(y_vals):
                        if max_yval is None:
                            max_yval = y_val
                        else:
                            max_yval += y_val

                        series_x.append(x_vals[i])
                        series_y.append(max_yval)
                        all_y.append(max_yval)

                    cdf_series_data.append((model_name, series_x, series_y))

        if pdf:
            return pdf_series_data, list(all_x), list(all_y)
        elif cdf:
            return cdf_series_data, list(all_x), list(all_y)
        else:
            return [], [], []

    def generate_time_series(self, internal_id, dependent_variable):
        # Find the matching tc_candidate
        tc_candidate = next((tc for iid, tc in self.plotted_tc_candidates if iid == internal_id), None)

        if tc_candidate is None:
            return None

        # exclude points based on inclusion options
        exclusively_include_any_warm_or_cold_core = False
        exclusively_include_warm_core = False
        exclusively_include_cold_core = False
        if self.warm_core_previous_selected == 'Any':
            exclusively_include_any_warm_or_cold_core = True
        elif self.warm_core_previous_selected == 'Warm Core':
            exclusively_include_warm_core = True
        elif self.warm_core_previous_selected == 'Cold Core':
            exclusively_include_cold_core = True

        # Any would also include extratropical/frontal/etc... ?
        exclusively_include_any_cps = False
        exclusively_include_tropical_or_subtropical = False
        exclusively_include_tropical = False
        exclusively_include_subtropical = False
        if self.tc_phase_previous_selected == 'Any':
            exclusively_include_any_cps = True
        elif self.tc_phase_previous_selected == 'Tropical/Subtropical':
            exclusively_include_tropical_or_subtropical = True
        elif self.tc_phase_previous_selected == 'Tropical':
            exclusively_include_tropical = True
        elif self.tc_phase_previous_selected == 'Subtropical':
            exclusively_include_subtropical = True

        if exclusively_include_any_warm_or_cold_core and exclusively_include_any_cps:
            filter_by_phase = False
        else:
            filter_by_phase = True

        # Extract and convert datetimes
        datetimes = [
            self.convert_to_selected_timezone(tc['valid_time'], self.tz_previous_selected) for
            tc in tc_candidate]

        model_name = tc_candidate[0]['model_name']

        # Extract dependent variable values
        try:
            filtered_values = []
            filtered_datetimes = []
            for tc, dt in zip(tc_candidate, datetimes):
                exclude = False
                if filter_by_phase:
                    # for Unknowns we skip the exclusion

                    if 'warm_core' not in tc or tc['warm_core'] == 'Unknown':
                        skip_exclusion_by_warm_core = True
                    else:
                        skip_exclusion_by_warm_core = False

                    if 'tropical' not in tc or 'subtropical' not in tc or \
                        tc['tropical'] == 'Unknown' or tc['subtropical'] == 'Unknown':
                        skip_exclusion_by_cps = True
                    else:
                        skip_exclusion_by_cps = False

                    warm_core = False
                    tropical = False
                    subtropical = False

                    if not skip_exclusion_by_warm_core and not exclusively_include_any_warm_or_cold_core:
                        # 'Unknown' cases for EPS handled by skip_exclusion_* vars
                        if tc['warm_core'] == 'True':
                            warm_core = True
                        else:
                            warm_core = False
                        if tc['tropical'] == 'True':
                            tropical = True
                        else:
                            tropical = False

                    if not skip_exclusion_by_warm_core and not exclusively_include_any_warm_or_cold_core:
                        if exclusively_include_warm_core:
                            if not warm_core:
                                exclude = True
                        elif exclusively_include_cold_core:
                            if warm_core:
                                exclude = True

                    if not skip_exclusion_by_cps and not exclusively_include_any_cps:
                        if tc['tropical'] == 'True':
                            tropical = True
                        else:
                            tropical = False
                        if tc['subtropical'] == 'True':
                            subtropical = True
                        else:
                            subtropical = False

                    if not skip_exclusion_by_cps and not exclusively_include_any_cps:
                        if exclusively_include_tropical_or_subtropical:
                            if not (tropical or subtropical):
                                exclude = True
                        elif exclusively_include_tropical:
                            if not tropical:
                                exclude = True
                        elif exclusively_include_subtropical:
                            if not subtropical:
                                exclude = True

                if not exclude and dependent_variable in tc and tc[dependent_variable] is not None:
                    filtered_values.append(tc[dependent_variable])
                    filtered_datetimes.append(dt)

            if len(filtered_values) == 0:
                values = None
                datetimes = None
            else:
                values = filtered_values
                datetimes = filtered_datetimes
        except:
            traceback.print_exc()
            values = None
            datetimes = None

        # Return the time series data as a tuple (x, y)
        return (model_name, datetimes, values,)

    # get color index, num_ensembles, and num represented ensemble members
    def get_color_index_by_model_name(self, model_names_set):
        ensemble_names = set()
        num_represented = {}
        for model_name in model_names_set:
            if model_name[-5:] == 'TCGEN' or model_name == 'GLOBAL-DET' or model_name[-4:] == 'ATCF':
                ensemble_name = model_name
            elif model_name not in self.lookup_model_name_ensemble_name:
                # TODO: for now don't count ensembles for adeck data as we don't have a complete list of member names
                continue
            else:
                ensemble_name = self.lookup_model_name_ensemble_name[model_name]
            ensemble_names.add(ensemble_name)
            if ensemble_name not in num_represented:
                num_represented[ensemble_name] = 0
            num_represented[ensemble_name] += 1

        num_ensembles = len(ensemble_names)
        if len(model_names_set) <= 10 or num_ensembles <= 1:
            return None, num_ensembles, num_represented

        ensemble_to_enum = {}
        last_i = 0
        for i, ensemble_name in enumerate(ensemble_names):
            ensemble_to_enum[ensemble_name] = i
            last_i = i

        model_color_indices = {}
        # Must sort since model_names_set is not sorted and the series/legend will be in sorted order
        for model_name in sorted(model_names_set):
            if model_name not in self.lookup_model_name_ensemble_name:
                if model_name[-5:] == 'TCGEN' or model_name[-4:] == 'ATCF':
                    # PDF/CDF case with only 1 ensemble
                    last_i += 1
                    model_color_indices[model_name] = last_i
                elif model_name == 'GLOBAL-DET':
                    last_i += 1
                    model_color_indices[model_name] = last_i
                else:
                    # TODO: for now don't count ensembles for adeck data as we don't have a complete list of member names
                    model_color_indices[model_name] = 0
            else:
                model_color_indices[model_name] = ensemble_to_enum[self.lookup_model_name_ensemble_name[model_name]]

        return model_color_indices, num_ensembles, num_represented

    def get_current_tab_name(self):
        current_tab_index = self.notebook.index("current")
        return self.notebook_tab_names[current_tab_index]

    def on_cdf_range_changed(self, *args):
        self.do_update_cdf_chart = True

    def on_pdf_range_changed(self, *args):
        self.do_update_pdf_chart = True

    def on_tab_changed(self, event):
        tab_name = self.notebook.tab("current")["text"]
        if tab_name == 'PDF':
            if self.do_update_pdf_chart:
                self.update_pdf_chart()
        elif tab_name == 'CDF':
            if self.do_update_cdf_chart:
                self.update_cdf_chart()

    def get_timezones_with_offsets(self):
        """Get a list of all timezones with their UTC offsets."""
        timezones = []
        for tz in pytz.all_timezones:
            timezone = pytz.timezone(tz)
            offset = datetime.now(timezone).strftime('%z')
            hours, minutes = int(offset[:3]), int(offset[0] + offset[3:])
            formatted_offset = f"{'+' if hours >= 0 else '-'}{abs(hours):02}:{abs(minutes):02}"
            timezones.append(f"(UTC{formatted_offset}) {tz}")
        return timezones

    def get_timezones(self):
        """Get a list of all timezones with their UTC offsets."""
        return pytz.all_timezones

    @staticmethod
    def get_unique_datetimes(plotted_tc_candidates, selected_internal_storm_ids, tz_previous_selected):
        # Filter plotted_tc_candidates based on selected_internal_storm_ids
        filtered_candidates = [(iid, tc) for iid, tc in plotted_tc_candidates if
                               iid in selected_internal_storm_ids]

        # Collect all unique datetimes across all models
        all_datetimes = set()
        tz = tz_previous_selected
        for internal_id, tc_candidate in filtered_candidates:
            for tc in tc_candidate:
                valid_dt = tc['valid_time']
                converted_valid_dt = AnalysisDialog.convert_to_selected_timezone(valid_dt, tz)
                all_datetimes.add(converted_valid_dt)
        return sorted(all_datetimes)

    def ok(self, event=None):
        self.withdraw()
        self.update_idletasks()
        self.cancel()

    # Function to save the current figure
    def save(self, *args):
        current_tab_name = self.get_current_tab_name()
        # Use the current_tab_index to determine which figure to save
        if current_tab_name not in self.notebook_figs:
            return
        fig = self.notebook_figs[current_tab_name]

        current_time = datetime_utcnow().strftime("%Y-%m-%d-%H-%M-%S")
        # Create the screenshots folder if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        # Save the screenshot as a PNG file
        fig.savefig(f"screenshots/analysis-{current_time}-{current_tab_name}.png")

    def save_tz_session(self, *args):
        tz_name = self.tz_previous_selected.split(') ')[-1]
        print('save tz session', tz_name)
        App.save_analysis_tz('session', tz_name)

    def save_tz_setting(self, *args):
        tz_name = self.tz_previous_selected.split(') ')[-1]
        print('save tz setting', tz_name)
        App.save_analysis_tz('setting', tz_name)

    def show_converted_time(self):
        """Convert and show the current UTC time in the selected timezone."""
        now_utc = datetime_utcnow()  # current time in UTC (without tzinfo)
        converted_time = self.convert_to_selected_timezone(now_utc, self.tz_previous_selected)
        tk.messagebox.showinfo("Converted Time", f"Current time in {self.tz_previous_selected}:\n{converted_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def update_all_charts(self):
        self.update_vmax_chart()
        self.update_pres_chart()
        self.update_tc_size_chart()
        self.update_track_spread_chart()
        self.update_rvor_chart()
        self.update_pdf_chart()
        self.update_cdf_chart()

    def update_opt_preset_min_max(self, pdf=False, cdf=False):
        basin = self.basin_previous_selected

        if cdf:
            analysis = self.cdf_previous_selected
            preset = self.cdf_preset_previous_selected
            min, max = self.min_max_presets_by_preset_and_basin_and_analysis[preset][basin][analysis]
            self.cdf_min.set(min)
            self.cdf_max.set(max)
        if pdf:
            analysis = self.pdf_previous_selected
            preset = self.pdf_preset_previous_selected
            min, max = self.min_max_presets_by_preset_and_basin_and_analysis[preset][basin][analysis]
            self.pdf_min.set(min)
            self.pdf_max.set(max)

    def update_cdf_chart(self):
        self.do_update_cdf_chart = False
        with plt.style.context('dark_background'):
            if self.fig_cdf:
                plt.close(self.fig_cdf)
                self.fig_cdf = None
                self.notebook_figs['cdf'] = None
            if self.canvas_cdf is not None:
                self.canvas_cdf.get_tk_widget().destroy()
                self.canvas_cdf = None
            # Create a figure and axis object
            self.fig_cdf, self.ax_cdf = plt.subplots(figsize=(6, 4), dpi=100)

            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            track_series_data = []
            model_series_data = []
            model_names_set = set()
            # unlike the other analysis charts this does not generate the final time series for the chart
            # generate the intermediate data we need

            analysis_name = self.cdf_previous_selected
            analysis_min = self.cdf_min.get()
            analysis_max = self.cdf_max.get()
            analysis_range = False
            var_name = None
            var_proper_name = ''
            var_proper_long_desc = ''
            earliest_analysis = False
            if analysis_name == 'Earliest VMax @ 10m >= min':
                var_name = 'vmax10m'
                var_proper_name = 'VMax'
                var_proper_long_desc = f'Earliest VMax @ 10m >= {analysis_min} kt'
                earliest_analysis = True
            elif analysis_name == 'min <= Earliest VMax @ 10m <= max':
                var_name = 'vmax10m'
                var_proper_name = 'VMax'
                var_proper_long_desc = f'{analysis_min} <= Earliest VMax @ 10m <= {analysis_max} kt'
                earliest_analysis = True
            elif analysis_name == 'Earliest MSLP <= max':
                var_name = 'mslp_value'
                var_proper_name = 'MSLP'
                var_proper_long_desc = f'Earliest MSLP <= {analysis_max} hPa'
                earliest_analysis = True
            elif analysis_name == 'min <= Earliest MSLP <= max':
                var_name = 'mslp_value'
                var_proper_name = 'MSLP'
                var_proper_long_desc = f'{analysis_min} <= Earliest MSLP <= {analysis_max} hPa'
                earliest_analysis = True

            if var_name is None:
                # Should not reach
                return

            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, var_name)
                if y is not None:
                    # filter by analysis constraints (should return length 1 or 0 for Earliest analysis)
                    new_x, new_y = self.filter_series(first=earliest_analysis, min_val=analysis_min, max_val=analysis_max, x=x, y=y, cdf=True)
                    if new_y is not None:
                        any_data = True
                        #all_x.extend(new_x)
                        #all_y.extend(new_y)
                        track_series_data.append([model_name, new_x, new_y])
                        num_tracks_with_data += 1
                        model_names_set.add(model_name)

            if not any_data:
                return

            # Now convert the track_deries_data to model_series_data
            model_series_data = self.aggregate_tracks_to_model_series(track_series_data, first=earliest_analysis)

            model_series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            num_plotted_ensembles = 1
            # Process the data by ensemble
            global_det_case = False
            if self.total_ensembles >= 1 and self.possible_ensemble:
                # unlike the other charts, we are interested here in the TOTAL model count (not represented)
                # the series data is aggregated so we don't have individual models anyway
                if self.total_ensembles == 1:
                    series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                     model_series_freq=model_series_data,
                                                                     num_ensembles=1, cdf=True)
                    ens_names = [s[0] for s in series_data]
                    if self.ensemble_type == 'TCGEN' or self.ensemble_type == 'ATCF':
                        ens_name = self.previous_selected_combo
                    else:
                        ens_name = ens_names[-1]
                    total_num_models_in_ensemble = self.lookup_num_models_by_ensemble_name[ens_name]
                    ensemble_model_count_str = f'{self.total_ensembles} Ensemble ({total_num_models_in_ensemble} total members)'
                else:
                    # ALL-TCGEN case? (HARCODED) CHANGE IF ENSEMBLE SOURCES CHANGE
                    series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                                   model_series_freq=model_series_data,
                                                                                   num_ensembles=self.total_ensembles,
                                                                                   cdf=True)
                    ens_names = [s[0] for s in series_data]
                    if self.ensemble_type == 'TCGEN' or self.ensemble_type == 'ATCF':
                        ens_name = self.previous_selected_combo
                    else:
                        ens_name = ens_names[-1]
                    # Don't include member count in super ensemble as this is misleading since we are equally weighting by ensemble
                    ensemble_model_count_str = f'{self.total_ensembles} total Ensembles'
            else:
                # GLOBAL-DET case
                global_det_case = True
                series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                               model_series_freq=model_series_data,
                                                                               num_ensembles=0,
                                                                               cdf=True)
                ens_names = [s[0] for s in series_data]
                ensemble_model_count_str = f'{num_unique_models} models'

            num_colors = len(ens_names)
            distinct_colors = self.generate_distinct_colors(num_colors)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(ens_names)

            if not all_x:
                # no data
                return

            x_min = min(all_x)
            x_max = max(all_x)

            simplify_colors = len(series_data) > 8
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if global_det_case:
                    color_index = i
                else:
                    if not simplify_colors:
                        color_index = i
                    elif model_name == 'ALL-TCGEN' or model_name == 'ALL-ATCF':
                        color_index = 0
                    else:
                        # all members of the ensemble get the same color
                        color_index = num_colors - 1

                print(f"\nCDF for {model_name}, {var_proper_long_desc}:")
                print("")

                for x_val, y_val in zip(x, y):
                    print(f"  {x_val.strftime('%m/%d')}: {y_val:.2f}%")

                self.ax_cdf.plot(x, y, label=model_name, color=distinct_colors[color_index], marker='o', markersize=10)

            # Find the overall min/max of all x and y values
            x_min = min(all_x)
            x_max = max(all_x)
            y_max = max(all_y)
            y_lim_min = -5
            y_lim_max = 105
            # Set the x-axis to display dates
            self.ax_cdf.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_cdf.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            # self.ax_cdf.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_cdf.set_xlim([start_of_day, end_of_day])
            # Set y-axis major ticks and labels every 10 %
            self.ax_cdf.yaxis.set_major_locator(ticker.MultipleLocator(10))
            # Set y-axis minor ticks (no label) every 5 %
            self.ax_cdf.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_cdf.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_cdf.set_title(f'CDF of {var_proper_long_desc}, {ensemble_model_count_str}')
            self.ax_cdf.set_ylabel('%')
            self.ax_cdf.set_xlabel(self.tz_previous_selected)
            self.ax_cdf.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_cdf.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_cdf.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # Get the number of legend items
            num_items = len(self.ax_cdf.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_cdf.legend(
                loc='lower left',
                bbox_to_anchor=(1.052, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_cdf.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_cdf = FigureCanvasTkAgg(self.fig_cdf, master=self.cdf_frame)
            self.canvas_cdf.draw()
            self.canvas_cdf.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['cdf'] = self.fig_cdf

    def update_pdf_chart(self):
        self.do_update_pdf_chart = False
        with plt.style.context('dark_background'):
            if self.fig_pdf:
                plt.close(self.fig_pdf)
                self.fig_pdf = None
                self.notebook_figs['pdf'] = None
            if self.canvas_pdf is not None:
                self.canvas_pdf.get_tk_widget().destroy()
                self.canvas_pdf = None
            # Create a figure and axis object
            self.fig_pdf, self.ax_pdf = plt.subplots(figsize=(6, 4), dpi=100)

            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            track_series_data = []
            model_series_data = []
            model_names_set = set()
            # unlike the other analysis charts this does not generate the final time series for the chart
            # generate the intermediate data we need

            analysis_name = self.pdf_previous_selected
            analysis_min = self.pdf_min.get()
            analysis_max = self.pdf_max.get()
            analysis_range = False
            var_name = None
            var_proper_name = ''
            var_proper_long_desc = ''
            earliest_analysis = False
            if analysis_name == 'Earliest VMax @ 10m >= min':
                var_name = 'vmax10m'
                var_proper_name = 'VMax'
                var_proper_long_desc = f'Earliest VMax @ 10m >= {analysis_min} kt'
                earliest_analysis = True
            elif analysis_name == 'min <= Earliest VMax @ 10m <= max':
                var_name = 'vmax10m'
                var_proper_name = 'VMax'
                var_proper_long_desc = f'{analysis_min} <= Earliest VMax @ 10m <= {analysis_max} kt'
                earliest_analysis = True
            elif analysis_name == 'Earliest MSLP <= max':
                var_name = 'mslp_value'
                var_proper_name = 'MSLP'
                var_proper_long_desc = f'Earliest MSLP <= {analysis_max} hPa'
                earliest_analysis = True
            elif analysis_name == 'min <= Earliest MSLP <= max':
                var_name = 'mslp_value'
                var_proper_name = 'MSLP'
                var_proper_long_desc = f'{analysis_min} <= Earliest MSLP <= {analysis_max} hPa'
                earliest_analysis = True

            if var_name is None:
                # Should not reach
                return

            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, var_name)
                if y is not None:
                    # filter by analysis constraints (should return length 1 or 0 for Earliest analysis)
                    new_x, new_y = self.filter_series(first=earliest_analysis, min_val=analysis_min, max_val=analysis_max, x=x, y=y, pdf=True)
                    if new_y is not None:
                        any_data = True
                        track_series_data.append([model_name, new_x, new_y])
                        num_tracks_with_data += 1
                        model_names_set.add(model_name)

            if not any_data:
                return

            # Now convert the track_deries_data to model_series_data
            model_series_data = self.aggregate_tracks_to_model_series(track_series_data, first=earliest_analysis)

            model_series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            num_plotted_ensembles = 1
            # Process the data by ensemble
            global_det_case = False
            if self.total_ensembles >= 1 and self.possible_ensemble:
                # unlike the other charts, we are interested here in the TOTAL model count (not represented)
                # the series data is aggregated so we don't have individual models anyway
                if self.total_ensembles == 1:
                    series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                     model_series_freq=model_series_data,
                                                                     num_ensembles=1, pdf=True)
                    ens_names = [s[0] for s in series_data]
                    if self.ensemble_type == 'TCGEN' or self.ensemble_type == 'ATCF':
                        ens_name = self.previous_selected_combo
                    else:
                        ens_name = ens_names[-1]
                    total_num_models_in_ensemble = self.lookup_num_models_by_ensemble_name[ens_name]
                    ensemble_model_count_str = f'{self.total_ensembles} Ensemble ({total_num_models_in_ensemble} total members)'
                else:
                    # ALL-TCGEN case? (HARCODED) CHANGE IF ENSEMBLE SOURCES CHANGE
                    series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                     model_series_freq=model_series_data,
                                                                     num_ensembles=self.total_ensembles, pdf=True)
                    ens_names = [s[0] for s in series_data]
                    if self.ensemble_type == 'TCGEN' or self.ensemble_type == 'ATCF':
                        ens_name = self.previous_selected_combo
                    else:
                        ens_name = ens_names[-1]
                    # Don't include member count in super ensemble as this is misleading since we are equally weighting by ensemble
                    ensemble_model_count_str = f'{self.total_ensembles} total Ensembles'
            else:
                # GLOBAL-DET case
                global_det_case = True
                series_data, all_x, all_y = self.generate_freq_analysis_series(first=earliest_analysis,
                                                                 model_series_freq=model_series_data, num_ensembles=0,
                                                                 pdf=True)
                ens_names = [s[0] for s in series_data]
                ensemble_model_count_str = f'{num_unique_models} models'

            num_colors = len(ens_names)
            distinct_colors = self.generate_distinct_colors(num_colors)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(ens_names)

            if not all_x:
                # no data
                return

            x_min = min(all_x)
            x_max = max(all_x)

            # Create a dictionary to store the offsets for overlapping x-values
            offsets = {}

            min_marker_size = 25
            max_marker_size = 100
            marker_size = max_marker_size
            marker_decr = 10

            # JITTER CODE
            # Loop over the series data to find overlapping x-values
            for i, series in enumerate(series_data):
                model_name, x, y = series
                print(f"\nPDF for {model_name}, {var_proper_long_desc}:")
                print("")

                for x_val, y_val in zip(x,y):
                    print(f"  {x_val.strftime('%m/%d')}: {y_val:.2f}%")

                    if x_val in offsets:
                        offsets[x_val].append((i, y_val))
                    else:
                        offsets[x_val] = [(i, y_val)]

            # Calculate the offsets for overlapping x-values
            offset_width = 4.0
            xoffsets = {}
            bin_width = 5.0
            # Calculate the offsets for overlapping x-values
            for x_val, series_indices in offsets.items():
                if len(series_indices) > 1:
                    # Bin the y-values by 5
                    y_bins = {}
                    for series_index, y_val in series_indices:
                        y_bin = int(round(y_val / bin_width) * bin_width)
                        if y_bin in y_bins:
                            y_bins[y_bin].append((series_index, y_val))
                        else:
                            y_bins[y_bin] = [(series_index, y_val)]

                    bin_marker_size = max_marker_size
                    for j, (y_bin, bin_series_indices) in enumerate(y_bins.items()):
                        num_bin_series_indices = len(bin_series_indices)
                        if num_bin_series_indices == 1:
                            # don't offset this bin
                            if x_val not in xoffsets:
                                xoffsets[x_val] = {}

                            if series_index not in xoffsets[x_val]:
                                xoffsets[x_val][series_index] = {}

                            xoffsets[x_val][series_index][y_bin] = timedelta(hours=0)
                        else:
                            offset_range = timedelta(hours=offset_width / num_bin_series_indices)
                            center = offset_range * (num_bin_series_indices - 1) / 2
                            for k, (series_index, y_val) in enumerate(bin_series_indices):
                                if x_val not in xoffsets:
                                    xoffsets[x_val] = {}

                                if series_index not in xoffsets[x_val]:
                                    xoffsets[x_val][series_index] = {}

                                xoffsets[x_val][series_index][y_bin] = (offset_range * k) - center
                                bin_marker_size -= marker_decr

                        marker_size = max(bin_marker_size, min_marker_size)

            # END JITTER CODE

            simplify_colors = len(series_data) > 8
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if global_det_case:
                    color_index = i
                else:
                    if not simplify_colors:
                        color_index = i
                    elif model_name == 'ALL-TCGEN' or model_name == 'ALL-ATCF':
                        color_index = 0
                    else:
                        # all members of the ensemble get the same color
                        color_index = num_colors - 1

                x_jittered = []
                # offset the data slightly horizontally if there are multiple values at the time
                for x_val, y_val in list(zip(x,y)):
                    y_bin = int(round(y_val / bin_width) * bin_width)
                    if x_val in xoffsets and len(xoffsets[x_val]) > 1 and i in xoffsets[x_val] and y_bin in xoffsets[x_val][i]:
                        x_jittered.append(x_val + xoffsets[x_val][i][y_bin])
                    else:
                        x_jittered.append(x_val)

                self.ax_pdf.scatter(x_jittered, y, label=model_name, color=distinct_colors[color_index], marker='o', s=marker_size)
                #self.ax_pdf.scatter(x, y, label=model_name, color=distinct_colors[color_index], marker='o', s=100)

            # Find the overall min/max of all x and y values
            y_max = max(all_y)
            y_lim_min = -5
            y_lim_max = 105
            # Set the x-axis to display dates
            self.ax_pdf.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_pdf.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            # self.ax_pdf.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_pdf.set_xlim([start_of_day, end_of_day])
            # Set y-axis major ticks and labels every 10 %
            self.ax_pdf.yaxis.set_major_locator(ticker.MultipleLocator(10))
            # Set y-axis minor ticks (no label) every 5 %
            self.ax_pdf.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_pdf.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_pdf.set_title(f'PDF of {var_proper_long_desc}, {ensemble_model_count_str}')
            self.ax_pdf.set_ylabel('%')
            self.ax_pdf.set_xlabel(self.tz_previous_selected)
            self.ax_pdf.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_pdf.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_pdf.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # Get the number of legend items
            num_items = len(self.ax_pdf.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_pdf.legend(
                loc='lower left',
                bbox_to_anchor=(1.052, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_pdf.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_pdf = FigureCanvasTkAgg(self.fig_pdf, master=self.pdf_frame)
            self.canvas_pdf.draw()
            self.canvas_pdf.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['pdf'] = self.fig_pdf

    def update_pres_chart(self):
        with plt.style.context('dark_background'):
            # Create a figure and axis object
            if self.fig_pres:
                plt.close(self.fig_pres)
                self.fig_pres = None
                self.notebook_figs['size'] = None
            if self.canvas_pres is not None:
                self.canvas_pres.get_tk_widget().destroy()
                self.canvas_pres = None
            self.fig_pres, self.ax_pres = plt.subplots(figsize=(6, 4), dpi=100)
            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            model_names_set = set()
            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, 'mslp_value')
                if y is not None:
                    any_data = True
                    all_x.extend(x)
                    all_y.extend(y)
                    series_data.append([model_name, x, y])
                    num_tracks_with_data += 1
                    model_names_set.add(model_name)
            if not any_data:
                return

            series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            # get color indices if have many models and more than one ensemble
            # also get the number of ensembles, and number of members represented from each ensemble (not the total in the ensemble)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(model_names_set)
            # self.possible_ensemble is there to make sure we don't label our own tracker which only has global deterministic models
            if num_represented and num_ensembles > 1 and self.possible_ensemble:
                # in rough alphabetical order (AP, CP, EE, NP)
                ensemble_model_counts = []
                for ens_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']:
                    if ens_name in num_represented:
                        ensemble_model_counts.append(str(num_represented[ens_name]))
                ensemble_model_count_str = ", ".join(ensemble_model_counts)
                if ensemble_model_count_str:
                    ensemble_model_count_str = f', {num_ensembles} Ensembles ({ensemble_model_count_str})'
            else:
                ensemble_model_count_str = ''

            ensemble_count_str = []
            if color_index_by_model_name:
                use_color_index = True
                num_colors = num_ensembles
            else:
                use_color_index = False
                num_colors = num_tracks_with_data

            distinct_colors = self.generate_distinct_colors(num_colors)
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if use_color_index:
                    color_index = color_index_by_model_name[model_name]
                else:
                    color_index = i
                self.ax_pres.plot(x, y, label=model_name, color=distinct_colors[color_index])

            # Find the overall min/max of all x and y values
            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            y_lim_min = -10 + ((y_min // 10) * 10)
            y_lim_max = 10 + ((y_max // 10) * 10)
            # Set the x-axis to display dates
            self.ax_pres.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_pres.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            self.ax_pres.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_pres.set_xlim([start_of_day, end_of_day])
            self.ax_pres.yaxis.set_major_locator(ticker.MultipleLocator(10))
            self.ax_pres.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_pres.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_pres.set_title(f'TC MSLP, {num_tracks_with_data}/{num_storm_ids} tracks, {num_unique_models} models{ensemble_model_count_str}\n\nMin MSLP: {int(round(y_min))} mb')
            self.ax_pres.set_ylabel('MSLP (mb)')
            self.ax_pres.set_xlabel(self.tz_previous_selected)
            self.ax_pres.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_pres.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_pres.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)
            # TODO: add hlines for NATL/EPAC pressure medians on initial classification

            # Get the number of legend items
            num_items = len(self.ax_pres.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_pres.legend(
                loc='lower left',
                bbox_to_anchor=(1.05, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_pres.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_pres = FigureCanvasTkAgg(self.fig_pres, master=self.pressure_frame)
            self.canvas_pres.draw()
            self.canvas_pres.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['pres'] = self.fig_pres

    def update_rvor_chart(self):
        # Create a figure and axis object
        with plt.style.context('dark_background'):
            if self.fig_rvor:
                plt.close(self.fig_rvor)
                self.fig_rvor = None
                self.notebook_figs['rvor'] = None
            if self.canvas_rvor is not None:
                self.canvas_rvor.get_tk_widget().destroy()
                self.canvas_rvor = None
            self.fig_rvor, self.ax_rvor = plt.subplots(figsize=(6, 4), dpi=100)
            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            model_names_set = set()
            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, 'rv850max')
                if y is not None:
                    any_data = True
                    all_x.extend(x)
                    all_y.extend(y)
                    series_data.append([model_name, x, y])
                    num_tracks_with_data += 1
                    model_names_set.add(model_name)
            if not any_data:
                return

            series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            # get color indices if have many models and more than one ensemble
            # also get the number of ensembles, and number of members represented from each ensemble (not the total in the ensemble)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(
                model_names_set)
            # self.possible_ensemble is there to make sure we don't label our own tracker which only has global deterministic models
            if num_represented and num_ensembles > 1 and self.possible_ensemble:
                # in rough alphabetical order (AP, CP, EE, NP)
                ensemble_model_counts = []
                for ens_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']:
                    if ens_name in num_represented:
                        ensemble_model_counts.append(str(num_represented[ens_name]))
                ensemble_model_count_str = ", ".join(ensemble_model_counts)
                if ensemble_model_count_str:
                    ensemble_model_count_str = f', {num_ensembles} Ensembles ({ensemble_model_count_str})'
            else:
                ensemble_model_count_str = ''

            ensemble_count_str = []
            if color_index_by_model_name:
                use_color_index = True
                num_colors = num_ensembles
            else:
                use_color_index = False
                num_colors = num_tracks_with_data

            distinct_colors = self.generate_distinct_colors(num_colors)
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if use_color_index:
                    color_index = color_index_by_model_name[model_name]
                else:
                    color_index = i
                self.ax_rvor.plot(x, y, label=model_name, color=distinct_colors[color_index])

            # Find the overall min/max of all x and y values
            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            y_lim_min = min(0, -10 + ((y_min // 10) * 10))
            y_lim_max = 10 + ((y_max // 10) * 10)
            # Set the x-axis to display dates
            self.ax_rvor.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_rvor.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            self.ax_rvor.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_rvor.set_xlim([start_of_day, end_of_day])
            if y_lim_max - y_lim_min > 150:
                self.ax_rvor.yaxis.set_major_locator(ticker.MultipleLocator(20))
                self.ax_rvor.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            else:
                self.ax_rvor.yaxis.set_major_locator(ticker.MultipleLocator(10))
                self.ax_rvor.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_rvor.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_rvor.set_title(f'TC RVOR (* 10^-5 m/s), {num_tracks_with_data}/{num_storm_ids} tracks, {num_unique_models} models{ensemble_model_count_str}\n\nMax RVOR: {round(y_max,1)} * 10^-5 m/s')
            self.ax_rvor.set_ylabel('Relative Vorticity (* 10^-5 m/s)')
            self.ax_rvor.set_xlabel(self.tz_previous_selected)
            self.ax_rvor.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_rvor.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_rvor.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # Get the number of legend items
            num_items = len(self.ax_rvor.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_rvor.legend(
                loc='lower left',
                bbox_to_anchor=(1.05, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_rvor.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_rvor = FigureCanvasTkAgg(self.fig_rvor, master=self.rvor_frame)
            self.canvas_rvor.draw()
            self.canvas_rvor.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['rvor'] = self.fig_rvor

    def update_track_spread_chart(self):
        # Create a figure and axis object
        with plt.style.context('dark_background'):
            if self.fig_track_spread:
                plt.close(self.fig_track_spread)
                self.fig_track_spread = None
                self.notebook_figs['track'] = None
            if self.canvas_track_spread is not None:
                self.canvas_track_spread.get_tk_widget().destroy()
                self.canvas_track_spread = None

            # The following is only used here to get counts
            num_tracks_with_data = 0
            series_data = []
            model_names_set = set()
            any_data = False
            filtered_storm_ids = []
            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, 'lon')
                if y is not None:
                    any_data = True
                    num_tracks_with_data += 1
                    model_names_set.add(model_name)
                    filtered_storm_ids.append(internal_storm_id)
            if not any_data:
                return

            num_unique_models = len(model_names_set)
            num_storm_ids = len(self.selected_internal_storm_ids)
            # get the number of ensembles, and number of members represented from each ensemble (not the total in the ensemble)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(
                model_names_set)
            # self.possible_ensemble is there to make sure we don't label our own tracker which only has global deterministic models
            if num_represented and num_ensembles > 1 and self.possible_ensemble:
                # in rough alphabetical order (AP, CP, EE, NP)
                ensemble_model_counts = []
                for ens_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']:
                    if ens_name in num_represented:
                        ensemble_model_counts.append(str(num_represented[ens_name]))
                ensemble_model_count_str = ", ".join(ensemble_model_counts)
                if ensemble_model_count_str:
                    ensemble_model_count_str = f', {num_ensembles} Ensembles ({ensemble_model_count_str})'
            else:
                ensemble_model_count_str = ''

            # Calculate the spreads now
            unique_datetimes = AnalysisDialog.get_unique_datetimes(self.plotted_tc_candidates, self.selected_internal_storm_ids, self.tz_previous_selected)
            if not unique_datetimes:
                return
            mean_track = AnalysisDialog.calculate_mean_track(self.plotted_tc_candidates, unique_datetimes, filtered_storm_ids)
            if not mean_track:
                return
            cross_track_spread, along_track_spread = self.calculate_spread(mean_track, filtered_storm_ids)
            if not cross_track_spread or not along_track_spread:
                return

            self.fig_track_spread, self.ax_track_spread = plt.subplots(figsize=(6, 4), dpi=100)

            # Plot cross-track and along-track spreads
            self.ax_track_spread.plot(unique_datetimes, cross_track_spread, label='Cross-Track Spread', color='blue',
                                linestyle='-', marker='o')
            self.ax_track_spread.plot(unique_datetimes, along_track_spread, label='Along-Track Spread', color='red',
                                linestyle='-', marker='x')

            # Set the x-axis to display dates
            self.ax_track_spread.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_track_spread.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            self.ax_track_spread.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = unique_datetimes[0].replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = unique_datetimes[-1].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_track_spread.set_xlim([start_of_day, end_of_day])

            # Set y-axis limits
            y_min = min(min(cross_track_spread, default=0), min(along_track_spread, default=0))
            y_max = max(max(cross_track_spread, default=0), max(along_track_spread, default=0))
            y_lim_min = min(0, y_min - 10)  # Adding buffer for visual clarity
            y_lim_max = y_max + 10
            if y_lim_max - y_lim_min > 150:
                self.ax_track_spread.yaxis.set_major_locator(ticker.MultipleLocator(20))
                self.ax_track_spread.yaxis.set_minor_locator(ticker.MultipleLocator(10))
            else:
                self.ax_track_spread.yaxis.set_major_locator(ticker.MultipleLocator(10))
                self.ax_track_spread.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_track_spread.set_ylim([y_lim_min, y_lim_max])

            # Set title and labels
            self.ax_track_spread.set_title(f'Track Spread Over Time, {num_tracks_with_data}/{num_storm_ids} tracks, {num_unique_models} models{ensemble_model_count_str}')
            self.ax_track_spread.set_ylabel('Cross-track, Along-track spread (n. mi.)')
            self.ax_track_spread.set_xlabel('Date')
            self.ax_track_spread.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_track_spread.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_track_spread.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # Get the number of legend items
            num_items = len(self.ax_track_spread.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_track_spread.legend(
                loc='lower left',
                bbox_to_anchor=(1.05, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_track_spread.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_track_spread = FigureCanvasTkAgg(self.fig_track_spread, master=self.track_frame)
            self.canvas_track_spread.draw()
            self.canvas_track_spread.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['track'] = self.fig_track_spread

    def update_tc_size_chart(self):
        with plt.style.context('dark_background'):
            if self.fig_size:
                plt.close(self.fig_size)
                self.fig_size = None
                self.notebook_figs['size'] = None
            if self.canvas_size is not None:
                self.canvas_size.get_tk_widget().destroy()
                self.canvas_size = None
            # Create a figure and axis object
            self.fig_size, self.ax_size = plt.subplots(figsize=(6, 4), dpi=100)
            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            model_names_set = set()
            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, 'roci')
                if y is not None:
                    any_data = True
                    all_x.extend(x)
                    all_y.extend(y)
                    series_data.append([model_name, x, y])
                    num_tracks_with_data += 1
                    model_names_set.add(model_name)
            if not any_data:
                return

            series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            # get color indices if have many models and more than one ensemble
            # also get the number of ensembles, and number of members represented from each ensemble (not the total in the ensemble)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(
                model_names_set)
            # self.possible_ensemble is there to make sure we don't label our own tracker which only has global deterministic models
            if num_represented and num_ensembles > 1 and self.possible_ensemble:
                # in rough alphabetical order (AP, CP, EE, NP)
                ensemble_model_counts = []
                for ens_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']:
                    if ens_name in num_represented:
                        ensemble_model_counts.append(str(num_represented[ens_name]))
                ensemble_model_count_str = ", ".join(ensemble_model_counts)
                if ensemble_model_count_str:
                    ensemble_model_count_str = f', {num_ensembles} Ensembles ({ensemble_model_count_str})'
            else:
                ensemble_model_count_str = ''

            ensemble_count_str = []
            if color_index_by_model_name:
                use_color_index = True
                num_colors = num_ensembles
            else:
                use_color_index = False
                num_colors = num_tracks_with_data

            distinct_colors = self.generate_distinct_colors(num_colors)
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if use_color_index:
                    color_index = color_index_by_model_name[model_name]
                else:
                    color_index = i
                self.ax_size.plot(x, y, label=model_name, color=distinct_colors[color_index])

            # Find the overall min/max of all x and y values
            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            #y_lim_min = min(0, -100 + ((y_min // 100) * 100))
            y_lim_min = 0
            y_lim_max = 100 + ((y_max // 100) * 100)
            # Set the x-axis to display dates
            self.ax_size.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_size.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            self.ax_size.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_size.set_xlim([start_of_day, end_of_day])
            self.ax_size.yaxis.set_major_locator(ticker.MultipleLocator(100))
            self.ax_size.yaxis.set_minor_locator(ticker.MultipleLocator(50))
            self.ax_size.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_size.set_title(f'TC ROCI (km), {num_tracks_with_data}/{num_storm_ids} tracks, {num_unique_models} models{ensemble_model_count_str}\n\n(Min, Max) ROCI: ({int(round(y_min))}, {int(round(y_max))}) km')
            self.ax_size.set_ylabel('ROCI (km)')
            self.ax_size.set_xlabel(self.tz_previous_selected)
            self.ax_size.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_size.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_size.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # ROCI scale (degrees of latitude)
            roci_scale = np.array([0, 2, 3, 6, 8]).astype(float)
            # convert degree latitude to km
            roci_scale = roci_scale * 111.1
            roci_labels = ['Very small', 'Small', 'Medium', 'Large', 'Very large']

            # Add horizontal gridlines at the ROCI scale with labels
            for i, (value, label) in enumerate(zip(roci_scale, roci_labels)):
                # convert to degrees from km
                if value >= self.ax_size.get_ylim()[0] and value <= self.ax_size.get_ylim()[1]:
                    self.ax_size.axhline(y=value, color='blue', linestyle='-', linewidth=1.5, zorder=0)
                # Calculate the y-position for the label, halfway between the current and next line
                if i < len(roci_scale) - 1:
                    y_pos = (value + (roci_scale[i + 1])) / 2
                else:
                    y_pos = value * 111.1  # for the last label, just use the line value
                if y_pos >= self.ax_size.get_ylim()[0] and value <= self.ax_size.get_ylim()[1]:
                    self.ax_size.text(1.02, y_pos / self.ax_size.get_ylim()[1], label, ha='left', va='center',
                                      transform=self.ax_size.transAxes)

            # Get the number of legend items
            num_items = len(self.ax_size.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_size.legend(
                loc='lower left',
                bbox_to_anchor=(1.1, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # Use tight_layout to ensure everything fits within the figure area
            # self.fig_vmax.tight_layout(rect=[0, 0, 0.95, 1])
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_size.subplots_adjust(left=70 / FULL_SCREEN_WIDTH, right=(0.80 - est_legend_width),
                                          bottom=60 / FULL_SCREEN_HEIGHT, top=0.9)

            # Use tight_layout to ensure everything fits within the figure area
            #self.fig_size.tight_layout(rect=[0, 0, 0.95, 1])
            # Create a canvas to display the plot in the frame
            self.canvas_size = FigureCanvasTkAgg(self.fig_size, master=self.size_frame)
            self.canvas_size.draw()
            self.canvas_size.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['size'] = self.fig_size

    def update_vmax_chart(self):
        with plt.style.context('dark_background'):
            if self.fig_vmax:
                plt.close(self.fig_vmax)
                self.fig_vmax = None
                self.notebook_figs['vmax'] = None
            if self.canvas_vmax is not None:
                self.canvas_vmax.get_tk_widget().destroy()
                self.canvas_vmax = None
            # Create a figure and axis object
            self.fig_vmax, self.ax_vmax = plt.subplots(figsize=(6, 4), dpi=100)

            # Initialize lists to store all x and y values
            all_x = []
            all_y = []
            # Plot the data for all storms
            any_data = False
            num_tracks_with_data = 0
            series_data = []
            model_names_set = set()
            for internal_storm_id in self.selected_internal_storm_ids:
                model_name, x, y = self.generate_time_series(internal_storm_id, 'vmax10m')
                if y is not None:
                    any_data = True
                    all_x.extend(x)
                    all_y.extend(y)
                    series_data.append([model_name, x, y])
                    num_tracks_with_data += 1
                    model_names_set.add(model_name)
            if not any_data:
                return

            series_data.sort(key=lambda x: x[0])
            num_unique_models = len(model_names_set)

            # get color indices if have many models and more than one ensemble
            # also get the number of ensembles, and number of members represented from each ensemble (not the total in the ensemble)
            color_index_by_model_name, num_ensembles, num_represented = self.get_color_index_by_model_name(model_names_set)
            # self.possible_ensemble is there to make sure we don't label our own tracker which only has global deterministic models
            if num_represented and num_ensembles > 1 and self.possible_ensemble:
                # in rough alphabetical order (AP, CP, EE, NP)
                ensemble_model_counts = []
                for ens_name in ['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN']:
                    if ens_name in num_represented:
                        ensemble_model_counts.append(str(num_represented[ens_name]))
                ensemble_model_count_str = ", ".join(ensemble_model_counts)
                if ensemble_model_count_str:
                    ensemble_model_count_str = f', {num_ensembles} Ensembles ({ensemble_model_count_str})'
            else:
                ensemble_model_count_str = ''

            ensemble_count_str = []
            if color_index_by_model_name:
                use_color_index = True
                num_colors = num_ensembles
            else:
                use_color_index = False
                num_colors = num_tracks_with_data

            distinct_colors = self.generate_distinct_colors(num_colors)
            for i, series in enumerate(series_data):
                model_name, x, y = series
                if use_color_index:
                    color_index = color_index_by_model_name[model_name]
                else:
                    color_index = i
                self.ax_vmax.plot(x, y, label=model_name, color=distinct_colors[color_index])

            # Find the overall min/max of all x and y values
            x_min = min(all_x)
            x_max = max(all_x)
            y_max = max(all_y)
            y_lim_min = 0
            y_lim_max = 10 + ((y_max // 10) * 10)
            # Set the x-axis to display dates
            self.ax_vmax.xaxis.set_major_locator(mdates.DayLocator())
            self.ax_vmax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # Set the x-axis to display minor ticks every 6 hours
            self.ax_vmax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            # Set the x-axis limits to start at the first day - 1 and end at the last day + 1
            start_of_day = x_min.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_of_day = x_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.ax_vmax.set_xlim([start_of_day, end_of_day])
            # Set y-axis major ticks and labels every 10 kt
            self.ax_vmax.yaxis.set_major_locator(ticker.MultipleLocator(10))
            # Set y-axis minor ticks (no label) every 5 kt
            self.ax_vmax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            self.ax_vmax.set_ylim([y_lim_min, y_lim_max])
            # Set title and labels
            num_storm_ids = len(self.selected_internal_storm_ids)
            self.ax_vmax.set_title(f'TC VMax @ 10m, {num_tracks_with_data}/{num_storm_ids} tracks, {num_unique_models} models{ensemble_model_count_str}\n\nPeak VMax: {int(round(y_max))} kt')
            self.ax_vmax.set_ylabel('VMax @ 10m (kt)')
            self.ax_vmax.set_xlabel(self.tz_previous_selected)
            self.ax_vmax.grid(which='major', axis='y', linestyle=':', linewidth=0.8)
            self.ax_vmax.grid(which='major', axis='x', linestyle=':', linewidth=1.0)
            self.ax_vmax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

            # Saffir-Simpson scale (kt)
            saffir_simpson_scale = [34, 64, 83, 96, 113, 137]
            saffir_simpson_labels = ['TS', 'CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5']
            # Add horizontal gridlines at the Saffir-Simpson scale with labels
            for value, label in zip(saffir_simpson_scale, saffir_simpson_labels):
                self.ax_vmax.axhline(y=value, color='blue', linestyle='-', linewidth=1.5, zorder=0)
                if value >= self.ax_vmax.get_ylim()[0] and value <= self.ax_vmax.get_ylim()[1]:
                    self.ax_vmax.text(1.02, value / self.ax_vmax.get_ylim()[1], label, ha='left', va='center',
                                      transform=self.ax_vmax.transAxes)

            # Get the number of legend items
            num_items = len(self.ax_vmax.get_legend_handles_labels()[0])
            # Calculate the number of columns based on the available space
            available_height = FULL_SCREEN_HEIGHT * 0.9
            max_items_per_column = available_height / 30
            num_columns = min(max(1, num_items // max_items_per_column), 5)
            # Create the legend with the calculated number of columns
            self.ax_vmax.legend(
                loc='lower left',
                bbox_to_anchor=(1.052, 0),
                borderaxespad=0,
                ncol=num_columns
            )
            # all fine tuned values since we haven't drawn/packed it yet
            est_legend_width = num_columns * 80 / FULL_SCREEN_WIDTH
            self.fig_vmax.subplots_adjust(left=90/FULL_SCREEN_WIDTH, right=(0.88 - est_legend_width), bottom=60/FULL_SCREEN_HEIGHT, top=0.9)

            # Create a canvas to display the plot in the frame
            self.canvas_vmax = FigureCanvasTkAgg(self.fig_vmax, master=self.intensity_frame)
            self.canvas_vmax.draw()
            self.canvas_vmax.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.notebook_figs['vmax'] = self.fig_vmax

# For extrema annotations (holds a collection of AnnotatedCircle)
class AnnotatedCircles:
    ax = None
    # this contains the plain Circle artist objects that each AnnotatedCircle contributes to
    circle_handles = None
    rtree_p = index.Property()
    rtree_idx = index.Index(properties=rtree_p)
    # only increment counter so we only have unique ids
    counter = 0

    class AnnotatedCircle:
        def __init__(self, draggable_annotation, circle_handle, rtree_id, internal_id=None, point_index=None):
            self.draggable_annotation_object = draggable_annotation
            self.circle_handle_object = circle_handle
            self.visible = True
            # id in the index for rtree_idx
            self.rtree_id = rtree_id
            self.internal_id = internal_id
            self.point_index = point_index

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
                # self.annotation_handles = None
                if App.draggable_annotations and self.draggable_annotation_object in App.draggable_annotations:
                    App.draggable_annotations.remove(self.draggable_annotation_object)
                self.draggable_annotation_object = None
            if self.circle_handle_object:
                try:
                    AnnotatedCircles.removeCircle(self)
                    self.circle_handle_object.remove()
                    removed = True
                except:
                    traceback.print_exc()
                    pass
                self.circle_handle_object = None
            if removed:
                AnnotatedCircles.delete_point(self.rtree_id)

    # add AnnotatedCircle
    @classmethod
    def add_point(cls, coords):
        cls.counter += 1
        cls.rtree_idx.insert(cls.counter, coords)
        return cls.counter

    @classmethod
    def add(cls, lat=None, lon=None, label=None, label_color=DEFAULT_ANNOTATE_TEXT_COLOR, internal_id=None, point_index=None):
        if lat is None or lon is None or label is None or App.ax is None:
            return None
        if cls.circle_handles is None:
            cls.circle_handles = {}
        if App.draggable_annotations is None:
            App.draggable_annotations = []

        # Since they are draggable, preference should be to annotate all (even overlapped)
        # no longer check for overlap based on lat/lon, only on internal id and point to make sure we don't annotate twice
        if cls.has_overlap(internal_id=internal_id, point_index=point_index):
            return None

        lon_offset, lat_offset = cls.calculate_offset_pixels()
        # calculate radius of pixels in degrees
        radius_pixels_degrees = cls.calculate_radius_pixels()
        circle_handle = Circle((lon, lat), radius=radius_pixels_degrees, color=DEFAULT_ANNOTATE_MARKER_COLOR,
                               fill=False, linestyle='dotted', linewidth=2, alpha=0.8,
                               transform=ccrs.PlateCarree())
        App.ax.add_patch(circle_handle)
        rtree_id = cls.add_point((lon, lat, lon, lat))
        if internal_id not in cls.circle_handles:
            cls.circle_handles[internal_id] = {}
        cls.circle_handles[internal_id][point_index] = circle_handle

        bbox_props = {
            'boxstyle': 'round,pad=0.3',
            'edgecolor': '#FFFFFF',
            'facecolor': '#000000',
            'alpha': 1.0
        }

        # Original annotation creation with DraggableAnnotation integration
        annotation_handle = App.ax.annotate(label, xy=(lon, lat),
                                            xytext=(lon + lon_offset, lat + lat_offset),
                                            textcoords='data', color=label_color,
                                            fontsize=12, ha='left', va='bottom', bbox=bbox_props)

        # Create DraggableAnnotation instance
        draggable_annotation = DraggableAnnotation(
            annotation_handle, (lon, lat), bbox_props)

        annotated_circle = cls.AnnotatedCircle(draggable_annotation, circle_handle, rtree_id, internal_id=internal_id, point_index=point_index)

        # create a way access the annotated_circle from the draggable annotation
        draggable_annotation.set_circle_annotation(annotated_circle)
        App.draggable_annotations.append(draggable_annotation)

        return annotated_circle
        # draw later as we will likely add multiple circles
        # self.canvas.draw()

    @classmethod
    def any_annotation_contains_point(cls, event):
        annotations = App.draggable_annotations
        if annotations:
            for annotation in annotations:
                if annotation and annotation.contains_point(event):
                    return True
        return False

    # calculate annotation offset of pixels in degrees (of where to place the annotation next to the circle)
    @classmethod
    def calculate_offset_pixels(cls):
        # Get current extent of the map in degrees and pixels
        extent = App.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = App.ax.get_window_extent()
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
    # calculate radius of pixels in degrees
    def calculate_radius_pixels(cls):
        # Get current extent of the map in degrees and pixels
        extent = App.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = App.ax.get_window_extent()
        lon_pixels = window_extent.width
        lat_pixels = window_extent.height

        # Calculate degrees per pixel in both x and y directions
        lon_deg_per_pixel = lon_diff / lon_pixels
        lat_deg_per_pixel = lat_diff / lat_pixels

        # Convert pixels to degrees
        radius_degrees = max(lon_deg_per_pixel, lat_deg_per_pixel) * DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS

        return radius_degrees

    @classmethod
    def changed_extent(cls):
        cls.circle_handles = None
        App.draggable_annotations = None
        cls.rtree_p = index.Property()
        cls.rtree_idx = index.Index(properties=cls.rtree_p)
        App.annotated_circle_objects = {}
        cls.counter = 0

    @classmethod
    def clear(cls):
        # if self.annotation_handles:
        if not App.draggable_annotations:
            return
        try:
            for annotation in App.draggable_annotations:
                annotation.remove()
        except:
            traceback.print_exc()
            pass
        App.draggable_annotations = None
        if cls.circle_handles:
            try:
                for internal_id, point_index_circles in cls.circle_handles.items():
                    for point_index, circle_object in point_index_circles.items():
                        #print(internal_id, point_index, circle_object)
                        circle_object.remove()
            except:
                traceback.print_exc()
                pass
            cls.circle_handles = None
        if cls.rtree_p:
            cls.rtree_p = index.Property()
        if cls.rtree_idx:
            cls.rtree_idx = index.Index(properties=cls.rtree_p)

        App.redraw_fig_canvas(stale_bg=True)

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

    @classmethod
    def get_circle_handles(cls):
        if cls.circle_handles:
            return cls.circle_handles
        return None

    @classmethod
    def has_overlap(cls, internal_id=None, point_index=None):
        if not App.annotated_circle_objects:
            return False

        if internal_id in App.annotated_circle_objects and point_index in App.annotated_circle_objects[internal_id]:
            return True

        return False

    @classmethod
    def removeCircle(cls, circle):
        if circle.internal_id in cls.circle_handles:
            if circle.point_index in cls.circle_handles[circle.internal_id]:
                del cls.circle_handles[circle.internal_id][circle.point_index]

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

        self.body_frame = ttk.Frame(self, style='CanvasFrame.TFrame')
        self.body_frame.pack(padx=5, pady=10, fill="both", expand=True)

        self.button_frame = ttk.Frame(self, style='CanvasFrame.TFrame')
        self.button_frame.pack(padx=5, pady=10)

        self.body(self.body_frame, )
        self.create_buttonbox(self.button_frame)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.grab_set()
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.wait_visibility()
        self.center_window()
        self.wait_window(self)

    def apply(self):
        self.result = {
            'annotation_label_options': [option for option, var in self.annotation_label_checkboxes.items() if
                                         var.get()],
            'settings': {key: var.get() for key, var in self.settings.items()}
        }

    def body(self, master):
        #frame = ttk.Frame(master)
        #frame.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(master, style="TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Create a frame for each tab
        map_settings_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        map_settings_frame.grid()
        self.notebook.add(map_settings_frame, text="Map")

        # Create a frame for each tab
        overlay_settings_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        overlay_settings_frame.grid()
        self.notebook.add(overlay_settings_frame, text="Overlay")

        annotation_colors_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        annotation_colors_frame.grid()
        self.notebook.add(annotation_colors_frame, text="Annotation Colors")

        extrema_annotations_frame = ttk.Frame(self.notebook, style='CanvasFrame.TFrame')
        extrema_annotations_frame.grid()
        self.notebook.add(extrema_annotations_frame, text="Annotation Labels")

        r = 0
        ttk.Label(map_settings_frame, text="Circle patch radius (pixels):", style='TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(map_settings_frame, textvariable=self.settings['DEFAULT_CIRCLE_PATCH_RADIUS_PIXELS'], style='TEntry', width=5).grid(row=r, column=1)
        r += 1
        ttk.Label(map_settings_frame, text="Zoom in step factor:", style='TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(map_settings_frame, textvariable=self.settings['ZOOM_IN_STEP_FACTOR'], style='TEntry', width=5).grid(row=r, column=1)
        r += 1
        ttk.Label(map_settings_frame, text="Min grid line spacing (inches):", style='TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(map_settings_frame, textvariable=self.settings['MIN_GRID_LINE_SPACING_INCHES'], style='TEntry', width=5).grid(row=r, column=1)
        r += 1

        for i in range(2):
            map_settings_frame.grid_columnconfigure(i, pad=10)
        for i in range(r):
            map_settings_frame.grid_rowconfigure(i, pad=5)

        overlay_var_text = {
            'RVOR_CYCLONIC_CONTOURS': 'RVOR Cyclonic Contours',
            'RVOR_CYCLONIC_LABELS': 'RVOR Cyclonic Labels',
            'RVOR_ANTICYCLONIC_CONTOURS': 'RVOR Anti-Cyclonic Contours',
            'RVOR_ANTICYCLONIC_LABELS': 'RVOR Anti-cyclonic Labels',
        }

        r = 0
        for varname, option in overlay_var_text.items():
            var = self.settings[varname]
            cb = ttk.Checkbutton(overlay_settings_frame, text=option, variable=var, style='TCheckbutton')
            cb.grid(row=r, column=0, sticky='w')
            r += 1

        ttk.Label(overlay_settings_frame, text="Min. contour pixels X:", style='TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(overlay_settings_frame, textvariable=self.settings['MINIMUM_CONTOUR_PX_X'], style='TEntry', width=5).grid(row=r, column=1)
        r += 1

        ttk.Label(overlay_settings_frame, text="Min. contour pixels Y:", style='TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(overlay_settings_frame, textvariable=self.settings['MINIMUM_CONTOUR_PX_Y'], style='TEntry', width=5).grid(row=r, column=1)
        r += 1

        for i in range(2):
            overlay_settings_frame.grid_columnconfigure(i, pad=10)
        for i in range(r):
            overlay_settings_frame.grid_rowconfigure(i, pad=5)

        r = 0
        ttk.Label(annotation_colors_frame, text="(Circle hover) Marker color:", style='TLabel').grid(row=r, column=0, sticky='w')
        self.default_annotate_marker_color_label = tk.Label(annotation_colors_frame, text="", width=10,
                                                            bg=self.settings['DEFAULT_ANNOTATE_MARKER_COLOR'].get())
        self.default_annotate_marker_color_label.grid(row=r, column=1)
        self.default_annotate_marker_color_label.bind("<Button-1>",
                                                      lambda event: self.choose_color('DEFAULT_ANNOTATE_MARKER_COLOR'))
        r += 1

        ttk.Label(annotation_colors_frame, text="Default annotation text color:", style='TLabel').grid(row=r, column=0, sticky='w')
        self.default_annotate_text_color_label = tk.Label(annotation_colors_frame, text="", width=10,
                                                          bg=self.settings['DEFAULT_ANNOTATE_TEXT_COLOR'].get())
        self.default_annotate_text_color_label.grid(row=r, column=1)
        self.default_annotate_text_color_label.bind("<Button-1>",
                                                    lambda event: self.choose_color('DEFAULT_ANNOTATE_TEXT_COLOR'))
        r += 1

        ttk.Label(annotation_colors_frame, text="TC start text color:", style='TLabel').grid(row=r, column=0, sticky='w')
        self.annotate_dt_start_color_label = tk.Label(annotation_colors_frame, text="", width=10,
                                                      bg=self.settings['ANNOTATE_DT_START_COLOR'].get())
        self.annotate_dt_start_color_label.grid(row=r, column=1)
        self.annotate_dt_start_color_label.bind("<Button-1>",
                                                lambda event: self.choose_color('ANNOTATE_DT_START_COLOR'))
        r += 1

        ttk.Label(annotation_colors_frame, text="Earliest named text color:", style='TLabel').grid(row=r, column=0, sticky='w')
        self.annotate_earliest_named_color_label = tk.Label(annotation_colors_frame, text="", width=10,
                                                            bg=self.settings['ANNOTATE_EARLIEST_NAMED_COLOR'].get())
        self.annotate_earliest_named_color_label.grid(row=r, column=1)
        self.annotate_earliest_named_color_label.bind("<Button-1>",
                                                      lambda event: self.choose_color('ANNOTATE_EARLIEST_NAMED_COLOR'))
        r += 1

        ttk.Label(annotation_colors_frame, text="Vmax text color:", style='TLabel').grid(row=r, column=0, sticky='w')
        self.annotate_vmax_color_label = tk.Label(annotation_colors_frame, text="", width=10,
                                                  bg=self.settings['ANNOTATE_VMAX_COLOR'].get())
        self.annotate_vmax_color_label.grid(row=r, column=1)
        self.annotate_vmax_color_label.bind("<Button-1>", lambda event: self.choose_color('ANNOTATE_VMAX_COLOR'))
        r += 1

        for i in range(2):
            annotation_colors_frame.grid_columnconfigure(i, pad=10)
        for i in range(r):
            annotation_colors_frame.grid_rowconfigure(i, pad=10)

        self.color_labels = {
            'DEFAULT_ANNOTATE_MARKER_COLOR': self.default_annotate_marker_color_label,
            'DEFAULT_ANNOTATE_TEXT_COLOR': self.default_annotate_text_color_label,
            'ANNOTATE_DT_START_COLOR': self.annotate_dt_start_color_label,
            'ANNOTATE_EARLIEST_NAMED_COLOR': self.annotate_earliest_named_color_label,
            'ANNOTATE_VMAX_COLOR': self.annotate_vmax_color_label,
        }

        r = 0
        # Add your widgets for the "Extrema Annotations" tab here
        self.annotation_label_checkboxes = {}
        for option in self.annotation_label_options:
            var = tk.IntVar()
            cb = ttk.Checkbutton(extrema_annotations_frame, text=option, variable=var, style='TCheckbutton')
            cb.grid(row=r, column=0, sticky='w')
            r += 1
            self.annotation_label_checkboxes[option] = var
            if option in self.selected_annotation_label_options:
                var.set(1)

        for i in range(2):
            extrema_annotations_frame.grid_columnconfigure(i, pad=10)
        for i in range(r):
            extrema_annotations_frame.grid_rowconfigure(i, pad=5)

    def cancel(self, event=None):
        self.parent.focus_set()
        self.destroy()

    def center_window(self):
        self.update_idletasks()

        # Get the dimensions of the dialog
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height() + 20

        # Get the dimensions of the parent window
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate the position to center the dialog on the parent window
        x = self.parent.winfo_x() + (parent_width // 2) - (dialog_width // 2)
        y = self.parent.winfo_y() + (parent_height // 2) - (dialog_height // 2)

        # Set the position of the dialog
        self.geometry(f'{dialog_width}x{dialog_height}+{x}+{y}')

    def choose_color(self, setting_name):
        color_obj = self.settings[setting_name]
        if color_obj:
            color = colorchooser.askcolor(color=color_obj.get())[1]
            if color:
                color_obj.set(color)
                self.color_labels[setting_name].config(bg=color)

    def create_buttonbox(self, master):
        self.buttonbox = ttk.Frame(master, style='CanvasFrame.TFrame')
        w = ttk.Button(self.buttonbox, text="Restore All Defaults (requires restart)", command=self.restore_defaults, style='TButton')
        w.pack(side=tk.LEFT, padx=5, pady=5)
        ok_w = ttk.Button(self.buttonbox, text="OK", command=self.ok, style='TButton')
        ok_w.pack(side=tk.LEFT, padx=5, pady=5)
        w = ttk.Button(self.buttonbox, text="Cancel", command=self.cancel, style='TButton')
        w.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        # Set focus on the OK button
        ok_w.focus_set()

        self.buttonbox.pack()
        return self.buttonbox

    def ok(self, event=None):
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    def restore_defaults(self):
        import os
        os.remove('settings_tcviewer.json')
        self.cancel()

# The draggable, annotated textbox used by AnnotatedCircles
class DraggableAnnotation:
    def __init__(self, annotation, original_point, bbox_props):
        self.blocking = False
        self.circle_annotation = None
        self.original_annotation = annotation
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
        self.dragging_annotation = App.ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            xycoords=xycoords,
            textcoords=textcoords,
            bbox=self.bbox_props,  # Use the bbox_props passed to the constructor
            **props
        )
        self.dragging_annotation.set_visible(False)

        # self.bring_to_front()

        # Connect to the event handlers
        self.cid_press = App.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = App.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = App.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def block_for_dragging(self):
        # only start blocking when we have a line
        block_for_dragging = EventManager.block_events('dragging_annotation')
        if not block_for_dragging:
            raise ValueError("Failed to block events for DraggingAnnotation")
        self.blocking = True
        return True

    def bring_to_front(self):
        if App.draggable_annotations:
            top_zorder = max(ann.zorder for ann in App.draggable_annotations) + 1
            self.zorder = top_zorder
            self.original_annotation.set_zorder(top_zorder)
            self.dragging_annotation.set_zorder(top_zorder)
        else:
            # first
            top_zorder = 10
            self.zorder = top_zorder
            self.original_annotation.set_zorder(top_zorder)
            self.dragging_annotation.set_zorder(top_zorder)

    def on_press(self, event):
        if not self.visible:
            return

        if not App.draggable_annotations:
            return

        if self != self.get_topmost_annotation(event):
            return

        contains, attrd = self.original_annotation.contains(event)
        if not contains:
            return

        xy_orig = self.original_annotation.xy
        self.press = (xy_orig, self.original_annotation.get_position(), event.xdata, event.ydata)
        self.original_annotation.set_visible(False)
        if self.line:
            self.line.set_visible(False)
        App.redraw_fig_canvas()
        self.background = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)
        self.dragging_annotation.set_position(self.original_annotation.get_position())
        self.dragging_annotation.set_visible(True)
        if self.line:
            self.line.set_visible(True)
        App.redraw_fig_canvas()
        self.set_dragging(True)

    def calculate_radius_pixels(self):
        # Get current extent of the map in degrees and pixels
        extent = App.ax.get_extent()
        lon_diff = extent[1] - extent[0]
        lat_diff = extent[3] - extent[2]

        # Get window extent in pixels
        window_extent = App.ax.get_window_extent()
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

    def get_topmost_annotation(self, event):
        annotations = App.draggable_annotations
        # need a separate is dragging flag as bbox is unstable when we are blitting
        # (as it can become invisible during the setup for blitting (in an async event with the other handlers)
        # the bbox will result erroneously in a 1 pixel box causing the wrong annotation to drag
        if not annotations:
            return None
        valid_annotations = [ann for ann in annotations if ann and (ann.is_dragging() or ann.contains_point(event))]
        if not valid_annotations:
            return None
        return max(valid_annotations, key=lambda ann: ann.zorder)

    def has_focus(self, event):
        if not self.visible:
            return False

        if self != self.get_topmost_annotation(event):
            return False

        contains, attrd = self.original_annotation.contains(event)
        if not contains:
            return False

        return True

    def is_dragging(self):
        return self.dragging

    def on_motion(self, event):
        if self.press is None:
            return

        (x0, y0), (x0_cur, y0_cur), xpress, ypress = self.press

        xlim = App.ax.get_xlim()
        ylim = App.ax.get_ylim()

        inbound = False
        if event.inaxes == App.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

        if not (inbound) or not (xpress) or not (ypress):
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
        self.line = App.ax.plot(
            [edge_x, new_x],
            [edge_y, new_y],
            linestyle='--',
            color=DEFAULT_ANNOTATE_TEXT_COLOR,
            transform=ccrs.PlateCarree(),
            )[0]

        App.ax.figure.canvas.restore_region(self.background)
        App.ax.draw_artist(self.dragging_annotation)
        App.ax.draw_artist(self.line)
        App.ax.figure.canvas.blit(App.ax.bbox)

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
        App.redraw_fig_canvas(stale_bg=True)

    def remove(self):
        self.unblock_for_dragging()
        App.ax.figure.canvas.mpl_disconnect(self.cid_press)
        App.ax.figure.canvas.mpl_disconnect(self.cid_release)
        App.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        if self.line:
            self.line.remove()
        self.original_annotation.remove()
        self.dragging_annotation.remove()

    def set_circle_annotation(self, circle_annotation):
        self.circle_annotation = circle_annotation

    def set_dragging(self, dragging):
        if dragging and not self.dragging:
            self.block_for_dragging()
        elif self.dragging and not dragging:
            self.unblock_for_dragging()
        self.dragging = dragging

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

    def unblock_for_dragging(self):
        if self.blocking:
            EventManager.unblock_events()
            self.blocking = False

# Manage mutex blocking of mouse motion events
# this disables status label updates for dragging/selecting operations that are supposed to be fast
# normally this is during blittering operations,
# such as, like dragging annotations, making loop selections, measurements, etc.
class EventManager:
    blocked = False
    blocking_purpose = None

    @classmethod
    def block_events(cls, purpose):
        if cls.blocked:
            return False
        cls.blocked = True
        cls.blocking_purpose = purpose
        return True

    @classmethod
    def get_blocking_purpose(cls):
        return cls.blocking_purpose

    @classmethod
    def reset(cls):
        cls.blocked = False
        cls.blocking_purpose = False

    @classmethod
    def unblock_events(cls):
        if not cls.blocked:
            raise ValueError("Events are not blocked")
        cls.blocked = False
        cls.blocking_purpose = None

# Blitting measure tool (single instance)
class MeasureTool:
    def __init__(self):
        self.measure_mode = False
        self.start_point = None
        self.end_point = None
        self.line = None
        self.distance_text = None
        self.distance = 0
        # Save background
        self.bg = None
        self.blocking = False

    def block_for_measure(self):
        if self.measure_mode:
            if not EventManager.get_blocking_purpose():
                blocking_for_measure = EventManager.block_events('measure')
                if not blocking_for_measure:
                    raise ValueError("Failed to block events for MeasureTool")
            self.blocking = True
            return True
        return False

    @staticmethod
    def calculate_distance(start_point, end_point):
        geod = cgeo.Geodesic()
        line = LineString([start_point, end_point])
        total_distance = geod.geometry_length(line)
        nautical_miles = total_distance / 1852.0
        return nautical_miles

    def changed_extent(self):
        if self.line or self.distance_text:
            self.remove_artists()
            App.redraw_fig_canvas(stale_bg=True)
            # update background region
            self.bg = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)
            self.create_artists()

    def create_artists(self):
        self.create_line_artist()
        self.create_distance_text_artist(self.distance)

    def create_line_artist(self):
        if App.ax and self.start_point and self.end_point:
            self.line = App.ax.plot([self.start_point[0], self.end_point[0]],
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
            extent = App.ax.get_extent()
            lon_diff = extent[1] - extent[0]
            lat_diff = extent[3] - extent[2]

            lon_deg_per_pixel = lon_diff / App.ax.get_window_extent().width
            lat_deg_per_pixel = lat_diff / App.ax.get_window_extent().height

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

            self.distance_text = App.ax.text(mid_point[0] + offset_x_deg, mid_point[1] + offset_y_deg,
                                              f"{distance:.2f} NM", color='white', fontsize=12,
                                              ha='center', va='center', rotation=angle,
                                              bbox=dict(facecolor='black', alpha=0.5))

    def in_measure_mode(self):
        return self.measure_mode

    def on_click(self, event):
        if self.measure_mode:
            if event.button == 1:
                self.start_point = (event.xdata, event.ydata)
                self.end_point = (event.xdata, event.ydata)
                return True
            elif event.button == 3:
                self.reset_measurement()
        return False

    def on_key_press(self, event):
        if event.key == 'shift':
            self.measure_mode = True
            self.block_for_measure()
            if self.line:
                # updating existing line
                # hide to get new blit, then do on motion
                self.changed_extent()
                inbound = False
                xlim = App.ax.get_xlim()
                ylim = App.ax.get_ylim()
                if event.inaxes == App.ax:
                    # Check if mouse coordinates are within figure bounds
                    try:
                        inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
                    except:
                        inbound = False
                self.on_motion(event, inbound)

    def on_key_release(self, event):
        if event.key == 'shift':
            self.measure_mode = False
            self.unblock_for_measure()
            if self.line:
                # store blit of line
                App.redraw_fig_canvas(stale_bg=True)

    def on_motion(self, event, inbound):
        if not self.measure_mode:
            return False

        if not self.start_point:
            return False

        if not inbound:
            return False

        self.end_point = (event.xdata, event.ydata)

        try:
            if not self.line:
                self.bg = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)
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
            App.ax.draw_artist(self.line)
        if self.distance_text:
            App.ax.draw_artist(self.distance_text)
        App.ax.figure.canvas.blit(App.ax.bbox)
        return True

    def remove_artists(self):
        self.remove_line_artist()
        self.remove_distance_text_artist()

    def remove_distance_text_artist(self):
        if self.distance_text:
            try:
                self.distance_text.remove()
            except:
                pass
            self.distance_text = None

    def remove_line_artist(self):
        if self.line:
            try:
                self.line.remove()
            except:
                pass
            self.line = None

    def reset_measurement(self):
        self.remove_artists()
        App.ax.set_yscale('linear')
        self.start_point = None
        self.end_point = None
        self.line = None
        self.distance = 0
        App.redraw_fig_canvas(stale_bg=True)

    def restore_region(self):
        if self.bg:
            App.ax.figure.canvas.restore_region(self.bg)

    def unblock_for_measure(self):
        if self.blocking:
            self.blocking = False
            if EventManager.get_blocking_purpose():
                EventManager.unblock_events()

# Plot tchp, depth of the 26th isobar data
class NetCDFPlotter:
    def __init__(self, filepath, data_type):
        self.filepath = filepath
        self.tchp = None
        self.d26 = None
        self.ohc = None
        self.iso26C = None
        self.sst = None
        self.lat = None
        self.lon = None
        self.first_datetime = None
        self.data_type = data_type
        self.globe = None

    def load_data(self):
        # Open the NetCDF file and load the netcdf datasets
        with nc.Dataset(self.filepath, 'r') as ds:
            if self.data_type in ['tchp', 'd26']:
                # this is defunct until we figure out what the problem with the dataset is (d26 is currently == tchp)
                self.tchp = ds.variables['Tropical_Cyclone_Heat_Potential'][:][0]
                self.d26 = ds.variables['D26'][:][0]
                self.lat = ds.variables['latitude'][:]
                self.lon = ds.variables['longitude'][:]
            elif self.data_type in ['ohc', 'iso26C']:
                self.ohc = ds.variables['ohc'][:][0]
                self.iso26C = ds.variables['iso26C'][:][0]
                self.lat = ds.variables['latitude'][:]
                self.lon = ds.variables['longitude'][:]
                if 'crs' in ds.variables:
                    # Extract the CRS parameters
                    semimajor_axis = ds.variables['crs'].semi_major_axis
                    inverse_flattening = ds.variables['crs'].inverse_flattening
                    longitude_of_prime_meridian = ds.variables['crs'].longitude_of_prime_meridian
                    self.globe = ccrs.Globe(semimajor_axis=semimajor_axis,
                                       semiminor_axis=semimajor_axis / (1 + 1 / inverse_flattening),
                                       ellipse=None)
            elif self.data_type == 'sst':
                # Get the sst data for the day and convert it from K to C
                self.sst = ds.variables['analysed_sst'][:][0]  - 273.15
                self.lat = ds.variables['lat'][:]
                self.lon = ds.variables['lon'][:]

            # Assume the time variable is named 'time' and is in hours since a reference date
            time_var = ds.variables['time'][:]
            time_units = ds.variables['time'].units

            # Convert the time to a pandas datetime
            time_dates = nc.num2date(time_var, time_units)

            # Convert cftime to datetime
            datetime_dates = [datetime(*date.timetuple()[:6]) for date in time_dates]
            if len(datetime_dates) != 0:
                self.first_datetime = datetime_dates[0]

    def clip_data(self, data, data_min=None, data_max=None):
        # Clip the data based on provided min and max
        if data_min is not None:
            data = np.maximum(data, data_min)
        if data_max is not None:
            data = np.minimum(data, data_max)
        return data

    def bin_data(self, data, bins=None):
        # If binning is enabled, simplify the data based on the bins
        if bins:
            binned_data = np.zeros_like(data)
            for i, (bin_min, bin_max) in enumerate(bins):
                binned_data[(data >= bin_min) & (data < bin_max)] = (bin_min + bin_max) / 2
            return binned_data
        return data

    # Create the colormap function
    def create_colormap(self, dataset, data_min, data_max, bins, scale_min, scale_max):
        if dataset == 'sst':
            colors = ['white', 'cyan', 'blue', 'green', 'yellow', 'orange', 'red', 'purple', 'pink']
        else:
            colors = ['black', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'purple', 'pink']

        if bins is not None:
            # Number of bins, and hence color segments, should match the intervals in bins
            num_colors = len(bins)

            # Create a continuous colormap from the defined colors
            continuous_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

            # Extract the lower and upper bounds from the bin tuples
            bin_edges = np.array([b[0] for b in bins] + [bins[-1][1]])  # Include the upper bound of the last bin

            # Normalize the bin edges to [0, 1] based on data_min and data_max
            norm_bin_edges = (bin_edges - scale_min) / (scale_max - scale_min)

            # Sample colors based on the normalized bin edges
            sampled_colors = [continuous_cmap(norm) for norm in norm_bin_edges]

            # Create a ListedColormap using the sampled colors
            cmap = mcolors.ListedColormap(sampled_colors[:num_colors])

            norm = mcolors.BoundaryNorm(bin_edges, cmap.N)


        else:
            # Use continuous colormap
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
            norm = mcolors.Normalize(vmin=scale_min, vmax=scale_max)

        return cmap, norm

    def plot_data(self, ax=None, dataset='tchp', data_min=None, data_max=None, bins=None, opacity=1.0, scale_min=0,
                  scale_max=300):
        # Select the dataset (tchp or d26)
        if dataset == 'tchp':
            data = self.tchp
            label = 'TCHP\n(kJ cm^-2)'
        elif dataset == 'd26':
            data = self.d26
            label = 'D26\n(m)'
        elif dataset == 'ohc':
            data = self.ohc
            label = 'OHC\n(kJ cm^-2)'
        elif dataset == 'iso26C':
            data = self.iso26C
            label = 'ISO26C\n(m)'
        elif dataset == 'sst':
            data = self.sst
            label = 'SST\n(deg C)'
        else:
            # invalid config
            return

        if self.first_datetime is not None:
            formatted_date = self.first_datetime.strftime('%Y-%m-%d')
            label += f'\n{formatted_date}'

        data = self.clip_data(data, data_min, data_max)

        data = self.bin_data(data, bins)

        data_range_min = np.nanmin(data)
        data_range_max = np.nanmax(data)

        vmin = data_min if data_min is not None else data_range_min
        vmax = data_max if data_max is not None else data_range_max

        # Create the colormap
        cmap, norm = self.create_colormap(dataset, data_range_min, data_range_max, bins, scale_min, scale_max)
        if self.globe is not None:
            mesh = ax.pcolormesh(self.lon, self.lat, data, cmap=cmap, norm=norm, alpha=opacity,
                                 transform=ccrs.PlateCarree(globe=self.globe))
        else:
            mesh = ax.pcolormesh(self.lon, self.lat, data, cmap=cmap, norm=norm, alpha=opacity,
                                 transform=ccrs.PlateCarree())

        # Create the color bar
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, fraction = 0.1)
        cbar.ax.set_position([0.45, 0.9, 0.45, 0.1])  # [left, bottom, width, height] of the color bar

        # Set tick label colors to white
        cbar.ax.tick_params(labelcolor='white')  # Set color of color bar ticks

        if bins is not None:
            bin_edges = np.array([b[0] for b in bins] + [bins[-1][1]])
            cbar.set_ticks(bin_edges)

        cbar.set_label(label, color='white')

        # Set bbox for the color bar label using the correct attribute
        cbar.ax.xaxis.label.set_bbox(dict(facecolor='dimgray', edgecolor='white', alpha=1))  # Dark grey background
        cbar.ax.xaxis.set_label_coords(1.09, 0.75)  # Adjust this to position the label below the color bar

        cbar.ax.tick_params(labelcolor='white')  # Set color of color bar ticks

        for label in cbar.ax.get_xticklabels():
            label.set_color('white')  # Set label color to white
            label.set_bbox(dict(facecolor='dimgray', edgecolor='white', alpha=1.0))  # Dark grey background

        # Set the color of the color bar ticks to white
        for tick in cbar.ax.get_yticklabels():
            tick.set_color('white')


# Helper class for (partially) interpolating and filtering points on tc candidates by time and fields
class PartialInterpolationTrackFilter:
    @staticmethod
    def filter_by_field(time_filtered_candidates, field_data, by_all_all, by_all_any, by_any_all,
                                   by_any_any):
        """
        Filter candidates based on the specified criteria.

        Parameters:
        - time_filtered_candidates: List of tuples containing candidate tracks.
        - field_data: Dictionary containing field key ranges.
        - by_all_all: Boolean indicating if all points must satisfy all fields.
        - by_all_any: Boolean indicating if all points must satisfy at least one field.
        - by_any_all: Boolean indicating if any point must satisfy all fields.
        - by_any_any: Boolean indicating if any pair of points must satisfy any fields.

        Returns:
        - List of filtered candidates.
        """
        field_filtered_candidates = []

        for internal_id, points in time_filtered_candidates:
            # Skip candidates if the track has no points
            if not points:
                continue

            if by_all_all and PartialInterpolationTrackFilter.satisfies_all_fields(points, field_data):
                field_filtered_candidates.append((internal_id, points))
            elif by_all_any and PartialInterpolationTrackFilter.satisfies_any_fields(points, field_data):
                field_filtered_candidates.append((internal_id, points))
            elif by_any_all and PartialInterpolationTrackFilter.satisfies_any_all(points, field_data):
                field_filtered_candidates.append((internal_id, points))
            elif by_any_any and PartialInterpolationTrackFilter.satisfies_any_any(points, field_data):
                field_filtered_candidates.append((internal_id, points))

        return field_filtered_candidates

    @staticmethod
    def interpolate_point(pt1, pt2, target_time, field_keys):
        """Interpolate values between two points based on target_time."""
        interpolated = {}
        time_diff = (pt2['valid_time'] - pt1['valid_time']).total_seconds()
        factor = (target_time - pt1['valid_time']).total_seconds() / time_diff

        # Interpolate each field based on the factor
        for key in field_keys:
            if key in pt1 and key in pt2:
                interpolated[key] = pt1[key] + (pt2[key] - pt1[key]) * factor

        # Always include valid_time and init_time
        interpolated['valid_time'] = target_time
        interpolated['init_time'] = pt1['init_time']  # Assuming we use pt1's init_time
        return interpolated

    # Partially (only include points in dt range) interpolate points for tc candidates (filter by time)
    # This is run first, then filtered by fields
    @staticmethod
    def filter_by_time(filtered_candidates, start_dt, end_dt, field_keys, all_must_overlap):
        interpolated_tracks = []

        for internal_id, tc in filtered_candidates:
            # If start or end is outside the track range, skip
            if start_dt > tc[-1]['valid_time'] or end_dt < tc[0]['valid_time']:
                continue

            if all_must_overlap:
                if tc[0]['valid_time'] >= start_dt and tc[-1]['valid_time'] <= end_dt:
                    interpolated_tracks.append((internal_id, tc))
                # either track is absolutely in range or not
                continue

            new_track = []
            i = 0

            # Find where the start_dt fits
            while i < len(tc) and tc[i]['valid_time'] < start_dt:
                i += 1

            # Handle the case where start_dt is before the first point
            if i == 0:
                new_track.append({k: v for k, v in tc[0].items() if k in field_keys or k == 'valid_time'})
                i = 1
            elif i < len(tc):
                # Interpolate start_dt
                interpolated_start = PartialInterpolationTrackFilter.interpolate_point(tc[i - 1], tc[i], start_dt, field_keys)
                new_track.append(interpolated_start)

            # Add subsequent points until end_dt
            while i < len(tc) and tc[i]['valid_time'] <= end_dt:
                new_track.append({k: v for k, v in tc[i].items() if k in field_keys or k == 'valid_time'})
                i += 1

            # Handle the case where end_dt is between two points
            if i < len(tc) and tc[i - 1]['valid_time'] < end_dt < tc[i]['valid_time']:
                interpolated_end = PartialInterpolationTrackFilter.interpolate_point(tc[i - 1], tc[i], end_dt, field_keys)
                new_track.append(interpolated_end)

            # Add track to final output if we added points
            if new_track:
                interpolated_tracks.append((internal_id, new_track))

        return interpolated_tracks

    @staticmethod
    def ranges_overlap(range1, range2):
        """Check if two ranges overlap."""
        return max(range1[0], range2[0]) <= min(range1[1], range2[1])

    @staticmethod
    def satisfies_all_fields(points, field_data):
        """Check if all points satisfy all field conditions."""
        for pt in points:
            for key, value in field_data.items():
                if not (value['min'] <= pt[key] <= value['max']):
                    return False
        return True

    @staticmethod
    def satisfies_any_fields(points, field_data):
        """Check if at least one field is satisfied for all points."""
        for pt in points:
            if any(value['min'] <= pt[key] <= value['max'] for key, value in field_data.items()):
                continue
            return False
        return True

    @staticmethod
    def satisfies_any_all(points, field_data):
        """Check if any point satisfies all field conditions."""
        for pt in points:
            if all(value['min'] <= pt[key] <= value['max'] for key, value in field_data.items()):
                return True
        return False

    @staticmethod
    def satisfies_all_any(points, field_data):
        """Check if all points have at least one overlapping field."""
        for key in field_data.keys():
            if all(not (value['min'] <= pt[key] <= value['max']) for pt in points):
                return False
        return True

    @staticmethod
    def satisfies_any_any(points, field_data):
        """Check if any pair of points satisfies any field conditions."""
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for key in field_data.keys():
                    if PartialInterpolationTrackFilter.ranges_overlap(
                            (points[i][key], points[j][key]),
                            (field_data[key]['min'], field_data[key]['max'])
                    ):
                        return True
        return False

# Blitting box for map (for zooms / selections)
class SelectionBox:
    def __init__(self):
        block_for_zoom = EventManager.block_events('zoom')
        if not block_for_zoom:
            raise ValueError("Failed to block events for SelectionBox")
        else:
            self.box = None
            self.lon1, self.lat1, self.lon2, self.lat2 = None, None, None, None
            self.has_latlons = False
            self.bg = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)  # Save background

    def destroy(self):
        self.remove()
        self.restore_region()  # Restore regions without drawing
        App.ax.figure.canvas.blit(App.ax.bbox)
        EventManager.unblock_events()

    def draw_box(self):
        if self.box:
            try:
                self.box.remove()
            except:
                pass
        self.box = App.ax.plot(
            [self.lon1, self.lon2, self.lon2, self.lon1, self.lon1],
            [self.lat1, self.lat1, self.lat2, self.lat2, self.lat1],
            color='yellow', linestyle='--', transform=ccrs.PlateCarree())
        for artist in self.box:
            App.ax.draw_artist(artist)
        App.ax.figure.canvas.blit(App.ax.bbox)

    def is_2d(self):
        if self.lon1 and self.lon2 and self.lat1 and self.lat2:
            return not (np.isclose(self.lon1, self.lon2) or np.isclose(self.lat1, self.lat2))
        else:
            return False

    def restore_region(self):
        App.ax.figure.canvas.restore_region(self.bg)

    def remove(self):
        if self.box:
            try:
                for artist in self.box:
                    artist.remove()
            except:
                pass

    def update_box(self, lon1, lat1, lon2, lat2):
        self.remove()
        self.restore_region()  # Restore first
        self.lon1, self.lat1, self.lon2, self.lat2 = lon1, lat1, lon2, lat2
        if lon1 and lat1 and lon2 and lat2:
            self.has_latlons = True
            self.draw_box()  # Draw the box
        App.ax.figure.canvas.blit(App.ax.bbox)

# Blitting selection loop to select storm tracks
class SelectionLoops:
    last_ax = None
    selection_loop_objects = []
    last_loop = None
    selecting = False
    blocking = False
    visible = True

   # using shapely polygon as Polygon, and matplotlib polygon as MPLPolygon
    class SelectionLoop:
        def __init__(self, event, polys=None):
            self.last_ax = App.ax
            self.verts = []
            self.polygons = []
            self.preview = None
            self.preview_artists = []
            self.closed_artists = []
            self.background = None
            self.closed = False
            self.alpha = 0.35
            self.bg = None
            if event is not None:
                self.verts.append((event.xdata, event.ydata))
            elif polys is not None:
                for poly in polys:
                    self.verts.extend(poly.exterior.coords)

                self.set_polygons(polys)

        def changed_extent(self):
            # preserve the polygons on the map across all map & data changes
            if self.last_ax != App.ax:
                self.last_ax = App.ax
                if self.closed_artists:
                    self.remove()
                    App.redraw_fig_canvas()
                    # update background region
                    self.bg = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)
                    self.update_closed_artists()

        # find loops and return a list of (Shapely) Polygons
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

        def get_polygons(self):
            return self.polygons

        def on_motion(self, event):
            self.verts.append((event.xdata, event.ydata))
            self.update_preview()
            return True

        def on_release(self, polys=None):
            if polys is None:
                polys = self.close_loop()

            self.set_polygons(polys)
            self.update_closed_artists()
            self.closed = True
            # clean up memory
            self.background = None

        def remove(self):
            self.remove_preview_artists()
            self.remove_closed_artists()

        def remove_closed_artists(self):
            if self.closed_artists:
                for artist in self.closed_artists:
                    artist.remove()
                self.closed_artists = []
                self.closed = []

        def remove_preview_artists(self):
            if self.preview_artists:
                for artist in self.preview_artists:
                    artist.remove()
                self.preview_artists = []
                self.preview = []

        def set_polygons(self, polygons):
            self.polygons = polygons

        def set_visible(self, visible):
            for patch in self.closed_artists:
                patch.set_visible(visible)

        def update_closed_artists(self):
            for poly in self.polygons:
                patch = MPLPolygon(poly.exterior.coords, alpha=self.alpha)
                self.closed_artists.append(App.ax.add_patch(patch))
            self.remove_preview_artists()
            #App.ax.figure.canvas.draw_idle()
            App.redraw_fig_canvas(stale_bg=True)

        def update_preview(self):
            if self.background is None:
                self.background = App.ax.figure.canvas.copy_from_bbox(App.ax.bbox)
            if self.preview_artists is not None:
                App.ax.figure.canvas.restore_region(self.background)

            self.remove_preview_artists()
            self.preview = self.close_loop()
            if self.preview:
                for poly in self.preview:
                    patch = MPLPolygon(poly.exterior.coords, alpha=self.alpha)
                    artist = App.ax.add_patch(patch)
                    self.preview_artists.append(artist)
                    App.ax.draw_artist(artist)
                App.ax.figure.canvas.blit(App.ax.bbox)
                return
            else:
                App.ax.figure.canvas.restore_region(self.background)
                return

    @classmethod
    def add_poly(cls, geos):
        cls.last_loop = cls.SelectionLoop(None, polys=list(geos))
        cls.selection_loop_objects.append(cls.last_loop)
        cls.last_loop.on_release(polys=list(geos))

    @classmethod
    def block(cls):
        try:
            EventManager.block_events('selection_loop')
            cls.blocking = True
        except:
            traceback.print_exc()
            cls.blocking = False

    @classmethod
    def changed_extent(cls):
        if cls.last_ax != App.ax:
            cls.visible = True

        cls.last_ax = App.ax
        for selection_loop_obj in cls.selection_loop_objects:
            selection_loop_obj.changed_extent()

    @classmethod
    def clear(cls):
        for selection_loop_obj in cls.selection_loop_objects:
            selection_loop_obj.remove()
        cls.selection_loop_objects = []
        cls.last_loop = None
        cls.selecting = False
        cls.unblock()
        #App.ax.figure.canvas.draw_idle()
        App.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def get_polygons(cls):
        all_polygons = []
        for selection_loop_obj in cls.selection_loop_objects:
            all_polygons.extend(selection_loop_obj.get_polygons())
        return all_polygons

    @classmethod
    def is_empty(cls):
        if cls.selection_loop_objects:
            return False
        else:
            return True

    @classmethod
    def on_click(cls, event):
        if event.button == 1:  # left click
            cls.last_loop = cls.SelectionLoop(event)
            cls.selection_loop_objects.append(cls.last_loop)
            cls.selecting = True
            cls.block()
            return True
        elif event.button == 3: # right click
            if cls.selecting:
                cls.unblock()
            cls.selecting = False
            if cls.selection_loop_objects:
                # remove last item added (an undo type feature; no redo)
                cls.selection_loop_objects[-1].remove()
                del cls.selection_loop_objects[-1]
                if cls.selection_loop_objects:
                    cls.last_loop = cls.selection_loop_objects[-1]
                else:
                    cls.last_loop = None
                #App.ax.figure.canvas.draw_idle()
                App.redraw_fig_canvas(stale_bg=True)
            return True

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
    def toggle_visible(cls):
        if not cls.selecting:
            cls.visible = not (cls.visible)
            for selection_loop_obj in cls.selection_loop_objects:
                selection_loop_obj.set_visible(cls.visible)
            #App.ax.figure.canvas.draw_idle()
            App.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def unblock(cls):
        if cls.blocking:
            try:
                EventManager.unblock_events()
            except:
                traceback.print_exc()
        cls.blocking = False

# A sorted cyclic dict that has the item number enumerated (sorted by value)
# has a get() the next enumerated number and the key of the item in a cycle
# used for cycling over a dict of items (cycling over hover matches)
class SortedCyclicEnumDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_value = lambda x: x[1]  # default sort by value
        self._index = 0

    def _reorder(self):
        sorted_items = sorted(self.items(), key=self._sort_value)
        temp_dict = OrderedDict(sorted_items)
        self.clear()
        for key, value in temp_dict.items():
            super().__setitem__(key, value)
        self._index = 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._reorder()

    def get_first_key(self):
        if not self:
            return None

        return next(iter(self.items()))[0]

    # get previous key without moving backwards (assumes have gotten at least one tuple with next)
    def get_prev_enum_key_tuple(self):
        if not self:
            return None
        sorted_items = list(self.items())
        prev_index = (self._index - 1) % len(self)
        key, value = sorted_items[prev_index]
        return (prev_index + 1, key)

    def next_enum_key_tuple(self):
        if not self:
            return None
        sorted_items = list(self.items())
        idx = self._index % len(self)
        key, value = sorted_items[idx]
        self._index = (self._index + 1) % len(self)
        return (idx + 1, key)

# Helper class to calculate wind field geometries from wind radii (tcgen/ensemble wind radii)
class WindField:
    @staticmethod
    # aggregate the points them by quadrant (pairs of vertices, each belonging to a connecting line)
    def aggregate_points_by_quadrant(quad_poly_point_pairs):
        quad_lines = [ [[ ], [ ], [ ]], [[ ], [ ], [ ]], [[ ], [ ], [ ]], [[ ], [ ], [ ]] ]
        for j, quad_point_triple in enumerate(quad_poly_point_pairs):

            for (step_dt, arc_start_point, arc_end_point) in quad_point_triple:
                quad_lines[j][0].append(step_dt)
                quad_lines[j][1].append(arc_start_point)
                quad_lines[j][2].append(arc_end_point)

        return quad_lines

    @staticmethod
    def aggregate_points_in_pairs(paths):
        quad_poly_point_pairs = [[ ], [ ], [ ], [ ]]
        last_current_point_paths = None
        for step_dt, current_point_paths in paths.items():
            # get the quadrant points that extend outwards forming the vertices for the arc
            last_current_point_paths = current_point_paths
            for k, current_quad_path in enumerate(current_point_paths):
                quad_poly_point_pairs[k].append(
                    [step_dt, current_quad_path[0][1], current_quad_path[0][4]])

        return quad_poly_point_pairs

    # approximate all Bezier curves in the Path
    @staticmethod
    def approximate_bezier_path(path, num_points=100):
        """Approximates Bezier segments in a Path."""
        vertices = []
        i = 0
        while i < len(path.vertices):
            code = path.codes[i]
            if code == Path.CURVE4:  # Handle cubic Bezier curves
                if i + 3 < len(path.vertices):  # Ensure enough points for CURVE4
                    p0 = path.vertices[i - 1]  # Start point
                    p1, p2, p3 = path.vertices[i:i + 3]  # Two control points and the end point
                    for t in np.linspace(0, 1, num_points):
                        point = WindField.cubic_bezier(t, p0, p1, p2, p3)
                        vertices.append(point)
                    i += 3  # Move index past the control points and end point
                else:
                    #print(f"Warning: Not enough points for CURVE4 at index {i}")
                    break
            else:
                vertices.append(path.vertices[i])  # For other path segments, just add the vertex
            i += 1

        return np.array(vertices)

    @staticmethod
    def concat_quads_and_connecting_paths(paths, connecting_paths):
        combined_vertices = []
        combined_codes = []
        concat_paths = []
        for path_dict in [paths, connecting_paths]:
            for dt, quad_paths in path_dict.items():
                for quad_path in quad_paths:
                    vertices, codes = quad_path
                    combined_vertices.extend(vertices)
                    combined_codes.extend(codes)
                    path = Path(vertices, codes)
                    concat_paths.append(path)

        return concat_paths

    def construct_connecting_vertices(quad_lines):
        # construct the connecting polys for each interpolated step and quadrant using the connecting lines
        connecting_quad_poly_vertices = {}
        # changed to NOT do a large path around the whole track, rather do small connecting polys for each interpolation
        # this allows for (fine tuned) interpolation statistics (however much that adds given we are only using 6 hour wind radii)
        for j, quad_line in enumerate(quad_lines):
            for k in range(len(quad_line[0]) - 1):
                end_idx = k
                quad_dt_end = quad_line[0][end_idx]
                quad_end_p1 = quad_line[1][end_idx]
                quad_end_p2 = quad_line[2][end_idx]

                start_idx = k+1
                quad_dt_start = quad_line[0][start_idx]
                quad_start_p1 = quad_line[1][start_idx]
                quad_start_p2 = quad_line[2][start_idx]

                # take middle of datetimes for connecting
                time_diff = quad_dt_end - quad_dt_start

                # calculate the middle time
                quad_dt = quad_dt_start + time_diff / 2

                if quad_dt not in connecting_quad_poly_vertices:
                    connecting_quad_poly_vertices[quad_dt] = [[ ], [ ], [ ], [ ]]

                poly = [quad_end_p1] + [quad_end_p2] + \
                    [quad_start_p2] + [quad_start_p1] + \
                    [quad_end_p1]

                connecting_quad_poly_vertices[quad_dt][j].extend(poly)

        return connecting_quad_poly_vertices

    # interpolate points on a cubic Bezier curve
    @staticmethod
    def cubic_bezier(t, p0, p1, p2, p3):
        """Parametric cubic Bezier curve equation."""
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # generate the path object for the connecting poly
    @staticmethod
    def generate_connecting_paths(connecting_quad_poly_vertices):
        connecting_paths = {}
        # connecting_quad_poly_vertices is a dict (by interpolated dt) of a list of polys (one per quadrant)
        last_connecting_quad = None
        for quad_dt, connecting_quads in connecting_quad_poly_vertices.items():
            for k, connecting_quad in enumerate(connecting_quads):
                num_points = len(connecting_quad)

                if num_points > 1:
                    codes = [Path.MOVETO] + ([Path.LINETO] * (num_points - 2)) + [Path.CLOSEPOLY]
                    # Create the Path object
                    last_connecting_quad = connecting_quad
                    if quad_dt not in connecting_paths:
                        connecting_paths[quad_dt] = [[ ], [ ], [ ], [ ]]

                    connecting_paths[quad_dt][k] = (connecting_quad, codes)

        return connecting_paths

    @staticmethod
    def generate_cone_path_from_paths(concat_paths):
        # Approximate cubic Bezier curves in the path by generating intermediate points
        # for each curve (Path.CURVE4) segment. This creates smooth paths by interpolating
        # between control points. The result is a dense set of points that approximate
        # all the curves in the path.

        # For each path in concat_paths (which represent individual closed shapes),
        # we approximate the Bezier curves and use the points to create Shapely Polygons.

        # Using Shapely’s unary_union, we combine the polygons to get the outermost boundary
        # (the "outline") that surrounds all the paths.

        # Finally, we convert this outer boundary to a Matplotlib Path and display it

        polygons = []
        for path in concat_paths:
            # 1. Approximate Bezier curves in the path
            approx_vertices = WindField.approximate_bezier_path(path, num_points=100)

            # 2. Create a Shapely Polygon for each interpolated closed path
            polygons.append(Polygon(approx_vertices))

        # 3. Find the union of all polygons (outermost boundary)
        polys = []
        for poly in polygons:
            if poly and WindField.geometry_length(poly) > 2:
                p_list = antimeridian.segment_shape(poly)
                for p in p_list:
                    po = Polygon(p)
                    if p and WindField.geometry_length(po) > 2:
                        polys.append(po)

        try:
            valid_polygons = [make_valid(polygon) for polygon in polys]
            actual_polygons = []
            for valid_geom in valid_polygons:
                if isinstance(valid_geom, Polygon):
                    #print("Valid Polygon")
                    actual_polygons.append(valid_geom)
                elif isinstance(valid_geom, MultiPolygon):
                    #print("Valid MultiPolygon")
                    actual_polygons.append(valid_geom)
                else:
                    pass

            union_polygon = unary_union(actual_polygons)
        except GEOSException:
            print("# polygons:", len(actual_polygons))
            union_polygon = []





        # 4. Convert the exterior of the union polygon to a Matplotlib Path
        if isinstance(union_polygon, Polygon):
            outer_boundary_vertices = np.array(union_polygon.exterior.coords)
        else:
            outer_boundary_vertices = []
            for polygon in union_polygon.geoms:
                outer_boundary_vertices.extend(polygon.exterior.coords)
            outer_boundary_vertices = np.array(outer_boundary_vertices)

        num_points = len(outer_boundary_vertices)
        if num_points > 1:
            codes = [Path.MOVETO] + ([Path.LINETO] * (num_points - 2)) + [Path.CLOSEPOLY]
        else:
            codes = None

        if codes:
            outer_boundary_path = Path(outer_boundary_vertices, codes)
        else:
            outer_boundary_path = None

        return outer_boundary_path

    # https://gis.stackexchange.com/questions/119453/count-the-number-of-points-in-a-multipolygon-in-shapely
    @staticmethod
    def geometry_flatten(geom):
        if hasattr(geom, 'geoms'):  # Multi<Type> / GeometryCollection
            for g in geom.geoms:
                yield from WindField.geometry_flatten(g)
        elif hasattr(geom, 'interiors'):  # Polygon
            yield geom.exterior
            yield from geom.interiors
        else:  # Point / LineString
            yield geom

    @staticmethod
    def geometry_length(geom):
        return sum(len(g.coords) for g in WindField.geometry_flatten(geom))

    @staticmethod
    def get_wind_radii_paths_and_gpds_for_steps(wind_radii_selected_list = [34, 50, 64], lat_lon_with_time_step_list = []):
        # Assume ret_dict is your input dictionary
        if lat_lon_with_time_step_list is None or len(lat_lon_with_time_step_list) == 0:
            return None

        model_name = lat_lon_with_time_step_list[0]['model_name']
        #label model

        ret_dict = WindField.interpolate_points_in_speed_list(wind_radii_selected_list, lat_lon_with_time_step_list)

        ret_gpds = {}
        for wind_radii_speed, (wind_radii_path, filler_path, outer_boundary_path) in ret_dict.items():
            # Create a GeoDataFrame for each of the three paths
            points = []
            path_indices = []
            for path_index, path_list in wind_radii_path.items():
                for (vertices, codes) in path_list:
                    real_path = Path(vertices, codes)
                    approx_path = WindField.approximate_bezier_path(real_path, num_points=20)
                    points.extend(approx_path)
                    path_indices.extend([path_index] * len(approx_path))

            geometry = [Point(x) for x in points]

            if len(geometry) > 0:
                wind_radii_gpd = gpd.GeoDataFrame(
                    geometry=geometry,
                    index=path_indices,
                    columns=['geometry']
                )

                wind_radii_gpd['wind_radii_speed'] = wind_radii_speed
                wind_radii_gpd['model_name'] = model_name
                wind_radii_gpd['valid_time'] = wind_radii_gpd.index
            else:
                wind_radii_gpd = None

            points = []
            path_indices = []
            for path_index, path_list in filler_path.items():
                for (vertices, codes) in path_list:
                    real_path = Path(vertices, codes)
                    approx_path = WindField.approximate_bezier_path(real_path, num_points=20)
                    points.extend(approx_path)
                    path_indices.extend([path_index] * len(approx_path))

            geometry = [Point(x) for x in points]

            if len(geometry) > 0:
                filler_gpd = gpd.GeoDataFrame(
                    geometry=geometry,
                    #index=wind_radii_path.keys(),
                    index=path_indices,
                    columns=['geometry']
                )

                filler_gpd['wind_radii_speed'] = wind_radii_speed
                filler_gpd['model_name'] = model_name
                filler_gpd['valid_time'] = filler_gpd.index
            else:
                filler_gpd = None

            if outer_boundary_path is not None:
                outer_boundary = WindField.approximate_bezier_path(outer_boundary_path, num_points = 100)
                outer_boundary_gpd = gpd.GeoDataFrame(
                    geometry=[Point(x) for x in outer_boundary],
                    index=range(len(outer_boundary)),
                    columns=['geometry']
                )
                outer_boundary_gpd['wind_radii_speed'] = wind_radii_speed
                outer_boundary_gpd['model_name'] = model_name
                outer_boundary_gpd['valid_time'] = None
            else:
                outer_boundary_gpd = None

            # Store the GeoDataFrames in a dictionary
            ret_gpds[wind_radii_speed] = [wind_radii_gpd, filler_gpd, outer_boundary_gpd]

        # returns the dict of the 3 paths and the dict of the 3 gpds
        return ret_dict, ret_gpds

    @staticmethod
    def interpolate_points_at_speed(wind_radii_speed, lat_lon_with_time_step_list):
        geod = Geodesic.WGS84
        quad_angles = list(zip([0, 90, 180, 270], ['NE', 'SE', 'SW', 'NW']))

        # Create a Plate Carree projection
        proj = ccrs.PlateCarree()

        # paths representing Paths (interpolated polys) at interpolated times
        paths = {}
        for i, point in reversed(list(enumerate(lat_lon_with_time_step_list))):
            wind_radii_key = f'wind_radii_{str(int(round(wind_radii_speed,0)))}'
            current_point = point
            if i > 0:
                prev_point = lat_lon_with_time_step_list[i-1]
            else:
                prev_point = current_point

            # Calculate the time difference (num steps to interpolate between points)
            current_dt = current_point['valid_time']
            prev_dt = prev_point['valid_time']
            time_diff = round(
                (current_dt - prev_dt).total_seconds() / INTERP_PERIOD_SECONDS, 0)

            # Generate the interpolation steps (0 to 1 ratios)
            step_ratios = np.linspace(0, 1, int(time_diff) + 1)

            # Initialize the list of lists to store the interpolated values
            interp_points = []

            any_valid_wind = False
            if wind_radii_key in current_point:
                current_wind_radii = current_point[wind_radii_key]
            else:
                current_wind_radii = [0.0, 0.0, 0.0, 0.0]

            if wind_radii_key in prev_point:
                prev_wind_radii = prev_point[wind_radii_key]
            else:
                prev_wind_radii = [0.0, 0.0, 0.0, 0.0]

            if prev_wind_radii != [0.0, 0.0, 0.0, 0.0] or current_wind_radii != [0.0, 0.0, 0.0, 0.0]:
                any_valid_wind = True

            last_step_ratio = step_ratios[-1]

            # Calculate the intermediate point using geod.Direct
            g = geod.Inverse(current_point['lat'], current_point['lon'], prev_point['lat'], prev_point['lon'])
            azimuth = g['azi1']
            distance = g['s12']

            # Loop through each step and interpolate the wind radii values
            for step_n, step_ratio in enumerate(step_ratios):
                interpolated_wind_radii_values = [[], [], [], []]
                for n in range(4):
                    if current_wind_radii is not None and prev_wind_radii is not None:
                        interpolated_wind_radii_values[n] = current_wind_radii[n] + (prev_wind_radii[n] - current_wind_radii[n]) * step_ratio

                if step_ratio > 0:
                    intermediate_point = geod.Direct(current_point['lat'], current_point['lon'], azimuth, distance * (step_ratio / last_step_ratio))

                    interpolated_point = {
                        'lon': intermediate_point['lon2'],
                        'lat': intermediate_point['lat2'],
                        wind_radii_key: interpolated_wind_radii_values,
                        'any_valid_wind': any_valid_wind
                    }
                else:
                    interpolated_point = {
                        'lon': current_point['lon'],
                        'lat': current_point['lat'],
                        wind_radii_key: current_wind_radii,
                        'any_valid_wind': any_valid_wind
                    }

                # subtract as we are doing it in reverse order
                step_dt = current_dt - timedelta(seconds=INTERP_PERIOD_SECONDS * step_n)

                interp_points.append((step_dt, interpolated_point))

            for (step_dt, interp_point) in interp_points:
                wind_radii = interp_point[wind_radii_key]
                point_paths = []
                if not interp_point['any_valid_wind'] or wind_radii is None:
                    # skip when no wind field
                    continue

                for j, (angle, quad) in enumerate(quad_angles):
                    # Calculate the radius in degrees for each direction
                    if wind_radii[j] is not None and wind_radii[j] and wind_radii[j] > 0:
                        try:
                            radius = wind_radii[j]
                            d = radius*4*(np.sqrt(2)-1)/3
                        except:
                            print('\n')
                            print(wind_radii)
                            print(j)
                            print(wind_radii[j])
                    else:
                        d = 0
                        radius = 0

                    # bezier wedge (quarter circle) approximations for each quadrant
                    result1 = geod.Direct(interp_point['lat'], interp_point['lon'], angle, radius)
                    result2 = geod.Direct(interp_point['lat'], interp_point['lon'], angle+90, radius)
                    control_point1 = geod.Direct(result1['lat2'], result1['lon2'], angle+90, d)
                    control_point2 = geod.Direct(result2['lat2'], result2['lon2'], angle, d)
                    vertices = [
                        (interp_point['lon'], interp_point['lat']), # center
                        (result1['lon2'], result1['lat2']), # clock wise arc in NEQ order
                        (control_point1['lon2'], control_point1['lat2']),
                        (control_point2['lon2'], control_point2['lat2']),
                        (result2['lon2'], result2['lat2']),
                        (interp_point['lon'], interp_point['lat']),  # back to center
                    ]

                    codes = [
                        Path.MOVETO,  # center
                        Path.LINETO,  # north/south point
                        Path.CURVE4,  # arc to east/west point
                        Path.CURVE4,  # arc to east/west point
                        Path.CURVE4,  # arc to east/west point
                        Path.CLOSEPOLY  # back to center
                    ]

                    point_paths.append((vertices, codes))

                paths[step_dt] = point_paths

        # we have the paths already for the wind fields at each interpolated step

        # what we need is also to generate 'cone' like paths using the paths

        # aggregate the points pairwise
        quad_poly_point_pairs = WindField.aggregate_points_in_pairs(paths)

        # create the connecting lines
        quad_lines = WindField.aggregate_points_by_quadrant(quad_poly_point_pairs)

        # construct the vertices for the connecting polygons (the cone-filling)
        connecting_quad_poly_vertices = WindField.construct_connecting_vertices(quad_lines)

        # generate the connecting paths
        connecting_paths = WindField.generate_connecting_paths(connecting_quad_poly_vertices)

        # join the quad paths (the synoptic shape of the wind radii field) with the cone from interpolation
        concat_paths = WindField.concat_quads_and_connecting_paths(paths, connecting_paths)

        # generates a single (boundary) cone from the concat paths
        boundary_path = WindField.generate_cone_path_from_paths(concat_paths)

        # return values:
        # paths is the (interpolated) wind radii paths
        # connecting_paths is the filler for cone (connecting wind radii paths to different hours)
        # boundary_path is the (simplified) cone boundary

        return paths, connecting_paths, boundary_path

    @staticmethod
    def interpolate_points_in_speed_list(wind_radii_speed_list, lat_lon_with_time_step_list):
        ret_dict = {}
        if len(lat_lon_with_time_step_list) > 0:
            for wind_radii_speed in wind_radii_speed_list:
                wind_radii_path, filler_path, outer_boundary_path = WindField.interpolate_points_at_speed(wind_radii_speed, lat_lon_with_time_step_list)
                # convert "paths" to gpd to be useful for statistics

                ret_dict[wind_radii_speed] = (wind_radii_path, filler_path, outer_boundary_path)

        return ret_dict

# Main app class

class App:
    saved_hidden = {}
    mean_track_obj = None
    blit_show_mean_track = False
    plotter_sst = None
    plotter_tchp_d26 = None
    plotters_ohc = []
    plotters_iso26C = []
    if LOAD_NETCDF:
        file_path = SST_NC_PATH
        # load tchp, d26 together
        plotter_sst = NetCDFPlotter(file_path, 'sst')
        plotter_sst.load_data()

        file_path = TCHP_NC_PATH
        plotter_tchp_d26 = NetCDFPlotter(file_path, 'tchp')
        plotter_tchp_d26.load_data()

        file_paths = OHC_NC_PATHS
        for file_path in file_paths:
            plotter_ohc = NetCDFPlotter(file_path, 'ohc')
            plotter_ohc.load_data()
            plotters_ohc.append(plotter_ohc)

    lon_lat_tc_records = []
    str_tree = None
    wind_field_records = []
    wind_field_str_trees = None

    root = None
    # manually hidden tc candidates and annotations
    hidden_tc_candidates = set()

    level_vars = None
    top_frame = None
    tools_frame = None
    canvas_frame = None
    canvas = None
    adeck_mode_frame = None
    exit_button_adeck = None
    reload_button_adeck = None
    label_adeck_mode = None
    adeck_selected_combobox = None
    adeck_selection_info_label = None
    switch_to_genesis_button = None
    adeck_config_button = None
    genesis_mode_frame = None
    exit_button_genesis = None
    reload_button_genesis = None
    label_genesis_mode = None
    prev_genesis_cycle_button = None
    latest_genesis_cycle_button = None
    genesis_models_label = None
    genesis_selected_combobox = None
    adeck_selection_info_label = None
    genesis_selection_info_label = None
    switch_to_adeck_button = None
    genesis_config_button = None
    add_marker_button = None
    toggle_selection_loop_button = None
    label_mouse_coords_prefix = None
    label_mouse_coords = None
    label_mouse_hover_info_prefix = None
    label_mouse_hover_matches = None
    label_mouse_hover_info_coords = None
    label_mouse_hover_info_valid_time_prefix = None
    label_mouse_hover_info_valid_time = None
    label_mouse_hover_info_model_init_prefix = None
    label_mouse_hover_info_model_init = None
    label_mouse_hover_info_vmax10m_prefix = None
    label_mouse_hover_info_vmax10m = None
    label_mouse_hover_info_mslp_prefix = None
    label_mouse_hover_info_mslp = None
    label_mouse_hover_info_roci_prefix = None
    label_mouse_hover_info_roci = None
    label_mouse_hover_info_isobar_delta_prefix = None
    label_mouse_hover_info_isobar_delta = None
    fig = None
    ax = None
    canvas = None
    axes_size = None
    # global extent
    global_extent = [-180, 180, -90, 90]
    # default mode (AB DECK mode)
    mode = "ADECK"
    recent_storms = None
    adeck = None
    bdeck = None
    adeck_previous_selected = None
    genesis_previous_selected = None
    adeck_storm = None
    genesis_model_cycle_time = None
    genesis_model_cycle_time_navigate = None
    zoom_selection_box = None
    last_cursor_lon_lat = (0.0, 0.0)
    lastgl = None
    # track first fetch (don't update staleness until a reload, combobox selection, or model cycle change)
    have_deck_data = False
    have_genesis_data = False
    # track whether there is new tcvitals,adecks,bdecks data
    deck_timer_id = None
    genesis_timer_id = None
    stale_urls = dict()
    stale_urls['tcvitals'] = set()
    stale_urls['adeck'] = set()
    stale_urls['bdeck'] = set()
    stale_adeck2 = False
    stale_genesis_data = dict()
    stale_genesis_data['global-det'] = []
    stale_genesis_data['tcgen'] = []
    # keys are the urls we will check for if it is stale, values are datetime objects
    dt_mods_tcvitals = {}
    dt_mods_adeck = {}
    dt_mods_adeck2 = {}
    dt_mods_bdeck = {}
    # keys are the type of genesis data we will check for if data is stale ('global-det' or 'tcgen'), values are datetime objects
    dt_mods_genesis = {}
    # measure tool
    measure_mode = False
    start_point = None
    end_point = None
    line = None
    distance_text = None
    # input event tracking
    cid_press = None
    cid_release = None
    cid_motion = None
    cid_key_press = None
    cid_key_release = None
    # keep list of all lat_lon_with_time_step_list (plural) used to create points in last drawn map
    #   note, each item is also a list of dicts (not enumerated)
    plotted_tc_candidates = []
    # r-tree index
    rtree_p = index.Property()
    rtree_idx = index.Index(properties=rtree_p)
    # Mapping from rtree point index to (internal_id, tc_index, tc_candidate_point_index)
    rtree_tuple_point_id = 0
    rtree_tuple_index_mapping = {}

    scatter_objects = {}
    line_collection_objects = {}
    annotated_circle_objects = {}
    # circle patch for selected marker
    circle_handle = None
    last_circle_lon = None
    last_circle_lat = None
    # track overlapped points (by index, pointing to the plotted_tc_candidates)
    #   this will hold information on the marker where the cursor previously pointed to (current circle patch),
    #   and which one of the possible matches was (is currently) viewed
    nearest_point_indices_overlapped = SortedCyclicEnumDict()
    # settings for plotting
    time_step_marker_colors = [
        '#ffff00',
        '#ba0a0a', '#e45913', '#fb886e', '#fdd0a2',
        '#005b1c', '#07a10b', '#9cd648', '#a5ee96',
        '#0d3860', '#2155c4', '#33aaff', '#7acaff',
        '#710173', '#b82cae', '#c171cf', '#ffb9ee',
        '#636363', '#969696', '#bfbfbf', '#e9e9e9'
    ]
    time_step_legend_fg_colors = [
        '#000000',
        '#ffffff', '#ffffff', '#000000', '#000000',
        '#ffffff', '#ffffff', '#000000', '#000000',
        '#ffffff', '#ffffff', '#000000', '#000000',
        '#ffffff', '#ffffff', '#000000', '#000000',
        '#ffffff', '#000000', '#000000', '#000000'
    ]
    # Will change when clicked
    time_step_opacity = [
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
    time_step_ranges = [
        (float('-inf'), -1),
        (0, 23),
        (24, 47),
        (48, 71),
        (72, 95),
        (96, 119),
        (120, 143),
        (144, 167),
        (168, 191),
        (192, 215),
        (216, 239),
        (240, 263),
        (264, 287),
        (288, 311),
        (312, 335),
        (336, 359),
        (360, 383),
        (384, 407),
        (408, 431),
        (432, 455),
        (456, float('inf'))
    ]
    time_step_legend_objects = []
    # a pair of dicts of selected rvor levels' contours, with contour ids as keys
    overlay_rvor_contour_objs = []
    overlay_rvor_label_objs = []
    overlay_rvor_contour_dict = None

    overlay_rvor_contour_visible = True
    overlay_rvor_label_visible = True
    overlay_rvor_label_last_alpha = 1.0

    rvor_dialog_open = False
    hide_by_field_dialog_open = False
    analysis_dialog_open = False

    selection_loop_mode = False
    # Bbox for circle patch blitting
    background_for_blit = None

    draggable_annotations = []

    @classmethod
    def annotate_single_storm_extrema(cls, point_index=None):
        global ANNOTATE_COLOR_LEVELS
        if point_index is None or len(point_index) != 3:
            return
        internal_id, tc_index, tc_point_index = point_index
        if not cls.plotted_tc_candidates or (tc_index + 1) > len(cls.plotted_tc_candidates):
            return

        results = {}
        for short_name in DISPLAYED_FUNCTIONAL_ANNOTATIONS:
            result_tuple = annotations_result_val[short_name](cls.plotted_tc_candidates[tc_index][1])
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

                result_point = cls.plotted_tc_candidates[tc_index][1][result_idx]
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
        cleared_once = False
        for label_point_index, (point_label, color_level) in point_index_labels.items():
            point = cls.plotted_tc_candidates[tc_index][1][label_point_index]
            added = True
            # check if already annotated
            #   there is a question of how we want to use annotations (one or many per (nearby or same) point?)
            #   in AnnotatedCircles, we use has_overlap() to prevent annotating twice
            annotated_circle = AnnotatedCircles.add(lat=point['lat'], lon=point['lon'], label=point_label,
                                                    label_color=ANNOTATE_COLOR_LEVELS[color_level], internal_id=internal_id, point_index=label_point_index)
            # handle case to make sure we don't add doubles
            if annotated_circle is None and not cleared_once:
                # in this case let's re-annotate by clearning the existing annotations for this track (internal_id)
                if internal_id in cls.annotated_circle_objects:
                    removed_any = False
                    for point_index, ac in cls.annotated_circle_objects[internal_id].items():
                        ac.remove()
                        removed_any = True

                    del cls.annotated_circle_objects[internal_id]
                    if removed_any:
                        cleared_once = True
                        annotated_circle = AnnotatedCircles.add(lat=point['lat'], lon=point['lon'], label=point_label,
                                                                label_color=ANNOTATE_COLOR_LEVELS[color_level],
                                                                internal_id=internal_id, point_index=label_point_index)

            if not cls.annotated_circle_objects:
                cls.annotated_circle_objects = {}
            if internal_id not in cls.annotated_circle_objects:
                cls.annotated_circle_objects[internal_id] = {}
            cls.annotated_circle_objects[internal_id][label_point_index] = annotated_circle

        if added:
            cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def annotate_storm_extrema(cls):
        if len(cls.plotted_tc_candidates) == 0:
            return

        # note: point_index is a tuple of tc_index, tc_point_index
        if len(cls.nearest_point_indices_overlapped) == 0:
            # annotate all storm extrema in current view
            for tc_index in range(len(cls.plotted_tc_candidates)):
                internal_id, tc_candidate = cls.plotted_tc_candidates[tc_index]
                if len(tc_candidate):
                    if cls.any_storm_points_in_bounds(tc_index):
                        point_index = (internal_id, tc_index, 0)
                        cls.annotate_single_storm_extrema(point_index=point_index)
        else:
            # annotate storm extrema of previously selected
            num, cursor_point_index = cls.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            cls.annotate_single_storm_extrema(point_index=cursor_point_index)

    @classmethod
    def any_modal_open(cls):
        return cls.hide_by_field_dialog_open or cls.analysis_dialog_open or cls.rvor_dialog_open

    @classmethod
    def any_storm_points_in_bounds(cls, tc_index):
        if not cls.plotted_tc_candidates:
            return False
        if (tc_index + 1) > len(cls.plotted_tc_candidates):
            return False
        if not cls.ax:
            return False

        any_in_bound = False

        xlim = cls.ax.get_xlim()
        ylim = cls.ax.get_ylim()
        try:
            internal_id, tc_candidate = cls.plotted_tc_candidates[tc_index]
            for point in tc_candidate:
                if len(cls.hidden_tc_candidates) == 0 or internal_id not in cls.hidden_tc_candidates:
                    lat = point['lat']
                    lon = point['lon']
                    any_in_bound = any_in_bound or (xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1])
        except:
            traceback.print_exc()
            pass

        return any_in_bound

    @classmethod
    def blit_circle_patch(cls, stale_bg=False):
        if cls.ax is None:
            return

        if stale_bg or cls.background_for_blit is None or cls.circle_handle is None:
            cls.background_for_blit = cls.ax.figure.canvas.copy_from_bbox(cls.ax.bbox)
            if not cls.circle_handle:
                return

        if cls.circle_handle:
            cls.ax.draw_artist(cls.circle_handle)
            cls.ax.figure.canvas.blit(cls.ax.bbox)
        elif cls.background_for_blit:
            cls.ax.figure.canvas.restore_region(cls.background_for_blit)
            cls.ax.figure.canvas.blit(cls.ax.bbox)

    @classmethod
    def blit_mean_track(cls, stale_bg=False, init=False):
        if cls.ax is None:
            return

        if not init:
            cls.mean_track_obj = None
            mean_track = None
        else:
            if SelectionLoops.is_empty():
                return

            filtered_candidates = []
            internal_ids = cls.get_selected_internal_storm_ids()
            num_tracks = 0
            if internal_ids:
                num_tracks = len(internal_ids)

            if num_tracks == 0:
                return

            filtered_candidates = [(iid, tc) for iid, tc in cls.plotted_tc_candidates if
                                   iid in internal_ids]

            tz = 'UTC'
            timezone = pytz.timezone(tz)
            offset = datetime.now(timezone).strftime('%z')
            hours, minutes = int(offset[:3]), int(offset[0] + offset[3:])
            formatted_offset = f"{'+' if hours >= 0 else '-'}{abs(hours):02}:{abs(minutes):02}"
            tz_previous_selected = f"(UTC{formatted_offset}) {tz}"

            all_datetimes = AnalysisDialog.get_unique_datetimes(filtered_candidates, internal_ids, tz_previous_selected)
            mean_track = AnalysisDialog.calculate_mean_track_interp(filtered_candidates, all_datetimes, internal_ids)
            lat_lon_with_time_step_list = []
            len_mean_track = len(mean_track)
            valid_day = datetime.fromisoformat(cls.valid_day)
            for i in range(len_mean_track):
                prev_lon = None
                prev_lat = None
                if i > 0:
                    lon = mean_track[i]['lon']
                    prev_lon = mean_track[i - 1]['lon']
                    if prev_lon:
                        prev_lon_f = float(prev_lon)
                        if abs(prev_lon_f - lon) > 270:
                            if prev_lon_f < lon:
                                prev_lon = prev_lon_f + 360
                            else:
                                prev_lon = prev_lon_f - 360

                mean_track[i]['prev_lat'] = mean_track[i - 1]['lat']
                mean_track[i]['prev_lon'] = prev_lon

                hours_diff = (mean_track[i]['valid_time'] - valid_day).total_seconds() / 3600
                # round to the nearest hour
                hours_diff_rounded = round(hours_diff)

                mean_track[i]['hours_after_valid_day'] = hours_diff_rounded

        if stale_bg or cls.background_for_blit is None:
            cls.background_for_blit = cls.ax.figure.canvas.copy_from_bbox(cls.ax.bbox)
            if init:
                cls.mean_track_obj = None

        if cls.mean_track_obj is None and mean_track is not None:
            lon_lat_tuples = []
            llwtsl_indices = []
            #cls.ax.draw_artist(mean_track_obj)
            vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'),
                                 (float('inf'), '*')]
            vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2606 5']
            marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

            scatter_objects = []
            # do in reversed order so most recent items get rendered on top
            for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                opacity = 1.0
                lons = {}
                lats = {}
                for llwtsl_idx, point in reversed(list(enumerate(mean_track))):
                    hours_after = point['hours_after_valid_day']
                    # if start <= time_step <= end:
                    # use hours after valid_day instead
                    if start <= hours_after <= end:
                        marker = "*"
                        for upper_bound, vmaxmarker in vmax_kt_threshold:
                            marker = vmaxmarker
                            if point['vmax10m'] < upper_bound:
                                break
                        if marker not in lons:
                            lons[marker] = []
                            lats[marker] = []
                        lons[marker].append(point['lon'])
                        lats[marker].append(point['lat'])
                        lon_lat_tuples.append([point['lon'], point['lat']])
                        # track with indices are visible so we can construct a visible version
                        llwtsl_indices.append(llwtsl_idx)

                for vmaxmarker in lons.keys():
                    scatter = cls.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker,
                                             facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                             s=marker_sizes[vmaxmarker] ** 2, alpha=opacity, antialiased=False)
                    scatter_objects.append(scatter)
                    #cls.ax.draw_artist(scatter)

            line_collection_objects = []
            for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                line_color = cls.time_step_marker_colors[i]
                opacity = 1.0
                strokewidth = 3

                line_segments = []
                for point in reversed(mean_track):
                    hours_after = point['hours_after_valid_day']
                    if start <= hours_after <= end:
                        if point['prev_lon']:
                            # Create a list of line segments
                            line_segments.append([(point['prev_lon'], point['prev_lat']),
                                                  (point['lon'], point['lat'])])

                # Create a LineCollection
                lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                # Add the LineCollection to the axes
                line_collection = cls.ax.add_collection(lc)
                line_collection_objects.append(line_collection)
                #cls.ax.draw_artist(line_collection)

            cls.mean_track_obj = [scatter_objects, line_collection_objects]

        if cls.background_for_blit:
            cls.ax.figure.canvas.restore_region(cls.background_for_blit)

            if cls.mean_track_obj:
                scatters, lines = cls.mean_track_obj
                for scatter in scatters:
                    cls.ax.draw_artist(scatter)

                for line_collection in lines:
                    cls.ax.draw_artist(line_collection)

                cls.ax.figure.canvas.blit(cls.ax.bbox)

    @staticmethod
    def calculate_distance(start_point, end_point):
        geod = cgeo.Geodesic()
        line = LineString([start_point, end_point])
        total_distance = geod.geometry_length(line)
        nautical_miles = total_distance / 1852.0
        return nautical_miles

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

    @classmethod
    def check_for_stale_deck_data(cls):
        if cls.deck_timer_id is not None:
            cls.root.after_cancel(cls.deck_timer_id)
        cls.deck_timer_id = cls.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_deck_data)
        if cls.dt_mods_tcvitals:
            for url, old_dt_mod in cls.dt_mods_tcvitals.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        cls.stale_urls['tcvitals'] = cls.stale_urls['tcvitals'] | {url}
        if cls.dt_mods_adeck:
            for url, old_dt_mod in cls.dt_mods_adeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        cls.stale_urls['adeck'] = cls.stale_urls['adeck'] | {url}
        if cls.dt_mods_adeck2:
            for local_filename, old_dt_mod in cls.dt_mods_adeck2.items():
                new_dt_mod = os.path.getmtime(local_filename)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        cls.stale_adeck2 = True
        if cls.dt_mods_bdeck:
            for url, old_dt_mod in cls.dt_mods_bdeck.items():
                new_dt_mod = http_get_modification_date(url)
                if new_dt_mod:
                    if new_dt_mod > old_dt_mod:
                        cls.stale_urls['bdeck'] = cls.stale_urls['bdeck'] | {url}

        cls.update_reload_button_color_for_deck()

    @classmethod
    def check_for_stale_genesis_data(cls):
        if cls.genesis_timer_id is not None:
            cls.root.after_cancel(cls.genesis_timer_id)

        cls.genesis_timer_id = cls.root.after(LOCAL_TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_genesis_data)

        if cls.dt_mods_genesis:
            # genesis_source_type is either 'global-det' or 'tcgen'
            for genesis_source_type, model_init_times, model_completed_times, \
                ensemble_completed_times, completed_ensembles in cls.get_latest_genesis_data_times():

                max_dt = datetime.min

                if not model_completed_times:
                    continue

                for model_name, completed_times in model_completed_times.items():
                    if type(completed_times) == datetime:
                        new_max_dt = completed_times
                    else:
                        new_max_dt = max(completed_times)
                    max_dt = max(max_dt, new_max_dt)

                new_dt_mod = max_dt

                if genesis_source_type in cls.dt_mods_genesis:
                    old_dt_mod = cls.dt_mods_genesis[genesis_source_type]
                else:
                    old_dt_mod = datetime.min

                if new_dt_mod != datetime.min and new_dt_mod > old_dt_mod:
                    cls.stale_genesis_data[genesis_source_type] = new_dt_mod

        else:
            for genesis_source_type, model_init_times, model_completed_times, \
                ensemble_completed_times, completed_ensembles in cls.get_latest_genesis_data_times():

                if not model_completed_times:
                    continue

                max_dt = datetime.min
                for model_name, completed_times in model_completed_times.items():
                    if type(completed_times) == datetime:
                        new_max_dt = completed_times
                    else:
                        new_max_dt = max(completed_times)
                    max_dt = max(max_dt, new_max_dt)

                new_dt_mod = max_dt
                if new_dt_mod != datetime.min:
                    cls.stale_genesis_data[genesis_source_type] = max_dt

        cls.update_reload_button_color_for_genesis()

    @classmethod
    def clear_circle_patch(cls):
        if cls.circle_handle:
            # restore region
            if cls.background_for_blit:
                cls.ax.figure.canvas.restore_region(cls.background_for_blit)
                cls.ax.figure.canvas.blit(cls.ax.bbox)

            #cls.circle_handle.remove()
            cls.circle_handle = None
            # cls.ax.set_yscale('linear')
            # cls.redraw_fig_canvas()

        cls.last_circle_lon = None
        cls.last_circle_lat = None

    @classmethod
    def clear_plotted_list(cls):
        cls.plotted_tc_candidates = []
        cls.rtree_p = index.Property()
        cls.rtree_idx = index.Index(properties=cls.rtree_p)
        cls.rtree_tuple_point_id = 0
        cls.rtree_tuple_index_mapping = {}
        # reset all labels
        cls.update_tc_status_labels()
        cls.clear_circle_patch()
        cls.lon_lat_tc_records = []
        cls.wind_field_records = {}
        cls.wind_field_strtrees = {}

    @classmethod
    def clear_storm_extrema_annotations(cls):
        AnnotatedCircles.clear()
        cls.annotated_circle_objects = {}

    @classmethod
    def adeck_combo_selected_models_event(cls, event):
        current_value = cls.adeck_selected_combobox.get()
        if current_value == cls.adeck_previous_selected:
            # user did not change selection
            cls.set_focus_on_map()
            return
        else:
            cls.adeck_previous_selected = current_value
            cls.display_map()
            if not cls.have_deck_data:
                cls.update_deck_data()
            cls.hidden_tc_candidates = set()
            cls.display_deck_data()
            cls.set_focus_on_map()

    @classmethod
    def genesis_combo_selected_models_event(cls, event):
        current_value = cls.genesis_selected_combobox.get()
        if current_value == cls.genesis_previous_selected:
            # user did not change selection
            cls.set_focus_on_map()
            return
        else:
            cls.genesis_previous_selected = current_value
            if not cls.have_genesis_data:
                # track first fetch
                cls.update_genesis_data_staleness()
            model_cycle = cls.genesis_model_cycle_time
            if model_cycle is None:
                model_cycles = get_tc_model_init_times_relative_to(datetime.now(), cls.genesis_previous_selected)
                if model_cycles['next'] is None:
                    model_cycle = model_cycles['at']
                else:
                    model_cycle = model_cycles['next']
            if model_cycle:
                # clear map
                cls.redraw_map_with_data(model_cycle=model_cycle)
                cls.set_focus_on_map()

    @classmethod
    def create_adeck_mode_widgets(cls):
        cls.adeck_mode_frame = ttk.Frame(cls.top_frame, style="TopFrame.TFrame")

        cls.exit_button_adeck = ttk.Button(cls.adeck_mode_frame, text="EXIT", command=cls.root.quit, style="TButton")
        cls.exit_button_adeck.pack(side=tk.LEFT, padx=5, pady=5)

        cls.reload_button_adeck = ttk.Button(cls.adeck_mode_frame, text="(RE)LOAD", command=cls.reload_adeck,
                                              style="White.TButton")
        cls.reload_button_adeck.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_adeck_mode = ttk.Label(cls.adeck_mode_frame, text="ADECK MODE. Models: 0", style="TLabel")
        cls.label_adeck_mode.pack(side=tk.LEFT, padx=5, pady=5)

        cls.adeck_selected_combobox = ttk.Combobox(cls.adeck_mode_frame, width=14, textvariable=cls.adeck_selected,
                                                    state='readonly', style='Black.TCombobox', height=20)
        cls.adeck_selected_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        cls.adeck_selected_combobox['state'] = 'readonly'  # Set the state according to configure colors
        cls.adeck_selected_combobox['values'] = (
        'ALL', 'STATISTICAL', 'GLOBAL', 'GEFS-MEMBERS', 'GEFS-ATCF', 'GEPS-ATCF', 'FNMOC-ATCF', 'ALL-ATCF', 'REGIONAL', 'CONSENSUS', 'OFFICIAL')
        cls.adeck_selected_combobox.current(10)
        cls.adeck_previous_selected = cls.adeck_selected.get()

        cls.adeck_selected_combobox.bind("<<ComboboxSelected>>", cls.adeck_combo_selected_models_event)

        cls.adeck_selection_info_label = ttk.Label(cls.adeck_mode_frame,
                                                    text="",
                                                    style="TLabel")
        cls.adeck_selection_info_label.pack(side=tk.LEFT, padx=5, pady=5)

        cls.adeck_selection_info_label = ttk.Label(cls.adeck_mode_frame,
                                                      text="",
                                                      style="TLabel")
        cls.adeck_selection_info_label.pack(side=tk.LEFT, padx=5, pady=5)

        cls.switch_to_genesis_button = ttk.Button(cls.adeck_mode_frame, text="SWITCH TO GENESIS MODE",
                                                   command=cls.switch_mode, style="TButton")
        cls.switch_to_genesis_button.pack(side=tk.RIGHT, padx=5, pady=5)

        cls.adeck_config_button = ttk.Button(cls.adeck_mode_frame, text="CONFIG \u2699",
                                              command=cls.show_config_adeck_dialog, style="TButton")
        cls.adeck_config_button.pack(side=tk.RIGHT, padx=5, pady=5)

    @classmethod
    def create_genesis_mode_widgets(cls):
        cls.genesis_mode_frame = ttk.Frame(cls.top_frame, style="TopFrame.TFrame")

        cls.exit_button_genesis = ttk.Button(cls.genesis_mode_frame, text="EXIT", command=cls.root.quit,
                                              style="TButton")
        cls.exit_button_genesis.pack(side=tk.LEFT, padx=5, pady=5)

        cls.reload_button_genesis = ttk.Button(cls.genesis_mode_frame, text="(RE)LOAD", command=cls.reload_genesis,
                                                style="White.TButton")
        cls.reload_button_genesis.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_genesis_mode = ttk.Label(cls.genesis_mode_frame, text="GENESIS MODE: Start valid day: YYYY-MM-DD",
                                            style="TLabel")
        cls.label_genesis_mode.pack(side=tk.LEFT, padx=5, pady=5)

        cls.prev_genesis_cycle_button = ttk.Button(cls.genesis_mode_frame, text="PREV CYCLE",
                                                    command=cls.prev_genesis_cycle, style="TButton")
        cls.prev_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        cls.prev_genesis_cycle_button = ttk.Button(cls.genesis_mode_frame, text="NEXT CYCLE",
                                                    command=cls.next_genesis_cycle, style="TButton")
        cls.prev_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        cls.latest_genesis_cycle_button = ttk.Button(cls.genesis_mode_frame, text="LATEST CYCLE",
                                                      command=cls.latest_genesis_cycle, style="TButton")
        cls.latest_genesis_cycle_button.pack(side=tk.LEFT, padx=5, pady=5)

        cls.genesis_models_label = ttk.Label(cls.genesis_mode_frame,
                                              text="Models: GFS [--/--Z], ECM[--/--Z], NAV[--/--Z], CMC[--/--Z]",
                                              style="TLabel")
        cls.genesis_models_label.pack(side=tk.LEFT, padx=5, pady=5)

        cls.genesis_selected_combobox = ttk.Combobox(cls.genesis_mode_frame, width=14, textvariable=cls.genesis_selected,
                                                    state='readonly', style='Black.TCombobox')
        cls.genesis_selected_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        cls.genesis_selected_combobox['state'] = 'readonly'  # Set the state according to configure colors
        cls.genesis_selected_combobox['values'] = (
            'GLOBAL-DET', 'ALL-TCGEN', 'GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN')
        cls.genesis_selected_combobox.current(0)
        cls.genesis_previous_selected = cls.genesis_selected.get()
        cls.genesis_selected_combobox.bind("<<ComboboxSelected>>", cls.genesis_combo_selected_models_event)

        cls.genesis_selection_info_label = ttk.Label(cls.genesis_mode_frame,
                                                      text="",
                                                      style="TLabel")
        cls.genesis_selection_info_label.pack(side=tk.LEFT, padx=5, pady=5)

        cls.switch_to_adeck_button = ttk.Button(cls.genesis_mode_frame, text="SWITCH TO ADECK MODE",
                                                 command=cls.switch_mode, style="TButton")
        cls.switch_to_adeck_button.pack(side=tk.RIGHT, padx=5, pady=5)

        cls.genesis_config_button = ttk.Button(cls.genesis_mode_frame, text="CONFIG \u2699",
                                                command=cls.show_config_genesis_dialog, style="TButton")
        cls.genesis_config_button.pack(side=tk.RIGHT, padx=5, pady=5)

    @classmethod
    def create_tools_widgets(cls):
        # cls.tools_frame = ttk.Frame(cls.tools_frame, style="Tools.TFrame")

        cls.toggle_selection_loop_button = ttk.Button(cls.tools_frame, text="\u27B0 SELECT",
                                                       command=cls.toggle_selection_loop_mode, style="TButton")
        cls.toggle_selection_loop_button.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_coords_prefix = ttk.Label(cls.tools_frame, text="Cursor position:", style="TLabel")
        cls.label_mouse_coords_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_coords = ttk.Label(cls.tools_frame, text="(-tt.tttt, -nnn.nnnn)",
                                            style="FixedWidthWhite.TLabel")
        cls.label_mouse_coords.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_prefix = ttk.Label(cls.tools_frame, text="(Hover) Matches:", style="TLabel")
        cls.label_mouse_hover_info_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_matches = ttk.Label(cls.tools_frame, text="0  ", style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_matches.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_coords = ttk.Label(cls.tools_frame, text="(-tt.tttt, -nnn.nnnn)",
                                                       style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_coords.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_valid_time_prefix = ttk.Label(cls.tools_frame, text="Valid time:", style="TLabel")
        cls.label_mouse_hover_info_valid_time_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_valid_time = ttk.Label(cls.tools_frame, text="YYYY-MM-DD hhZ",
                                                           style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_valid_time.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_model_init_prefix = ttk.Label(cls.tools_frame, text="Model init:", style="TLabel")
        cls.label_mouse_hover_info_model_init_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_model_init = ttk.Label(cls.tools_frame, text="YYYY-MM-DD hhZ",
                                                           style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_model_init.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_vmax10m_prefix = ttk.Label(cls.tools_frame, text="Vmax @ 10m:", style="TLabel")
        cls.label_mouse_hover_info_vmax10m_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_vmax10m = ttk.Label(cls.tools_frame, text="---.- kt",
                                                        style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_vmax10m.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_mslp_prefix = ttk.Label(cls.tools_frame, text="MSLP:", style="TLabel")
        cls.label_mouse_hover_info_mslp_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_mslp = ttk.Label(cls.tools_frame, text="----.- hPa",
                                                     style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_mslp.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_roci_prefix = ttk.Label(cls.tools_frame, text="ROCI:", style="TLabel")
        cls.label_mouse_hover_info_roci_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_roci = ttk.Label(cls.tools_frame, text="---- km", style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_roci.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_isobar_delta_prefix = ttk.Label(cls.tools_frame, text="Isobar delta:",
                                                                    style="TLabel")
        cls.label_mouse_hover_info_isobar_delta_prefix.pack(side=tk.LEFT, padx=5, pady=5)

        cls.label_mouse_hover_info_isobar_delta = ttk.Label(cls.tools_frame, text="--- hPa",
                                                             style="FixedWidthWhite.TLabel")
        cls.label_mouse_hover_info_isobar_delta.pack(side=tk.LEFT, padx=5, pady=5)

    @classmethod
    def create_widgets(cls):
        cls.top_frame = ttk.Frame(cls.root, style="TopFrame.TFrame")
        cls.top_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        # Middle frame
        cls.tools_frame = ttk.Frame(cls.root, style="ToolsFrame.TFrame")
        cls.tools_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        cls.create_adeck_mode_widgets()
        cls.create_genesis_mode_widgets()
        cls.create_tools_widgets()

        cls.update_mode()

        cls.canvas_frame = ttk.Frame(cls.root, style="CanvasFrame.TFrame")
        cls.canvas_frame.pack(fill=tk.X, expand=True, anchor=tk.NW)

        cls.canvas = None
        cls.fig = None

    @classmethod
    def cycle_to_next_overlapped_point(cls):
        # called when user hovers on overlapped points and hits the TAB key
        total_num_overlapped_points = len(cls.nearest_point_indices_overlapped)
        if total_num_overlapped_points > 1:
            overlapped_point_num, nearest_point_index = cls.nearest_point_indices_overlapped.next_enum_key_tuple()
            internal_id, tc_index, point_index = nearest_point_index
            cls.update_tc_status_labels(tc_index, point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = cls.plotted_tc_candidates[tc_index][1][point_index]
            lon = point['lon']
            lat = point['lat']
            cls.update_circle_patch(lon=lon, lat=lat)

    @classmethod
    def display_adt(cls, url):
        adt_track = parse_adt(url)
        if len(adt_track) == 0:
            return
        lat_lon_with_time_step_list = []
        len_adt_track = len(adt_track)
        valid_day = datetime.fromisoformat(cls.valid_day)
        for i in range(len_adt_track):
            prev_lon = None
            prev_lat = None
            if i > 0:
                lon = adt_track[i]['lon']
                prev_lon = adt_track[i - 1]['lon']
                if prev_lon:
                    prev_lon_f = float(prev_lon)
                    if abs(prev_lon_f - lon) > 270:
                        if prev_lon_f < lon:
                            prev_lon = prev_lon_f + 360
                        else:
                            prev_lon = prev_lon_f - 360

            adt_track[i]['prev_lat'] = adt_track[i - 1]['lat']
            adt_track[i]['prev_lon'] = prev_lon

            hours_diff = (adt_track[i]['valid_time'] - valid_day).total_seconds() / 3600
            # round to the nearest hour
            hours_diff_rounded = round(hours_diff)

            adt_track[i]['hours_after_valid_day'] = hours_diff_rounded

        lon_lat_tuples = []
        llwtsl_indices = []
        # cls.ax.draw_artist(adt_track)
        vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'),
                             (float('inf'), '*')]
        vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2606 5']
        marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

        scatter_objects = []
        # do in reversed order so most recent items get rendered on top
        for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
            opacity = 0.3
            opacity_archer = 0.6
            lons = {}
            lats = {}
            lons_archer = {}
            lats_archer = {}
            opacities = {}
            for llwtsl_idx, point in reversed(list(enumerate(adt_track))):
                hours_after = point['hours_after_valid_day']
                # if start <= time_step <= end:
                # use hours after valid_day instead
                if start <= hours_after <= end:
                    marker = "*"
                    for upper_bound, vmaxmarker in vmax_kt_threshold:
                        marker = vmaxmarker
                        if point['vmax10m'] < upper_bound:
                            break
                    if point['fixmethod'] == 'ARCHER':
                        if marker not in lons_archer:
                            lons_archer[marker] = []
                            lats_archer[marker] = []
                        lons_archer[marker].append(point['lon'])
                        lats_archer[marker].append(point['lat'])
                    else:
                        if marker not in lons:
                            lons[marker] = []
                            lats[marker] = []
                        lons[marker].append(point['lon'])
                        lats[marker].append(point['lat'])

                    lon_lat_tuples.append([point['lon'], point['lat']])
                    # track with indices are visible so we can construct a visible version
                    llwtsl_indices.append(llwtsl_idx)

            for vmaxmarker in lons.keys():
                scatter = cls.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker,
                                         facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                         s=marker_sizes[vmaxmarker] ** 2, alpha=opacity, antialiased=False)
                scatter_objects.append(scatter)
                cls.ax.draw_artist(scatter)

            for vmaxmarker in lons_archer.keys():
                scatter = cls.ax.scatter(lons_archer[vmaxmarker], lats_archer[vmaxmarker], marker=vmaxmarker,
                                         facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                         s=marker_sizes[vmaxmarker] ** 2, alpha=opacity_archer, antialiased=False)
                scatter_objects.append(scatter)
                cls.ax.draw_artist(scatter)

        line_collection_objects = []
        for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
            line_color = cls.time_step_marker_colors[i]
            opacity = 1.0
            strokewidth = 2

            line_segments = []
            for point in reversed(adt_track):
                hours_after = point['hours_after_valid_day']
                if start <= hours_after <= end:
                    if point['prev_lon']:
                        # Create a list of line segments
                        line_segments.append([(point['prev_lon'], point['prev_lat']),
                                              (point['lon'], point['lat'])])

            # Create a LineCollection
            # for ADT change linestyle to dotted
            lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity, linestyle='dotted')
            # Add the LineCollection to the axes
            line_collection = cls.ax.add_collection(lc)
            line_collection_objects.append(line_collection)
            cls.ax.draw_artist(line_collection)

        # update blit
        cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def display_custom_boundaries(cls, label_column=None):
        if custom_gdf is not None:
            for geometry in custom_gdf.geometry:
                if isinstance(geometry, Polygon):
                    cls.ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none',
                                           linewidth=2)
                else:
                    cls.ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='magenta', facecolor='none',
                                           linewidth=2)

        if label_column:
            for idx, row in custom_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                cls.ax.text(x, y, row[label_column], transform=ccrs.PlateCarree(), fontsize=8, color='magenta')

    # display custom rvor overlay from shape files
    @classmethod
    def display_custom_overlay(cls):
        if not overlay_gdfs:
            return
        cls.init_rvor_contour_dict()
        global RVOR_CYCLONIC_CONTOURS
        global RVOR_CYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_LABELS
        global RVOR_ANTICYCLONIC_CONTOURS
        global SELECTED_PRESSURE_LEVELS
        if SELECTED_PRESSURE_LEVELS == []:
            return

        extent, min_span_lat_deg, min_span_lon_deg = cls.get_contour_min_span_deg()

        do_overlay_shapes = {
            'rvor_c_poly': RVOR_CYCLONIC_CONTOURS,
            'rvor_c_points': RVOR_CYCLONIC_LABELS,
            'rvor_ac_poly': RVOR_ANTICYCLONIC_CONTOURS,
            'rvor_ac_points': RVOR_ANTICYCLONIC_LABELS
        }

        renderable_ids = cls.overlay_rvor_contour_dict['renderable_ids']
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
                cls.overlay_rvor_contour_dict['ids'].add(contour_id)
                cls.overlay_rvor_contour_dict['contour_span_lons'][contour_id] = span_lon
                cls.overlay_rvor_contour_dict['contour_span_lats'][contour_id] = span_lat

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
            contour_visible = cls.overlay_rvor_contour_visible
            not_at_global_extent = not (
                    extent[0] == cls.global_extent[0] and
                    extent[1] == cls.global_extent[1] and
                    extent[2] == cls.global_extent[2] and
                    extent[3] == cls.global_extent[3]
            )
            label_visible = contour_visible and cls.overlay_rvor_label_visible and not_at_global_extent
            if label_visible:
                alpha_label_visible = 1.0
            else:
                alpha_label_visible = 0.0
            cls.overlay_rvor_label_last_alpha = alpha_label_visible

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
                    # is_visible = contour_visible and is_renderable
                    # only limit labels to detail
                    is_visible = contour_visible
                    obj = cls.ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor=edge_color, facecolor='none',
                                                 linewidth=2, visible=is_visible)
                    cls.overlay_rvor_contour_dict['contour_objs'][contour_id] = [obj]
                else:
                    # Draw labels
                    edge_color = edge_colors[str(row['level'])]
                    is_visible = label_visible and is_renderable
                    x, y = geom.x, geom.y
                    label = row['label']
                    # obj = cls.ax.text(x, y, label, transform=ccrs.PlateCarree(), color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor=edge_color, edgecolor='black', pad=2), alpha=alpha_label_visible)
                    obj = cls.ax.text(x, y, label, transform=ccrs.PlateCarree(), color='black', fontsize=12,
                                       ha='center', va='center',
                                       bbox=dict(facecolor=edge_color, edgecolor='black', pad=2), visible=is_visible)
                    obj_bbox = obj.get_bbox_patch()
                    # obj_bbox.set_alpha(alpha_label_visible)
                    obj_bbox.set_visible(is_visible)
                    cls.overlay_rvor_contour_dict['label_objs'][contour_id] = [obj, obj_bbox]

        cls.ax.set_yscale('linear')

    @classmethod
    def display_deck_data(cls):
        vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'),
                             (float('inf'), '*')]
        vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2606 5']
        marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

        valid_datetime, num_all_models, num_models, tc_candidates = cls.get_selected_model_candidates_from_decks()
        start_of_day = valid_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_day = start_of_day.isoformat()
        cls.valid_day = valid_day

        cls.clear_plotted_list()
        lon_lat_tc_records = []
        numc = 0
        day_minus_toggle_3rd_mode = (cls.time_step_opacity[0] == 1.0) and all(q==0.3 for q in cls.time_step_opacity[1:])
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
                        # TODO conditionally hidden by interface (first forecast valid time > user selected horizon (latest first valid time))
                        # valid_time_str = disturbance_candidates[0][1]['valid_time']
                        # valid_time_datetime.fromisoformat(valid_time_str)

                        # check for manually hidden
                        if len(cls.hidden_tc_candidates) != 0 and internal_id in cls.hidden_tc_candidates:
                            continue

                    lat_lon_with_time_step_list = []
                    lon_lat_tuples = []
                    # handle when we are hiding certain time steps or whether the storm itself should be hidden based on time step
                    have_displayed_points = False
                    flip_for_best = False
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
                        candidate_info['valid_time'] = datetime.fromisoformat(valid_time)
                        candidate_info['time_step'] = time_step_int
                        if 'basin' in candidate.keys():
                            candidate_info['basin'] = candidate['basin']
                        else:
                            candidate_info['basin'] = ""

                        # calculate the difference in hours
                        hours_diff = (candidate_info['valid_time'] - datetime.fromisoformat(
                            valid_day)).total_seconds() / 3600
                        # round to the nearest hour
                        hours_diff_rounded = round(hours_diff)
                        candidate_info['hours_after_valid_day'] = hours_diff_rounded

                        if 'roci' in candidate and candidate['roci'] and float(candidate['roci']):
                            candidate_info['roci'] = float(candidate['roci']) / 1000.0
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

                        prev_lat = candidate_info['lat']
                        prev_lon = lon

                        # check whether we want to display it (first valid time limit)
                        for i, (start, end) in list(enumerate(cls.time_step_ranges)):
                            # opacity == 0.3 case (hide all points beyond legend valid time)
                            # for D- we will treat it differently for a-deck mode
                            # this is so we can selectively show only or hide all but old data from best tracks
                            # this flips the function of toggling the legend for D- only for the third mode in adeck mode
                            if i == 0 and day_minus_toggle_3rd_mode:
                                flip_for_best = True
                                continue

                            hours_after = candidate_info['hours_after_valid_day']
                            if start <= hours_after <= end:
                                if cls.time_step_opacity[i] == 1.0:
                                    lat_lon_with_time_step_list.append(candidate_info)
                                    have_displayed_points = True
                                elif have_displayed_points and cls.time_step_opacity[i] == 0.6:
                                    # for D- this effectively hides model data and shows best track data
                                    lat_lon_with_time_step_list.append(candidate_info)
                                elif flip_for_best:
                                    lat_lon_with_time_step_list.append(candidate_info)
                                else:
                                    # opacity == 0.3 case (hide all points beyond legend valid time)
                                    break

                    if lat_lon_with_time_step_list:
                        cls.update_plotted_list(internal_id, lat_lon_with_time_step_list)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                        opacity = 1.0
                        lons = {}
                        lats = {}
                        for point in reversed(lat_lon_with_time_step_list):
                            hours_after = point['hours_after_valid_day']
                            # if start <= time_step <= end:
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
                                    lons[marker].append(point['lon'])
                                    lats[marker].append(point['lat'])
                                    lon_lat_tuples.append([point['lon'], point['lat']])

                        for vmaxmarker in lons.keys():
                            scatter = cls.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker,
                                                      facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                                      s=marker_sizes[vmaxmarker] ** 2, alpha=opacity, antialiased=False)
                            if internal_id not in cls.scatter_objects:
                                cls.scatter_objects[internal_id] = []
                            cls.scatter_objects[internal_id].append(scatter)

                    # do in reversed order so most recent items get rendered on top
                    for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                        line_color = cls.time_step_marker_colors[i]
                        opacity = 1.0
                        strokewidth = 0.5

                        line_segments = []
                        for point in reversed(lat_lon_with_time_step_list):
                            hours_after = point['hours_after_valid_day']
                            if start <= hours_after <= end:
                                if point['prev_lon']:
                                    # Create a list of line segments
                                    line_segments.append([(point['prev_lon'], point['prev_lat']),
                                                          (point['lon'], point['lat'])])
                                    """
                                    plt.plot([point['prev_lon'], point['lon']], [point['prev_lat'], point['lat']],
                                             color=color, linewidth=strokewidth, marker='', markersize = 0, alpha=opacity)
                                    """

                        # Create a LineCollection
                        lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                        # Add the LineCollection to the axes
                        line_collection = cls.ax.add_collection(lc)
                        if internal_id not in cls.line_collection_objects:
                            cls.line_collection_objects[internal_id] = []
                        cls.line_collection_objects[internal_id].append(line_collection)

                    # add the visible points for the track to the m-tree
                    if lon_lat_tuples and len(lon_lat_tuples) > 1:
                        line_string = LineString(lon_lat_tuples)
                        try:
                            cut_lines = cut_line_string_at_antimeridian(line_string)
                        except:
                            print(lon_lat_tuples)
                            print(line_string)
                            print(line_string.coords)
                        for cut_line in cut_lines:
                            record = dict()
                            record['geometry'] = cut_line
                            record['value'] = internal_id
                            lon_lat_tc_records.append(record)

        cls.lon_lat_tc_records = lon_lat_tc_records
        cls.str_tree = STRtree([record["geometry"] for record in lon_lat_tc_records])

        labels_positive = [f' D+{str(i): >2} ' for i in
                           range(len(cls.time_step_marker_colors) - 1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        cls.time_step_legend_objects = []

        for i, (color, label) in enumerate(zip(reversed(cls.time_step_marker_colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i * 20
            time_step_opacity = list(reversed(cls.time_step_opacity))[i]
            if time_step_opacity == 1.0:
                edgecolor = "#FFFFFF"
            elif time_step_opacity == 0.6:
                edgecolor = "#FF77B0"
            else:
                edgecolor = "#A63579"
            legend_object = cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels',
                                             color=list(reversed(cls.time_step_legend_fg_colors))[i],
                                             fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                                           edgecolor=edgecolor,
                                                                                           facecolor=color, alpha=1.0))
            cls.time_step_legend_objects.append(legend_object)

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i * 35
            cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                             fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                            edgecolor='#FFFFFF', facecolor='#000000',
                                                                            alpha=1.0))

        cls.redraw_fig_canvas(stale_bg=True)
        cls.label_adeck_mode.config(text=f"ADECK MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime(
            '%Y-%m-%d') + f". Models: {num_models}/{num_all_models}")
        cls.update_selection_info_label()

    @classmethod
    def display_genesis_data(cls, model_cycle):
        # vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, 'p'), (113.0, 'o'), (137.0, 'D'), (float('inf'), '+')]
        vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'),
                             (float('inf'), '*')]
        vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2605 5']
        marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

        expected_model_names = []
        if cls.genesis_previous_selected != 'GLOBAL-DET':
            is_ensemble = True
            if cls.genesis_previous_selected != 'ALL-TCGEN':
                is_all_tcgen = False
            else:
                is_all_tcgen = True

            expected_model_names = tcgen_models_by_ensemble[cls.genesis_previous_selected]
        else:
            is_ensemble = False
            expected_model_names = global_det_members

        # disturbance_candidates = get_disturbances_from_db(model_name, model_timestamp)
        # now = datetime_utcnow()
        # start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # valid_day = start_of_day.isoformat()
        # model_init_times, tc_candidates = get_tc_candidates_from_valid_time(now.isoformat())
        # model_init_times, tc_candidates = get_tc_candidates_at_or_before_init_time(model_cycle)
        (model_init_times, earliest_recent_ensemble_init_times, model_completed_times, ensemble_completed_times, \
         completed_ensembles, tc_candidates) = get_tc_candidates_at_or_before_init_time(cls.genesis_previous_selected, model_cycle)
        # if model_init_times != last_model_init_times:

        """
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
        """
        model_dates = {}
        most_recent_model_timestamps = {}

        for model_name in expected_model_names:
            model_dates[model_name] = None
            most_recent_model_timestamps[model_name] = datetime.min

        for genesis_source_type, model_init_times, model_completed_times, \
            ensemble_completed_times, completed_ensembles in cls.get_latest_genesis_data_times(is_ensemble=is_ensemble, model_cycle=model_cycle):

            if not model_init_times:
                continue
            for model_name, model_init_time in model_init_times.items():
                if model_name not in expected_model_names:
                    continue

                model_timestamp = datetime.fromisoformat(model_init_time)
                ensemble_name = None
                if model_name in model_name_to_ensemble_name:
                    ensemble_name = model_name_to_ensemble_name[model_name]

                if (ensemble_completed_times and ensemble_name in ensemble_completed_times and
                        len(ensemble_completed_times[ensemble_name]) == 1):
                    #completed_time = ensemble_completed_times[ensemble_name][0]
                    completed_time = completed_ensembles[ensemble_name]
                else:
                    completed_time = None
                if completed_time and ensemble_name and model_timestamp < completed_time:
                    # this is the edge case where there was 0 predictions (empty file) but the ensemble was completed
                    most_recent_model_timestamps[model_name] = completed_time
                    model_dates[model_name] = completed_time.strftime('%d/%HZ')
                elif most_recent_model_timestamps[model_name] == datetime.min:
                    most_recent_model_timestamps[model_name] = model_timestamp
                    if genesis_source_type != 'GLOBAL-DET':
                        model_dates[model_name] = model_timestamp.strftime('%d/%HZ')
                elif most_recent_model_timestamps[model_name] < model_timestamp:
                    most_recent_model_timestamps[model_name] = model_timestamp
                    if genesis_source_type != 'GLOBAL-DET':
                        model_dates[model_name] = model_timestamp.strftime('%d/%HZ')

        # need to handle GLOBAL-DET case and TCGEN case
        # should match model cycle
        missing_models = []
        # filter out any models that aren't available
        for model_name, dt in most_recent_model_timestamps.items():
            if dt == datetime.min:
                missing_models.append(model_name)

        # remove from list completely missing models
        for model_name in missing_models:
            del most_recent_model_timestamps[model_name]

        if is_ensemble:
            # tcgen case
            if is_all_tcgen:
                most_recent_model_cycle = datetime.min
                oldest_model_cycle = datetime.max
                for ens_name, ens_init_time_str in earliest_recent_ensemble_init_times.items():
                    ens_init_time = datetime.fromisoformat(ens_init_time_str)
                    most_recent_model_cycle = max(most_recent_model_cycle, ens_init_time)
                    oldest_model_cycle = min(oldest_model_cycle, ens_init_time)
            else:
                tcgen_ensemble = cls.genesis_previous_selected
                ens_init_time_str = earliest_recent_ensemble_init_times[tcgen_ensemble]
                ens_init_time = datetime.fromisoformat(ens_init_time_str)
                most_recent_model_cycle = ens_init_time
                oldest_model_cycle = ens_init_time
        else:
            most_recent_model_cycle = max(most_recent_model_timestamps.values())
            oldest_model_cycle = min(most_recent_model_timestamps.values())

        start_of_day = oldest_model_cycle.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_day = start_of_day.isoformat()
        cls.valid_day = valid_day
        # model_init_times, tc_candidates = get_tc_candidates_from_valid_time(now.isoformat())
        #model_init_times, tc_candidates = get_tc_candidates_at_or_before_init_time(most_recent_model_cycle)
        model_init_times, earliest_recent_ensemble_init_times, model_completed_times, ensemble_completed_times, completed_ensembles, \
            tc_candidates = get_tc_candidates_at_or_before_init_time(cls.genesis_previous_selected, model_cycle)

        most_recent_timestamp = None

        cls.clear_plotted_list()
        lon_lat_tc_records = []
        wind_field_gpd_dicts = []
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
            elif model_dates[model_name] is None:
                model_dates[model_name] = datetime.fromisoformat(model_timestamp).strftime('%d/%HZ')

            numdisturb = 0
            prev_lat = None
            prev_lon = None
            if disturbance_candidates:
                numdisturb = len(disturbance_candidates)

            # check if it should be hidden
            if numdisturb > 0:
                # TODO conditionally hidden by interface (first forecast valid time > user selected horizon (latest first valid time))
                # valid_time_str = disturbance_candidates[0][1]['valid_time']
                # valid_time_datetime.fromisoformat(valid_time_str)

                # check for manually hidden
                internal_id = numc
                if len(cls.hidden_tc_candidates) != 0 and internal_id in cls.hidden_tc_candidates:
                    continue

                lat_lon_with_time_step_list = []
                lon_lat_tuples = []
                llwtsl_indices = []
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
                    candidate_info['valid_time'] = datetime.fromisoformat(valid_time_str)
                    candidate_info['time_step'] = time_step_int
                    candidate_info['basin'] = candidate['basin']

                    # calculate the difference in hours
                    hours_diff = (candidate_info['valid_time'] - datetime.fromisoformat(
                        valid_day)).total_seconds() / 3600
                    # round to the nearest hour
                    hours_diff_rounded = round(hours_diff)
                    candidate_info['hours_after_valid_day'] = hours_diff_rounded

                    roci = candidate['roci']
                    if roci > 0:
                        candidate_info['roci'] = candidate['roci'] / 1000
                    else:
                        candidate_info['roci'] = None

                    rv850max = candidate['rv850max']
                    if rv850max > 0:
                        candidate_info['rv850max'] = candidate['rv850max'] * 100000
                    else:
                        candidate_info['rv850max'] = None
                    vmaxkt = candidate['vmax10m_in_roci'] * 1.9438452
                    candidate_info['vmax10m_in_roci'] = vmaxkt
                    candidate_info['vmax10m'] = vmaxkt
                    candidate_info['mslp_value'] = candidate['mslp_value']
                    isobar_delta = candidate['closed_isobar_delta']
                    if isobar_delta > 0:
                        candidate_info['closed_isobar_delta'] = candidate['closed_isobar_delta']
                    else:
                        candidate_info['closed_isobar_delta'] = None

                    if 'warm_core' in candidate:
                        candidate_info['warm_core'] = candidate['warm_core']
                    else:
                        candidate_info['warm_core'] = 'Unknown'

                    if 'tropical' in candidate:
                        candidate_info['tropical'] = candidate['tropical']
                    else:
                        candidate_info['tropical'] = 'Unknown'

                    if 'subtropical' in candidate:
                        candidate_info['subtropical'] = candidate['subtropical']
                    else:
                        candidate_info['subtropical'] = 'Unknown'

                    if PROCESS_TCGEN_WIND_RADII:
                        wind_radii_speed_indices = ['34', '50', '64']
                        for wind_speed in wind_radii_speed_indices:
                            wind_radii_key = f'wind_radii_{wind_speed}'
                            if wind_radii_key in candidate:
                                candidate_info[wind_radii_key] = candidate[wind_radii_key]
                            else:
                                candidate_info[wind_radii_key] = None

                    if prev_lon:
                        prev_lon_f = float(prev_lon)
                        if abs(prev_lon_f - lon) > 270:
                            if prev_lon_f < lon:
                                prev_lon = prev_lon_f + 360
                            else:
                                prev_lon = prev_lon_f - 360

                    candidate_info['prev_lat'] = prev_lat
                    candidate_info['prev_lon'] = prev_lon

                    prev_lat = candidate_info['lat']
                    prev_lon = lon

                    # check whether we want to display it (first valid time limit)
                    for i, (start, end) in list(enumerate(cls.time_step_ranges)):
                        hours_after = candidate_info['hours_after_valid_day']
                        if start <= hours_after <= end:
                            if cls.time_step_opacity[i] == 1.0:
                                lat_lon_with_time_step_list.append(candidate_info)
                                have_displayed_points = True
                            elif have_displayed_points and cls.time_step_opacity[i] == 0.6:
                                lat_lon_with_time_step_list.append(candidate_info)
                            else:
                                # opacity == 0.3 case (hide all points beyond legend valid time)
                                break

                if lat_lon_with_time_step_list:
                    cls.update_plotted_list(internal_id, lat_lon_with_time_step_list)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                    opacity = 1.0
                    lons = {}
                    lats = {}
                    for llwtsl_idx, point in reversed(list(enumerate(lat_lon_with_time_step_list))):
                        hours_after = point['hours_after_valid_day']
                        # if start <= time_step <= end:
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
                            lons[marker].append(point['lon'])
                            lats[marker].append(point['lat'])
                            lon_lat_tuples.append([point['lon'], point['lat']])
                            # track with indices are visible so we can construct a visible version
                            llwtsl_indices.append(llwtsl_idx)

                    for vmaxmarker in lons.keys():
                        scatter = cls.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker,
                                                  facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                                  s=marker_sizes[vmaxmarker] ** 2, alpha=opacity, antialiased=False)
                        if internal_id not in cls.scatter_objects:
                            cls.scatter_objects[internal_id] = []
                        cls.scatter_objects[internal_id].append(scatter)

                # do in reversed order so most recent items get rendered on top
                for i, (start, end) in reversed(list(enumerate(cls.time_step_ranges))):
                    line_color = cls.time_step_marker_colors[i]
                    opacity = 1.0
                    strokewidth = 0.5

                    line_segments = []
                    for point in reversed(lat_lon_with_time_step_list):
                        hours_after = point['hours_after_valid_day']
                        if start <= hours_after <= end:
                            if point['prev_lon']:
                                # Create a list of line segments
                                line_segments.append([(point['prev_lon'], point['prev_lat']),
                                                      (point['lon'], point['lat'])])
                                """
                                plt.plot([point['prev_lon'], point['lon']], [point['prev_lat'], point['lat']],
                                         color=color, linewidth=strokewidth, marker='', markersize = 0, alpha=opacity)
                                """

                    # Create a LineCollection
                    lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                    # Add the LineCollection to the axes
                    line_collection = cls.ax.add_collection(lc)
                    if internal_id not in cls.line_collection_objects:
                        cls.line_collection_objects[internal_id] = []
                    cls.line_collection_objects[internal_id].append(line_collection)

                # add the visible points for the track to the m-tree
                if lon_lat_tuples and len(lon_lat_tuples) > 1:
                    line_string = LineString(lon_lat_tuples)
                    try:
                        cut_lines = cut_line_string_at_antimeridian(line_string)
                    except:
                        print(lon_lat_tuples)
                        print(line_string)
                        print(line_string.coords)
                    for cut_line in cut_lines:
                        record = dict()
                        record['geometry'] = cut_line
                        record['value'] = internal_id
                        lon_lat_tc_records.append(record)

                # TODO: handle anti-meridean for WindField geometries
                #  this doesn't seem a too important edge case unless it is breaking
                # get the visible time step list for lat_lon_with_time_step_list
                llwtsl_indices.sort()
                visible_llwtsl = [lat_lon_with_time_step_list[llwtsl_idx] for llwtsl_idx in llwtsl_indices]
                if visible_llwtsl:
                    # TODO MODIFY (ONLY FOR TESTING)
                    path_dicts, gpd_dicts = WindField.get_wind_radii_paths_and_gpds_for_steps(
                        lat_lon_with_time_step_list = visible_llwtsl, wind_radii_selected_list = [34])
                    
                    # TODO: probabilistic wind radii graphic with boundary path (path patch)
                    wind_field_gpd_dicts.append(gpd_dicts)

        cls.lon_lat_tc_records = lon_lat_tc_records
        cls.str_tree = STRtree([record["geometry"] for record in lon_lat_tc_records])

        wind_field_records = {}
        if PROCESS_TCGEN_WIND_RADII:
            for wind_speed in [34, 50, 64]:  # assuming these are the wind speeds
                gpds_to_concat = []
                for gpd in wind_field_gpd_dicts:
                    if wind_speed in gpd:
                        # 0 is the gpd corresponding to the interpolated wind radii Path s
                        gpds_to_concat.append(gpd[wind_speed][0])
                        # 1 is the gpd corresponding to the Path s connecting the interpolations (cone)
                        gpds_to_concat.append(gpd[wind_speed][1])

                if gpds_to_concat and len(gpds_to_concat) > 0:
                    df = pd.concat(gpds_to_concat, ignore_index=True)
                    cls.wind_field_records[wind_speed] = df
                    cls.wind_field_strtrees[wind_speed] = STRtree(df["geometry"].values)
                else:
                    cls.wind_field_records[wind_speed] = None
                    cls.wind_field_strtrees[wind_speed] = None

        labels_positive = [f' D+{str(i): >2} ' for i in
                           range(len(cls.time_step_marker_colors) - 1)]  # Labels corresponding to colors
        labels = [' D-   ']
        labels.extend(labels_positive)

        cls.time_step_legend_objects = []

        for i, (color, label) in enumerate(zip(reversed(cls.time_step_marker_colors), reversed(labels))):
            x_pos, y_pos = 100, 150 + i * 20

            time_step_opacity = list(reversed(cls.time_step_opacity))[i]
            if time_step_opacity == 1.0:
                edgecolor = "#FFFFFF"
            elif time_step_opacity == 0.6:
                edgecolor = "#FF77B0"
            else:
                edgecolor = "#A63579"
            legend_object = cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels',
                                             color=list(reversed(cls.time_step_legend_fg_colors))[i],
                                             fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                                           edgecolor=edgecolor,
                                                                                           facecolor=color, alpha=1.0))
            cls.time_step_legend_objects.append(legend_object)

        # Draw the second legend items inline using display coordinates
        for i, label in enumerate(reversed(vmax_labels)):
            x_pos, y_pos = 160, 155 + i * 35
            cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                             fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                            edgecolor='#FFFFFF', facecolor='#000000',
                                                                            alpha=1.0))

        cls.redraw_fig_canvas(stale_bg=True)

        cls.label_genesis_mode.config(
            text="GENESIS MODE: Start valid day: " + datetime.fromisoformat(valid_day).strftime('%Y-%m-%d'))
        cls.genesis_model_cycle_time = most_recent_model_cycle
        if not is_ensemble:
            # Update model times label
            cls.genesis_models_label.config(
                text=f"Models: GFS [{model_dates['GFS']}], ECM[{model_dates['ECM']}], NAV[{model_dates['NAV']}], CMC[{model_dates['CMC']}]")
        elif cls.genesis_previous_selected != 'GLOBAL-DET':
            ens_dates = []
            for ens_name, init_time in earliest_recent_ensemble_init_times.items():
                ens_dates.append((ens_name, datetime.fromisoformat(init_time).strftime('%d/%HZ')))

            model_labels_str = ''
            model_labels = []
            for ens_name, ens_min_date in ens_dates:
                model_labels.append(f'{ens_name} [{ens_min_date}]')

            join_model_labels = ", ".join(model_labels)
            if join_model_labels:
                model_labels_str = f'Models: {join_model_labels}'

            cls.genesis_models_label.config(text=model_labels_str)

        cls.update_selection_info_label()

    @classmethod
    def display_map(cls):
        cls.scatter_objects = {}
        cls.line_collection_objects = {}
        # reset for circle patch (hover) blitting
        cls.background_for_blit = None
        cls.circle_handle = None
        cls.last_circle_lat = None
        cls.last_circle_lon = None
        cls.mean_track_obj = None
        cls.blit_show_mean_track = False

        if cls.canvas:
            cls.canvas.get_tk_widget().pack_forget()

        # Adjust figure size to fill the screen
        screen_width = cls.root.winfo_screenwidth()
        screen_height = cls.root.winfo_screenheight()

        if cls.fig:
            try:
                plt.close(cls.fig)
                cls.measure_tool.reset_measurement()
            except:
                pass

        cls.fig = plt.figure(figsize=(screen_width / CHART_DPI, screen_height / CHART_DPI), dpi=CHART_DPI,
                              facecolor='black')

        # cls.fig.add_subplot(111, projection=ccrs.PlateCarree())
        cls.ax = plt.axes(projection=ccrs.PlateCarree())
        cls.measure_tool.changed_extent()
        cls.ax.set_anchor("NW")

        cls.ax.autoscale_view(scalex=False, scaley=False)
        cls.ax.set_yscale('linear')
        cls.ax.set_facecolor('black')

        if LOAD_NETCDF and DISPLAY_NETCDF is not None:
            # Draw the netcdf data for SSTs, or TCHP, or Depth of the 26th isobar (D26)
            plotters = []
            plotter = None
            if DISPLAY_NETCDF == 'sst':
                plotter = cls.plotter_sst
                abs_scale_min = -5
                abs_scale_max = 35
                bins = [(-5, 0), (0, 5), (5, 10), (10, 20), (20, 25), (25, 26), (26, 27),
                        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 99)]

                # fine bins
                if FINE_SST_BINS:
                    abs_scale_min = 28
                    abs_scale_max = 32
                    edges = np.linspace(28, 32, 21)
                    bins = []
                    for i in list(range(len(edges))):
                        if (i+1) != len(edges):
                            bins.append((edges[i], edges[i+1]))
                        
            elif DISPLAY_NETCDF in ['tchp', 'd26']:
                plotter = cls.plotter_tchp_d26
                abs_scale_min = 0
                abs_scale_max = 300
                bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                          (90, 100), (100, 125), (125, 150), (150, 175), (175, 200),
                          (200, 225), (225, 250), (250, 275), (275, 300), (300, 999)]

            elif DISPLAY_NETCDF in ['ohc', 'iso26C']:
                plotters = cls.plotters_ohc
                abs_scale_min = 0
                abs_scale_max = 300
                bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                        (90, 100), (100, 125), (125, 150), (150, 175), (175, 200),
                        (200, 225), (225, 250), (250, 275), (275, 300), (300, 999)]

            # Plot with clipping, binning, and opacity

            if not BIN_NETCDF_DATA:
                bins = None

            if plotter:
                plotter.plot_data(ax=cls.ax, dataset=DISPLAY_NETCDF, data_min=None, data_max=None, bins=bins, opacity=1,
                                  scale_min=abs_scale_min, scale_max=abs_scale_max)
            elif plotters:
                for plotter in plotters:
                    plotter.plot_data(ax=cls.ax, dataset=DISPLAY_NETCDF, data_min=None, data_max=None, bins=bins,
                                      opacity=1, scale_min=abs_scale_min, scale_max=abs_scale_max)


        # Draw a rectangle patch around the subplot to visualize its boundaries
        extent = cls.ax.get_extent()

        rect = Rectangle((-180, -90), 360, 180, linewidth=2, edgecolor='white', facecolor='none')
        cls.ax.add_patch(rect)

        # Adjust aspect ratio of the subplot
        cls.ax.set_aspect('equal')

        # Draw red border around figure
        # cls.fig.patch.set_edgecolor('red')
        # cls.fig.patch.set_linewidth(2)

        # cls.ax.stock_img()
        cls.ax.set_extent(cls.global_extent)

        # Add state boundary lines
        states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces',
            scale='50m',
            facecolor='none'
        )

        cls.ax.add_feature(states, edgecolor='gray')

        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none'
        )
        cls.ax.add_feature(country_borders, edgecolor='white', linewidth=0.5)
        cls.ax.add_feature(cfeature.COASTLINE, edgecolor='yellow', linewidth=0.5)

        cls.update_axes()

        cls.canvas = FigureCanvasTkAgg(cls.fig, master=cls.canvas_frame)
        cls.redraw_app_canvas()
        cls.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        cls.cid_press = cls.canvas.mpl_connect("button_press_event", cls.on_click)
        cls.cid_release = cls.canvas.mpl_connect("button_release_event", cls.on_release)
        cls.cid_motion = cls.canvas.mpl_connect("motion_notify_event", cls.on_motion)
        cls.cid_key_press = cls.canvas.mpl_connect('key_press_event', cls.on_key_press)
        cls.cid_key_release = cls.canvas.mpl_connect('key_release_event', cls.on_key_release)

        # we only know how big the canvas frame/plot is after drawing/packing, and we need to wait until drawing to fix the size
        cls.canvas_frame.update_idletasks()
        # fix the size to the canvas frame
        frame_hsize = float(cls.canvas_frame.winfo_width()) / CHART_DPI
        frame_vsize = float(cls.canvas_frame.winfo_height()) / CHART_DPI
        h = [Size.Fixed(0), Size.Fixed(frame_hsize)]
        v = [Size.Fixed(0), Size.Fixed(frame_vsize)]
        divider = Divider(cls.fig, (0, 0, 1, 1), h, v, aspect=False)
        cls.ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        cls.clear_storm_extrema_annotations()
        AnnotatedCircles.changed_extent()
        SelectionLoops.changed_extent()
        cls.update_selection_info_label()
        cls.measure_tool.changed_extent()
        cls.redraw_app_canvas()
        cls.display_custom_boundaries()
        cls.display_custom_overlay()
        cls.axes_size = cls.ax.get_figure().get_size_inches() * cls.ax.get_figure().dpi
        # make sure focus is always on map after updating
        cls.set_focus_on_map()

    @classmethod
    def enter_adt_url(cls):
        # defunct, kept for reference, or in case something breaks
        url = simpledialog.askstring("Enter (OSPO) ADT URL", "URL:")
        if url and url.strip() is not None:
            cls.display_adt(url)
        else:
            cls.set_focus_on_map()

    @classmethod
    def get_adt(cls):
        #cls.enter_adt_url()
        ospo_urls, cimss_urls = get_adt_urls()
        if ospo_urls is not None:
            for ospo_url in ospo_urls:
                cls.display_adt(ospo_url)
        if cimss_urls is not None:
            for cimss_url in cimss_urls:
                cls.display_adt(cimss_url)

    @classmethod
    def get_contour_min_span_deg(cls):
        global MINIMUM_CONTOUR_PX_X
        global MINIMUM_CONTOUR_PX_Y
        # Get the current extent in degrees
        extent = cls.ax.get_extent(ccrs.PlateCarree())
        min_lon, max_lon, min_lat, max_lat = extent
        span_lon_extent = max_lon - min_lon
        span_lat_extent = max_lat - min_lat
        # Determine the minimum span in degrees for display
        ax_width, ax_height = cls.ax.get_figure().get_size_inches() * cls.ax.get_figure().dpi
        lon_pixel_ratio = span_lon_extent / ax_width
        lat_pixel_ratio = span_lat_extent / ax_height
        minimum_contour_extent = [MINIMUM_CONTOUR_PX_X, MINIMUM_CONTOUR_PX_Y]  # Minimum in pixels for lon, lat
        min_span_lon_deg = minimum_contour_extent[0] * lon_pixel_ratio
        min_span_lat_deg = minimum_contour_extent[1] * lat_pixel_ratio
        return extent, min_span_lat_deg, min_span_lon_deg

    # generator to return data on latest model/ensemble times (yields once per genesis source type)
    @classmethod
    def get_latest_genesis_data_times(cls, is_ensemble=None, model_cycle=None):
        if is_ensemble is not None:
            if is_ensemble:
                genesis_source_types = ['ALL-TCGEN']
            else:
                genesis_source_types = ['GLOBAL-DET']
        else:
            genesis_source_types = ['GLOBAL-DET', 'ALL-TCGEN']

        for genesis_source_type in genesis_source_types:
            conn = None
            model_names = list(model_data_folders_by_model_name.keys())
            model_init_times = {}

            model_completed_times = {}
            # list of completed times per ensemble for member completion for a recent cycle (may spread across more than one cycle)
            ensemble_completed_times = {}
            # get whether the ensemble is complete at a single cycle (only one init_time and ens_status is complete)
            completed_ensembles = {}

            ensemble_names = []
            if genesis_source_type == 'GLOBAL-DET':
                db_file_path = tc_candidates_db_file_path
                # filter out all model_names from tcgen and keep all models from own tracker
                model_names = [model_name for model_name in model_names if model_name[-5:] != 'TCGEN']
            else:
                # 'ALL-TCGEN' case
                db_file_path = tc_candidates_tcgen_db_file_path
                # filter out all model_names from own tracker and keep all tcgen models
                ensemble_names = [ensemble_name for ensemble_name in tcgen_models_by_ensemble.keys() if
                                  ensemble_name != 'ALL-TCGEN']
                model_names = tcgen_models_by_ensemble['ALL-TCGEN']

            try:
                # Connect to the SQLite database
                conn = sqlite3.connect(db_file_path)
                cursor = conn.cursor()

                for model_name in model_names:
                    if model_cycle is not None:
                        cursor.execute(
                            'SELECT DISTINCT init_date, completed_date FROM completed WHERE model_name = ? AND init_date <= ? ORDER BY init_date DESC LIMIT 1',
                            (model_name, datetime.isoformat(model_cycle)))
                    else:
                        cursor.execute(
                            'SELECT DISTINCT init_date, completed_date FROM completed WHERE model_name = ? ORDER BY init_date DESC LIMIT 1',
                            (model_name, ))
                    results = cursor.fetchall()
                    if results:
                        for row in results:
                            init_time = row[0]
                            completed_time = datetime.fromisoformat(row[1])
                            model_init_times[model_name] = init_time
                            model_completed_times[model_name] = completed_time

                for ensemble_name in ensemble_names:
                    ensemble_init_times = set()
                    ensemble_completed_times_by_init_date = {}
                    for model_name, model_init_time in model_init_times.items():
                        if model_name not in model_name_to_ensemble_name:
                            print("Warning could not find ensemble name for model name (skipping):", model_name)
                            continue
                        if ensemble_name == model_name_to_ensemble_name[model_name]:
                            ensemble_init_times.add(model_init_time)
                            if model_init_time not in ensemble_completed_times_by_init_date:
                                ensemble_completed_times_by_init_date[model_init_time] = []
                            ensemble_completed_times_by_init_date[model_init_time].append(model_completed_times[model_name])
                    if not ensemble_init_times:
                        continue

                    num_init_time_ens_is_complete = 0
                    max_ensemble_init_time = '0001-01-01T00:00:00'
                    for ensemble_init_time in ensemble_init_times:
                        cursor.execute(
                            f'SELECT completed FROM ens_status WHERE init_date = ? AND ensemble_name = ? ORDER BY init_date DESC LIMIT 1',
                            (ensemble_init_time, ensemble_name))
                        results = cursor.fetchall()
                        ens_is_completed = 0
                        if results:
                            for row in results:
                                ens_is_completed = row[0]

                        if ensemble_name not in ensemble_completed_times:
                            ensemble_completed_times[ensemble_name] = []
                        if ens_is_completed:
                            ensemble_completed_time = max(ensemble_completed_times_by_init_date[ensemble_init_time])
                            if len(ensemble_completed_times[ensemble_name]) > 0:
                                # skip over old models that have members with 0 predictions yet the ensemble is complete
                                if ensemble_init_time > max_ensemble_init_time:
                                    ensemble_completed_times[ensemble_name] = [ensemble_completed_time]
                                    max_ensemble_init_time = ensemble_init_time
                                    num_init_time_ens_is_complete = 1
                            else:
                                ensemble_completed_times[ensemble_name].append(ensemble_completed_time)
                                max_ensemble_init_time = max(max_ensemble_init_time, ensemble_init_time)
                                num_init_time_ens_is_complete += int(ens_is_completed)

                    if num_init_time_ens_is_complete == 1:
                        completed_ensembles[ensemble_name] = ensemble_completed_times[ensemble_name][0]

            except sqlite3.Error as e:
                print(f"SQLite error (get_latest_genesis_data_times): {e}")
            finally:
                if conn:
                    conn.close()

            # completed times are datetimes while init times are strings
            yield genesis_source_type, model_init_times, model_completed_times, ensemble_completed_times, completed_ensembles

    # uses str_tree and the SelectionLoops to get a list of storm internal ids that are in the lassod boundary
    #   this will only cover storms that have been rendered through display_*
    #   as those are the only ones we make internal ids for
    @classmethod
    def get_selected_internal_storm_ids(cls):
        selection_surface_polygons = SelectionLoops.get_polygons()

        result_items = set()
        lon_lat_tc_records = cls.lon_lat_tc_records
        str_tree = cls.str_tree
        if not lon_lat_tc_records:
            return
        tc_internal_ids = [record["value"] for record in lon_lat_tc_records]
        # internal id is an integer for genesis, and a tuple for abdeck (storm id, model id)
        for query_polygon in selection_surface_polygons:
            result = str_tree.query(query_polygon, predicate='intersects')
            if result.size > 0:
                result_ids = [tc_internal_ids[i] for i in result]
                for result_id in result_ids:
                    # skip internal ids that are not visible
                    if len(cls.hidden_tc_candidates) != 0:
                        if result_id not in cls.hidden_tc_candidates:
                            result_items.add(result_id)
                    else:
                        result_items.add(result_id)

        return list(result_items)

    # get model data from adeck, plus bdeck and tcvitals
    @classmethod
    def get_selected_model_candidates_from_decks(cls):
        selected_models = cls.get_selected_deck_model_list()
        # reference current datetime since we are checking for old invests
        valid_datetime = datetime_utcnow()
        earliest_model_valid_datetime = valid_datetime
        selected_model_data = {}
        actual_models = set()
        all_models = set()
        # assume invests older than this are different storms (must have correct datetime on computer)
        max_time_delta = timedelta(days=4)
        if cls.adeck and cls.adeck.keys():
            for storm_atcf_id in cls.adeck.keys():
                if storm_atcf_id in cls.adeck and cls.adeck[storm_atcf_id]:
                    for model_id, models in cls.adeck[storm_atcf_id].items():
                        if models:
                            model_is_from_old_invest = True
                            for valid_time, data in models.items():
                                dt = datetime.fromisoformat(valid_time)
                                # 9X are invests... may not have yet model data for new invest (and identify wrong date for old one)
                                is_invest = (int(storm_atcf_id[2:3]) == 9)
                                tcvitals_model_diff = earliest_model_valid_datetime - dt
                                is_old_invest = is_invest and (tcvitals_model_diff > max_time_delta)
                                model_is_from_old_invest = model_is_from_old_invest and is_old_invest
                                if dt < earliest_model_valid_datetime and not(is_old_invest):
                                    earliest_model_valid_datetime = dt
                            if not model_is_from_old_invest:
                                all_models.add(model_id)
                                if model_id in selected_models:
                                    if storm_atcf_id not in selected_model_data.keys():
                                        selected_model_data[storm_atcf_id] = {}
                                    selected_model_data[storm_atcf_id][model_id] = models
                                    actual_models.add(model_id)

        if cls.bdeck:
            for storm_atcf_id in cls.bdeck.keys():
                if storm_atcf_id in cls.bdeck and cls.bdeck[storm_atcf_id]:
                    for model_id, models in cls.bdeck[storm_atcf_id].items():
                        if models:
                            if model_id in selected_models:
                                if storm_atcf_id not in selected_model_data.keys():
                                    selected_model_data[storm_atcf_id] = {}
                                selected_model_data[storm_atcf_id][model_id] = models

        # tcvitals
        if cls.recent_storms:
            # max time delta for invests, to mark models corresponding to old invests
            for storm_atcf_id, data in cls.recent_storms.items():
                valid_date_str = data['valid_time']
                if storm_atcf_id not in selected_model_data.keys():
                    selected_model_data[storm_atcf_id] = {}
                selected_model_data[storm_atcf_id]['TCVITALS'] = {valid_date_str: data}
                # 9X are invests... may not have yet model data for new invest (and identify wrong date for old one)
                is_invest = (int(storm_atcf_id[2:3]) == 9)
                dt = datetime.fromisoformat(valid_date_str)
                tcvitals_model_diff = earliest_model_valid_datetime - dt
                is_old_invest = is_invest and (tcvitals_model_diff > max_time_delta)
                if dt < earliest_model_valid_datetime and not is_old_invest:
                    earliest_model_valid_datetime = dt

        # case where there are only invests
        if valid_datetime == datetime.max:
            valid_datetime = datetime_utcnow()

        return earliest_model_valid_datetime, len(all_models), len(actual_models), selected_model_data

    @classmethod
    def get_selected_deck_model_list(cls):
        selected_text = cls.adeck_selected.get()
        if selected_text == 'ALL':
            return included_intensity_models
        elif selected_text == 'GLOBAL':
            return global_models
        elif selected_text == 'GEFS-MEMBERS':
            return gefs_members_models
        elif selected_text == 'GEFS-ATCF':
            return gefs_atcf_members
        elif selected_text == 'GEPS-ATCF':
            return geps_atcf_members
        elif selected_text == 'FNMOC-ATCF':
            return fnmoc_atcf_members
        elif selected_text == 'ALL-ATCF':
            return all_atcf_ens_members
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

    # Hide (selected) tracks by field
    @classmethod
    def hide_by_field(cls, by_all_all = False, by_all_any = False, by_any_all = False, by_any_any = False):
        # Given selected tracks (selectop loop) or all the tracks (if no selection loops):
        # When there is no field keys and by_all_all, we treat the valid_time as the field key essentially (time range must overlap for all points)
        # When there are field keys, we interpolate and filter tracks on the valid_time, then by field depending on whether
        # (by_all_all) for all track points (in the time range) (per candidate), all fields must satisfy
        # (by_all_any) for all track poiints (in the time range) (per candidate), any field must satisfy
        # (by_any_all) for any track points (in the time range) (per candidate), all fields must satisfy
        # (by_all_any) for any track poiints (in the time range) (per candidate), any field must satisfy
        # We then take the disjoint union of the constrained candidates to get the set of candidates to hide

        field_data = {}
        any_valid = False
        for key, vars in cls.field_vars.items():
            val = vars['value'].get()
            tol = vars['tolerance'].get()
            min = vars['min'].get()
            max = vars['max'].get()
            fd = {}
            if key == 'vmax10m_ms':
                unit_conv = 1.9438452
                key = 'vmax10m'
            else:
                unit_conv = 1.0

            if len(str(val).strip()) > 0:
                conv_value = float(val) * unit_conv
                if len(str(tol).strip()) > 0:
                    tol = float(tol) * unit_conv
                else:
                    tol = 0.0

                fd['min'] = conv_value - tol
                fd['max'] = conv_value + tol
            else:
                any_min_max = False
                if len(str(min).strip()) > 0:
                    fd['min'] = float(min) * unit_conv
                    any_min_max = True
                if len(str(max).strip()) > 0:
                    fd['max'] = float(max) * unit_conv
                    any_min_max = True

                # only add the implicit min/max if we have one of the values as an input
                if any_min_max:
                    if 'min' not in fd:
                        fd['min'] = -np.inf
                    if 'max' not in fd:
                        fd['max'] = np.inf

            if len(fd.keys()) > 0:
                field_data[key] = fd
                any_valid = True

        time_data = {}
        for time_type, time_type_vars in cls.time_vars.items():
            time_data[time_type] = {}
            for k, v in time_type_vars.items():
                val = v.get().strip()
                if val != '':
                    time_data[time_type][k] = int(val)

            if time_type == 'point':
                expected_num_keys = 7
            else:
                expected_num_keys = 6

            if len(time_data[time_type].keys()) == expected_num_keys:
                any_valid = True
            elif time_type == 'point' and len(time_data[time_type].keys()) == 6 and 'tolerance' not in time_data[time_type].keys():
                time_data[time_type]['tolerance'] = 0.0
                any_valid = True
            else:
                # delete the constraint if there are empty (required) fields
                del time_data[time_type]

            # can only have a point or range for time type (in case user specifies all three)
            if 'point' in time_data.keys() and len(time_data.keys()) > 1:
                if 'min' in time_data.keys():
                    del time_data[time_type]['min']
                if 'max' in time_data.keys():
                    del time_data[time_type]['max']

        if not any_valid:
            return

        filtered_candidates = []
        if not SelectionLoops.is_empty():
            internal_ids = cls.get_selected_internal_storm_ids()
            num_tracks = 0
            if internal_ids:
                num_tracks = len(internal_ids)

            if num_tracks == 0:
                return

            # Aggregate candidates by models
            filtered_candidates = [(iid, tc) for iid, tc in cls.plotted_tc_candidates if
                                   iid in internal_ids]
        else:
            # Aggregate candidates by models
            for candidate in cls.plotted_tc_candidates:
                iid, tc = candidate
                if iid not in cls.hidden_tc_candidates:
                    filtered_candidates.append(candidate)

        if len(filtered_candidates) == 0:
            return

        # from filtered candidates are now the list of candidates we will find a list that meet the constraints
        # after finding the candidates that meet the constraints, we will hide the disjoint union of the two

        field_keys = field_data.keys()
        synoptic_only = False
        synpotic_point = False
        synoptic_min = False
        synoptic_max = False
        interpolated_time = False
        hide_by_time = False
        # first check if we need to interpolate the tracks (for time constraint)
        time_filtered_candidates = []
        if len(time_data.keys()) == 0:
            # no filtering by time
            time_filtered_candidates = filtered_candidates
        elif len(time_data.keys()) > 0:
            start_dt = datetime.min
            end_dt = datetime.max
            if 'point' in time_data.keys():
                k = 'point'
                dt = datetime(
                    year=time_data[k]['year'],
                    month=time_data[k]['month'],
                    day=time_data[k]['day'],
                    hour=time_data[k]['hour'],
                    minute=time_data[k]['minute'],
                    second=time_data[k]['second']
                )
                start_dt = dt - timedelta(seconds=time_data[k]['tolerance'])
                end_dt = dt + timedelta(seconds=time_data[k]['tolerance'])
            else:
                if 'min' in time_data.keys():
                    k = 'min'
                    start_dt = datetime(
                        year=time_data[k]['year'],
                        month=time_data[k]['month'],
                        day=time_data[k]['day'],
                        hour=time_data[k]['hour'],
                        minute=time_data[k]['minute'],
                        second=time_data[k]['second']
                    )
                if 'max' in time_data.keys():
                    k = 'max'
                    end_dt = datetime(
                        year=time_data[k]['year'],
                        month=time_data[k]['month'],
                        day=time_data[k]['day'],
                        hour=time_data[k]['hour'],
                        minute=time_data[k]['minute'],
                        second=time_data[k]['second']
                    )

            no_field_keys = len(field_keys) == 0
            all_must_overlap = no_field_keys and by_all_all
            time_filtered_candidates = PartialInterpolationTrackFilter.filter_by_time(filtered_candidates,
                                                                                      start_dt, end_dt, field_keys,
                                                                                      all_must_overlap)

        # have candidates meeting time constraint, now filter by candidates meeting field_keys constraint
        if len(time_filtered_candidates) > 0 and len(field_keys) > 0:
            # filter now by field_keys
            # only one of the by_X_X can be True
            field_filtered_candidates = PartialInterpolationTrackFilter.filter_by_field(
                time_filtered_candidates,
                field_data,
                by_all_all=by_all_all,
                by_all_any=by_all_any,
                by_any_all=by_any_all,
                by_any_any=by_any_any
            )
        else:
            field_filtered_candidates = time_filtered_candidates

        # Get the disjoint union
        field_filtered_ids = {internal_id for internal_id, _ in field_filtered_candidates}
        filtered_ids = {internal_id for internal_id, _ in filtered_candidates}

        # Return the symmetric difference, which gives us the disjoint union of ids
        internal_ids_to_hide = field_filtered_ids.symmetric_difference(filtered_ids)

        if len(internal_ids_to_hide) > 0:
            # hide the candidates
            cls.hide_tc_candidates(internal_ids_to_hide)

    # Function to set the current time for a specific time variable
    @classmethod
    def hide_set_current_time(cls, time_var):
        current_time = datetime_utcnow()
        time_var['year'].set(current_time.year)
        time_var['month'].set(current_time.month)
        time_var['day'].set(current_time.day)
        time_var['hour'].set(current_time.hour)
        time_var['minute'].set(current_time.minute)
        time_var['second'].set(current_time.second)

    # hide selected, or show all if none selected; if candidates_to_hide is passed, only hide those candidates
    @classmethod
    def hide_tc_candidates(cls, candidates_to_hide = None):
        # two modes now
        # when no selection loops present: hide mouse hovered storm, or unhide all
        # when selection loops present: hide mouse hovered storm, or if no hover, hide all tracks in selection
        total_num_overlapped_points = len(cls.nearest_point_indices_overlapped)
        if total_num_overlapped_points == 0 or candidates_to_hide:
            if not SelectionLoops.is_empty() or candidates_to_hide:
                if candidates_to_hide:
                    to_hide_internal_ids = candidates_to_hide
                else:
                    to_hide_internal_ids = cls.get_selected_internal_storm_ids()

                for to_hide_internal_id in to_hide_internal_ids:
                    #cursor_internal_id, tc_index, tc_point_index = cursor_point_index
                    cls.hidden_tc_candidates.add(to_hide_internal_id)

                    if cls.scatter_objects:
                        for internal_id, scatters in cls.scatter_objects.items():
                            if to_hide_internal_id == internal_id:
                                try:
                                    for scatter in scatters:
                                        scatter.set_visible(False)
                                except:
                                    traceback.print_exc()

                    if cls.line_collection_objects:
                        for internal_id, line_collections in cls.line_collection_objects.items():
                            if to_hide_internal_id == internal_id:
                                try:
                                    for line_collection in line_collections:
                                        line_collection.set_visible(False)
                                except:
                                    traceback.print_exc()

                    if cls.annotated_circle_objects:
                        if to_hide_internal_id in cls.annotated_circle_objects:
                            for point_index, annotated_circle in cls.annotated_circle_objects[to_hide_internal_id].items():
                                try:
                                    annotated_circle.set_visible(False)
                                except:
                                    traceback.print_exc()

                cls.update_selection_info_label()
                cls.redraw_fig_canvas(stale_bg=True)
            # unhide all
            elif len(cls.hidden_tc_candidates) > 0:
                if cls.scatter_objects:
                    for internal_id, scatters in cls.scatter_objects.items():
                        try:
                            for scatter in scatters:
                                scatter.set_visible(True)
                        except:
                            traceback.print_exc()

                if cls.line_collection_objects:
                    for internal_id, line_collections in cls.line_collection_objects.items():
                        try:
                            for line_collection in line_collections:
                                line_collection.set_visible(True)
                        except:
                            traceback.print_exc()

                if cls.annotated_circle_objects:
                    for internal_id, point_index_circles in cls.annotated_circle_objects.items():
                        for point_index, annotated_circle in point_index_circles:
                            try:
                                if annotated_circle:
                                    annotated_circle.set_visible(True)
                            except:
                                traceback.print_exc()

                cls.hidden_tc_candidates = set()
                (lon, lat) = cls.last_cursor_lon_lat
                cls.update_labels_for_mouse_hover(lat=lat, lon=lon)
                cls.update_selection_info_label()
                cls.redraw_fig_canvas(stale_bg=True)
        else:
            num, cursor_point_index = cls.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            if cursor_point_index:
                cursor_internal_id, tc_index, tc_point_index = cursor_point_index
                cls.hidden_tc_candidates.add(cursor_internal_id)

                if cls.scatter_objects:
                    for internal_id, scatters in cls.scatter_objects.items():
                        if cursor_internal_id == internal_id:
                            try:
                                for scatter in scatters:
                                    scatter.set_visible(False)
                            except:
                                traceback.print_exc()

                if cls.line_collection_objects:
                    for internal_id, line_collections in cls.line_collection_objects.items():
                        if cursor_internal_id == internal_id:
                            try:
                                for line_collection in line_collections:
                                    line_collection.set_visible(False)
                            except:
                                traceback.print_exc()

                if cls.annotated_circle_objects:
                    if cursor_internal_id in cls.annotated_circle_objects:
                        for point_index, annotated_circle in cls.annotated_circle_objects[cursor_internal_id].items():
                            try:
                                annotated_circle.set_visible(False)
                            except:
                                traceback.print_exc()

                (lon, lat) = cls.last_cursor_lon_lat
                cls.update_labels_for_mouse_hover(lat=lat, lon=lon)
                cls.update_selection_info_label()
                cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def init(cls, root):
        cls.root = root
        cls.root.title("tcviewer")
        cls.root.attributes('-fullscreen', True)
        cls.root.configure(bg="black")
        # bind main app keys
        cls.root.bind("p", cls.take_screenshot)
        cls.root.bind("l", cls.toggle_rvor_labels)
        cls.root.bind("v", cls.toggle_rvor_contours)
        # cls.root.bind("v", cls.toggle_rvor_contours)
        cls.root.bind("V", cls.show_rvor_dialog)

        cls.root.bind("a", cls.show_analysis_dialog)

        cls.root.bind('z', cls.zoom_to_basin_dialog)
        cls.root.bind('b', cls.select_basin_dialog)

        cls.root.bind('4', cls.set_netcdf_display_sst)
        cls.root.bind('5', cls.set_netcdf_display_ohc)
        cls.root.bind('6', cls.set_netcdf_display_iso26C)
        cls.root.bind('r', cls.set_netcdf_display_none)

        cls.adeck_selected = tk.StringVar()
        cls.genesis_selected = tk.StringVar()

        # settings
        cls.load_settings()

        cls.init_rvor_contour_dict()

        # setup widges
        cls.create_widgets()
        # setup tools
        cls.measure_tool = MeasureTool()

        # display initial map
        cls.display_map()

    @classmethod
    def init_rvor_contour_dict(cls):
        cls.overlay_rvor_contour_dict = defaultdict(dict)
        # noinspection PyTypeChecker
        cls.overlay_rvor_contour_dict['ids'] = set()
        # noinspection PyTypeChecker
        cls.overlay_rvor_contour_dict['renderable_ids'] = set()
        cls.overlay_rvor_contour_dict['contour_span_lons'] = {}
        cls.overlay_rvor_contour_dict['contour_span_lats'] = {}
        cls.overlay_rvor_contour_dict['contour_objs'] = defaultdict(dict)
        cls.overlay_rvor_contour_dict['label_objs'] = defaultdict(dict)

    @classmethod
    def latest_genesis_cycle(cls):
        cls.update_genesis_data_staleness()
        model_cycles = get_tc_model_init_times_relative_to(datetime.now(), cls.genesis_previous_selected)
        if model_cycles['next'] is None:
            model_cycle = model_cycles['at']
        else:
            model_cycle = model_cycles['next']

        if model_cycle:
            # clear map
            cls.redraw_map_with_data(model_cycle=model_cycle)

    # rehide hidden candidates saved in slot
    @classmethod
    def load_hidden(cls, slot=None):
        cls.hide_tc_candidates(cls.saved_hidden[slot])

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
            global ANALYSIS_TZ
            with open('settings_tcviewer.json', 'r') as f:
                try:
                    settings = json.load(f)
                except:
                    print("Could not load JSON from settings_tcviewer.json. Using defuaults.")
                    return

                for key, val in settings.items():
                    if key == 'annotation_label_options':
                        DISPLAYED_FUNCTIONAL_ANNOTATIONS = [option for option in displayed_functional_annotation_options
                                                            if option in val]
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

    @classmethod
    def on_analysis_dialog_close(cls, dialog):
        cls.analysis_dialog_open = False
        dialog.destroy()
        # focus back on app
        cls.set_focus_on_map()

    @classmethod
    def on_hide_by_field_dialog_close(cls, dialog):
        cls.hide_by_field_dialog_open = False
        dialog.destroy()
        # focus back on app
        cls.set_focus_on_map()

    @classmethod
    def on_rvor_dialog_close(cls, dialog):
        cls.rvor_dialog_open = False
        dialog.destroy()
        # focus back on app
        cls.set_focus_on_map()

    @classmethod
    def next_genesis_cycle(cls):
        nav_time = cls.genesis_model_cycle_time_navigate
        cls.update_genesis_data_staleness()
        if cls.genesis_model_cycle_time is None:
            cls.genesis_model_cycle_time = datetime.now()
            model_cycles = get_tc_model_init_times_relative_to(cls.genesis_model_cycle_time, cls.genesis_previous_selected)
            if model_cycles['next'] is None:
                model_cycle = model_cycles['at']
            else:
                model_cycle = model_cycles['next']
        else:
            model_cycles = get_tc_model_init_times_relative_to(nav_time, cls.genesis_previous_selected)
            if model_cycles['next'] is None:
                # nothing after current cycle
                return
            model_cycle = model_cycles['next']

        if model_cycle:
            if model_cycle != cls.genesis_model_cycle_time:
                cls.genesis_model_cycle_time_navigate = model_cycle
                cls.redraw_map_with_data(model_cycle=model_cycle)

    @classmethod
    def on_key_press(cls, event):
        if cls.any_modal_open():
            return

        if event.key == 't':
            cls.get_adt()
            return

        if event.key == 'k':
            cls.show_hide_by_field_dialog()
            return

        if event.key == 'm':
            cls.blit_show_mean_track = not cls.blit_show_mean_track
            cls.blit_mean_track(init=cls.blit_show_mean_track)
            return

        if event.key == '1':
            cls.save_hidden(slot=1)
            return
        if event.key == '2':
            cls.save_hidden(slot=2)
            return
        if event.key == '8':
            cls.load_hidden(slot=1)
            return
        if event.key == '9':
            cls.load_hidden(slot=2)
            return

        if event.key == 'escape':
            # abort a zoom
            if cls.zoom_selection_box is not None:
                cls.zoom_selection_box.destroy()
                cls.zoom_selection_box = None
            # cls.fig.canvas.draw()
            cls.blit_circle_patch()
            return

        if event.key == '0':
            cls.zoom_out(max_zoom=True)
            return

        if event.key == '-':
            cls.zoom_out(step_zoom=True)
            return

        if event.key == '=':
            cls.zoom_in(step_zoom=True)
            return

        if event.key == 'n':
            cls.cycle_to_next_overlapped_point()
            return

        if event.key == 's':
            cls.toggle_selection_loop_mode()
            return

        if event.key == 'h':
            if cls.selection_loop_mode:
                SelectionLoops.toggle_visible()
            else:
                cls.hide_tc_candidates()
            return

        if event.key == 'i':
            if not cls.selection_loop_mode:
                cls.print_track_stats()
            return

        if event.key == 'x':
            # annotate storm extrema
            cls.annotate_storm_extrema()
            return

        if event.key == 'c':
            if cls.selection_loop_mode:
                SelectionLoops.clear()
                cls.update_selection_info_label()
            # clear storm extrema annotation(s)
            # check if any annotations has focus
            elif cls.annotated_circle_objects:
                removed_any = False
                # find a track to clear if any has focus
                for internal_id, point_index_circles in cls.annotated_circle_objects.items():
                    removed_circle = None
                    removed_point_index = None
                    for point_index, annotated_circle in list(point_index_circles.items()):
                        if not annotated_circle:
                            continue
                        try:
                            if annotated_circle.annotation_has_focus(event):
                                removed_any = True
                                # we can't let annotation object pick it up,
                                #  since this causes a race condition since we are removing it
                                removed_point_index = point_index
                        except:
                            traceback.print_exc()

                        if removed_point_index is not None:
                            del point_index_circles[removed_point_index]
                            removed_point_index = None
                            if len(point_index_circles) == 0:
                                del cls.annotated_circle_objects[internal_id]

                            annotated_circle.remove()

                if not removed_any:
                    cls.clear_storm_extrema_annotations()
                
                cls.ax.set_yscale('linear')
            else:
                cls.clear_storm_extrema_annotations()

            cls.redraw_fig_canvas(stale_bg=True)
            cls.blit_circle_patch()
            return

        if not cls.zoom_selection_box:
            cls.measure_tool.on_key_press(event)

    @classmethod
    def on_key_release(cls, event):
        if cls.any_modal_open():
            return

        if not cls.zoom_selection_box:
            cls.measure_tool.on_key_release(event)

    @classmethod
    def on_click(cls, event):
        if cls.any_modal_open():
            return

        xlim = cls.ax.get_xlim()
        ylim = cls.ax.get_ylim()

        if event.inaxes == cls.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

            if inbound:
                if cls.selection_loop_mode:
                    SelectionLoops.on_click(event)
                    if event.button == 3:
                        cls.update_selection_info_label()
                    return
                try:
                    do_measure = cls.measure_tool.on_click(event)
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
                        if cls.time_step_legend_objects:

                            for i, time_step_legend_object in list(enumerate(reversed(cls.time_step_legend_objects))):
                                bbox = time_step_legend_object.get_window_extent()
                                if bbox.contains(event.x, event.y):
                                    changed_opacity = True
                                    # clicked on legend
                                    changed_opacity_index = i
                                    # for almost all models/modes, cycle between:
                                    #   1.0:    all visible
                                    #   0.6:    hide all storms tracks with start valid_time later than selected
                                    #   0.3:    hide all points beyond valid_time later than selected
                                    next_opacity_index = min(changed_opacity_index + 1, len(cls.time_step_opacity) - 1)
                                    if cls.time_step_opacity[i] != 1.0:
                                        # not cycling
                                        new_opacity = 0.6
                                    else:
                                        if cls.time_step_opacity[next_opacity_index] == 1.0:
                                            new_opacity = 0.6
                                        elif cls.time_step_opacity[next_opacity_index] == 0.6:
                                            new_opacity = 0.3
                                        else:
                                            new_opacity = 1.0
                                    break

                            if changed_opacity:
                                for i, time_step_legend_object in list(
                                        enumerate(reversed(cls.time_step_legend_objects))):
                                    if i > changed_opacity_index:
                                        cls.time_step_opacity[i] = new_opacity
                                    else:
                                        cls.time_step_opacity[i] = 1.0

                        if changed_opacity:
                            # update map as we have changed what is visible
                            model_cycle = None
                            if cls.mode == "GENESIS":
                                model_cycle = cls.genesis_model_cycle_time

                            cls.redraw_map_with_data(model_cycle=model_cycle)

                        else:
                            # zooming
                            try:
                                cls.zoom_selection_box = SelectionBox()
                                # handle case we are not blocking (something other action is blocking)
                                cls.zoom_selection_box.update_box(event.xdata, event.ydata, event.xdata, event.ydata)
                            except:
                                # some other action is blocking
                                pass
                                traceback.print_exc()

                    elif event.button == 3:  # Right click
                        cls.zoom_out(step_zoom=True)

    @classmethod
    def on_motion(cls, event):
        if cls.any_modal_open():
            return

        xlim = cls.ax.get_xlim()
        ylim = cls.ax.get_ylim()

        if event.inaxes == cls.ax:
            # Check if mouse coordinates are within figure bounds
            try:
                inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
            except:
                inbound = False

            lon = event.xdata
            lat = event.ydata
            cls.last_cursor_lon_lat = (lon, lat)
            cls.update_labels_for_mouse_hover(lat=lat, lon=lon)

            if cls.selection_loop_mode:
                if inbound:
                    SelectionLoops.on_motion(event)
                return
            elif cls.zoom_selection_box:
                x0 = cls.zoom_selection_box.lon1
                y0 = cls.zoom_selection_box.lat1
                if inbound:
                    x1, y1 = event.xdata, event.ydata
                else:
                    # out of bound motion
                    return
                if type(x0) == type(x1) == type(y0) == type(y1):
                    if cls.zoom_selection_box is not None:
                        cls.zoom_selection_box.update_box(x0, y0, x1, y1)

                    # blitting object will redraw using cached buffer
                    # cls.fig.canvas.draw_idle()
            else:
                cls.measure_tool.on_motion(event, inbound)

    @classmethod
    def on_release(cls, event):
        if cls.any_modal_open():
            return

        xlim = cls.ax.get_xlim()
        ylim = cls.ax.get_ylim()

        # Check if mouse coordinates are within figure bounds
        try:
            inbound = (xlim[0] <= event.xdata <= xlim[1] and ylim[0] <= event.ydata <= ylim[1])
        except:
            inbound = False

        if cls.selection_loop_mode:
            SelectionLoops.on_release(event)
            cls.update_selection_info_label()
            return

        doing_measurement = cls.measure_tool.in_measure_mode()
        if doing_measurement:
            pass
        else:
            if event.button == 1 and cls.zoom_selection_box:  # Left click release
                if event.xdata is None or event.ydata is None or not inbound:
                    if not cls.zoom_selection_box.is_2d():
                        cls.zoom_selection_box.destroy()
                        cls.zoom_selection_box = None
                        return

                cls.zoom_in()

    @classmethod
    def prev_genesis_cycle(cls):
        cls.update_genesis_data_staleness()
        nav_time = cls.genesis_model_cycle_time_navigate
        if not nav_time:
            nav_time = cls.genesis_model_cycle_time
        if cls.genesis_model_cycle_time is None:
            cls.genesis_model_cycle_time = datetime.now()
            model_cycles = get_tc_model_init_times_relative_to(cls.genesis_model_cycle_time, cls.genesis_previous_selected)
            if model_cycles['previous'] is None:
                model_cycle = model_cycles['at']
            else:
                model_cycle = model_cycles['previous']
        else:
            model_cycles = get_tc_model_init_times_relative_to(nav_time, cls.genesis_previous_selected)
            if model_cycles['previous'] is None:
                # nothing before current cycle
                return
            model_cycle = model_cycles['previous']

        if model_cycle:
            if model_cycle != cls.genesis_model_cycle_time:
                cls.genesis_model_cycle_time_navigate = model_cycle
                cls.redraw_map_with_data(model_cycle=model_cycle)

    # print track stats to terminal for hovered track
    @classmethod
    def print_track_stats(cls):
        if len(cls.plotted_tc_candidates) == 0:
            return

        # note: point_index is a tuple of tc_index, tc_point_index
        if len(cls.nearest_point_indices_overlapped) != 0:
            # annotate storm extrema of previously selected
            num, cursor_point_index = cls.nearest_point_indices_overlapped.get_prev_enum_key_tuple()
            if cursor_point_index is not None:
                cursor_internal_id, tc_index, tc_point_index = cursor_point_index
                internal_id, lat_lon_with_time_step_list = cls.plotted_tc_candidates[tc_index]
                # print out statistics for track
                storm_ace = 0.0
                storm_vmax10m = 0.0
                storm_vmax_time_earliest = None
                disturbance_start_time = None
                disturbance_end_time = None
                earliest_named_time = None
                for candidate_info in lat_lon_with_time_step_list:
                    if disturbance_start_time is None:
                        disturbance_start_time = candidate_info['valid_time']

                    disturbance_end_time = candidate_info['valid_time']
                    if 'vmax10m' in candidate_info:
                        point_vmax = candidate_info['vmax10m']
                        if point_vmax >= 34.0:
                            storm_ace += pow(10, -4) * np.power(point_vmax, 2)
                            if earliest_named_time is None:
                                earliest_named_time = candidate_info['valid_time']
                        if point_vmax > storm_vmax10m:
                            storm_vmax_time_earliest = candidate_info['valid_time']
                            storm_vmax10m = point_vmax

                print(f"Disturbance (Internal ID: {internal_id}) (Model Track stats):")
                print(f'    ACE (10^-4): {storm_ace:0.1f}')
                print(f'    Peak VMax @ 10m: {storm_vmax10m:0.1f} kt')
                print(f'    Disturbance start: {disturbance_start_time}')
                print(f'    Disturbance end: {disturbance_end_time}')
                if earliest_named_time is not None:
                    print(f'    Earliest Named time: {earliest_named_time}')
        else:
            internal_ids = cls.get_selected_internal_storm_ids()
            num_tracks = 0
            if internal_ids:
                num_tracks = len(internal_ids)

            if num_tracks == 0:
                return

            # Aggregate candidates by models
            filtered_candidates = [(iid, tc) for iid, tc in cls.plotted_tc_candidates if
                                   iid in internal_ids]

            if len(filtered_candidates) == 0:
                return

            if cls.mode == "ADECK":
                previous_selected_combo = cls.adeck_previous_selected
            else:
                previous_selected_combo = cls.genesis_previous_selected

            if previous_selected_combo == 'GLOBAL-DET':
                possible_ensemble = False
                total_ensembles = 0
                model_names = model_data_folders_by_model_name.keys()
                model_names = [model_name for model_name in model_names if model_name[-5:] != 'TCGEN']
                total_models = len(model_names)
                ensemble_type = 'GLOBAL-DET'
                lookup_model_name_ensemble_name = global_det_lookup
                # TODO: test; this won't work but is for redundancy, needs fixing
                lookup_num_models_by_ensemble_name = num_active_global_det_members
            elif previous_selected_combo[-5:] == 'TCGEN':
                possible_ensemble = True
                total_models = len(tcgen_models_by_ensemble[previous_selected_combo])
                if previous_selected_combo == 'ALL-TCGEN':
                    # Should be 4 (EPS, FNMOC, GEFS, GEPS; don't count ALL)
                    total_ensembles = len(tcgen_models_by_ensemble.keys()) - 1
                    if len(tcgen_active_ensembles_all) != total_ensembles:
                        total_ensembles = len(tcgen_active_ensembles_all)
                else:
                    total_ensembles = 1

                ensemble_type = 'TCGEN'
                lookup_model_name_ensemble_name = model_name_to_ensemble_name
                lookup_num_models_by_ensemble_name = tcgen_num_models_by_ensemble
            elif previous_selected_combo[-4:] in 'ATCF':
                # unofficial adeck (GEFS, GEPS, FNMOC)
                # not handling "GEFS-MEMBERS" (no official a-track gefs members as no guarantee which members are present?)
                possible_ensemble = True
                total_models = len(atcf_ens_models_by_ensemble[previous_selected_combo])
                if previous_selected_combo == 'ALL-ATCF':
                    # Should be 3 (FNMOC, GEFS, GEPS; don't count ALL)
                    total_ensembles = len(atcf_ens_num_models_by_ensemble.keys()) - 1
                    if len(atcf_active_ensembles_all) != total_ensembles:
                        total_ensembles = len(atcf_active_ensembles_all)
                else:
                    total_ensembles = 1

                ensemble_type = 'ATCF'
                lookup_model_name_ensemble_name = atcf_ens_model_name_to_ensemble_name
                lookup_num_models_by_ensemble_name = atcf_ens_num_models_by_ensemble
            else:
                # TODO: for now don't count ensembles for adeck data as we don't have a complete list of member names
                total_ensembles = 0
                possible_ensemble = False
                ensemble_type = 'UNKNOWN'
                # won't work for now until fixed in code
                lookup_model_name_ensemble_name = model_name_to_ensemble_name

            by_model_stats = {}
            for internal_id, tc in filtered_candidates:
                if tc and tc[0] and 'model_name' in tc[0]:
                    model_name = tc[0]['model_name']
                    ensemble_name = None
                    if possible_ensemble and model_name in lookup_model_name_ensemble_name:
                        ensemble_name = lookup_model_name_ensemble_name[model_name]

                    lat_lon_with_time_step_list = tc
                    # print out statistics for track
                    storm_ace = 0.0
                    storm_vmax10m = 0.0
                    storm_vmax_time_earliest = None
                    disturbance_start_time = None
                    disturbance_end_time = None
                    earliest_named_time = None
                    for candidate_info in lat_lon_with_time_step_list:
                        if disturbance_start_time is None:
                            disturbance_start_time = candidate_info['valid_time']

                        disturbance_end_time = candidate_info['valid_time']
                        if 'vmax10m' in candidate_info:
                            point_vmax = candidate_info['vmax10m']
                            if point_vmax >= 34.0:
                                storm_ace += pow(10, -4) * np.power(point_vmax, 2)
                                if earliest_named_time is None:
                                    earliest_named_time = candidate_info['valid_time']
                            if point_vmax > storm_vmax10m:
                                storm_vmax_time_earliest = candidate_info['valid_time']
                                storm_vmax10m = point_vmax

                    if model_name not in by_model_stats:
                        by_model_stats[model_name] = {
                            'storm_ace': [storm_ace],
                            'storm_vmax10m': [storm_vmax10m],
                            'storm_vmax_time_earliest': [storm_vmax_time_earliest],
                            'disturbance_start_time': [disturbance_start_time],
                            'earliest_named_time': [earliest_named_time],
                            'disturbance_end_time': [disturbance_end_time]
                        }
                    else:
                        by_model_stats[model_name]['storm_ace'].append(storm_ace)
                        by_model_stats[model_name]['storm_vmax10m'].append(storm_vmax10m)
                        by_model_stats[model_name]['storm_vmax_time_earliest'].append(storm_vmax_time_earliest)
                        by_model_stats[model_name]['disturbance_start_time'].append(disturbance_start_time)
                        by_model_stats[model_name]['earliest_named_time'].append(earliest_named_time)
                        by_model_stats[model_name]['disturbance_end_time'].append(disturbance_end_time)

                    if ensemble_name and 'ensemble_name' not in by_model_stats[model_name]:
                        by_model_stats[model_name]['ensemble_name'] = ensemble_name

            by_ensemble_stats = {}
            for model_name, model_stats in by_model_stats.items():
                if possible_ensemble:
                    ensemble_name = model_stats['ensemble_name']
                else:
                    ensemble_name = ensemble_type

                if ensemble_name not in by_ensemble_stats:
                    by_ensemble_stats[ensemble_name] = {
                        'storm_ace': [model_stats['storm_ace']],
                        'storm_vmax10m': [model_stats['storm_vmax10m']],
                        'storm_vmax_time_earliest': [model_stats['storm_vmax_time_earliest']],
                        'disturbance_start_time': [model_stats['disturbance_start_time']],
                        'earliest_named_time': [model_stats['earliest_named_time']],
                        'disturbance_end_time': [model_stats['disturbance_end_time']]
                    }
                else:
                    by_ensemble_stats[ensemble_name]['storm_ace'].append(model_stats['storm_ace'])
                    by_ensemble_stats[ensemble_name]['storm_vmax10m'].append(model_stats['storm_vmax10m'])
                    by_ensemble_stats[ensemble_name]['storm_vmax_time_earliest'].append(model_stats['storm_vmax_time_earliest'])
                    by_ensemble_stats[ensemble_name]['disturbance_start_time'].append(model_stats['disturbance_start_time'])
                    by_ensemble_stats[ensemble_name]['earliest_named_time'].append(model_stats['earliest_named_time'])
                    by_ensemble_stats[ensemble_name]['disturbance_end_time'].append(model_stats['disturbance_end_time'])

            # Initialize dictionaries to store aggregated statistics
            ensemble_stats = {}
            all_ensemble_stats = {}

            # Loop through each ensemble
            for ensemble_name, stats in by_ensemble_stats.items():
                # Initialize dictionaries to store aggregated statistics for the current ensemble
                model_agg = {}
                storm_values = defaultdict(list)
                weights = {}

                # Calculate aggregated statistics for the current ensemble
                for key, values in stats.items():
                    # Remove None values
                    valid_values = []
                    for value_list in values:
                        valid_list = []
                        if value_list is not None:
                            for v in value_list:
                                if v is not None:
                                    valid_list.append(v)

                        if len(valid_list) > 0:
                            valid_values.append(valid_list)

                    # Calculate weight for the current parameter
                    num_models_with_data = len(valid_values)
                    if possible_ensemble:
                        weight = 1 / lookup_num_models_by_ensemble_name[ensemble_name]
                    else:
                        weight = 1 / num_models_with_data

                    weights[key] = weight
                    flattened_valid_values = []
                    for valid_vals in valid_values:
                        flattened_valid_values.extend(valid_vals)

                    valid_values = np.array(flattened_valid_values)
                    have_data = len(valid_values) > 0
                    if key == 'storm_ace':
                        model_agg[key] = np.sum(valid_values)
                    elif key == 'storm_vmax10m':
                        model_agg[key] = np.max(valid_values) if have_data else None
                    elif key == 'storm_vmax_time_earliest':
                        model_agg[key] = np.min(valid_values) if have_data else None
                    elif key == 'disturbance_start_time':
                        model_agg[key] = np.min(valid_values) if have_data else None
                    elif key == 'earliest_named_time':
                        model_agg[key] = np.min(valid_values) if have_data else None
                    elif key == 'disturbance_end_time':
                        model_agg[key] = np.max(valid_values) if have_data else None

                    # Store all valid values for the current parameter
                    storm_values[key] = np.array(valid_values)

                ensemble_agg_mean = {}
                ensemble_agg_median = {}
                ensemble_agg_min = {}
                ensemble_agg_max = {}
                weighted_ensemble_agg_mean = {}
                weighted_ensemble_agg_median = {}
                weighted_ensemble_agg_min = {}
                weighted_ensemble_agg_max = {}

                for key, values in storm_values.items():
                    # Calculate ensemble statistics (mean, median, min, max)
                    if 'time' in key:
                        if len(values) > 0 and isinstance(values[0], datetime):
                            # Convert datetime objects to Unix timestamps
                            timestamps = [value.timestamp() for value in values]

                            # Calculate mean, median, min, max of timestamps
                            mean_timestamp = np.mean(timestamps)
                            median_timestamp = np.median(timestamps)
                            min_timestamp = np.min(timestamps)
                            max_timestamp = np.max(timestamps)

                            # Convert mean, median, min, max timestamps back to datetime objects
                            ensemble_agg_mean[key] = datetime.fromtimestamp(mean_timestamp)
                            ensemble_agg_median[key] = datetime.fromtimestamp(median_timestamp)
                            ensemble_agg_min[key] = datetime.fromtimestamp(min_timestamp)
                            ensemble_agg_max[key] = datetime.fromtimestamp(max_timestamp)

                            weighted_ensemble_agg_mean[key] = datetime.fromtimestamp(mean_timestamp)
                            weighted_ensemble_agg_median[key] = datetime.fromtimestamp(median_timestamp)
                            weighted_ensemble_agg_min[key] = datetime.fromtimestamp(min_timestamp)
                            weighted_ensemble_agg_max[key] = datetime.fromtimestamp(max_timestamp)
                        else:
                            # no time (empty array)
                            pass
                    else:
                        # Calculate mean, median, min, max for non-datetime values
                        if key == 'storm_ace':
                            ensemble_agg_mean[key] = np.mean(values)
                            ensemble_agg_median[key] = np.median(values)
                            ensemble_agg_min[key] = np.min(values)
                            ensemble_agg_max[key] = np.max(values)

                            weighted_ensemble_agg_mean[key] = model_agg[key] * weights[key]
                            # only the mean has significance, these are just placeholders
                            weighted_ensemble_agg_median[key] = model_agg[key] * weights[key]
                            weighted_ensemble_agg_min[key] = model_agg[key] * weights[key]
                            weighted_ensemble_agg_max[key] = model_agg[key] * weights[key]
                        else:
                            ensemble_agg_mean[key] = np.mean(values)
                            ensemble_agg_median[key] = np.median(values)
                            ensemble_agg_min[key] = np.min(values)
                            ensemble_agg_max[key] = np.max(values)

                            weighted_ensemble_agg_mean[key] = np.mean(values) * weights[key]
                            weighted_ensemble_agg_median[key] = np.median(values) * weights[key]
                            weighted_ensemble_agg_min[key] = np.min(values) * weights[key]
                            weighted_ensemble_agg_max[key] = np.max(values) * weights[key]

                # Store aggregated statistics for the current ensemble
                ensemble_stats[ensemble_name] = {
                    'model_agg': model_agg,
                    'ensemble_agg_mean': ensemble_agg_mean,
                    'ensemble_agg_median': ensemble_agg_median,
                    'ensemble_agg_min': ensemble_agg_min,
                    'ensemble_agg_max': ensemble_agg_max,
                    'weighted_ensemble_agg_mean': weighted_ensemble_agg_mean,
                    'weighted_ensemble_agg_median': weighted_ensemble_agg_median,
                    'weighted_ensemble_agg_min': weighted_ensemble_agg_min,
                    'weighted_ensemble_agg_max': weighted_ensemble_agg_max,
                }

            # Calculate aggregated statistics for super ensemble
            if len(by_ensemble_stats) > 1:
                all_ensemble_stats['ALL'] = {}
                for key in ensemble_stats[list(by_ensemble_stats.keys())[0]].keys():
                    all_ensemble_stats['ALL'][key] = {}
                    for ensemble_name, stats in ensemble_stats.items():
                        for param, value in stats[key].items():
                            if param not in all_ensemble_stats['ALL'][key]:
                                all_ensemble_stats['ALL'][key][param] = []
                            all_ensemble_stats['ALL'][key][param].append(value)

                    for param, values in all_ensemble_stats['ALL'][key].items():
                        if key.startswith('weighted'):
                            if isinstance(values[0], datetime):
                                # Convert datetime objects to Unix timestamps
                                timestamps = [value.timestamp() for value in values]
                                # Calculate weighted mean of timestamps
                                weighted_mean_timestamp = sum(timestamps) / len(by_ensemble_stats)
                                # Convert weighted mean timestamp back to datetime object
                                all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(
                                    weighted_mean_timestamp)
                            else:
                                all_ensemble_stats['ALL'][key][param] = sum(values) / len(by_ensemble_stats)
                        elif key.startswith('ensemble_agg'):
                            if isinstance(values[0], datetime):
                                # Convert datetime objects to Unix timestamps
                                timestamps = [value.timestamp() for value in values]
                                if key.endswith('mean'):
                                    # Calculate mean of timestamps
                                    mean_timestamp = np.mean(timestamps)
                                    # Convert mean timestamp back to datetime object
                                    all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(
                                        mean_timestamp)
                                elif key.endswith('median'):
                                    # Calculate median of timestamps
                                    median_timestamp = np.median(timestamps)
                                    # Convert median timestamp back to datetime object
                                    all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(
                                        median_timestamp)
                                elif key.endswith('min'):
                                    # Calculate min of timestamps
                                    min_timestamp = np.min(timestamps)
                                    # Convert min timestamp back to datetime object
                                    all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(
                                        min_timestamp)
                                elif key.endswith('max'):
                                    # Calculate max of timestamps
                                    max_timestamp = np.max(timestamps)
                                    # Convert max timestamp back to datetime object
                                    all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(
                                        max_timestamp)
                            else:
                                if key.endswith('mean'):
                                    all_ensemble_stats['ALL'][key][param] = np.mean(values)
                                elif key.endswith('median'):
                                    all_ensemble_stats['ALL'][key][param] = np.median(values)
                                elif key.endswith('min'):
                                    all_ensemble_stats['ALL'][key][param] = min(values)
                                elif key.endswith('max'):
                                    all_ensemble_stats['ALL'][key][param] = max(values)
                        else:
                            if 'time' in param:
                                if len(values) > 0:
                                    # and isinstance(values[0], datetime)):
                                    # Convert datetime objects to Unix timestamps
                                    timestamps = [value.timestamp() for value in values if isinstance(value, datetime)]
                                    # Calculate mean of timestamps
                                    mean_timestamp = sum(timestamps) / len(by_ensemble_stats)
                                    # Convert mean timestamp back to datetime object
                                    all_ensemble_stats['ALL'][key][param] = datetime.fromtimestamp(mean_timestamp)
                            else:
                                all_ensemble_stats['ALL'][key][param] = sum(values) / len(by_ensemble_stats)


            # Print aggregated statistics for each ensemble
            for ensemble_name, stats in ensemble_stats.items():
                print(f"Ensemble: {ensemble_name}")
                for key, values in stats.items():
                    print(f"  {key}:")
                    for param, value in values.items():
                        if param != 'storm_ace' or (key not in \
                            ['weighted_ensemble_agg_median', 'weighted_ensemble_agg_min', 'weighted_ensemble_agg_max']):

                            if param == 'storm_ace':
                                print(f"    {param} (10^-4): {value:0.1f}")
                            elif param == 'storm_vmax10m':
                                print(f"    {param} (kts): {value:0.1f}")
                            else:
                                print(f"    {param}: {value}")

                print()

            # Print aggregated statistics for all ensembles
            if len(by_ensemble_stats) > 1:
                print("Ensemble: ALL")
                for key, values in all_ensemble_stats['ALL'].items():
                    print(f"  {key}:")
                    for param, value in values.items():

                        if param != 'storm_ace' or (key not in \
                            ['weighted_ensemble_agg_median', 'weighted_ensemble_agg_min', 'weighted_ensemble_agg_max']):

                            if param in ['storm_ace' or 'storm_vmax10m']:
                                print(f"    {param}: {value:0.1f}")
                            else:
                                print(f"    {param}: {value}")
                print()

    @classmethod
    def redraw_app_canvas(cls):
        cls.canvas.draw()
        cls.blit_circle_patch()

    @classmethod
    def redraw_fig_canvas(cls, stale_bg=False):
        cls.ax.figure.canvas.draw()

        cls.blit_circle_patch(stale_bg=stale_bg)
        cls.blit_mean_track(stale_bg=stale_bg)

    @classmethod
    def redraw_map_with_data(cls, model_cycle=None):
        cls.hidden_tc_candidates = set()
        cls.display_map()
        if cls.mode == "GENESIS" and model_cycle:
            cls.display_genesis_data(model_cycle)
        elif cls.mode == "ADECK":
            cls.display_deck_data()

    @classmethod
    def reload(cls):
        if cls.mode == "ADECK":
            cls.update_deck_data()
            cls.redraw_map_with_data()
        elif cls.mode == "GENESIS":
            cls.display_map()
            cls.update_genesis_data_staleness()
            model_cycle = cls.genesis_model_cycle_time
            if model_cycle is None:
                model_cycles = get_tc_model_init_times_relative_to(datetime.now(), cls.genesis_previous_selected)
                if model_cycles['next'] is None:
                    model_cycle = model_cycles['at']
                else:
                    model_cycle = model_cycles['next']
            if model_cycle:
                # clear map
                cls.redraw_map_with_data(model_cycle=model_cycle)

    @classmethod
    def reload_adeck(cls):
        if cls.deck_timer_id is not None:
            cls.root.after_cancel(cls.deck_timer_id)
        cls.deck_timer_id = cls.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_deck_data)

        cls.reload()
        cls.set_focus_on_map()

    @classmethod
    def reload_genesis(cls):
        if cls.genesis_timer_id is not None:
            cls.root.after_cancel(cls.genesis_timer_id)
        cls.genesis_timer_id = cls.root.after(LOCAL_TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_genesis_data)

        cls.reload()
        cls.set_focus_on_map()

    @classmethod
    def rvor_labels_new_extent(cls):
        cls.update_rvor_contour_renderable_after_zoom()
        extent = cls.ax.get_extent(ccrs.PlateCarree())
        contour_visible = cls.overlay_rvor_contour_visible
        not_at_global_extent = not (
                extent[0] == cls.global_extent[0] and
                extent[1] == cls.global_extent[1] and
                extent[2] == cls.global_extent[2] and
                extent[3] == cls.global_extent[3]
        )
        label_visible = contour_visible and cls.overlay_rvor_label_visible and not_at_global_extent
        if label_visible:
            alpha_label_visible = 1.0
        else:
            alpha_label_visible = 0.0
        # if cls.overlay_rvor_label_last_alpha != alpha_label_visible:
        try:
            renderable_ids = cls.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, obj_list in cls.overlay_rvor_contour_dict['label_objs'].items():
                # limit detail for labels based on zoom extent
                is_renderable = (contour_id in renderable_ids)
                is_visible = label_visible and is_renderable
                for obj in obj_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass

        cls.overlay_rvor_label_last_alpha = alpha_label_visible

    @classmethod
    def save_analysis_tz(cls, duration, tz):
        global ANALYSIS_TZ
        ANALYSIS_TZ = tz
        if duration == "setting":
            with open('settings_tcviewer.json', 'r') as f:
                settings = json.load(f)
            if settings:
                settings['settings']['ANALYSIS_TZ'] = ANALYSIS_TZ
                with open('settings_tcviewer.json', 'w') as f:
                    json.dump(settings, f, indent=4)

    @classmethod
    def save_hidden(cls, slot=None):
        cls.saved_hidden[slot] = copy.deepcopy(cls.hidden_tc_candidates)

    @classmethod
    def select_basin_dialog(cls, event=None):
        dialog = tk.Toplevel(cls.root)
        dialog.title('Select Basin')
        dialog.geometry(f"300x200+{int(cls.root.winfo_width()/2-150)}+{int(cls.root.winfo_height()/2-100)}")

        listbox = tk.Listbox(dialog, bg=default_bg, fg=default_fg)
        for basin_name in sorted(basin_extents.keys(), key=lambda x: ['N', 'E', 'W', 'I', 'S'].index(x[0])):
            listbox.insert(tk.END, basin_name)

        listbox.pack(fill=tk.BOTH, expand=1)
        listbox.selection_set(0)
        listbox.focus_set()
        listbox.see(0)

        def cancel():
            dialog.destroy()

        def select_basin(event):
            selected_basin = listbox.get(listbox.curselection())
            SelectionLoops.add_poly(basin_polys[selected_basin])
            cls.update_selection_info_label()
            cancel()

        listbox.bind('<Return>', select_basin)
        listbox.bind('<Up>', lambda event: listbox.select_set(0))
        listbox.bind('<Down>', lambda event: listbox.select_set(0))
        listbox.bind("<Escape>", lambda event: cancel())
        ttk.Button(dialog, text='OK', command=select_basin, style='TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(dialog, text='Cancel', command=cancel, style='TButton').pack(side=tk.LEFT, padx=5, pady=5)

    @classmethod
    def set_netcdf_display_iso26C(cls, event=None):
        global DISPLAY_NETCDF
        DISPLAY_NETCDF = 'iso26C'
        cls.redraw_map_with_data(model_cycle=cls.genesis_model_cycle_time)

    @classmethod
    def set_netcdf_display_none(cls, event=None):
        global DISPLAY_NETCDF
        DISPLAY_NETCDF = None
        cls.redraw_map_with_data(model_cycle=cls.genesis_model_cycle_time)

    @classmethod
    def set_netcdf_display_ohc(cls, event=None):
        global DISPLAY_NETCDF
        DISPLAY_NETCDF = 'ohc'
        cls.redraw_map_with_data(model_cycle=cls.genesis_model_cycle_time)

    @classmethod
    def set_netcdf_display_sst(cls, event=None):
        global DISPLAY_NETCDF
        DISPLAY_NETCDF = 'sst'
        cls.redraw_map_with_data(model_cycle=cls.genesis_model_cycle_time)

    @classmethod
    def set_focus_on_map(cls):
        cls.canvas.get_tk_widget().focus_set()

    @classmethod
    def show_analysis_dialog(cls, event=None):
        if not cls.analysis_dialog_open:
            cls.analysis_dialog_open = True
        else:
            return

        root_width = cls.root.winfo_screenwidth()
        root_height = cls.root.winfo_screenheight()
        if cls.mode == "ADECK":
            previous_selected_combo = cls.adeck_previous_selected
        else:
            previous_selected_combo = cls.genesis_previous_selected
        dialog = AnalysisDialog(cls.root, cls.plotted_tc_candidates, root_width, root_height, previous_selected_combo)

        # fix focus back to map
        cls.set_focus_on_map()

        cls.analysis_dialog_open = False

    @classmethod
    def show_config_adeck_dialog(cls):
        cls.show_config_dialog()
        cls.set_focus_on_map()

    @classmethod
    def show_config_dialog(cls):
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
        global ANALYSIS_TZ
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
            'ANALYSIS_TZ': tk.StringVar(value=ANALYSIS_TZ)
        }
        root_width = cls.root.winfo_screenwidth()
        root_height = cls.root.winfo_screenheight()

        dialog = ConfigDialog(cls.root, displayed_functional_annotation_options, DISPLAYED_FUNCTIONAL_ANNOTATIONS,
                              settings, root_width, root_height)
        if dialog.result:
            result = dialog.result
            updated_annotated_colors = False
            for key, vals in result.items():
                if key == 'annotation_label_options':
                    DISPLAYED_FUNCTIONAL_ANNOTATIONS = [option for option in displayed_functional_annotation_options if
                                                        option in vals]
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
        cls.set_focus_on_map()

    @classmethod
    def show_config_genesis_dialog(cls):
        cls.show_config_dialog()
        cls.set_focus_on_map()

    @classmethod
    def show_hide_by_field_dialog(cls, event=None):
        cls.root.update_idletasks()

        if not cls.hide_by_field_dialog_open:
            cls.hide_by_field_dialog_open = True
        else:
            return

        dialog = tk.Toplevel(cls.root, bg="#000000")
        dialog.protocol("WM_DELETE_WINDOW", lambda: cls.on_hide_by_field_dialog_close(dialog))
        dialog.title("Hide (Selected) Tracks by Field")

        height = 540
        width = 925
        dialog.geometry(f"{width}x{height}")  # Set the width to 300 and height to 250
        dialog.geometry(
            f"+{tk_root.winfo_x() - width // 2 + tk_root.winfo_width() // 2}+{tk_root.winfo_y() - height // 2 + tk_root.winfo_height() // 2}")

        # modal
        dialog.grab_set()


        # Create a frame to hold the checkboxes
        frame = ttk.Frame(dialog, style='CanvasFrame.TFrame')
        frame.grid()
        frame.focus_set()
        frame.focus()

        # Field Constraints Section
        # Header Labels
        n = 0
        ttk.Label(frame, text="Field", font=("Arial", 12, 'bold')).grid(row=n, column=0, padx=5, pady=5)
        ttk.Label(frame, text="Value", font=("Arial", 12, 'bold')).grid(row=n, column=1, padx=5, pady=5)
        ttk.Label(frame, text="Tol (+/-)", font=("Arial", 12, 'bold')).grid(row=n, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Min", font=("Arial", 12, 'bold')).grid(row=n, column=3, padx=5, pady=5)
        ttk.Label(frame, text="Max", font=("Arial", 12, 'bold')).grid(row=n, column=4, padx=5, pady=5)

        # Field rows with default tolerances
        cls.fields = {
            'lat': ("Latitude (deg):", 0.2),
            'lon': ("Longitude (deg):", 0.2),
            'mslp_value': ("MSLP (hPa):", 2.0),
            'vmax10m': ("VMAX @ 10m (kt):", 5.0),
            'vmax10m_ms': ("VMAX @ 10m (m/s):", 2.6),
            'roci': ("ROCI (km):", 0),
            'rmw': ("RMW (km):", 0)
        }

        # Entries
        cls.field_vars = {}

        n += 1
        for i, (key, (label, tolerance)) in enumerate(cls.fields.items(), start=n):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=5, pady=5)

            value_var = tk.StringVar()
            tolerance_var = tk.StringVar(value=str(tolerance))
            min_var = tk.StringVar()
            max_var = tk.StringVar()

            ttk.Entry(frame, textvariable=value_var, width=8).grid(row=i, column=1, padx=5, pady=5)
            ttk.Entry(frame, textvariable=tolerance_var, width=4).grid(row=i, column=2, padx=5, pady=5)
            ttk.Entry(frame, textvariable=min_var, width=8).grid(row=i, column=3, padx=5, pady=5)
            ttk.Entry(frame, textvariable=max_var, width=8).grid(row=i, column=4, padx=5, pady=5)

            cls.field_vars[key] = {
                'value': value_var,
                'tolerance': tolerance_var,
                'min': min_var,
                'max': max_var
            }
            n += 1

        # Time Constraints Section
        # Labels for 'Time Constraints'
        ttk.Label(frame, text="Time Constraints", font=("Arial", 12, 'bold')).grid(row=n, column=0, padx=5, pady=10,
                                                                                   columnspan=9)
        n += 1
        time_labels = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Sec', 'Tol (+/-) (s)']
        for i, time_label in enumerate(time_labels):
            ttk.Label(frame, text=time_labels[i], font=("Arial", 12, 'bold')).grid(row=n, column=i+2, padx=5, pady=5)

        n += 1
        current_time = datetime_utcnow()
        cls.time_vars = {
            'point': {
                'year': tk.StringVar(),
                'month': tk.StringVar(),
                'day': tk.StringVar(),
                'hour': tk.StringVar(),
                'minute': tk.StringVar(),
                'second': tk.StringVar(),
                'tolerance': tk.StringVar(value="0")
            },
            'min': {
                'year': tk.StringVar(),
                'month': tk.StringVar(),
                'day': tk.StringVar(),
                'hour': tk.StringVar(),
                'minute': tk.StringVar(),
                'second': tk.StringVar(),
            },
            'max': {
                'year': tk.StringVar(),
                'month': tk.StringVar(),
                'day': tk.StringVar(),
                'hour': tk.StringVar(),
                'minute': tk.StringVar(),
                'second': tk.StringVar(),
            }
        }

        n += 1
        # Row for point time
        ttk.Label(frame, text="Point Time").grid(row=n, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Now", command=lambda: cls.hide_set_current_time(cls.time_vars['point'])).grid(row=n,
                                                                                                     column=1,
                                                                                                     padx=5,
                                                                                                     pady=5)
        for i, (key, var) in enumerate(cls.time_vars['point'].items()):
            ttk.Entry(frame, textvariable=var, width=4).grid(row=n, column=i + 2, padx=5, pady=5, sticky="ew")

        n += 1
        # Row for min time
        ttk.Label(frame, text="Min Time").grid(row=n, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Now", command=lambda: cls.hide_set_current_time(cls.time_vars['min'])).grid(row=n,
                                                                                                     column=1,
                                                                                                     padx=5,
                                                                                                     pady=5)
        for i, (key, var) in enumerate(cls.time_vars['min'].items()):
            if key != 'tolerance':
                ttk.Entry(frame, textvariable=var, width=4).grid(row=n, column=i + 2, padx=5, pady=5, sticky="ew")

        n += 1
        # Row for max time
        ttk.Label(frame, text="Max Time").grid(row=n, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Now", command=lambda: cls.hide_set_current_time(cls.time_vars['max'])).grid(row=n,
                                                                                                     column=1,
                                                                                                     padx=5,
                                                                                                     pady=5)
        for i, (key, var) in enumerate(cls.time_vars['max'].items()):
            if key != 'tolerance':
                ttk.Entry(frame, textvariable=var, width=4).grid(row=n, column=i + 2, padx=5, pady=5, sticky="ew")

        n += 1
        # buttons
        all_all_btn = ttk.Button(frame, text="All T All F",
                            command=lambda: [cls.hide_by_field(by_all_all=True), cls.on_hide_by_field_dialog_close(dialog)],
                            style='TButton', width=12)
        all_all_btn.grid(row=n, column=0, padx=20, pady=5)

        all_any_btn = ttk.Button(frame, text="All T Any F",
                            command=lambda: [cls.hide_by_field(by_all_any=True), cls.on_hide_by_field_dialog_close(dialog)],
                            style='TButton', width=12)
        all_any_btn.grid(row=n, column=1, padx=20, pady=5)

        any_all_btn = ttk.Button(frame, text="Any T All F",
                                 command=lambda: [cls.hide_by_field(by_any_all=True), cls.on_hide_by_field_dialog_close(dialog)],
                                 style='TButton', width=12)
        any_all_btn.grid(row=n, column=2, padx=20, pady=5)

        any_any_btn = ttk.Button(frame, text="Any T Any F",
                             command=lambda: [cls.hide_by_field(by_any_any=True), cls.on_hide_by_field_dialog_close(dialog)],
                             style='TButton', width=12)
        any_any_btn.grid(row=n, column=3, padx=20, pady=5)

        # Cancel button
        cancel_btn = ttk.Button(frame, text="Cancel", command=lambda e: cls.on_hide_by_field_dialog_close(dialog), style='TButton', width=8)
        cancel_btn.grid(row=n, column=4, padx=20, pady=5)

        dialog.bind("<Escape>", lambda e: cls.on_hide_by_field_dialog_close(dialog))

    @classmethod
    def show_rvor_dialog(cls, event=None):
        cls.root.update_idletasks()

        if not cls.rvor_dialog_open:
            cls.rvor_dialog_open = True
        else:
            return
        # Create the toplevel dialog
        global SELECTED_PRESSURE_LEVELS
        dialog = tk.Toplevel(cls.root, bg="#000000")
        dialog.protocol("WM_DELETE_WINDOW", lambda: cls.on_rvor_dialog_close(dialog))
        dialog.title("Selected RVOR levels:")
        height = 275
        width = 240
        dialog.geometry(f"{width}x{height}")  # Set the width to 300 and height to 250
        dialog.geometry(
            f"+{tk_root.winfo_x() - width // 2 + tk_root.winfo_width() // 2}+{tk_root.winfo_y() - height // 2 + tk_root.winfo_height() // 2}")
        # modal
        dialog.grab_set()

        # Create a frame to hold the checkboxes
        frame = ttk.Frame(dialog, style='CanvasFrame.TFrame')
        frame.grid()

        # Create the checkboxes and their corresponding variables
        cls.level_vars = {}
        chk_first = None
        for i, level in enumerate([925, 850, 700, 500, 200], start=1):
            val = 1 if level in SELECTED_PRESSURE_LEVELS else 0
            var = tk.IntVar(value=val)
            chk = ttk.Checkbutton(frame, text=f"{level} mb", variable=var, style='TCheckbutton')
            var.set(val)
            if not chk_first:
                chk_first = chk
            chk.grid(row=i, column=0, sticky="n")

            # Center the labels
            frame.columnconfigure(0, weight=1)
            chk.columnconfigure(0, weight=1)
            cls.level_vars[level] = var

        for i in range(9):
            frame.grid_columnconfigure(i, pad=15)
            frame.grid_rowconfigure(i, pad=10)

        # Focus on the first checkbox
        frame.focus_set()
        frame.focus()  # Set focus on the frame
        chk_first.focus_set()

        # OK button
        ok_btn = ttk.Button(dialog, text="OK",
                           command=lambda: [cls.update_rvor_levels(), cls.on_rvor_dialog_close(dialog)], style='TButton')
        ok_btn.grid(row=8, column=0)

        # Cancel button
        cancel_btn = ttk.Button(dialog, text="Cancel", command=dialog.destroy, style='TButton')
        cancel_btn.config(width=ok_btn.cget("width"))  # Set the width of the Cancel button to match the OK button
        cancel_btn.grid(row=8, column=1)

        dialog.bind("<Return>", lambda e: [cls.update_rvor_levels(), cls.on_rvor_dialog_close(dialog)])
        dialog.bind("<Escape>", lambda e: cls.on_rvor_dialog_close(dialog))

    @classmethod
    def switch_mode(cls):
        cls.mode = "GENESIS" if cls.mode == "ADECK" else "ADECK"
        cls.update_mode()
        cls.set_focus_on_map()

    @classmethod
    def take_screenshot(cls, *args):
        # Get the current UTC date and time
        current_time = datetime_utcnow().strftime("%Y-%m-%d-%H-%M-%S")

        # Create the screenshots folder if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        # Capture the screenshot
        screenshot = ImageGrab.grab()

        # Save the screenshot as a PNG file
        screenshot.save(f"screenshots/{current_time}.png")

        # Create a custom dialog window
        dialog = tk.Toplevel(cls.root, bg="#000000")
        dialog.title("tcviewer")
        dialog.geometry("200x100")  # Set the dialog size

        # Center the dialog
        x = (tk_root.winfo_screenwidth() - 200) // 2
        y = (tk_root.winfo_screenheight() - 100) // 2
        dialog.geometry(f"+{x}+{y}")

        label = ttk.Label(dialog, text="Screenshot saved", style='TLabel')
        label.pack(pady=10)
        button = ttk.Button(dialog, text="OK", command=dialog.destroy, style='TButton')
        button.pack(pady=10)

        # Set focus on the OK button
        button.focus_set()

        # Bind the Enter key to the button's command
        dialog.bind("<Return>", lambda event: button.invoke())

        dialog.bind("<Escape>", lambda event: dialog.destroy())

        # Bind the FocusOut event to the dialog's destroy method
        dialog.bind("<FocusOut>", lambda event: dialog.destroy())

    @classmethod
    def toggle_rvor_contours(cls, *args):
        new_vis = not cls.overlay_rvor_contour_visible
        try:
            renderable_ids = cls.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, objs_list in cls.overlay_rvor_contour_dict['contour_objs'].items():
                # is_renderable = (contour_id in renderable_ids)
                # is_visible = new_vis and is_renderable
                # don't limit contour detail by extent (only limit labels)
                is_visible = new_vis
                for obj in objs_list:
                    obj.set_visible(is_visible)
            cls.overlay_rvor_label_visible = new_vis
            for contour_id, objs_list in cls.overlay_rvor_contour_dict['label_objs'].items():
                is_renderable = (contour_id in renderable_ids)
                is_visible = new_vis and is_renderable
                for obj in objs_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass
        cls.overlay_rvor_contour_visible = new_vis
        cls.ax.set_yscale('linear')
        cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def toggle_rvor_labels(cls, *args):
        new_vis = not cls.overlay_rvor_label_visible
        if new_vis:
            new_alpha = 1.0
        else:
            new_alpha = 0.0
        try:
            renderable_ids = cls.overlay_rvor_contour_dict['renderable_ids']
            for contour_id, obj_list in cls.overlay_rvor_contour_dict['label_objs'].items():
                is_renderable = (contour_id in renderable_ids)
                is_visible = new_vis and is_renderable
                for obj in obj_list:
                    obj.set_visible(is_visible)
        except:
            traceback.print_exc()
            pass
        cls.overlay_rvor_label_last_alpha = new_alpha
        cls.overlay_rvor_label_visible = new_vis
        # fixes bug in matplotlib after modifying many artists (especially test boxes)
        cls.ax.set_yscale('linear')
        cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def toggle_selection_loop_mode(cls):
        cls.selection_loop_mode = not (cls.selection_loop_mode)
        cls.update_toggle_selection_loop_button_color()
        cls.set_focus_on_map()

    @classmethod
    def update_axes(cls):
        gl = cls.lastgl

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
                cls.ax._gridliners.remove(gl)
            except:
                pass
        cls.ax.set_yscale('linear')

        gl = cls.ax.gridlines(draw_labels=["bottom", "left"], x_inline=False, y_inline=False, auto_inline=False,
                               color='white', alpha=0.5, linestyle='--')
        # Move axis labels inside the subplot
        cls.ax.tick_params(axis='both', direction='in', labelsize=16)
        # https://github.com/SciTools/cartopy/issues/1642
        gl.xpadding = -10  # Ideally, this would move labels inside the map, but results in hidden labels
        gl.ypadding = -10  # Ideally, this would move labels inside the map, but results in hidden labels

        gl.xlabel_style = {'color': 'orange'}
        gl.ylabel_style = {'color': 'orange'}

        # in pixels
        window_extent = cls.ax.get_window_extent()
        # in degrees
        extent = cls.ax.get_extent()

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

        # fitted_grid_line_degrees = min(multiples, key=lambda x: (x - min_grid_line_spacing_pixels if x >= min_grid_line_spacing_pixels else float('inf')))
        fitted_grid_line_degrees = min([multiple for multiple, spacing in zip(multiples, grid_line_spacing_inches) if
                                        spacing >= MIN_GRID_LINE_SPACING_INCHES], default=float('inf'))
        if fitted_grid_line_degrees == float('inf'):
            # must pick a reasonable number
            fitted_grid_line_degrees = multiples[-1]

        gl.xlocator = plt.MultipleLocator(fitted_grid_line_degrees)
        gl.ylocator = plt.MultipleLocator(fitted_grid_line_degrees)

        cls.lastgl = gl

        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
        lat_formatter = LatitudeFormatter(direction_label=True)
        lon_formatter = LongitudeFormatter(direction_label=True)
        cls.ax.xaxis.set_major_formatter(lon_formatter)
        cls.ax.yaxis.set_major_formatter(lat_formatter)

    @classmethod
    def update_circle_patch(cls, lon=None, lat=None):
        if cls.last_circle_lon == lon and cls.last_circle_lat == lat:
            return

        if cls.circle_handle:
            # restore region
            if cls.background_for_blit:
                cls.ax.figure.canvas.restore_region(cls.background_for_blit)
                cls.ax.figure.canvas.blit(cls.ax.bbox)

            # cls.circle_handle.remove()

        # cls.ax.set_yscale('linear')

        cls.circle_handle = Circle((lon, lat), radius=cls.calculate_radius_pixels(),
                                    color=DEFAULT_ANNOTATE_MARKER_COLOR, fill=False, linestyle='dotted', linewidth=2,
                                    alpha=0.8,
                                    transform=ccrs.PlateCarree())
        
        # as we are not using cartopy's Circle patches we have to update axes ourself
        cls.circle_handle.axes = cls.ax
        #cls.ax.add_patch(cls.circle_handle)
        #cls.redraw_fig_canvas()

        cls.last_circle_lon = lon
        cls.last_circle_lat = lat

        cls.blit_circle_patch()

    @classmethod
    def update_deck_data(cls):
        # track which data is stale (tc vitals, adeck, bdeck)
        # unfortunately we need to get all each type (vitals, adeck, bdeck) from both mirrors, since
        # it's unknown which mirror actually has most up-to-date data from the modification date alone
        updated_urls_tcvitals = set()
        updated_urls_adeck = set()
        updated_urls_bdeck = set()

        # logic for updating classes
        do_update_tcvitals = do_update_adeck = do_update_adeck2 = do_update_bdeck = False
        if not cls.have_deck_data:
            # first fetch of data
            do_update_tcvitals = do_update_adeck = do_update_adeck2 = do_update_bdeck = True
            if cls.deck_timer_id is not None:
                cls.root.after_cancel(cls.deck_timer_id)
            cls.deck_timer_id = cls.root.after(TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_deck_data)
        else:
            # refresh status of stale data one more time since the user has requested a reload
            cls.check_for_stale_deck_data()
            if cls.dt_mods_tcvitals and cls.stale_urls['tcvitals']:
                do_update_tcvitals = True
            if cls.dt_mods_adeck and cls.stale_urls['adeck']:
                do_update_adeck = True
            if cls.dt_mods_adeck2 and cls.stale_adeck2:
                do_update_adeck2 = True
            if cls.dt_mods_bdeck and cls.stale_urls['bdeck']:
                do_update_bdeck = True

        # Get recent storms (tcvitals)
        if do_update_tcvitals:
            new_dt_mods_tcvitals, new_recent_storms = get_recent_storms(tcvitals_urls)
            if new_dt_mods_tcvitals:
                old_dt_mods = copy.deepcopy(cls.dt_mods_tcvitals)
                cls.dt_mods_tcvitals.update(new_dt_mods_tcvitals)
                updated_urls_tcvitals = diff_dicts(old_dt_mods, cls.dt_mods_tcvitals)
                if updated_urls_tcvitals:
                    cls.recent_storms = new_recent_storms
        # Get A-Deck and B-Deck files
        if do_update_adeck or do_update_adeck2 or do_update_bdeck:

            new_dt_mods_adeck, new_dt_mods_adeck2, new_dt_mods_bdeck, new_adeck, new_bdeck = get_deck_files(
                cls.recent_storms, adeck_urls, bdeck_urls, do_update_adeck, do_update_adeck2, do_update_bdeck)

            if new_dt_mods_adeck and do_update_adeck:
                old_dt_mods = copy.deepcopy(cls.dt_mods_adeck)
                cls.dt_mods_adeck.update(new_dt_mods_adeck)
                updated_urls_adeck = diff_dicts(old_dt_mods, cls.dt_mods_adeck)
                if updated_urls_adeck:
                    cls.adeck = new_adeck

            if new_dt_mods_adeck2 and do_update_adeck2:
                old_dt_mods = copy.deepcopy(cls.dt_mods_adeck2)
                cls.dt_mods_adeck2.update(new_dt_mods_adeck2)
                updated_files_adeck2 = diff_dicts(old_dt_mods, cls.dt_mods_adeck2)
                if updated_files_adeck2:
                    # the official adeck and unofficial adeck are combined so this if fine
                    cls.adeck = new_adeck

            if new_dt_mods_bdeck and do_update_bdeck:
                old_dt_mods = copy.deepcopy(cls.dt_mods_bdeck)
                cls.dt_mods_bdeck.update(new_dt_mods_bdeck)
                updated_urls_bdeck = diff_dicts(old_dt_mods, cls.dt_mods_bdeck)
                if updated_urls_bdeck:
                    cls.bdeck = new_bdeck

        if cls.dt_mods_tcvitals or cls.dt_mods_adeck or dt_mods_adeck2 or cls.dt_mods_bdeck:
            # at least something was downloaded
            cls.have_deck_data = True

        cls.stale_urls['tcvitals'] = cls.stale_urls['tcvitals'] - set(updated_urls_tcvitals)
        cls.stale_urls['adeck'] = cls.stale_urls['adeck'] - set(updated_urls_adeck)
        cls.stale_urls['bdeck'] = cls.stale_urls['bdeck'] - set(updated_urls_bdeck)
        cls.stale_adeck2 = False
        cls.update_reload_button_color_for_deck()

    @classmethod
    def update_genesis_data_staleness(cls):
        # track which data is stale (global-det or all-tcgen)
        # this is mainly a simple clearing of staleness (doesn't keep track of which set the user views)
        if cls.genesis_timer_id is not None:
            cls.root.after_cancel(cls.genesis_timer_id)
        cls.genesis_timer_id = cls.root.after(LOCAL_TIMER_INTERVAL_MINUTES * 60 * 1000, cls.check_for_stale_genesis_data)

        cls.stale_genesis_data['global-det'] = []
        cls.stale_genesis_data['tcgen'] = []

        if ('global-det' in cls.dt_mods_genesis and cls.dt_mods_genesis['global-det']) or \
                ('tcgen' in cls.dt_mods_genesis and cls.dt_mods_genesis['tcgen']):
            # at least we have some data
            cls.have_genesis_data = True

        cls.update_reload_button_color_for_genesis()

    @classmethod
    def update_labels_for_mouse_hover(cls, lat=None, lon=None):
        if not (lat) or not (lon):
            return

        # Update label for mouse cursor position on map first
        cls.label_mouse_coords.config(text=f"({lat:>8.4f}, {lon:>9.4f})")

        if EventManager.get_blocking_purpose():
            # blocking for zoom, measure
            return

        # Next, find nearest point (within some bounding box, as we want to be selective)
        # Define a bounding box around the cursor for initial query (in degrees)
        buffer = MOUSE_SELECT_IN_DEGREES  # Adjust this value based on desired precision
        bounding_box = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

        # Query the R-tree for points within the bounding box
        possible_matches = list(cls.rtree_idx.intersection(bounding_box, objects=True))

        # Calculate the geodesic distance and find the nearest point (nearest_point_index)
        min_distance = float('inf')
        # a sorted cyclic dict that has the item number enumerated
        # has a get() the next enumerated number and the key (the point_index tuple) in a cycle (sorted by value, which will be a datetime)
        cls.nearest_point_indices_overlapped = SortedCyclicEnumDict()
        for item in possible_matches:
            unmapped_point_index = item.id
            internal_id, tc_index, point_index = cls.rtree_tuple_index_mapping[unmapped_point_index]
            point = cls.plotted_tc_candidates[tc_index][1][point_index]
            item_is_overlapping = False
            if internal_id in cls.hidden_tc_candidates:
                continue
            if len(cls.nearest_point_indices_overlapped):
                overlapping_internal_id, overlapping_tc_index, overlapping_point_index = cls.nearest_point_indices_overlapped.get_first_key()
                possible_overlapping_point = cls.plotted_tc_candidates[overlapping_tc_index][1][
                    overlapping_point_index]
                lon_diff = round(abs(possible_overlapping_point['lon'] - point['lon']), 3)
                lat_diff = round(abs(possible_overlapping_point['lat'] - point['lat']), 3)
                if lon_diff == 0.0 and lat_diff == 0.0:
                    item_is_overlapping = True

            # check to see if it is an almost exact match (~3 decimals in degrees) to approximate whether it is an overlapped point
            if item_is_overlapping:
                # this will likely be an overlapped point in the grid
                cls.nearest_point_indices_overlapped[(internal_id, tc_index, point_index)] = point['valid_time']
                # min distance should not significantly change (we are using the first point as reference for overlapping)
            else:
                distance = cls.calculate_distance((lon, lat), (point['lon'], point['lat']))
                if distance < min_distance:
                    # not an overlapping point but still closer to cursor, so update
                    # first clear any other points since this candidate is closer and does not have an overlapping point
                    cls.nearest_point_indices_overlapped = SortedCyclicEnumDict()
                    cls.nearest_point_indices_overlapped[(internal_id, tc_index, point_index)] = point['valid_time']
                    min_distance = distance

        # Update the labels if a nearest point is found within the threshold
        total_num_overlapped_points = len(cls.nearest_point_indices_overlapped)
        if total_num_overlapped_points > 0:
            overlapped_point_num, nearest_point_index = cls.nearest_point_indices_overlapped.next_enum_key_tuple()
            internal_id, tc_index, point_index = nearest_point_index
            cls.update_tc_status_labels(tc_index, point_index, overlapped_point_num, total_num_overlapped_points)
            # get the nearest_point
            point = cls.plotted_tc_candidates[tc_index][1][point_index]
            lon = point['lon']
            lat = point['lat']
            cls.update_circle_patch(lon=lon, lat=lat)
        else:
            # clear the label if no point is found? No.
            #   Not only will this prevent the constant reconfiguring of labels, it allows the user more flexibility
            # cls.update_tc_status_labels()
            # Do clear the circle though as it might be obtrusive
            cls.clear_circle_patch()

    @classmethod
    def update_mode(cls):
        if cls.mode == "ADECK":
            cls.genesis_mode_frame.pack_forget()
            cls.adeck_mode_frame.pack(side=tk.TOP, fill=tk.X)
        else:
            cls.adeck_mode_frame.pack_forget()
            cls.genesis_mode_frame.pack(side=tk.TOP, fill=tk.X)

    @classmethod
    def update_plotted_list(cls, internal_id, tc_candidate):
        # zero indexed
        tc_index = len(cls.plotted_tc_candidates)
        for point_index, point in enumerate(tc_candidate):  # Iterate over each point in the track
            lat, lon = point['lat'], point['lon']
            # Can't use a tuple (tc_index, point_index) as the index so use a mapped index
            cls.rtree_idx.insert(cls.rtree_tuple_point_id, (lon, lat, lon, lat))
            cls.rtree_tuple_index_mapping[cls.rtree_tuple_point_id] = (internal_id, tc_index, point_index)
            cls.rtree_tuple_point_id += 1

        cls.plotted_tc_candidates.append((internal_id, tc_candidate))

    @classmethod
    def update_reload_button_color_for_deck(cls):
        if cls.stale_urls['adeck']:
            cls.reload_button_adeck.configure(style='Red.TButton')
        elif cls.stale_urls['bdeck']:
            cls.reload_button_adeck.configure(style='Orange.TButton')
        elif cls.stale_urls['tcvitals']:
            cls.reload_button_adeck.configure(style='Yellow.TButton')
        else:
            cls.reload_button_adeck.configure(style='White.TButton')

    @classmethod
    def update_reload_button_color_for_genesis(cls):
        if cls.stale_genesis_data['global-det']:
            cls.reload_button_genesis.configure(style='Red.TButton')
        elif cls.stale_genesis_data['tcgen']:
            cls.reload_button_genesis.configure(style='Orange.TButton')
        else:
            cls.reload_button_genesis.configure(style='White.TButton')

    @classmethod
    def update_rvor_contour_renderable_after_zoom(cls):
        if cls.overlay_rvor_contour_dict and 'ids' in cls.overlay_rvor_contour_dict:
            extent, min_span_lat_deg, min_span_lon_deg = cls.get_contour_min_span_deg()
            renderable_ids = set()
            span_lons = cls.overlay_rvor_contour_dict['contour_span_lons']
            span_lats = cls.overlay_rvor_contour_dict['contour_span_lats']
            for contour_id in cls.overlay_rvor_contour_dict['ids']:
                span_lon = span_lons[contour_id]
                span_lat = span_lats[contour_id]
                if span_lon >= min_span_lon_deg and span_lat >= min_span_lat_deg:
                    renderable_ids.add(contour_id)
            cls.overlay_rvor_contour_dict['renderable_ids'] = renderable_ids

    # Update rvor levels
    @classmethod
    def update_rvor_levels(cls):
        # Update the global variable with the new values
        global SELECTED_PRESSURE_LEVELS

        new_SELECTED_PRESSURE_LEVELS = [level for level, var in cls.level_vars.items() if var.get()]

        if SELECTED_PRESSURE_LEVELS == new_SELECTED_PRESSURE_LEVELS:
            return

        SELECTED_PRESSURE_LEVELS = new_SELECTED_PRESSURE_LEVELS

        contours = cls.overlay_rvor_contour_dict['contour_objs']
        labels = cls.overlay_rvor_contour_dict['label_objs']
        for objs_dict in [contours, labels]:
            for _, obj_list in objs_dict.items():
                # for obj in obj_list:
                # don't remove bbox as that will automatically get removed
                try:
                    obj_list[0].remove()
                except:
                    traceback.print_exc()
        cls.ax.set_yscale('linear')
        cls.display_custom_overlay()
        cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def update_selection_info_label(cls):
        # cls.genesis_selection_info_label.config(text="---.- kt")
        internal_ids = cls.get_selected_internal_storm_ids()
        num_tracks = 0
        if internal_ids:
            num_tracks = len(internal_ids)
        num_tracks_str = ""

        if num_tracks > 0:
            num_tracks_str=f"# Selected: Tracks: {num_tracks}"
            filtered_candidates = [(iid, tc) for iid, tc in cls.plotted_tc_candidates if
                                   iid in internal_ids]
            selected_model_names = set()
            for internal_id, tc in filtered_candidates:
                if tc and tc[0] and 'model_name' in tc[0]:
                    selected_model_names.add(tc[0]['model_name'])
            num_selected_model_names = len(selected_model_names)
            num_tracks_str += f", Models: {num_selected_model_names}"
        if cls.mode == "GENESIS":
            cls.genesis_selection_info_label.config(text=num_tracks_str)
        elif cls.mode == "ADECK":
            cls.adeck_selection_info_label.config(text=num_tracks_str)

    @classmethod
    def update_tc_status_labels(cls,tc_index=None, tc_point_index=None, overlapped_point_num=0, total_num_overlapped_points=0):
        # may not have init interface yet
        try:
            if tc_index is None or tc_point_index is None or len(cls.plotted_tc_candidates) == 0:
                cls.label_mouse_hover_matches.config(text="0  ", style="FixedWidthWhite.TLabel")
                cls.label_mouse_hover_info_coords.config(text="(-tt.tttt, -nnn.nnnn)")
                cls.label_mouse_hover_info_valid_time.config(text="YYYY-MM-DD hhZ")
                cls.label_mouse_hover_info_model_init.config(text="YYYY-MM-DD hhZ")
                cls.label_mouse_hover_info_vmax10m.config(text="---.- kt")
                cls.label_mouse_hover_info_mslp.config(text="----.- hPa")
                cls.label_mouse_hover_info_roci.config(text="---- km")
                cls.label_mouse_hover_info_isobar_delta.config(text="--- hPa")
            else:
                # list of dicts (points in time) for tc candidate
                internal_id, tc_candidate = cls.plotted_tc_candidates[tc_index]
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

                cls.label_mouse_hover_matches.config(text=f"{overlapped_point_num}/{total_num_overlapped_points}")
                if total_num_overlapped_points > 1:
                    cls.label_mouse_hover_matches.config(style="FixedWidthRed.TLabel")
                else:
                    cls.label_mouse_hover_matches.config(style="FixedWidthWhite.TLabel")
                cls.label_mouse_hover_info_coords.config(text=f"({lat:>8.4f}, {lon:>9.4f})")
                if valid_time:
                    cls.label_mouse_hover_info_valid_time.config(text=valid_time)
                else:
                    cls.label_mouse_hover_info_valid_time.config(text="YYYY-MM-DD hhZ")
                if init_time:
                    cls.label_mouse_hover_info_model_init.config(text=f"{model_name:>4} {init_time}")
                else:
                    if model_name == "TCVITALS" and valid_time:
                        cls.label_mouse_hover_info_model_init.config(text=f"{model_name:>4} {valid_time}")
                    else:
                        cls.label_mouse_hover_info_model_init.config(text="YYYY-MM-DD hhZ")
                if vmax10m:
                    cls.label_mouse_hover_info_vmax10m.config(text=f"{vmax10m:>5.1f} kt")
                else:
                    cls.label_mouse_hover_info_vmax10m.config(text="---.- kt")
                if mslp:
                    cls.label_mouse_hover_info_mslp.config(text=f"{mslp:>6.1f} hPa")
                else:
                    cls.label_mouse_hover_info_mslp.config(text="----.- hPa")
                if roci:
                    cls.label_mouse_hover_info_roci.config(text=f"{roci:>4.0f} km")
                else:
                    cls.label_mouse_hover_info_roci.config(text="---- km")
                if isobar_delta:
                    cls.label_mouse_hover_info_isobar_delta.config(text=f"{isobar_delta:>3.0f} hPa")
                else:
                    cls.label_mouse_hover_info_isobar_delta.config(text="--- hPa")
        except:
            traceback.print_exc()
            pass

    @classmethod
    def update_toggle_selection_loop_button_color(cls):
        if cls.selection_loop_mode:
            cls.toggle_selection_loop_button.configure(style='YellowAndBorder.TButton')
        else:
            cls.toggle_selection_loop_button.configure(style='WhiteAndBorder.TButton')

    @classmethod
    def zoom_in(cls, step_zoom=False, extents=None):
        if extents:
            cls.ax.set_extent([extents[0], extents[1], extents[2], extents[3]], crs=ccrs.PlateCarree())
            x0 = extents[0]
            y0 = extents[1]
            x1 = extents[2]
            y1 = extents[3]

            extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]

            # Calculate the aspect ratio of the zoom rectangle
            zoom_aspect_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])

            # Calculate the aspect ratio of the canvas frame
            frame_width_pixels = cls.canvas_frame.winfo_width()
            frame_height_pixels = cls.canvas_frame.winfo_height()
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

            cls.ax.set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree())

            cls.clear_storm_extrema_annotations()
            cls.update_axes()
            AnnotatedCircles.changed_extent()
            SelectionLoops.changed_extent()
            cls.update_selection_info_label()
            cls.measure_tool.changed_extent()
            cls.redraw_fig_canvas(stale_bg=True)
            cls.rvor_labels_new_extent()
            cls.redraw_fig_canvas(stale_bg=True)

        elif step_zoom:
            extent = cls.ax.get_extent()
            lon_diff = extent[1] - extent[0]
            lat_diff = extent[3] - extent[2]
            lon_center, lat_center = cls.last_cursor_lon_lat
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

            cls.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
            cls.clear_storm_extrema_annotations()
            cls.update_axes()
            AnnotatedCircles.changed_extent()
            SelectionLoops.changed_extent()
            cls.update_selection_info_label()
            cls.measure_tool.changed_extent()
            cls.redraw_fig_canvas(stale_bg=True)
            cls.rvor_labels_new_extent()
            cls.redraw_fig_canvas(stale_bg=True)

        # elif cls.zoom_rect and (None not in cls.zoom_rect) and len(cls.zoom_rect) == 4:
        # 2d checks if valid rect and that x's aren't close and y's aren't close
        elif cls.zoom_selection_box:
            if not cls.zoom_selection_box.is_2d():
                cls.zoom_selection_box.destroy()
                cls.zoom_selection_box = None
                return

            x0 = cls.zoom_selection_box.lon1
            y0 = cls.zoom_selection_box.lat1
            x1 = cls.zoom_selection_box.lon2
            y1 = cls.zoom_selection_box.lat2

            cls.zoom_selection_box.destroy()
            cls.zoom_selection_box = None

            extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]

            # Calculate the aspect ratio of the zoom rectangle
            zoom_aspect_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])

            # Calculate the aspect ratio of the canvas frame
            frame_width_pixels = cls.canvas_frame.winfo_width()
            frame_height_pixels = cls.canvas_frame.winfo_height()
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

            cls.ax.set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree())

            cls.clear_storm_extrema_annotations()
            cls.update_axes()
            AnnotatedCircles.changed_extent()
            SelectionLoops.changed_extent()
            cls.update_selection_info_label()
            cls.measure_tool.changed_extent()
            cls.redraw_fig_canvas(stale_bg=True)
            cls.rvor_labels_new_extent()
            cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def zoom_out(cls, max_zoom=False, step_zoom=False):
        extent = cls.ax.get_extent()
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
                cls.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
                cls.clear_storm_extrema_annotations()
                cls.update_axes()
                AnnotatedCircles.changed_extent()
                SelectionLoops.changed_extent()
                cls.update_selection_info_label()
                cls.measure_tool.changed_extent()
                cls.redraw_fig_canvas(stale_bg=True)
                cls.rvor_labels_new_extent()
                cls.redraw_fig_canvas(stale_bg=True)

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
                cls.ax.set_extent(new_extent, crs=ccrs.PlateCarree())
                cls.clear_storm_extrema_annotations()
                cls.update_axes()
                AnnotatedCircles.changed_extent()
                SelectionLoops.changed_extent()
                cls.update_selection_info_label()
                # do measure last as we are going to remove and redraw it
                cls.measure_tool.changed_extent()
                cls.redraw_fig_canvas(stale_bg=True)
                cls.rvor_labels_new_extent()
                cls.redraw_fig_canvas(stale_bg=True)

    @classmethod
    def zoom_to_basin_dialog(cls, event=None):
        dialog = tk.Toplevel(cls.root)
        dialog.title('Zoom to Basin')
        dialog.geometry(f"300x200+{int(cls.root.winfo_width()/2-150)}+{int(cls.root.winfo_height()/2-100)}")

        listbox = tk.Listbox(dialog, bg=default_bg, fg=default_fg)
        # show in order (first letter)
        for basin_name in sorted(basin_extents.keys(), key=lambda x: ['N', 'E', 'W', 'I', 'S'].index(x[0])):
            listbox.insert(tk.END, basin_name)

        listbox.pack(fill=tk.BOTH, expand=1)
        listbox.selection_set(0)
        listbox.focus_set()
        listbox.see(0)

        def cancel():
            dialog.destroy()

        def select_basin(event):
            selected_basin = listbox.get(listbox.curselection())
            cancel()
            cls.zoom_in(extents=basin_extents[selected_basin])

        listbox.bind('<Return>', select_basin)
        listbox.bind('<Up>', lambda event: listbox.select_set(0))
        listbox.bind('<Down>', lambda event: listbox.select_set(0))
        listbox.bind("<Escape>", lambda event: cancel())
        ttk.Button(dialog, text='OK', command=select_basin).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(dialog, text='Cancel', command=cancel).pack(side=tk.LEFT, padx=5, pady=5)


if AUTO_DOWNLOAD_LATEST_OHC:
    print("Updating latest OHC files...")
    OHC_NC_PATHS = download_latest_ohc_nc_files()
    print("Done updating OHC files.")

if AUTO_DOWNLOAD_LATEST_SST:
    print("Updating latest SST file...")
    SST_NC_PATH = download_latest_sst_nc_file()
    print("Done updating SST file.")

# Start program GUI
if __name__ == "__main__":
    tk_root = tk.Tk()

    # For analysis plots before drawing we want to guess the width, height available full screen
    FULL_SCREEN_WIDTH = tk_root.winfo_screenwidth()
    FULL_SCREEN_HEIGHT = tk_root.winfo_screenheight()

    available_fonts = font.families()
    mono_fonts = [x.lower() for x in available_fonts if 'mono' in x]
    mono_font = "Latin Modern Mono"
    if mono_font not in mono_fonts and mono_fonts:
        mono_font = mono_fonts[0]

    # Style configuration for ttk widgets
    tk_style = ttk.Style()
    tk_style.theme_use('clam')  # Ensure using a theme that supports customization
    default_bg = "#000000"
    default_fg = "#FFFFFF"
    tk_style.configure("TListbox", background=default_bg, foreground=default_fg)
    tk_style.configure("TButton", background=default_bg, foreground=default_fg)

    tk_style.configure("White.TButton", background=default_bg, foreground="white")
    tk_style.configure("Red.TButton", background=default_bg, foreground="red")
    tk_style.configure("Orange.TButton", background=default_bg, foreground="orange")
    tk_style.configure("Yellow.TButton", background=default_bg, foreground="yellow")

    tk_style.configure("YellowAndBorder.TButton", background=default_bg, foreground="yellow", bordercolor="yellow")
    tk_style.configure("WhiteAndBorder.TButton", background=default_bg, foreground="white", bordercolor="white")

    # change hover color
    tk_style.map('TButton', background=[('active', '#444444')])  # change hover color to dark grey

    tk_style.configure("TLabel", background=default_bg, foreground=default_fg)
    tk_style.configure("TEntry", background=default_bg, foreground=default_fg, fieldbackground=default_bg)
    tk_style.configure("FixedWidthWhite.TLabel", font=(mono_font, 12), background=default_bg,
                       foreground="white")
    tk_style.configure("FixedWidthRed.TLabel", font=(mono_font, 12), background=default_bg, foreground="red")

    tk_style.configure("TCheckbutton", background=default_bg, foreground=default_fg)
    # change hover color
    tk_style.map('TCheckbutton', background=[('active', '#555555')],
                 foreground=[('active', default_fg)])  # change hover color to dark grey

    tk_style.configure("TopFrame.TFrame", background=default_bg, foreground=default_fg)
    tk_style.configure("ToolsFrame.TFrame", background=default_bg, foreground=default_fg)
    tk_style.configure("CanvasFrame.TFrame", background=default_bg, foreground=default_fg)

    tk_style.configure("TMessaging", background=default_bg, foreground=default_fg)

    #tk_style.configure('Black.TCombobox', foreground=default_fg)
    tk_style.map('Black.TCombobox', fieldbackground=[('readonly', default_bg)])
    tk_style.map('Black.TCombobox', foreground=[('readonly', default_fg)])
    tk_style.map('Black.TCombobox', selectbackground=[('readonly', default_bg)])
    tk_style.map('Black.TCombobox', selectforeground=[('readonly', default_fg)])

    # Configure the notebook style
    tk_style.configure('TNotebook', background=default_bg, foreground='grey')
    tk_style.configure('TNotebook.Tab', background=default_bg, foreground=default_fg)
    tk_style.map('TNotebook.Tab', background=[('selected', default_bg)], foreground=[('selected', 'pink')])
    tk_style.configure("CanvasFrame", background=default_bg, foreground=default_fg)
    tk_style.configure("CanvasFrame.TFrame", background=default_bg, foreground=default_fg)
    tk_style.map('CanvasFrame.TFrame', background=[('selected', default_bg)], foreground=[('selected', 'pink')])

    tk_root.configure(background=default_bg, borderwidth=0)
    tk_root.option_add('*background', '#000000')
    tk_root.option_add('*foreground', '#FFFFFF')

    app = App.init(tk_root)

    tk_root.mainloop()
