# Generates (TC type) disturbances from global model data (GFS, CMC, ECM, NAV):
# Uses data gathered from download.py

# EXPERIMENTAL
# Work in progress (do not use)
# Only tested on linux (python 3.9.1, and python (conda/pypc) 3.12.4)

# find TC genesis from models based on work similar to FSU's methodology

# only focused on 1. so far:
# [1] https://journals.ametsoc.org/view/journals/wefo/28/6/waf-d-13-00008_1.xml
# [2] https://journals.ametsoc.org/view/journals/wefo/31/3/waf-d-15-0157_1.xml?tab_body=fulltext-display
# [3] https://journals.ametsoc.org/view/journals/wefo/32/1/waf-d-16-0072_1.xml

# for ROCI:
# [4] https://journals.ametsoc.org/view/journals/wefo/28/6/waf-d-13-00008_1.xml

# units are metric, but scale may be mixed between internal representations of data, thresholds, and printouts

# for downloads of navgem (metview distance didn't work with the navgem grib1 files):
# converting to grib2 fixes this issue (do along with downloading: add .grib2 and delete original)
# grib_set -s edition=2 US058GMET-GR1mdl.0018_0056_00000F0RL2023110112_0102_000000-000000pres_msl US058GMET-GR1mdl.0018_0056_00000F0RL2023110112_0102_000000-000000pres_msl.grib2

# partial TODO
#    recalculate overnight...
#
#    chart disturbances
#        todo assigning unique storm tracks (moving box method to bin disturbances...?)
#        map.save for properties mouse hover(low priority)
#
#    extend to handle multiple time_steps/bufrs for TC detection
#        individually a single criteria being met is a disturbance,
#             start of 24 continuous hours of meeting criterias a tc, need to do box tracking? (multiple possibilities)
#                  calculate from disturbances db
#                  generate storm ids
#             store as separate db
#
#    automatically download model files (UKMET seems infeasible as it costs much money)
#        GFS done, working
#             delete old data automatically? (archive for now for testing)
#        for partial splits (NAVGEM/CMC) need to note expected number of files (so we don't miss out on 10m wind speed for instance or relative vorticity)
#             need to modify calc_disturbances_by_model_name_date_and_time_steps() so we can only compute once we have the expected number of gribs per model
#                 use expected_num_grib_files_by_model_name dict`
#

# thresholds for disturbances (reversed from text output from FSU)
disturbance_thresholds_path = 'disturbance_thresholds.json'

disturbances_db_file_path = 'disturbances.db'

# this is for accessing by model and timestep
tc_disturbances_db_file_path = 'tc_disturbances.db'
# this is for accessing by model and storm (internal component id)
tc_candidates_db_file_path = 'tc_candidates.db'

# cylindrical (no spherical corrections), same as GEMPAK (used by FSU apparently)
default_vorticity_method = 'metpy'
# metview's vorticity method (differs from metpy/GEPAK):
#vorticity_method = 'metview'

# shape file for placing lat,lon in basins which we are classifying
# each has an attr`ibute called 'basin_name', with CPAC & EPAC combined as EPAC
# basins are NATL, EPAC, WPAC, IO, SH
shape_file = 'shapes/basins.shp'

model_data_folders_by_model_name = {
    'GFS': '/home/db/metview/JRPdata/globalmodeldata/gfs',
    'ECM': '/home/db/metview/JRPdata/globalmodeldata/ecm',
    'CMC': '/home/db/metview/JRPdata/globalmodeldata/cmc',
    'NAV': '/home/db/metview/JRPdata/globalmodeldata/nav'
}

# where to save the disturbance maps
disturbance_maps_folder = '/home/db/Documents/JRPdata/cyclone-genesis/disturbance_maps'

# path to font for charts
font_path = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'

# for debugging MSLP isobars (depth first search)
debug_isobar_images = False

# save vorticity calculations
debug_save_vorticity = False

# output calculation timing for debugging & optimization
debug_calc_exec_time = False

# write networkx graphs (original and reduced) of TC tracks (in graphs/ folder)
write_graphs = True

graphs_folder = '/home/db/Documents/JRPdata/cyclone-genesis/graphs'

# save disturbance maps
save_disturbance_maps = True

# Several libraries were used for debugging/development that might be optionally removed

import json
import pygrib
import re
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import traceback
import sys
import copy
import time
import os

import warnings

from datetime import datetime, timedelta
import sqlite3

from matplotlib.patches import Circle

import metview as mv

import matplotlib.pyplot as plt
import pytz

# may need to modify lzma.py in python folder to get this to work (pip install backports-lzma)
# this is for metpy
try:
    import lzma
except ImportError:
    import backports.lzma as lzma

gdf = gpd.read_file(shape_file)

disturbance_criteria_names_map = {
    'WS': 'vmax925',
    'THKN': 'gp250_850_thickness',
    'RV': 'rv850max'
}

# shape file attribute names to threshold names
shape_basin_names_to_threshold_names_map = {
    'NATL': 'AL',
    'WPAC': 'WP',
    'IO': 'IO',
    'SH': 'SH',
    'EPAC': 'EP'
}

model_time_step_str_format = {
    'GFS': '03',
    'ECM': '00',
    'CMC': '03',
    'NAV': '03'
}

# not including GFS analysis files (.anl)
total_model_time_steps = {
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
all_time_steps_by_model = {
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

# NAVGEM has relative vorticity grib (if using USGODAE); if using NCEP we need the u/v wind speed gribs for 850
# NCEP might be faster for NAVGEM (and for consistency of vorticity calculations, calculate it ourselves)
expected_num_grib_files_by_model_name = {
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
    'index'
]

# hours between model runs
model_interval_hours = {
    'GFS': 6,
    'ECM': 12,
    'CMC': 12,
    'NAV': 6
}

time_step_re_str_by_model_name = {
    'GFS': r'.*?\.f(?P<time_step>\d\d\d)_',
    'ECM': r'.*?-(?P<time_step>\d+)h-oper-fc',
    'CMC': r'.*?\d+_P(?P<time_step>\d+)',
    'NAV': r'.*?navgem_\d{10}f(?P<time_step>\d\d\d)'
}

with open(disturbance_thresholds_path, 'r') as f:
    disturbance_thresholds = json.loads(f.read())

colocated_box_radii = [5]
next_colocated_box_radii = [3]
largest_colocated_radius = colocated_box_radii[-1]
largest_next_colocated_radius = next_colocated_box_radii[-1]
# edges between nodes in same timestep (box of colocated radius degrees)
largest_colocated_edge_type = f'colocated_{largest_colocated_radius}'
# these are edges to the next timestep
next_colocated_edge_type = f'next_colocated_{largest_next_colocated_radius}'
# colors for nx graphs (disjoint segments are 'split' and colored red)
nx_default_color = 'skyblue'
nx_split_color = 'red'

# must set to [] anywhere there is an exception and the stack doesn't get popped properly
debug_time_start_stack = []

# Reset the warnings to its original state here
warnings.resetwarnings()

meters_to_knots = 1.943844

# avoid deprecation
def datetime_utcnow():
    return datetime.now(pytz.utc).replace(tzinfo=None)

#########################################################
########       CALCULATE DISTURBANCES            ########
#########################################################

def traverse_contour(grid):
    x_lookup = [0,1,0,-1]
    y_lookup = [-1,0,1,0]

    grid_shape = np.shape(grid)
    visited = np.zeros(grid_shape, dtype=bool)

    y_last = grid_shape[0] - 1
    x_last = grid_shape[1] - 1

    center_y = grid_shape[0] // 2
    center_x = grid_shape[1] // 2

    north_stack = []
    south_stack = []
    east_stack = []
    west_stack = []
    cardinal_stacks = [north_stack, east_stack, south_stack, west_stack]

    for direction in range(4):
        y = center_y
        x = center_x

        current_stack = cardinal_stacks[direction]
        stack2 = []

        found_isosurface = False
        while True:
            visited[y][x] = True
            if not grid[y][x]:
                last_direction = (direction-1)%4
                new_y = y + y_lookup[last_direction]
                new_x = x + x_lookup[last_direction]
                if (0 <= new_y <= y_last) and (0 <= new_x <= x_last):
                    current_stack.append((last_direction, new_y, new_x))

                last_direction = (direction+1)%4
                new_y = y + y_lookup[last_direction]
                new_x = x + x_lookup[last_direction]
                if (0 <= new_y <= y_last) and (0 <= new_x <= x_last):
                    stack2.append((last_direction, new_y, new_x))

                new_y = y + y_lookup[direction]
                new_x = x + x_lookup[direction]
                if (0 <= new_y <= y_last) and (0 <= new_x <= x_last) and not visited[new_y][new_x]:
                    y = new_y
                    x = new_x
                else:
                    reached_bounds = True
                    break
            else:
                found_isosurface = True
                break

        if (not found_isosurface) and reached_bounds:
            return False

        current_stack.extend(stack2)

    while north_stack or south_stack or east_stack or west_stack:
        for stack in cardinal_stacks:
            if stack:
                last_direction, y, x = stack.pop()
                if visited[y][x]:
                    continue

                visited[y][x] = True
                if not grid[y][x]:
                    if (y == 0) or (x == 0) or (y == y_last) or (x == x_last):
                        return False

                    for neighbor_direction in [(last_direction + 1) % 4, last_direction, (last_direction - 1) % 4]:
                        neighbor_y = y + y_lookup[neighbor_direction]
                        neighbor_x = x + x_lookup[neighbor_direction]
                        if (0 <= neighbor_y <= y_last) and (0 <= neighbor_x <= x_last):
                            if not visited[neighbor_y][neighbor_x]:
                                stack.append((neighbor_direction, neighbor_y, neighbor_x))

    return True

# checks for a closed isobar (not at integer steps) using traverse_contour
# this will only return True or False, as opposed to the dfs version (which finds the contour)
#    this is since the visited is not the contour of the isosurface because of the search method
def has_closed_isobar_traverse_contour(mslp_data, grid_resolution, candidate, isobar_threshold=2.0, isobar_search_radius_degrees = 15):
    isobar_neighborhood_size = calculate_neighborhood_size(isobar_search_radius_degrees, grid_resolution)

    try:
        x = candidate['x_value']
        y = candidate['y_value']
        minima_value = mslp_data[x][y]

        # Create a modified neighborhood for isobar calculation

        x_min = x - isobar_neighborhood_size
        x_max = x + isobar_neighborhood_size + 1
        y_min = y - isobar_neighborhood_size
        y_max = y + isobar_neighborhood_size + 1

        in_bounds = ((x_min >= 0) and
            (x_max <= mslp_data.shape[0]) and
            (y_min >= 0) and
            (y_max <= mslp_data.shape[1]))

        if in_bounds:
            # the normal case (not edges of array)
            neighborhood = mslp_data[x_min:x_max, y_min:y_max]
        else:
            # handle indices at the boundaries
            neighborhood = extract_2d_neighborhood(mslp_data, (x, y), isobar_neighborhood_size)

        # create a binary image (True are values that are at the threshold, with the center being False)
        binary_neighborhood = (neighborhood - minima_value - isobar_threshold) >= 0

        return traverse_contour(binary_neighborhood)

    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return None

# finds a closed isobar (not at integer steps) using dfs
# this will return a visited array of the neighborhood
#  the visited are those values less than the isobar_threshold difference from the center enclosed by an isoline
#  if there is no closed isobar within the neighborhood it will return None
# this is modified from previous version, so takes the (float) neighborhood as input, along with the center minima value, radius, and threshold from minima
def find_closed_isobar_in_neighborhood(neighborhood, minima_value, isobar_neighborhood_size, isobar_threshold = 2.0):
    def dfs(x, y):
        nonlocal visited
        stack = [(x, y)]

        while stack:
            x, y = stack.pop()
            visited[x][y] = True

            # Check N, S, E, W neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for nx, ny in neighbors:
                if 0 <= nx < isobar_neighborhood_size * 2 + 1 and 0 <= ny < isobar_neighborhood_size * 2 + 1:
                    if not visited[nx][ny]:
                        if nx == 0 or ny == 0 or nx == isobar_neighborhood_size * 2 or ny == isobar_neighborhood_size * 2:
                            return False
                        if not binary_neighborhood[nx][ny]:
                            stack.append((nx, ny))
        return True

    try:
        # create a binary image (True are values that are at the threshold, with the center being False)
        binary_neighborhood = (neighborhood - minima_value - isobar_threshold) >= 0

        # Initialize visited array for DFS
        visited = np.zeros_like(binary_neighborhood, dtype=bool)

        # Flag to track if a closed path is found
        path_found = dfs(isobar_neighborhood_size, isobar_neighborhood_size)

        if path_found:
            return copy.deepcopy(visited)
        else:
            return None

    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return None

# todo remove other storms?
# only for finding disturbance center 'mass' coordinate, similar to [4]
def find_feature_center(neighborhood):
    # Find the center of the feature (assuming it's connected)
    float_x, float_y = np.argwhere(neighborhood).mean(axis=0)
    return (int(round(float_x)), int(round(float_y)))

# this is different than [4] (closed isobar where all radial legs are met)
# this allows for a closed isobar that is not perfectly round, extending outward up until before the isobar is no longer closed (above the threshold)
# each leg is extended to the radius used to create the neighborhood (15 deg or ~1667 km)
# [4] legs extends to 1500km (~13.3 deg)
def find_average_radius(mslp_fs, candidate, neighborhood, num_legs=576):
    # Get the center of the neighborhood
    neighborhood_center = (neighborhood.shape[0] // 2, neighborhood.shape[1] // 2)

    # need to get distances from mass center in mslp_data
    # use mv.distance and mslp_fs to do so
    # todo? this can likely be optimized further
    mslp_ds = mslp_fs.to_dataset()
    mv_lats = mslp_ds.latitude.to_numpy()
    mv_lons = mslp_ds.longitude.to_numpy()
    mass_x = candidate['mass_x']
    mass_y = candidate['mass_y']
    mass_lat, mass_lon = array_indices_to_lat_lon(mass_x, mass_y, mv_lats, mv_lons)
    dist_fs = mslp_fs.distance(mass_lat, mass_lon)
    distances_ds = dist_fs.to_dataset()
    if type(distances_ds) is not np.ndarray:
        for var in distances_ds:
            if 'msl' in var:
                distances = distances_ds[var].to_numpy()
                break
    else:
        distances = distances_ds

    # Calculate the angle step between legs
    angle_step = 2 * np.pi / num_legs

    # Initialize a list to store distances for each leg (meters)
    leg_distances = []
    # also store in units of grid_resolution
    leg_distances_units = []

    for i in range(num_legs):
        # Calculate the direction vector for the current leg
        direction = np.array([np.cos(i * angle_step), np.sin(i * angle_step)])

        # Initialize the leg distance
        leg_distance_units = 0

        current_point = copy.deepcopy(neighborhood_center)
        neighborhood_last_x = neighborhood_center[0]
        neighborhood_last_y = neighborhood_center[1]
        while True:
            current_point = current_point + direction
            x, y = current_point.astype(int)

            # Check if the current point is out of bounds or False
            if x < 0 or x >= neighborhood.shape[0] or y < 0 or y >= neighborhood.shape[1] or not neighborhood[x, y]:
                break

            # Increment the leg distance
            leg_distance_units += 1
            neighborhood_last_x = x
            neighborhood_last_y = y

        # calculate the leg_distance (for append) in units of meters
        # transform the leg coordinates from neighborhood coordinates to the global coordinates
        neighborhood_center_x = int(neighborhood.shape[0] // 2)
        neighborhood_center_y = int(neighborhood.shape[1] // 2)
        unwrapped_leg_x = mass_x + (neighborhood_last_x - neighborhood_center_x)
        unwrapped_leg_y = mass_y + (neighborhood_last_y - neighborhood_center_y)
        # calculate the leg_distance (for append) in units of meters
        leg_x = int(unwrapped_leg_x % mv_lats.shape[0])
        # mv lons are 1d whereas pygrib lons are 2d
        leg_y = int(unwrapped_leg_y % mv_lons.shape[0])
        leg_distance = distances[leg_x][leg_y]
        # if leg_x and leg_y is at the center this is supposed to be 0 (but it is nan)
        if not (leg_distance > 0):
            leg_distance = 0
        leg_distances.append(leg_distance)

        leg_distances_units.append(leg_distance_units)

    # Calculate the average of leg distances
    average_radius = np.mean(leg_distances)
    average_radius_units = np.mean(leg_distances_units)
    return average_radius, average_radius_units

# Typhoon Tip was < 10 deg. (use 15 to cover even larger possible future storms)
def get_outermost_closed_isobar(mslp_data, grid_resolution, candidate, isobar_search_radius_degrees = 15):
    mslp_candidate = copy.deepcopy(candidate)

    # get neighborhood once
    isobar_neighborhood_size = calculate_neighborhood_size(isobar_search_radius_degrees, grid_resolution)
    x = mslp_candidate['x_value']
    y = mslp_candidate['y_value']
    mslp_center_minima_value = mslp_data[x][y]

    # Create a modified neighborhood for isobar calculation

    x_min = x - isobar_neighborhood_size
    x_max = x + isobar_neighborhood_size + 1
    y_min = y - isobar_neighborhood_size
    y_max = y + isobar_neighborhood_size + 1

    in_bounds = ((x_min >= 0) and
        (x_max <= mslp_data.shape[0]) and
        (y_min >= 0) and
        (y_max <= mslp_data.shape[1]))

    if in_bounds:
        # the normal case (not edges of array)
        neighborhood = mslp_data[x_min:x_max, y_min:y_max]
    else:
        # handle indices at the boundaries
        neighborhood = extract_2d_neighborhood(mslp_data, (x, y), isobar_neighborhood_size)

    # center of closest closed isobar (1 hPa away)
    visited = find_closed_isobar_in_neighborhood(neighborhood, mslp_center_minima_value, isobar_neighborhood_size, isobar_threshold=1)
    neighborhood_mass_x, neighborhood_mass_y = find_feature_center(visited)
    neighborhood_center_x = int(visited.shape[0] // 2)
    neighborhood_center_y = int(visited.shape[1] // 2)
    unwrapped_mass_x = candidate['x_value'] + (neighborhood_mass_x - neighborhood_center_x)
    unwrapped_mass_y = candidate['y_value'] + (neighborhood_mass_y - neighborhood_center_y)
    mass_x = int(unwrapped_mass_x % mslp_data.shape[0])
    mass_y = int(unwrapped_mass_y % mslp_data.shape[1])
    mslp_candidate['mass_x'] = mass_x
    mslp_candidate['mass_y'] = mass_y

    last_visited = None
    last_isobar_threshold = None
    for isobar_threshold in range(2, 200):
        visited = find_closed_isobar_in_neighborhood(neighborhood, mslp_center_minima_value, isobar_neighborhood_size, isobar_threshold=isobar_threshold)
        if visited is not None:
            last_visited = visited
            last_isobar_threshold = isobar_threshold
        if visited is None:
            break

    return mslp_candidate, last_isobar_threshold, last_visited

# Define a context manager to temporarily ignore the warning
class SuppressRuntimeWarningVortCalcs:
    def __enter__(self):
        self.original_filters = warnings.filters[:]
        # first two warning are from metpy
        warning_filter_vort_divide = (
            "ignore",
            ".*invalid value encountered in divide.*"
        )
        warnings.filterwarnings(*warning_filter_vort_divide)

        #/tmp/ipykernel_701280/3056967088.py:670: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
        warning_filter_vort_units = (
            "ignore",
            ".*unit of the quantity is stripped.*"
        )
        warnings.filterwarnings(*warning_filter_vort_units)

        # warning from metview
        warnings_filter_vort_grib = (
            "ignore",
            ".*GRIB write support is experimental.*"
        )
        warnings.filterwarnings(*warnings_filter_vort_grib)

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.filters = self.original_filters

def get_calculated_and_missing_time_steps(model_name, model_timestamp):
    model_hour = f'{model_timestamp.hour:02}'
    # remove from this the calculated timesteps
    model_time_steps_missing = set(all_time_steps_by_model[model_name][model_hour])

    disturbance_candidates = get_disturbances_from_db(model_name, model_timestamp)
    time_steps_calculated = set()
    if disturbance_candidates:
        for time_step_str, candidates in disturbance_candidates.items():
            time_step_int = int(time_step_str)
            time_steps_calculated.add(time_step_int)
            # discard will not throw an error even if it is not there
            model_time_steps_missing.discard(time_step_int)
    return sorted(list(time_steps_calculated)), sorted(list(model_time_steps_missing))

# the full timestep string as used in the file names we are parsing from the model data
# include any leading zeros up that make sense (only up to what the model covers)
def convert_model_time_step_to_str(model_name, model_time_step):
    str_format = model_time_step_str_format[model_name]
    return f'{model_time_step:{str_format}}'

# store results in db
def calc_disturbances_by_model_name_date_and_time_steps(model_name, model_timestamp, model_time_steps):
    # test storage for disturbances database
    # last gfs directory (0.5deg)
    debug_timing()

    model_base_dir = model_data_folders_by_model_name[model_name]
    model_date_str_with_hour = datetime.strftime(model_timestamp, '%Y%m%d%H')
    model_date_str = datetime.strftime(model_timestamp, '%Y%m%d')
    model_dir = os.path.join(model_base_dir, model_date_str_with_hour)
    model_hour = datetime.strftime(model_timestamp, '%H')
    model_time_step_re = re.compile(time_step_re_str_by_model_name[model_name])
    all_candidates = {}

    if not os.path.exists(model_dir):
        return all_candidates

    already_calculated_model_time_steps, missing_model_time_steps = get_calculated_and_missing_time_steps(model_name, model_timestamp)
    for model_time_step in model_time_steps:
        model_time_step_int = int(model_time_step)
        if already_calculated_model_time_steps is not None:
            if model_time_step_int in already_calculated_model_time_steps:
                continue

        # do so here since may be live updating
        files = os.listdir(model_dir)
        debug_timing()
        if type(model_time_step) is str:
            model_time_step_str = convert_model_time_step_to_str(model_name, int(model_time_step))
        elif type(model_time_step) is int:
            model_time_step_str = convert_model_time_step_to_str(model_name, model_time_step)

        # may not be a trailing slash or not
        if not os.path.isdir(model_dir):
            model_timestamp_str = os.path.basename(os.path.dirname(model_dir))
        else:
            model_timestamp_str = os.path.basename(model_dir)

        model_step_timestamp = datetime.strptime(model_timestamp_str, '%Y%m%d%H')

        grib_files = []

        for f in files:
            f_ext = f.split('.')[-1]
            if f_ext not in ignored_file_extensions:
                res = re.match(model_time_step_re, f)
                if res:
                    file_model_time_step_str = res['time_step']
                    if file_model_time_step_str == model_time_step_str:
                        grib_files.append(os.path.join(model_dir, f))

        if grib_files:
            res = re.match(model_time_step_re, grib_files[0])
            if res:
                if expected_num_grib_files_by_model_name[model_name] == len(grib_files):
                    model_time_step_str = f"{int(res['time_step']):02}"
                    #print_with_process_id("========================================================================================")
                    print_with_process_id(f"Calculating disturbances (TC candidates) {model_name} for {model_date_str} {model_hour}Z at +{model_time_step}h")
                    #print_with_process_id("========================================================================================")
                    candidates = get_disturbance_candidates_from_split_gribs(grib_files, model_name)
                    if candidates is not None:
                        update_disturbances_db(model_name, model_step_timestamp, model_time_step, candidates)
                        all_candidates[model_time_step_str] = candidates
                else:
                    print_with_process_id(f"Missing files for {model_name} at time step {model_time_step} in {model_dir}")
            else:
                print_with_process_id(f"Could not parse time step for {model_name} at time step {model_time_step} in {model_dir}")
        else:
            print_with_process_id(f"Could not find grib files for {model_name} at time step {model_time_step} in {model_dir}")

        # getting the time taken for executing the code in seconds
        debug_timing(f'Total time taken for one step {model_name}')

    debug_timing(f'Total time taken for all steps {model_name}')

    return all_candidates

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
        print_with_process_id(f"SQLite error (get_disturbances_from_db): {e}")
    finally:
        if conn:
            conn.close()

    return retrieved_data

def update_disturbances_db(model_name, model_timestamp, model_time_step, candidates):
    conn = None
    model_time_step_str = str(model_time_step)
    try:
        # store disturbance candidates in database
        conn = sqlite3.connect(disturbances_db_file_path)
        cursor = conn.cursor()
        # Parallelization:
        # As we are reading in a dict and appending in parallel...
        #   we need to make this exclusive to make sure only the latest dict is used
        #   otherwise we might clobber our work for previous time-steps for the same model & init time
        cursor.execute('BEGIN EXCLUSIVE TRANSACTION')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disturbances (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                data JSON,
                date TEXT,
                is_complete TEXT,
                UNIQUE(model_name, date)
            )
        ''')

        ds = model_timestamp.isoformat()

        cursor.execute('SELECT data FROM disturbances WHERE model_name = ? AND date = ?', (model_name, ds))
        result = cursor.fetchone()

        if result:
            retrieved_data = json.loads(result[0])
        else:
            retrieved_data = {}

        retrieved_data[model_time_step_str] = candidates
        json_data = json.dumps(retrieved_data)

        num_steps_complete = len(retrieved_data.keys())
        hour_str = datetime.strftime(model_timestamp, '%H')
        is_complete = 0
        if num_steps_complete == total_model_time_steps[model_name][hour_str]:
            is_complete = 1

        cursor.execute('INSERT OR REPLACE INTO disturbances (model_name, data, date, is_complete) VALUES (?, ?, ?, ?)', (model_name, json_data, ds, is_complete))
        conn.commit()

    except sqlite3.Error as e:
        print_with_process_id(f"SQLite error (update_disturbances_db): {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

# call with no params before a func call, and then must pass it a func_str on the next call once func is complete
def debug_timing(func_str = None):
    if not debug_calc_exec_time:
        return
    if func_str is None:
        debug_time_start_stack.append(time.time())
        return

    debug_time_start = debug_time_start_stack.pop()
    debug_time_end = time.time()
    print_with_process_id(f'{func_str} execution time (seconds): {debug_time_end - debug_time_start:.1f}')

# rv_to_sign just maps to positive and negative, without it it does an (arbitrary) color mapping
def create_image(array, title, rv_to_sign=True, radius=None):
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.RdBu

    if array.dtype == bool:
        # Convert boolean values to 1.0 (True) and -1.0 (False)
        image = np.where(array, -1.0, 1.0)
    else:
        if rv_to_sign:
            # Map float values to 1.0 (>=0) and -1.0 (<0)
            image = np.where(array >= 0, 1.0, -1.0)
        else:
            image = array

    plt.imshow(image, cmap=cmap, interpolation='none')
    plt.title(title)

    for i in range(array.shape[0] + 1):
        plt.axhline(i - 0.5, color='black', lw=0.5)
        plt.axvline(i - 0.5, color='black', lw=0.5)

    plt.gca().invert_yaxis()
    plt.axis('off')

    if radius is not None:
        circle = Circle((array.shape[1] / 2 - 0.5, array.shape[0] / 2 - 0.5), radius, fill=False, color='yellow')
        plt.gca().add_patch(circle)

    plt.show()

def list_available_parameters(grib_file):
    try:
        # Open the GRIB file
        grbs = pygrib.open(grib_file)

        # Initialize a list to store parameter information
        parameter_info = []

        # Iterate through the GRIB messages and extract parameter information
        for grb in grbs:
            parameter_name = grb.name
            parameter_shortName = grb.shortName
            parameter_unit = grb.units
            level_type = grb.levelType
            level = grb.level
            print_with_process_id(grb)
            parameter_info.append({
                "Parameter Name": parameter_name,
                "Parameter shortName": parameter_shortName,
                "Unit": parameter_unit,
                "Level Type": level_type,
                "Level": level
            })

        # Close the GRIB file
        grbs.close()

        # Print the information for parameters
        for info in parameter_info:
            print_with_process_id("Parameter Name:", info["Parameter Name"])
            print_with_process_id("Parameter shortName:", info["Parameter shortName"])
            print_with_process_id("Unit:", info["Unit"])
            print_with_process_id("Level Type:", info["Level Type"])
            print_with_process_id("Level:", info["Level"])
            print_with_process_id("")

    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)

def convert_to_signed_lon(lon):
    # Convert longitudes from 0-360 range to -180 to +180 range
    return (lon + 180) % 360 - 180

# this is in a number of cells radius from the center (so a neighborhood size of 1 would be 3x3)
def calculate_neighborhood_size(degree_radius, grid_resolution):
    # Calculate the radius in grid points
    # changed to round() rather than truncating to integer
    radius_in_grid_points = int(round(degree_radius / grid_resolution))
    return radius_in_grid_points

def array_indices_to_lat_lon(x, y, lats, lons):
    if len(lats.shape) == 2:
        lat = lats[x, y]
        lon = lons[x, y]
    else:
        lat = lats[x]
        lon = lons[y]
    signed_lon = convert_to_signed_lon(lon)
    return lat, signed_lon

# print candidates
def print_candidates(mslp_minima_list, lats = None, lons = None, meet_all_disturbance_thresholds = False, no_numbering=False):
    n = 0
    for candidate in mslp_minima_list:
        if meet_all_disturbance_thresholds:
            if not candidate['criteria']['all']:
                continue

        basin_str = ""
        if 'basin' in candidate:
            basin = candidate['basin']
            basin_str += f'{basin} Basin, '

        n += 1
        mslp_value = candidate["mslp_value"]

        if lats is not None and lons is not None:
            x = candidate["x_value"]
            y = candidate["y_value"]
            lat, lon = array_indices_to_lat_lon(x, y, lats, lons)
        else:
            lat = candidate['lat']
            lon = candidate['lon']

        formatted_mslp = f"{mslp_value:.1f}".rjust(6, ' ')
        rv_str = ""
        if 'rv850max'in candidate:
            rv_str = "850 RV MAX (*10^-5 1/s): "
            rv_value = candidate['rv850max'] * np.power(10.0,5)
            formatted_rv = f"{rv_value:2.2f}".rjust(6, ' ')
            rv_str += formatted_rv

        thickness_str = ""
        if 'gp250_850_thickness' in candidate:
            thickness_str = ", 250-850 hPa Thickness (m): "
            thickness_value = candidate['gp250_850_thickness']
            formatted_thickness = f"{thickness_value:2.2f}".rjust(6, ' ')
            thickness_str += formatted_thickness

        vmax_str = ""
        if 'vmax925' in candidate:
            vmax_str = ", 925 hPa WS MAX (m/s): "
            vmax_value = candidate['vmax925']
            formatted_vmax = f"{vmax_value:3.2f}".rjust(6, ' ')
            vmax_str += formatted_vmax

        roci_str = ""
        if 'roci' in candidate:
            roci_str = ", ~ROCI (km): "
            roci_value = candidate['roci']
            formatted_roci = f"{roci_value/1000:3.0f}".rjust(4, ' ')
            roci_str += formatted_roci

        # vmax 10m is not strictly in ROCI (it's in a box formed from ROCI grid units rather than meters)
        vmax10m_in_roci_str = ""
        if 'vmax10m_in_roci' in candidate:
            vmax10m_in_roci_str = "10m WS MAX (knots) in ~ROCI: "
            vmax10m_in_roci_value = candidate['vmax10m_in_roci']
            formatted_vmax10m = f"{vmax10m_in_roci_value * meters_to_knots:3.1f}".rjust(6, ' ')
            vmax10m_in_roci_str += formatted_vmax10m

        # MSLP at outermost closed isobar - MSLP minimum
        closed_isobar_delta_str = ""
        if 'closed_isobar_delta' in candidate:
            closed_isobar_delta_str = ", Isobar delta (hPa) (MSLP for OCI - minimum): "
            closed_isobar_delta_value = candidate['closed_isobar_delta']
            formatted_closed_isobar_delta = f"{closed_isobar_delta_value:3.0f}".rjust(3, ' ')
            closed_isobar_delta_str += formatted_closed_isobar_delta

        if no_numbering:
            numbering_str = "    "
        else:
            numbering_str = f"#{n: >2}, "
        print_with_process_id(f"{numbering_str}{basin_str}Latitude (deg:): {lat: >6.1f}, Longitude (deg): {lon: >6.1f}, MSLP (hPa): {formatted_mslp}{roci_str}\n        {rv_str}{thickness_str}{vmax_str}\n        {vmax10m_in_roci_str}{closed_isobar_delta_str}")

        if debug_isobar_images:
            create_image(candidate['neighborhood'], 'Neighborhood')
            create_image(candidate['visited'], 'Visited')

# this function is for checking find_mslp_minima_with_closed_isobars
def find_mslp_minima(mslp_data, minima_neighborhood_size=1):
    try:
        # Lists to store MSLP minima, latitudes, and longitudes
        candidates = []

        # Loop through each grid point
        for x in range(minima_neighborhood_size, mslp_data.shape[0] - minima_neighborhood_size):
            for y in range(minima_neighborhood_size, mslp_data.shape[1] - minima_neighborhood_size):
                mslp_value = mslp_data[x, y]  # MSLP value at the current point
                neighborhood = mslp_data[x - minima_neighborhood_size:x + minima_neighborhood_size + 1,
                                         y - minima_neighborhood_size:y + minima_neighborhood_size + 1]

                # Check if the MSLP value is the minimum within the neighborhood
                if mslp_value == neighborhood.min():
                    candidate = {
                        "mslp_value": float(mslp_value),
                        "x_value": x,
                        "y_value": y
                    }
                    candidates.append(candidate)


        return candidates
    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return [], [], []

def print_grid(grid):
    for row in grid:
        print_with_process_id("".join(map(lambda x: "1" if x else "0", row)))

# use traverse_contour() for speed (the non fast version uses dfs)
def find_mslp_minima_with_closed_isobars_fast(mslp_data, grid_resolution, isobar_threshold=2.0, isobar_search_radius_degrees = 5, minima_neighborhood_size = 1):
    isobar_neighborhood_size = calculate_neighborhood_size(isobar_search_radius_degrees, grid_resolution)
    mslp_shape_x = mslp_data.shape[0]
    mslp_shape_y = mslp_data.shape[1]
    try:
        # List to store MSLP minima as dictionaries
        mslp_minima_list = []

        # Create a list of candidates for MSLP minima
        candidates = []

        # Loop through each grid point
        for x in range(mslp_shape_x):
            for y in range(mslp_shape_y):
                mslp_value = mslp_data[x, y]  # MSLP value at the current point

                x_min = x - minima_neighborhood_size
                x_max = x + minima_neighborhood_size + 1
                y_min = y - minima_neighborhood_size
                y_max = y + minima_neighborhood_size + 1

                in_bounds = ((x_min >= 0) and
                    (x_max <= mslp_shape_x) and
                    (y_min >= 0) and
                    (y_max <= mslp_shape_y))

                if in_bounds:
                    # the normal case (not edges of array)
                    neighborhood = mslp_data[x_min:x_max, y_min:y_max]
                else:
                    # handle indices at the boundaries
                    neighborhood = extract_2d_neighborhood(mslp_data, (x, y), minima_neighborhood_size)

                # Check if the MSLP value is the minimum within the neighborhood
                if mslp_value == neighborhood.min():
                    candidates.append((x, y, float(mslp_value)))

        # Loop through the candidates to find isobars
        for x, y, minima_value in candidates:
            # Create a modified neighborhood for isobar calculation

            x_min = x - isobar_neighborhood_size
            x_max = x + isobar_neighborhood_size + 1
            y_min = y - isobar_neighborhood_size
            y_max = y + isobar_neighborhood_size + 1

            in_bounds = ((x_min >= 0) and
                (x_max <= mslp_shape_x) and
                (y_min >= 0) and
                (y_max <= mslp_shape_y))

            if in_bounds:
                # the normal case (not edges of array)
                neighborhood = mslp_data[x_min:x_max, y_min:y_max]
            else:
                # handle indices at the boundaries
                neighborhood = extract_2d_neighborhood(mslp_data, (x, y), isobar_neighborhood_size)

            # create a binary image (True are values that are at the threshold, with the center being False)
            binary_neighborhood = (neighborhood - minima_value - isobar_threshold) >= 0
            path_found = traverse_contour(binary_neighborhood)

            if path_found:
                # Store MSLP minima data as a dictionary
                candidate = {
                    "mslp_value": float(minima_value),
                    "x_value": x,
                    "y_value": y
                }
                mslp_minima_list.append(candidate)

        return mslp_minima_list

    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return []

# minima_neighborhood_size set to 1 finds the smallest area that could be called a relative minima in MSLP
# this can create many duplicates for the same cyclone
# [1] uses a 3 degree buffer over space and time to group together these disturbances as a cyclone:
# Quoting [1]; "a 3° buffer around the position at forecast hour 12. Note that the position at forecast hour 18 is on or within the 3° buffer. Thus, these two points are considered to be the same TC. A new buffer is then drawn around the location at forecast hour 18 (green box), and a search is done for any positions on or within the buffer at forecast hour 24. This process is repeated until no points are found on or within the buffer."
def find_mslp_minima_with_closed_isobars(mslp_data, grid_resolution, isobar_threshold=2.0, isobar_search_radius_degrees = 5, minima_neighborhood_size = 1):
    isobar_neighborhood_size = calculate_neighborhood_size(isobar_search_radius_degrees, grid_resolution)
    #print_with_process_id('isobar neighborsize', isobar_neighborhood_size)
    def dfs(x, y):
        nonlocal visited
        stack = [(x, y)]

        while stack:
            x, y = stack.pop()
            visited[x][y] = True

            # Check N, S, E, W neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for nx, ny in neighbors:
                if 0 <= nx < isobar_neighborhood_size * 2 + 1 and 0 <= ny < isobar_neighborhood_size * 2 + 1:
                    if not visited[nx][ny]:
                        if nx == 0 or ny == 0 or nx == isobar_neighborhood_size * 2 or ny == isobar_neighborhood_size * 2:
                            return False
                        if not binary_neighborhood[nx][ny]:
                            stack.append((nx, ny))
        return True

    try:
        # List to store MSLP minima as dictionaries
        mslp_minima_list = []

        # Create a list of candidates for MSLP minima
        candidates = []

        # Loop through each grid point
        for x in range(mslp_data.shape[0]):
            for y in range(mslp_data.shape[1]):
                mslp_value = mslp_data[x, y]  # MSLP value at the current point

                x_min = x - minima_neighborhood_size
                x_max = x + minima_neighborhood_size + 1
                y_min = y - minima_neighborhood_size
                y_max = y + minima_neighborhood_size + 1

                in_bounds = ((x_min >= 0) and
                    (x_max <= mslp_data.shape[0]) and
                    (y_min >= 0) and
                    (y_max <= mslp_data.shape[1]))

                if in_bounds:
                    # the normal case (not edges of array)
                    neighborhood = mslp_data[x_min:x_max, y_min:y_max]
                else:
                    # handle indices at the boundaries
                    neighborhood = extract_2d_neighborhood(mslp_data, (x, y), minima_neighborhood_size)

                # Check if the MSLP value is the minimum within the neighborhood
                if mslp_value == neighborhood.min():
                    candidates.append((x, y, float(mslp_value)))

        # Loop through the candidates to find isobars
        for x, y, minima_value in candidates:
            # Create a modified neighborhood for isobar calculation

            x_min = x - isobar_neighborhood_size
            x_max = x + isobar_neighborhood_size + 1
            y_min = y - isobar_neighborhood_size
            y_max = y + isobar_neighborhood_size + 1

            in_bounds = ((x_min >= 0) and
                (x_max <= mslp_data.shape[0]) and
                (y_min >= 0) and
                (y_max <= mslp_data.shape[1]))

            if in_bounds:
                # the normal case (not edges of array)
                neighborhood = mslp_data[x_min:x_max, y_min:y_max]
            else:
                # handle indices at the boundaries
                neighborhood = extract_2d_neighborhood(mslp_data, (x, y), isobar_neighborhood_size)

            binary_neighborhood = (neighborhood - minima_value - isobar_threshold) >= 0

            # Initialize visited array for DFS
            visited = np.zeros_like(binary_neighborhood, dtype=bool)

            # Flag to track if a closed path is found
            path_found = dfs(isobar_neighborhood_size, isobar_neighborhood_size)

            if path_found:
                # Store MSLP minima data as a dictionary
                if debug_isobar_images:
                    candidate = {
                        "mslp_value": float(minima_value),
                        "x_value": x,
                        "y_value": y,
                        "neighborhood": copy.deepcopy(neighborhood - minima_value - isobar_threshold),
                        "visited": copy.deepcopy(visited)
                    }
                else:
                    candidate = {
                        "mslp_value": float(minima_value),
                        "x_value": x,
                        "y_value": y
                    }
                mslp_minima_list.append(candidate)


        return mslp_minima_list

    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return []


# calculate vorticity (accounting for projection distortion) using metpy (uses finite differences method)
# returns vorticity using an ellipsoid based on WGS84
def calculate_vorticity(u_wind_850, v_wind_850, lats=None, lons=None, vorticity_method=default_vorticity_method):
    if vorticity_method == 'metview':
        u_wind_850_fs = u_wind_850
        v_wind_850_fs = v_wind_850
        xds_u = u_wind_850_fs.to_dataset()
        xds_v = v_wind_850_fs.to_dataset()

        with SuppressRuntimeWarningVortCalcs():
            vort_fieldset = mv.vorticity(xds_u, xds_v)

        vort_dataset = vort_fieldset.to_dataset()
        masked_vort_850 = vort_dataset['vo'].to_masked_array()
        np.ma.set_fill_value(masked_vort_850, -9999.0)
        # return as a masked array (this will hide NaNs: needed for getting the correct max and saving the calculation)
        return masked_vort_850
    elif vorticity_method == 'metpy':
        # similar to gempak
        # supposed to require cylindrical corrections (but this is what the thresholded values)
        # Convert wind components to units
        u_wind_850_with_units = units.Quantity(u_wind_850, 'm/s')
        v_wind_850_with_units = units.Quantity(v_wind_850, 'm/s')

        # it seems FSU is using GFS 0.5deg, and using vorticity calculations with a spherical geodesic
        dx, dy = mpcalc.lat_lon_grid_deltas(lons * units.degrees, lats * units.degrees)

        # Calculate relative vorticity (use metpy)
        with SuppressRuntimeWarningVortCalcs():
            vort_850 = np.array(mpcalc.vorticity(u_wind_850_with_units, v_wind_850_with_units, dx=dx, dy=dy))
            masked_vort_850 = np.ma.masked_invalid(vort_850)

        np.ma.set_fill_value(masked_vort_850, -9999.0)
        # return as a masked array (this will hide NaNs: needed for getting the correct max and saving the calculation)
        return masked_vort_850

# Function to find RV maximums in neighborhoods for a list of candidates
def find_rv_maximums_in_neighborhoods(mslp_minima_list, rv, grid_resolution, relative_vorticity_radius_degrees = 2):
    neighborhood_size = calculate_neighborhood_size(relative_vorticity_radius_degrees, grid_resolution)
    updated_mslp_minima_list = []

    for candidate in mslp_minima_list:
        x, y, _ = candidate["x_value"], candidate["y_value"], candidate["mslp_value"]

        # Extract the neighborhood for the current candidate
        x_min = x - neighborhood_size
        x_max = x + neighborhood_size + 1
        y_min = y - neighborhood_size
        y_max = y + neighborhood_size + 1

        in_bounds = ((x_min >= 0) and
            (x_max <= rv.shape[0]) and
            (y_min >= 0) and
            (y_max <= rv.shape[1]))

        if in_bounds:
            # the normal case (not edges of array)
            neighborhood = rv[x_min:x_max, y_min:y_max]
        else:
            # handle indices at the boundaries
            neighborhood = extract_2d_neighborhood(rv, (x, y), neighborhood_size)

        # Find the maximum RV value within the neighborhood
        # need to adjust by latitude (vorticity will be negative for southern hemisphere)
        lat = candidate["lat"]
        if lat < 0:
            rv_max = neighborhood.min()
        else:
            rv_max = neighborhood.max()

        # Update a copy of the candidate's dictionary with rv maximum value
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['rv850max'] = float(rv_max)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# calculate relative max thickness for 250hPa - 850hPa for a list of candidates
def find_gp_250_850_max_thickness(mslp_minima_list, geopotential_250, geopotential_850, grid_resolution, degrees_radius=2):
    # Calculate the neighborhood size based on degrees_radius and grid_resolution
    neighborhood_size = calculate_neighborhood_size(degrees_radius, grid_resolution)

    updated_mslp_minima_list = []

    # Iterate over the list of candidates
    for candidate in mslp_minima_list:
        # Extract the x and y indices of the candidate
        x, y = candidate['x_value'], candidate['y_value']

        # Extract the neighborhoods for 250 hPa and 850 hPa
        x_min = x - neighborhood_size
        x_max = x + neighborhood_size + 1
        y_min = y - neighborhood_size
        y_max = y + neighborhood_size + 1

        in_bounds = ((x_min >= 0) and
            (x_max <= geopotential_250.shape[0]) and
            (y_min >= 0) and
            (y_max <= geopotential_250.shape[1]))

        if in_bounds:
            # the normal case (not edges of array)
            neighborhood_250 = geopotential_250[x_min:x_max, y_min:y_max]
            neighborhood_850 = geopotential_850[x_min:x_max, y_min:y_max]
        else:
            # handle indices at the boundaries
            neighborhood_250 = extract_2d_neighborhood(geopotential_250, (x, y), neighborhood_size)
            neighborhood_850 = extract_2d_neighborhood(geopotential_850, (x, y), neighborhood_size)

        # Calculate the 250–850-hPa thickness for each cell in the neighborhood
        thickness = neighborhood_250 - neighborhood_850

        # Find the maximum thickness value in the neighborhood
        max_thickness = thickness.max()

        # Update a copy of the candidate's dictionary with the maximum thickness value
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['gp250_850_thickness'] = float(max_thickness)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# find max wind at 925 hPa for a list of candidates
def find_max_wind_925(mslp_minima_list, u_wind_925, v_wind_925, grid_resolution, degrees_radius=5):
    # Calculate the neighborhood size based on radius_degrees and grid_resolution for wind
    neighborhood_size = calculate_neighborhood_size(degrees_radius, grid_resolution)

    updated_mslp_minima_list = []

    # Iterate over the list of candidates
    for candidate in mslp_minima_list:
        # Extract the neighborhoods for u-wind and v-wind at 925 hPa

        # Extract the x and y indices of the candidate
        x = candidate['x_value']
        y = candidate['y_value']

        x_min = x - neighborhood_size
        x_max = x + neighborhood_size + 1
        y_min = y - neighborhood_size
        y_max = y + neighborhood_size + 1

        in_bounds = ((x_min >= 0) and
            (x_max <= u_wind_925.shape[0]) and
            (y_min >= 0) and
            (y_max <= u_wind_925.shape[1]))

        if in_bounds:
            # the normal case (not edges of array)
            neighborhood_u_wind = u_wind_925[x_min:x_max, y_min:y_max]
            neighborhood_v_wind = v_wind_925[x_min:x_max, y_min:y_max]
        else:
            # handle indices at the boundaries
            neighborhood_u_wind = extract_2d_neighborhood(u_wind_925, (x, y), neighborhood_size)
            neighborhood_v_wind = extract_2d_neighborhood(v_wind_925, (x, y), neighborhood_size)

        # Calculate wind speed in the neighborhood
        wind_speed = np.sqrt(neighborhood_u_wind ** 2 + neighborhood_v_wind ** 2)

        # Find the maximum wind speed value in the neighborhood
        max_wind_speed = wind_speed.max()

        # Update a copy of the candidate's dictionary with the maximum wind speed value
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['vmax925'] = float(max_wind_speed)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# find max wind at 10m for a list of candidates using ROCI calculations
# vmax 10m is not strictly in ROCI (it's in a box formed from ROCI grid units rather than meters)
def find_max_wind_10m_in_roci(mslp_minima_list, u_wind_10m, v_wind_10m):
    if u_wind_10m is None or v_wind_10m is None:
        return mslp_minima_list

    updated_mslp_minima_list = []

    # Iterate over the list of candidates
    for candidate in mslp_minima_list:
        if 'roci_grid_units' not in candidate:
            updated_mslp_minima_list.append(candidate)
            continue

        radius_grid_units_float = candidate['roci_grid_units']
        radius_units_int = int(np.round(radius_grid_units_float))

        # The neighborhood size is the ROCI (grid units)
        neighborhood_size = radius_units_int

        # Extract the neighborhoods for u-wind and v-wind at 925 hPa

        # Extract the x and y indices of the candidate
        x = candidate['x_value']
        y = candidate['y_value']

        x_min = x - neighborhood_size
        x_max = x + neighborhood_size + 1
        y_min = y - neighborhood_size
        y_max = y + neighborhood_size + 1

        in_bounds = ((x_min >= 0) and
            (x_max <= u_wind_10m.shape[0]) and
            (y_min >= 0) and
            (y_max <= u_wind_10m.shape[1]))

        if in_bounds:
            # the normal case (not edges of array)
            neighborhood_u_wind = u_wind_10m[x_min:x_max, y_min:y_max]
            neighborhood_v_wind = v_wind_10m[x_min:x_max, y_min:y_max]
        else:
            # handle indices at the boundaries
            neighborhood_u_wind = extract_2d_neighborhood(u_wind_10m, (x, y), neighborhood_size)
            neighborhood_v_wind = extract_2d_neighborhood(v_wind_10m, (x, y), neighborhood_size)

        # Calculate wind speed in the neighborhood
        wind_speed = np.sqrt(neighborhood_u_wind ** 2 + neighborhood_v_wind ** 2)

        # Find the maximum wind speed value in the neighborhood
        max_wind_speed = wind_speed.max()

        # Update a copy of the candidate's dictionary with the maximum wind speed value
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['vmax10m_in_roci'] = float(max_wind_speed)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

def calc_roci(mslp_minima_list, mslp, mslp_fs, grid_resolution):
    updated_mslp_minima_list = []

    # Iterate over the list of candidates
    for candidate in mslp_minima_list:
        # Calculate: the outermost closed isobar (dfs only)
        #   the 'mass' location of the innermost closed isobar
        #   and, the isobar difference between the MSLP minimum and the outermost closed isobar
        # Note: a fast version of this using traverse_contour doesn't have a speed up over dfs as it will be closed for most of the isobars checked
        mass_candidate, max_closed_isobar_difference, visited_outermost_closed_isobar = get_outermost_closed_isobar(mslp, grid_resolution, candidate)

        average_radius, average_radius_units = find_average_radius(mslp_fs, mass_candidate, visited_outermost_closed_isobar)

        # Update a copy of the candidate's dictionary with the ROCI in m and in units (model grid units)
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['roci'] = float(average_radius)
        updated_candidate['roci_grid_units'] = float(average_radius_units)
        updated_candidate['mass_x'] = mass_candidate['mass_x']
        updated_candidate['mass_y'] = mass_candidate['mass_y']
        # MSLP at OCI - MSLP minimum (this is threshold (integer) difference for the outermost closed isobar from the MSLP minimum)
        updated_candidate['closed_isobar_delta'] = int(max_closed_isobar_difference)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# add basin name, and also removes candidates that aren't in one of the basins
# add lat,lon also
def add_basin_name(mslp_minima_list, lats, lons):
    updated_mslp_minima_list = []

    for candidate in mslp_minima_list:
        x = candidate["x_value"]
        y = candidate["y_value"]

        lat, lon = array_indices_to_lat_lon(x, y, lats, lons)
        basin_name = get_basin_name_from_lat_lon(lat, lon)

        # remove candidate if not in a basin we are considering
        if not basin_name:
            continue

        # Update a copy of the candidate's dictionary with rv maximum value
        updated_candidate = copy.deepcopy(candidate)
        updated_candidate['basin'] = basin_name
        updated_candidate['lat'] = lat
        updated_candidate['lon'] = lon

        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# Function to calculate the booleans on disturbance thresholds are met or not for each criteria and for all
# Must calculate and update with the basin classification first
def calc_disturbance_threshold_booleans(mslp_minima_list, model_name):
    updated_mslp_minima_list = []

    for candidate in mslp_minima_list:
        # Update a copy of the candidate's dictionary with rv maximum value
        updated_candidate = copy.deepcopy(candidate)
        all_met = True

        basin_name = candidate["basin"]
        lat = candidate["lat"]

        for criteria_abbrev, criteria_value in disturbance_thresholds[model_name][basin_name].items():
            criteria_name = disturbance_criteria_names_map[criteria_abbrev]
            flip_sign = 1
            if criteria_name == 'rv850max':
                # thresholds in the JSON for RV are scaled by 10^5
                criteria_value *= pow(10, -5)
                if lat < 0:
                    flip_sign = -1
            criteria_bool = False
            # for rv max flip sign for southern hemisphere (for other criteria, no)
            if updated_candidate[criteria_name] * flip_sign >= criteria_value:
                criteria_bool = True
            if 'criteria' not in updated_candidate:
                updated_candidate['criteria'] = {}
            updated_candidate['criteria'][criteria_name] = criteria_bool
            all_met = (all_met and criteria_bool)

        updated_candidate['criteria']['all'] = all_met
        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    return updated_mslp_minima_list

# returns the basin name from lat, lon if in a basin that is covered by the thresholds, otherwise returns None
def get_basin_name_from_lat_lon(lat, lon):
    # Create a Point geometry for the latitude and longitude
    point = Point(lon, lat)

    # Check if the point is within any of the polygons
    result = gdf[gdf.geometry.covers(point)]
    if not result.empty:
        shape_basin_name = result['basin_name'].iloc[0]
        # this is used for thresholds so return the threshold name
        return shape_basin_names_to_threshold_names_map[shape_basin_name]
    else:
        return None

# use u_wind_850 grib as reference and modify message to store vorticity
def save_vorticity_to_grib(u_wind_850_grib_file_path = None, grib_dest_file_path = None, relative_vorticity = None):
    if u_wind_850_grib_file_path is None or grib_dest_file_path is None or relative_vorticity is None:
        return

    grbs = pygrib.open(u_wind_850_grib_file_path)
    modified_grbs = []
    # Define the updated attributes for relative vorticity
    new_param_id = 138
    new_short_name = 'vo'

    for grb in grbs:
        # Modify parameter information
        grb.shortName = new_short_name
        grb.paramId = new_param_id
        # Replace values
        # doesn't seem to handle masked -nan values properly (encoding error), so fill with -9999
        grb.values = relative_vorticity.filled()
        modified_grbs.append(grb)

    # Create a new GRIB file with modified data
    with open(grib_dest_file_path, 'wb') as output_file:
        for modified_grb in modified_grbs:
            output_file.write(modified_grb.tostring())

# filter out candidates not meeting criteria
def get_disturbance_candidates_from_split_gribs(grib_files, model_name):
    # grid_resolution in degrees. MSLP is converted hPa, rest are converted to metric. Relative vorticity (1/s) is not scaled as threshold (not * 10^5)
    debug_timing()
    grid_resolution, lats, lons, mslp, mslp_fs, u_wind_925, v_wind_925, u_wind_850, u_wind_850_fs, v_wind_850, v_wind_850_fs, geopotential_250, geopotential_850, relative_vorticity_850, u_wind_10m, v_wind_10m, u_wind_850_grib_file_path = load_and_extract_split_grib_data(grib_files, model_name)
    debug_timing('load_and_extract_split_grib_data()')

    # check if an error was caught
    if lats is None:
        return None

    debug_timing()
    #mslp_minima_list_with_closed_isobars = find_mslp_minima_with_closed_isobars(mslp, grid_resolution['mslp'])
    # use a faster version (50 seconds vs 100+s for CMC) that relies on that most of the candidates will not satisfy it
    mslp_minima_list_with_closed_isobars = find_mslp_minima_with_closed_isobars_fast(mslp, grid_resolution['mslp'])
    debug_timing('find_mslp_minima_with_closed_isobars()')

    # if vorticity is missing, calculate relative vorticity for the entire data set first
    if relative_vorticity_850 is None:
        # this may issue warnings for divide by 0 for vorticity calculations
        #print_with_process_id("Calculating Relative vorticity for 850 hPa")
        debug_timing()
        if default_vorticity_method == 'metpy':
            rv_850 = calculate_vorticity(u_wind_850, v_wind_850, lats=lats, lons=lons)
        elif default_vorticity_method == 'metview':
            rv_850 = calculate_vorticity(u_wind_850_fs, v_wind_850_fs)
        debug_timing('calculate_vorticity()')
        # save the vorticity calculated to grib?
        if debug_save_vorticity:
            grib_dir = os.path.dirname(u_wind_850_grib_file_path)
            grib_dest_file_path = os.path.join(grib_dir, 'calculated_relative_vorticity.grib')
            save_vorticity_to_grib(u_wind_850_grib_file_path = u_wind_850_grib_file_path, grib_dest_file_path = grib_dest_file_path, relative_vorticity = rv_850)
    else:
        rv_850 = relative_vorticity_850

    debug_timing()
    # get lat/lon and basin name (exclude ones not in a basin we cover)
    mslp_minima_list_in_basins = add_basin_name(mslp_minima_list_with_closed_isobars, lats, lons)
    debug_timing('add_basin_name()')

    # calculate relative vorticity maximum for each candidate
    debug_timing()
    mslp_minima_list_in_basins = find_rv_maximums_in_neighborhoods(mslp_minima_list_in_basins, rv_850, grid_resolution['gh250'])
    debug_timing('find_rv_maximums_in_neighborhoods()')

    debug_timing()
    # Find the maximum relative thickness (250 - 850 hPa) for each candidate
    mslp_minima_list_in_basins = find_gp_250_850_max_thickness(mslp_minima_list_in_basins, geopotential_250, geopotential_850, grid_resolution['gh250'])
    debug_timing('find_gp_250_850_max_thickness()')

    debug_timing()
    # find the maximum 925 hPa wind speed
    mslp_minima_list_in_basins = find_max_wind_925(mslp_minima_list_in_basins, u_wind_925, v_wind_925, grid_resolution['uwind925'])
    debug_timing('find_max_wind_925()')

    debug_timing()
    mslp_minima_list_with_disturbance_threshold_booleans = calc_disturbance_threshold_booleans(mslp_minima_list_in_basins, model_name)
    debug_timing(f'calc_disturbance_threshold_booleans()')
    # print_candidates(mslp_minima_list_with_disturbance_threshold_booleans, lats, lons, meet_all_disturbance_thresholds = True)

    debug_timing()
    # exclude disturbances not meeting all criteria
    updated_mslp_minima_list = []
    # Iterate over the list of candidates
    for candidate in mslp_minima_list_with_disturbance_threshold_booleans:
        # exclude candidates not meeting all criteria
        if not candidate['criteria']['all']:
            #print_with_process_id("Failed candidate:")
            #print_with_process_id(json.dumps(candidate,indent=4))
            continue

        # Update a copy of the candidate's dictionary with LAT, LON
        updated_candidate = copy.deepcopy(candidate)

        # Add the updated candidate to the list
        updated_mslp_minima_list.append(updated_candidate)

    debug_timing('Remove candidates not meeting all criteria')

    debug_timing()
    updated_mslp_minima_list_with_roci = calc_roci(updated_mslp_minima_list, mslp, mslp_fs, grid_resolution['mslp'])
    debug_timing('calc_roci()')

    debug_timing()
    updated_mslp_minima_list_with_10m_wind = find_max_wind_10m_in_roci(updated_mslp_minima_list_with_roci, u_wind_10m, v_wind_10m)
    debug_timing('find_max_wind_10m_in_roci()')

    #print_with_process_id('\n')
    return updated_mslp_minima_list_with_10m_wind

def get_level_from_fieldset(fs):
    level_list = mv.grib_get(fs,['level'])
    # may be nested like [['850']]
    level = int(np.array(level_list).flatten()[0])
    return level

def load_and_extract_split_grib_data(grib_files, model_name):
    try:
        # Initialize variables for relevant parameters
        # we want the field set for mslp for distance calculations, and u/v wind 850 for vorticity calculations
        # otherwise the variables are numpy 2-D arrays (y,x), but we just use internally [x][y] for calculations for simplicity
        # so candidate x and y values really refers to lats and lons which normally are thought of as y and x
        mslp = None
        mslp_fs = None
        u_wind_925 = None
        v_wind_925 = None
        u_wind_850 = None
        v_wind_850 = None
        u_wind_850_fs = None
        v_wind_850_fs = None
        u_wind_10m = None
        v_wind_10m = None

        geopotential_250 = None
        geopotential_850 = None
        relative_vorticity_850 = None

        mslp_units = None
        wind_925_units = None
        wind_850_units = None
        wind_10m_units = None

        mslp_lats = None
        mslp_lons = None
        u_wind_925_lats = None
        u_wind_925_lons = None
        v_wind_925_lats = None
        v_wind_925_lons = None
        u_wind_850_lats = None
        u_wind_850_lons = None
        v_wind_850_lats = None
        v_wind_850_lons = None
        geopotential_250_lats = None
        geopotential_250_lons = None
        geopotential_850_lats = None
        geopotential_850_lons = None
        relative_vorticity_850_lats = None
        relative_vorticity_850_lons = None
        u_wind_10m_lats = None
        u_wind_10m_lons = None
        v_wind_10m_lats = None
        v_wind_10m_lons = None

        # only used for reference to optionally store calculated relative vorticity
        u_wind_850_grib_file_path = None

        grid_resolution = {}

        for grib_file in grib_files:
            # ukmets 10m adjusted does not have proper names in grib (or look up fails)
            u_component = False
            v_component = False
            units = None

            # Open the GRIB file
            #grbs = pygrib.open(grib_file)
            fs = mv.read(grib_file)
            # Extract relevant parameters (modify the parameter names and levels accordingly)

            short_name = None
            # assume only one message per grib (gribs split by param and level)
            xds = fs.to_dataset()
            for name in xds:
                short_name = name
            xda = xds[short_name]

            level = get_level_from_fieldset(fs)

            res = float(xda.GRIB_iDirectionIncrementInDegrees)
            name = xda.long_name
            units = xda.GRIB_units
            if "u-component" in grib_file:
                u_component = True
                units = "m/s"
            if "v-component" in grib_file:
                v_component = True
                units = "m/s"
            level_type = xda.GRIB_typeOfLevel
            values = xda.to_numpy()

            # DO NOT USE fieldset for latitudes as it doesn't work: must operate on xarray data array
            lats = xda.latitude.to_numpy()
            lons = xda.longitude.to_numpy()
            if level == 0 and name in ['Pressure reduced to MSL', 'Mean sea level pressure']:
                mslp = values
                mslp_fs = fs
                mslp_units = units
                mslp_lats = lats
                mslp_lons = lons
                grid_resolution['mslp'] = res
            elif name == 'U component of wind' and level == 925:
                u_wind_925 = values
                wind_925_units = units
                u_wind_925_lats = lats
                u_wind_925_lons = lons
                grid_resolution['uwind925'] = res
            elif name == 'V component of wind' and level == 925:
                v_wind_925 = values
                v_wind_925_lats = lats
                v_wind_925_lons = lons
                grid_resolution['vwind925'] = res
            elif name == 'U component of wind' and level == 850:
                u_wind_850 = values
                u_wind_850_fs = fs
                wind_850_units = units
                u_wind_850_lat = lats
                u_wind_850_lons = lons
                u_wind_850_grib_file_path = grib_file
                grid_resolution['uwind850'] = res
            elif name == 'V component of wind' and level == 850:
                v_wind_850 = values
                v_wind_850_fs = fs
                v_wind_850_lats = lats
                v_wind_850_lons = lons
                grid_resolution['vwind850'] = res
            elif name == 'Geopotential height' and level == 250:
                geopotential_250 = values
                geopotential_250_lats = lats
                geopotential_250_lons = lons
                grid_resolution['gh250'] = res
            elif name == 'Geopotential height' and level == 850:
                geopotential_850 = values
                geopotential_850_lats = lats
                geopotential_850_lons = lons
                grid_resolution['gh850'] = res
            elif name == 'Vorticity (relative)' and level == 850:
                relative_vorticity_850 = values
                relative_vorticity_850_lats = lats
                relative_vorticity_850_lons = lons
                grid_resolution['rv850'] = res
            elif name == '10 metre U wind component' or (level_type == 'heightAboveGround' and level == 10 and u_component):
                u_wind_10m = values
                wind_10m_units = units
                u_wind_10m_lats = lats
                u_wind_10m_lons = lons
                grid_resolution['uwind10m'] = res
            elif name == '10 metre V wind component' or (level_type == 'heightAboveGround' and level == 10 and v_component):
                v_wind_10m = values
                v_wind_10m_lats = lats
                v_wind_10m_lons = lons
                grid_resolution['vwind10m'] = res

        # Check if units are not in hPa and convert if necessary
        if mslp_units != "hPa":
            #print_with_process_id(f"Converting MSLP units from {mslp_units} to hPa")
            if mslp_units == "Pa":
                # Convert from Pa to hPa
                mslp *= 0.01
                mslp_units = "hPa"
            else:
                print_with_process_id("Warning: Units of MSLP are not in Pa or hPa. Please verify the units for accurate results.")

        # Check if units are not in m/s for 925 hPa wind components and convert if necessary
        if not (re.search(r"(m/s|m s\*\*-1)", wind_925_units)):
            print_with_process_id(f"Converting 925 hPa wind components units from {wind_925_units} to m/s")
            if re.search(r"knots|knot", wind_925_units, re.I):
                # Convert from knots to m/s (1 knot ≈ 0.514444 m/s)
                u_wind_925 *= 0.514444
                v_wind_925 *= 0.514444
                wind_925_units = "m/s"
            else:
                print_with_process_id("Warning: Units of 925 hPa wind components are not in knots, m/s, or m s**-1. Please verify the units for accurate results.")

        # Check if units are not in m/s for 925 hPa wind components and convert if necessary
        if wind_850_units and not (re.search(r"(m/s|m s\*\*-1)", wind_850_units)):
            print_with_process_id(f"Converting 850 hPa wind components units from {wind_850_units} to m/s")
            if re.search(r"knots|knot", wind_925_units, re.I):
                # Convert from knots to m/s (1 knot ≈ 0.514444 m/s)
                u_wind_850 *= 0.514444
                v_wind_850 *= 0.514444
                wind_850_units = "m/s"
            else:
                print_with_process_id("Warning: Units of 925 hPa wind components are not in knots, m/s, or m s**-1. Please verify the units for accurate results.")

        # Check if units are not in m/s for 925 hPa wind components and convert if necessary
        if wind_10m_units and not (re.search(r"(m/s|m s\*\*-1)", wind_10m_units)):
            print_with_process_id(f"Converting 10 m wind components units from {wind_10m_units} to m/s")
            if re.search(r"knots|knot", wind_10m_units, re.I):
                # Convert from knots to m/s (1 knot ≈ 0.514444 m/s)
                u_wind_10m *= 0.514444
                v_wind_10m *= 0.514444
                wind_10m_units = "m/s"
            else:
                print_with_process_id("Warning: Units of 925 hPa wind components are not in knots, m/s, or m s**-1. Please verify the units for accurate results.")

        # make sure shapes are all the same
        lat_shapes = [x.shape for x in
            [
                mslp_lats,
                u_wind_925_lats,
                v_wind_925_lats,
                u_wind_850_lats,
                v_wind_850_lats,
                geopotential_250_lats,
                geopotential_850_lats,
                relative_vorticity_850_lats,
                u_wind_10m_lats,
                v_wind_10m_lats
            ] if x is not None]
        lon_shapes = [x.shape for x in
            [
                mslp_lons,
                u_wind_925_lons,
                v_wind_925_lons,
                u_wind_850_lons,
                v_wind_850_lons,
                geopotential_250_lons,
                geopotential_850_lons,
                relative_vorticity_850_lons,
                u_wind_10m_lons,
                v_wind_10m_lons
            ] if x is not None]

        # check to make sure all the same shape
        if len(set(lat_shapes)) != 1:
            # lats and lons different shapes!
            print_with_process_id(f"Error: getting disturbance candidates: lats have different shapes for: {grib_files}")
            return [None] * 17

        # check to make sure all the same shape
        if len(set(lon_shapes)) != 1:
            # lats and lons different shapes!
            print_with_process_id(f"Error: getting disturbance candidates: lons have different shapes for: {grib_files}")
            return [None] * 17

        lats = mslp_lats
        lons = mslp_lons

        return grid_resolution, lats, lons, mslp, mslp_fs, u_wind_925, v_wind_925, u_wind_850, u_wind_850_fs, v_wind_850, v_wind_850_fs, geopotential_250, geopotential_850, relative_vorticity_850, u_wind_10m, v_wind_10m, u_wind_850_grib_file_path
    except Exception as e:
        print_with_process_id(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)
        return [None] * 17

def extract_1d_neighborhood(arr, center_index, neighborhood_size):
    num_elements = arr.shape[0]
    neighborhood = np.zeros(2 * neighborhood_size + 1, dtype=arr.dtype)

    for i in range(-neighborhood_size, neighborhood_size + 1):
        index = (center_index + i) % num_elements
        neighborhood[i + neighborhood_size] = arr[index]

    return neighborhood

def extract_2d_neighborhood(arr, center_indices, neighborhood_size):
    """
    Extract a 2D neighborhood around a center index in a 2D array.

    Parameters:
    - arr: The 2D input array.
    - center_indices: A tuple (center_x, center_y) specifying the center indices.
    - neighborhood_size: The size of the neighborhood (half on each side of the center).

    Returns:
    - neighborhood: The extracted 2D neighborhood.
    """
    center_x, center_y = center_indices
    x_indices = range(center_x - neighborhood_size, center_x + neighborhood_size + 1)

    # Initialize an empty 2D neighborhood
    neighborhood = np.empty((2 * neighborhood_size + 1, 2 * neighborhood_size + 1), dtype=arr.dtype)

    num_elements = arr.shape[0]
    # Extract the 1D neighborhoods for each row
    for i, x in enumerate(x_indices):
        index = x % num_elements
        neighborhood[i] = extract_1d_neighborhood(arr[index], center_y, neighborhood_size)

    return neighborhood

## AUTO UPDATE CODE ##

model_names_to_update = ['GFS', 'ECM', 'CMC', 'NAV']
# only try to calculate from (approx.) previous # of hours (so we don't loop through entire models folder)
hours_to_look_back = 48
# polling interval in minutes to recalculate (look for new gribs and update the disturbances)
polling_interval = 5

# Parallelized on model time steps
# Note: If there is a failture relating to metview and EOF, delete any metview created idx files (not the GFS idx files): i.e., ending in ".grib2.923a8.idx"

# Global variable for the number of worker proceses
n_worker_processes = 4  # If using 1, don't use this and use the non parallel version

import concurrent.futures
import multiprocessing
import signal

# Create a queue to store tasks
task_queue = multiprocessing.Queue()

# Maintain a set to track enqueued tasks
enqueued_tasks = set()

process_exiting_event = multiprocessing.Event()

process_lock = multiprocessing.Lock()

# Counter for completed tasks
completed_tasks_counter = multiprocessing.Value('i', 0)
completed_tasks_counter_lock = multiprocessing.Lock()

# Counter for number of total tasks enqueued
n_total_tasks = multiprocessing.Value('i', 0)
n_total_tasks_lock = multiprocessing.Lock()

def print_with_process_id(*args, **kwargs):
    try:
        process_number = os.getpid()
        print(f"P#{process_number}: ", end='')
        print(*args, **kwargs)
    except:
        print("No PID: ", end='')
        print(*args, **kwargs)

# Signal handler for Ctrl-C
def signal_handler(sig, frame):
    print_with_process_id("Received Ctrl-C. Waiting for current tasks to finish...")

    process_exiting_event.set()

    for _ in range(n_worker_processes + 1):
        task_queue.put(None)

    try:
        sys.exit(0)
    except Exception as e:
        print("Error exiting:")
        traceback.print_exc(limit=None, file=None, chain=True)

def worker():
    try:
        print_with_process_id("Worker process started.")
        while not process_exiting_event.is_set():
            try:
                # Get a task from the queue (blocking)
                task = task_queue.get()

                if task:
                    model_name, start_model_timestamp, model_time_step = task
                    process_model(model_name, start_model_timestamp, model_time_step)

                    with completed_tasks_counter_lock:
                        completed_tasks_counter.value += 1

                else:
                    # Only get None when exiting (Ctrl-C)
                    break

            except Exception as e:
                traceback.print_exc(limit=None, file=None, chain=True)
                #print_with_process_id(f"Error in worker: {e}")

        print_with_process_id("Worker finished")

    except Exception as e:
        print_with_process_id("Exception")
        traceback.print_exc(limit=None, file=None, chain=True)
        print_with_process_id(f"Worker process encountered an exception.")

def enqueue_task(model_name, start_model_timestamp, model_time_step):
    # Use a tuple to represent the task
    task = (model_name, start_model_timestamp, model_time_step)

    # Use a lock to ensure atomicity of set operations
    with process_lock:
        # Check if the task is already enqueued
        if task not in enqueued_tasks:
            with n_total_tasks_lock:
                n_total_tasks.value += 1
            task_queue.put(task)
            enqueued_tasks.add(task)

def process_model(model_name, start_model_timestamp, model_time_step):
    # only passing a single time step (as a list), as this allows us to keep the same code and parallelize it by time step
    all_candidates = calc_disturbances_by_model_name_date_and_time_steps(model_name, start_model_timestamp, [model_time_step])

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Start worker processes
with concurrent.futures.ProcessPoolExecutor(max_workers=n_worker_processes) as executor:
    futures = [executor.submit(worker) for _ in range(n_worker_processes)]

    print_with_process_id("Main process starting.")

    while not process_exiting_event.is_set():
        # Clear the set of enqueued tasks
        # This is needed as we don't want to enqueue tasks multiple times that results in multiple workers working the same queue item
        with process_lock:
            enqueued_tasks.clear()

        exiting = False

        for model_name in model_names_to_update:
            model_interval = model_interval_hours[model_name]
            current_time = datetime_utcnow()
            prev_interval_hour = (current_time.hour // model_interval) * model_interval
            start_model_timestamp = current_time.replace(hour=prev_interval_hour, minute=0, second=0, microsecond=0)
            start_model_timestamp -= timedelta(hours=hours_to_look_back)

            next_interval_hour = (current_time.hour // model_interval + 1) * model_interval

            if next_interval_hour == 24:
                next_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                next_interval_hour = 0
            else:
                next_time = current_time.replace(hour=next_interval_hour, minute=0, second=0, microsecond=0)

            end_timestamp = next_time if next_time >= start_model_timestamp else start_model_timestamp

            interval = timedelta(hours=model_interval)
            timestamps = []

            current_timestamp = start_model_timestamp
            while current_timestamp <= end_timestamp:
                timestamps.append(current_timestamp)
                current_timestamp += interval

            for model_timestamp in timestamps:
                model_hour_str = f'{model_timestamp.hour:02}'
                time_steps = all_time_steps_by_model[model_name][model_hour_str]

                for model_time_step in time_steps:
                    # Enqueue the task only if it's not already in the set
                    if not process_exiting_event.is_set():
                        enqueue_task(model_name, model_timestamp, model_time_step)

                    else:
                        exiting = True
                        break

                if exiting:
                    break

            if exiting:
                break

        if exiting:
            break

        time.sleep(30)
        completed_task_diff = 0
        with n_total_tasks_lock:
            with completed_tasks_counter_lock:
                completed_task_diff = n_total_tasks.value - completed_tasks_counter.value

        # avoid a greater likelihood of a race condition (repeating previous work) by waiting to add more tasks when there are only a few left
        if completed_task_diff <= n_worker_processes and completed_task_diff != 0:
            # a few tasks left but not zero
            time.sleep(60 * (polling_interval - 0.5))
        else:
            if completed_task_diff != 0:
                # many left (more than n_worker processes), wait until queue is empty before clearing enqueued tasks
                while not task_queue.empty():
                    # give some time to complete tasks before polling again
                    time.sleep(60 * (polling_interval - 0.5))
            else:
                # either none processed or a few processed very fast (wait for more data)
                time.sleep(60 * (polling_interval - 0.5))

# Perform any finalization logic after all processes have finished
print_with_process_id("Main process finished. Exiting")
