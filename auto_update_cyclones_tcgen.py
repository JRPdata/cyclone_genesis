# EXPERIMENTAL
# Work in progress (do not use)

# Process TCGEN products from NCEP (ensemble GFDL tracker) and ECMWF into a SQL db (for tcviewer)

# all units are stored in metric (data Vmax @ 10m printed in kts), but
# scale may be mixed between internal representations of data, thresholds, and printouts

# this is for accessing by model and storm (internal component id)
tc_candidates_db_file_path = 'tc_candidates_tcgen.db'

# shape file for placing lat,lon in basins which we are classifying
# each has an attr`ibute called 'basin_name', with CPAC & EPAC combined as EPAC
# basins are NATL, EPAC, WPAC, IO, SH
shape_file = 'shapes/basins.shp'

model_data_folders_by_model_name = {
    'EPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/eps-tcgen',
    'GEFS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen',
    'GEPS-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/geps-tcgen',
    'FNMOC-TCGEN': '/home/db/metview/JRPdata/globalmodeldata/fnmoc-tcgen'
}

# 'G' (track type "HC" instead of AL or EP) is default when no basin given (referencing GFDL source code: gettrk_main.f)
gfdl_basin_to_atcf_basin = {
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

# GFDL data has a warm core flag; this prunes any disturbances (points) BEFORE the time criteria is considered
prune_disturbances_without_warm_core = True

# Cache size for get_basin_name_from_lat_lon (use a power of 2)
# It is number of entries, not bytes; this may need to be made smaller if run out of memory)
cache_size = 1048576
# Here is the memory usage for a few different lru cache sizes based on a test:
# Cache size: 65536, Memory usage: 2.75 MB
# Cache size: 1048576, Memory usage: 44.00 MB
# Cache size: 16777216, Memory usage: 704.00 MB

# round some of the floats to 4 decimal places (lat, lon, mslp, rmw, moving_speed, direction, cps paramaters)
round_float_places = 4

# show start,end,max info for each TC for each member
verbose_print = False

# show file paths being processed
print_file_paths = True

# output calculation timing for debugging & optimization
debug_calc_exec_time = False

# write networkx graphs (original and reduced) of TC tracks (in graphs/ folder)
write_graphs = False

graphs_folder = '/home/db/Documents/JRPdata/cyclone-genesis/graphs'

# save disturbance maps
save_disturbance_maps = False

# save disturbance maps
save_tc_maps = False

import json
import pygrib
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

import traceback
import copy
import time
import os
import re

# cache results for calculating basin
from functools import lru_cache

import pytz

import warnings

from datetime import datetime, timedelta
import sqlite3

import io
import base64

import networkx as nx

from pyproj import Geod

from eccodes import codes_bufr_new_from_file, codes_get, codes_get_array, codes_release, codes_set, \
    CodesInternalError

# for converting lat/lons to basins
gdf = gpd.read_file(shape_file)

# shape file attribute names to threshold names
shape_basin_names_to_threshold_names_map = {
    'NATL': 'AL',
    'WPAC': 'WP',
    'IO': 'IO',
    'SH': 'SH',
    'EPAC': 'EP'
}

total_model_members_by_time_step = {
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
    }
}

gefs_members = ['GFSO', 'AC00', 'AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AP21', 'AP22', 'AP23', 'AP24', 'AP25', 'AP26', 'AP27', 'AP28', 'AP29', 'AP30']
geps_members = ['CMC', 'CC00', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06', 'CP07', 'CP08', 'CP09', 'CP10', 'CP11', 'CP12', 'CP13', 'CP14', 'CP15', 'CP16', 'CP17', 'CP18', 'CP19', 'CP20']
eps_members = ['ECHR', 'ECME', 'EE01', 'EE02', 'EE03', 'EE04', 'EE05', 'EE06', 'EE07', 'EE08', 'EE09', 'EE10', 'EE11', 'EE12', 'EE13', 'EE14', 'EE15', 'EE16', 'EE17', 'EE18', 'EE19', 'EE20', 'EE21', 'EE22', 'EE23', 'EE24', 'EE25', 'EE26', 'EE27', 'EE28', 'EE29', 'EE30', 'EE31', 'EE32', 'EE33', 'EE34', 'EE35', 'EE36', 'EE37', 'EE38', 'EE39', 'EE40', 'EE41', 'EE42', 'EE43', 'EE44', 'EE45', 'EE46', 'EE47', 'EE48', 'EE49', 'EE50']
fnmoc_members = ['NGX', 'NC00', 'NP01', 'NP02', 'NP03', 'NP04', 'NP05', 'NP06', 'NP07', 'NP08', 'NP09', 'NP10', 'NP11', 'NP12', 'NP13', 'NP14', 'NP15', 'NP16', 'NP17', 'NP18', 'NP19', 'NP20']

# Create a dictionary to map short strings to list names
model_name_to_ensemble_name = {}
for list_name, lst in zip(['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN'], [gefs_members, geps_members, eps_members, fnmoc_members]):
    for model_name in lst:
        model_name_to_ensemble_name[model_name] = list_name

# number of files expected per time-step (or member for tcgen)
expected_num_grib_files_by_model_name = {
    'EPS-TCGEN': 9999,
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
    'EPS-TCGEN': r'.*?A_JSXX\d+(?P<member>EC..)',
    'GEFS-TCGEN': r'.*?storms\.(?P<member>[a-z0-9]+)\.',
    'GEPS-TCGEN': r'.*?storms\.(?P<member>[a-z0-9]+)\.',
    'FNMOC-TCGEN': r'.*?storms\.(?P<member>[a-z0-9]+)\.'
}

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

# close old maps generated
old_maps = []

old_tc_maps = []

meters_to_knots = 1.943844

gfdl_column_names = [
    "Track Type",  # Column 1 (Either TG (tc genesis), ML (mid latitude tc genesis), HC (no basin), or Basin (AL, WP, SI, etc) (not very useful)
    "Num_Cyclogenesis",  # Column 2 (Also a short version of ATCF ID for invests/named storms)
    "Cyclogenesis_ID",  # Column 3
    "Model_Init_Time",  # Column 4 (fort.64 Column 3)
    "Record_Type",  # Column 5 (fort.64 Column 4) (this is the "technique type", which is 03 for models)
    "Model_ATCF_Name",  # Column 6 (fort.64 Column 5)
    "Forecast_Hour",  # Column 7 (fort.64 Column 6)
    "Vortex_Center_Lat",  # Column 8 (fort.64 Column 7)
    "Vortex_Center_Lon",  # Column 9 (fort.64 Column 8)
    "Max_10m_Wind",  # Column 10 (fort.64 Column 9)
    "Min_MSLP",  # Column 11 (fort.64 Column 10)
    "Storm_Type",  # Column 12 (fort.64 Column 11)
    "Threshold_Wind_Speed",  # Column 13 (fort.64 Column 12)
    "Radii_Quadrant",  # Column 14 (fort.64 Column 13)
    "NE_Radius",  # Column 15 (fort.64 Column 14)
    "SE_Radius",  # Column 16 (fort.64 Column 15)
    "SW_Radius",  # Column 17 (fort.64 Column 16)
    "NW_Radius",  # Column 18 (fort.64 Column 17)
    "Pressure_Last_Closed_Isobar",  # Column 19 (POUTER)
    "Radius_Last_Closed_Isobar",  # Column 20 (ROCI)
    "Radius_Max_Wind",  # Column 21 (RMW)
    "Phase_Space_Parameter_B",  # Column 22
    "Thermal_Wind_Lower_Troposphere",  # Column 23
    "Thermal_Wind_Upper_Troposphere",  # Column 24
    "Warm_Core_Presence",  # Column 25
    "Storm_Moving_Direction",  # Column 26
    "Storm_Moving_Speed",  # Column 27
    "Mean_850hPa_Vorticity",  # Column 28
    "Max_850hPa_Vorticity",  # Column 29
    "Mean_700hPa_Vorticity",  # Column 30
    "Max_700hPa_Vorticity",  # Column 31
    "UNKNOWN_C32", # Column 32
    "UNKNOWN_C33", # Column 33
    "UNKNOWN_C34", # Colunn 34
    "UNKNOWN_C35", # Column 35
    "UNKNOWN_C36",  # Column 36
    "UNKNOWN_C37",  # Column 37
    "UNKNOWN_C38",  # Colunn 38
    "UNKNOWN_C39",  # Column 39
]

#########################################################
########       CALCULATE DISTURBANCES            ########
#########################################################

def datetime_utcnow():
    return datetime.now(pytz.utc).replace(tzinfo=None)

# the full timestep string as used in the file names we are parsing from the model data
# include any leading zeros up that make sense (only up to what the model covers)
def convert_model_time_step_to_str(model_name, model_time_step):
    str_format = model_time_step_str_format[model_name]
    return f'{model_time_step:{str_format}}'

# call with no params before a func call, and then must pass it a func_str on the next call once func is complete
def debug_timing(func_str = None):
    if not debug_calc_exec_time:
        return
    if func_str is None:
        debug_time_start_stack.append(time.time())
        return

    debug_time_start = debug_time_start_stack.pop()
    debug_time_end = time.time()
    print(f'{func_str} execution time (seconds): {debug_time_end - debug_time_start:.1f}')

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
            print(grb)
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
            print("Parameter Name:", info["Parameter Name"])
            print("Parameter shortName:", info["Parameter shortName"])
            print("Unit:", info["Unit"])
            print("Level Type:", info["Level Type"])
            print("Level:", info["Level"])
            print("\n")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc(limit=None, file=None, chain=True)

def convert_to_signed_lon(lon):
    # Convert longitudes from 0-360 range to -180 to +180 range
    return (lon + 180) % 360 - 180

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
def print_candidates(mslp_minima_list, lats = None, lons = None, no_numbering=False):
    n = 0
    for candidate in mslp_minima_list:
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
        print(f"{numbering_str}{basin_str}Latitude (deg:): {lat: >6.1f}, Longitude (deg): {lon: >6.1f}, MSLP (hPa): {formatted_mslp}{roci_str}\n        {rv_str}{thickness_str}{vmax_str}\n        {vmax10m_in_roci_str}{closed_isobar_delta_str}")

# returns the basin name from lat, lon if in a basin that is covered by the thresholds, otherwise returns None
@lru_cache(maxsize=None)
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

#########################################################
######## PROCESS DISTURBANCES INTO TC CANDIDATES ########
#########################################################

def get_unprocessed_disturbance_model_runs():
    ensemble_status_dicts = get_num_files_processed_in_ensembles()
    # Sort the data from disturbances
    sorted_data, model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp = get_all_disturbances_sorted_by_timestamp(ensemble_status_dicts)

    if not sorted_data:
        return [], {}, {}

    tc_completed_by_model_name = get_model_timestamp_of_completed_tc_by_model_name()
    unprocessed_data = []
    for entry in sorted_data:
        model_name = entry['model_name']
        ensemble_name = entry['ensemble_name']
        model_timestamp = entry['model_timestamp']
        if model_name in tc_completed_by_model_name:
            if model_timestamp in tc_completed_by_model_name[model_name]:
                # check first to make sure if we want to reprocess a folder (previously only partially complete)
                if ensemble_name in ensemble_status_dicts and \
                    model_timestamp in ensemble_status_dicts[ensemble_name] and \
                    model_file_paths_by_model_name_and_timestamp[ensemble_name][model_timestamp] == ensemble_status_dicts[ensemble_name][model_timestamp]:
                    continue
        unprocessed_data.append(entry)

    return unprocessed_data, model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp

def calc_tc_candidates():
    unprocessed_data, model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp = get_unprocessed_disturbance_model_runs()
    if len(unprocessed_data) == 0:
        return
    if verbose_print:
        print("Calculate TC candidates (and simplified tracks)")
        print(f"# Model runs to process: {len(unprocessed_data)}")
        print("")
    for row in unprocessed_data:
        graph = create_graph_and_add_candidates(row)
        connect_graph_colocated(graph)
        connect_graph_next_timestep(graph)
        remove_disturbances_not_meeting_time_criteria(graph)
        process_and_simplify_graph(graph)

    if verbose_print:
        print("")
        print("Completed all model runs.")

    # Update processed
    update_ensemble_status(model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp)

# connect nodes at same timestep that are colocated
def connect_graph_colocated(graph):
    # Create colocation_# nodes for nodes for the same time step that are in same area (box radius # degrees)
    # we are only using DiGraph so we can only support one edge at a time (use multi for multiple radii edges)

    # Create a dictionary to group node names by time_step
    nodes_by_time_step = {}

    # Group nodes by time_step
    for node in graph.nodes():
        model_name, model_timestamp, time_step, lat, lon = node.split("_")
        if time_step not in nodes_by_time_step:
            nodes_by_time_step[time_step] = []
        nodes_by_time_step[time_step].append(node)

    # Loop through time_steps and nodes
    for time_step, node_list in nodes_by_time_step.items():
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]

                # Access node attributes from the graph
                node1_attributes = graph.nodes[node1]
                node2_attributes = graph.nodes[node2]

                # Compare lat/lon and get colocated box radii bins
                box_radii_bins = get_colocated_box_radii_bins(node1_attributes, node2_attributes, colocated_box_radii)

                if box_radii_bins is not None:
                    # Create edges with types "colocated_#"
                    for radius in box_radii_bins:
                        edge_type = f"colocated_{radius}"
                        graph.add_edge(node1, node2, edge_type=edge_type)
                        graph.add_edge(node2, node1, edge_type=edge_type)

# connect nodes on the next timestep that are colocated
def connect_graph_next_timestep(graph):
    # Create connections between disturbances between successive (6hr) timestamps that are in a 3 degree radii box
    # Loop through each graph
    # Create a dictionary to group node names by time_step
    nodes_by_time_step = {}

    # Group nodes by time_step
    for node in graph.nodes():
        model_name, model_timestamp, time_step, lat, lon = node.split("_")
        if time_step not in nodes_by_time_step:
            nodes_by_time_step[time_step] = []
        nodes_by_time_step[time_step].append(node)

    # Convert time step keys to integers for proper sorting
    time_step_keys = list(map(int, nodes_by_time_step.keys()))

    # Sort the time step keys in ascending order
    time_step_keys.sort()

    # Create time step pairs with smaller value first
    time_step_pairs = [(current, future) for current, future in zip(time_step_keys, time_step_keys[1:])
                      if (future - current) == 6]

    # Step 2: Loop through each time_step pair
    for current_time_step, future_time_step in time_step_pairs:
        current_nodes = nodes_by_time_step[str(current_time_step)]
        future_nodes = nodes_by_time_step[str(future_time_step)]

        # Step 3: Loop through nodes in the current time step
        for node1 in current_nodes:
            # Convert node1 attributes to a dictionary
            node1_attributes = graph.nodes[node1]

            # Step 4: Compare with all nodes (with replacement) in the future time step
            for node2 in future_nodes:
                # Convert node2 attributes to a dictionary
                node2_attributes = graph.nodes[node2]

                # Step 5: Check for colocated box radii bins of 3
                box_radii_bins = get_colocated_box_radii_bins(node1_attributes, node2_attributes, next_colocated_box_radii)
                if box_radii_bins is not None:
                    if largest_next_colocated_radius in box_radii_bins:
                        # Connect nodes with an edge
                        graph.add_edge(node1, node2, edge_type=next_colocated_edge_type)

def remove_disturbances_not_meeting_time_criteria(graph):
    # Remove disturbances that are not part of a path through time with at least 4 edges (5 nodes)
    # this is a model TC that exists for 24 hours (5 time steps) using the colocated 3-degree criteria (criteria #5 in [1])

    # Create a directed acyclic subgraph with only "next_colocated_3" edges
    subgraph = graph.edge_subgraph([(node1, node2) for node1, node2, data in graph.edges(data=True) if data["edge_type"] == next_colocated_edge_type])

    # Step 1: Find the topological order of the subgraph and in reverse
    topological_order = list(nx.topological_sort(subgraph))
    topological_order_reversed = list(reversed(topological_order))

    # Step 2: Initialize dictionaries to keep track of distances and predecessors
    distance = {node: 0 for node in subgraph.nodes}
    successor = {}

    # Step 3: Calculate the longest distance starting at each node
    for node in topological_order_reversed:
        successors = [(neighbor, distance[neighbor] + 1) for neighbor in subgraph.neighbors(node) if subgraph[node][neighbor]["edge_type"] == next_colocated_edge_type]
        if successors:
            max_successor = max(successors, key=lambda x: x[1])
            distance[node] = max_successor[1]
            successor[node] = max_successor[0]

    # Step 4: Calculate the maximum path length that each node is part of
    for current_node in topological_order:
        for future_node in subgraph.neighbors(current_node):
            if subgraph[current_node][future_node]["edge_type"] == next_colocated_edge_type:
                # Update the distance of the future node to the path length that is the longest
                distance[future_node] = max(distance[future_node], distance[current_node])

    # Step 5: Find the nodes that are part of paths with distance >= 4
    valid_nodes = [node for node in subgraph.nodes if distance[node] >= 4]

    # Step 6: Remove nodes that are not part of valid paths from the original graph
    nodes_to_remove = [node for node in graph.nodes() if node not in valid_nodes]
    graph.remove_nodes_from(nodes_to_remove)

def process_and_simplify_graph(graph):
    def get_first_node_valid_time(component):
        first_node = min(component, key=lambda node: datetime.fromisoformat(graph.nodes[node]["valid_time"]))
        return datetime.fromisoformat(graph.nodes[first_node]["valid_time"])

    # use for distance calculations to simplify nodes in connected TCs/lows
    g = Geod(ellps='WGS84')

    has_error = False
    if verbose_print:
        print("")
        print("=========================")
        print(graph.graph['name'])
        print("=========================")

    model_name = graph.graph['model_name']
    init_time = graph.graph['init_time']

    orig_file_graph_path = os.path.join(graphs_folder, f"Orig_{graph.graph['name']}.gexf")
    if write_graphs:
        if not os.path.exists(orig_file_graph_path):
            nx.write_gexf(graph, orig_file_graph_path, encoding="utf-8", prettyprint=True)

    # Extract connected components in the graph (these are possible TCs, with possibly merged lows/other TCs)
    components = list(nx.weakly_connected_components(graph))
    components.sort(key=get_first_node_valid_time)
    components_were_modified = True
    num_removed_nodes = 0
    num_removed_next_edges = 0
    num_removed_colocated_edges = 0
    num_components = len(components)
    # separate nodes that (likely) shouldn't be connected
    while components_were_modified:
        components_were_modified = False
        for component in components:
            color_by_nodes = {}
            for node in graph:
                color_by_nodes[node] = nx_default_color

            # there are multiple nodes per time step, find the nodes for each time step first
            nodes_by_valid_time = {}
            for node in component:
                valid_time = graph.nodes[node]['valid_time']
                if valid_time not in nodes_by_valid_time:
                    nodes_by_valid_time[valid_time] = []

                cur = nodes_by_valid_time[valid_time]
                cur.append(node)
                nodes_by_valid_time[valid_time] = cur
            # get the valid_times for all nodes
            node_valid_times = sorted(list(nodes_by_valid_time.keys()), key=lambda dt: datetime.fromisoformat(dt))

            # for each disjoint set remove nodes from a component
            #     timestep nodes to subgraph
            #     get subgraph of current time_step nodes, get components of subgraph:
            #      for each component (disjoint set):
            #          if num_nodes == 1 continue
            #          get node mslp_minimum and set of all other nodes in component (reducible nodes)
            #          get list of all component neighbor predecessor and successor nodes, next_colocated_3
            #             get subgraph comprised of mslp node + all component neighbor predecessor/successor nodes
            #               if num_components == 1 (that means it's fully connected through the minimum)
            #                  remove the set of reducible nodes from the main component

            # remove reducible nodes on a connected path
            #    reducible nodes are nodes at the same timestep connected to the MSLP minimum that can be removed without removing nodes
            repeat_disconnect = True
            while repeat_disconnect is True:
                # have to loop to take care of both 'merger' joins and splits
                component_changed = False

                for valid_time in node_valid_times:
                    time_step_nodes = nodes_by_valid_time[valid_time]
                    if len(time_step_nodes) == 0:
                        continue
                    time_step_graph = graph.subgraph(time_step_nodes)
                    # this is the disjoint set
                    time_step_components = list(nx.weakly_connected_components(time_step_graph))
                    for time_step_component in time_step_components:
                        if len(time_step_component) == 1:
                            continue

                        min_mslp_node = min(time_step_component, key=lambda node: time_step_graph.nodes[node]['mslp'])
                        # time step component predecessors and successors
                        successors = set()
                        predecessors = set()
                        for node in time_step_component:
                            node_successors = [neighbor for neighbor in graph.neighbors(node) if graph[node][neighbor]["edge_type"] == next_colocated_edge_type]
                            successors.update(node_successors)
                            node_predecessors = [neighbor for neighbor in graph.predecessors(node) if graph[neighbor][node]["edge_type"] == next_colocated_edge_type]
                            predecessors.update(node_predecessors)

                        # the set of the min mslp node in the timestep component and
                        #    all the successors/predecessors for the time_step_component
                        mslp_and_successors_and_predecessors = set()
                        mslp_and_successors_and_predecessors.update([min_mslp_node])
                        successors_and_predecessors = set()
                        successors_and_predecessors.update(successors)
                        successors_and_predecessors.update(predecessors)
                        mslp_and_successors_and_predecessors.update(successors_and_predecessors)
                        reducible_graph = graph.subgraph(list(mslp_and_successors_and_predecessors))
                        reducible_graph_components = list(nx.weakly_connected_components(reducible_graph))
                        if len(reducible_graph_components) == 1:
                            # it is reducible
                            reducible_nodes = copy.deepcopy(time_step_component)
                            reducible_nodes.remove(min_mslp_node)
                            #print('Removing nodes:',reducible_nodes)
                            graph.remove_nodes_from(list(reducible_nodes))
                            components_were_modified = True
                            # also remove from component
                            for reducible_node in reducible_nodes:
                                component.remove(reducible_node)
                            num_removed_nodes += len(reducible_nodes)
                            component_changed = True

                # update nodes_by_valid_time if necessary
                if component_changed:
                    repeat_disconnect = True
                    # there are multiple nodes per time step, find the nodes for each time step first
                    nodes_by_valid_time = {}
                    for node in component:
                        valid_time = graph.nodes[node]['valid_time']
                        if valid_time not in nodes_by_valid_time:
                            nodes_by_valid_time[valid_time] = []

                        cur = nodes_by_valid_time[valid_time]
                        cur.append(node)
                        nodes_by_valid_time[valid_time] = cur
                    # get the valid_times for all nodes
                    node_valid_times = sorted(list(nodes_by_valid_time.keys()), key=lambda dt: datetime.fromisoformat(dt))
                else:
                    repeat_disconnect = False

            repeat_remove_colocated_edges = True
            while repeat_remove_colocated_edges:
                repeat_remove_colocated_edges = False
                edges_to_remove = set()

                connected_steps = {}
                for valid_time in node_valid_times:
                    valid_time_nodes = nodes_by_valid_time[valid_time]
                    sg = graph.subgraph(valid_time_nodes).to_undirected()
                    connected_steps[valid_time] = nx.is_connected(sg)

                not_connected_segments = [None]
                for valid_time, is_connected in connected_steps.items():
                    last_segment = not_connected_segments[-1]
                    if is_connected:
                        if last_segment is not None:
                            not_connected_segments.append(None)

                    else:
                        if last_segment is None:
                            last_segment = []

                        last_segment.append(valid_time)
                        not_connected_segments[-1] = last_segment

                # find segment with length of 5 or greater
                segment_lengths = [len(x) for x in not_connected_segments if x is not None]
                max_segment_length = 0
                if segment_lengths:
                    max_segment_length = max(segment_lengths)

                if max_segment_length < 5:
                    #print("No cases with segments >= 5.")
                    continue

                for not_connected_segment in not_connected_segments:
                    if not_connected_segment is None:
                        continue

                    if len(not_connected_segment) < 5:
                        continue

                    segment_nodes = []
                    for valid_time in not_connected_segment:
                        segment_nodes.extend(nodes_by_valid_time[valid_time])

                    for node in segment_nodes:
                        color_by_nodes[node] = nx_split_color

                    segment_first_valid_time = not_connected_segment[0]
                    segment_last_valid_time = not_connected_segment[-1]
                    segment_first_index = node_valid_times.index(segment_first_valid_time)
                    segment_last_index = node_valid_times.index(segment_last_valid_time)
                    #segment_predecessor_index = segment_first_index - 1
                    segment_successor_index = segment_last_index + 1
                    #if segment_predecessor_index >= 0:
                    #    segment_predecessor_valid_time = node_valid_times[segment_predecessor_index]
                    #    print("  Node bunch before segment:", nodes_by_valid_time[segment_predecessor_valid_time])

                    #print("  Node bunch at start:", nodes_by_valid_time[segment_first_valid_time])
                    #print("  Node bunch at end of segment:", nodes_by_valid_time[segment_last_valid_time])
                    nodes_at_end_of_segment = nodes_by_valid_time[segment_last_valid_time]

                    if segment_successor_index < len(nodes_by_valid_time):
                        #segment_successor_valid_time = node_valid_times[segment_successor_index]
                        #print("  Node bunch after segment:", nodes_by_valid_time[segment_successor_valid_time])

                        # disconnect 'bridge' nodes at a merger (at start of a merger)
                        #nodes_after_end_of_segment = nodes_by_valid_time[segment_successor_valid_time]
                        # first get the nodes from the end of the segment and their disjoint sets (components)
                        time_step_graph = graph.subgraph(nodes_at_end_of_segment)
                        time_step_components = list(nx.weakly_connected_components(time_step_graph))
                        for i, time_step_component in enumerate(time_step_components):
                            # create a set of the nodes in all other components from this timestep
                            time_step_other_disjoint_sets = set()
                            for k, other_time_step_component in enumerate(time_step_components):
                                if i != k:
                                    time_step_other_disjoint_sets.update(other_time_step_component)

                            # create a list of successors that belong to the current disjoint set (time_step_component)
                            time_step_disjoint_set_successors = set()
                            for node in time_step_component:
                                disjoint_successor_nodes = [neighbor for neighbor in graph.neighbors(node) if graph[node][neighbor]["edge_type"] == next_colocated_edge_type]
                                time_step_disjoint_set_successors.update(disjoint_successor_nodes)

                            for node in time_step_component:
                                successor_nodes = [neighbor for neighbor in graph.neighbors(node) if graph[node][neighbor]["edge_type"] == next_colocated_edge_type]
                                for successor_node in successor_nodes:
                                    successor_node_predecessors = [neighbor for neighbor in graph.predecessors(successor_node) if graph[neighbor][successor_node]["edge_type"] == next_colocated_edge_type]
                                    successor_node_predecessors_set = set()
                                    successor_node_predecessors_set.update(successor_node_predecessors)
                                    if successor_node_predecessors_set.isdisjoint(time_step_other_disjoint_sets):
                                        # this successor node potentially has bridging edges
                                        # delete colocate_edges to other successor nodes not connected to the current disjoint set
                                        successor_node_colocated_nodes = [neighbor for neighbor in graph.neighbors(successor_node) if graph[successor_node][neighbor]["edge_type"] == largest_colocated_edge_type and neighbor not in time_step_disjoint_set_successors]
                                        if successor_node_colocated_nodes:
                                            for colocated_node in successor_node_colocated_nodes:
                                                edges_to_remove.add((successor_node, colocated_node))

                if edges_to_remove:
                    repeat_remove_colocated_edges = True
                    components_were_modified = True
                    for source_node, target_node in edges_to_remove:
                        num_removed_colocated_edges += 1
                        graph.remove_edge(source_node, target_node)
                else:
                    # find nodes in segments with predecessors (also in segment) belonging to different disjoint sets (a merge that can't be disconnected without simplifying first?)
                    for not_connected_segment in not_connected_segments:
                        if not_connected_segment is None:
                            continue

                        if len(not_connected_segment) < 5:
                            continue

                        # skip first valid time since we are removing next_colocate_3 coming from predecessors only in not_connected_segments
                        for valid_time in not_connected_segment[1:]:
                            segment_nodes = nodes_by_valid_time[valid_time]
                            time_step_graph = graph.subgraph(segment_nodes)
                            #time_step_components = list(nx.weakly_connected_components(time_step_graph))
                            for c in components:
                                for node in c:
                                    node_predecessors = [neighbor for neighbor in graph.predecessors(node) if graph[neighbor][node]["edge_type"] == next_colocated_edge_type]
                                    predecessors_time_step_graph = graph.subgraph(node_predecessors)
                                    predecessors_time_step_components = list(nx.weakly_connected_components(predecessors_time_step_graph))
                                    # skip nodes that don't have more than one disjoint predecessor
                                    if len(predecessors_time_step_components) == 1:
                                        continue

                                    node_mslp = graph.nodes[node]['mslp']
                                    node_lat = graph.nodes[node]['lat']
                                    node_lon = graph.nodes[node]['lon']

                                    # find which predecessor (component) is a better fit
                                    # if there is a predecessor that is a good fit in a disjoint set (component), keep the next_colocated_3 edges for the component
                                    mslp_diffs = []
                                    distance_diffs = []
                                    for i, predecessors_time_step_component in enumerate(predecessors_time_step_components):
                                        component_mslp_diffs = []
                                        component_distance_diffs = []
                                        for k, predecessor_node in enumerate(predecessors_time_step_component):
                                            predecessor_node_mslp = graph.nodes[predecessor_node]['mslp']
                                            predecessor_node_lat = graph.nodes[predecessor_node]['lat']
                                            predecessor_node_lon = graph.nodes[predecessor_node]['lon']
                                            node_mslp_diff = abs(predecessor_node_mslp - node_mslp)
                                            az1, az2, node_distance_diff =  g.inv(node_lon, node_lat, predecessor_node_lon, predecessor_node_lat)
                                            component_mslp_diffs.append(node_mslp_diff)
                                            component_distance_diffs.append(node_distance_diff)
                                        mslp_diffs.append(component_mslp_diffs)
                                        distance_diffs.append(component_distance_diffs)

                                    # for each (predecessor) node in the component, a 'good' candidate node satisfies:
                                    #   mslp and distance diffs that are smaller than all other nodes in other components
                                    component_candidate_node_indexes = []
                                    for i, component_mslp_diffs in enumerate(mslp_diffs):
                                        for k, node_mslp_diff in enumerate(component_mslp_diffs):
                                            node_distance_diff = distance_diffs[i][k]
                                            poor_candidate = False
                                            # compare against all other component nodes
                                            for x, cmp_component_mslp_diffs in enumerate(mslp_diffs):
                                                if x == i:
                                                    continue

                                                for y, cmp_node_mslp_diff in enumerate(cmp_component_mslp_diffs):
                                                    cmp_node_distance_diff = distance_diffs[x][y]
                                                    if cmp_node_mslp_diff <= node_mslp_diff or cmp_node_distance_diff <= node_distance_diff:
                                                        poor_candidate = True
                                                        break

                                                if poor_candidate:
                                                    break

                                            if not poor_candidate:
                                                component_candidate_node_indexes.append((i,k))

                                        if component_candidate_node_indexes:
                                            break

                                    if not component_candidate_node_indexes:
                                        continue

                                    # the component (predecessor) that is close, mslp and distance wise, to node
                                    component_candidate_index = component_candidate_node_indexes[0][0]
                                    # remove edges from other disjoint sets
                                    for i, predecessors_time_step_component in enumerate(predecessors_time_step_components):
                                        if i == component_candidate_index:
                                            continue

                                        for predecessor_node in predecessors_time_step_component:
                                            edges_to_remove.add((predecessor_node, node))

                    if edges_to_remove:
                        repeat_remove_colocated_edges = True
                        components_were_modified = True
                        for source_node, target_node in edges_to_remove:
                            graph.remove_edge(source_node, target_node)
                            num_removed_next_edges += 1

    # components have been simplified
    # refresh components as some components might have been split into multiple components
    components = list(nx.weakly_connected_components(graph))
    components.sort(key=get_first_node_valid_time)
    #components_were_modified = True
    if len(components) != num_components:
        if verbose_print:
            print(f"# components (before separation): {num_components}")
        num_components = len(components)
        if verbose_print:
            print(f"# components (after separation): {num_components}")
    else:
        if verbose_print:
            print(f"# components: {num_components}")

    for i, component in enumerate(components):
        nodes_by_valid_time = {}
        for node in component:
            valid_time = graph.nodes[node]['valid_time']
            if valid_time not in nodes_by_valid_time:
                nodes_by_valid_time[valid_time] = []

            cur = nodes_by_valid_time[valid_time]
            cur.append(node)
            nodes_by_valid_time[valid_time] = cur

        node_valid_times = sorted(list(nodes_by_valid_time.keys()), key=lambda dt: datetime.fromisoformat(dt))
        # create a simplified track by MSLP minimum (this may fail for nearby storms if not separated properly)
        max_10m_wind_speed_node = None
        simplified_nodes_by_valid_time = {}
        for valid_time in node_valid_times:
            valid_time_nodes = nodes_by_valid_time[valid_time]
            mslp_node = min(valid_time_nodes, key=lambda node: float(graph.nodes[node]['data']['mslp_value']))
            simplified_nodes_by_valid_time[valid_time] = mslp_node
            if 'vmax10m_in_roci' in graph.nodes[mslp_node]['data']:
                if max_10m_wind_speed_node is None:
                    max_10m_wind_speed_node = mslp_node

                if float(graph.nodes[mslp_node]['data']['vmax10m_in_roci']) > float(graph.nodes[max_10m_wind_speed_node]['data']['vmax10m_in_roci']):
                    max_10m_wind_speed_node = mslp_node

        node_valid_times = sorted(list(simplified_nodes_by_valid_time.keys()), key=lambda dt: datetime.fromisoformat(dt))
        # then, get the first and last nodes with the lowest MSLP value in their timestep
        first_time_step_nodes = nodes_by_valid_time[node_valid_times[0]]
        last_time_step_nodes = nodes_by_valid_time[node_valid_times[-1]]
        first_node = min(first_time_step_nodes, key=lambda node: float(graph.nodes[node]['data']['mslp_value']))
        last_node = min(last_time_step_nodes, key=lambda node: float(graph.nodes[node]['data']['mslp_value']))

        first_valid_time = graph.nodes[first_node]["valid_time"]
        last_valid_time = graph.nodes[last_node]["valid_time"]
        if verbose_print:
            print(f"  Start Valid Time: {first_valid_time}")
            print_candidates([graph.nodes[first_node]["data"]], no_numbering=True)
            print(f"  Last Valid Time: {last_valid_time}")
            print_candidates([graph.nodes[last_node]["data"]], no_numbering=True)

        max_10m_wind_speed = None
        if max_10m_wind_speed_node is not None:
            max_10m_wind_speed_valid_time = graph.nodes[max_10m_wind_speed_node]["valid_time"]
            if verbose_print:
                print(f"  Max 10m Wind Speed Valid Time: {max_10m_wind_speed_valid_time}")
                print_candidates([graph.nodes[max_10m_wind_speed_node]["data"]], no_numbering=True)
            max_10m_wind_speed = float(graph.nodes[max_10m_wind_speed_node]["data"]['vmax10m_in_roci'])

        if verbose_print:
            print("")

        # recover candidate data via simplified_nodes_by_valid_time
        tc_disturbance_candidates = []
        for valid_time in node_valid_times:
            node = simplified_nodes_by_valid_time[valid_time]
            tc_disturbance_candidates.append((
                graph.nodes[node]['time_step'], graph.nodes[node]['valid_time'], graph.nodes[node]['data']
            ))

        # add the simplified track to the tc_candidates.db
        ret = add_tc_candidate(model_name, init_time, i, max_10m_wind_speed, tc_disturbance_candidates)
        has_error = has_error or ret

    if num_removed_nodes:
        if verbose_print:
            print('# Removed nodes: ', num_removed_nodes)

    if num_removed_next_edges:
        if verbose_print:
            print('# Removed Next Edges: ', num_removed_next_edges)

    if num_removed_colocated_edges:
        if verbose_print:
            print('# Removed Colocated Edges: ', num_removed_colocated_edges)

    # mark model run completed if no errors
    if not has_error:
        update_tc_completed(model_name, init_time)

    # store the modified graph
    file_graph_path = os.path.join(graphs_folder, f"{graph.graph['name']}.gexf")
    if write_graphs:
        nx.write_gexf(graph, file_graph_path, encoding="utf-8", prettyprint=True)

    if verbose_print:
        print("")

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
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, '_COMPLETE_')):
                    model_dirs.append(os.path.join(root, dir))
                elif model_name == 'EPS-TCGEN':
                    partial_model_dirs.append(os.path.join(root, dir))
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

    tcgen_file_ext = ['txt', 'bin']
    ecm_hres_re = re.compile(r'.*?JSXX\d\dECMF')
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            f_ext = file.split('.')[-1]
            if f_ext in tcgen_file_ext:
                if is_complete:
                    model_file_paths.append(os.path.join(root, file))
                elif re.match(ecm_hres_re, file):
                    # files named *JSXXnnECMF* are HRES (ECMP are ensemble), we only want HRES for incomplete case
                    model_file_paths.append(os.path.join(root, file))

    return sorted(model_file_paths)

# reference gettrk_main.f subroutine output_atcf_gen()
# and HWRF users guide 4.0
# after the first 31 usual parameters (fort.66 sample in appendix D of the users guide)
# there are 8 more floating point values
# Note: Both documentation and comments in the code mention scaling the vorticity * 10^-5, but the code is * 10^-6
#   The vorticity values output only make sense if this is the case (checked with an active TC)
# There seems to be some concern about the calculation for the last closed isobar (so we will not transfer it over into our database):
#   ROCI calculation from GDFL's tracker (the units seem fine, and there is no scaling):
#   Using data from GFS 2024-16 00Z +6H on Ernesto (AL052024) as a reference this seems wrong or is a completely different measure then expected
#      (TC vitals from 06Z has the ROCI at 446 km and my own analysis of GFS puts it at 482 km (NCEP GFS 0.25 00Z+06H))
#   Pressure of last closed isobar also seems problematic (GFDL at 977; mine at 1112, tcvitals environmental pressure at 1112 also)
# Convert most units to metric (excepting threshold wind speed for wind radii 34/50/64)
def read_gdfl_txt_file_to_df(model_file_path):
    if print_file_paths:
        print('            ', model_file_path)
    kt_to_ms = 0.514444
    nmi_to_meters = 1852.0

    # As we will be using some of the strings we don't want any leading spaces
    try:
        df = pd.read_csv(model_file_path, header=None, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return None

    # Column numbers referenced in comments are 1 indexed per HWRF doc
    # Coerce columns 5, 7, 10-11, 13, 15-18 to integer
    for col in [4, 6] + list(range(9, 11)) + [12] + list(range(14, 19)):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Coerce columns 19-21 and 26-31 to numeric, replacing -999 with NaN
    for col in list(range(19, 22)) + list(range(25, 31)):
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(-999, np.nan)

    # Coerce columns 22-24 to numeric, replacing -9999 with NaN
    for col in range(22, 23):
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(-9999, np.nan)

    # Convert column 4 to datetime
    df[3] = pd.to_datetime(df[3], format='%Y%m%d%H')

    # Convert columns 22-24 to np.float32, divide by 10 (these are Hart's phase space parameters multiplied by 10)
    for col in range(21, 24):
        df[col] = np.float32(df[col]) / 10

    # Column 25 is warm core flag character (Y = Warm core, N = No warm core, U = Undetermined), leave as is

    # Convert column 27 to np.float32, multiply by 10 (moving speed is in units 10^1 m/s)
    df[26] = np.float32(df[26]) / 10

    # Convert columns 28-31 to np.float64, divide by 10^6 (these are the storm's relative vorticity values in units 10^-6 m/s)
    for col in range(27, 31):
        df[col] = np.float64(df[col]) / 1e6

    # Convert columns 10 from kt to ms (np.float32) (this is the column for VMax @ 10 m)
    df[9] = np.float32(df[9]) * kt_to_ms

    # Convert columns 15-18, 20-21 from n.mi to meters (np.float64) (Wind Radii, ROCI, and RMW are all in units n. miles)
    for col in list(range(14, 18)) + [19, 20]:
        df[col] = np.float64(df[col]) * nmi_to_meters

    # Now that the types are set and converted mostly, rename the columns
    if len(gfdl_column_names) == len(df.columns):
        df.columns = gfdl_column_names
    else:
        print("Error: Length of columns different from expected when processing file:", model_file_path)
        return None

    # Create the useful columns
    df['Model_Valid_Time'] = df['Model_Init_Time'] + pd.to_timedelta(df['Forecast_Hour'], unit='h')

    # Create new columns with empty strings for "ATCF_ID" and "Basin"
    df['ATCF_ID'] = ''
    df['Basin'] = ''
    df['Named_Storm'] = False
    df['Invest'] = False

    # Create new columns for signed lat and lon with np.float32 type
    df['Lat_Signed'] = np.float32(np.nan)
    df['Lon_Signed'] = np.float32(np.nan)

    # Iterate over rows to fill signed lat and lon columns
    for index, row in df.iterrows():
        lat = row['Vortex_Center_Lat']
        lon = row['Vortex_Center_Lon']

        # Convert lat and lon to signed values
        if lat[-1] == 'N':
            df.at[index, 'Lat_Signed'] = np.float32(lat[:-1]) / 10
        elif lat[-1] == 'S':
            df.at[index, 'Lat_Signed'] = -np.float32(lat[:-1]) / 10

        if lon[-1] == 'E':
            df.at[index, 'Lon_Signed'] = np.float32(lon[:-1]) / 10
        elif lon[-1] == 'W':
            df.at[index, 'Lon_Signed'] = -np.float32(lon[:-1]) / 10

    # Iterate over rows to fill "ATCF_ID" and "Basin" columns
    for index, row in df.iterrows():
        num_cyclogenesis = row['Num_Cyclogenesis']
        has_basin = False
        # Check if the value ends with a letter
        basin_short = num_cyclogenesis[-1]
        if basin_short.isalpha():
            # Check if the letter is in the basin mapping dictionary
            if basin_short in gfdl_basin_to_atcf_basin:
                basin_name = gfdl_basin_to_atcf_basin[basin_short]
                year = row['Model_Init_Time'].year
                atcf_number_str = num_cyclogenesis[:-1]
                atcf_number = int(atcf_number_str)
                if atcf_number < 70:
                    df.at[index, 'Named_Storm'] = True
                elif atcf_number >= 90:
                    df.at[index, 'Invest'] = True
                df.at[index, 'ATCF_ID'] = f"{basin_name}{atcf_number_str}{year}"

            lat = df.at[index, 'Lat_Signed']
            lon = df.at[index, 'Lon_Signed']

            # Change the basin name depending on the storm center (don't keep originating basin)
            basin_name_new = get_basin_name_from_lat_lon(lat, lon)
            if basin_name_new is not None:
                basin_name = basin_name
            df.at[index, 'Basin'] = basin_name
        else:
            lat = df.at[index, 'Lat_Signed']
            lon = df.at[index, 'Lon_Signed']
            basin_name = get_basin_name_from_lat_lon(lat, lon)
            if basin_name is not None:
                df.at[index, 'Basin'] = basin_name

    return df

def df_to_disturbances(ensemble_model_name, model_timestamp, model_member, df):
    # a row is all the disturbances generated by that model
    #  the main keys are the model forecast hour, and the value is a list of dicts
    #  each dict is then a point for that hour for each track

    disturbance = dict()
    disturbance["model_name"] = model_member
    disturbance["ensemble_name"] = ensemble_model_name
    # model_timestamp is the folder name in format YYYYMMDDHH
    dt = datetime.strptime(model_timestamp, '%Y%m%d%H')
    model_timestamp_isoformat = dt.isoformat()
    disturbance["model_timestamp"] = model_timestamp_isoformat

    if ensemble_model_name == "EPS-TCGEN":
        # EPS dataframe has different data, structure
        data = dict()

        for forecast_hour, tc_df in df.groupby('Forecast_Hour'):
            forecast_hour_dicts = []
            for index, row in tc_df.iterrows():
                mslp = row['Min_MSLP']

                if row['ATCF_ID'] and len(row['ATCF_ID']) >= 4:
                    atcf_id = row['ATCF_ID']
                else:
                    atcf_id = ''

                wind_radii_34 = [row['NE_34'], row['SE_34'], row['SW_34'], row['NW_34']]
                wind_radii_50 = [row['NE_50'], row['SE_50'], row['SW_50'], row['NW_50']]
                wind_radii_64 = [row['NE_64'], row['SE_64'], row['SW_64'], row['NW_64']]

                forecast_hour_dict = {
                    'mslp_value': float(np.round(mslp, round_float_places)),
                    'lat': np.round(row['Lat'], round_float_places),
                    'lon': np.round(row['Lon'], round_float_places),
                    'basin': row['Basin'],
                    'named': row['Named_Storm'],
                    'invest': None,
                    'atcf_id': atcf_id,
                    'vmax10m_in_roci': np.round(row['Max_10m_Wind'], 1),
                    'roci': float(np.nan),
                    'closed_isobar_delta': float(np.nan),
                    'rv850max': float(np.nan),
                    'rv700max': float(np.nan),
                    'rmw': np.round(row['Radius_Max_Wind'], round_float_places),
                    'cps_b': float(np.nan),
                    'cps_vtl': float(np.nan),
                    'cps_vtu': float(np.nan),
                    'warm_core': 'Unknown',
                    'storm_direction': float(np.nan),
                    'storm_speed': float(np.nan),
                    'wind_radii_34': wind_radii_34,
                    'wind_radii_50': wind_radii_50,
                    'wind_radii_64': wind_radii_64,
                    'lat_max_wind': np.round(row['Lat_of_Max_Wind'], round_float_places),
                    'lon_max_wind': np.round(row['Lon_of_Max_Wind'], round_float_places),
                    'criteria': {'all': True}
                }

                forecast_hour_dicts.append(forecast_hour_dict)

            data[str(int(forecast_hour))] = list(forecast_hour_dicts)

        disturbance["data"] = data
    else:
        data = dict()
        for forecast_hour, tc_df in df.groupby('Forecast_Hour'):
            forecast_hour_dicts = []
            for index, row in tc_df.iterrows():
                # Before doing anything for this record, check it against the pruning criteria
                mslp = row['Min_MSLP']
                criteria = True
                if row['Pressure_Last_Closed_Isobar'] and row['Pressure_Last_Closed_Isobar'] > 100:
                    closed_isobar_delta = row['Pressure_Last_Closed_Isobar'] - mslp
                    if closed_isobar_delta < 2:
                        criteria = False
                else:
                    closed_isobar_delta = float(np.nan)

                warm_core = 'Unknown'
                if row['Warm_Core_Presence']:
                    if row['Warm_Core_Presence'] == 'Y':
                        warm_core = 'True'
                    elif row['Warm_Core_Presence'] == 'N':
                        warm_core = 'False'
                        if prune_disturbances_without_warm_core:
                            criteria = False

                # Prune disturbances by warm core and MSLP
                # For MSLP ave to have at least 2 hPa difference between storm MSLP and POUTER)
                if not criteria:
                    continue

                if row['ATCF_ID'] and len(row['ATCF_ID']) >= 4:
                    atcf_id = row['ATCF_ID']
                else:
                    atcf_id = ''

                if row['Radius_Last_Closed_Isobar'] and row['Radius_Last_Closed_Isobar'] > 1:
                    roci = row['Radius_Last_Closed_Isobar']
                else:
                    roci = float(np.nan)

                # only NEQ34 is seen...
                wind_radii = [np.round(row['NE_Radius'], round_float_places),
                              np.round(row['SE_Radius'], round_float_places),
                              np.round(row['SW_Radius'], round_float_places),
                              np.round(row['NW_Radius'], round_float_places)]
                radii_speed = int(row['Threshold_Wind_Speed'])
                # this will be wind_radii_34, fill in missing columns
                wind_radii_col_name = f'wind_radii_{radii_speed}'
                missing_speeds = list(set([34, 50, 64]) - set([radii_speed]))
                # fill in None so we have same number of wind radii as EPS
                wind_radii_col_missing1 = missing_speeds[0]
                wind_radii_col_missing2 = missing_speeds[1]

                forecast_hour_dict = {
                    'mslp_value': float(np.round(mslp, round_float_places)),
                    'lat': np.round(row['Lat_Signed'], round_float_places),
                    'lon': np.round(row['Lon_Signed'], round_float_places),
                    'basin': row['Basin'],
                    'named': row['Named_Storm'],
                    'invest': row['Invest'],
                    'atcf_id': atcf_id,
                    'vmax10m_in_roci': np.round(row['Max_10m_Wind'], 1),
                    'roci': np.round(roci, 0),
                    'closed_isobar_delta': closed_isobar_delta,
                    'rv850max': row['Max_850hPa_Vorticity'],
                    'rv700max': row['Max_700hPa_Vorticity'],
                    'rmw': np.round(row['Radius_Max_Wind'], 0),
                    'cps_b': np.round(row['Phase_Space_Parameter_B'], round_float_places),
                    'cps_vtl': np.round(row['Thermal_Wind_Lower_Troposphere'], round_float_places),
                    'cps_vtu': np.round(row['Thermal_Wind_Lower_Troposphere'], round_float_places),
                    'warm_core': warm_core,
                    'storm_direction': np.round(row['Storm_Moving_Direction'], round_float_places),
                    'storm_speed': np.round(row['Storm_Moving_Speed'], round_float_places),
                    wind_radii_col_name: wind_radii,
                    wind_radii_col_missing1: None,
                    wind_radii_col_missing2: None,
                    'lat_max_wind': None,
                    'lon_max_wind': None,
                    'criteria': {'all': criteria}
                }

                forecast_hour_dicts.append(forecast_hour_dict)

            data[str(int(forecast_hour))] = list(forecast_hour_dicts)

        disturbance["data"] = data

    return disturbance

# sometimes we get a scalar instead of an array when all of the values are the same
def fill_array_from_scalar(array, num):
    if array.shape[0] == num:
        return array
    elif array.shape[0] == 1:
        if num is None or (num is not None and (np.isnan(num) or num < -999)):
            num = np.nan
        return array.repeat(np.round(num, 0))

# Reads a tropical cyclone bufr .bin file (ECMWF) into a dataframe
# References:
# [1] https://confluence.ecmwf.int/display/ECC/bufr_read_tropical_cyclone
# [2] https://confluence.ecmwf.int/display/FCST/Update+to+Tropical+Cyclone+tracks
# https://confluence.ecmwf.int/display/FCST/New+Tropical+Cyclone+Wind+Radii+product
def read_tc_bufr_to_df(file_path):
    if print_file_paths:
        print('            ', file_path)
    # developed in ecmwf_bufr_dev.ipynb notebook (referenced ECMWF code above)
    # open BUFR file
    f = open(file_path, 'rb')

    # use to calculate RMW from lat/lon vmax
    g = Geod(ellps='WGS84')

    cnt = 0
    rows = []

    # clockwise direction quadrant map (start of swath)
    quadrant_map = {
        0: 'NE',
        90: 'SE',
        180: 'SW',
        270: 'NW'
    }

    # m/s to kt map for wind radii wind speed thresholds
    wind_speed_threshold_map = {
        18: 34,
        26: 50,
        33: 64
    }

    # 4 quads
    num_quad = len(quadrant_map.keys())

    # 3 wind thresholds
    num_wind_radii_per_quad = len(wind_speed_threshold_map.keys())
    analysis_time = True

    # loop for the messages in the file
    while 1:
        # get handle for message
        bufr = codes_bufr_new_from_file(f)
        if bufr is None:
            break

        # we need to instruct ecCodes to expand all the descriptors
        # i.e. unpack the data values
        codes_set(bufr, 'unpack', 1)

        numObs = codes_get(bufr, "numberOfSubsets")
        year = codes_get(bufr, "year")
        month = codes_get(bufr, "month")
        day = codes_get(bufr, "day")
        hour = codes_get(bufr, "hour")
        minute = codes_get(bufr, "minute")

        # Edge case: this is a guess but might be wrong for any late December storms that crossed into the new year
        short_storm_id = codes_get(bufr, "stormIdentifier")
        if short_storm_id:
            short_storm_id = short_storm_id.strip()

        # How many different timePeriod in the data
        # this is actually +1 the number of actual #n#timePeriod since the +00 forecast hour has none
        numberOfPeriods = 0
        while True:
            numberOfPeriods = numberOfPeriods + 1
            try:
                codes_get_array(bufr, "#%d#timePeriod" % numberOfPeriods)
            except CodesInternalError as err:
                break

        # Get ensembleMemberNumber (this is a list of member numbers)
        memberNumber = codes_get_array(bufr, "ensembleMemberNumber")
        memberNumberLen = len(memberNumber)

        # for tc genesis, this is a scalar
        ensembleForecastType = codes_get_array(bufr, "ensembleForecastType")
        ensembleForecastType = fill_array_from_scalar(ensembleForecastType, memberNumberLen)

        long_storm_name = codes_get(bufr, "longStormName")
        if long_storm_name:
           long_storm_name = long_storm_name.strip()

        is_named_storm = False
        basin_name = None
        atcf_id = ''
        last_basin_names = {}
        # not possible to identify discriminate Invests from the data given the naming schema
        if long_storm_name:
            if short_storm_id != long_storm_name:
                # Named
                if re.match(r'^[a-zA-Z]+$', long_storm_name):
                    is_named_storm = True
                if is_named_storm or long_storm_name[0] == '9':
                    # If it's not a named storm, it is either: 1) a TD or 2) 'potential TC' given an ATCF ID
                    # It may still have the long name of the invest so use the short_storm_id
                    basin_short = short_storm_id[-1]
                    if basin_short.isalpha():
                        # Check if the letter is in the basin mapping dictionary
                        if basin_short in gfdl_basin_to_atcf_basin:
                            basin_name = gfdl_basin_to_atcf_basin[basin_short]
                            year_str = str(int(year))
                            atcf_number_str = short_storm_id[:-1]
                            atcf_number = int(atcf_number_str)
                            atcf_id = f"{basin_name}{atcf_number_str}{year_str}"

        model_names = []
        for i in range(len(memberNumber)):
            if ensembleForecastType[i] == 4:
                # Perturbatin member
                mem_num = memberNumber[i]
                model_names.append(f'EE{mem_num:02d}')
            elif ensembleForecastType[i] == 1:
                # Control member
                model_names.append(f'ECME')
            elif ensembleForecastType[i] == 0:
                # Could not find the model technique name for HRES model ... making up as ECHR
                model_names.append(f'ECHR')

        # Get the analysis (+00Z initial perturbations) and forecast hours
        for i in range(0, numberOfPeriods):
            wind_radii = {}
            # the offsets (ranks) for repeated field names (enumerated by bufr reader)
            rank1 = i * 2 + 2
            rank3 = i * 2 + 3
            rank_wind_threshold = (i * num_wind_radii_per_quad) + 1
            rank_wind_radii = (i * num_quad * num_wind_radii_per_quad) + 1
            rank_quad = (i * 2 * num_quad * num_wind_radii_per_quad) + 1

            if i != 0:
                ivalues = codes_get_array(bufr, "#%d#timePeriod" % i)
                timePeriod = ivalues[0] if len(ivalues) == 1 else ivalues[1]
            else:
                timePeriod = 0

            # Fetch arrays for each variable
            lats = codes_get_array(bufr, "#%d#latitude" % rank1)
            lons = codes_get_array(bufr, "#%d#longitude" % rank1)
            mslp = codes_get_array(bufr, "#%d#pressureReducedToMeanSeaLevel" % (i + 1))
            lat_vmax = codes_get_array(bufr, "#%d#latitude" % rank3)
            lon_vmax = codes_get_array(bufr, "#%d#longitude" % rank3)
            vmax10m = codes_get_array(bufr, "#%d#windSpeedAt10M" % (i + 1))

            lats = fill_array_from_scalar(lats, memberNumberLen)
            lons = fill_array_from_scalar(lons, memberNumberLen)
            mslp = fill_array_from_scalar(mslp, memberNumberLen)
            lat_vmax = fill_array_from_scalar(lat_vmax, memberNumberLen)
            lon_vmax = fill_array_from_scalar(lon_vmax, memberNumberLen)
            vmax10m = fill_array_from_scalar(vmax10m, memberNumberLen)

            # wind radii labels
            # there are 2 azimuth per quadrant as a span, 4 quadrants per period, 3 thresholds
            # (the first of the two is going clockwise from North == 0 degrees)
            # order is by threshold then by quadrant (threshold1-quad1, threshold1, quad2, etc..)
            for r in range(num_wind_radii_per_quad):
                wind_num = rank_wind_threshold + r
                wind_threshold_ms = codes_get(bufr, "#%d#windSpeedThreshold" % wind_num)
                wind_threshold_kt = wind_speed_threshold_map[int(wind_threshold_ms)]
                for q in range(num_quad):
                    az_num = rank_quad + (r * num_quad * 2) + (2 * q)
                    azimuth = codes_get(bufr, "#%d#bearingOrAzimuth" % az_num)

                    quad_name = quadrant_map[int(azimuth)]
                    col_name = f'{quad_name}_{wind_threshold_kt}'

                    radii_num = rank_wind_radii + (r * num_quad) + q
                    wind_in_quad = codes_get_array(bufr,
                                                   "#%d#effectiveRadiusWithRespectToWindSpeedsAboveThreshold" % radii_num)
                    wind_in_quad = fill_array_from_scalar(wind_in_quad, memberNumberLen)
                    wind_radii[col_name] = wind_in_quad

            # Loop over the fetched lists to construct rows
            for i in range(len(memberNumber)):
                valid_lat_lon = True
                row_lat = lats[i]
                if row_lat is None or (row_lat is not None and (np.isinf(row_lat) or row_lat < -500)):
                    row_lat = np.nan
                    valid_lat_lon = False

                row_lon = lons[i]
                if row_lon is None or (row_lon is not None and (np.isinf(row_lon) or row_lon < -500)):
                    row_lon = np.nan
                    valid_lat_lon = False

                valid_lat_lon_vmax = True
                row_lat_vmax = lat_vmax[i]
                if row_lat_vmax is None or (row_lat_vmax is not None and (np.isinf(row_lat_vmax) or row_lat_vmax < -500)):
                    row_lat_vmax = np.nan
                    valid_lat_lon_vmax = False

                row_lon_vmax = lon_vmax[i]
                if row_lon_vmax is None or (row_lon_vmax is not None and (np.isinf(row_lon_vmax) or row_lon_vmax < -500)):
                    row_lon_vmax = np.nan
                    valid_lat_lon_vmax = False

                rmw = float(0)
                if i not in last_basin_names:
                    last_basin_names[i] = basin_name
                if valid_lat_lon:
                    if valid_lat_lon_vmax:
                        az12, az21, rmw = g.inv(row_lon, row_lat, row_lon_vmax, row_lat_vmax)
                        if np.isnan(rmw):
                            rmw = float(0)
                        else:
                            rmw = float(np.round(rmw, 0))
                    basin_name_new = get_basin_name_from_lat_lon(row_lat, row_lon)
                    if basin_name_new is not None:
                        last_basin_names[i] = basin_name_new

                member_basin_name = last_basin_names[i]

                min_mslp = mslp[i]
                if np.isnan(min_mslp) or np.isinf(min_mslp) or min_mslp < -99:
                    min_mslp = np.nan

                vmax = vmax10m[i]
                if np.isnan(vmax) or np.isinf(vmax) or vmax < -99:
                    vmax = np.nan

                row = {
                    'Storm_Name': long_storm_name,
                    "Short_Storm_ID": short_storm_id,
                    'ATCF_ID': atcf_id,
                    'Model_ATCF_Name': model_names[i],
                    'Model_Init_Time': pd.Timestamp(year, month, day, hour, minute),
                    'Model_Valid_Time': pd.Timestamp(year, month, day, hour, minute) + pd.Timedelta(hours=timePeriod),
                    'Forecast_Hour': np.int32(int(np.round(timePeriod))),
                    'Named_Storm': is_named_storm,
                    'Member_Num': np.int16(memberNumber[i]),
                    'Lat': np.float32(row_lat),
                    'Lon': np.float32(row_lon),
                    'Basin': member_basin_name,
                    'Min_MSLP': np.float32(min_mslp),
                    'Max_10m_Wind': np.float32(vmax),
                    'Lat_of_Max_Wind': np.float32(row_lat_vmax),
                    'Lon_of_Max_Wind': np.float32(row_lon_vmax),
                    'Radius_Max_Wind': rmw,
                    'Valid': valid_lat_lon
                }

                for col_name in wind_radii.keys():
                    value = wind_radii[col_name][i]
                    if value is None or (value is not None and (np.isnan(value) or value < -99)):
                        value = np.nan
                    row[col_name] = np.round(np.float32(value), 0)

                #for col_name in wind_radii.keys():
                #    row[col_name] = np.int32(wind_radii[col_name][i])
                rows.append(row)

        cnt += 1

        # release the BUFR message
        codes_release(bufr)

    # close the file
    f.close()

    df = pd.DataFrame(rows)

    # fix units for MSLP to hPa, mb from Pa
    df['Min_MSLP'] /= 100

    return df


def get_disturbances_from_gfdl_txt_files(ensemble_model_name, model_files_by_stamp):
    '''
    TG, 0121, 2024081518_F090_106N_1260W_FOF, 2024081518, 03, AC00, 114, 132N, 1253W,  43, 1002, XX,  34, NEQ, 0051, 0101, 0000, 0000, 1008,  130,   45,      3,    395,     27, N,  358,   43,  341,  557,  257,  558,    258.4,    258.1,    -20.6,     -9.0,     20.3,     79.0,    258.4,    258.8
    TG, 0121, 2024081518_F090_106N_1260W_FOF, 2024081518, 03, AC00, 120, 138N, 1258W,  41, 1004, XX,  34, NEQ, 0059, 0109, 0000, 0000, 1010,  107,   50,     85,    461,    266, Y, -999, -999,  339,  531,  279,  459,    258.9,    258.3,    -23.9,    -10.7,     17.3,     77.4,    258.9,    259.3
    SI,  90S, 2024081518_F000_053S_0734E_90S, 2024081518, 03, AC00, 000,  55S,  738E,  31, 1004, XX,  34, NEQ, 0000, 0000, 0000, 0000, 1009,  145,   70,   -999,  -9999,  -9999, Y,  182,   21,  276,  487,  247,  406,    258.6,    258.1,    -34.0,    -12.9,     82.7,     92.6,    258.6,    259.2
    HC,  02G, 2024081518_F000_221N_1089E_02G, 2024081518, 03, AC00, 000, 219N, 1087E,  13, 1003, XX,  34, NEQ, 0000, 0000, 0000, 0000, 1004,  341,  142,   -999,  -9999,  -9999, N,  100,   18,   68,  127,   56,   89,    260.3,    259.8,     -8.6,      1.9,     41.3,     76.2,    260.3,    260.4
    '''
    disturbances = []

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
            #gfso_test = '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen/2024081600/2024081600_storms.gfso.atcf_gen.altg.2024081600.txt'
            #gefs_test = '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen/2024081600/2024081600_storms.ac00.atcf_gen.altg.2024081600.txt'
            df = read_gdfl_txt_file_to_df(model_file_path)
            if df is not None:
                disturbances_from_model = df_to_disturbances(ensemble_model_name, model_timestamp, model_member, df)
                disturbances.append(disturbances_from_model)

    return disturbances

def get_disturbances_from_bufr_files(ensemble_model_name, model_files_by_stamp):
    ensemble_model_name = "EPS-TCGEN"
    # disturbances should be a list of models' disturbances
    # each entry should be a member of the ensemble at a specific model init time
    # each such entry is a dict that contains the disturbances for that model

    disturbances = []

    # for EPS we have to first process all the dfs in a directory (a specific model init time)
    # then we have to merge the dfs and then split by model member to process each member's tracks
    for model_timestamp, model_file_paths in model_files_by_stamp.items():
        # match whether this is the deterministic or the ensemble model
        model_type_re = re.compile(model_member_re_str_by_model_name[ensemble_model_name])
        eps_dfs = []
        hres_dfs = []
        for model_file_path in model_file_paths:
            f = os.path.basename(model_file_path)
            res = re.match(model_type_re, f)
            is_ensemble = False
            if res:
                model_type = res['member'].upper()
                if model_type == 'ECEP':
                    # EPS (model ensemble)
                    is_ensemble = True
                elif model_type == 'ECMF':
                    # HRES (deterministic)
                    pass
                else:
                    print("Warning: Skipping file! Could not match member for: ", model_file_path)
                    continue
            else:
                print("Warning: Skipping file! Could not match member for: ", model_file_path)
                continue
            #gfso_test = '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen/2024081600/2024081600_storms.gfso.atcf_gen.altg.2024081600.txt'
            #gefs_test = '/home/db/metview/JRPdata/globalmodeldata/gefs-tcgen/2024081600/2024081600_storms.ac00.atcf_gen.altg.2024081600.txt'
            df = read_tc_bufr_to_df(model_file_path)
            if df is not None:
                # Drop all invalid rows
                df = df[df['Valid']]
                if is_ensemble:
                    eps_dfs.append(df)
                else:
                    hres_dfs.append(df)

        # Create an empty dictionary to store the concatenated DataFrames
        df_by_member_name = {}

        # Iterate over the unique Model_ATCF_Name values
        all_dfs = eps_dfs + hres_dfs
        model_names = set()
        for df in all_dfs:
            new_names = set(df['Model_ATCF_Name'])
            model_names = model_names.union(new_names)
        for model_name in model_names:
            hres_dfs_filtered = [df for df in hres_dfs if df['Model_ATCF_Name'].eq(model_name).any()]
            eps_dfs_filtered = [df for df in eps_dfs if df['Model_ATCF_Name'].eq(model_name).any()]

            # Concatenate filtered DataFrames
            if hres_dfs_filtered and eps_dfs_filtered:
                concat_df = pd.concat(hres_dfs_filtered + eps_dfs_filtered).drop_duplicates(keep='first')
            elif hres_dfs_filtered:
                concat_df = pd.concat(hres_dfs_filtered).drop_duplicates(keep='first')
            elif eps_dfs_filtered:
                concat_df = pd.concat(eps_dfs_filtered).drop_duplicates(keep='first')
            else:
                continue

            member_df = concat_df[concat_df['Model_ATCF_Name'] == model_name]

            df_by_member_name[model_name] = member_df

        for model_name, df in df_by_member_name.items():
            disturbances_from_model = df_to_disturbances(ensemble_model_name, model_timestamp, model_name, df)
            disturbances.append(disturbances_from_model)

    return disturbances

# returns all disturbances (a list of disturbance dicts, sorted by model init timestamp) for unprocessed runs
def get_all_disturbances_sorted_by_timestamp(ensemble_status_dicts):
    # get completed model folders first (not processed yet)
    # ECM HRES should be an exception as it is separate from genesis
    # i.e.  files named *JSXXnnECMF* are HRES and completed earlier than the ensemble members!
    completed_folders_by_model_name, partial_folders_by_model_name = get_completed_unprocessed_model_folders_by_model_name()
    if not completed_folders_by_model_name and not partial_folders_by_model_name:
        return [], {}, {}

    model_file_paths_by_model_name_and_timestamp = {}
    # as above but excluding partial directories (used to keep track of complete folders)
    complete_file_paths_by_model_name_and_timestamp = {}
    # get all relevant files first
    # we use model name here but we mean ensemble name
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
                    model_file_paths_by_timestamp[model_timestamp] = model_file_paths
            if model_file_paths_by_timestamp:
                # check here whether we should skip (processed same exact number of files already)
                if ensemble_status_dicts and model_name in ensemble_status_dicts and model_timestamp in ensemble_status_dicts[model_name]:
                    num_files_processed = ensemble_status_dicts[model_name][model_timestamp]
                else:
                    num_files_processed = 0
                # either process a new directory, or reprocess a directory that has new files (all files reprocessed)
                # the number of components (tracks) in the candidates table should not decrease so it should be fine
                if num_files_processed != len(model_file_paths_by_timestamp[model_timestamp]):
                    model_file_paths_by_model_name_and_timestamp[model_name] = model_file_paths_by_timestamp
                    if is_partial == 0:
                        complete_file_paths_by_model_name_and_timestamp[model_name] = model_file_paths_by_timestamp

    if not model_file_paths_by_model_name_and_timestamp:
        return [], {}, {}

    disturbances = []
    for model_name, model_files_by_timestamp in model_file_paths_by_model_name_and_timestamp.items():
        if model_name == "EPS-TCGEN":
            new_disturbances = get_disturbances_from_bufr_files(model_name, model_files_by_timestamp)
        else:
            new_disturbances = get_disturbances_from_gfdl_txt_files(model_name, model_files_by_timestamp)
        if new_disturbances:
            disturbances.extend(new_disturbances)

    #model_name, model_timestamp, data = row
    #retrieved_data = {
    #    "model_name": model_name,
    #    "model_timestamp": model_timestamp,
    #    "data": json.loads(data)
    #}
    #all_retrieved_data.append(retrieved_data)

    sorted_disturbances = sorted(disturbances, key=lambda x: x["model_timestamp"])

    # return also the files we have intermediately, but not completely processed
    return sorted_disturbances, model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp

# Function to create graph for a model run (a row in disturbances database) and add candidates as nodes
def create_graph_and_add_candidates(row):
    model_name = row["model_name"]
    model_timestamp = row["model_timestamp"]
    data = row["data"]

    # Create a new graph for this row
    G = nx.DiGraph()
    G.graph["name"] = f"{model_name}_{model_timestamp}"
    G.graph["model_name"] = f"{model_name}"
    G.graph["init_time"] = f"{model_timestamp}"

    # Parse the model_timestamp as a datetime
    model_timestamp_datetime = datetime.fromisoformat(model_timestamp)

    for time_step, candidates in data.items():
        # Convert the time_step to an integer (it's in hours)
        time_step_hours = int(time_step)
        # Calculate the valid_time by adding the time_step as a timedelta
        valid_time_datetime = model_timestamp_datetime + timedelta(hours=time_step_hours)
        # Convert the valid_time back to the desired format
        valid_time = valid_time_datetime.isoformat()

        for candidate in candidates:
            lat = candidate.get("lat", "")
            lon = candidate.get("lon", "")
            mslp = float(candidate.get("mslp_value", ""))

            # Construct the node name based on the specified format
            node_name = f"{model_name}_{model_timestamp}_{time_step}_{lat}_{lon}"
            # Add the node to the graph
            G.add_node(node_name, lat=lat, lon=lon, mslp=mslp, time_step=time_step, valid_time=valid_time, init_time=model_timestamp, data=candidate)

    return G

def get_colocated_box_radii_bins(node1, node2, predefined_radii=[2]):
    # Extract lat and lon from node attributes and convert to float
    # lat and lon must be signed (lat between -90 and 90, and lon between -180 and 180)
    lat1 = float(node1["lat"])
    lon1 = float(node1["lon"])
    lat2 = float(node2["lat"])
    lon2 = float(node2["lon"])

    # Predefined radii
    #predefined_radii = [2, 3, 5]
    #predefined_radii = [2, 3]
    colocated_radii_bins = []

    for radius in predefined_radii:
        # Calculate the distance in degrees for latitude and longitude
        lat_distance = max(lat1, lat2) - min(lat1, lat2)
        lon_distance = max(lon1, lon2) - min(lon1, lon2)

        # Handle wrapping for latitude and longitude
        lat_distance_wrapped = min(lat_distance, abs(lat_distance - 180))
        lon_distance_wrapped = min(lon_distance, abs(lon_distance - 360))

        # Check if node2 is within the box for this radius
        if lat_distance_wrapped <= radius and lon_distance_wrapped <= radius:
            colocated_radii_bins.append(radius)

    if colocated_radii_bins:
        return colocated_radii_bins
    else:
        return None

# get dict of the completed ensembles (this only has the number of files processed)
# dict of number of files processed by ensemble name and init date
def get_num_files_processed_in_ensembles():
    processed_ensembles_dicts = get_ensemble_status()
    # create a dict by model_name from the rows
    processed_ensembles_dicts_by_name = {}
    if processed_ensembles_dicts:
        for processed_ensembles_dict in processed_ensembles_dicts:
            ensemble_name = processed_ensembles_dict['ensemble_name']
            ensemble_init_date = processed_ensembles_dict['ensemble_init_date']
            num_files_processed = processed_ensembles_dict['num_files_processed']
            if ensemble_name not in processed_ensembles_dicts_by_name:
                processed_ensembles_dicts_by_name[ensemble_name] = {}

            processed_ensembles_dicts_by_name[ensemble_name][ensemble_init_date] = num_files_processed
    return processed_ensembles_dicts_by_name

# get status of how many files last processed for the ensembles
def get_ensemble_status():
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ens_status (
                id INTEGER PRIMARY KEY,
                ensemble_name TEXT,
                init_date TEXT,
                num_files_processed INTEGER,
                expected_num_files_processed INTEGER,
                completed INTEGER,
                UNIQUE(ensemble_name, init_date)
            )
        ''')

        cursor.execute('SELECT ensemble_name, init_date, num_files_processed FROM ens_status ORDER BY init_date')
        results = cursor.fetchall()
        if results:
            # Process data for each row
            for row in results:
                ensemble_name, ensemble_timestamp, num_files_processed = row
                dt = datetime.fromisoformat(ensemble_timestamp)
                init_date = dt.strftime('%Y%m%d%H')

                retrieved_data = {
                    "ensemble_name": ensemble_name,
                    "ensemble_init_date": init_date,
                    "num_files_processed": num_files_processed
                }
                all_retrieved_data.append(retrieved_data)

    except sqlite3.Error as e:
        print(f"SQLite error (get_completed_ensembles): {e}")
    finally:
        if conn:
            conn.close()

    return all_retrieved_data

# store list of completed model runs in tc_candidates and tc_disturbances databases
def update_ensemble_status(model_file_paths_by_model_name_and_timestamp, complete_file_paths_by_model_name_and_timestamp):
    conn = None
    try:
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ens_status (
                id INTEGER PRIMARY KEY,
                ensemble_name TEXT,
                init_date TEXT,
                num_files_processed INTEGER,
                expected_num_files_processed INTEGER,
                completed INTEGER,
                UNIQUE(ensemble_name, init_date)
            )
        ''')
        # track ensemble status of files processed
        for ensemble_name, file_paths_by_timestamp in model_file_paths_by_model_name_and_timestamp.items():
            for model_init_time, file_paths in file_paths_by_timestamp.items():
                dt = datetime.strptime(model_init_time, '%Y%m%d%H')
                ensemble_timestamp = dt.isoformat()

                cursor.execute('SELECT num_files_processed FROM ens_status WHERE ensemble_name = ? AND init_date = ?', (ensemble_name, ensemble_timestamp))
                result = cursor.fetchone()
                num_processed = len(file_paths)
                if result is None or result != num_processed:
                    expected_num = -1
                    if ensemble_name in complete_file_paths_by_model_name_and_timestamp and \
                        model_init_time in complete_file_paths_by_model_name_and_timestamp[ensemble_name]:
                        expected_num = len(complete_file_paths_by_model_name_and_timestamp[ensemble_name][model_init_time])

                    if expected_num == num_processed:
                        completed = 1
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        completed = 0
                    cursor.execute('INSERT OR REPLACE INTO ens_status (ensemble_name, init_date, num_files_processed, expected_num_files_processed, completed) VALUES (?, ?, ?, ?, ?)', (ensemble_name, ensemble_timestamp, num_processed, expected_num, completed))
                    conn.commit()
                    if completed == 1:
                        print(f"Completed {formatted_date} - {ensemble_name}")

    except sqlite3.Error as e:
        print(f"SQLite error (update_ensemble_status): {e}")
    finally:
        if conn:
            conn.close()

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

# get list of completed TC candidates (tracks)
def get_completed_tc():
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                init_date TEXT,
                completed_date TEXT,
                UNIQUE(model_name, init_date)
            )
        ''')

        # Query all rows from the 'disturbances' table and order by 'model_timestamp'
        cursor.execute('SELECT model_name, init_date, completed_date FROM completed ORDER BY init_date')
        results = cursor.fetchall()
        if results:
            # Process data for each row
            for row in results:
                model_name, model_timestamp, completed_date = row
                retrieved_data = {
                    "model_name": model_name,
                    "model_timestamp": model_timestamp,
                    "completed_timestamp": completed_date
                }
                all_retrieved_data.append(retrieved_data)

    except sqlite3.Error as e:
        print(f"SQLite error (get_completed_tc): {e}")
    finally:
        if conn:
            conn.close()

    return all_retrieved_data

# store list of completed model runs in tc_candidates and tc_disturbances databases
def update_tc_completed(model_name, model_init_time):
    conn = None
    try:
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                init_date TEXT,
                completed_date TEXT,
                UNIQUE(model_name, init_date)
            )
        ''')
        # track which components are already finished
        cursor.execute('SELECT completed_date FROM completed WHERE model_name = ? AND init_date = ?', (model_name, model_init_time))
        result = cursor.fetchone()
        if not result:
            dt = datetime_utcnow()
            completed_date = dt.isoformat()
            cursor.execute('INSERT OR REPLACE INTO completed (model_name, init_date, completed_date) VALUES (?, ?, ?)', (model_name, model_init_time, completed_date))
            conn.commit()

    except sqlite3.Error as e:
        print(f"SQLite error (update_tc_completed): {e}")
    finally:
        if conn:
            conn.close()

# store (simplified) TC track candidate as disturbances and as a track (two different databases)
# component_num (from 0) is an id for each component (tc candidate track) of the model for that model's timestep
def add_tc_candidate(model_name, model_init_time, component_num, max_10m_wind_speed, tc_disturbance_candidates):
    has_error = False

    # store tc disturbance candidates in database similar to disturbances (this is a database for accessing by model and timestep)
    conn = None
    try:
        # store tc disturbance candidates in database by model name, component (storm) id, valid time (access by model/id/valid_time)
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tc_candidates (
                id INTEGER PRIMARY KEY,
                ensemble_name TEXT,
                model_name TEXT,
                data JSON,
                init_date TEXT,
                component_id INTEGER,
                start_time_step INTEGER,
                start_valid_date TEXT,
                start_basin TEXT,
                start_lat REAL,
                start_lon REAL,
                ws_max_10m REAL,
                UNIQUE(model_name, init_date, component_id)
            )
        ''')

        start_time_step = int(tc_disturbance_candidates[0][0])
        start_valid_time = tc_disturbance_candidates[0][1]
        start_disturbance = tc_disturbance_candidates[0][2]
        start_basin = start_disturbance['basin']
        start_lat = float(start_disturbance['lat'])
        start_lon = float(start_disturbance['lon'])
        try:
            json_data = json.dumps(tc_disturbance_candidates)
        except Exception as e:
            # Error serializing?
            print(f"Error: {e}")
            traceback.print_exc(limit=None, file=None, chain=True)
            print(tc_disturbance_candidates)
            exit(1)

        ensemble_name = model_name_to_ensemble_name[model_name]

        cursor.execute('SELECT component_id FROM tc_candidates WHERE model_name = ? AND init_date = ? AND component_id = ?', (model_name, model_init_time, component_num))
        result = cursor.fetchone()
        if not result:
            cursor.execute('INSERT OR REPLACE INTO tc_candidates (ensemble_name, model_name, data, init_date, component_id, start_time_step, start_valid_date, start_basin, start_lat, start_lon, ws_max_10m) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (ensemble_name, model_name, json_data, model_init_time, component_num, start_time_step, start_valid_time, start_basin, start_lat, start_lon, max_10m_wind_speed))
            conn.commit()

    except sqlite3.Error as e:
        has_error = True
        print(f"SQLite error (add_tc_candidate: tc_candidates): {e}")
    finally:
        if conn:
            conn.close()

    return has_error

#######################################
### CALCULATE TCs FROM DISTURBANCES ###
#######################################

# Only not already computed calculate disturbances from complete model runs

## AUTO UPDATE CODE ##

# polling interval in minutes to calculate (look for disturbances that have completed runs but have not yet computed tc candidates)
polling_interval = 5

last_model_init_times = []

if __name__ == "__main__":
    while True:
        #print("\nChecking for new disturbance data from completed model runs")
        calc_tc_candidates()
        break

        time.sleep(60 * polling_interval)
