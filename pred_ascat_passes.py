# ASCAT Pass Predictions for Tropical Cyclones (included genesis)
# EXPERIMENTAL!
# Demo: http://jrpdata.free.nf/ascatpred
# Hover/click/touch over global map to get storm's ASCAT predictions
# Covers METOP-B, METOP-C, HY-2B, HY-2C, OCEANSAT-3

# The credit for the main idea or method for swath prediction is owed to STAR/NESDIS/NOAA's ASCAT pass predictions page: https://manati.star.nesdis.noaa.gov/datasets/SatellitePrediction.php
# Uses the same numbers for METOP-B and METOP-C for swath widths and nadir blind range
# HY-2B, HY-2C, OCEANSAT-3 all have the same swath widths (1800 km in diameter, we use half this value as a radius)
# There are some limitations such as track accuracy (especially further out) and the use of the RMW to calculate a successful "pass", which may be inaccurate or not the desired measured for depending on the use case. Suggested alterations include, multiplying the interpolated RMW by a constant factor to get a larger wind field (more possible misses), use a different parameter, or generate a radius of interest in a heuristic using existing parameters and/or climatology?

# Updated TLEs are downloaded each run (from EUMETSAT directly for METOP-B,C, rest from celestrak.org)

# auto_update_cyclones_tcgen.py uses data from GEFS genesis tracker (NCEP) to generate tracks into a db
# Some tracks may be missing from corresponind GEFS genesis tracks, especially in cases of merges, as we use a 24 hour cutoff and (in a very ad-hoc way) handle merging tropical cyclones

# This code borrows code from tcviewer.py to read in the db for the GFSO model and compute the ASCAT predictions based on an interpolated track at 1 minute intervals
# Spatial-temporal Intersections are done using the (interpolations of the) Radius of Max Wind (RMW) parameter from the GEFS tracker output
# For choosing the sensing time, the time is essentially truncated to the minute covers the most of the storm's RMW (the point coordinate of the (1-minute duration, quadrilateral) swaths' corner closest to the interpolated MSLP center).  A more complicated method, rounding to the nearest minute, is possible but likely not needed, so not implemented.

import traceback
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from matplotlib.patches import Circle
from rtree import index
from shapely.geometry import box, LineString, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree
from skyfield.api import load, EarthSatellite
from tqdm import tqdm
import antimeridian
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ftplib
import geopandas as gpd
import json
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import requests
import sqlite3
import warnings

# Predict ASCAT passes (METOP-B, METOP-C, HY-2B, HY-2C, OCEANSAT-3) for TCs (including genesis)
# Uses RMW from forecast track from tc genesis models (NCEP TC genesis tracker -> own database)

# start_dt for passes is the beginning of the valid day (00Z) from the latest model

# TODO: more statistics
model_cycle_str = None

### CONFIG ##
# days offset for satellite passes, in case we want an earlier day or later day
offset = 0
# number of days of passes to consider (from UTC 00Z of current UTC datetime)
# 1 = 24 hours (from 00Z), 2 = 48 hours, etc
total_days = 5

# specify model time for prediction (comment out to use latest)
#model_cycle_str = '2024-09-30 18:00:00'
# which ensemble to work with (COULD BE ANY OF GEFS-TCGEN, EPS-TCGEN, GEPS-TCGEN, FNMOC-TCGEN)
genesis_previous_selected = 'GEFS-TCGEN'
# model to use for predicting successful passes
model_member_name_for_prediction = 'GFSO'

# save pass images
save_swath_images = False
save_ascat_swath_images_folder = './ascat_swaths'
save_model_passes = True
save_model_passes_folder = './model_ascat_passes'

# disable swath chart plotting
do_plot_swath_chart = False
# show swaths chart (ascat swaths)
show_swath_chart = False
# show model pass predictions (each pass image)
show_pass_predictions = False

if model_cycle_str is not None:
    model_cycle = datetime.strptime(model_cycle_str, "%Y-%m-%d %H:%M:%S")
else:
    model_cycle = None

# Define swath information for each satellite (m) (nadir blind range on each side and swath width from nadir (radius) in km)
swath_info = {
    "METOP-B": {"blind_range": 336000, "swath_width": 550000},
    "METOP-C": {"blind_range": 336000, "swath_width": 550000},
    "HAIYANG-2B": {"blind_range": 0, "swath_width": 900000},
    "HAIYANG-2C": {"blind_range": 0, "swath_width": 900000},
    "OCEANSAT-3": {"blind_range": 0, "swath_width": 900000},
}

region = 'GLOBAL'
#region = 'TROPICS'

# used to update images (force reload on stale) in browser (to be replaced)
candidate_init_time_dt = datetime.now()

# Define the region you want to plot (e.g., Europe)
if region == 'TROPICS':
    lon_min, lon_max = -140, 5
    lat_min, lat_max = 0, 50
elif region == 'GLOBAL':
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -90, 90


urls = {
    'METOP-B': 'https://service.eumetsat.int/tle/data_out/latest_m01_tle.txt',
    'METOP-C': 'https://service.eumetsat.int/tle/data_out/latest_m03_tle.txt',
    'HAIYANG-2B': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=43655&FORMAT=2LE',
    'HAIYANG-2C': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=46469&FORMAT=2LE',
    'OCEANSAT-3': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=54361&FORMAT=2LE'
}

# tcgen data (processed from NCEP/NOAA, ECMWF) by model and storm
tc_candidates_tcgen_db_file_path = './tc_candidates_tcgen.db'

# own tracker for genesis data - this accesses data by model and storm (internal component id)
# (won't use as it is unofficial)
tc_candidates_db_file_path = 'tc_candidates.db'

os.makedirs(save_ascat_swath_images_folder, exist_ok=True)
os.makedirs(save_model_passes_folder, exist_ok=True)


# RUN ONCE PER DAY

def fetch_tle_data():

    tle_data_dict = {}

    for satellite, tle_url in urls.items():
        response = requests.get(tle_url, verify=False)
        tle_lines = response.text.strip().split('\n')
        if len(tle_lines) == 3:
            tle_data_dict[satellite] = (tle_lines[0], tle_lines[1], tle_lines[2])
        elif len(tle_lines) == 2:
            tle_data_dict[satellite] = (tle_lines[0], tle_lines[1])

    return tle_data_dict

# Test fetching TLE data
all_tle_data = fetch_tle_data()

if len(all_tle_data.keys()) != len(urls.keys()):
    raise Exception("Could not get all TLEs... Aborting.")

#print(all_tle_data)



def calculate_nadir_paths(tle_data, start_dt, end_dt, time_step_seconds=60):
    ts = load.timescale()

    nadir_paths = {}

    for satellite_name, tle_lines in tle_data.items():
        satellite = EarthSatellite(tle_lines[0], tle_lines[1], satellite_name, ts)
        nadir_path = []

        current_time = start_dt
        while current_time <= end_dt:
            t = ts.utc(current_time)
            geocentric = satellite.at(t)
            subpoint = geocentric.subpoint()
            nadir_path.append((current_time, subpoint.latitude.degrees, subpoint.longitude.degrees))
            current_time += timedelta(seconds=time_step_seconds)

        nadir_paths[satellite_name] = nadir_path

    return nadir_paths


def calculate_pass_times(nadir_paths):
    pass_times = {}

    for satellite_name, path in nadir_paths.items():
        pass_times[satellite_name] = []
        for timestamp, lat, lon in path:
            if abs(lat - 45) < 0.1 or abs(lat) < 0.1 or abs(lat + 45) < 0.1:
                pass_times[satellite_name].append((timestamp, lat, lon))

    return pass_times



def in_bounds(lat, lon):
    if lat >= lat_min and lat <= lat_max and lon >= lon_min and lon <= lon_max:
        return True
    else:
        return False

def calculate_swaths_and_plot(nadir_paths, start_dt, label_interval=30, opacity=0.3):
    global do_plot_swath_chart
    if do_plot_swath_chart:
        fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set the extent of the map
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, linestyle='-', edgecolor='black')
        ax.gridlines(draw_labels=True, xlocs=np.arange(lon_min, lon_max + 5, 5), ylocs=np.arange(lat_min, lat_max + 5, 5))
    else:
        # placeholder
        ax = None

    colors = ['blue', 'purple', 'red', 'orange', 'yellow', 'green', 'pink', 'turquoise', 'lavender', 'brown', 'black', 'teal', 'magenta', 'lime', 'gray', 'amber']
    added_labels = set()
    gdfs = []
    for i, (satellite_name, path) in enumerate(nadir_paths.items()):
        lats = [p[1] for p in path]
        lons = [p[2] for p in path]
        times = [p[0] for p in path]

        # Split the path into ascending and descending segments
        segments = split_ascending_descending(lats, lons, times)
        last_label_time = datetime.min.replace(tzinfo=pytz.utc)

        for j, (segment_type, segment_lons, segment_lats, segment_times) in enumerate(segments):
            color_index = (i * 2) + (0 if segment_type == 'ascending' else 1)
            color = colors[color_index]
            label = f"{satellite_name} ({segment_type})"

            if do_plot_swath_chart:
                # Create a shapely LineString from the segment path
                line_string = LineString(zip(segment_lons, segment_lats))
                try:
                    fixed_line = antimeridian.fix_line_string(line_string)
                except Exception:
                    print("line_string, type:", type(line_string))
                    print("len lons,lats:", len(segment_lons), len(segment_lats))
                    print(line_string.coords[:])
                    print(traceback.format_exc())


                # Plot the fixed line(s)
                plot_fixed_line(ax, fixed_line, color)

            # Shade the swath halves
            #shade_swath(color_index, ax, segment_lons, segment_lats, satellite_name, swath_info[satellite_name], opacity)
            gdf = shade_swath(color_index, ax, segment_lons, segment_lats, satellite_name, swath_info[satellite_name], opacity, j, segment_times, start_dt, segment_type)
            gdfs.append(gdf)

            if do_plot_swath_chart:
                # Add a label for the satellite's ascending/descending pass
                if label not in added_labels:
                    added_labels.add(label)
                    ax.plot([], [], label=label, color=color)

                # Annotate pass times
                annotate_pass_times(ax, segment_times, segment_lats, segment_lons, start_dt, color, last_label_time, label_interval)

    if do_plot_swath_chart:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.title(f'Satellite Nadir Paths with ASCAT swaths (Ascending and Descending)\nRelative to {start_dt}')
        timestamp = start_dt.strftime("%Y%m%d_%H_%M_%S")

        if save_swath_images:
            file_name = os.path.join(save_ascat_swath_images_folder, f'ascat-{region}-{timestamp}.png')
            plt.savefig(file_name, bbox_inches='tight')

        if show_swath_chart:
            plt.show()
        else:
            plt.close(fig)

    return gdfs

def plot_fixed_line(ax, fixed_line, color):
    """Plots the fixed line(s) for the nadir path."""
    if fixed_line.geom_type == 'MultiLineString':
        for line in fixed_line.geoms:
            lons, lats = line.xy
            ax.plot(lons, lats, color=color, transform=ccrs.PlateCarree(), linestyle=':')
    else:
        lons, lats = fixed_line.xy
        ax.plot(lons, lats, color=color, transform=ccrs.PlateCarree(), linestyle=':')

def is_counterclockwise(points):
    """Check if the points are in counterclockwise order using the shoelace formula."""
    total = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # Next point, wrapping around to the first
        total += (x2 - x1) * (y2 + y1)  # Calculate the area contribution
    return total < 0  # Return True if the area is negative (counterclockwise)

#def shade_swath(color_index, ax, segment_lons, segment_lats, satellite_name, swath_details, opacity):
def shade_swath(color_index, ax, segment_lons, segment_lats, satellite_name, swath_details, opacity, j, segment_times, start_dt, segment_type):
    """Shades the swath halves around the nadir path."""
    global do_plot_swath_chart
    blind_range = swath_details["blind_range"]
    swath_width = swath_details["swath_width"]

    has_blind_range = True
    if blind_range == 0:
        # hy/oceansat circular swaths (no blind range)
        has_blind_range = False

    colors = ['blue', 'purple', 'red', 'orange', 'yellow', 'green', 'pink', 'turquoise', 'lavender', 'brown', 'black', 'teal', 'magenta', 'lime', 'gray', 'amber']
    color = colors[color_index % len(colors)]

    geod = Geodesic.WGS84

    # for the gdf
    rows = []
    
    # Loop through each segment starting from the second point
    for i in range(1, len(segment_lons)):
        # Current point
        lon1, lat1 = segment_lons[i], segment_lats[i]
        # Previous point
        lon0, lat0 = segment_lons[i - 1], segment_lats[i - 1]

        # Calculate the forward and backward distances to determine swath edges
        # Calculate the direction vector
        direction = geod.Inverse(lat0, lon0, lat1, lon1)
        azimuth = direction['azi1']  # Get the azimuth from the previous to current point

        std_unroll_lats = (Geodesic.STANDARD | Geodesic.LONG_UNROLL)

        if has_blind_range:
            # Calculate left and right swaths
            left_swath_1 = geod.Direct(lat0, lon0, azimuth - 90, blind_range, outmask = std_unroll_lats)
            left_swath_2 = geod.Direct(lat0, lon0, azimuth - 90, blind_range + swath_width, outmask = std_unroll_lats)
            left_swath_3 = geod.Direct(lat1, lon1, azimuth - 90, blind_range, outmask = std_unroll_lats)
            left_swath_4 = geod.Direct(lat1, lon1, azimuth - 90, blind_range + swath_width, outmask = std_unroll_lats)

            right_swath_1 = geod.Direct(lat0, lon0, azimuth + 90, blind_range, outmask = std_unroll_lats)
            right_swath_2 = geod.Direct(lat0, lon0, azimuth + 90, blind_range + swath_width, outmask = std_unroll_lats)
            right_swath_3 = geod.Direct(lat1, lon1, azimuth + 90, blind_range, outmask = std_unroll_lats)
            right_swath_4 = geod.Direct(lat1, lon1, azimuth + 90, blind_range + swath_width, outmask = std_unroll_lats)
        else:
            # (left_swath == both swaths to simplify var names)
            left_swath_1 = geod.Direct(lat0, lon0, azimuth + 90, swath_width, outmask = std_unroll_lats)
            left_swath_2 = geod.Direct(lat0, lon0, azimuth - 90, swath_width, outmask = std_unroll_lats)
            left_swath_3 = geod.Direct(lat1, lon1, azimuth + 90, swath_width, outmask = std_unroll_lats)
            left_swath_4 = geod.Direct(lat1, lon1, azimuth - 90, swath_width, outmask = std_unroll_lats)

        if has_blind_range:
            left_points = [
                (left_swath_1['lon2'], left_swath_1['lat2']),
                (left_swath_2['lon2'], left_swath_2['lat2']),
                (left_swath_4['lon2'], left_swath_4['lat2']),
                (left_swath_3['lon2'], left_swath_3['lat2']),
                (left_swath_1['lon2'], left_swath_1['lat2']),
            ]
            right_points = [
                (right_swath_1['lon2'], right_swath_1['lat2']),
                (right_swath_2['lon2'], right_swath_2['lat2']),
                (right_swath_4['lon2'], right_swath_4['lat2']),
                (right_swath_3['lon2'], right_swath_3['lat2']),
                (right_swath_1['lon2'], right_swath_1['lat2']),
            ]
        else:
            left_points = [
                (left_swath_1['lon2'], left_swath_1['lat2']),
                (left_swath_2['lon2'], left_swath_2['lat2']),
                (left_swath_4['lon2'], left_swath_4['lat2']),
                (left_swath_3['lon2'], left_swath_3['lat2']),
                (left_swath_1['lon2'], left_swath_1['lat2']),
            ]

        # booleans are just for commentary to understand flow
        #did_reverse_left = False
        if not is_counterclockwise(left_points):
            #did_reverse_left = True
            left_points.reverse()

        if has_blind_range:
            #did_reverse_right = False
            if not is_counterclockwise(right_points):
                #did_reverse_right = True
                right_points.reverse()

        left_polygon = Polygon(left_points)
        if has_blind_range:
            right_polygon = Polygon(right_points)

        # Fix the polygon for antimeridian crossing

        # Capture warnings during polygon fixing
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Filter warnings to catch only those related to winding
            warnings.simplefilter("always")  # Enable all warnings

            try:
                fixed_left_polygon = antimeridian.fix_polygon(left_polygon)
            except:
                print("left_polygon, type:", type(left_polygon))
                print("left_polygon.exterior.coords:")
                print(list(left_polygon.exterior.coords))
                print(traceback.format_exc())


            # Log the warnings
            for warning in caught_warnings:
                if "wound clockwise" in str(warning.message):
                    # this is buggy around the anti-meridian ..
                    #print(f"L Warning {did_reverse_left}: {warning.message} for points: {left_points}")
                    #print(lon0, lat1, lon1, lat1)
                    pass



        # Capture warnings during polygon fixing
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Filter warnings to catch only those related to winding
            warnings.simplefilter("always")  # Enable all warnings
            try:
                if has_blind_range:
                    fixed_right_polygon = antimeridian.fix_polygon(right_polygon)
            except:
                print("right_polygon, type:", type(right_polygon))
                print("right_polygon.exterior.coords:")
                print(list(left_polygon.exterior.coords))
                print(traceback.format_exc())


            # Log the warnings
            for warning in caught_warnings:
                if "wound clockwise" in str(warning.message):
                    # this is buggy around the anti-meridian ..
                    #print(f"R Warning {did_reverse_right}: {warning.message} for points: {right_points}")
                    #print(lon0, lat1, lon1, lat1)
                    pass


        # Plot the swaths
        if isinstance(fixed_left_polygon, MultiPolygon):
            for poly_num, poly in enumerate(fixed_left_polygon.geoms):
                if do_plot_swath_chart:
                    ax.add_patch(plt.Polygon(list(poly.exterior.coords), color=color, alpha=opacity))
                rows.append({
                    'interp_id': int((segment_times[i] - start_dt).total_seconds()),
                    'valid_time': segment_times[i],
                    'satellite_name': satellite_name,
                    'pass_number': j,
                    'ascending': 1 if segment_type == 'ascending' else 0,
                    'swath_side': 0,
                    'swath_side_poly_num': poly_num,
                    'geometry': poly
                })
        else:
            if do_plot_swath_chart:
                ax.add_patch(plt.Polygon(list(fixed_left_polygon.exterior.coords), color=color, alpha=opacity))
            rows.append({
                'interp_id': int((segment_times[i] - start_dt).total_seconds()),
                'valid_time': segment_times[i],
                'satellite_name': satellite_name,
                'pass_number': j,
                'ascending': 1 if segment_type == 'ascending' else 0,
                'swath_side': 0,
                'swath_side_poly_num': 0,
                'geometry': fixed_left_polygon
            })

        if has_blind_range:
            if isinstance(fixed_right_polygon, MultiPolygon):
                for poly_num, poly in enumerate(fixed_right_polygon.geoms):
                    if do_plot_swath_chart:
                        ax.add_patch(plt.Polygon(list(poly.exterior.coords), color=color, alpha=opacity))
                    rows.append({
                        'interp_id': int((segment_times[i] - start_dt).total_seconds()),
                        'valid_time': segment_times[i],
                        'satellite_name': satellite_name,
                        'pass_number': j,
                        'ascending': 1 if segment_type == 'ascending' else 0,
                        'swath_side': 1,
                        'swath_side_poly_num': poly_num,
                        'geometry': poly
                    })
            else:
                if do_plot_swath_chart:
                    ax.add_patch(plt.Polygon(list(fixed_right_polygon.exterior.coords), color=color, alpha=opacity))
                rows.append({
                    'interp_id': int((segment_times[i] - start_dt).total_seconds()),
                    'valid_time': segment_times[i],
                    'satellite_name': satellite_name,
                    'pass_number': j,
                    'ascending': 1 if segment_type == 'ascending' else 0,
                    'swath_side': 1,
                    'swath_side_poly_num': 0,
                    'geometry': fixed_right_polygon
                })

    # Create the GeoDataFrame from the collected rows
    gdf = gpd.GeoDataFrame(rows, columns=[
        'interp_id', 'valid_time', 'satellite_name', 'pass_number', 'ascending', 'swath_side', 'swath_side_poly_num', 'geometry'
    ])

    return gdf


def annotate_pass_times(ax, segment_times, segment_lats, segment_lons, start_dt, color, last_label_time, label_interval):
    """Annotates the pass times at regular intervals."""
    time_delta = timedelta(minutes=label_interval)
    for time, lat, lon in zip(segment_times, segment_lats, segment_lons):
        if time >= last_label_time + time_delta:
            time_offset = time - start_dt
            hours, remainder = divmod(time_offset.total_seconds(), 3600)
            minutes = remainder // 60
            time_str = f"{int(hours):+}:{int(minutes):02d}"

            last_label_time = time
            if in_bounds(lat, lon):
                text = ax.text(lon, lat, time_str, fontsize=8, color=color, transform=ccrs.PlateCarree())

                text.set_path_effects([
                    path_effects.Stroke(linewidth=1, foreground='black'),
                    path_effects.Normal()
                ])

def split_ascending_descending(lats, lons, times):
    """Splits the nadir path into ascending and descending segments."""
    segments = []
    current_segment = {"type": "ascending" if lats[1] > lats[0] else "descending", "lons": [], "lats": [], "times": []}

    for i in range(len(lats) - 1):
        current_segment["lons"].append(lons[i])
        current_segment["lats"].append(lats[i])
        current_segment["times"].append(times[i])

        if (lats[i+1] > lats[i] and current_segment["type"] == "descending") or \
                (lats[i+1] < lats[i] and current_segment["type"] == "ascending"):
            segments.append((current_segment["type"], current_segment["lons"], current_segment["lats"], current_segment["times"]))
            current_segment = {
                "type": "ascending" if lats[i+1] > lats[i] else "descending",
                "lons": [lons[i]], "lats": [lats[i]], "times": [times[i]]
            }

    current_segment["lons"].append(lons[-1])
    current_segment["lats"].append(lats[-1])
    current_segment["times"].append(times[-1])
    segments.append((current_segment["type"], current_segment["lons"], current_segment["lats"], current_segment["times"]))

    return segments



### code from tcviewer (modified for standalone pass calculation)

hidden_tc_candidates = set()
plotted_tc_candidates = []
# r-tree index
rtree_p = index.Property()
rtree_idx = index.Index(properties=rtree_p)
# Mapping from rtree point index to (internal_id, tc_index, tc_candidate_point_index)
rtree_tuple_point_id = 0
rtree_tuple_index_mapping = {}


PROCESS_TCGEN_WIND_RADII = False

global_det_members = ['GFS', 'CMC', 'ECM', 'NAV']
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

model_name_to_ensemble_name = {}
for list_name, lst in zip(['GEFS-TCGEN', 'GEPS-TCGEN', 'EPS-TCGEN', 'FNMOC-TCGEN'], [gefs_members, geps_members, eps_members, fnmoc_members]):
    for model_name in lst:
        model_name_to_ensemble_name[model_name] = list_name

# add these to group them under the same organization (ensemble) even though they are not part of the ensemble
model_name_to_ensemble_name['GFS'] = 'GEFS-TCGEN'
model_name_to_ensemble_name['ECM'] = 'EPS-TCGEN'
model_name_to_ensemble_name['NAV'] = 'FNMOC-TCGEN'

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

cls_time_step_opacity = [
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

def get_genesis_cycle(model_cycle):
    if model_cycle:
        display_genesis_data(model_cycle)

def latest_genesis_cycle():
    model_cycles = get_tc_model_init_times_relative_to(datetime.now(), genesis_previous_selected)
    if model_cycles['next'] is None:
        model_cycle = model_cycles['at']
    else:
        model_cycle = model_cycles['next']

    if model_cycle:
        # clear map
        display_genesis_data(model_cycle)


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

            # for tcgen case, handle case when no predictions from a member but have a complete ensemble
            if tcgen_ensemble:
                cursor.execute(
                    f'SELECT init_date FROM ens_status WHERE ensemble_name = ? AND completed = 1 AND init_date <= ? ORDER BY init_date DESC LIMIT 1',
                    (ensemble_name, datetime.isoformat(interval_end)))
                results = cursor.fetchall()
                #ens_is_completed = 0
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
    # earliest_recent_ensemble_init_times denotes the earliest member init time among all most recent members' init times (or if it's an empty ensemble last complete run time)
    return model_init_times, earliest_recent_ensemble_init_times, model_completed_times, ensemble_completed_times, completed_ensembles, all_retrieved_data




def get_latest_genesis_data_times(is_ensemble=None, model_cycle=None):
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



def display_genesis_data(model_cycle):
    # vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, 'p'), (113.0, 'o'), (137.0, 'D'), (float('inf'), '+')]
    vmax_kt_threshold = [(34.0, 'v'), (64.0, '^'), (83.0, 's'), (96.0, '<'), (113.0, '>'), (137.0, 'D'),
                         (float('inf'), '*')]
    #vmax_labels = ['\u25BD TD', '\u25B3 TS', '\u25A1 1', '\u25C1 2', '\u25B7 3', '\u25C7 4', '\u2605 5']
    #marker_sizes = {'v': 6, '^': 6, 's': 8, '<': 10, '>': 12, 'D': 12, '*': 14}

    #expected_model_names = []
    if genesis_previous_selected != 'GLOBAL-DET':
        is_ensemble = True
        if genesis_previous_selected != 'ALL-TCGEN':
            is_all_tcgen = False
        else:
            is_all_tcgen = True

        expected_model_names = tcgen_models_by_ensemble[genesis_previous_selected]
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
     completed_ensembles, tc_candidates) = get_tc_candidates_at_or_before_init_time(genesis_previous_selected, model_cycle)
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
            ensemble_completed_times, completed_ensembles in get_latest_genesis_data_times(is_ensemble=is_ensemble, model_cycle=model_cycle):

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
                # this is the edge case where there were 0 predictions (empty file) but the ensemble was completed
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
            tcgen_ensemble = genesis_previous_selected
            ens_init_time_str = earliest_recent_ensemble_init_times[tcgen_ensemble]
            ens_init_time = datetime.fromisoformat(ens_init_time_str)
            most_recent_model_cycle = ens_init_time
            oldest_model_cycle = ens_init_time
    else:
        most_recent_model_cycle = max(most_recent_model_timestamps.values())
        oldest_model_cycle = min(most_recent_model_timestamps.values())

    start_of_day = oldest_model_cycle.replace(hour=0, minute=0, second=0, microsecond=0)
    valid_day = start_of_day.isoformat()
    # model_init_times, tc_candidates = get_tc_candidates_from_valid_time(now.isoformat())
    #model_init_times, tc_candidates = get_tc_candidates_at_or_before_init_time(most_recent_model_cycle)
    model_init_times, earliest_recent_ensemble_init_times, model_completed_times, ensemble_completed_times, completed_ensembles, \
        tc_candidates = get_tc_candidates_at_or_before_init_time(genesis_previous_selected, model_cycle)

    most_recent_timestamp = None

    clear_plotted_list()
    #lon_lat_tc_records = []
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
            if len(hidden_tc_candidates) != 0 and internal_id in hidden_tc_candidates:
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

                rmw = candidate['rmw']
                if rmw > 0:
                    candidate_info['rmw'] = candidate['rmw'] / 1000
                else:
                    candidate_info['rmw'] = None

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
                for i, (start, end) in list(enumerate(time_step_ranges)):
                    hours_after = candidate_info['hours_after_valid_day']
                    if start <= hours_after <= end:
                        if cls_time_step_opacity[i] == 1.0:
                            lat_lon_with_time_step_list.append(candidate_info)
                            have_displayed_points = True
                        elif have_displayed_points and cls_time_step_opacity[i] == 0.6:
                            lat_lon_with_time_step_list.append(candidate_info)
                        else:
                            # opacity == 0.3 case (hide all points beyond legend valid time)
                            break

            if lat_lon_with_time_step_list:
                update_plotted_list(internal_id, lat_lon_with_time_step_list)

            # do in reversed order so most recent items get rendered on top
            for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
                #opacity = 1.0
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

                """for vmaxmarker in lons.keys():
                    scatter = cls.ax.scatter(lons[vmaxmarker], lats[vmaxmarker], marker=vmaxmarker,
                                              facecolors='none', edgecolors=cls.time_step_marker_colors[i],
                                              s=marker_sizes[vmaxmarker] ** 2, alpha=opacity, antialiased=False)
                    if internal_id not in cls.scatter_objects:
                        cls.scatter_objects[internal_id] = []
                    cls.scatter_objects[internal_id].append(scatter)"""

            '''# do in reversed order so most recent items get rendered on top
            for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
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

                            """plt.plot([point['prev_lon'], point['lon']], [point['prev_lat'], point['lat']],
                                     color=color, linewidth=strokewidth, marker='', markersize = 0, alpha=opacity)"""


                # Create a LineCollection
                lc = LineCollection(line_segments, color=line_color, linewidth=strokewidth, alpha=opacity)
                # Add the LineCollection to the axes
                line_collection = cls.ax.add_collection(lc)
                if internal_id not in cls.line_collection_objects:
                    cls.line_collection_objects[internal_id] = []
                cls.line_collection_objects[internal_id].append(line_collection)'''

            # add the visible points for the track to the m-tree
            if lon_lat_tuples and len(lon_lat_tuples) > 1:
                line_string = LineString(lon_lat_tuples)
                cut_lines = []
                try:
                    cut_lines = cut_line_string_at_antimeridian(line_string)
                except:
                    #print(lon_lat_tuples)
                    #print(line_string)
                    #print(line_string.coords)
                    pass
                for cut_line in cut_lines:
                    record = dict()
                    record['geometry'] = cut_line
                    record['value'] = internal_id
                    #lon_lat_tc_records.append(record)

            # TODO: handle anti-meridean for WindField geometries
            #  this doesn't seem a too important edge case unless it is breaking
            # get the visible time step list for lat_lon_with_time_step_list
            llwtsl_indices.sort()
            #visible_llwtsl = [lat_lon_with_time_step_list[llwtsl_idx] for llwtsl_idx in llwtsl_indices]
            """if visible_llwtsl:
                # TODO MODIFY (ONLY FOR TESTING)
                path_dicts, gpd_dicts = WindField.get_wind_radii_paths_and_gpds_for_steps(
                    lat_lon_with_time_step_list = visible_llwtsl, wind_radii_selected_list = [34])

                # TODO: probabilistic wind radii graphic with boundary path (path patch)
                wind_field_gpd_dicts.append(gpd_dicts)"""

    """App.lon_lat_tc_records = lon_lat_tc_records"""
    #str_tree = STRtree([record["geometry"] for record in lon_lat_tc_records])

    """wind_field_records = {}
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
                App.wind_field_records[wind_speed] = df
                App.wind_field_strtrees[wind_speed] = STRtree(df["geometry"].values)
            else:
                App.wind_field_records[wind_speed] = None
                App.wind_field_strtrees[wind_speed] = None"""

    labels_positive = [f' D+{str(i): >2} ' for i in
                       range(len(time_step_marker_colors) - 1)]  # Labels corresponding to colors
    labels = [' D-   ']
    labels.extend(labels_positive)

    #time_step_legend_objects = []

    for i, (color, label) in enumerate(zip(reversed(time_step_marker_colors), reversed(labels))):
        #x_pos, y_pos = 100, 150 + i * 20

        time_step_opacity = list(reversed(cls_time_step_opacity))[i]
        """if time_step_opacity == 1.0:
            edgecolor = "#FFFFFF"
        elif time_step_opacity == 0.6:
            edgecolor = "#FF77B0"
        else:
            edgecolor = "#A63579"""
        """legend_object = cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels',
                                         color=list(reversed(cls.time_step_legend_fg_colors))[i],
                                         fontsize=8, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                                       edgecolor=edgecolor,
                                                                                       facecolor=color, alpha=1.0))
        time_step_legend_objects.append(legend_object)"""

    # Draw the second legend items inline using display coordinates
    """for i, label in enumerate(reversed(vmax_labels)):
        x_pos, y_pos = 160, 155 + i * 35
        cls.ax.annotate(label, xy=(x_pos, y_pos), xycoords='figure pixels', color='white',
                         fontsize=12, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                                        edgecolor='#FFFFFF', facecolor='#000000',
                                                                        alpha=1.0))"""

    #genesis_model_cycle_time = most_recent_model_cycle
    if not is_ensemble:
        # Update model times label
        pass
    elif genesis_previous_selected != 'GLOBAL-DET':
        ens_dates = []
        for ens_name, init_time in earliest_recent_ensemble_init_times.items():
            ens_dates.append((ens_name, datetime.fromisoformat(init_time).strftime('%d/%HZ')))

        #model_labels_str = ''
        model_labels = []
        for ens_name, ens_min_date in ens_dates:
            model_labels.append(f'{ens_name} [{ens_min_date}]')

        join_model_labels = ", ".join(model_labels)
        """if join_model_labels:
            model_labels_str = f'Models: {join_model_labels}'"""

        pass



def clear_plotted_list():
    global plotted_tc_candidates
    global rtree_p
    global rtree_idx
    global rtree_tuple_point_id
    global rtree_tuple_index_mapping
    #global lon_lat_tc_records
    #global wind_field_records
    #global wind_field_strtrees

    plotted_tc_candidates = []
    rtree_p = index.Property()
    rtree_idx = index.Index(properties=rtree_p)
    rtree_tuple_point_id = 0
    rtree_tuple_index_mapping = {}
    # reset all labels
    """update_tc_status_labels()
    cls.clear_circle_patch()"""
    #lon_lat_tc_records = []
    #wind_field_records = {}
    #wind_field_strtrees = {}


def update_plotted_list(internal_id, tc_candidate):
    global plotted_tc_candidates
    global rtree_p
    global rtree_idx
    global rtree_tuple_point_id
    global rtree_tuple_index_mapping
    #global lon_lat_tc_records
    #global wind_field_records
    #global wind_field_strtrees

    # zero indexed
    tc_index = len(plotted_tc_candidates)
    for point_index, point in enumerate(tc_candidate):  # Iterate over each point in the track
        lat, lon = point['lat'], point['lon']
        # Can't use a tuple (tc_index, point_index) as the index so use a mapped index
        rtree_idx.insert(rtree_tuple_point_id, (lon, lat, lon, lat))
        rtree_tuple_index_mapping[rtree_tuple_point_id] = (internal_id, tc_index, point_index)
        rtree_tuple_point_id += 1

    plotted_tc_candidates.append((internal_id, tc_candidate))


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



# get the model data from database and process it for use (code from tcviewer)

if model_cycle is not None:
    get_genesis_cycle(model_cycle)
else:
    latest_genesis_cycle()



# Old plan for reference (included plan to use entire ensembles/super-ensemble for a probabilistic pass prediction with statistics)
# too slow? to compute so settle for use with only a single model:

# interpolate the model track points to every minute (including roci (km) and vmax)

#
# in either of the two methods, will have to loop over each ascending or descending pass separately first (create a strtree)
# and then an inner loop by storm track makes the most sense
# for each storm track we need a dict that has the relevant ascending/descending pass info from methods below

# two methods... (we use (2))
# (1) (fast) find swaths ONLY containing (interpolated) MSLP centers (swaths must contain the center, this excludes partial passes that don't have the model MSLP center);
#     this gives a "center fit" calculation that is more natural (i.e. 1 would be perfectly centered, 0 would be a center on the edge of the swath)
#     also possible is a roci or rmw (circle) coverage calculation based on the ratio of the swath width at the CPA to the diameter of the circle (roci and/or rmw)

# find all swath polygons that contain each track point (add them to str_tree, need a dict for their index with a nested dict of satellite info: name, time, ascending or descending)
# for each matching swath polygon get the middle of the swath's line (along the track) (geodesic)
#    calculate the point along the swath's line's closest approach (cpa) (swath_middle_cpa_point)
#        (a) calculate the distance and heading between the swath_middle_cpa_point and the track point
#        (b) calculate the swath distance between the swath_middle_cpa_point along the heading intersecting the nearest edge of the swath polygon
#        (c) calculate the swath distance between the swath_middle_cpa_point along the heading+180 in intersecting the nearest edge of the swath polygon
#        ratio of (a) to ((b)+(c)) is the center fit calculation
#        ratio of (a) to roci or rmw is the coverage calculation for (1)
#        circle at swath_middle_cpa_point intersected with circle at storm point


# (2) (slow) (find ~ all partial swaths): create many different boxes (MBRs) (roci/rmw) for each of the interpolated points (MSLP centers)
#     spatial join between two STRtrees then pruning temporally and then finely by intersection of shapes (approximate the storm circle with an n-gon, will need to look up the storm center and radius AGAIN) to get precise matches

#     if there are hundreds of tracks (maybe ~200 in a single GEFS ensemble?), and may each extend to 120 hours (if GEFS-TCGEN) ~ 7200 minutes
#         this means 7200 intersection operations per satellite pass
#            with 7200 intersect operations with the STR tree (per track)  * 200 tracks * # satellite passes (each ascending+descending = 4 every 24-hour period)
#            ~= 5.8 million intersections

#      for all matching 'intersects' (intersecting polygons), we prune those without matching times (valid_intersects), and then prune again on the precise intersections since STRTree will use boxes (we need the precise intersections rather than the MBR)
#         from that list we find the one with the greatest area, and designate that match as the CPA
#         from this we can calculate the center fit calculation again (it may be negative...)
#         for the coverage calculation we need for each valid_intersects:
#                 get the circle corresponding interpolated track circle (as storm_circle from roci or rmw) and append each as storm_circle_intersect_list
#                 get the union of storm_circle and valid_intersect and store append into a list of union_valid_intersects
#         union all the union_valid_intersects (all_unions_valid_intersects)
#         union all the storm_circle from storm_circle_intersect_list (all_union_storm_circle)
#         then get the the ratio of the all_unions_valid_intersects area to the all_union_storm_circle area to get the coverage

# once we get (1) or (2) then we have the data for the passes for each candidate (all members candidates mixed together)
# what we need to do then is bin the candidates into "storms"
# this needs to be done, again, spatiotemporally using a heuristic
#   get the starting lon/lat and the heading (calculated either from the geodesic using the next point or from the 'storm_direction' for GFDL/NCEP members (NO EPS members have this))
#      given we don't have EPS members storm_direction let's calculate it for each storm ourselves from the first pair of points
#      heading is useful to disambiguate intersecting storms, otherwise we will lump storms that are already existing AND intersecting together that are in the same bin
# once we have the candidates binned into storms, we can combine the coverage data together
#   then compute useful statistics (min/median/mean/IQR,max) for each storm by pass
#   then we can aggregate all this into a df for the period (24 hours?):
#      the df will rows for each storm ID, with a row for each pass with timing, ascending, coverage probabilities for center and for partial coverage amount (roci/rmw)

# plotted_tc_candidates[0][1][0]

# Define start and end time for today in UTC
#dt_utc = datetime.now(pytz.utc)
if len(plotted_tc_candidates) == 0:
    raise Exception("No genesis events predicted...")

dt_utc = plotted_tc_candidates[0][1][0]['init_time'].replace(tzinfo=pytz.utc)

dt_utc += timedelta(days=offset)

start_dt = dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
end_dt = dt_utc.replace(hour=23, minute=59, second=59, microsecond=999999)

end_dt += timedelta(days=total_days-1)

# calculate nadir paths, then pass times, and then the calculate and plot the swaths
nadir_paths = calculate_nadir_paths(all_tle_data, start_dt, end_dt)
pass_times = calculate_pass_times(nadir_paths)
swath_gdfs = calculate_swaths_and_plot(nadir_paths, start_dt, label_interval=10, opacity=0.3)





## Code for calculating the spatial-temporal join with the pass swaths and plotting intersecting matches (using storm RMW)

def find_min_max_valid_time(swath_gdfs):
    # Initialize min and max valid_time using the first gdf in the list
    min_valid_time = swath_gdfs[0].iloc[0]['valid_time']
    max_valid_time = swath_gdfs[0].iloc[-1]['valid_time']

    # Loop through each gdf in swath_gdfs
    for gdf in swath_gdfs:
        # Get the first and last valid_time in the current gdf
        current_min_time = gdf.iloc[0]['valid_time']
        current_max_time = gdf.iloc[-1]['valid_time']

        # Update the global min and max times
        if current_min_time < min_valid_time:
            min_valid_time = current_min_time
        if current_max_time > max_valid_time:
            max_valid_time = current_max_time

    return min_valid_time, max_valid_time

def prune_matching_rows_by_time(all_matching_rows, swath_gdfs):
    pruned_rows_by_mbr_and_time = []  # List to hold pruned rows for each swath_gdf
    have_matches = False

    # Loop through each set of matching rows by enumerating
    for idx, matching_rows in enumerate(all_matching_rows):
        pruned_matching_rows = []  # List to hold pruned tuples for current swath_gdf

        swath_gdf = swath_gdfs[idx]  # Get the corresponding swath_gdf for this index

        # Inner loop over the set of tuples in matching_rows
        for row_number, interp_id in matching_rows:
            # Get the interp_id value from the swath_gdf at the row_number
            if row_number < len(swath_gdf):  # Check for valid row number
                swath_row_interp_id = swath_gdf.at[row_number, 'interp_id']

                # Check if interp_id matches
                if swath_row_interp_id == interp_id:
                    pruned_matching_rows.append((row_number, interp_id))
                    have_matches = True
                    #print(f"\nmatch: swath {idx}: {swath_row_interp_id} vs model {interp_id}")
                else:
                    #print(f"\nmiss: swath {swath_row_interp_id} vs model {interp_id}")
                    pass

        # Append the pruned list for this swath_gdf to the main list
        pruned_rows_by_mbr_and_time.append(pruned_matching_rows)

    return have_matches, pruned_rows_by_mbr_and_time

def find_intersecting_candidates(strtree_list, candidate_gdf, strtree_min_times, strtree_max_times):
    all_matching_rows = []  # List to hold results for each STRtree
    # Loop through each STRtree in the list
    for k, str_tree in enumerate(strtree_list):
        matching_row_nums = []  # List to hold matching (row_number, interp_id) tuples for current STRtree
        # Loop through each geometry in candidate_gdf
        strtree_min_time = strtree_min_times[k]
        strtree_max_time = strtree_max_times[k]
        current_min_time = candidate_gdf.iloc[0]['valid_time']
        current_max_time = candidate_gdf.iloc[-1]['valid_time']

        # check to see if it is out of bounds timewise
        if current_min_time > strtree_max_time or current_max_time < strtree_min_time:
            # complete miss for this pass
            all_matching_rows.append(matching_row_nums)
            continue


        for idx, row in candidate_gdf.iterrows():
            valid_time = row['valid_time']
            # Update the global min and max times
            if valid_time < min_valid_time or valid_time > strtree_max_time:
                continue

            query_polygon = row['geometry']  # Get the geometry for querying

            # Query the STRtree with 'intersects' predicate
            result = str_tree.query(query_polygon, predicate='intersects')

            if result.size > 0:  # If there are intersecting geometries
                # Create tuples of (swath_gdf_row_number, interp_id) for each match
                for result_id in result:
                    matching_row_nums.append((result_id, row['interp_id']))

        # Append the matching tuples to the list for this STRtree
        all_matching_rows.append(matching_row_nums)

    return all_matching_rows


def create_strtrees_for_swaths(swath_gdfs):
    strtree_list = []
    strtree_min_times = []
    strtree_max_times = []

    for gdf in swath_gdfs:
        # Create an STRtree for the 'geometry' column in the current GeoDataFrame
        strtree = STRtree(gdf['geometry'])
        strtree_list.append(strtree)

        min_time, max_time = find_min_max_valid_time([gdf])
        strtree_min_times.append(min_time)
        strtree_max_times.append(max_time)

    return strtree_list, strtree_min_times, strtree_max_times


def meters_to_degrees(lat, distance):
    """Convert meters to degrees for latitude and longitude."""
    # Convert latitude distance to degrees
    lat_deg = distance / 111320  # Approx. meters per degree latitude

    # Convert longitude distance to degrees
    lon_deg = distance / (111320 * np.cos(np.radians(lat)))  # Approx. meters per degree longitude
    return lat_deg, lon_deg


def generate_circle_wgs84(lat, lon, radius_km, num_sides=24):
    """Generate an approximated circle as a polygon using WGS84 and geographiclib for accurate geodesic distances."""
    geod = Geodesic.WGS84
    points = []

    # Create a 12-sided polygon approximation of the circle
    for angle in range(0, 360, int(360 / num_sides)):
        point = geod.Direct(lat, lon, angle, radius_km * 1000)  # Convert km to meters
        points.append((point['lon2'], point['lat2']))

    # Close the loop by appending the first point again
    points.append(points[0])

    return Polygon(points)


# not truly precise, but (arbitrarily) approximate (more precise than bounding box)
def precise_intersect_check(pruned_rows_by_gdf, swath_gdfs, candidate_gdf):
    """Check for precise intersection between candidate RMW circle and swath polygons."""
    all_intersecting_candidates = []

    have_matches = False

    for swath_idx, pruned_rows in enumerate(pruned_rows_by_gdf):
        swath_gdf = swath_gdfs[swath_idx]
        intersecting_candidates = []

        for row_number, interp_id in pruned_rows:
            swath_geometry = swath_gdf.geometry.iloc[row_number]

            # Get candidate data for the corresponding interp_id
            candidate_row = candidate_gdf.loc[candidate_gdf['interp_id'] == interp_id]
            if candidate_row.empty:
                continue

            candidate_lat = candidate_row['lat'].values[0]
            candidate_lon = candidate_row['lon'].values[0]
            rmw = candidate_row['rmw'].values[0]  # RMW is in kilometers

            # Create a Point for the candidate's lat/lon
            candidate_point = Point(candidate_lon, candidate_lat)

            # First quick check: is the center of the circle within the swath polygon?
            if swath_geometry.contains(candidate_point):
                # Candidate's center is within the swath, so we count this as an intersection
                intersecting_candidates.append((row_number, interp_id))
                have_matches = True
                continue

            # If not, check the precise intersection with the circle's perimeter
            # increase sides for generate_circle_wgs84 for more precision
            rmw_circle = generate_circle_wgs84(candidate_lat, candidate_lon, rmw)

            # Check if the approximated circle intersects with the swath geometry
            if swath_geometry.intersects(rmw_circle):
                intersecting_candidates.append((row_number, interp_id))
                have_matches = True

        all_intersecting_candidates.append(intersecting_candidates)

    return have_matches, all_intersecting_candidates

def prune_overlapped_sensing(precise_pruned_rows_by_gdf, swath_gdfs, candidate_gdf):
    # find rows that are duplicates
    # for a given swath there are some overlapping (sequential) on the sensing times
    # since we are already processing this per storm track it becomes trivial
    # find sequential sensing times (interp_id +- 60) for each swath
    # keep the swath that has the closest point to the candidate gdf interpolated center
    all_unique_rows = []
    geod = Geodesic.WGS84

    for swath_idx, pruned_rows in enumerate(precise_pruned_rows_by_gdf):
        swath_gdf = swath_gdfs[swath_idx]
        unique_rows = []
        duplicate_tuple_rows = []

        # remember, interp_id is just the second count since start of valid day (60 second increments)
        # find interp_id that are sequential (60 seconds apart)

        # make sure they are in order of interp_id
        pruned_rows.sort(key=lambda x: x[1])

        # keep list of tuples to find duplicate sensing times
        # this is a more general method than needed, given the loop structure

        current_sequence = []

        # remember one row in a swath has a geometry of a single 5 point polygon (a closed quadrilateral)
        for row_number, interp_id in pruned_rows:
            if len(current_sequence) == 0:
                # if the current sequence is empty we will add first and check on the next iteration
                # need to process this list once after loop finishes
                current_sequence.append((row_number, interp_id))
            elif interp_id - current_sequence[-1][1] == 60:
                # found a duplicate
                current_sequence.append((row_number, interp_id))
            else:
                # current row is not a duplicate (any current sequence ended with previous row)
                if len(current_sequence) > 1:
                    # overlapping sensing times
                    duplicate_tuple_rows.append(current_sequence)
                elif len(current_sequence) == 0:
                    unique_rows.append(current_sequence[0])

                current_sequence = [(row_number, interp_id)]

        # handle the last sequence
        if len(current_sequence) > 1:
            duplicate_tuple_rows.append(current_sequence)
        elif len(current_sequence) == 1:
            unique_rows.append(current_sequence[0])

        # process duplicate_tuple_rows to select the rows to keep and add it to unique_rows
        if len(duplicate_tuple_rows) > 0:
            for duplicate_tuple_seq in duplicate_tuple_rows:
                # keep a list of distances from candidate center to the swaths (the closest corner of each swath)
                min_corner_distances = []
                for idx, (row_number, interp_id) in enumerate(duplicate_tuple_seq):
                    swath_geometry = swath_gdf.geometry.iloc[row_number]
                    candidate_row = candidate_gdf.loc[candidate_gdf['interp_id'] == interp_id]
                    candidate_lat = candidate_row['lat'].values[0]
                    candidate_lon = candidate_row['lon'].values[0]

                    # Get the boundary coordinates of the polygon
                    boundary_coords = swath_geometry.exterior.coords

                    # Initialize minimum distance to infinity
                    min_distance = float('inf')

                    # Loop over the boundary coordinates of the swath
                    for coord in boundary_coords:
                        distance = geod.Inverse(candidate_lat, candidate_lon, coord[1], coord[0])['s12']

                        min_distance = min(min_distance, distance)

                    # Append the minimum distance to the list
                    min_corner_distances.append((min_distance, idx))

                # get the minimum
                min_idx = min(min_corner_distances, key=lambda x: x[0])[1]
                unique_rows.append(duplicate_tuple_seq[min_idx])

        # keep it sorted by time (interp_id)
        unique_rows.sort(key=lambda x: x[1])

        all_unique_rows.append(unique_rows)

    return all_unique_rows


def plot_candidate_swath_matches(pruned_rows_by_gdf, swath_gdfs, candidate_gdf, start_lat, start_lon, start_basin):
    # Collect all matches across swaths along with valid_time
    all_matches = []

    for swath_idx, pruned_rows in enumerate(pruned_rows_by_gdf):
        swath_gdf = swath_gdfs[swath_idx]

        for row_number, interp_id in pruned_rows:
            valid_time = swath_gdf['valid_time'].iloc[row_number]
            all_matches.append((valid_time, swath_idx, row_number, interp_id))

    # Sort the matches globally by valid_time
    all_matches_sorted = sorted(all_matches, key=lambda x: x[0])

    # Set up the plotting area
    for valid_time, swath_idx, row_number, interp_id in all_matches_sorted:
        # Get the corresponding swath_gdf
        swath_gdf = swath_gdfs[swath_idx]

        sat_name = swath_gdf['satellite_name'].values[0]
        ascending_str = 'Ascending' if swath_gdf['ascending'].values[0] else 'Descending'
        sat_str = f'{sat_name} {ascending_str}'

        # Get the geometry for the current swath_gdf
        #swath_geometry = swath_gdf.geometry.iloc[row_number]

        # Retrieve the candidate's details from candidate_gdf using interp_id
        candidate_row = candidate_gdf.loc[candidate_gdf['interp_id'] == interp_id]
        if candidate_row.empty:
            continue

        # Extract necessary values for plotting
        candidate_lat = candidate_row['lat'].values[0]
        candidate_lon = candidate_row['lon'].values[0]
        rmw = candidate_row['rmw'].values[0] * 1000  # Convert to meters
        roci = candidate_row['roci'].values[0] * 1000  # Convert to meters

        model_name = candidate_row['model_name'].values[0]
        model_init_time = candidate_row['model_init_time'].values[0]

        # Convert to datetime object
        model_init_time = datetime.fromtimestamp(model_init_time.astype('datetime64[ns]').item() / 1e9, pytz.utc)

        # Format as string
        model_init_time_str = model_init_time.strftime("%Y-%m-%d %H:%M %Z")
        subfolder_time_str = model_init_time.strftime("%Y_%m_%d__%HZ")
        subfolder_name = f'{subfolder_time_str}_{model_name}'

        # Convert rmw and roci from meters to degrees
        rmw_lat_deg, rmw_lon_deg = meters_to_degrees(candidate_lat, rmw)
        roci_lat_deg, roci_lon_deg = meters_to_degrees(candidate_lat, roci)

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the swath geometry
        swath_gdf.boundary.plot(ax=ax, color='green', linewidth=2, label='Swath Geometry')

        # Plot the candidate's position
        #candidate_point = Point(candidate_lon, candidate_lat)
        ax.scatter(candidate_lon, candidate_lat, color='black', marker='o', label='Candidate Position')

        # Add circles for RMW and ROCI
        rmw_circle = Circle((candidate_lon, candidate_lat), radius=rmw_lat_deg, color='red', fill=False, linewidth=2, label='RMW Circle')
        roci_circle = Circle((candidate_lon, candidate_lat), radius=roci_lat_deg, color='blue', fill=False, linewidth=2, label='ROCI Circle')

        # Get the valid_time as string for the current swath_gdf row
        valid_time_str = valid_time.strftime("%Y-%m-%d %H:%M %Z")
        file_pass_valid_time_str = valid_time.strftime("%Y_%m_%d__%H_%M")

        ax.add_patch(rmw_circle)
        ax.add_patch(roci_circle)

        # minimum is 6 degrees (3 deg. on either side)
        max_deg_radius = 3
        if not np.isnan(rmw_lat_deg):
            max_deg_radius = max(max_deg_radius, rmw_lat_deg)
        if not np.isnan(roci_lat_deg):
            max_deg_radius = max(max_deg_radius, roci_lat_deg)

        # Set limits and labels
        ax.set_xlim(candidate_lon - max_deg_radius, candidate_lon + max_deg_radius)  # Adjust limits based on your data
        ax.set_ylim(candidate_lat - max_deg_radius, candidate_lat + max_deg_radius)  # Adjust limits based on your data
        ax.set_title(f'Pass time: {valid_time_str}\n{sat_str}\nModel: {model_name}: {model_init_time_str}\n(Model) Storm MSLP Center lat, lon: {candidate_lat:.1f}, {candidate_lon:.1f}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right')

        # Show the plot
        plt.grid()

        if save_model_passes:
            sat_file_str = sat_str.replace('-', '_').replace(' ', '_')
            file_name = f'pass_{start_basin}_lon_lat_{start_lon:.1f}_{start_lat:.1f}_{file_pass_valid_time_str}_{sat_file_str}.png'
            folder_path = os.path.join(save_model_passes_folder, subfolder_name)
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path, bbox_inches='tight')

            update_passes_log(save_model_passes_folder, model_name, model_init_time_str, start_basin, start_lon, start_lat, candidate_lon, candidate_lat, valid_time_str, subfolder_name, file_name, sat_str)



        if show_pass_predictions:
            plt.show()
        else:
            plt.close(fig)

def calculate_pass_predictions_and_plot():
    geod = Geodesic.WGS84  # Initialize the WGS84 model

    have_reset_once = False
    global candidate_init_time_dt
    global model_member_name_for_prediction
    
    model_candidate_idx_list = []
    for candidate_idx in range(len(plotted_tc_candidates)):
        candidate_model_name = plotted_tc_candidates[candidate_idx][1][0]['model_name']
        if candidate_model_name == model_member_name_for_prediction:
            model_candidate_idx_list.append(candidate_idx)
            candidate_init_time_dt = plotted_tc_candidates[candidate_idx][1][0]['init_time'].replace(tzinfo=pytz.utc)

    #num_candidates = len(model_candidate_idx_list)

    for candidate_idx in tqdm(model_candidate_idx_list):
        candidate = plotted_tc_candidates[candidate_idx]
        internal_id, tc_points = candidate

        interpolated_data = []

        model_name = tc_points[0]['model_name']
        model_init_time = tc_points[0]['init_time'].replace(tzinfo=pytz.utc)
        start_lon = tc_points[0]['lon']
        start_lat = tc_points[0]['lat']
        #start_valid_time = tc_points[0]['valid_time']
        #end_valid_time = tc_points[-1]['valid_time']
        start_basin = tc_points[0]['basin']

        if not have_reset_once:
            model_init_time_str = model_init_time.strftime("%Y-%m-%d %H:%M %Z")
            load_and_reset_passes_log(save_model_passes_folder, model_name, model_init_time_str)
            have_reset_once = True

        for i in range(len(tc_points) - 1):
            # Get start and end points for interpolation
            p1 = tc_points[i]
            p2 = tc_points[i+1]

            # Calculate the time delta in seconds
            time_delta = (p2['valid_time'] - p1['valid_time']).total_seconds()
            step_count = int(time_delta / 60)  # Number of minutes between points

            # Compute the geodesic between the two lat/lon points
            geodesic_info = geod.Inverse(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
            total_distance_m = geodesic_info['s12']  # Distance in meters
            azimuth = geodesic_info['azi1']  # Azimuth (bearing) in degrees
            for step in range(step_count):
                # Interpolating valid_time
                interp_valid_time = p1['valid_time'].replace(tzinfo=pytz.utc) + timedelta(minutes=step)

                # only add interpolation points for the passes (swaths) we are considering
                if interp_valid_time < min_valid_time or interp_valid_time > max_valid_time:
                    continue

                # Calculate the interpolation ratio for this step
                step_ratio = step / step_count

                # Interpolate vmax10m, roci, rmw using linear interpolation
                interp_vmax10m = p1['vmax10m'] + step_ratio * (p2['vmax10m'] - p1['vmax10m'])
                roci1 = p1['roci']
                if roci1 is None:
                    roci1 = 0.0

                roci2 = p2['roci']
                if roci2 is None:
                    roci2 = 0.0

                interp_roci = roci1 + step_ratio * (roci2 - roci1)
                interp_rmw = p1['rmw'] + step_ratio * (p2['rmw'] - p1['rmw'])

                # Interpolate the distance and use Geodesic Direct to find the new lat/lon
                interp_distance_m = step_ratio * total_distance_m
                interp_position = geod.Direct(p1['lat'], p1['lon'], azimuth, interp_distance_m)
                interp_lat = interp_position['lat2']
                interp_lon = interp_position['lon2']

                # Calculate bounding box geometry from interpolated rmw
                rmw_in_meters = interp_rmw * 1000

                # Latitude difference for the bounding box (north and south)
                north_position = geod.Direct(interp_lat, interp_lon, 0, rmw_in_meters)
                south_position = geod.Direct(interp_lat, interp_lon, 180, rmw_in_meters)
                north_lat = north_position['lat2']
                south_lat = south_position['lat2']

                # Longitude difference for the bounding box (east and west)
                lon_degree_distance = rmw_in_meters / (111320 * np.cos(np.radians(interp_lat)))
                east_lon = interp_lon + lon_degree_distance
                west_lon = interp_lon - lon_degree_distance

                # Create bounding box using shapely's box
                bounding_box = box(west_lon, south_lat, east_lon, north_lat)

                # Calculate interp_id based on the time difference from start_dt
                interp_id = int((interp_valid_time - start_dt).total_seconds())

                # Append interpolated data to the list
                interpolated_data.append({
                    'interp_id': interp_id,
                    'internal_id': internal_id,
                    'valid_time': interp_valid_time,
                    'vmax10m': interp_vmax10m,
                    'roci': interp_roci,
                    'rmw': interp_rmw,
                    'lat': interp_lat,
                    'lon': interp_lon,
                    'model_name': model_name,
                    'model_init_time': model_init_time,
                    'geometry': bounding_box
                })

        if len(interpolated_data) == 0:
            continue

        candidate_gdf = gpd.GeoDataFrame(interpolated_data)

        all_matching_rows = find_intersecting_candidates(strtree_list, candidate_gdf, strtree_min_times, strtree_max_times)

        have_matches, pruned_rows_by_gdf = prune_matching_rows_by_time(all_matching_rows, swath_gdfs)

        if not have_matches:
            continue

        have_matches, precise_pruned_rows_by_gdf = precise_intersect_check(pruned_rows_by_gdf, swath_gdfs, candidate_gdf)

        if not have_matches:
            continue

        pruned_overlap_rows_by_gdf = prune_overlapped_sensing(precise_pruned_rows_by_gdf, swath_gdfs, candidate_gdf)

        #print(f"Storm in {start_basin} starting at {start_valid_time}:\nStart lat,lon: {start_lat}, {start_lon}\nEnd valid time: {end_valid_time}")
        plot_candidate_swath_matches(pruned_overlap_rows_by_gdf, swath_gdfs, candidate_gdf, start_lat, start_lon, start_basin)


def update_passes_log(save_model_passes_folder, model_name, model_init_time_str, start_basin, start_lon, start_lat, candidate_lon, candidate_lat, valid_time_str, subfolder_name, file_name, sat_str):
    passes_log_file = os.path.join(save_model_passes_folder, 'passes_log.json')

    # Load the existing data
    passes_log = load_passes_log(passes_log_file)

    # Create a new entry if the model_name key doesn't exist
    if model_name not in passes_log:
        passes_log[model_name] = {}

    # Create a new entry if the model_init_time_str key doesn't exist
    if model_init_time_str not in passes_log[model_name]:
        passes_log[model_name][model_init_time_str] = []

    # Append the new data to the list
    passes_log[model_name][model_init_time_str].append({
        'start_basin': start_basin,
        'start_lon': start_lon,
        'start_lat': start_lat,
        'candidate_lon': candidate_lon,
        'candidate_lat': candidate_lat,
        'valid_time_str': valid_time_str,
        'subfolder_name': subfolder_name,
        'file_name': file_name,
        'sat_str': sat_str
    })

    # Save the updated data
    with open(passes_log_file, 'w') as f:
        json.dump(passes_log, f, indent=4)


# reset passes for current model init time (in case of a rerun is necessary)
def load_and_reset_passes_log(save_model_passes_folder, model_name, model_init_time_str):
    passes_log_file = os.path.join(save_model_passes_folder, 'passes_log.json')

    # Create the file if it doesn't exist
    if not os.path.exists(passes_log_file):
        with open(passes_log_file, 'w') as f:
            json.dump({}, f)

    # Load the existing data
    with open(passes_log_file, 'r') as f:
        passes_log = json.load(f)

    # Create a new entry if the model_name key doesn't exist
    if model_name not in passes_log:
        passes_log[model_name] = {}

    # Reset the model_init_time_str key if it exists
    if model_init_time_str in passes_log[model_name]:
        passes_log[model_name][model_init_time_str] = []

    # Save the updated data
    with open(passes_log_file, 'w') as f:
        json.dump(passes_log, f, indent=4)

    return passes_log


def load_passes_log(passes_log_file):
    # Create the file if it doesn't exist
    if not os.path.exists(passes_log_file):
        with open(passes_log_file, 'w') as f:
            json.dump({}, f)

    # Load the existing data
    with open(passes_log_file, 'r') as f:
        passes_log = json.load(f)

    return passes_log

# TODO: Statistics:
# Center fit
# Get area, coverage also

# Only create strtrees for the valid times we are interested in
min_valid_time, max_valid_time = find_min_max_valid_time(swath_gdfs)
strtree_list, strtree_min_times, strtree_max_times = create_strtrees_for_swaths(swath_gdfs)
# Get the predicted passes
calculate_pass_predictions_and_plot()








# global image dims (don't change)
image_width = 900
image_height = 450
# how small a circle, in radius in pixels, for mouse hover / click on global map
circle_size = 10

# Load the existing data
passes_log_file = os.path.join(save_model_passes_folder, 'passes_log.json')
with open(passes_log_file, 'r') as f:
    passes_log = json.load(f)

# Get the latest model run
latest_model_run = list(passes_log.keys())[-1]

# Get the latest pass
latest_pass = list(passes_log[latest_model_run].keys())[-1]


# Create a dictionary to store unique start_lon, start_lat pairs
start_lon_lat_pairs = {}
for candidate in passes_log[latest_model_run][latest_pass]:
    start_lon_lat_pair = f"{candidate['start_lat']}, {candidate['start_lon']}"
    if start_lon_lat_pair not in start_lon_lat_pairs:
        start_lon_lat_pairs[start_lon_lat_pair] = []
    start_lon_lat_pairs[start_lon_lat_pair].append(latest_pass)

# Sort the start_lon_lat_pairs by longitude
sorted_start_lon_lat_pairs = sorted(start_lon_lat_pairs.keys(), key=lambda x: float(x.split(', ')[1]))


# Create a figure with a Cartopy map
fig = plt.figure(figsize=(image_width/100, image_height/100))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Remove the axes border
ax.axis('off')

# Adjust the layout so that the subplot fills the entire figure
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Add some map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)

# Set the extent of the map
ax.set_extent([-180, 180, -90, 90])

# Add markers for each candidate location
for candidate in passes_log[latest_model_run][latest_pass]:
    ax.plot(candidate['candidate_lon'], candidate['candidate_lat'], 'o', color='red', markersize=2)

# Save the figure as an image
image_file_path = os.path.join(save_model_passes_folder, 'global_image.png')
plt.savefig(image_file_path, dpi=100, bbox_inches=None, pad_inches=0)

dt_expires = dt_utc + timedelta(hours=6)
expires_string = dt_expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
meta_tag = f'<meta http-equiv="Expires" content="{expires_string}">\n'

# Create an HTML file with the image map and a right-hand side frame
html_file_path = os.path.join(save_model_passes_folder, 'index.html')
with open(html_file_path, 'w') as f:
    f.write('<html><head><title>ASCAT Pass Predictions for TCs\n</title>')
    f.write(meta_tag)
    f.write('<style>\n')
    f.write('#pass-select { font-size: 14px; padding: 5x; }\n')
    f.write('#prev-button, #next-button { font-size: 12px; padding: 5px; }\n')
    f.write('''.pass-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    width: 100%;
    height: 100%;
    margin-bottom: 10px;
    display: block;
}''')

    f.write('</style>\n')
    f.write('<body style="margin: 0; padding: 0; style="overflow-y: hidden">\n')

    f.write('<div style="overflow-y: hidden">\n')

    f.write('<div id="global-map-frame" style="width: 55%; height: 90vh; overflow-x: auto; overflow-y: auto; float: left;">\n')
    f.write('<h1 style="font-size: 20px; text-align: center;">ASCAT Predictions for Tropical Cyclones</h1>\n')
    f.write('<h3 style="font-size: 18px; text-align: center;">EXPERIMENTAL! Includes genesis tracks.</h3>\n')
    f.write('<h3 style="font-size: 18px; text-align: center;">METOP-B, METOP-C, HAIYANG-2B, HAIYANG-2C, OCEANSAT-3</h3>\n')
    f.write(f'<h2 style="font-size: 18px; text-align: center;">Predictions from {latest_pass}</h1>\n')

    timestamp = candidate_init_time_dt.strftime("%Y%m%d%H%M%S%f")
    url = f"global_image.png?_t={timestamp}"

    f.write(
        f'<img src="{url}" ontouchstart="" usemap="#map" style="width: {image_width}px; height: {image_height}px; margin-left: 20px; margin-top: auto; margin-bottom: auto; border: none;">\n')

    f.write('<map name="map">\n')


    # Get the image coordinates from the latitude and longitude values
    for candidate in passes_log[latest_model_run][latest_pass]:
        x, y = ax.transData.transform((candidate['candidate_lon'], candidate['candidate_lat']))
        x = int(x)
        # flip y coordinates as cartopy uses y=0 to refer to the bottom, whereas HTML image coordinates use the top
        y = image_height - int(y) - 1

        f.write(f'<area shape="circle" coords="{x},{y},{circle_size}" href="#" onclick="showPass(\'{candidate["start_lat"]}, {candidate["start_lon"]}\')" onmouseover="showPass(\'{candidate["start_lat"]}, {candidate["start_lon"]}\')">')



    f.write('</map>\n')
    f.write('</div>\n')
    f.write('<div style="width: 45%; height: 90vh; float: right; overflow-y: auto;">\n')
    f.write('<div style="width: 95%; height: 40px; background-color: #f0f0f0; padding: 10px;">\n')
    f.write('<label for="pass-select" style="font-size: 16px;">Storm start lat, lon:</label>&nbsp;\n')
    f.write('<select id="pass-select" onchange="showPassList(this.value)">\n')
    for start_lon_lat_pair in sorted_start_lon_lat_pairs:
        f.write(f'<option value="{start_lon_lat_pair}">{start_lon_lat_pair}</option>\n')
    f.write('</select>\n')
    f.write('<button id="prev-button" onclick="showPrevPass()">Prev Pass</button>\n')
    f.write('<button id="next-button" onclick="showNextPass()">Next Pass</button>\n')
    f.write('</div>\n')
    f.write('<div id="pass-list" style="width: 95%; height: calc(90vh - 120px); overflow-y: auto;">\n')
    for i, start_lon_lat_pair in enumerate(sorted_start_lon_lat_pairs):
        if i == 0:
            f.write(f'<div id="{start_lon_lat_pair}" style="padding: 10px; display: block;">\n')
        else:
            f.write(f'<div id="{start_lon_lat_pair}" style="padding: 10px; display: none;">\n')
        images = set()
        for candidate in passes_log[latest_model_run][latest_pass]:
            if candidate["start_lon"] == float(start_lon_lat_pair.split(", ")[1]) and candidate["start_lat"] == float(start_lon_lat_pair.split(", ")[0]):
                image_path = f'{candidate["subfolder_name"]}/{candidate["file_name"]}'
                if image_path not in images:
                    images.add(image_path)
                    f.write(f'<img src="{image_path}" class="pass-image">\n')
        f.write('</div>\n')
    f.write('</div>\n')
    f.write('</div>\n')
    f.write('</div>\n')
    f.write('<script>\n')
    f.write('var currentIndex = 0;\n')
    f.write('function showPass(startLonLatPair) {\n')
    f.write('  document.getElementById("pass-select").value = startLonLatPair;\n')
    f.write('  showPassList(startLonLatPair);\n')
    f.write('}\n')
    f.write('function showPassList(startLonLatPair) {\n')
    f.write('  var passList = document.getElementById("pass-list");\n')
    f.write('  var passDivs = passList.children;\n')
    f.write('  for (var i = 0; i < passDivs.length; i++) {\n')
    f.write('    if (passDivs[i].id == startLonLatPair) {\n')
    f.write('      passDivs[i].style.display = "block";\n')
    f.write('    } else {\n')
    f.write('      passDivs[i].style.display = "none";\n')
    f.write('    }\n')
    f.write('  }\n')
    f.write('  currentIndex = 0;\n')
    f.write('  showPrevPass();\n')
    f.write('}\n')
    f.write('function showPrevPass() {\n')
    f.write('  var passList = document.getElementById("pass-list");\n')
    f.write('  var passDiv = passList.querySelector("div[style*=\'display: block\']");\n')
    f.write('  var images = passDiv.children;\n')
    f.write('  if (currentIndex > 0) {\n')
    f.write('    currentIndex--;\n')
    f.write('  }\n')
    f.write('  images[currentIndex].scrollIntoView();\n')
    f.write('}\n')
    f.write('function showNextPass() {\n')
    f.write('  var passList = document.getElementById("pass-list");\n')
    f.write('  var passDiv = passList.querySelector("div[style*=\'display: block\']");\n')
    f.write('  var images = passDiv.children;\n')
    f.write('  if (currentIndex < images.length - 1) {\n')
    f.write('    currentIndex++;\n')
    f.write('  }\n')
    f.write('  images[currentIndex].scrollIntoView();\n')
    f.write('}\n')
    f.write('document.addEventListener("DOMContentLoaded", function() {\n')
    f.write('  var map = document.querySelector("map");\n')
    f.write('  var areas = map.querySelectorAll("area");\n')
    f.write('  areas.forEach(function(area) {\n')
    f.write('    area.addEventListener("mouseover", function() {\n')
    f.write('      area.style.cursor = "pointer";\n')
    f.write('    });\n')
    f.write('    area.addEventListener("mouseout", function() {\n')
    f.write('      area.style.cursor = "default";\n')
    f.write('    });\n')
    f.write('  });\n')
    f.write('});\n')
    f.write('</script>\n')
    f.write('</body></html>\n')







def upload_to_ftp(ftp_address, ftp_port, ftp_username, ftp_password, http_proj_folder):
    # Connect to the FTP server
    ftp = ftplib.FTP()
    ftp.connect(ftp_address, ftp_port)
    ftp.login(ftp_username, ftp_password)

    # Create the project folder if it doesn't exist
    if not ftp.nlst('/' + http_proj_folder.split('/')[0]):
        ftp.mkd('/' + http_proj_folder.split('/')[0])

    if not ftp.nlst('/' + http_proj_folder):
        ftp.mkd('/' + http_proj_folder)

    # Get the latest model run
    latest_model_run = list(passes_log.keys())[-1]

    # Get the latest pass
    latest_pass = sorted(passes_log[latest_model_run].keys())[-1]

    # Get the subfolder name
    subfolder_name = passes_log[latest_model_run][latest_pass][0]['subfolder_name']

    if not ftp.nlst(f'/{http_proj_folder}/{subfolder_name}'):
        ftp.mkd(f'/{http_proj_folder}/{subfolder_name}')

    # Upload the image files first
    for pass_info in passes_log[latest_model_run][latest_pass]:
        local_file_path = os.path.join(save_model_passes_folder, subfolder_name, pass_info["file_name"])
        remote_file_path = f'/{http_proj_folder}/{subfolder_name}/{pass_info["file_name"]}'
        ftp.storbinary('STOR ' + remote_file_path, open(local_file_path, 'rb'))

    # Upload the global image
    local_file_path = os.path.join(save_model_passes_folder, 'global_image.png')
    remote_file_path = f'/{http_proj_folder}/global_image.png'

    if ftp.nlst(remote_file_path):
        ftp.delete(remote_file_path)

    ftp.storbinary('STOR ' + remote_file_path, open(local_file_path, 'rb'))

    # Upload the HTML files last
    html_files = ['index.html', 'map.html']
    for file in html_files:
        local_file_path = os.path.join(save_model_passes_folder, file)
        remote_file_path = f'/{http_proj_folder}/{file}'
        ftp.storbinary('STOR ' + remote_file_path, open(local_file_path, 'rb'))

    # Get a list of subdirectories in the project folder
    subdirs = ftp.nlst('/' + http_proj_folder)

    subdirs = [subdir for subdir in subdirs if '.' not in subdir]


    # Delete any old subdirectories and their contents
    for subdir in subdirs:
        if subdir != subfolder_name:
            files = ftp.nlst(f'/{http_proj_folder}/{subdir}')
            files = [file for file in files if file[-4:] == '.png']
            for file in files:
                ftp.delete(f'/{http_proj_folder}/{subdir}/{file}')

            ftp.rmd(f'/{http_proj_folder}/{subdir}')

    # Close the FTP connection
    ftp.quit()



upload_url = 'ftpupload.net'
upload_port = 21

with open('cred.txt', 'r') as f:
    ftp_username = f.readline().strip()
    ftp_password = f.readline().strip()


upload_to_ftp(upload_url, upload_port, ftp_username, ftp_password, 'htdocs/ascatpred')

