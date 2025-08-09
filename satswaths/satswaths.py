# (Rough) Swaths from ASCAT (METOP-B,C, HAIYANG-2B,C, OCEANSAT-3), AMSR (GCOM-W1), GMI (GPM)
# Trapezoidal swaths (timing) will be inaccurate for conical scanning (non METOP) and are only rough estimates
# EXPERIMENTAL!
# Demo: http://jrpdata.free.nf/satswaths

# Updated TLEs are downloaded each run (from EUMETSAT directly for METOP-B,C, the rest from celestrak.org

# This code borrows code from pred_ascat_passes.py to calculate the swaths
# For choosing the sensing time, the time is essentially truncated to the minute covers the most of the storm's RMW (the point coordinate of the (1-minute duration, quadrilateral) swaths' corner closest to the interpolated MSLP center).  A more complicated method, rounding to the nearest minute, is possible but likely not needed, so not implemented.

import ftplib
import gzip
import os
import shutil
import traceback
import warnings
from datetime import datetime, timedelta

import antimeridian
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
from geographiclib.geodesic import Geodesic
from matplotlib.patches import Circle
from rtree import index
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.strtree import STRtree
from skyfield.api import EarthSatellite, load

# suppress http(s) warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

### CONFIG ##
# days offset for satellite passes, in case we want an earlier day or later day
offset = 0
# number of days of passes to consider (from UTC 00Z of current UTC datetime)
# 1 = 24 hours (from 00Z), 2 = 48 hours, etc
total_days = 5

# archive swath (for validation, debugging)
save_archive_swath_files = True
save_archive_path = './archive'

# save pass images
save_swath_images = False
save_ascat_swath_images_folder = './ascat_swaths'

# disable swath chart plotting
do_plot_swath_chart = False
# show swaths chart (ascat swaths)
show_swath_chart = False

# Define swath information for each satellite (m) (nadir blind range on each side and swath width from nadir (radius) in km)
# For sattelites  except METOP-B,C this is the HALF swath width; for satellites with PMW we use typically the typical min. effective swath widths (36/89 GHz)
# Different swath widths for SMOS (https://www.eoportal.org/satellite-missions/smos#mission-capabilities lists it as 1050 km)
#   For own effective swath width calculations for SMOS L2 NRT wind speed (operationally run by ESA), I get ~1128 km
swath_info = {
    "METOP-B": {"blind_range": 336000, "swath_width": 550000},
    "METOP-C": {"blind_range": 336000, "swath_width": 550000},
    "HAIYANG-2B": {"blind_range": 0, "swath_width": 900000},
    "HAIYANG-2C": {"blind_range": 0, "swath_width": 900000},
    "OCEANSAT-3": {"blind_range": 0, "swath_width": 900000},
    "SMAP": {"blind_range": 0, "swath_width": 500000}, # ~1000km (effective swath can vary for the RSS Ocean AWS product but is above 1000 for latitudes < 10 deg. N, however from 10N to 50N it drops from ~980km to 950km)
    "SMOS": {"blind_range": 0, "swath_width": 564000}, # ~1000km (see above; using typical min. effective swath width for ESA L2 NRT WS ~ 1128 km)
    "GPM-CORE": {"blind_range": 0, "swath_width": 500000}, # 442500 minimum (885km nominal, but using typical min. effective swath width for microwave imagery: 1000 km)
    "GCOM-W1": {"blind_range": 0, "swath_width": 800000} # 725000 minimum (1450km nominal, but using typical min. effective swath width for microwave imagery: 1600 km)
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
    'OCEANSAT-3': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=54361&FORMAT=2LE',
    'SMAP': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=40376&FORMAT=2LE',
    'SMOS': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=36036&FORMAT=2LE',
    'GPM-CORE': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=39574&FORMAT=2LE',
    'GCOM-W1': 'https://celestrak.org/NORAD/elements/gp.php?CATNR=38337&FORMAT=2LE',
}

os.makedirs(save_ascat_swath_images_folder, exist_ok=True)


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

    # must match the number of unique swaths (for metop-b,c this is each 4)
    colors = ['blue', 'purple', 'red', 'orange', 'yellow', 'green', 'pink', 'turquoise', 'lavender', 'brown', 'black', 'teal', 'magenta', 'lime', 'gray', 'amber', 'purple', 'cyan', 'gold', 'lime']
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

# r-tree index
rtree_p = index.Property()
rtree_idx = index.Index(properties=rtree_p)
# Mapping from rtree point index to (internal_id, tc_index, tc_candidate_point_index)
rtree_tuple_point_id = 0
rtree_tuple_index_mapping = {}

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


## OLD-OLD NOTES FROM pred_ascat_passes.py

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
#if len(plotted_tc_candidates) == 0:
#    raise Exception("No genesis events predicted...")

# dt_utc = plotted_tc_candidates[0][1][0]['init_time'].replace(tzinfo=pytz.utc)

dt_now = datetime.now(pytz.utc)
dt_utc = datetime.now(pytz.utc)

dt_utc += timedelta(days=offset)

start_dt = dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
# get one day earlier than current day UTC (may be inaccurate with new TLEs)
start_dt = start_dt - timedelta(days=1)
end_dt = dt_utc.replace(hour=23, minute=59, second=59, microsecond=999999)

end_dt += timedelta(days=total_days-1)

print()
print("Calculating new nadir paths, pass times, swaths.")

# calculate nadir paths, then pass times, and then the calculate and plot the swaths
nadir_paths = calculate_nadir_paths(all_tle_data, start_dt, end_dt)
pass_times = calculate_pass_times(nadir_paths)
swath_gdfs = calculate_swaths_and_plot(nadir_paths, start_dt, label_interval=10, opacity=0.3)


# If not already done
all_swath_gdf = gpd.GeoDataFrame(pd.concat(swath_gdfs, ignore_index=True))
columns_to_keep = ['valid_time', 'satellite_name', 'ascending', 'geometry']
gdf_clean = all_swath_gdf[columns_to_keep].copy()
gdf_clean['valid_time'] = gdf_clean['valid_time'].dt.strftime(r'%Y-%m-%dT%H:%M:%SZ')

gdf_clean.to_file("swaths.geojson", driver="GeoJSON")

# Compress it with gzip
with open("swaths.geojson", "rb") as f_in:
    ts_str = dt_now.strftime(r'%Y%m%dT%H%M%SZ')
    output_filename = "swaths.geojson.gz"
    archive_filename = f"swaths.geojson.{ts_str}.gz"
    with gzip.open(output_filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if save_archive_swath_files:
        os.makedirs(save_archive_path, exist_ok=True)
        shutil.copy(output_filename, os.path.join(save_archive_path, archive_filename))

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

    # Upload the files last
    files = ['swaths.geojson.gz']
    for file in files:
        local_file_path = file
        remote_file_path = f'/{http_proj_folder}/{file}'
        ftp.storbinary('STOR ' + remote_file_path, open(local_file_path, 'rb'))

    # Close the FTP connection
    ftp.quit()



upload_url = 'ftpupload.net'
upload_port = 21

with open('cred.txt', 'r') as f:
    ftp_username = f.readline().strip()
    ftp_password = f.readline().strip()

print("Uploading new swaths to website...")
upload_to_ftp(upload_url, upload_port, ftp_username, ftp_password, 'htdocs/satswaths')

formatted_date = dt_now.strftime("%Y-%m-%d %H:%M:%S")
print(f"{formatted_date} - Done uploading new swaths")
