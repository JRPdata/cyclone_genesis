# EXPERIMENTAL
# Work in progress (do not use)

# find TC genesis from models based on work similar to FSU's methodology
# Note: uses roughly same thresholds but for different model resolutions, so they are more sensitive than [1]

# only focused on 1. so far:
# [1] https://journals.ametsoc.org/view/journals/wefo/28/6/waf-d-13-00008_1.xml
# [2] https://journals.ametsoc.org/view/journals/wefo/31/3/waf-d-15-0157_1.xml?tab_body=fulltext-display
# [3] https://journals.ametsoc.org/view/journals/wefo/32/1/waf-d-16-0072_1.xml

# for ROCI:
# [4] https://journals.ametsoc.org/view/journals/wefo/28/6/waf-d-13-00008_1.xml

# all units are stored in metric (data Vmax @ 10m printed in kts), but
# scale may be mixed between internal representations of data, thresholds, and printouts

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

# where to save the disturbance maps
tc_maps_folder = '/home/db/Documents/JRPdata/cyclone_genesis_charts'

# path to font for charts
font_path = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'

# for debugging MSLP isobars (depth first search)
debug_isobar_images = False

# save vorticity calculations
debug_save_vorticity = False

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

import traceback
import copy
import time
import os

import warnings

#%matplotlib inline

from datetime import datetime, timedelta
import sqlite3

from matplotlib.patches import Circle

# for generating charts of disturbances
from IPython.display import display, clear_output

from ipyleaflet import Map, LayersControl, basemaps, basemap_to_tiles, LayerGroup, FullScreenControl, LegendControl, ImageOverlay, WidgetControl, GeoJSON
from PIL import Image, ImageDraw, ImageFont
from ipywidgets import HTML, Layout

import io
import base64

import networkx as nx
import matplotlib.pyplot as plt
from pyproj import Geod

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

# close old maps generated
old_maps = []

old_tc_maps = []

meters_to_knots = 1.943844

# get list of completed TC candidates (tracks)
def get_tc_candidates_by_basin(basin_name, limit=10):
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()

        # Query all rows from the 'disturbances' table and order by 'model_timestamp'
        cursor.execute('SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE start_basin = ? ORDER BY init_date DESC LIMIT ?', (basin_name, limit))
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
        print(f"SQLite error (get_tc_candidates_by_basin): {e}")
    finally:
        if conn:
            conn.close()

    return all_retrieved_data

def max_model_time_steps_from_timestamp_str(model_name, timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    hour = dt.hour
    hour_str = f'{hour:02}'
    return all_time_steps_by_model[model_name][hour_str][-1]

# get list of completed TC candidates (tracks)
def get_tc_candidates_from_last_n_model_init_times(num_init_times=1):
    all_retrieved_data = []  # List to store data from all rows
    conn = None
    model_names = list(model_data_folders_by_model_name.keys())
    model_init_times = {}
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()

        for model_name in model_names:
            cursor.execute('SELECT DISTINCT init_date FROM completed WHERE model_name = ? ORDER BY init_date DESC LIMIT ?', (model_name, num_init_times))
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
                    # Query all rows from the 'disturbances' table and order by 'model_timestamp'
                    cursor.execute('SELECT model_name, init_date, start_valid_date, ws_max_10m, data FROM tc_candidates WHERE model_name = ? AND init_date = ? ORDER BY init_date DESC', (model_name, init_date))
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
        print(f"SQLite error (get_tc_candidates_from_last_n_model_init_times): {e}")
    finally:
        if conn:
            conn.close()

    return model_init_times, all_retrieved_data

############################################################
########   CHART DISTURBANCES & CYCLONE CANDIDATES  ########
############################################################

def get_title_overlay(title_text):
    # Create a blank image with a white background
    width, height = 360, 360
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define the text and font
    font_size = 28
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0,0), title_text, font=font, align='center')

    # Calculate the position to center the text vertically
    x_min, y_min, x_max, y_max = text_bbox
    x = (width - (x_max - x_min)) // 2
    y = (height - (y_max - y_min)) // 2

    # Increase the size of the rectangle to add more space around the text
    padding = 10  # Adjust this value to control the size of the rectangle
    rectangle_x0 = x - padding
    rectangle_y0 = y - padding
    rectangle_x1 = x_max + x + padding
    rectangle_y1 = y_max + y + padding

    # Fill the background with a white rectangle
    draw.rectangle([rectangle_x0, rectangle_y0, rectangle_x1, rectangle_y1], fill=(255, 255, 255, 255))

    # Add a thin red border
    border_width = 1
    draw.rectangle([rectangle_x0, rectangle_y0, rectangle_x1, rectangle_y1], outline=(255, 0, 0, 255), width=border_width)

    # Draw the text on the image
    draw.multiline_text((x, y), title_text, font=font, fill=(0, 0, 0, 255))  # Black text

    # Convert the PIL image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = image_bytes.getvalue()
    image_bytes.close()

    # Encode the image data in base64
    image_base64 = base64.b64encode(image_data).decode()

    # Create a data URI for the image
    image_data_uri = f"data:image/png;base64,{image_base64}"

    # Calculate the bounds for the ImageOverlay
    bounds = [(8,40), (72,130)]

    # Create an ImageOverlay with the data URI for the image and calculated bounds
    title_overlay = ImageOverlay(
        name='Title',
        url=image_data_uri,
        bounds=bounds  # Use the calculated bounds
    )
    return title_overlay

def generate_tc_chart(tc_candidates, saveonly=False):
    def update_html(feature, **kwargs):
        html1.value = '''
            <div id='DISTURBANCE_PROPS'>
            <p>Model: {}</p>
            <p>Init Time: {}</p>
            <p>Valid Time: {}</p>
            <p>Time step (h): +{}</p>
            <p>Lat, Lon: {}, {}</p>
            <p>ROCI (km): {}</p>
            <p>Isobar Delta (hPa)): {}</p>
            <p>MSLP (hPa): {}</p>
            <p>Vmax (m/s) @ 10m: {}</p>
            </div>
        '''.format(
            feature['properties']['model_name'],
            feature['properties']['init_time'],
            feature['properties']['valid_time'],
            feature['properties']['time_step'],
            feature['properties']['lat'], feature['properties']['lon'],
            feature['properties']['roci'],
            feature['properties']['closed_isobar_delta'],
            feature['properties']['mslp_value'],
            feature['properties']['vmax10m_in_roci']
        )

    for m in old_tc_maps:
        m.close()
        m.close_all()

    lat_lon_with_time_step_list = []
    most_recent_timestamp = None
    for tc in tc_candidates:
        model_name = tc['model_name']
        model_timestamp = tc['model_timestamp']
        disturbance_candidates = tc['disturbance_candidates']
        if not most_recent_timestamp:
            most_recent_timestamp = model_timestamp

        if datetime.fromisoformat(most_recent_timestamp) < datetime.fromisoformat(model_timestamp):
            most_recent_timestamp = model_timestamp

        if disturbance_candidates:
            prev_lat = None
            prev_lon = None
            prev_lon_repeat2 = None
            prev_lon_repeat3 = None
            for time_step_str, valid_time_str, candidate in disturbance_candidates:
                time_step_int = int(time_step_str)
                lon = candidate['lon']
                candidate_info = {}
                candidate_info['model_name'] = model_name
                candidate_info['init_time'] = model_timestamp
                candidate_info['lat'] = f"{candidate['lat']:3.2f}"
                candidate_info['lon'] = f"{lon:3.2f}"
                candidate_info['lon_repeat'] = candidate_info['lon']
                candidate_info['valid_time'] = valid_time_str
                candidate_info['time_step'] = time_step_int
                candidate_info['roci'] = f"{candidate['roci']/1000:3.1f}"
                candidate_info['vmax10m_in_roci'] = f"{candidate['vmax10m_in_roci']:3.2f}"
                candidate_info['closed_isobar_delta'] = candidate['closed_isobar_delta']
                candidate_info['mslp_value'] = f"{candidate['mslp_value']:4.1f}"

                if prev_lon:
                    prev_lon_f = float(prev_lon)
                    if abs(prev_lon_f - lon) > 270:
                        if prev_lon_f < lon:
                            prev_lon = f"{prev_lon_f + 360:3.2f}"
                        else:
                            prev_lon = f"{prev_lon_f - 360:3.2f}"

                candidate_info['prev_lat'] = prev_lat
                candidate_info['prev_lon'] = prev_lon
                candidate_info['prev_lon_repeat'] = prev_lon

                prev_lat = candidate_info['lat']
                prev_lon = lon

                lat_lon_with_time_step_list.append(candidate_info)
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

    # Create the initial map
    # darkmatter is dark, positron is light
    # CartoDB.DarkMatter
    # CartoDB.Positron
    darkmatter = basemap_to_tiles(basemaps.CartoDB.DarkMatter)
    darkmatter.base = True
    darkmatter.name = 'Dark'

    positron = basemap_to_tiles(basemaps.CartoDB.Positron)
    positron.base = True
    positron.name = 'Light'

    layout=Layout(height='800px')

    m = Map(center=(0, 200), zoom=1.5, layout=layout, layers=[positron, darkmatter])

    html1 = HTML(
    '''
    <div id='DISTURBANCE_PROPS'>
    <h4>Disturbance Info:</h4>
    Mouse hover over a point</div>
    '''
    )
    html1.layout.margin = '0px 20px 20px 20px'

    control1 = WidgetControl(widget=html1, position='bottomright')
    m.add(control1)

    #model_date = datetime.strftime(model_timestamp, '%Y-%m-%d')
    #model_hour = f'{model_timestamp.hour:02}'
    title_text = f'TC candidate tracks\nLast : {most_recent_timestamp[:-6]}'

    m.add(get_title_overlay(title_text))

    colors = ['red', 'yellow', 'springgreen', 'darkturquoise', 'blue', 'orchid']

    # Define 6 different time_step ranges and their corresponding colors
    time_step_ranges = [
        (0, 48),
        (49, 120),
        (121, 168),
        (169, 240),
        (241, 336),
        (337, float('inf'))  # Anything larger than 57
    ]

    range_str = [
        '2d',
        '5d',
        '7d',
        '10d',
        '14d',
        '14+d'
    ]

    # Create a LegendControl with color names
    legend = LegendControl(
        {
            "2 days": "red",
            "5 days": "yellow",
            "7 days": "springgreen",
            "10 days": "darkturquoise",
            "14 days": "blue",
            "14+ days": "orchid"
        },
        title="Days+Init",
        position="bottomleft"
    )

    # Create custom marker styles and feature collections for each time_step range
    custom_marker_styles = []
    feature_collections = []
    layer_groups = []

    # do in reversed order so most recent items get rendered on top
    for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
        style = {
            'color': colors[i],
            'fill': True,
            'fillColor': colors[i],
            'opacity': 0.6,
            'radius': 1  # Adjust the radius as needed
        }
        custom_marker_styles.append(style)

        features = []
        for point in reversed(lat_lon_with_time_step_list):
            time_step = point['time_step']
            if start <= time_step <= end:
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [point['lon_repeat'], point['lat']]
                    },
                    'properties': point
                }

                features.append(feature)

        name = f'{range_str[i]}'
        feature_collection = GeoJSON(data={
            'type': 'FeatureCollection',
            'features': features
        }, point_style=style)
        feature_collection.on_hover(update_html)
        feature_collections.append(feature_collection)

        layer_group = LayerGroup(layers=(feature_collection,), name=name)
        layer_groups.append(layer_group)
        m.add(layer_group)


    # Create custom marker styles and feature collections for each time_step range
    custom_marker_styles = []
    feature_collections = []
    layer_groups = []

    # do in reversed order so most recent items get rendered on top
    for i, (start, end) in reversed(list(enumerate(time_step_ranges))):
        style = {
            'color': colors[i],
            'fill': True,
            'fillColor': colors[i],
            'opacity': 0.2,
            'stroke-width': 7,
            'strokeWidth': 9,
            'strokewidth': 11,
            'width': 13
        }
        custom_marker_styles.append(style)

        features = []
        for point in reversed(lat_lon_with_time_step_list):
            time_step = point['time_step']
            if start <= time_step <= end:
                if point['prev_lon_repeat']:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [point['prev_lon_repeat'], point['prev_lat']], [point['lon_repeat'], point['lat']]
                                ]
                        }
                    }
                    features.append(feature)

        name = 'Tracks'
        feature_collection = GeoJSON(data={
            'type': 'FeatureCollection',
            'features': features
        }, style=style)
        feature_collections.append(feature_collection)

        layer_group = LayerGroup(layers=(feature_collection,), name=name)
        layer_groups.append(layer_group)
        m.add(layer_group)

    # Add legend to the map
    m.add(legend)
    m.add(LayersControl())
    m.add(FullScreenControl())

    # Display the interactive map
    if not saveonly:
        clear_output()
        display(m)

    old_tc_maps.append(m)

    # save the map if wanted
    if save_tc_maps:
        file_name = f'{most_recent_timestamp}_c{datetime.utcnow().isoformat()}.html'
        save_file_path = os.path.join(tc_maps_folder, file_name)
        m.save(save_file_path)
        return save_file_path
    else:
        return None

#########################################################
########       CALCULATE DISTURBANCES            ########
#########################################################

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
        print(f"SQLite error (get_disturbances_from_db): {e}")
    finally:
        if conn:
            conn.close()

    return retrieved_data

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

# rv_to_sign just maps to positive and negative, without it, it does an (arbitrary) color mapping
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
        print(f"{numbering_str}{basin_str}Latitude (deg:): {lat: >6.1f}, Longitude (deg): {lon: >6.1f}, MSLP (hPa): {formatted_mslp}{roci_str}\n        {rv_str}{thickness_str}{vmax_str}\n        {vmax10m_in_roci_str}{closed_isobar_delta_str}")

        if debug_isobar_images:
            create_image(candidate['neighborhood'], 'Neighborhood')
            create_image(candidate['visited'], 'Visited')

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

#########################################################
######## PROCESS DISTURBANCES INTO TC CANDIDATES ########
#########################################################

def get_unprocessed_disturbance_model_runs():
    # Sort the data from disturbances
    sorted_data = get_all_disturbances_sorted_by_timestamp()

    tc_completed_by_model_name = get_model_timestamp_of_completed_tc_by_model_name()
    unprocessed_data = []
    for entry in sorted_data:
        model_name = entry['model_name']
        model_time_stamp = entry['model_timestamp']
        if model_name in tc_completed_by_model_name:
            if model_time_stamp in tc_completed_by_model_name[model_name]:
                continue
        unprocessed_data.append(entry)

    return unprocessed_data

def calc_tc_candidates():
    unprocessed_data = get_unprocessed_disturbance_model_runs()
    if len(unprocessed_data) == 0:
        return
    print("Calculate TC candidates (and simplified tracks)")
    print(f"# Model runs to process: {len(unprocessed_data)}")
    print("")
    for row in unprocessed_data:
        graph = create_graph_and_add_candidates(row)
        connect_graph_colocated(graph)
        connect_graph_next_timestep(graph)
        remove_disturbances_not_meeting_time_criteria(graph)
        process_and_simplify_graph(graph)

    print("")
    print("Completed all model runs.")

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
        print(f"# components (before separation): {num_components}")
        num_components = len(components)
        print(f"# components (after separation): {num_components}")
    else:
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
        print(f"  Start Valid Time: {first_valid_time}")
        print_candidates([graph.nodes[first_node]["data"]], no_numbering=True)
        print(f"  Last Valid Time: {last_valid_time}")
        print_candidates([graph.nodes[last_node]["data"]], no_numbering=True)

        max_10m_wind_speed = None
        if max_10m_wind_speed_node is not None:
            max_10m_wind_speed_valid_time = graph.nodes[max_10m_wind_speed_node]["valid_time"]
            print(f"  Max 10m Wind Speed Valid Time: {max_10m_wind_speed_valid_time}")
            print_candidates([graph.nodes[max_10m_wind_speed_node]["data"]], no_numbering=True)
            max_10m_wind_speed = float(graph.nodes[max_10m_wind_speed_node]["data"]['vmax10m_in_roci'])
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
        print('# Removed nodes: ', num_removed_nodes)

    if num_removed_next_edges:
        print('# Removed Next Edges: ', num_removed_next_edges)

    if num_removed_colocated_edges:
        print('# Removed Colocated Edges: ', num_removed_colocated_edges)

    # mark model run completed if no errors
    if not has_error:
        update_tc_completed(model_name, init_time)

    # store the modified graph
    file_graph_path = os.path.join(graphs_folder, f"{graph.graph['name']}.gexf")
    if write_graphs:
        nx.write_gexf(graph, file_graph_path, encoding="utf-8", prettyprint=True)

    print("")

# returns all disturbances by timestamp for completed model runs ONLY
def get_all_disturbances_sorted_by_timestamp():
    conn = None
    all_retrieved_data = []  # List to store data from all rows

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(disturbances_db_file_path)
        cursor = conn.cursor()

        # Query all rows from the 'disturbances' table and order by 'model_timestamp'
        cursor.execute('SELECT model_name, date, data FROM disturbances WHERE is_complete = 1 ORDER BY date')
        results = cursor.fetchall()
        if results:
            # Process data for each row
            for row in results:
                model_name, model_timestamp, data = row
                retrieved_data = {
                    "model_name": model_name,
                    "model_timestamp": model_timestamp,
                    "data": json.loads(data)
                }
                all_retrieved_data.append(retrieved_data)

    except sqlite3.Error as e:
        print(f"SQLite error (get_all_disturbances_sorted_by_timestamp): {e}")
    finally:
        if conn:
            conn.close()

    return all_retrieved_data

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

# get dict of the completed TC candidate (tracks) by model name
def get_model_timestamp_of_completed_tc_by_model_name():
    completed_tc_dicts = get_tc_completed()
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
def get_tc_completed():
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
        print(f"SQLite error (get_tc_completed): {e}")
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
            dt = datetime.utcnow()
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
        conn = sqlite3.connect(tc_disturbances_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tc_disturbances (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                data JSON,
                date TEXT,
                done_components JSON,
                UNIQUE(model_name, date)
            )
        ''')
        # track which components are already finished
        cursor.execute('SELECT done_components FROM tc_disturbances WHERE model_name = ? AND date = ?', (model_name, model_init_time))
        result = cursor.fetchone()
        if result:
            done_components = json.loads(result[0])
        else:
            done_components = []

        if component_num not in done_components:
            done_components.append(component_num)
            done_components_json_data = json.dumps(done_components)

            for time_step, valid_time, disturbance_candidate in tc_disturbance_candidates:
                model_time_step_str = str(time_step)

                cursor.execute('SELECT data FROM tc_disturbances WHERE model_name = ? AND date = ?', (model_name, model_init_time))
                result = cursor.fetchone()

                if result:
                    retrieved_data = json.loads(result[0])
                else:
                    retrieved_data = {}

                if model_time_step_str not in retrieved_data:
                    retrieved_data[model_time_step_str] = []

                candidates = retrieved_data[model_time_step_str]
                candidates.append(disturbance_candidate)

                retrieved_data[model_time_step_str] = candidates

                json_data = json.dumps(retrieved_data)

                cursor.execute('INSERT OR REPLACE INTO tc_disturbances (model_name, data, date, done_components) VALUES (?, ?, ?, ?)', (model_name, json_data, model_init_time, done_components_json_data))
                conn.commit()

    except sqlite3.Error as e:
        has_error = True
        print(f"SQLite error (add_tc_candidate: tc_disturbances): {e}")
    finally:
        if conn:
            conn.close()

    conn = None
    try:
        # store tc disturbance candidates in database by model name, component (storm) id, valid time (access by model/id/valid_time)
        conn = sqlite3.connect(tc_candidates_db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tc_candidates (
                id INTEGER PRIMARY KEY,
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
        json_data = json.dumps(tc_disturbance_candidates)

        cursor.execute('SELECT component_id FROM tc_candidates WHERE model_name = ? AND init_date = ? AND component_id = ?', (model_name, model_init_time, component_num))
        result = cursor.fetchone()
        if not result:
            cursor.execute('INSERT OR REPLACE INTO tc_candidates (model_name, data, init_date, component_id, start_time_step, start_valid_date, start_basin, start_lat, start_lon, ws_max_10m) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (model_name, json_data, model_init_time, component_num, start_time_step, start_valid_time, start_basin, start_lat, start_lon, max_10m_wind_speed))
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

from bs4 import BeautifulSoup

def update_index_html(target_directory, save_file_path, mstr):
    # Store the current working directory
    original_directory = os.getcwd()

    try:
        # Change to the target directory
        os.chdir(target_directory)

        # Ensure target directory exists
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Read existing index.html content
        index_file_path = os.path.join(target_directory, 'index.html')
        if os.path.exists(index_file_path):
            with open(index_file_path, 'r') as index_file:
                existing_content = index_file.read()
        else:
            existing_content = ''

        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(existing_content, 'html.parser')

        # Find or create a div with a specific name
        target_div = soup.find('div', {'id': 'model_updates'})
        if target_div is None:
            # Create the HTML structure with the title
            soup = BeautifulSoup(
                f'<!DOCTYPE html>\n<html>\n<head>\n<title>Experimental TC charts from global models</title>\n</head>\n<body>\n</body>\n<h2>Experimental TC charts from global models</h2>\n<p> Do not use or rely on this!</p>\n<p> See <a href="https://github.com/JRPdata/cyclone_genesis">cyclone_genesis</a> and the notebook there for references.</p>\n<br><div id="model_updates"></div></html>',
                'html.parser'
            )
            target_div = soup.find('div', {'id': 'model_updates'})

        # Create a new link element
        link = soup.new_tag('a', href=os.path.basename(save_file_path))
        link.string = mstr

        br = soup.new_tag('br')
        br.append(BeautifulSoup("\n", 'html.parser'))
        target_div.insert(0, br)

        # Prepend the new link to the div
        target_div.insert(0, link)

        # Write the modified HTML back to the file
        with open(index_file_path, 'w') as index_file:
            index_file.write(str(soup))

        commit_changes(target_directory, mstr, index_file_path, save_file_path)

    finally:
        # Change back to the original working directory
        os.chdir(original_directory)

def get_existing_links(target_directory):
    index_file_path = os.path.join(target_directory, 'index.html')
    if os.path.exists(index_file_path):
        with open(index_file_path, 'r') as index_file:
            lines = index_file.readlines()
            # Extract existing links
            existing_links = [line.strip() for line in lines if line.startswith('<a href=')]
            return ''.join(existing_links)
    return ''

def commit_changes(target_directory, commit_message, *files):
    # Change to the target directory
    os.chdir(target_directory)

    # Stage files
    for file in files:
        os.system(f'git add {os.path.basename(file)}')

    # Commit changes
    os.system(f'git commit -m "{commit_message}"')

    os.system('git push')

last_model_init_times = []
while True:
    #print("\nChecking for new disturbance data from completed model runs")
    calc_tc_candidates()
    if save_tc_maps:
        # most recent N model runs
        num_init_times_per_model = 1
        model_init_times, recent_candidates = get_tc_candidates_from_last_n_model_init_times(num_init_times_per_model)
        if model_init_times != last_model_init_times:
            #for model_name, init_time in model_init_times.items():
            #    print(model_name, init_time)
            save_file_path = generate_tc_chart(recent_candidates, saveonly = True)
            mlist = []
            for model_name, init_time in model_init_times.items():
                mlist.append(model_name + " " + init_time[0][:-6])
            mstr = ", ".join(mlist)
            update_index_html(tc_maps_folder, save_file_path, mstr)

        last_model_init_times = model_init_times

    time.sleep(60 * polling_interval)
