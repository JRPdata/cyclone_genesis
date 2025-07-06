# Plot (passive Microwave) GCOM-W1 PCT (Polarized Corrected Temperature) imagery for 89 GHz, 36.5 GHz channels
# EXPERIMENTAL! (DO NOT USE)
# Some of JAXA's AMTK reference tool code (AMTK_AMSR2_Ver1.14) was ported over to python to align the 36.5 GHz data, which isn't geolocated
# (i.e. "coregistration" routines used for alignment of low freq. channels)

# Uses JAXA L1B data in HDF5 format
# ie. /standard/GCOM-W/GCOM-W.AMSR2/L1B/2/2025/07/GW1AM2_202507060651_015D_L1SGBTBR_2220220.h5 from JAXA's (free) distribution server
# thank you JAXA!

# Tip: Use JAXA's tool to find pass numbers of interest: https://www.eorc.jaxa.jp/AMSR/am_orbit/amsr_orbit_ja.html

# Credits to some various authors in comments below

# Note: Color scheme is not an exact match to NRL though does attempt to references older paper's algorithm for PCT.
# for 89GHz uses both horns to plot data (higher res than 36.5 GHz)

import sys
from tqdm import tqdm
import os
from datetime import datetime, timezone, timedelta
import h5py
import pandas as pd
import numpy as np
import math
from scipy.interpolate import griddata
from scipy.spatial import distance
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

## THETA (Θ) is used for calculation of the PCT
# optionally, use other numbers
USE_POPULAR_THETA = True

# Spencer et al. (1989); also called Pct85
DEFAULT_THETA_89 = 0.818
DEFAULT_THETA_36 = 1.18

# Plot a bounding box, with a radius of lat,lon degrees from center
PLOT_RADIUS_DEG = 5

# from Lee et al, 2002 to match NRL color scheme (clip and scale to these ranges)
PCT_MIN89, PCT_MAX89 = 220, 310
TBV_MIN89, TBV_MAX89 = 270, 290
TBH_MIN89, TBH_MAX89 = 240, 300

PCT_MIN36, PCT_MAX36 = 260, 280
TBV_MIN36, TBV_MAX36 = 160, 300
TBH_MIN36, TBH_MAX36 = 180, 310

# DEFUNCT FOR TESTING (PLOTTING RAW TB)
#K_MIN=145
#K_MAX=310
#R_MIN, R_MAX = 150, 275
#G_MIN, G_MAX = 170, 310
#B_MIN, B_MAX = 170, 310

# not used, built into data
# (for reference; TB data are integers; scale by this factor to obtain unscaled value)
DEFAULT_INT_SCALE_FACTOR = 0.01 # K

# Optional fine-tuned theta values (non-NRL scheme?)
# Cecil and Chronis (2018)
csv89_path = "microwave_theta_89ghz.csv"
csv36_path = "microwave_theta_36ghz.csv"

# === Load theta table ===
THETA89_DF = pd.read_csv(csv89_path)
THETA36_DF = pd.read_csv(csv36_path)


# === Port from AMTK (JAXA) ===

CHANNELS = ["6G", "7G", "10G", "18G", "23G", "36G"]

### ports from AMTK tool (JAXA)

AM2_LATLON_06 = 40010
AM2_LATLON_07 = 40020
AM2_LATLON_10 = 40030
AM2_LATLON_18 = 40040
AM2_LATLON_23 = 40050
AM2_LATLON_36 = 40060
AM2_LATLON_89A = 40070
AM2_LATLON_89B = 40080

DATASET_MAPS = {
    AM2_LATLON_89A: {
        'lat': "Latitude of Observation Point for 89A",
        'lon': "Longitude of Observation Point for 89A"
    },
    AM2_LATLON_89B: {
        'lat': "Latitude of Observation Point for 89B",
        'lon': "Longitude of Observation Point for 89B"
    }
    # No entry for AM2_LATLON_36, it will be generated from 89A + 89B
}

def common_prm_check(hdf_file, from_scan, to_scan, dataset_no):
    """
    Perform basic sanity checks on input parameters before accessing HDF5 data.

    Parameters:
        hdf_file (h5py.File): Open HDF5 file handle (or any non-None object)
        from_scan (int): Starting scan index
        to_scan (int): Ending scan index
        dataset_no (int): Dataset ID/magic constant (e.g., AM2_LATLON_89A)

    Returns:
        int: 0 for success, or a negative error code
    """

    AM2_SCANNUM_NOCHK = -9999

    ERROR_ILLEGAL_SCANNUM_S = -211
    ERROR_ILLEGAL_HDF_FILEID = -213
    ERROR_NOT_ACCESSLABEL = -217
    SUCCESS = 0

    if from_scan != AM2_SCANNUM_NOCHK and to_scan != AM2_SCANNUM_NOCHK:
        if from_scan > to_scan:
            return ERROR_ILLEGAL_SCANNUM_S

    if hdf_file is None:
        return ERROR_ILLEGAL_HDF_FILEID

    if dataset_no == 0:
        return ERROR_NOT_ACCESSLABEL

    return SUCCESS


def parse_coregistration_string(value):
    """
    Parse co-registration string into dict of channel -> value

    Parameters:
        value (str): e.g., "6G-1.51820,7G-1.40210,..."

    Returns:
        dict: mapping from channel name to float value
              e.g., {'6G': 1.5182, '7G': 1.4021, ..., '36G': 1.4004}
    """
    result = {}
    for entry in value.split(','):
        for ch in CHANNELS:
            if entry.startswith(f"{ch}-") or entry.startswith(f"{ch}--"):
                num_str = entry[len(ch) + 1:]
                result[ch] = float(num_str)
                break
    return result

def get_coef_from_attrs(h5_attrs):
    """
    Extract A1 and A2 coefficients for each AMSR2 channel from HDF5 attributes.

    Parameters:
        h5_attrs (dict-like): h5py-style .attrs from the HDF5 file

    Returns:
        dict: {'6G': [a1, a2], ..., '36G': [a1, a2] }
    """
    # Extract the raw strings from attributes
    a1_raw = h5_attrs['CoRegistrationParameterA1'][0]
    a2_raw = h5_attrs['CoRegistrationParameterA2'][0]

    # Parse into per-channel dictionaries
    a1_map = parse_coregistration_string(a1_raw)
    a2_map = parse_coregistration_string(a2_raw)

    # Combine into final structure
    coef = {}
    for ch in CHANNELS:
        coef[ch] = [
            a1_map.get(ch, 0.0),
            a2_map.get(ch, 0.0)
        ]

    return coef

def calc_outer_vector(p1, p2):
    """
    Cross product (outer product) of two 3D vectors
    """
    return np.cross(p1, p2)

def calc_inner_vector(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1 /= np.linalg.norm(p1)
    p2 /= np.linalg.norm(p2)
    return np.dot(p1, p2)

def radian2latlon(pos):
    """
    Convert 3D unit vector to (lat, lon) in degrees.

    Parameters:
        pos: list or array of [x, y, z]

    Returns:
        (lat, lon): latitude and longitude in degrees
    """
    x, y, z = pos
    lon_rad = math.atan2(y, x)

    if abs(x) < np.finfo(float).eps:
        if abs(y) < np.finfo(float).eps:
            lat_deg = 90.0 if z > 0 else -90.0
        else:
            lat_deg = math.degrees(math.atan((z * math.sin(lon_rad)) / y))
    else:
        lat_deg = math.degrees(math.atan((z * math.cos(lon_rad)) / x))

    lon_deg = math.degrees(lon_rad)

    return lat_deg, lon_deg

def conv_registration(pos_odd, pos_even, a1, a2):
    """
    Apply coregistration correction using 89GHz A/B positions to get low-freq pointing vector.

    Parameters:
        pos_odd:  np.array of shape (3,) → unit vector for odd (e.g., 89A)
        pos_even: np.array of shape (3,) → unit vector for even (e.g., 89B)
        a1, a2:   floats → co-registration coefficients for the target channel

    Returns:
        pos_low: np.array of shape (3,) → adjusted pointing vector
    """
    pos_odd = np.asarray(pos_odd, dtype=np.float64)
    pos_even = np.asarray(pos_even, dtype=np.float64)

    ex = pos_odd.copy()
    ez = np.cross(pos_odd, pos_even)
    len_ez = np.linalg.norm(ez)

    if len_ez < 1e-6:
        return ex  # fallback: no rotation
    else:
        ez /= len_ez
        ey = np.cross(ez, ex)

        cos_theta = np.dot(pos_odd / np.linalg.norm(pos_odd),
                           pos_even / np.linalg.norm(pos_even))
        theta = np.arccos(cos_theta)

        a1_theta = a1 * theta
        a2_theta = a2 * theta

        cos_a1 = np.cos(a1_theta)
        sin_a1 = np.sin(a1_theta)
        cos_a2 = np.cos(a2_theta)
        sin_a2 = np.sin(a2_theta)

        pos_low = (
            cos_a2 * (cos_a1 * ex + sin_a1 * ey) +
            sin_a2 * ez
        )
        return pos_low

def get_latlon(f, from_scan, to_scan, dataset_no, coef36=None):
    """
    Reads lat/lon from HDF5 or computes coregistered 36 GHz coordinates from 89A and 89B.

    Parameters:
        f: h5py.File
        from_scan, to_scan: int
        dataset_no: int → channel type
        coef36: tuple (a1, a2) used for 36 GHz registration if dataset_no is AM2_LATLON_36

    Returns:
        lat, lon: 2D arrays
    """
    if dataset_no in (AM2_LATLON_89A, AM2_LATLON_89B):
        ds_map = DATASET_MAPS[dataset_no]
        lat_data = f[ds_map['lat']][from_scan:to_scan + 1, :]
        lon_data = f[ds_map['lon']][from_scan:to_scan + 1, :]

        scale_lat = f[ds_map['lat']].attrs.get('scale_factor', 1.0)
        scale_lon = f[ds_map['lon']].attrs.get('scale_factor', 1.0)

        lat_scaled = lat_data * scale_lat
        lon_scaled = lon_data * scale_lon
        return lat_scaled, lon_scaled

    elif dataset_no == AM2_LATLON_36:
        if coef36 is None:
            raise ValueError("Must supply coef36 (a1, a2) for 36GHz coregistration")


        #for i in tqdm(range(shape[0]), desc="Calculating latlon grid for 36.5 GHz"):
        # Use 89A to generate it
        return get_registration_lat_lon(f, from_scan, to_scan, dataset_no)
        lat_89a, lon_89a = get_registration_lat_lon(f, from_scan, to_scan, AM2_LATLON_89A)

        shape = lat_89a.shape
        lat_36 = np.zeros_like(lat_89a)
        lon_36 = np.zeros_like(lon_89a)

        #for i in range(shape[0]):
        # for now a progress bar until we optimize it so its fast
        for i in tqdm(range(shape[0]), desc="Calculating latlon grid for 36.5 GHz"):
            for j in range(shape[1]):
                pos_odd = latlon2radian(lat_89a[i, j], lon_89a[i, j])
                pos_even = latlon2radian(lat_89b[i, j], lon_89b[i, j])
                pos_low = conv_registration(pos_odd, pos_even, coef36[0], coef36[1])
                lat, lon = radian2latlon(pos_low)
                lat_36[i, j] = lat
                lon_36[i, j] = lon

        return lat_36, lon_36

    else:
        raise ValueError(f"Unknown dataset_no: {dataset_no}")

def latlon2radian(lat, lon):
    """
    Convert latitude and longitude (degrees) to 3D Cartesian coordinates on unit sphere.

    Parameters:
        lat: latitude in degrees
        lon: longitude in degrees

    Returns:
        pos: list or numpy array of length 3: [x, y, z]
    """

    gclat = math.radians(lat)  # degrees to radians
    gclon = math.radians(lon)

    x = math.cos(gclat) * math.cos(gclon)
    y = math.cos(gclat) * math.sin(gclon)
    z = math.sin(gclat)

    return [x, y, z]

# instead of pointer
class AM2_COMMON_LATLON:
    def __init__(self, lat, lon):
        self.lat = np.float64(lat)
        self.lon = np.float64(lon)

def LatLon_Conversion_89G(info89a_odd, info89a_even, info_coef):
    """
    Calculate co-registered lat/lon for low frequencies (36.5 GHz) based on 89 GHz-A odd/even samples.

    Parameters:
        info89a_odd (AM2CommonLatLon): Odd scan point
        info89a_even (AM2CommonLatLon): Even scan point
        info_coef (LatLonCoef): A1 and A2 coefficients

    Returns:
        [float, float]: returns [lat, lon] on success, raises Exception otherwise
    """
    pos89a_odd = np.zeros(3)
    pos89a_even = np.zeros(3)
    #pos_low = np.zeros(3)
    
    AM2_DEF_LATLON_ABNML = -9991.0
    ERROR_ILLEGAL_LAT = -227
    ERROR_ILLEGAL_LON = -228

    # Check for abnormal values
    if info89a_odd[0] <= AM2_DEF_LATLON_ABNML:
        # Error code, [abnormal lat,lon]
        raise Exception(0, [info89a_odd[0], info89a_odd[0]])
    elif info89a_odd[1] <= AM2_DEF_LATLON_ABNML:
        raise Exception(0, [info89a_odd[0], info89a_odd[0]])
    elif info89a_even[0] <= AM2_DEF_LATLON_ABNML:
        raise Exception(0, [info89a_even[0], info89a_even[0]])
    elif info89a_even[1] <= AM2_DEF_LATLON_ABNML:
        raise Exception(0, [info89a_even[0], info89a_even[0]])

    # Check for illegal range
    if abs(info89a_odd[0]) > 90.0:
        raise Exception(ERROR_ILLEGAL_LAT, [])
    if abs(info89a_odd[1]) > 180.0:
        raise Exception(ERROR_ILLEGAL_LON, [])
    if abs(info89a_even[0]) > 90.0:
        raise Exception(ERROR_ILLEGAL_LAT, [])
    if abs(info89a_even[1]) > 180.0:
        raise Exception(ERROR_ILLEGAL_LON, [])

    # Convert to radian-based unit sphere vector
    try:
        pos89a_odd = latlon2radian(info89a_odd[0], info89a_odd[1])
        pos89a_even = latlon2radian(info89a_even[0], info89a_even[1])
    except:
        print(info89a_odd)
        print(info89a_even)
        raise Exception()

    # Perform co-registration shift
    pos_low = conv_registration(pos89a_odd, pos89a_even, info_coef[0], info_coef[1])

    # Convert back to lat/lon
    return radian2latlon(pos_low)

def get_registration_lat_lon(hdf_file_id, from_scan, to_scan, dataset_no):
    """
    Generate the co-registered lat/lon grid for 36 GHz (or other channels) using 89A data.

    Parameters:
        hdf_file_id: Open HDF5 file object (h5py.File)
        from_scan (int): Starting scan index
        to_scan (int): Ending scan index
        dataset_no (int): Dataset type identifier (e.g. AM2_LATLON_36)

    Returns:
        np.ndarray of AM2CommonLatLon: Half-resolution (coregistered) lat/lon grid
    """
    size = [0, 0, 0]
    #rank = AMTK_getDimSize(hdf_file_id, AM2_LATLON_89A, size)
    latlon_keys = DATASET_MAPS[AM2_LATLON_89A]
    size[0] = hdf_file_id[latlon_keys['lat']].shape[0]
    size[1] = hdf_file_id[latlon_keys['lon']].shape[0]
    rank = len(hdf_file_id[latlon_keys['lat']].shape)
    
    if rank < 0:
        raise Exception("No data found (rank 0 for lat. data)")

    height, width = size[0], size[1]
    read_scan_size = to_scan - from_scan + 1

    # Allocate high-res temp buffer (from 89A)
    highres_latlon = get_latlon(hdf_file_id, from_scan, to_scan, AM2_LATLON_89A)
    if highres_latlon is None:
        raise MemoryError("Failed to load 89A lat/lon data")

    # Choose the correct coefficient slot based on dataset
    channel_map = {
        AM2_LATLON_06: 0,
        AM2_LATLON_07: 1,
        AM2_LATLON_10: 2,
        AM2_LATLON_18: 3,
        AM2_LATLON_23: 4,
        AM2_LATLON_36: 5,
    }
    pnt = channel_map.get(dataset_no, 0)

    # Get coefficients
    coef_all = get_coef_from_attrs(hdf_file_id.attrs)
    
    coef = coef_all[CHANNELS[pnt]]

    # Allocate low-res output buffer (half-width)
    lats_89a, lons_89a = highres_latlon  # each is shape (n_scans, width)
    n_scans, width = lats_89a.shape
    lowres_lat = np.full((n_scans, width // 2), np.nan, dtype=np.float32)
    lowres_lon = np.full((n_scans, width // 2), np.nan, dtype=np.float32)

    # TODO: OPTIMIZE FOR SPEED; remove later
    for scan in tqdm(range(n_scans), desc="Calculating latlon grid for 36.5 GHz"):
    #for scan in range(n_scans):
        for pixel in range(0, width - 1, 2):
            # Extract lat/lon from both odd and even pixels
            info_odd = [lats_89a[scan, pixel], lons_89a[scan, pixel]]
            info_even = [lats_89a[scan, pixel + 1], lons_89a[scan, pixel + 1]]

            # Convert using the coefficient
            try:
                lat, lon = LatLon_Conversion_89G(info_odd, info_even, coef)
            except Exception as e:
                print(f"Warning: failed conversion at scan={scan} pixel={pixel}: {e}")
                import traceback
                print(traceback.format_exc())
                raise Exception("STOP")
                lat, lon = np.nan, np.nan

            lowres_lat[scan, pixel // 2] = lat
            lowres_lon[scan, pixel // 2] = lon

    return lowres_lat, lowres_lon

# === Helper functions ===

def tai93_to_utc(tai_seconds):
    TAI_EPOCH = datetime(1993, 1, 1, tzinfo=timezone.utc)
    return TAI_EPOCH + timedelta(seconds=tai_seconds - 37)  # 37s = TAI-UTC (as of 2025)

def lookup_theta(lat, month, df, freq=89):
    if USE_POPULAR_THETA:
        if freq == 89:
            return DEFAULT_THETA_89
        elif freq == 36:
            return DEFAULT_THETA_36
    else:
        lat_col = df["Lat"]
        idx = np.abs(lat_col - lat).argmin()
        print(df.iloc[idx, month])
        return df.iloc[idx, month]

def compute_pct(tb_v, tb_h, theta):
    return ((1 + theta) * tb_v) - (theta * tb_h)

def normalize(x, vmin, vmax):
    return np.clip((x - vmin) / (vmax - vmin), 0, 1)

def get_center_scan_time(lat, lon, scan_time, center_lat, center_lon,
                         points_per_scan=486, tai_utc_offset=37):
    """
    Find the scan time (UTC) closest to the image center.

    Parameters:
        lat, lon          : 1D arrays of lat/lon values (same shape)
        scan_time         : 1D array of scan times (TAI seconds since 1993-01-01)
        center_lat/lon    : float, center of the image in degrees
        points_per_scan   : int, number of data points per scan (default: 486 for 89 GHz)
        tai_utc_offset    : int, (leap) seconds to subtract from TAI to get UTC (default: 37s)

    Returns:
        utc_time (datetime) : UTC time of scan nearest to image center
        scan_index (int)             : Index into scan_time array
    """
    
    # Flatten lat/lon arrays to 1D
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()

    # Create (N, 2) array of all swath coordinates
    points = np.column_stack((lat_flat, lon_flat))  # shape: (2040*486, 2)

    # Target center point
    target = np.array([center_lat, center_lon])  # shape: (2,)

    # Compute distance to center
    dists = distance.cdist([target], points)[0]

    # Closest point index
    closest_idx = np.argmin(dists)

    # Convert flat index to scan index
    points_per_scan = lat.shape[1]  # = 486
    scan_idx = closest_idx // points_per_scan


    # Get scan time in seconds since 1993-01-01
    scan_sec = scan_time[scan_idx]

    # Convert to UTC datetime
    epoch = datetime(1993, 1, 1, tzinfo=timezone.utc)
    utc_time = epoch + timedelta(seconds=scan_sec - tai_utc_offset)

    return utc_time, scan_idx


## FOR DEV & TESTING

def save_latlon_cache(filepath, lat, lon):
    np.savez_compressed(filepath, lat=lat, lon=lon)
    print(f"Saved lat/lon cache to {filepath}")

def load_latlon_cache(filepath):
    if os.path.exists(filepath):
        data = np.load(filepath)
        print(f"Loaded lat/lon cache from {filepath}")
        return data['lat'], data['lon']
    else:
        return None, None


def plot_pct(file_path, center_lat, center_lon):

    with h5py.File(file_path, "r") as f:
        # Determine scan range (you might already know this)
        from_scan, to_scan = 0, f["Scan Time"].shape[0] - 1

        # === Read 89A/B lat/lon ===
        lat_89a, lon_89a = get_latlon(f, from_scan, to_scan, AM2_LATLON_89A)
        lat_89b, lon_89b = get_latlon(f, from_scan, to_scan, AM2_LATLON_89B)

        # === Get scan time near center ===
        scan_dt, scan_idx = get_center_scan_time(lat_89a, lon_89a, f["Scan Time"], center_lat, center_lon)

        # === Read BT data and scale ===
        tb_89av = f["Brightness Temperature (89.0GHz-A,V)"][:] * f["Brightness Temperature (89.0GHz-A,V)"].attrs['SCALE FACTOR']
        tb_89ah = f["Brightness Temperature (89.0GHz-A,H)"][:] * f["Brightness Temperature (89.0GHz-A,H)"].attrs['SCALE FACTOR']
        tb_89bv = f["Brightness Temperature (89.0GHz-B,V)"][:] * f["Brightness Temperature (89.0GHz-B,V)"].attrs['SCALE FACTOR']
        tb_89bh = f["Brightness Temperature (89.0GHz-B,H)"][:] * f["Brightness Temperature (89.0GHz-B,H)"].attrs['SCALE FACTOR']
        tb_36v = f["Brightness Temperature (36.5GHz,V)"][:] * f["Brightness Temperature (36.5GHz,V)"].attrs['SCALE FACTOR']
        tb_36h = f["Brightness Temperature (36.5GHz,H)"][:] * f["Brightness Temperature (36.5GHz,H)"].attrs['SCALE FACTOR']

        # === Read coregistration coefficients for 36GHz ===
        coef36 = get_coef_from_attrs(f.attrs)["36G"]  # you must define this helper to extract from f.attrs[]

        # === Get 36.5 GHz lat,lon from 89A lat,lon and coregistration ===
        # TODO: Optimize for speed
        # This is very slow and by far the slowest part of the program (because its a close port of the original c code)
        # for now there is a TQDM to let us know far along we are
        lat_36, lon_36 = get_latlon(f, from_scan, to_scan, AM2_LATLON_36, coef36=coef36)
        
        #cache_file = "latlon_36_cache.npz"
        #lat_36, lon_36 = load_latlon_cache(cache_file)
        #if lat_36 is None or lon_36 is None:
        #    lat_36, lon_36 = get_latlon(f, from_scan, to_scan, AM2_LATLON_36, coef36=coef36)
        #    save_latlon_cache(cache_file, lat_36, lon_36)

    # === Convert scan time to UTC & get month ===
    #scan_dt = tai93_to_utc(time_val)
    month = scan_dt.month  # 1-12

    # === Get theta for center latitude ===
    theta_89 = lookup_theta(center_lat, month, THETA89_DF, freq=89)
    theta_36 = lookup_theta(center_lat, month, THETA36_DF, freq=36)

    # === Compute PCT ===
    pct_89a = compute_pct(tb_89av, tb_89ah, theta_89)
    pct_89b = compute_pct(tb_89bv, tb_89bh, theta_89)
    pct_36 = compute_pct(tb_36v, tb_36h, theta_36)

    # === Subset within bounding box ===
    mask_89a = (
        (lat_89a >= center_lat - PLOT_RADIUS_DEG) & (lat_89a <= center_lat + PLOT_RADIUS_DEG) &
        (lon_89a >= center_lon - PLOT_RADIUS_DEG) & (lon_89a <= center_lon + PLOT_RADIUS_DEG)
    )
    # === Subset within bounding box ===
    mask_89b = (
        (lat_89b >= center_lat - PLOT_RADIUS_DEG) & (lat_89b <= center_lat + PLOT_RADIUS_DEG) &
        (lon_89b >= center_lon - PLOT_RADIUS_DEG) & (lon_89b <= center_lon + PLOT_RADIUS_DEG)
    )

    mask_36 = (
        (lat_36 >= center_lat - PLOT_RADIUS_DEG) & (lat_36 <= center_lat + PLOT_RADIUS_DEG) &
        (lon_36 >= center_lon - PLOT_RADIUS_DEG) & (lon_36 <= center_lon + PLOT_RADIUS_DEG)
    )

    target_lon_min = center_lon - PLOT_RADIUS_DEG
    target_lon_max = center_lon + PLOT_RADIUS_DEG
    target_lat_min = center_lat - PLOT_RADIUS_DEG
    target_lat_max = center_lat + PLOT_RADIUS_DEG

    for freq in [89, 36]:

        if freq == 89:
            freq_str = '89 GHz'
            # Concatenate lat/lon
            lat_all = np.concatenate([lat_89a[mask_89a], lat_89b[mask_89b]])
            lon_all = np.concatenate([lon_89a[mask_89a], lon_89b[mask_89b]])

            # Concatenate PCT and TBs
            pct_all = np.concatenate([pct_89a[mask_89a], pct_89b[mask_89b]])
            tb_v_all = np.concatenate([tb_89av[mask_89a], tb_89bv[mask_89b]])
            tb_h_all = np.concatenate([tb_89ah[mask_89a], tb_89bh[mask_89b]])

            pct_min_temp = PCT_MIN89
            pct_max_temp = PCT_MAX89
            tbv_min_temp = TBV_MIN89
            tbv_max_temp = TBV_MAX89
            tbh_min_temp = TBH_MIN89
            tbh_max_temp = TBH_MAX89
        
        elif freq == 36:
            freq_str = '36.5 GHz'
            lat_all = lat_36[mask_36]
            lon_all = lon_36[mask_36]
            
            pct_all = pct_36[mask_36]
            tb_v_all = tb_36v[mask_36]
            tb_h_all = tb_36h[mask_36]
            
            pct_min_temp = PCT_MIN36
            pct_max_temp = PCT_MAX36
            tbv_min_temp = TBV_MIN36
            tbv_max_temp = TBV_MAX36
            tbh_min_temp = TBH_MIN36
            tbh_max_temp = TBH_MAX36
        
        
        # Normalize each component
        # red is reverse tonality
        r = normalize(pct_all, pct_min_temp, pct_max_temp)       # RED/PINK from PCT (inverted) (89=RED, PINK=36.5)
        g = normalize(tb_v_all, tbv_min_temp, tbv_max_temp)      # GREEN from TB_V
        b = normalize(tb_h_all, tbh_min_temp, tbh_max_temp)      # BLUE from TB_H



        # Defunct, left for reference for plotting raw TB
        # for raw TB
        #im1 = ax.scatter(lon_89a[mask_89a], lat_89a[mask_89a], c=pct_89a[mask_89a], s=1,
        #                cmap="jet", vmin=K_MIN, vmax=K_MAX, transform=ccrs.PlateCarree())

        #im2 = ax.scatter(lon_89b[mask_89b], lat_89b[mask_89b], c=pct_89b[mask_89b], s=1,#
        #                cmap="jet", vmin=K_MIN, vmax=K_MAX, transform=ccrs.PlateCarree())
        #plt.colorbar(im, label=f"PCT 89GHz (K)")

        # Stack into RGB image (shape: [n_points, 3])
        # rgb = np.stack((r, g, b), axis=-1)


        # Define output grid
        lon_grid = np.linspace(min(lon_all), max(lon_all), 500)
        lat_grid = np.linspace(min(lat_all), max(lat_all), 500)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Interpolate each color channel separately
        r_grid = 1 - griddata((lon_all, lat_all), r, (lon_mesh, lat_mesh), method='nearest')
        g_grid = griddata((lon_all, lat_all), g, (lon_mesh, lat_mesh), method='nearest')
        b_grid = griddata((lon_all, lat_all), b, (lon_mesh, lat_mesh), method='nearest')

        # Stack to RGB image
        #rgb_grid = np.stack((r_grid, g_grid, b_grid), axis=-1)
        rgb_grid = np.clip(np.stack((r_grid, g_grid, b_grid), axis=-1), 0, 1)



        # === Plot ===
        plt.figure(figsize=(10, 8))

        # --- Your existing code for ax ---
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([center_lon - PLOT_RADIUS_DEG, center_lon + PLOT_RADIUS_DEG,
                    center_lat - PLOT_RADIUS_DEG, center_lat + PLOT_RADIUS_DEG], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

        xticks = np.arange(np.floor(lon_min), np.ceil(lon_max)+1, 1)
        yticks = np.arange(np.floor(lat_min), np.ceil(lat_max)+1, 1)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        ax.gridlines(xlocs=xticks, ylocs=yticks, color='black', linestyle='--', linewidth=0.5)

        ax.set_extent([target_lon_min, target_lon_max, target_lat_min, target_lat_max], crs=ccrs.PlateCarree())

        # In imshow:
        ax.imshow(
            rgb_grid,
            origin='lower',
            extent=[target_lon_min, target_lon_max, target_lat_min, target_lat_max],
            transform=ccrs.PlateCarree(),
            interpolation='none'  # don't blur pixels
        )

        plt.title(f"PCT {freq_str} - GCOM-W1 (JAXA)\n{scan_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        plt.tight_layout()
    
    plt.show()



# command line version
def print_usage():
    print("Usage: python3 pct.py <file_path> <center_lat> <center_lon>")
    print("  <file_path>   - Path to the .h5 file")
    print("  <center_lat>  - Center latitude (e.g., 15.5)")
    print("  <center_lon>  - Center longitude (e.g., -106.6)")

def main():
    try:
        if len(sys.argv) != 4:
            raise ValueError("Invalid number of arguments")

        file_path = sys.argv[1]
        center_lat = float(sys.argv[2])
        center_lon = float(sys.argv[3])

        plot_pct(file_path, center_lat, center_lon)

    except ValueError as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        print_usage()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        print_usage()

if __name__ == "__main__":
    main()

