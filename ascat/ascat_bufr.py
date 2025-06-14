# parse KNMI ASCAT .bufr (much more difficult than the netcdf format; see the notebook)
# prints off top solutions' likelihood with >= 34 kt
# has some leftover commented code from the related notebook for .nc files

import eccodes
#from eccodes import codes_bufr_new_from_file, codes_release, CodesInternalError
from eccodes import *
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import numpy as np

from datetime import datetime

### SETTINGS 

metric = False

# for example (TS Dalila 2025, NHC has genesis based partly on their own (OSPO?) processing of ASCAT wind vectors)
lon_min = -110
lon_max = -95
lat_min = 5
lat_max = 20

# some portion over land may be too exclusive for near coasts (should really use landFraction parameter instead intelligently)
qc_flags = [
    #'distance_to_gmf_too_large',
    #'data_are_redundant',
    #'no_meteorological_background_used',
    'rain_detected',
    #'not_usable_for_visualisation',
    #'small_wind_less_than_or_equal_to_3_m_s',
    #'large_wind_greater_than_30_m_s',
    #'wind_inversion_not_successful',
    'some_portion_of_wvc_is_over_ice',
    'some_portion_of_wvc_is_over_land',
    'variational_quality_control_fails',
    'knmi_quality_control_fails',
    #'product_monitoring_event_flag',
    #'product_monitoring_not_used',
    #'any_beam_noise_content_above_threshold',
    #'poor_azimuth_diversity',
    'not_enough_good_sigma0_for_wind_retrieval'
]

# 1 is the selected ambiguity (highest likelihood)
# change this to see other solutions (but be wary of likelihood)
select_ambiguity=1
file_path = "OASWC12_20250613_145400_66090_M01.bufr"

#####


# from KNMI docs
FLAG_BITMASKS = {
    'not_enough_good_sigma0_for_wind_retrieval': 2**22,
    'poor_azimuth_diversity': 2**21,
    'any_beam_noise_content_above_threshold': 2**20,
    'product_monitoring_not_used': 2**19,
    'product_monitoring_event_flag': 2**18,
    'knmi_quality_control_fails': 2**17,
    'variational_quality_control_fails': 2**16,
    'some_portion_of_wvc_is_over_land': 2**15,
    'some_portion_of_wvc_is_over_ice': 2**14,
    'wind_inversion_not_successful': 2**13,
    'large_wind_greater_than_30_m_s': 2**12,
    'small_wind_less_than_or_equal_to_3_m_s': 2**11,
    'no_meteorological_background_used': 2**8,
    'data_are_redundant': 2**7,
    'distance_to_gmf_too_large': 2**6,
    'rain_detected': 2**5,  # not documented, but assumed by position in some products
}



def print_bufr_keys(filepath):
    with open(filepath, 'rb') as f:
        all_keys = set()
        while True:
            bufr_id = eccodes.codes_bufr_new_from_file(f)
            if bufr_id is None:
                break

            eccodes.codes_set(bufr_id, "unpack", 1)  # Important to unpack BUFR descriptors

            #print("\nAvailable keys:")
            keys_iter = eccodes.codes_keys_iterator_new(bufr_id)

            while eccodes.codes_keys_iterator_next(keys_iter):
                key = eccodes.codes_keys_iterator_get_name(keys_iter)
                #print(key)
                all_keys.add(key)

            eccodes.codes_keys_iterator_delete(keys_iter)
            eccodes.codes_release(bufr_id)

        print("\n".join(sorted(all_keys)))

#print_bufr_keys("OASWC12_20250613_145400_66090_M01.bufr")
# buffer keys for reference (no delayed descriptors, i.e. #4#windSpeedAt10M, etc. for 1-4):
"""
7777
antennaBeamAzimuth
ascatExtrapolatedReferenceFunctionPresence
ascatKpEstimateQuality
ascatSatelliteOrbitAndAttitudeQuality
ascatSigma0Usability
ascatSolarArrayReflectionContamination
ascatSyntheticDataQuantity
ascatTelemetryPresenceAndQuality
ascatUseOfSyntheticData
backscatter
backscatterDistance
beamCollocation
beamIdentifier
bufrHeaderCentre
bufrHeaderSubCentre
bufrTemplate
bufrdcExpandedDescriptors
centre
compressedData
createNewData
crossTrackCellNumber
dataCategory
dataKeys
dataSubCategory
databaseIdentification
day
defaultSequence
delayedDescriptorReplicationFactor
directionOfMotionOfMovingObservingPlatform
ed
edition
estimatedErrorInSigma0At40DegreesIncidenceAngle
estimatedErrorInSlopeAt40DegreesIncidenceAngle
estimatedErrorInSurfaceSoilMoisture
expandedAbbreviations
expandedCodes
expandedCrex_scales
expandedCrex_units
expandedCrex_widths
expandedNames
expandedOriginalCodes
expandedOriginalReferences
expandedOriginalScales
expandedOriginalWidths
expandedTypes
expandedUnits
frozenLandSurfaceFraction
generatingApplication
globalDomain
heightOfAtmosphere
hour
iceAgeAParameter
iceProbability
indexOfSelectedWindVector
internationalDataSubCategory
inundationAndWetlandFraction
landFraction
latitude
lengthDescriptors
likelihoodComputedForSolution
localSectionPresent
localTablesVersionNumber
longitude
lossPerUnitLengthOfAtmosphere
masterTableNumber
masterTablesVersionNumber
masterTablesVersionNumberLatest
md5Data
md5Structure
meanSurfaceSoilMoisture
minute
modelWindDirectionAt10M
modelWindSpeedAt10M
month
numberOfSubsets
numberOfUnexpandedDescriptors
numberOfVectorAmbiguities
observedData
orbitNumber
pixelSizeOnHorizontal1
radarIncidenceAngle
radiometricResolutionNoiseValue
rainFallDetection
reservedSection3
satelliteIdentifier
satelliteInstruments
second
section1Length
section1Padding
section2Length
section3Flags
section3Length
section3Padding
section4Length
section4Padding
section5Length
sequences
slopeAt40DegreesIncidenceAngle
snowCover
softwareIdentification
soilMoistureCorrectionFlag
soilMoistureProcessingFlag
soilMoistureQuality
soilMoistureSensitivity
subCentre
surfaceSoilMoisture
tableNumber
templatesLocalDir
templatesMasterDir
topographicComplexity
totalLength
typicalCentury
typicalDate
typicalDateTime
typicalDay
typicalHour
typicalMinute
typicalMonth
typicalSecond
typicalTime
typicalYear
typicalYearOfCentury
unexpandedDescriptors
updateSequenceNumber
windDirectionAt10M
windSpeedAt10M
windVectorCellQuality
year
"""

def read_bufr_file(filepath, select_ambiguity=1):
    lat_all = []
    lon_all = []
    ws_all = []
    wd_all = []
    qc_all = []
    time_flat_all = []
    le_all = []
    satid = ''

    with open(filepath, "rb") as f:
        i = 0
        while True:
            i+=1
            bufr_id = codes_bufr_new_from_file(f)
            if bufr_id is None:
                break
        
            missing = -1e100
            #codes_set(bufr_id, "bitmapPresent", 1)
            codes_set(bufr_id, "missingValue", missing)
            codes_set(bufr_id, "unpack", 1)

            num_subsets = codes_get(bufr_id, "numberOfSubsets")
            # Number of wind ambiguities (replicated wind vectors)
            try:
                n_rep = codes_get(bufr_id, "delayedDescriptorReplicationFactor")
            except CodesInternalError:
                n_rep = 1  # No replication in this subset

            lat  = codes_get_array(bufr_id, "latitude")
            if len(lat) == 1:
                lat = np.full(num_subsets, lat[0])
            lon  = codes_get_array(bufr_id, "longitude")
            if len(lon) == 1:
                lon = np.full(num_subsets, lon[0])
            
            qc  = codes_get_array(bufr_id, "windVectorCellQuality")
            if len(qc) == 1:
                qc = np.full(num_subsets, qc[0])
            
            times_flat = broadcast_time_fields(bufr_id, num_subsets)
            
            satid = codes_get(bufr_id, 'satelliteIdentifier')
            
            for j in range(n_rep):
                if select_ambiguity != j+1:
                    continue
                try:
                    ws = codes_get_array(bufr_id, f"#{j+1}#windSpeedAt10M")
                    wd = codes_get_array(bufr_id, f"#{j+1}#windDirectionAt10M")
                    le = codes_get_array(bufr_id, f"#{j+1}#likelihoodComputedForSolution")
                    
                    ws[ws == missing] = np.nan
                    wd[wd == missing] = np.nan
                    le[le == missing] = np.nan
                    
                    if len(ws) == 0:
                        ws = np.full(num_subsets, np.nan)
                    elif len(ws) == 1:
                        ws = np.full(num_subsets, ws[0])
                        
                    if len(wd) == 0:
                        wd = np.full(num_subsets, np.nan)
                    elif len(wd) == 1:
                        wd = np.full(num_subsets, wd[0])
                    
                    if len(le) == 0:
                        le = np.full(num_subsets, np.nan)
                    elif len(le) == 1:
                        le = np.full(num_subsets, np.pow(10, le[0]))
                    else:
                        le = np.pow(10, le)
                        
                    lat_all.append(lat)
                    lon_all.append(lon)
                    ws_all.append(ws)
                    wd_all.append(wd)
                    qc_all.append(qc)
                    le_all.append(le)
                    time_flat_all.append(times_flat)
                except CodesInternalError:
                    import traceback
                    print(f"#{j+1}#windSpeedAt10M")
                    print("Error on", i, j)
                    traceback.print_exc()
                    exit(1)
                    continue
                
                break

            codes_release(bufr_id)

    # Concatenate everything (flattening into 1d arrays)
    lat_all = np.concatenate(lat_all)
    lon_all = np.concatenate(lon_all)
    ws_all = np.concatenate(ws_all)
    wd_all = np.concatenate(wd_all)
    qc_all = np.concatenate(qc_all)
    time_flat_all = np.concatenate(time_flat_all)

    return {
        'latitude': lat_all,
        'longitude': lon_all,
        'windSpeedAt10M': ws_all,
        'windDirectionAt10M': wd_all,
        'windVectorCellQuality': qc_all,
        'time_flat': time_flat_all,
        'le': le_all,
        'satelliteIdentifier': satid
    }



def mask_bufr_qc(wind_speed, qc_values, active_flags):
    """
    Masks wind_speed using the bitmask values from qc_values based on selected flags.
    Missing values (all bits set) are always masked.
    """
    wind_speed_masked = wind_speed.copy()
    mask = np.zeros_like(qc_values, dtype=bool)

    # Always mask missing values (24 bits set)
    mask |= (qc_values == 16777215)

    # Apply user-selected QC flags
    for flag in active_flags:
        bitmask = FLAG_BITMASKS.get(flag)
        if bitmask is not None:
            mask |= (qc_values & bitmask) != 0
        else:
            print(f"Warning: Flag '{flag}' not recognized.")

    wind_speed_masked[mask] = np.nan
    return wind_speed_masked


def mask_wind_speed(qc, qc_flags, flag_masks, flag_meanings, wind_speed):
    wind_speed_masked = wind_speed.copy()
    # convert the string to a list
    flag_meanings = flag_meanings.split(' ')
    for flag_meaning, flag_mask in zip(flag_meanings, flag_masks):
        if flag_meaning in qc_flags:
            wind_speed_masked[np.where((qc & flag_mask) != 0)] = np.nan
    return wind_speed_masked



def plot_ascat(file_path, metric, lon_min, lon_max, lat_min, lat_max, qc_flags, select_ambiguity=1):

    if not metric:
        # Define wind speed ranges in kt
        ranges = [0, 12, 25, 34, 50, 64, 100, 120, 200]
    else:
        # in m/s
        ranges = [0, 6.17, 12.85, 17.42, 25.93, 32.87, 51.44, 61.72, 102.96]

    # Create a PlateCarree projection
    proj = ccrs.PlateCarree()

    # Define custom color map
    colors = [
        (0.47, 0.53, 1),  #  Blue
        (0.5, 0.9, 0.5),  # Light Green
        (1, 0.8, 0),  # Light Yellow
        (1, 0, 0),  # Red
        (0.5, 0, 0.5),  # Purple
        (1, 0.5, 0.7),  # Light Pink
        (0.5, 0.5, 0.5),  # Grey
        (0, 0.5, 0.5),  # Teal
        (0, 0.2, 0.2)  # Dark Teal/Black
    ]

    # Create color map and norm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(ranges, len(colors))

    fig = plt.figure(figsize=(8, 12))

    # Create a GeoAxes object
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # process data

    data = read_bufr_file(file_path, select_ambiguity=select_ambiguity)

    wind_speed = data['windSpeedAt10M']
    lats = data['latitude']
    lons = data['longitude']
    le = data['le']
    qc = data['windVectorCellQuality']
    ambiguities_per_wvc = len(wind_speed) // len(lats)  # Should be 4
    lats = np.repeat(lats, ambiguities_per_wvc)
    lons = np.repeat(lons, ambiguities_per_wvc)
    qc = np.repeat(qc, ambiguities_per_wvc)


    # Wrap longitudes to [-180, 180] range
    lons_wrapped = np.where(lons > 180, lons - 360, lons)

    if not metric:
        # Convert wind speed values from m/s to kt
        wind_speed = wind_speed * 1.94384

    le = np.ma.masked_invalid(le)

    # Mask invalid wind speed values
    wind_speed_masked = np.ma.masked_invalid(wind_speed)

    time_flat = data['time_flat']
    wind_speed_masked = mask_bufr_qc(wind_speed, qc, qc_flags)

    
    # Find the indices of the points within the extent
    lons_wrapped_flat = lons_wrapped.flatten()
    lats_flat = lats.flatten()
    indices = np.where((lons_wrapped_flat >= ax.get_xlim()[0]) & (lons_wrapped_flat <= ax.get_xlim()[1]) & (lats_flat >= ax.get_ylim()[0]) & (lats_flat <= ax.get_ylim()[1]))
    
    vmax = np.nanmax(wind_speed_masked.flatten()[indices])
    vmax_le = le.flatten()[indices][np.nanargmax(wind_speed_masked.flatten()[indices])]

    wind_speed_area = wind_speed_masked.flatten()[indices]
    le_area = le.flatten()[indices]

    # Check if there are any wind speeds >= 34
    if np.any(wind_speed_area >= 34):
        mask = (wind_speed_area >= 34) & (~np.isnan(wind_speed_area))
        indices_of_interest = np.where(mask)[0]
        vmax_values = wind_speed_area[indices_of_interest]
        le_values = le_area[indices_of_interest]
    else:
        # Get the top 20 vmax values
        sorted_indices = np.argsort(-wind_speed_area)
        top_20_indices = sorted_indices[:20]
        vmax_values = wind_speed_area[top_20_indices]
        le_values = le_area[top_20_indices]

    # Print the vmax and likelihood values
    print("Top likelihoods for points >= 34")
    for vmax, le_value in zip(vmax_values, le_values):
        print(f"vmax: {vmax:3.1f}, Likelihood: {le_value:1.3f}")
    
    # will return an error if empty
    time_min = np.min(time_flat[indices])
    time_max = np.max(time_flat[indices])

    # Get the start and end indices for the swath extent
    start_date = time_min
    end_date = time_max

    # Format the dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')


    im = ax.scatter(lons_wrapped.flatten(), lats.flatten(), c=wind_speed_masked.flatten(), cmap=cmap, norm=norm, transform=proj, s=2)

    cax = fig.add_axes([0.1, 0.1, 0.8, 0.01])

    # Create color bar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax,
        orientation="horizontal",
        ticks=ranges[:-1],
    )

    cbar.ax.tick_params(axis='x', which='major', pad=5)
    cbar.ax.set_xticklabels(cbar.ax.get_xticks(), ha='center')

    if not metric:
        unit = 'kt'
        cbar.set_label('Wind Speed (knots)')
    else:
        unit = 'm/s'
        cbar.set_label('Wind Speed (m/s)')

    # grid
    ax.grid(which='major', axis='both', linestyle='-', linewidth=0.1, color='gray', alpha=0.8)

    #1 degree grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('{:.0f}'))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('{:.0f}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:.0f}'.format(y)))

    # get the short title of the source data
    #short = nc_file.title_short_name
    mapping = {3: 'METOP-B', 5: 'METOP-C'}
    short = mapping[data['satelliteIdentifier']]
    ax.set_title(f'{short}\nSwathe: {start_date_str} - {end_date_str}\nVMax: {vmax:3.1f} {unit} (Likelihood: {vmax_le:3.3f})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # for some reason this doesn't work outside a notebook
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    fig.text(x=1.0, y=0.8, s="QC Flags:\n" + "\n".join(qc_flags), ha='left', va='top', fontsize=12)

    # Show the plot
    plt.show()


def broadcast_time_fields(bufr_id, length):
    def broadcast(key):
        vals = codes_get_array(bufr_id, key)
        if vals is None:
            return np.full(length, np.nan)
        elif len(vals) == 1:
            return np.full(length, vals[0])
        else:
            return np.array(vals)

    year   = broadcast("year")
    month  = broadcast("month")
    day    = broadcast("day")
    hour   = broadcast("hour")
    minute = broadcast("minute")
    second = broadcast("second")

    times = []
    for y, m, d, h, mi, s in zip(year, month, day, hour, minute, second):
        try:
            times.append(datetime(int(y), int(m), int(d), int(h), int(mi), int(s)))
        except Exception:
            times.append(None)
    return np.array(times)


plot_ascat(file_path, metric, lon_min, lon_max, lat_min, lat_max, qc_flags, select_ambiguity=select_ambiguity)