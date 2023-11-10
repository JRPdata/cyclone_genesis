#!/bin/bash

# U and V files for 850 mb
grib_u_file="gfs.t00z.pgrb2.0p25.f072_ugrd_850_mb.grib2"
grib_v_file="gfs.t00z.pgrb2.0p25.f072_vgrd_850_mb.grib2"
# file name to store output path (not all characters accepted)
output_vor_grib_file="gfs.t00z.pgrb2.0p25.f072_vor_850_mb.grib2"

# temporary name to combine the grib files into a gemfile
output_gem_file="gfsuv.grd"

rm $output_gem_file 2>/dev/null

dcgrib2 $output_gem_file < $grib_u_file

dcgrib2 $output_gem_file < $grib_v_file

gdgrib <<EOF
GDATTIM  = LAST
GLEVEL   = 850
GVCORD   = PRES
GFUNC    = VOR(VECR(UREL,VREL))
GDFILE   = $output_gem_file
GAREA    = grid
PROJ     = ced/0;0;0
PDSVAL   = VOR@850
PRECSN   = D/5
GBTBLS   = wmogrib2.tbl
GBFILE   = $output_vor_grib_file
LIST
RUN


EOF
