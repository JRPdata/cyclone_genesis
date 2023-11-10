#!/bin/bash
#doesn't work... produces corrupted file

# U and V files for 850 mb
grib_u_file="gfs.t00z.pgrb2.0p25.f072_ugrd_850_mb.grib2"
grib_v_file="gfs.t00z.pgrb2.0p25.f072_vgrd_850_mb.grib2"
# file name to store output path (not all characters accepted)
output_vor_grib_file="gfs.t00z.pgrb2.0p25.f072_vor_850_mb.grib2"

# temporary name to combine the grib files into a gemfile
output_gem_file="gfsuv.grd"

rm $output_gem_file 2>/dev/null

## NEED TO CREATE A TEMPORARY GEM (GRD) FILE BY CONVERTING THE GRIB2 FILES THAT CONTAIN UGRD AND VGRD

# Specify the parameters as per your example
nagrib2 <<EOF
GBFILE   =  $grib_u_file
GDOUTF   =  $output_gem_file
PROJ     =  ced/0;0;0
GRDAREA  =  -90.;0.;90.;-0.25
KXKY     =  1440;721
MAXGRD   =  3000
CPYFIL   =
GAREA    =  grid
OUTPUT   =  tf
G2TBLS   =  g2varsncepwww.tbl
G2DIAG   =
LIST
RUN
EOF

nagrib2 <<EOF
# Specify the parameters as per your example
GBFILE   =  $grib_v_file
GDOUTF   =  $output_gem_file
PROJ     =  ced/0;0;0
GRDAREA  =  -90.;0.;90.;-0.25
KXKY     =  1440;721
MAXGRD   =  3000
CPYFIL   =
GAREA    =  grid
OUTPUT   =  tf
G2TBLS   =  g2varsncepwww.tbl
G2DIAG   =
LIST
RUN
EOF


gdgrib <<EOF
GDATTIM  = LAST
GLEVEL   = 850
## SAVE THE DATA TO A GRIB FILE WITH 5 DECIMALS PRECISION
GVCORD   = PRES
GFUNC    = VOR(VECR(UGRD,VGRD))
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
