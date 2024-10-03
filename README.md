# cyclone_genesis

DO NOT USE

EXPERIMENTAL

A collection of programs related to tropical cyclone genesis. (See requirements.txt for the list of main programs that are not deprecated.)

A viewer (tcviewer.py) is useful for viewing official tracks on a PC (fullscreen) as well as the genesis tracks and ensembles.

ASCAT pass prediction of tropical cyclones (pred_ascat_passes.py). See comments in code for details. Demo here: [http://jrpdata.free.nf/ascatpred](http://jrpdata.free.nf/ascatpred)

Find tropical cyclones (potential/TC genesis and active) from global models (auto_update_disturbances_parallel.py and auto_update_cyclones.py). See references at top of code.

A processor (to a database with json data) (auto_update_cyclones_tcgen.py) of public TC genesis tracker data from ensembles: NCEP (GEPS (CMC ensemble), FNMOC (NAV ensemble), GEFS (GFS ensemble)) and ECMWF (EPS (ECM ensemble))

A downloader to download global model data and tc genesis tracker data from various sources (download.py).

## Notes

Note: Our own version of a tracker (based on methodology similar to FSU) from the global model data uses thresholds from FSU but for higher resolution models so it is more sensitive.

See requirements.txt for some of the requirements. There are external dependencies (to python) as well, see download.py for some of them

Note: UKMET was not added to the download programs and the various scripts, but it should work, provided split gribs. It costs too much money to obtain the data.

## 2024-07-29:

Updated which programs are deprecated.

Now has a viewer (tcviewer) for tropical cyclone tracks (both official and genesis tracks from own tracker and processed NCEP/ECMWF genesis tracker data). It is still very much a work in progress. Most of the features are accessible by keyboard shortcuts.
