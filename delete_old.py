#!/usr/bin/env python3
setting_hours_old = 48
# add to crontab, crontab line:
# 0 0,12 * * * /usr/bin/python3 /home/db/Documents/JRPdata/cyclone-genesis/delete_old.py >> /home/db/Documents/JRPdata/cyclone-genesis/delete_old.log 2>&1

model_data_folders_by_model_name = {
    'GFS': '/home/db/metview/JRPdata/globalmodeldata/gfs',
    'ECM': '/home/db/metview/JRPdata/globalmodeldata/ecm',
    'CMC': '/home/db/metview/JRPdata/globalmodeldata/cmc',
    'NAV': '/home/db/metview/JRPdata/globalmodeldata/nav'
}

import os
import sys
from datetime import datetime, timedelta

def delete_old_files(folder_path, hours_old):
    try:
        # Validate input folder path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")

        # Validate input hours_old
        if not isinstance(hours_old, (int, float)) or hours_old < 0:
            raise ValueError("Please provide a non-negative numeric value for setting for hours old.")

        # Get the current time
        current_time = datetime.now()

        # Iterate over files in the folder and its subfolders
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".grib2"):
                    file_path = os.path.join(root, filename)
                    try:
                        # Get the last modification time of the file
                        last_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                        # Calculate the age of the file in hours
                        age_in_hours = (current_time - last_modified_time).total_seconds() / 3600

                        # Check if the file is older than the specified hours_old
                        if age_in_hours > hours_old:
                            print(f"Deleting old file: {file_path}")
                            os.remove(file_path)
                    except:
                        pass

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Replace '/your/folder/path' and 24 with your desired folder path and hours_old value
    for model_name, folder_path in model_data_folders_by_model_name.items():
        try:
            delete_old_files(folder_path, setting_hours_old)
        except:
            pass
