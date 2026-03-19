
from build_tierras_db import main as build_tierras_db_main

from datetime import datetime 
from astropy.time import Time 
import subprocess
import argparse

'''
    a wrapper function to update the database
'''
ap = argparse.ArgumentParser()
ap.add_argument("-single_field", required=False, default='', help="If passed, run pipeline on specified field only.")
ap.add_argument("-start_field", required=False, default='', help="If you pass a name, the code will skip all targets in the target list preceding the passed field. This is just for convenience if you need to stop running the code in-person and start running in a remote session, or if the code crashes halway through.")
ap.add_argument("-force_reweight", required=False, default='False', help="If True, force the reweighting of the reference stars for every field in the target list.")
ap.add_argument('-date', required=False, default=None, help='Date of data to process. If not passed, will default to last night.' )
ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flattened directory.')

args = ap.parse_args()
date = args.date 

# if no date was passed, grab last night's date
if date is None:
    last_night = (Time(datetime.now()) - 1).value
    date = str(last_night.year)+str(last_night.month).zfill(2)+str(last_night.day).zfill(2)

# part 3: update the database
print('Updating the database...')
args = f'-date {date}'
build_tierras_db_main(args.split())

email = "patrick.tamburo@cfa.harvard.edu juliana.garcia-mejia@cfa.harvard.edu"  
subject = f"[Tierras]: Completed data procesing for {date}"
command = f'echo | mutt {email} -s "{subject}"'
subprocess.run(command, shell=True, check=True)