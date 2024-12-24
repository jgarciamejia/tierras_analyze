from ap_phot import main as ap_phot_main
from analyze_global import main as analyze_global_main
from build_tierras_db import main as build_tierras_db_main

import numpy as np 
import os 
from datetime import datetime 
from astropy.time import Time 
import subprocess

'''
    a wrapper function to do photometry, make light curves, and update the database with last night's data
'''

last_night = (Time(datetime.now()) - 1).value
date = str(last_night.year)+str(last_night.month).zfill(2)+str(last_night.day).zfill(2)

# specify the date list
ffname = 'flat0000'
phot_type = 'fixed'

# part 1: do photometry 
print('Doing photometry...')
if phot_type == 'fixed':
    ap_radii = np.arange(5,21)
elif phot_type == 'variable':
    ap_radii = np.arange(0.5, 1.6, 0.1)

target_list = sorted(os.listdir(f'/data/tierras/flattened/{date}'))
for j in range(len(target_list)):
    target = target_list[j]
    if target == 'TARGET' or target == 'TARGET_red':
        continue
    if 'TEST' in target:
        continue
    if target == 'POI-2':
        rp_mag_limit = 17.06
    else:
        rp_mag_limit = 17.00
    args = f'-target {target} -date {date} -ffname {ffname} -rp_mag_limit {rp_mag_limit} -ap_radii {" ".join(map(str,ap_radii))} -phot_type {phot_type} -plot_source_detection False'
    print(args)
    ap_phot_main(args.split())

# part 2: make global light curves
print('Making light curves...')
targets = []
try:
    targets.extend(sorted(os.listdir(f'/data/tierras/photometry/{date}')))
except:
    print(f'No photometry directories found on {date}...')

target_list = sorted(np.unique(targets))

print(f'Found {len(target_list)} unique targets across the given date list.')

for j in range(len(target_list)):
    target = target_list[j]
    if 'TEST' in target or 'TARGET' in target:
        continue
    # if target == 'TIC33743172':
    #  continue
    print(f'Making global light curves for {target} (field {j+1} of {len(target_list)})')
    args = f'-field {target} -SAME False -cut_contaminated False -minimum_night_duration 0 -ffname {ffname}'
    print(args)
    analyze_global_main(args.split())

# part 3: update the database
print('Updating the database...')
args = f'-date {date}'
build_tierras_db_main(args.split())

email = "patrick.tamburo@cfa.harvard.edu"  # Replace with your email address
subject = f"Completed data procesing for {date}"
command = f'echo | mutt {email} -s "{subject}"'
subprocess.run(command, shell=True, check=True)
