from ap_phot import main as ap_phot_main
from analyze_global import main as analyze_global_main
from build_tierras_db import main as build_tierras_db_main

import numpy as np 
import os 
from datetime import datetime 
from astropy.time import Time 
import subprocess
import argparse
from ap_phot import t_or_f

'''
    a wrapper function to do photometry, make light curves, and update the database with last night's data
'''
ap = argparse.ArgumentParser()
ap.add_argument("-single_field", required=False, default='', help="If passed, run pipeline on specified field only.")
ap.add_argument("-start_field", required=False, default='', help="If you pass a name, the code will skip all targets in the target list preceding the passed field. This is just for convenience if you need to stop running the code in-person and start running in a remote session, or if the code crashes halway through.")
ap.add_argument("-skip_photometry", required=False, default='False', help="If True, skip photometry step (for testing)")
ap.add_argument("-force_reweight", required=False, default='False', help="If True, force the reweighting of the reference stars for every field in the target list.")

args = ap.parse_args()
single_field = args.single_field
start_field = args.start_field
skip_photometry = t_or_f(args.skip_photometry)
force_reweight = t_or_f(args.force_reweight)

last_night = (Time(datetime.now()) - 1).value
date = str(last_night.year)+str(last_night.month).zfill(2)+str(last_night.day).zfill(2)

# specify the date list
ffname = 'flat0000'
phot_type = 'fixed'

# read in priority target list 
with open('analysis_priority_fields.txt', 'r') as f:
    priority_targets = f.readlines()
priority_targets = [i.strip() for i in priority_targets][::-1]

if not skip_photometry:
    # part 1: do photometry 
    print('Doing photometry...')
    if phot_type == 'fixed':
        ap_radii = np.arange(5,21)
    elif phot_type == 'variable':
        ap_radii = np.arange(0.5, 1.6, 0.1)


    if single_field == '':
        target_list = sorted(os.listdir(f'/data/tierras/flattened/{date}'))
        # move any fields in analysis_priority_fields.txt to the front of the list 
        
        for i in range(len(priority_targets)):
            shift_ind = np.where([j == priority_targets[i] for j in target_list])[0]
            if len(shift_ind) == 0: # if the target is in the priority list but was not observed, skip it
                continue 
            target_list.remove(priority_targets[i])
            target_list.insert(0, priority_targets[i])
    else:
        target_list = [single_field]

    if start_field != '':
        ind = np.where(np.array(target_list) == start_field)[0][0]
        target_list = target_list[ind:]

    for j in range(len(target_list)):
        target = target_list[j]
        if target == 'TARGET' or target == 'TARGET_red':
            continue
        if 'TEST' in target:
            continue
        # TODO: how to automatically set the rp_mag_limit for really faint/bright targets?
        if target == 'POI-2':
            rp_mag_limit = 17.06
        if target == 'HD60779':
            rp_mag_limit = 14
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

# regenerate the target list and do any necessary priority re-sorting
# not necessarily all the targets will have had photometry done on them so it needs to be regenerated 
    #e.g. if all the images were way off the desired guiding position

if single_field == '':
    target_list = sorted(np.unique(targets))
    for i in range(len(priority_targets)):
        shift_ind = np.where([j == priority_targets[i] for j in target_list])[0]
        if len(shift_ind) == 0: # if the target is in the priority list but was not observed, skip it 
            continue
        target_list.remove(priority_targets[i])
        target_list.insert(0, priority_targets[i])
else:
    target_list = [single_field]

print(f'Found {len(target_list)} unique targets across the given date list.')

# if skip_photometry:
    # target_list = target_list[targ_cut:]
    # print(' YOU NEED TO REMOVE THE ABOVE LINE WHEN YOURE DONE TESTING ')
    # breakpoint() 

if start_field != '':
    ind = np.where(np.array(target_list) == start_field)[0][0]
    target_list = target_list[ind:]

for j in range(len(target_list)):
    target = target_list[j]
    if 'TEST' in target or 'TARGET' in target:
        continue
    # if target == 'TIC33743172':
    #  continue
    print(f'Making global light curves for {target} (field {j+1} of {len(target_list)})')
    args = f'-field {target} -SAME False -cut_contaminated False -minimum_night_duration 0 -ffname {ffname} -force_reweight {force_reweight}'
    print(args)
    analyze_global_main(args.split())

# part 3: update the database
print('Updating the database...')
args = f'-date {date}'
build_tierras_db_main(args.split())

email = "patrick.tamburo@cfa.harvard.edu juliana.garcia-mejia@cfa.harvard.edu"  
subject = f"[Tierras]: Completed data procesing for {date}"
command = f'echo | mutt {email} -s "{subject}"'
subprocess.run(command, shell=True, check=True)
