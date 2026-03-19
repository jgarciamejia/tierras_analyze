
from analyze_global import main as analyze_global_main

import numpy as np 
import os 
from datetime import datetime, timedelta 
from astropy.time import Time 
import argparse
from ap_phot import t_or_f

'''
    a wrapper function to do make light curves
'''
ap = argparse.ArgumentParser()
ap.add_argument("-single_field", required=False, default='', help="If passed, run pipeline on specified field only.")
ap.add_argument("-start_field", required=False, default='', help="If you pass a name, the code will skip all targets in the target list preceding the passed field. This is just for convenience if you need to stop running the code in-person and start running in a remote session, or if the code crashes halway through.")
ap.add_argument("-force_reweight", required=False, default='False', help="If True, force the reweighting of the reference stars for every field in the target list.")
ap.add_argument('-date', required=False, default=None, help='Date of data to process. If not passed, will default to last night.' )
ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flattened directory.')
ap.add_argument('-high_priority_only', required=False, default='False', help='Whether or not to only update light curves for targets in analysis_priority_fields.txt')
ap.add_argument('-all_fields_last_week', required=False, default=False, help='Whether or not to update light curves for all fields from the last week.')


args = ap.parse_args()
single_field = args.single_field
start_field = args.start_field
force_reweight = t_or_f(args.force_reweight)
date = args.date 
ffname = args.ffname
high_priority_only = t_or_f(args.high_priority_only)
all_fields_last_week = t_or_f(args.all_fields_last_week)

# if no date was passed, grab last night's date
if date is None:
    last_night = (Time(datetime.now()) - 1).value
    date = str(last_night.year)+str(last_night.month).zfill(2)+str(last_night.day).zfill(2)

# read in priority target list 
with open('/home/ptamburo/tierras/tierras_analyze/analysis_priority_fields.txt', 'r') as f:
    priority_targets = f.readlines()
priority_targets = [i.strip() for i in priority_targets][::-1]


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

n_hi_pri = 0
if single_field == '':
    target_list = sorted(np.unique(targets))
    for i in range(len(priority_targets)):
        shift_ind = np.where([j == priority_targets[i] for j in target_list])[0]
        if len(shift_ind) == 0: # if the target is in the priority list but was not observed, skip it 
            continue
        target_list.remove(priority_targets[i])
        target_list.insert(0, priority_targets[i])
        n_hi_pri += 1
else:
    target_list = [single_field]

if start_field != '':
    ind = np.where(np.array(target_list) == start_field)[0][0]
    target_list = target_list[ind:]

# only include high priority targets
if high_priority_only:
    target_list = target_list[:n_hi_pri]

# make a target list from all fields observed in the past week
if all_fields_last_week:
    date_stop = datetime.strptime(date, '%Y%m%d')
    date_start = date_stop-timedelta(days=7)
    date = date_start
    target_list = []
    while date <= date_stop:
        date_str = date.strftime('%Y%m%d')
        if os.path.exists(f'/data/tierras/photometry/{date_str}'):
            target_list.extend(os.listdir(f'/data/tierras/photometry/{date_str}'))
        date += timedelta(days=1)
    target_list = np.unique(target_list)

print(f'Updating light curves for {len(target_list)} fields.')

for j in range(len(target_list)):
    target = target_list[j]
    if 'TEST' in target or 'TARGET' in target:
        continue
    # if target == 'TIC33743172':
    #  continue
    print(f'Making global light curves for {target} (field {j+1} of {len(target_list)})')
    args = f'-field {target} -cut_contaminated False -minimum_night_duration 0 -ffname {ffname} -force_reweight {force_reweight}'
    print(args)
    analyze_global_main(args.split())


