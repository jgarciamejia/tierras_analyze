# tierras_analyze/analyze_thwomp.py
import argparse
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from scipy.interpolate import CubicSpline
from astropy.io import fits
import pyarrow.parquet as pq

from analyze_global import identify_target_gaia_id
from ap_phot import set_tierras_permissions, t_or_f, tierras_binner


def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-field', required=True,
                    help='Target field name (e.g. HIP47080), not the _ref field.')
    ap.add_argument('-ffname', required=False, default='flat0000',
                    help='Flattened directory name.')
    ap.add_argument('-use_nights', required=False, default=None,
                    help='Comma-separated list of dates to include, e.g. 20260416,20260518')
    ap.add_argument('-minimum_night_duration', required=False, default=0, type=float,
                    help='Min cumulative exposure time per night (hours).')
    ap.add_argument('-ap_rad', required=False, default=None, type=float,
                    help='Fix aperture radius (pixels). If None, auto-select by 5-min scatter.')
    ap.add_argument('-rp_bright_limit', required=False, default=13.0, type=float,
                    help='G_RP limit for non-target sources to receive light curves.')
    ap.add_argument('-force_reweight', required=False, default='False',
                    help='Reserved for future use.')
    args = ap.parse_args(raw_args)

    field = args.field
    ffname = args.ffname
    minimum_night_duration = args.minimum_night_duration
    ap_rad = args.ap_rad
    rp_bright_limit = args.rp_bright_limit

    fpath = '/data/tierras/flattened/'
    ref_field = f'{field}_ref'

    # ── 1. Discover target field dates ─────────────────────────────────────────
    date_list = glob(f'/data/tierras/photometry/**/{field}/{ffname}')
    date_list = np.array(sorted(date_list, key=lambda x: int(x.split('/')[4])))

    if args.use_nights is not None:
        use_nights = args.use_nights.replace(' ', '').split(',')
        keep = [j for j in range(len(date_list))
                if any(n in date_list[j] for n in use_nights)]
        date_list = date_list[keep]

    if os.path.exists(f'/data/tierras/fields/{field}/ignore_dates.txt'):
        with open(f'/data/tierras/fields/{field}/ignore_dates.txt') as f:
            ignore_dates = [ln.strip() for ln in f.readlines()]
        delete_inds = [i for i, p in enumerate(date_list) if p.split('/')[4] in ignore_dates]
        date_list = np.delete(date_list, delete_inds)

    dates = np.array([p.split('/')[4] for p in date_list])
    print(f'Found {len(dates)} nights for {field}: {list(dates)}')

    if len(date_list) == 0:
        raise RuntimeError(f'No photometry found for {field} under ffname={ffname}. '
                           f'Check that data exists at /data/tierras/photometry/**/{field}/{ffname}')

    # ── 2. Read source catalogs; find common source IDs across all nights ───────
    source_dfs, source_ids = [], []
    for path in date_list:
        source_file = glob(path + '/**sources.csv')[0]
        df = pd.read_csv(source_file)
        source_dfs.append(df)
        source_ids.append(list(df['source_id']))

    common_source_ids = np.array(source_ids[0])
    for sid_list in source_ids[1:]:
        mask = np.array([sid in sid_list for sid in common_source_ids])
        common_source_ids = common_source_ids[mask]

    # index mapping: source_inds[i][k] = column index in night i's parquet for common source k
    source_inds = []
    for df in source_dfs:
        id_to_idx = {sid: idx for idx, sid in enumerate(df['source_id'])}
        source_inds.append([id_to_idx[sid] for sid in common_source_ids if sid in id_to_idx])

    n_sources = len(common_source_ids)
    print(f'{n_sources} sources common across all nights.')

    # ── 3. Count total images and determine aperture file list ─────────────────
    # Use first night's phot files as the template for aperture sizes
    first_phot_files = sorted(
        [f for f in glob(date_list[0] + '/**phot**.parquet') if 'variable' not in f],
        key=lambda x: float(x.split('_')[-1].split('.parquet')[0])
    )
    if not first_phot_files:
        raise RuntimeError(f'No photometry files found for {field} on {dates[0]}.')

    if ap_rad is not None:
        radii = np.array([float(f.split('_')[-1].split('.parquet')[0]) for f in first_phot_files])
        df_ind = int(np.where(radii == ap_rad)[0][0])
        n_dfs = 1
        phot_files_template = [first_phot_files[df_ind]]
    else:
        n_dfs = len(first_phot_files)
        df_ind = None
        phot_files_template = first_phot_files

    n_ims = 0
    for path in date_list:
        pf = [f for f in glob(path + '/**phot**.parquet') if 'variable' not in f]
        if pf:
            n_ims += len(pq.read_table(pf[0]))

    print(f'{n_ims} total images across {len(dates)} nights, {n_dfs} aperture(s).')

    # ── 4. Identify the Tierras target star ────────────────────────────────────
    hdr = fits.open(glob(fpath + f'{dates[-1]}/{field}/{ffname}/*.fit')[0])[0].header
    targ_x_pix = hdr['CAT-X']
    targ_y_pix = hdr['CAT-Y']
    tierras_target_id = identify_target_gaia_id(
        field, source_dfs[-1], x_pix=targ_x_pix, y_pix=targ_y_pix)
    print(f'Target Gaia ID: {tierras_target_id}')

    # ── 5. Select sources for output (target + G_RP < rp_bright_limit) ────────
    targ_df_idx = np.where(source_dfs[0]['source_id'] == tierras_target_id)[0][0]
    common_rp = np.array([
        source_dfs[0].iloc[np.where(source_dfs[0]['source_id'] == sid)[0][0]]['phot_rp_mean_mag']
        for sid in common_source_ids
    ])
    targ_common_idx = np.where(common_source_ids == tierras_target_id)[0][0]
    output_source_inds = np.where(
        (common_rp < rp_bright_limit) | (common_source_ids == tierras_target_id)
    )[0]
    print(f'Will produce light curves for {len(output_source_inds)} sources '
          f'(target + G_RP < {rp_bright_limit}).')

    return  # placeholder; removed in Task 3
