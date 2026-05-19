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

    # placeholder so the file is importable; rest of main() added in later tasks
    return
