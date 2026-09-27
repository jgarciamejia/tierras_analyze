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
import matplotlib.pyplot as plt 
plt.ion()
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import celerite2
from celerite2 import terms

from analyze_global import identify_target_gaia_id
from ap_phot import set_tierras_permissions, t_or_f, tierras_binner

def neg_log_like(params, t_train, y_train, sigma_train):
    log_amp, log_tau, log_jitter = params
    amp, tau, jitter = np.exp(log_amp), np.exp(log_tau), np.exp(log_jitter)

    kernel = terms.Matern32Term(sigma=amp, rho=tau)
    gp = celerite2.GaussianProcess(kernel, mean=np.mean(y_train))

    # add jitter in quadrature to the known per-point errors
    yerr_eff = np.sqrt(sigma_train**2 + jitter**2)

    try:
        gp.compute(t_train, yerr=yerr_eff)
    except celerite2.driver.LinAlgError:
        return 1e10

    return -gp.log_likelihood(y_train)

def fit_gp(t_train, y_train, sigma_train, n_restarts=25, seed=0):
    rng = np.random.default_rng(seed)
    baseline_span = t_train.max() - t_train.min()
    median_cadence = np.median(np.diff(t_train))

    amp_guess = np.std(y_train)
    amp_upper = 5*amp_guess
    err_median = np.median(sigma_train)
    jitter_upper = err_median

    tau_lower = 1/24
    tau_upper = 150

    bounds = [
        (np.log(1e-6), np.log(amp_upper)),           # log_amp
        (np.log(tau_lower), np.log(tau_upper)),      # log_tau
        (np.log(1e-7), np.log(jitter_upper)),        # log_jitter
    ]

    best = None
    for _ in range(n_restarts):
        x0 = [
            np.log(rng.uniform(1e-4, amp_upper * 0.5)),
            np.log(rng.uniform(tau_lower, tau_upper)),
            np.log(rng.uniform(1e-7, jitter_upper * 0.5)),
        ]
        res = minimize(
            neg_log_like, x0,
            args=(t_train, y_train, sigma_train),
            method='L-BFGS-B', bounds=bounds
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res

    return best, bounds

def predict_mean_std(t_star, t_train, y_train, sigma_train, params):
    log_amp, log_tau, log_jitter = params
    amp, tau, jitter = np.exp(log_amp), np.exp(log_tau), np.exp(log_jitter)

    kernel = terms.Matern32Term(sigma=amp, rho=tau)
    gp = celerite2.GaussianProcess(kernel, mean=np.mean(y_train))
    gp.compute(t_train, yerr=sigma_train)

    mean_pred, var_pred = gp.predict(y_train, t=t_star, return_var=True)
    return mean_pred, np.sqrt(var_pred)

def predict_full_cov(t_star, t_train, y_train, sigma_train, params):
    log_amp, log_tau, log_jitter = params
    amp, tau, jitter = np.exp(log_amp), np.exp(log_tau), np.exp(log_jitter)

    kernel = terms.Matern32Term(sigma=amp, rho=tau)
    gp = celerite2.GaussianProcess(kernel, mean=np.mean(y_train))
    gp.compute(t_train, yerr=sigma_train)

    mean_pred, cov_pred = gp.predict(y_train, t=t_star, return_cov=True)
    return mean_pred, cov_pred

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
    ap.add_argument('-force_reweight', required=False, default='False',
                    help='Reserved for future use.')
    args = ap.parse_args(raw_args)

    field = args.field
    ffname = args.ffname
    minimum_night_duration = args.minimum_night_duration
    ap_rad = args.ap_rad

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
    else:
        n_dfs = len(first_phot_files)
        df_ind = None

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

    # ── 5. Select sources for output (target only) ────────────────────────────
    targ_common_idx = np.where(common_source_ids == tierras_target_id)[0][0]
    output_source_inds = np.array([targ_common_idx])
    print(f'Will produce light curve for target (Gaia ID {tierras_target_id}).')

    # ── 6. Allocate target photometry arrays ───────────────────────────────────
    ancillary_cols = ['Filename', 'BJD TDB', 'Airmass', 'Exposure Time',
                      'HA', 'Dome Humid', 'FWHM X', 'FWHM Y', 'WCS Flag']

    times            = np.zeros(n_ims, dtype='float64')
    airmasses        = np.zeros(n_ims, dtype='float16')
    exposure_times   = np.zeros(n_ims, dtype='float16')
    filenames        = np.empty(n_ims, dtype=object)
    ha               = np.zeros(n_ims, dtype='float16')
    humidity         = np.zeros(n_ims, dtype='float16')
    fwhm_x           = np.zeros(n_ims, dtype='float16')
    fwhm_y           = np.zeros(n_ims, dtype='float16')
    flux             = np.zeros((n_dfs, n_ims, n_sources), dtype='float32')
    flux_err         = np.zeros_like(flux)
    non_linear_flags = np.zeros_like(flux, dtype='bool')
    saturated_flags  = np.zeros_like(flux, dtype='bool')
    x_pos            = np.zeros((n_ims, n_sources), dtype='float32')
    y_pos            = np.zeros_like(x_pos)
    sky              = np.zeros_like(x_pos)
    wcs_flags        = np.zeros(n_ims, dtype='bool')

    # times_list holds VIEWS into times[] so subtracting x_offset later updates them in-place
    times_list = []
    start = 0
    t1 = time.time()

    for i, path in enumerate(date_list):
        print(f'Reading target photometry from {dates[i]} ({i+1}/{len(date_list)}).')
        phot_files = sorted(
            [f for f in glob(path + '/**phot**.parquet') if 'variable' not in f],
            key=lambda x: float(x.split('_')[-1].split('.parquet')[0])
        )
        if not phot_files:
            continue

        ancillary_file = glob(path + '/**ancillary**.parquet')[0]
        ancillary_tab  = pq.read_table(ancillary_file, columns=ancillary_cols)

        use_cols = []
        for s in source_inds[i]:
            p = f'S{s}'
            use_cols.extend([f'{p} Source-Sky', f'{p} Source-Sky Err',
                             f'{p} NL Flag',    f'{p} Sat Flag',
                             f'{p} Sky',        f'{p} X', f'{p} Y'])

        for j in range(n_dfs): # replace with n_dfs
            file_idx = df_ind if df_ind is not None else j
            data_tab = pq.read_table(phot_files[file_idx], columns=use_cols, memory_map=True)
            stop = start + len(data_tab)

            # ancillary data only filled on j==0 to avoid duplicate writes
            if j == 0:
                times[start:stop]          = np.array(ancillary_tab['BJD TDB'])
                times_list.append(times[start:stop])  
                airmasses[start:stop]      = np.array(ancillary_tab['Airmass'])
                exposure_times[start:stop] = np.array(ancillary_tab['Exposure Time'])
                filenames[start:stop]      = np.array(ancillary_tab['Filename'])
                ha[start:stop]             = np.array(ancillary_tab['HA'])
                humidity[start:stop]       = np.array(ancillary_tab['Dome Humid'])
                fwhm_x[start:stop]         = np.array(ancillary_tab['FWHM X'])
                fwhm_y[start:stop]         = np.array(ancillary_tab['FWHM Y'])
                wcs_flags[start:stop]      = np.array(ancillary_tab['WCS Flag'])

            for k in range(n_sources):
                s = source_inds[i][k]
                flux[j, start:stop, k]             = np.array(data_tab[f'S{s} Source-Sky'])
                flux_err[j, start:stop, k]         = np.array(data_tab[f'S{s} Source-Sky Err'])
                non_linear_flags[j, start:stop, k] = np.array(data_tab[f'S{s} NL Flag'])
                saturated_flags[j, start:stop, k]  = np.array(data_tab[f'S{s} Sat Flag'])
                if j == 0:
                    x_pos[start:stop, k] = np.array(data_tab[f'S{s} X'])
                    y_pos[start:stop, k] = np.array(data_tab[f'S{s} Y'])
                    sky[start:stop, k]   = np.array(data_tab[f'S{s} Sky'])

        start = stop  # advance after all apertures for this night

    print(f'Target read-in: {time.time()-t1:.1f}s')
    print(f'times[0]={times[0]:.6f}, times[-1]={times[-1]:.6f}, '
          f'flux[0,0,0]={flux[0,0,0]:.1f}')
    
    # write out a global ancillary .csv 
    global_ancillary_path = f'/data/tierras/fields/{field}/global_ancillary_data.csv'
    global_ancillary_data = pd.DataFrame(np.array([filenames, times, exposure_times, airmasses, ha, humidity, fwhm_x, fwhm_y, wcs_flags]).T, columns=['Filename', 'BJD TDB', 'Exposure Time', 'Airmass', 'Hour Angle', 'Humidity', 'FWHM X', 'FWHM Y', 'WCS Flag'])	
    global_ancillary_data.to_csv(global_ancillary_path, index=0)
    
    # ── 7. Read reference field photometry ─────────────────────────────────────
    ref_date_list = glob(f'/data/tierras/photometry/**/{ref_field}/{ffname}')
    ref_date_list = np.array(sorted(ref_date_list, key=lambda x: int(x.split('/')[4])))
    ref_dates = np.array([p.split('/')[4] for p in ref_date_list])

    if len(ref_date_list) == 0:
        raise RuntimeError(
            f'No photometry found for reference field {ref_field} under ffname={ffname}. '
            f'Run ap_phot on {ref_field} before analyze_thwomp.')

    if os.path.exists(f'/data/tierras/fields/{ref_field}/ignore_dates.txt'):
        with open(f'/data/tierras/fields/{ref_field}/ignore_dates.txt') as f:
            ref_ignore_dates = [ln.strip() for ln in f.readlines()]
        ref_delete_inds = [i for i, p in enumerate(ref_date_list)
                           if p.split('/')[4] in ref_ignore_dates]
        ref_date_list = np.delete(ref_date_list, ref_delete_inds)
        ref_dates = np.array([p.split('/')[4] for p in ref_date_list])

    print(f'Found {len(ref_dates)} nights for {ref_field}.')

    # source catalog for ref field
    ref_source_dfs, ref_source_ids = [], []
    for path in ref_date_list:
        source_file = glob(path + '/**sources.csv')[0]
        df = pd.read_csv(source_file)
        ref_source_dfs.append(df)
        ref_source_ids.append(list(df['source_id']))

    common_source_ids_ref = np.array(ref_source_ids[0])
    for sid_list in ref_source_ids[1:]:
        mask_ref = np.array([sid in sid_list for sid in common_source_ids_ref])
        common_source_ids_ref = common_source_ids_ref[mask_ref]

    ref_source_inds = []
    for df in ref_source_dfs:
        id_to_idx = {sid: idx for idx, sid in enumerate(df['source_id'])}
        ref_source_inds.append([id_to_idx[sid] for sid in common_source_ids_ref])

    n_sources_ref = len(common_source_ids_ref)

    # count ref images and determine n_dfs_ref
    n_ims_ref = 0
    for path in ref_date_list:
        pf = [f for f in glob(path + '/**phot**.parquet') if 'variable' not in f]
        if pf:
            n_ims_ref += len(pq.read_table(pf[0]))

    ref_first_phot = [
        f for f in glob(ref_date_list[0] + '/**phot**.parquet') if 'variable' not in f]
    n_dfs_ref = len(ref_first_phot)

    # allocate
    times_ref    = np.zeros(n_ims_ref, dtype='float64')
    flux_ref     = np.zeros((n_dfs_ref, n_ims_ref, n_sources_ref), dtype='float32')
    flux_err_ref = np.zeros_like(flux_ref)
    times_list_ref = []   # views into times_ref[]
    start_ref = 0

    t2 = time.time()
    for i, path in enumerate(ref_date_list):
        print(f'Reading ref photometry from {ref_dates[i]} ({i+1}/{len(ref_date_list)}).')
        ref_phot_files = [
            f for f in glob(path + '/**phot**.parquet')
            if 'variable' not in f
            and float(f.split('_')[-1].split('.parquet')[0])]
        if not ref_phot_files:
            continue

        anc_ref = pq.read_table(
            glob(path + '/**ancillary**.parquet')[0],
            columns=['BJD TDB']
        )

        use_cols_ref = []
        for s in ref_source_inds[i]:
            p = f'S{s}'
            use_cols_ref.extend([f'{p} Source-Sky', f'{p} Source-Sky Err'])

        for j in range(n_dfs_ref):
            data_ref = pq.read_table(ref_phot_files[j], columns=use_cols_ref, memory_map=True)
            stop_ref = start_ref + len(data_ref)

            if j == 0:
                times_ref[start_ref:stop_ref] = np.array(anc_ref['BJD TDB'])
                times_list_ref.append(times_ref[start_ref:stop_ref])  # VIEW

            for k in range(n_sources_ref):
                s = ref_source_inds[i][k]
                flux_ref[j, start_ref:stop_ref, k]     = np.array(data_ref[f'S{s} Source-Sky'])
                flux_err_ref[j, start_ref:stop_ref, k] = np.array(data_ref[f'S{s} Source-Sky Err'])

        start_ref = stop_ref

    print(f'Ref read-in: {time.time()-t2:.1f}s')
    print(f'times_ref[0]={times_ref[0]:.6f}, flux_ref[0,0,0]={flux_ref[0,0,0]:.1f}')

     # ── 8. Load target field weights ────────────────────────────────────────
    targ_weights_path = (f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/weights.csv')

    if not os.path.exists(targ_weights_path):
        raise RuntimeError(
            f'Reference field weights not found at {targ_weights_path}. '
            f'Run analyze_global on {field} before analyze_thwomp.')
    targ_weights_df = pd.read_csv(targ_weights_path)
    targ_ref_ids = np.array(targ_weights_df['Ref ID'])
    ap_col = ref_first_phot[0].split('_')[-1].split('.parquet')[0]
    if ap_col not in targ_weights_df.columns:
        raise RuntimeError(f'Aperture {ap_col} not in weights CSV.')
    targ_weights_j = np.array(targ_weights_df[ap_col])

    targ_weight_map = {rid: w for rid, w in zip(targ_ref_ids, targ_weights_j)}
    targ_weights_ordered = np.array([targ_weight_map.get(sid, 0.0) for sid in common_source_ids])

    # ── 9. Load reference field weights ────────────────────────────────────────
    ref_weights_path = (f'/data/tierras/fields/{ref_field}/sources/lightcurves/{ffname}/weights.csv')
    if not os.path.exists(ref_weights_path):
        raise RuntimeError(
            f'Reference field weights not found at {ref_weights_path}. '
            f'Run analyze_global on {ref_field} before analyze_thwomp.')
    ref_weights_df = pd.read_csv(ref_weights_path)
    # columns: 'Ref ID', '5.0', '6.0', ..., '20.0'
    # rows: one per reference star with a non-zero weight on at least one aperture

    # Map weight Gaia IDs to positions in common_source_ids_ref
    weight_ref_ids = np.array(ref_weights_df['Ref ID'])

    # ── 10. Subtract integer offset from times so both arrays share the same origin
    x_offset = int(np.floor(times[0]))
    times     -= x_offset   # times_list entries are views → also updated
    times_ref -= x_offset   # times_list_ref entries are views → also updated

    # ── 11. Build per-aperture, per-df, per-night interpolated ALC ─────────────────────
    EXTRAP_WARN_MIN = 5.0   # minutes; warn if target falls this far outside ref bounds

    alc_interp_all     = np.full((n_ims, n_dfs), np.nan, dtype='float64')
    alc_err_interp_all = np.full((n_ims, n_dfs), np.nan, dtype='float64')

    targ_weight_df_ind = np.where(targ_weights_df['Ref ID'] == tierras_target_id)[0][0]

    # ── 12. Quality masks ───────────────────────
    x_deviations = np.median(x_pos - np.nanmedian(x_pos, axis=0), axis=1)
    y_deviations = np.median(y_pos - np.nanmedian(y_pos, axis=0), axis=1)

    flux_ref_idx = 5 if (ap_rad is None and n_dfs > 5) else 0
    median_flux = (np.nanmedian(flux[flux_ref_idx], axis=1) /
                   np.nanmedian(np.nanmedian(flux[flux_ref_idx], axis=1)))
    flux_mask = np.zeros(n_ims, dtype='int')
    flux_mask[np.where(median_flux < 0.98)[0]] = 1

    pos_mask = np.zeros(n_ims, dtype='int')
    pos_mask[np.where((np.abs(x_deviations) > 20) | (np.abs(y_deviations) > 20))[0]] = 1

    fwhm_mask_arr = np.zeros(n_ims, dtype='int')
    fwhm_mask_arr[np.where(fwhm_x > 4)[0]] = 1

    short_night_mask = np.zeros(n_ims, dtype='bool')
    quality_mask = (wcs_flags == 1) | (pos_mask == 1) | (flux_mask == 1)
    mask_inv = ~quality_mask

    sigma_s = (0.09 * 130**(-2/3) * airmasses**(7/4) * (2 * exposure_times)**(-1/2) * np.exp(-2306 / 8000))

    best_std          = np.inf
    best_ap_label     = None
    best_corr_flux     = None
    best_corr_flux_err = None
    best_raw_flux      = None
    best_raw_flux_err  = None
    best_alc_col       = None
    best_alc_err_col   = None
    best_sat_flags     = None
    best_nl_flags      = None

    for i in range(n_dfs): # replace with n_dfs
        t_targ_full = []
        alc_targ_full = []
        alc_targ_err_full = []
        t_ref_full = []
        alc_ref_full = []
        alc_ref_err_full = []

        med_targ_refs_flux = np.median(np.nansum(flux[i, :, np.where(np.arange(n_sources) != targ_weight_df_ind)[0]], axis=1))
        med_targ_flux = np.median(flux[i, :, targ_weight_df_ind])
        med_ref_field_flux = np.median(np.nansum(flux_ref[i, :, :], axis=1))
        
        for n_idx in range(len(dates)):

            radius = ref_weights_df.keys()[i+1]
            night_date = dates[n_idx]

            ref_night_match = [ri for ri, rd in enumerate(ref_dates) if rd == night_date]
            if not ref_night_match:
                warnings.warn(f'{night_date}: no ref data; skipping night in THWOMP correction.')
                continue
            ri = ref_night_match[0]

            t_night = times_list[n_idx]
            t_ref_night = times_list_ref[ri]

            targ_inds = np.where((times >= t_night[0]) & (times <= t_night[-1]))[0]
            ref_inds  = np.where(
                (times_ref >= t_ref_night[0]) & (times_ref <= t_ref_night[-1])
            )[0]

            if len(ref_inds) < 2:
                print(f'{night_date}: fewer than 2 ref exposures; cannot interpolate.')
                continue    

            # load the targ_weights and renormalize after setting the target's weight in this aperture size to 0 
            targ_weights_arr = np.array(targ_weights_df[radius])
            targ_weights_arr[targ_weight_df_ind] = 0.0
            targ_weights_arr /= np.nansum(targ_weights_arr)

            alc_raw_targ   = flux[i, targ_inds, :] @ targ_weights_arr
            alc_err_raw_targ = np.sqrt((flux_err[i, targ_inds, :]**2) @ (targ_weights_arr**2))

            ref_weights_arr = np.array(ref_weights_df[radius])

            alc_raw_ref    = flux_ref[i, ref_inds, :] @ ref_weights_arr
            alc_err_raw_ref = np.sqrt((flux_err_ref[i, ref_inds, :]**2) @ (ref_weights_arr**2))

            valid = ~np.isnan(alc_raw_ref)
            if np.sum(valid) < 2:
                print(f'{night_date}: fewer than 2 non-NaN ALC points; skipping.')
                continue

            t_ref_v = times_ref[ref_inds][valid]
            alc_v_ref   = alc_raw_ref[valid]
            alc_e_v_ref = alc_err_raw_ref[valid]

            t_targ   = times[targ_inds]
            leading  = t_targ[t_targ < t_ref_v[0]]
            trailing = t_targ[t_targ > t_ref_v[-1]]

            if len(leading):
                gap_min = (t_ref_v[0] - leading.min()) * 24 * 60
                if gap_min > EXTRAP_WARN_MIN:
                    warnings.warn(
                        f'{night_date} ap={ap_col}: {len(leading)} target exposures '
                        f'extrapolated up to {gap_min:.1f} min before first ref.')
            if len(trailing):
                gap_min = (trailing.max() - t_ref_v[-1]) * 24 * 60
                if gap_min > EXTRAP_WARN_MIN:
                    warnings.warn(
                        f'{night_date} ap={ap_col}: {len(trailing)} target exposures '
                        f'extrapolated up to {gap_min:.1f} min after last ref.')

            t_targ_v = t_targ
            alc_v_targ = alc_raw_targ
            alc_e_v_targ = alc_err_raw_targ

            t_targ_full.extend(t_targ_v)
            alc_targ_full.extend(alc_v_targ)
            alc_targ_err_full.extend(alc_e_v_targ)
            t_ref_full.extend(t_ref_v)
            alc_ref_full.extend(alc_v_ref)
            alc_ref_err_full.extend(alc_e_v_ref)

        t_targ_full = np.array(t_targ_full)
        alc_targ_full = np.array(alc_targ_full)
        alc_targ_err_full = np.array(alc_targ_err_full)
        t_ref_full = np.array(t_ref_full)
        alc_ref_full = np.array(alc_ref_full)
        alc_ref_err_full = np.array(alc_ref_err_full)

        targ_flux = flux[i, :, targ_weight_df_ind]
        targ_flux_err = flux_err[i, :, targ_weight_df_ind]

        # normalize
        norm_targ_alc = np.nanmedian(alc_targ_full)
        norm_ref_alc  = np.nanmedian(alc_ref_full)
        norm_targ_flux = np.nanmedian(targ_flux)

        alc_targ_full /= norm_targ_alc
        alc_targ_err_full /= norm_targ_alc

        alc_ref_full /= norm_ref_alc
        alc_ref_err_full /= norm_ref_alc

        targ_flux /= norm_targ_flux
        targ_flux_err /= norm_targ_flux

        # do a GP regression to get the interpolation curve 

        t_train_raw     = np.concatenate([t_ref_full, t_targ_full])
        y_train_raw     = np.concatenate([alc_ref_full, alc_targ_full])
        sigma_train_raw = np.concatenate([alc_ref_err_full, alc_targ_err_full])

        sort_idx    = np.argsort(t_train_raw)
        t_train     = t_train_raw[sort_idx]
        y_train     = y_train_raw[sort_idx]
        sigma_train = sigma_train_raw[sort_idx]

        # --- 2. Fit hyperparameters ---
        fit_result, bounds = fit_gp(t_train, y_train, sigma_train, n_restarts=1)
        try:
            best_params = fit_result.x
        except:
            print('No solution found, continuing.')
            continue

        correction_mean, correction_std = predict_mean_std(times, t_train, y_train, sigma_train, best_params)

        # --- 4. Apply the correction (multiplicative, since these are normalized ALC fluxes ~1) ---
        targ_flux_corr_gp = targ_flux / correction_mean
        frac_err_targ_gp  = targ_flux_err / targ_flux
        frac_err_corr_gp  = correction_std / correction_mean
        targ_flux_corr_err_gp = targ_flux_corr_gp * np.sqrt(frac_err_targ_gp**2 + frac_err_corr_gp**2)

        # do a correction just using the reference stars in the target field
        targ_flux_corr_self = targ_flux / alc_targ_full
        frac_err_targ_self  = targ_flux_err / targ_flux
        frac_err_corr_self  = alc_targ_err_full / correction_mean
        targ_flux_corr_err_self = targ_flux_corr_self * np.sqrt(frac_err_targ_self**2 + frac_err_corr_self**2)

        # normalize 
        norm = np.nanmedian(targ_flux_corr_gp) 
        targ_flux_corr_gp /= norm 
        targ_flux_corr_err_gp /= norm
        
        # account for scintillation
        n_nonzero_j = (int(np.sum(np.array(ref_weights_df[radius]) > 0))
                       if radius in ref_weights_df.columns else 1)
        sigma_scint_j = 1.5 * sigma_s * np.sqrt(1.0 + 1.0 / max(n_nonzero_j, 1))

        targ_flux_corr_err_gp   = np.sqrt(targ_flux_corr_err_gp**2 + sigma_scint_j**2)
        targ_flux_corr_err_self = np.sqrt(targ_flux_corr_err_self**2 + sigma_scint_j**2)

        # 5-minute scatter on target, unmasked exposures with valid ALC
        use = mask_inv 
        if np.sum(use) < 4:
            continue
        _, by, _ = tierras_binner(
            times[use] + x_offset,
            targ_flux_corr_gp[use],
            bin_mins=5
        )
        scatter = np.nanstd(by)

        print(f'  ap={radius}: 5-min scatter on target = {scatter:.6f}')

        if scatter < best_std:
            best_ap_index           = i 
            best_std                = scatter
            best_ap_label           = radius
            best_corr_flux_gp       = targ_flux_corr_gp.astype('float32').copy()
            best_corr_flux_err_gp   = targ_flux_corr_err_gp.astype('float32').copy()
            best_corr_flux_self     = targ_flux_corr_self.astype('float32').copy()
            best_corr_flux_err_self = targ_flux_corr_err_self.astype('float32').copy()
            best_raw_flux           = targ_flux.astype('float32').copy()
            best_raw_flux_err       = targ_flux_err.astype('float32').copy()
            best_alc_col            = correction_mean.astype('float32').copy()
            best_alc_err_col        = correction_std.astype('float32').copy()
            best_t_ref_full         = t_ref_full.astype('float32').copy()
            best_t_targ_full        = t_targ_full.astype('float32').copy()
            best_alc_ref_full       = alc_ref_full.astype('float32').copy()
            best_alc_targ_full      = alc_targ_full.astype('float32').copy()
            best_alc_ref_err_full   = alc_ref_err_full.astype('float32').copy()
            best_alc_targ_err_full  = alc_targ_err_full.astype('float32').copy()
            best_correction_mean    = correction_mean.astype('float32').copy()
            best_correction_std     = correction_std.astype('float32').copy()  
            best_params_save        = best_params      

    if best_corr_flux_gp is None:
        raise RuntimeError('No valid aperture found. Check that ALC interpolation succeeded.')
    
    fig, ax = plt.subplots(2, 1, figsize=(10,12), sharex=True)
    ax[0].errorbar(best_t_targ_full, best_alc_targ_full, best_alc_targ_err_full, marker='.', ls='', label='ALC using reference stars in the target field')
    ax[0].errorbar(best_t_ref_full, best_alc_ref_full, best_alc_ref_err_full, marker='.', ls='', label='ALC using reference stars in the reference field')
    ax[0].errorbar(times, best_raw_flux, best_raw_flux_err, marker='.', ls='', label='Target flux')
    
    ax[0].grid(alpha=0.5)

    ax[1].errorbar(times, best_corr_flux_gp, best_corr_flux_err_gp, marker='.', ls='', color='k', alpha=0.5, label='GP correction')
    # ax[1].errorbar(times, best_corr_flux_self, best_corr_flux_err_self, marker='.', ls='', alpha=0.5, label='ALC from target field stars correction' )


    t_train_raw     = np.concatenate([best_t_ref_full, best_t_targ_full])
    y_train_raw     = np.concatenate([best_alc_ref_full, best_alc_targ_full])
    sigma_train_raw = np.concatenate([best_alc_ref_err_full, best_alc_targ_err_full])

    sort_idx    = np.argsort(t_train_raw)
    t_train     = t_train_raw[sort_idx]
    y_train     = y_train_raw[sort_idx]
    sigma_train = sigma_train_raw[sort_idx]

    correction_mean, correction_std = predict_mean_std(times, t_train, y_train, sigma_train, best_params_save) 

    ax[0].errorbar(times, correction_mean, correction_std, marker='.', color='k', ls='', label='GP model')
    ax[0].legend()  

    # do a hi-res model for display purposes
    t_pad = 0.05 * (t_train.max() - t_train.min())  # small padding beyond data range
    t_grid = np.linspace(t_train.min() - t_pad, t_train.max() + t_pad, 100000)
    grid_mean, grid_std = predict_mean_std(t_grid, t_train, y_train, sigma_train, best_params_save)

    ax[0].plot(t_grid, grid_mean, color='k')
    ax[0].fill_between(t_grid, grid_mean-grid_std, grid_mean+grid_std, color='k', alpha=0.1)

    ax[1].grid(alpha=0.5)

    print(f'Best aperture: {best_ap_label} (5-min scatter = {best_std:.6f})')

    x_breaks = np.where(np.gradient(times) > 0.2)[0][1::2]
    x_breaks = np.insert(x_breaks, 0, 0)
    x_breaks = np.append(x_breaks, len(times))
    n_nights = len(x_breaks) - 1
    bx  = np.zeros(n_nights)
    by_gp  = np.zeros(n_nights)
    bye_gp = np.zeros(n_nights)
    bx_self  = np.zeros(n_nights)
    by_self  = np.zeros(n_nights)
    bye_self = np.zeros(n_nights)
    for i in range(n_nights):
        start = x_breaks[i]
        end = x_breaks[i+1]
        bx[i]  = np.mean(times[start:end])
        by_gp[i]  = np.median(best_corr_flux_gp[start:end])
        bye_gp[i] = np.median(best_corr_flux_err_gp[start:end]) / np.sqrt(x_breaks[i+1]-x_breaks[i])
        by_self[i]  = np.median(best_corr_flux_self[start:end])
        bye_self[i] = np.median(best_corr_flux_err_self[start:end]) / np.sqrt(x_breaks[i+1]-x_breaks[i])

    ax[1].errorbar(bx, by_gp, bye_gp, marker='o', color='k', ls='', zorder=4)
    # ax[1].errorbar(bx, by_self, bye_self, marker='o', color='tab:blue', ls='', zorder=4)

    ax[1].axhline(1, lw=2, alpha=0.5, zorder=0, color='k')
    ax[0].axhline(1, lw=2, alpha=0.5, zorder=0, color='k')
    ax[1].legend()


    # ── 15. Create output directory ─────────────────────────────────────────────
    output_path = Path(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}')
    output_path.mkdir(parents=True, exist_ok=True)
    set_tierras_permissions(str(output_path))

    # ── 16. Write one CSV per output source ────────────────────────────────────
    for tt in output_source_inds:
        gaia_id = common_source_ids[tt]
        source_name = field if gaia_id == tierras_target_id else f'Gaia DR3 {gaia_id}'
        out_file = output_path / f'{source_name}_global_lc.csv'
        if out_file.exists():
            out_file.unlink()

        output_dict = {
            'BJD TDB':                  times + x_offset,
            'Flux':                     best_corr_flux_gp,
            'Flux Error':               best_corr_flux_err_gp,
            'Raw Flux (ADU)':           best_raw_flux,
            'Raw Flux Error (ADU)':     best_raw_flux_err,
            'ALC':                      best_alc_col,
            'ALC Error':                best_alc_err_col,
            'Sky Background (ADU/s)':   sky[:, tt] / exposure_times,
            'X':                        x_pos[:, tt],
            'Y':                        y_pos[:, tt],
            'WCS Flag':                 wcs_flags.astype(int),
            'Position Flag':            pos_mask,
            'FWHM Flag':                fwhm_mask_arr,
            'Flux Flag':                flux_mask,
            'Saturated Flag':           saturated_flags[best_ap_index, :, tt], 
            'Non-Linear Flag':          non_linear_flags[best_ap_index, :, tt]
        }

        with open(out_file, 'a') as fh:
            fh.write(f'# this light curve was made using circular_fixed_ap_phot_{best_ap_label}\n')
            pd.DataFrame(output_dict).to_csv(fh, index=False, na_rep='nan')
        set_tierras_permissions(out_file)
        print(f'Wrote {out_file}')

    gc.collect()
    breakpoint()

if __name__ == '__main__':
    main()