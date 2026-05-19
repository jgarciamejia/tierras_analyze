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
    # Build common_rp via O(n) merge rather than O(n²) per-source lookup
    rp_lookup = source_dfs[0].set_index('source_id')['phot_rp_mean_mag']
    common_rp = np.array(rp_lookup.reindex(common_source_ids))
    targ_common_idx = np.where(common_source_ids == tierras_target_id)[0][0]
    output_source_inds = np.where(
        (common_rp < rp_bright_limit) | (common_source_ids == tierras_target_id)
    )[0]
    print(f'Will produce light curves for {len(output_source_inds)} sources '
          f'(target + G_RP < {rp_bright_limit}).')

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

        for j in range(n_dfs):
            file_idx = df_ind if df_ind is not None else j
            data_tab = pq.read_table(phot_files[file_idx], columns=use_cols, memory_map=True)
            stop = start + len(data_tab)

            # ancillary data only filled on j==0 to avoid duplicate writes
            if j == 0:
                times[start:stop]          = np.array(ancillary_tab['BJD TDB'])
                times_list.append(times[start:stop])   # VIEW — updated when times -= x_offset
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
    # ── 7. Read reference field photometry ─────────────────────────────────────
    ref_date_list = glob(f'/data/tierras/photometry/**/{ref_field}/{ffname}')
    ref_date_list = np.array(sorted(ref_date_list, key=lambda x: int(x.split('/')[4])))
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

    ref_first_phot = sorted(
        [f for f in glob(ref_date_list[0] + '/**phot**.parquet') if 'variable' not in f],
        key=lambda x: float(x.split('_')[-1].split('.parquet')[0])
    )
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
        ref_phot_files = sorted(
            [f for f in glob(path + '/**phot**.parquet') if 'variable' not in f],
            key=lambda x: float(x.split('_')[-1].split('.parquet')[0])
        )
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
    # ── 8. Load reference field weights ────────────────────────────────────────
    weights_path = (f'/data/tierras/fields/{ref_field}/sources/lightcurves/'
                    f'{ffname}/weights.csv')
    if not os.path.exists(weights_path):
        raise RuntimeError(
            f'Reference field weights not found at {weights_path}. '
            f'Run analyze_global on {ref_field} before analyze_thwomp.')
    weights_df = pd.read_csv(weights_path)
    # columns: 'Ref ID', '5.0', '6.0', ..., '20.0'
    # rows: one per reference star with a non-zero weight on at least one aperture

    # Map weight Gaia IDs to positions in common_source_ids_ref
    weight_ref_ids = np.array(weights_df['Ref ID'])

    # ── 9. Subtract integer offset from times so both arrays share the same origin
    x_offset = int(np.floor(times[0]))
    times     -= x_offset   # times_list entries are views → also updated
    times_ref -= x_offset   # times_list_ref entries are views → also updated

    # ── 10. Build per-aperture, per-night interpolated ALC ─────────────────────
    EXTRAP_WARN_MIN = 5.0   # minutes; warn if target falls this far outside ref bounds

    # alc_interp_all[j, t] = interpolated ALC value for aperture j at target time t
    alc_interp_all     = np.full((n_dfs_ref, n_ims), np.nan, dtype='float64')
    alc_err_interp_all = np.full_like(alc_interp_all, np.nan)

    for j in range(n_dfs_ref):
        ap_col = ref_first_phot[j].split('_')[-1].split('.parquet')[0]  # e.g. '10.0'
        if ap_col not in weights_df.columns:
            print(f'Aperture {ap_col} not in weights CSV; skipping.')
            continue
        weights_j = np.array(weights_df[ap_col])

        # build ordered weight vector aligned to common_source_ids_ref
        weight_map = {rid: w for rid, w in zip(weight_ref_ids, weights_j)}
        weights_ordered = np.array([weight_map.get(sid, 0.0)
                                    for sid in common_source_ids_ref])

        for n_idx in range(len(dates)):
            night_date = dates[n_idx]

            # locate matching ref night by date
            ref_night_match = [ri for ri, rd in enumerate(ref_dates) if rd == night_date]
            if not ref_night_match:
                warnings.warn(f'{night_date}: no ref data; skipping night in THWOMP correction.')
                continue
            ri = ref_night_match[0]

            # index arrays for this target night and ref night
            t_night = times_list[n_idx]       # view: already offset-subtracted
            t_ref_night = times_list_ref[ri]  # view: already offset-subtracted

            targ_inds = np.where((times >= t_night[0]) & (times <= t_night[-1]))[0]
            ref_inds  = np.where(
                (times_ref >= t_ref_night[0]) & (times_ref <= t_ref_night[-1])
            )[0]

            if len(ref_inds) < 2:
                warnings.warn(f'{night_date}: fewer than 2 ref exposures; cannot interpolate.')
                continue

            # build raw ALC
            alc_raw     = flux_ref[j, ref_inds, :] @ weights_ordered
            alc_err_raw = np.sqrt((flux_err_ref[j, ref_inds, :]**2) @ (weights_ordered**2))

            valid = ~np.isnan(alc_raw)
            if np.sum(valid) < 2:
                warnings.warn(f'{night_date}: fewer than 2 non-NaN ALC points; skipping.')
                continue

            t_ref_v  = times_ref[ref_inds][valid]
            alc_v    = alc_raw[valid]
            alc_e_v  = alc_err_raw[valid]

            norm_ref = np.nanmedian(alc_v)
            alc_v   /= norm_ref
            alc_e_v /= norm_ref

            # extrapolation warnings (times in days; convert gap to minutes)
            t_targ = times[targ_inds]
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

            cs = CubicSpline(t_ref_v, alc_v, extrapolate=True)
            alc_interp_all[j, targ_inds]     = cs(t_targ)
            alc_err_interp_all[j, targ_inds] = np.median(alc_e_v)

    print('ALC interpolation done.')
    print(f'alc_interp_all[0, non-nan count]: '
          f'{np.sum(~np.isnan(alc_interp_all[0]))} of {n_ims}')
    # ── 11. Quality masks (mirrors analyze_global logic) ───────────────────────
    x_deviations = np.median(x_pos - np.nanmedian(x_pos, axis=0), axis=1)
    y_deviations = np.median(y_pos - np.nanmedian(y_pos, axis=0), axis=1)

    flux_ref_idx = 5 if (ap_rad is None and n_dfs > 5) else 0
    median_flux = (np.nanmedian(flux[flux_ref_idx], axis=1) /
                   np.nanmedian(np.nanmedian(flux[flux_ref_idx], axis=1)))
    flux_mask = np.zeros(n_ims, dtype='int')
    flux_mask[np.where(median_flux < 0.9)[0]] = 1

    pos_mask = np.zeros(n_ims, dtype='int')
    pos_mask[np.where((np.abs(x_deviations) > 20) | (np.abs(y_deviations) > 20))[0]] = 1

    fwhm_mask_arr = np.zeros(n_ims, dtype='int')
    fwhm_mask_arr[np.where(fwhm_x > 4)[0]] = 1

    quality_mask = (wcs_flags == 1) | (pos_mask == 1) | (fwhm_mask_arr == 1) | (flux_mask == 1)
    mask_inv = ~quality_mask
    print(f'Quality mask: {np.sum(mask_inv)}/{n_ims} exposures pass.')

    # ── 12. Drop nights below minimum_night_duration ───────────────────────────
    dates_to_remove = []
    for i, t_night in enumerate(times_list):
        night_inds = np.where((times >= t_night[0]) & (times <= t_night[-1]))[0]
        tot_exp = np.nansum(exposure_times[night_inds]) / 3600
        if tot_exp <= minimum_night_duration:
            print(f'{dates[i]} dropped: {tot_exp:.2f}h < {minimum_night_duration}h.')
            dates_to_remove.append(i)
    if dates_to_remove:
        dates      = np.delete(dates, dates_to_remove)
        date_list  = np.delete(date_list, dates_to_remove)
        times_list = [t for i, t in enumerate(times_list) if i not in dates_to_remove]

    # ── 13. Scintillation noise ─────────────────────────────────────────────────
    sigma_s = (0.09 * 130**(-2/3) * airmasses**(7/4)
               * (2 * exposure_times)**(-1/2) * np.exp(-2306 / 8000))

    # effective n_refs: count non-zero weights in weights_df (use first aperture col)
    first_ap_col = ref_first_phot[0].split('_')[-1].split('.parquet')[0]
    n_nonzero_weights = int(np.sum(np.array(weights_df[first_ap_col]) > 0))
    sigma_scint = (1.5 * sigma_s * np.sqrt(1.0 + 1.0 / max(n_nonzero_weights, 1)))

    # ── 14. Aperture loop — ALC correction and scatter minimisation ─────────────
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

    # iterate only over apertures present in both phot_files_template and alc_interp_all
    n_ap_loop = n_dfs_ref if ap_rad is None else 1

    for j in range(n_ap_loop):
        if ap_rad is not None:
            # map requested ap_rad to index in target phot files
            target_radii = np.array([
                float(f.split('_')[-1].split('.parquet')[0])
                for f in first_phot_files
            ])
            j_targ = int(np.where(target_radii == ap_rad)[0][0])
            ap_label = str(ap_rad)
        else:
            ap_label = ref_first_phot[j].split('_')[-1].split('.parquet')[0]
            # find matching aperture index in target phot files
            target_radii = np.array([
                float(f.split('_')[-1].split('.parquet')[0])
                for f in first_phot_files
            ])
            j_targ = int(np.where(target_radii == float(ap_label))[0][0]) if float(ap_label) in target_radii else None
            if j_targ is None:
                continue

        alc_j     = alc_interp_all[j]          # (n_ims,)
        alc_err_j = alc_err_interp_all[j]

        valid_alc = ~np.isnan(alc_j)

        F     = flux[j_targ]                   # (n_ims, n_sources)
        F_err = flux_err[j_targ]
        sat   = saturated_flags[j_targ].astype(bool)
        F_m     = np.where(sat, np.nan, F)
        F_err_m = np.where(sat, np.nan, F_err)

        alc_2d     = alc_j[:, None]
        alc_err_2d = alc_err_j[:, None]

        corr_flux     = F_m / alc_2d
        corr_flux_err = np.sqrt(
            (F_err_m / alc_2d)**2
            + (F_m * alc_err_2d / alc_2d**2)**2
            + (sigma_scint[:, None])**2
        )

        norms = np.nanmedian(corr_flux, axis=0)
        norms = np.where(norms == 0, 1.0, norms)
        corr_flux     /= norms
        corr_flux_err /= norms

        # 5-minute scatter on target, unmasked exposures with valid ALC
        use = mask_inv & valid_alc
        if np.sum(use) < 4:
            continue
        _, by, _ = tierras_binner(
            times[use] + x_offset,
            corr_flux[use, targ_common_idx],
            bin_mins=5
        )
        scatter = np.nanstd(by)

        print(f'  ap={ap_label}: 5-min scatter on target = {scatter:.6f}')

        if scatter < best_std:
            best_std          = scatter
            best_ap_label     = ap_label
            best_corr_flux     = corr_flux.astype('float32').copy()
            best_corr_flux_err = corr_flux_err.astype('float32').copy()
            best_raw_flux      = F_m.astype('float32').copy()
            best_raw_flux_err  = F_err_m.astype('float32').copy()
            best_alc_col       = alc_j.astype('float32').copy()
            best_alc_err_col   = alc_err_j.astype('float32').copy()
            best_sat_flags     = saturated_flags[j_targ].copy()
            best_nl_flags      = non_linear_flags[j_targ].copy()

    if best_corr_flux is None:
        raise RuntimeError('No valid aperture found. Check that ALC interpolation succeeded.')

    print(f'Best aperture: {best_ap_label} (5-min scatter = {best_std:.6f})')

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
            'Flux':                     best_corr_flux[:, tt],
            'Flux Error':               best_corr_flux_err[:, tt],
            'Raw Flux (ADU)':           best_raw_flux[:, tt],
            'Raw Flux Error (ADU)':     best_raw_flux_err[:, tt],
            'ALC':                      best_alc_col,
            'ALC Error':                best_alc_err_col,
            'Sky Background (ADU/s)':   sky[:, tt] / exposure_times,
            'X':                        x_pos[:, tt],
            'Y':                        y_pos[:, tt],
            'WCS Flag':                 wcs_flags.astype(int),
            'Position Flag':            pos_mask,
            'FWHM Flag':                fwhm_mask_arr,
            'Flux Flag':                flux_mask,
            'Saturated Flag':           best_sat_flags[:, tt].astype(int),
            'Non-Linear Flag':          best_nl_flags[:, tt].astype(int),
        }

        with open(out_file, 'a') as fh:
            fh.write(f'# this light curve was made using circular_fixed_ap_phot_{best_ap_label}\n')
            pd.DataFrame(output_dict).to_csv(fh, index=False, na_rep='nan')
        set_tierras_permissions(out_file)
        print(f'Wrote {out_file}')

    gc.collect()
