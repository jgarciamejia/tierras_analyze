# THWOMP Analysis Pipeline Design

**Date:** 2026-05-19  
**Repos affected:** `tierras_analyze` (primary), `tierras_red` (none)

---

## Background

THWOMP (Target Host with Off-chip Multi-aperture Photometry, or similar) is an observing mode for bright targets that would saturate or go non-linear in a standard long exposure. The telescope alternates between two pointings within each 7.5-minute slice:

1. **Reference field** (`{target}_ref`): bright target is dithered off the chip; one long (30 s) exposure is taken of a reference star field to accumulate flux in fainter reference stars.
2. **Target field** (`{target}`): telescope dithers back and reacquires the bright target; ten short exposures are taken.

This cycle repeats until the slice ends. The sequence is guaranteed to **start and end on a reference field exposure**; the only case where a target exposure falls outside the reference time bounds is a sequence failure at the end of a slice.

THWOMP slices are **interspersed with standard observations** of other fields throughout the night. Multiple THWOMP slices for the same target can occur on the same night, separated by large time gaps.

Because reference stars come from a separate pointing (`{target}_ref`) rather than from within the target field, the standard `analyze_global` ALC construction does not apply to THWOMP targets.

---

## Goals

- Produce ALC-corrected global light curves for the THWOMP target and all other bright sources (G_RP < 13) in the target field.
- Reuse pre-computed reference field weights from the standard `analyze_global` run on `{target}_ref`.
- Output in the same format as `analyze_global` so all downstream tools work without modification.
- Auto-detect THWOMP pairs; no operator flags required.
- Handle multi-night datasets the same way `analyze_global` does: one combined dataset spanning all nights.

---

## Architecture

### No changes to `tierras_red`

`ap_phot.py` already runs identically on `{target}` and `{target}_ref`. The standard photometry pipeline produces parquet files for both fields without modification.

### Changes to `tierras_analyze`

#### New file: `analyze_thwomp.py`

Contains a `main(raw_args=None)` function following the same conventions as `analyze_global.main`. See *Data Flow* section below.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `-field` | required | Target field name (e.g. `HIP47080`), not the `_ref` field |
| `-ffname` | `flat0000` | Flattened directory name |
| `-use_nights` | None | Comma-separated list of dates to include (pass a single date to restrict to one night) |
| `-minimum_night_duration` | 0 | Min cumulative exposure time per night (hours) |
| `-ap_rad` | None | Fix aperture size; if None, select by minimising 5-min scatter |
| `-force_reweight` | False | Not used directly (weights come from `_ref` pipeline), reserved for future |
| `-rp_bright_limit` | 13.0 | G_RP magnitude limit for producing light curves of non-target sources |

#### Edit: `process_data.py`

After the existing `analyze_global_main` loop, detect THWOMP pairs:

```python
# detect THWOMP targets: any field that has a sibling {field}_ref directory
thwomp_targets = [t for t in target_list
                  if not t.endswith('_ref')
                  and f'{t}_ref' in target_list]
for target in thwomp_targets:
    args = f'-field {target} -ffname {ffname}'
    analyze_thwomp_main(args.split())
```

`_ref` fields are already processed by the standard `analyze_global` loop and are not passed to `analyze_thwomp_main`.

#### Edit: `make_light_curves.py`

Same detection logic and call added after the `analyze_global_main` loop.

---

## Data Flow inside `analyze_thwomp.py`

### 1. Discover dates

Glob `/data/tierras/photometry/**/{field}/{ffname}`, sort chronologically. Apply `ignore_dates.txt` if present under `/data/tierras/fields/{field}/`. Apply `use_nights` and `minimum_night_duration` filters identically to `analyze_global`.

### 2. Read target field photometry

For each date, read:
- Ancillary parquet: `BJD TDB`, `Airmass`, `Exposure Time`, `HA`, `Dome Humid`, `FWHM X`, `FWHM Y`, `WCS Flag`, `Filename`
- All fixed-aperture phot parquets (files not containing `variable` in name), sorted by aperture size
- Sources CSV for Gaia IDs, G_RP mags, pixel positions

Build arrays across all nights:
- `times[n_ims]`, `airmasses[n_ims]`, `exposure_times[n_ims]`
- `flux[n_dfs, n_ims, n_sources]`, `flux_err`, `non_linear_flags`, `saturated_flags`
- `times_list`: list of per-night time arrays (for night-boundary bookkeeping)

Find common source IDs across all nights (same logic as `analyze_global`).

### 3. Identify target and bright sources

Call `identify_target_gaia_id(field, source_df, x_pix=CAT-X, y_pix=CAT-Y)` using the first image header on the most recent night.

Select sources for light curve output: the target (regardless of magnitude) plus all sources with `phot_rp_mean_mag < rp_bright_limit`.

### 4. Read reference field photometry

Discover dates independently for `{field}_ref` (same glob pattern). Read parquets into parallel arrays:
- `times_ref[n_ims_ref]`, `flux_ref[n_dfs, n_ims_ref, n_sources_ref]`, `flux_err_ref`, etc.

The ref field date list is built independently to handle edge cases where a ref field date has no paired target data or vice versa.

### 5. Load reference field weights

Read `/data/tierras/fields/{field}_ref/sources/lightcurves/{ffname}/weights.csv`.

If the file does not exist, raise a `RuntimeError` with a clear message: the standard `analyze_global` run on `{field}_ref` must complete before `analyze_thwomp` can run. This is guaranteed by the ordering in `process_data.py` (THWOMP calls come after the `analyze_global` loop).

Weights are stored per aperture size (columns are aperture radii as floats, rows are reference source Gaia IDs). Match column names to the aperture sizes in `phot_files_ref`.

### 6. Build ALC per night

For each aperture size `j` and each night `n`:

```
alc_night = flux_ref[j, night_ref_inds, :] @ weights_j
alc_err_night = sqrt(flux_err_ref[j, night_ref_inds, :]**2 @ weights_j**2)
```

Normalise by `nanmedian(alc_night)`. Remove NaN-valued ALC points before interpolation.

### 7. Interpolate ALC to target timestamps

For each night independently, fit `scipy.interpolate.CubicSpline` through `(times_ref_night, alc_night)` with `extrapolate=True`.

Evaluate at `times_target_night` to produce `alc_interp_night`.

**Extrapolation warning:** if any target timestamp falls more than 5 minutes outside the nearest reference timestamp for that night, emit a `warnings.warn` with the number of affected exposures and the maximum extrapolation gap. Do not discard those points.

Concatenate per-night interpolated ALCs into a single `alc_interp[n_ims]` array.

ALC uncertainty at target timestamps: use `median(alc_err_night)` as a constant approximation per night (same approach as the example script), propagated in quadrature with the target flux uncertainty.

### 8. Aperture loop and selection

For each aperture size `j`:

1. Normalise `targ_flux = flux[j, :, targ_ind] / nanmedian(flux[j, :, targ_ind])`
2. Add scintillation in quadrature:
   `sigma_s = 0.09 * 130**(-2/3) * airmasses**(7/4) * (2*exposure_times)**(-1/2) * exp(-2306/8000)`
3. Correct: `targ_flux_corr = targ_flux / alc_interp`
4. Propagate errors fully:
   `targ_flux_err_corr = sqrt((targ_flux_err/alc_interp)**2 + (targ_flux * alc_err_interp / alc_interp**2)**2)`
5. Compute 5-minute binned scatter using `tierras_binner` from `ap_phot.py`

If `-ap_rad` is not specified, select the aperture minimising scatter on the target source. Apply the same aperture to all other bright sources in the output.

### 9. Write output

For each source in the output set, write:

```
/data/tierras/fields/{field}/sources/lightcurves/{ffname}/{gaia_id}_global_lc.csv
```

Column format matches `analyze_global` output. Call `set_tierras_permissions` on each output file.

---

## Output directory

```
/data/tierras/fields/{field}/sources/lightcurves/{ffname}/
    {target_gaia_id}_global_lc.csv       ← THWOMP target
    {bright_source_gaia_id}_global_lc.csv  ← other G_RP < 13 sources
    # weights.csv is NOT written here; it lives under {field}_ref/
```

---

## Edge cases

| Case | Handling |
|---|---|
| Night ends without final reference exposure | `extrapolate=True` on spline; warn if gap > 5 min |
| `{field}_ref` weights file missing | Raise `RuntimeError` with clear message |
| Ref field observed on a night with no target data | That night's ref data is simply unused |
| Target field observed on a night with no ref data | Warn and skip that night's data from output |
| Source appears in some nights but not all | Standard common-source-ID logic excludes it |

---

## What is not changing

- `ap_phot.py` in `tierras_red` — no changes
- `analyze_global.py` — no changes; continues to process `{target}_ref` as a normal field and produce its `weights.csv`
- Database update logic in `process_data.py` — THWOMP output CSVs are in the standard location and format, so `build_tierras_db_main` picks them up automatically
