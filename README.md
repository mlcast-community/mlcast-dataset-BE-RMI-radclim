# RADCLIM to Zarr

This repository contains the code used to build a Zarr v3 version of the RADCLIM dataset.

## Important note on licensing

The license of this software is **not** the same as the license of the dataset itself.

The RADCLIM dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Please refer to the repository `LICENSE` file for the license that applies to this codebase.

## Project overview

The workflow in this repository consists of three main steps:

1. **Build inventory parquet files**
2. **Create an empty Zarr dataset**
3. **Populate the Zarr dataset with data from the source HDF files**

The final dataset is stored in **Zarr v3** format.

## Workflow

### 1. Build inventory parquet files

The first step is to build parquet files describing the available RADCLIM source files.

For each timestamp, the parquet inventory stores the location of the corresponding variable file in the RADCLIM archive. This mapping serves two purposes:

- it makes it easier to analyse the completeness of the dataset
- it is later used to populate the Zarr store efficiently

This step produces two parquet files:

- one for **5-minute** data
- one for **hourly** data

These parquet files are **not included** in this repository. Building them requires access to the internal RMI servers, because the stored paths point to files in the RADCLIM archive.

### 2. Create an empty Zarr dataset

After the inventory has been created, an empty Zarr dataset is initialized.

This step uses metadata from a sample ODIM HDF file to construct the grid definition, coordinates, and dataset structure.

The file `./example.hdf` is included as a reference file for constructing the empty Zarr dataset.

The main data variable is written with **Blosc** compression (Zstandard codec inside Blosc, with a configurable shuffle filter). Both the compression level and the shuffle modeare tunable — see [Compression settings](#compression-settings) below.

### 3. Populate the Zarr dataset

Once the empty Zarr store exists, it is populated with the RADCLIM data by reading the source HDF files listed in the inventory parquet.

The full creation and population workflow is executed from `src/main.py`.

## Repository structure

The Python source files are located in the `src/` subfolder:

- `src/build_inventory.py` – builds parquet inventories of available RADCLIM files
- `src/make_empty_zarr.py` – creates an empty Zarr v3 dataset
- `src/populate_zarr.py` – populates the Zarr store with raster data
- `src/main.py` – runs the end-to-end workflow
- `example.hdf` – sample HDF file used to derive the grid and projection metadata

Adjust these filenames if your local version differs slightly.

## Installation

All dependencies can be installed with [`uv`](https://github.com/astral-sh/uv).

For example:

```bash
uv sync
```

## Usage

The main workflow is run through:

```bash
python src/main.py
```

In summary, the process is:
- generate the parquet inventory files
- create the empty Zarr v3 dataset
- populate the dataset with data from the RADCLIM HDF archive

`src/main.py` accepts the following command-line options:

| Flag | Default | Description |
|---|---|---|
| `--variable` | `acrr_qpe_mfb_5m` | Inventory column to read, and the name of the output variable in the Zarr store (e.g. `acrr_qpe_mfb_5m` for the 5-minute accumulation, or `rate_qpe_mfb` for the 5-minute instantaneous rain rate). |
| `--store-path` | see `main.py` | Output Zarr store path. |
| `--clevel` | `9` | Blosc compression level, 0–9. |
| `--shuffle` | `bitshuffle` | Blosc shuffle filter: `noshuffle`, `shuffle`, or `bitshuffle`. |
| `--start-date` | none (full inventory) | Inclusive start date (e.g. `2020-01-01`) to scope the run to a subset of the inventory. |
| `--end-date` | none (full inventory) | Exclusive end date (e.g. `2020-02-01`) to scope the run to a subset of the inventory. |

Run `python src/main.py --help` for the full, up-to-date list.

Note that the output variable is now named after `--variable`, rather than always being called `rain_rate`.

### Testing on a subset of the dataset

The full inventory spans several years at 5-minute resolution, so a full build can take a long time. To iterate quickly — for example when comparing compression settings, or otherwise testing a change — scope a run to a short date range with `--start-date`/`--end-date`:

```bash
python src/main.py --variable acrr_qpe_mfb_5m --start-date 2020-01-01 --end-date 2020-02-01 --store-path ./BE-radclim-acrr_mfb-test.zarr
```

This builds a store containing only that date range, so you can validate the pipeline or compare output sizes/settings in minutes rather than days. Omit both flags to build against the full inventory.

### Compression settings

To compare compression settings, run multiple invocations against the same date range with different `--clevel`/`--shuffle` values and distinct `--store-path`s, then compare the resulting store sizes, e.g.:

```bash
python src/main.py --shuffle noshuffle   --start-date 2020-01-01 --end-date 2020-02-01 --store-path ./test_noshuffle.zarr
python src/main.py --shuffle shuffle     --start-date 2020-01-01 --end-date 2020-02-01 --store-path ./test_shuffle.zarr
python src/main.py --shuffle bitshuffle  --start-date 2020-01-01 --end-date 2020-02-01 --store-path ./test_bitshuffle.zarr

du -sh ./test_noshuffle.zarr ./test_shuffle.zarr ./test_bitshuffle.zarr
```

### Running multiple builds in parallel

`src/main.py` populates the store using a multiprocessing worker pool (`N_WORKERS` in the file). If you run several invocations concurrently (e.g. one per compression setting, as above), each spawns its own pool of workers, and each worker process imports NumPy — whose bundled BLAS library tries to create its own thread pool by default. Running several jobs at once can exhaust your account's process/thread limit (`ulimit -u`) well before it exhausts available CPU cores.

If you see errors like `pthread_create failed ... Resource temporarily unavailable` or `RLIMIT_NPROC`, cap BLAS threading per process before launching (this workload's parallelism already comes from the worker pool, not from BLAS):

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Notes

The parquet inventories are environment-specific and depend on access to internal RMI archive paths. The output dataset uses Zarr v3.
`example.hdf` is only used to derive metadata needed to initialize the empty dataset structure.

## Dataset license

The RADCLIM dataset is licensed under CC BY 4.0.
