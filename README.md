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

## Notes

The parquet inventories are environment-specific and depend on access to internal RMI archive paths. The output dataset uses Zarr v3.
example.hdf is only used to derive metadata needed to initialize the empty dataset structure.

## Dataset license

The RADCLIM dataset is licensed under CC BY 4.0.
