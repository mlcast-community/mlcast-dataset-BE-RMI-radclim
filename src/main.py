from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from zarr.codecs import BloscShuffle

from make_empty_zarr import (
    create_empty_geozarr_single_variable_from_inventory,
    read_odim_grid_spec,
)
from populate_zarr import populate_geozarr_from_inventory_parquet

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
EXAMPLE_HDF = Path("./example.hdf")
INVENTORY_PARQUET = Path("radclim_inventory_5min.parquet")
STORE_PATH = Path("./BE-radclim-mfb-rain_rate.zarr")
SCOPED_INVENTORY_DIR = Path("./inventory_subsets")

SOURCE_VAR = "acrr_qpe_mfb_5m"

N_WORKERS = 16
CHUNKSIZE = 100

# Blosc compression settings for the main data variable.
CLEVEL = 9
SHUFFLE = BloscShuffle.bitshuffle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and populate a GeoZarr store from the radclim inventory."
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=STORE_PATH,
        help=f"Output Zarr store path (default: {STORE_PATH}).",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=CLEVEL,
        help=f"Blosc compression level, 0-9 (default: {CLEVEL}).",
    )
    parser.add_argument(
        "--shuffle",
        choices=[s.value for s in BloscShuffle],
        default=SHUFFLE.value,
        help=f"Blosc shuffle filter (default: {SHUFFLE.value}).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive start date (e.g. 2020-01-01). Omit to use the full inventory.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Exclusive end date (e.g. 2020-02-01). Omit to use the full inventory.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=SOURCE_VAR,
        help=(
            "Inventory column / output variable name, e.g. 'acrr_qpe_mfb_5m' "
            f"or 'rate_qpe_mfb' (default: {SOURCE_VAR})."
        ),
    )
    return parser.parse_args()


def build_scoped_inventory(start_date: str | None, end_date: str | None) -> Path:
    """
    Write a date-filtered copy of ``INVENTORY_PARQUET`` and return its path.

    Returns ``INVENTORY_PARQUET`` unchanged when no date range is given.
    """
    if start_date is None and end_date is None:
        return INVENTORY_PARQUET

    inventory = pd.read_parquet(INVENTORY_PARQUET)
    if start_date is not None:
        inventory = inventory[inventory.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        inventory = inventory[inventory.index < pd.Timestamp(end_date)]

    SCOPED_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    scoped_path = SCOPED_INVENTORY_DIR / f"inventory_{start_date}_{end_date}.parquet"
    inventory.to_parquet(scoped_path)
    return scoped_path


def main() -> None:
    """
    Create an empty GeoZarr store and then populate it from the inventory parquet.
    """
    args = parse_args()
    store_path = args.store_path
    shuffle = BloscShuffle(args.shuffle)
    variable = args.variable
    inventory_parquet = build_scoped_inventory(args.start_date, args.end_date)
    print(f"Using inventory: {inventory_parquet}")
    print(f"Using variable: {variable}")

    grid = read_odim_grid_spec(str(EXAMPLE_HDF))

    dataset = create_empty_geozarr_single_variable_from_inventory(
        store_path=str(store_path),
        inventory_parquet=str(inventory_parquet),
        source_var=variable,
        grid=grid,
        out_var_name=variable,
        time_col=None,
        require_file=True,
        clevel=args.clevel,
        shuffle=shuffle,
    )
    print(dataset)

    summary = populate_geozarr_from_inventory_parquet(
        store_path=str(store_path),
        inventory_parquet=str(inventory_parquet),
        source_var=variable,
        out_var_name=variable,
        time_col=None,
        n_workers=N_WORKERS,
        chunksize=CHUNKSIZE,
    )
    print(summary)

if __name__ == "__main__":
    main()
