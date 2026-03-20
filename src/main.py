from __future__ import annotations

from pathlib import Path

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
STORE_PATH = Path("./BE-radclim-rain_rate.zarr")

SOURCE_VAR = "rate_qpe_mfb"
OUT_VAR_NAME = "rain_rate"

N_WORKERS = 16
CHUNKSIZE = 100


def main() -> None:
    """
    Create an empty GeoZarr store and then populate it from the inventory parquet.
    """
    grid = read_odim_grid_spec(str(EXAMPLE_HDF))

    dataset = create_empty_geozarr_single_variable_from_inventory(
        store_path=str(STORE_PATH),
        inventory_parquet=str(INVENTORY_PARQUET),
        source_var=SOURCE_VAR,
        grid=grid,
        out_var_name=OUT_VAR_NAME,
        time_col=None,
        require_file=True,
    )
    print(dataset)

    summary = populate_geozarr_from_inventory_parquet(
        store_path=str(STORE_PATH),
        inventory_parquet=str(INVENTORY_PARQUET),
        source_var=SOURCE_VAR,
        out_var_name=OUT_VAR_NAME,
        time_col=None,
        n_workers=N_WORKERS,
        chunksize=CHUNKSIZE,
    )
    print(summary)

if __name__ == "__main__":
    main()
