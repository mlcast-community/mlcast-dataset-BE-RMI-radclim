from __future__ import annotations

import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd
import zarr
from filelock import FileLock
from tqdm import tqdm


@dataclass(frozen=True)
class WorkerConfig:
    """
    Configuration shared with multiprocessing workers.

    Attributes
    ----------
    store_path
        Path to the target Zarr store.
    out_var_name
        Name of the output variable in the Zarr store.
    dataset
        ODIM dataset group name, usually ``dataset1``.
    data_name
        ODIM data group name, usually ``data1``.
    strict_time_match
        Whether to raise an error if a timestamp is not found in the Zarr time axis.
    strict_shape_match
        Whether to raise an error if the input raster shape differs from the Zarr target shape.
    lock_path
        Path to the file lock used to serialize writes.
    """

    store_path: str
    out_var_name: str
    dataset: str
    data_name: str
    strict_time_match: bool
    strict_shape_match: bool
    lock_path: str


# Shared per-process configuration, initialized via _init_worker().
_WORKER_CONFIG: WorkerConfig | None = None


def read_odim_data_as_float32(
    hdf_path: str,
    dataset: str = "dataset1",
    data_name: str = "data1",
    out_dtype: Any = np.float32,
) -> np.ndarray:
    """
    Read an ODIM HDF raster and convert it to physical units.

    The ODIM convention uses:

        physical_value = offset + gain * raw_value

    If present, the ``nodata`` and ``undetect`` values are masked to NaN.

    Parameters
    ----------
    hdf_path
        Path to the ODIM HDF file.
    dataset
        ODIM dataset group name.
    data_name
        ODIM data group name.
    out_dtype
        Output NumPy dtype, typically ``np.float32``.

    Returns
    -------
    np.ndarray
        2D raster in physical units with invalid values replaced by NaN.
    """
    data_path = f"{dataset}/{data_name}/data"
    metadata_path = f"{dataset}/{data_name}/what"

    with h5py.File(hdf_path, "r") as handle:
        raw = handle[data_path][...]
        attrs = handle[metadata_path].attrs

        gain = float(attrs.get("gain", 1.0))
        offset = float(attrs.get("offset", 0.0))
        nodata = attrs.get("nodata", None)
        undetect = attrs.get("undetect", None)

    raw = np.asarray(raw)

    invalid_mask = np.zeros(raw.shape, dtype=bool)
    if nodata is not None:
        invalid_mask |= raw == nodata
    if undetect is not None:
        invalid_mask |= raw == undetect

    data = (offset + gain * raw.astype(out_dtype)).astype(out_dtype, copy=False)

    if np.any(invalid_mask):
        data = data.astype(out_dtype, copy=False)
        data[invalid_mask] = np.nan

    return data


def _is_valid_hdf_path(value: Any) -> bool:
    """
    Return whether a value looks like a usable HDF file path.
    """
    if value is None:
        return False

    if isinstance(value, (str, np.str_)):
        text = str(value).strip()
        return bool(text) and text.lower() not in {"nan", "none", "null"}

    return False


def extract_first_path(cell: Any) -> str | None:
    """
    Extract the first valid path from an inventory cell.

    Supported inputs are:
    - a single string path
    - a list or tuple of paths

    Invalid, empty, or null-like values return ``None``.
    """
    if cell is None:
        return None

    try:
        if pd.isna(cell):
            return None
    except Exception:
        pass

    if isinstance(cell, (str, np.str_)):
        text = str(cell).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return text

    if isinstance(cell, (list, tuple)):
        for item in cell:
            if isinstance(item, (str, np.str_)):
                text = str(item).strip()
                if text and text.lower() not in {"nan", "none", "null"}:
                    return text

    return None


def append_odim_timestep_to_geozarr(
    store_path: str,
    hdf_path: str | None,
    time_ns: np.datetime64,
    out_var_name: str = "rain_rate",
    dataset: str = "dataset1",
    data_name: str = "data1",
    strict_time_match: bool = True,
    strict_shape_match: bool = True,
) -> int:
    """
    Write one ODIM raster into the correct ``(time, y, x)`` slot of an existing Zarr v3 store.

    Parameters
    ----------
    store_path
        Path to the target Zarr store.
    hdf_path
        Path to the ODIM HDF file. If invalid or empty, the write is skipped.
    time_ns
        Timestamp to place in the Zarr store. Must resolve to ``datetime64[ns]``.
    out_var_name
        Name of the data variable in the Zarr store.
    dataset
        ODIM dataset group name.
    data_name
        ODIM data group name.
    strict_time_match
        If True, raise an error when the timestamp is not present in the Zarr time coordinate.
        If False, return ``-1`` instead.
    strict_shape_match
        If True, raise an error when the source raster shape does not match the Zarr target.

    Returns
    -------
    int
        Written time index in the Zarr array, or ``-1`` when skipped.
    """
    if not _is_valid_hdf_path(hdf_path):
        return -1

    timestamp = np.asarray(time_ns).astype("datetime64[ns]").reshape(())

    store = zarr.storage.LocalStore(store_path)
    root = zarr.open_group(store, mode="r+", zarr_format=3)

    if "time" not in root:
        raise KeyError("Zarr store has no 'time' coordinate array.")
    if out_var_name not in root:
        raise KeyError(f"Zarr store has no data variable '{out_var_name}'.")

    zarr_times = np.asarray(root["time"][:]).astype("datetime64[ns]")
    time_index = int(np.searchsorted(zarr_times, timestamp))

    if time_index >= zarr_times.size or zarr_times[time_index] != timestamp:
        if strict_time_match:
            if zarr_times.size == 0:
                raise ValueError("Zarr 'time' array is empty.")
            nearest_index = int(np.argmin(np.abs(zarr_times - timestamp)))
            raise KeyError(
                f"time {timestamp} not found in Zarr 'time' coordinate. "
                f"Nearest is {zarr_times[nearest_index]} at index {nearest_index}."
            )
        return -1

    frame = read_odim_data_as_float32(
        hdf_path=hdf_path,
        dataset=dataset,
        data_name=data_name,
        out_dtype=np.float32,
    )

    target_array = root[out_var_name]
    _, target_ny, target_nx = target_array.shape

    if strict_shape_match and frame.shape != (target_ny, target_nx):
        raise ValueError(
            f"ODIM frame shape {frame.shape} does not match target "
            f"(ny, nx)=({target_ny}, {target_nx})."
        )

    target_array[time_index, :, :] = frame
    return time_index


def _read_inventory_times_and_paths(
    inventory_parquet: str,
    source_var: str,
    time_col: str | None = None,
) -> list[tuple[np.datetime64, str | None]]:
    """
    Read an inventory parquet and return ``(timestamp, path)`` pairs for one variable.

    Parameters
    ----------
    inventory_parquet
        Path to the inventory parquet file.
    source_var
        Inventory column containing the source HDF paths.
    time_col
        Optional explicit time column name. If omitted, the function first tries
        the DataFrame index when it is a DatetimeIndex, then a ``time`` column.

    Returns
    -------
    list[tuple[np.datetime64, str | None]]
        List of normalized nanosecond timestamps and extracted HDF paths.
    """
    inventory = pd.read_parquet(inventory_parquet)

    if time_col is None:
        if isinstance(inventory.index, pd.DatetimeIndex):
            times = inventory.index
        elif "time" in inventory.columns:
            times = pd.to_datetime(inventory["time"], utc=False)
        else:
            raise ValueError(
                "Could not infer times: parquet has no DatetimeIndex and no "
                "'time' column. Pass time_col='...'."
            )
    else:
        if time_col not in inventory.columns:
            raise ValueError(f"time_col='{time_col}' not found in parquet columns.")
        times = pd.to_datetime(inventory[time_col], utc=False)

    if source_var not in inventory.columns:
        raise ValueError(f"Variable '{source_var}' not found in parquet columns.")

    items: list[tuple[np.datetime64, str | None]] = []
    for timestamp, cell in zip(times, inventory[source_var].values):
        hdf_path = extract_first_path(cell)
        timestamp_ns = np.datetime64(pd.Timestamp(timestamp).to_datetime64()).astype(
            "datetime64[ns]"
        )
        items.append((timestamp_ns, hdf_path))

    return items


def _init_worker(config: WorkerConfig) -> None:
    """
    Initialize per-process worker configuration.
    """
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _worker_write_one(
    item: tuple[np.datetime64, str | None],
) -> tuple[str, np.datetime64, str | None, int | None, str | None]:
    """
    Write one inventory item from a multiprocessing worker.

    Returns
    -------
    tuple
        ``(status, timestamp, hdf_path, time_index, message)``

    Status is one of:
    - ``"written"``
    - ``"skipped"``
    - ``"error"``
    """
    global _WORKER_CONFIG

    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration was not initialized.")

    timestamp_ns, hdf_path = item

    if hdf_path is None:
        return ("skipped", timestamp_ns, None, None, None)

    try:
        lock = FileLock(_WORKER_CONFIG.lock_path)
        with lock:
            time_index = append_odim_timestep_to_geozarr(
                store_path=_WORKER_CONFIG.store_path,
                hdf_path=hdf_path,
                time_ns=timestamp_ns,
                out_var_name=_WORKER_CONFIG.out_var_name,
                dataset=_WORKER_CONFIG.dataset,
                data_name=_WORKER_CONFIG.data_name,
                strict_time_match=_WORKER_CONFIG.strict_time_match,
                strict_shape_match=_WORKER_CONFIG.strict_shape_match,
            )

        if time_index == -1:
            return ("skipped", timestamp_ns, hdf_path, None, "Skipped by writer (-1).")

        return ("written", timestamp_ns, hdf_path, int(time_index), None)

    except Exception as exc:
        return ("error", timestamp_ns, hdf_path, None, repr(exc))


def populate_geozarr_from_inventory_parquet(
    *,
    store_path: str,
    inventory_parquet: str,
    source_var: str,
    out_var_name: str = "rain_rate",
    time_col: str | None = None,
    dataset: str = "dataset1",
    data_name: str = "data1",
    n_workers: int | None = None,
    mp_start_method: str = "spawn",
    chunksize: int = 50,
    strict_time_match: bool = True,
    strict_shape_match: bool = True,
    show_errors: int = 20,
) -> dict[str, Any]:
    """
    Populate an existing GeoZarr store from an inventory parquet in parallel.

    The inventory is read row by row for one variable. Each worker attempts to
    load an ODIM HDF file and write it to the correct time slot in the target Zarr store.

    Notes
    -----
    Writes are serialized with a file lock to avoid store corruption. In the current
    design, reading and scaling the HDF data also happens while the lock is held, so
    write safety is prioritized over maximum throughput.

    Parameters
    ----------
    store_path
        Path to the target Zarr store.
    inventory_parquet
        Path to the inventory parquet file.
    source_var
        Inventory column containing source HDF paths.
    out_var_name
        Output data variable name in the Zarr store.
    time_col
        Optional explicit time column name.
    dataset
        ODIM dataset group name.
    data_name
        ODIM data group name.
    n_workers
        Number of worker processes. Defaults to ``cpu_count() - 1`` with a minimum of 1.
    mp_start_method
        Multiprocessing start method, typically ``"spawn"``.
    chunksize
        Chunk size for ``imap_unordered``.
    strict_time_match
        Whether to raise an error when a timestamp is missing from the Zarr store.
    strict_shape_match
        Whether to raise an error when a source raster shape mismatches the target.
    show_errors
        Maximum number of sample errors to include in the returned summary.

    Returns
    -------
    dict[str, Any]
        Summary statistics and a small sample of encountered errors.
    """
    items = _read_inventory_times_and_paths(
        inventory_parquet=inventory_parquet,
        source_var=source_var,
        time_col=time_col,
    )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    lock_path = store_path.rstrip("/").rstrip("\\") + ".write.lock"
    worker_config = WorkerConfig(
        store_path=store_path,
        out_var_name=out_var_name,
        dataset=dataset,
        data_name=data_name,
        strict_time_match=strict_time_match,
        strict_shape_match=strict_shape_match,
        lock_path=lock_path,
    )

    ctx = mp.get_context(mp_start_method)

    written_count = 0
    skipped_count = 0
    error_count = 0
    error_samples: list[tuple[str, str | None, str | None]] = []

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(worker_config,),
    ) as pool:
        results = pool.imap_unordered(_worker_write_one, items, chunksize=chunksize)

        for status, timestamp_ns, hdf_path, _, message in tqdm(
            results,
            total=len(items),
            desc=f"Populating {source_var}",
        ):
            if status == "written":
                written_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                error_count += 1
                if len(error_samples) < show_errors:
                    error_samples.append((str(timestamp_ns), hdf_path, message))

    return {
        "total_rows": len(items),
        "written": written_count,
        "skipped": skipped_count,
        "errors": error_count,
        "error_samples": error_samples,
        "lock_path": lock_path,
        "n_workers": n_workers,
        "chunksize": chunksize,
    }
