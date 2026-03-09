from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyproj import CRS, Transformer
from zarr.codecs import ZstdCodec

try:
    from zarr import consolidate_metadata
except ImportError:
    from zarr.convenience import consolidate_metadata


@dataclass(frozen=True)
class ODIMGridSpec:
    """
    Grid and CRS metadata extracted from an ODIM HDF file.

    Attributes
    ----------
    crs
        Parsed projected CRS.
    projdef
        Original projection definition string from the ODIM file.
    x
        1D x-coordinate array of cell centers.
    y
        1D y-coordinate array of cell centers.
    nx
        Number of grid cells along x.
    ny
        Number of grid cells along y.
    geotransform
        GDAL-style geotransform:
        (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
    data_path
        HDF path to the raster data array.
    what_path
        HDF path to the metadata group for the raster.
    """

    crs: CRS
    projdef: str
    x: np.ndarray
    y: np.ndarray
    nx: int
    ny: int
    geotransform: tuple[float, float, float, float, float, float]
    data_path: str
    what_path: str


def build_variable_attributes(variable_name: str) -> dict[str, str]:
    """
    Build CF- and project-style metadata for a RADCLIM variable.

    Parameters
    ----------
    variable_name
        Source variable name, e.g. ``rate_qpe_edk`` or ``acrr_qpe_edk_5m``.

    Returns
    -------
    dict[str, str]
        Attribute dictionary for the output Zarr variable.
        Returns an empty dictionary if the variable naming scheme is unknown.
    """
    name = variable_name.lower()

    if name.startswith("rate_"):
        quantity = "RATE"
        units = "kg m-2 h-1"
        standard_name = "rainfall_flux"
        base_name = "Rain rate"
    elif name.startswith("acrr_"):
        quantity = "ACRR"
        units = "kg m-2"
        standard_name = "precipitation_amount"
        base_name = "Accumulated precipitation amount"
    else:
        return {}

    product_code = None
    product_description = None
    if "_qpe" in name:
        product_code = "QPE"
        product_description = "Quantitative precipitation estimate"
    elif "_cap" in name:
        product_code = "CAP"
        product_description = "CAP (product type)"

    method_code = None
    method_description = None
    if "_edk" in name:
        method_code = "EDK"
        method_description = "External drift kriging"
    elif "_mfb" in name:
        method_code = "MFB"
        method_description = "Mean field bias correction"

    accumulation_period = None
    accumulation_text = None
    if quantity == "ACRR":
        if name.endswith("_5m"):
            accumulation_period = "PT5M"
            accumulation_text = "5 minutes"
        elif name.endswith("_1h"):
            accumulation_period = "PT1H"
            accumulation_text = "1 hour"

    long_name = base_name
    if product_code:
        long_name += f" ({product_code}"
        if method_code:
            long_name += f"_{method_code}"
        long_name += ")"
    elif method_code:
        long_name += f" ({method_code})"

    if accumulation_text:
        long_name += f" over {accumulation_text}"

    attrs = {
        "long_name": long_name,
        "units": units,
        "standard_name": standard_name,
        "quantity": quantity,
        "product": product_code,
        "product_description": product_description,
        "method": method_code,
        "method_description": method_description,
    }

    if accumulation_period:
        attrs["accumulation_period"] = accumulation_period

    return {key: value for key, value in attrs.items() if value is not None}


def read_odim_grid_spec(file_path: str, dataset: str = "dataset1") -> ODIMGridSpec:
    """
    Read ODIM HDF metadata and construct projected x/y coordinate arrays.

    Parameters
    ----------
    file_path
        Path to the ODIM HDF file.
    dataset
        Dataset group name inside the ODIM file.

    Returns
    -------
    ODIMGridSpec
        Grid geometry and CRS information derived from file metadata.
    """
    with h5py.File(file_path, "r") as handle:
        where_attrs = handle["where"].attrs

        projdef = where_attrs["projdef"]
        if isinstance(projdef, (bytes, np.bytes_)):
            projdef = projdef.decode("utf-8")

        ul_lon = float(where_attrs["UL_lon"])
        ul_lat = float(where_attrs["UL_lat"])
        ur_lon = float(where_attrs["UR_lon"])
        ur_lat = float(where_attrs["UR_lat"])
        ll_lon = float(where_attrs["LL_lon"])
        ll_lat = float(where_attrs["LL_lat"])

        data_path = f"{dataset}/data1/data"
        what_path = f"{dataset}/data1/what"

        ny, nx = handle[data_path].shape

    crs = CRS.from_user_input(projdef)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    x_ul, y_ul = transformer.transform(ul_lon, ul_lat)
    x_ur, y_ur = transformer.transform(ur_lon, ur_lat)
    x_ll, y_ll = transformer.transform(ll_lon, ll_lat)

    dx = (x_ur - x_ul) / nx
    dy = (y_ll - y_ul) / ny

    x = x_ul + (np.arange(nx) + 0.5) * dx
    y = y_ul + (np.arange(ny) + 0.5) * dy

    geotransform = (x_ul, dx, 0.0, y_ul, 0.0, dy)

    return ODIMGridSpec(
        crs=crs,
        projdef=projdef,
        x=x,
        y=y,
        nx=nx,
        ny=ny,
        geotransform=geotransform,
        data_path=data_path,
        what_path=what_path,
    )


def compute_lat_lon_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    projdef: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert projected 1D x/y coordinates into 2D latitude/longitude fields.

    Parameters
    ----------
    x
        1D x-coordinate array.
    y
        1D y-coordinate array.
    projdef
        Projection definition understood by pyproj.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two 2D arrays: ``(lat, lon)``.
    """
    crs = CRS.from_user_input(projdef)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    x_2d, y_2d = np.meshgrid(x, y)
    lon, lat = transformer.transform(x_2d.ravel(), y_2d.ravel())

    return lat.reshape(y_2d.shape), lon.reshape(x_2d.shape)


def _cell_contains_valid_path(value: Any) -> bool:
    """
    Check whether an inventory cell contains a usable file path or URI.

    Valid values are:
    - non-empty strings
    - lists/tuples containing at least one non-empty string

    Invalid values are:
    - None
    - NaN / NaT
    - empty strings
    - booleans
    """
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, (str, np.str_)):
        stripped = str(value).strip()
        return bool(stripped) and stripped.lower() not in {"nan", "none", "null"}

    if isinstance(value, (list, tuple)):
        return any(
            isinstance(item, (str, np.str_)) and bool(str(item).strip())
            for item in value
        )

    if isinstance(value, (bool, np.bool_)):
        return False

    return False


def extract_available_times_from_inventory(
    inventory_parquet: str,
    source_var: str,
    time_col: str | None = None,
) -> np.ndarray:
    """
    Extract timestamps for rows that truly contain source data.

    Unlike a simple time-column read, this function filters the inventory to
    keep only rows where ``source_var`` contains an actual path or URI.

    Parameters
    ----------
    inventory_parquet
        Path to the inventory parquet file.
    source_var
        Name of the column containing source file paths.
    time_col
        Optional explicit time column name. If omitted, the function first
        tries the index (when it is a DatetimeIndex), then a ``time`` column.

    Returns
    -------
    np.ndarray
        Sorted, unique timestamps as ``datetime64[ns]``.
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
        raise ValueError(
            f"source_var='{source_var}' not found in inventory parquet columns. "
            "Expected a variable column containing file paths or URIs."
        )

    if isinstance(times, pd.DatetimeIndex) and times.tz is not None:
        times = times.tz_localize(None)

    has_source_file = (
        inventory[source_var].map(_cell_contains_valid_path).to_numpy(dtype=bool)
    )
    filtered_times = pd.DatetimeIndex(times[has_source_file]).unique().sort_values()

    return filtered_times.values.astype("datetime64[ns]")


def infer_missing_timestamps_regular(time_values_ns: np.ndarray) -> np.ndarray:
    """
    Infer missing timestamps from a regular time series.

    The expected interval is taken as the smallest positive time difference.

    Parameters
    ----------
    time_values_ns
        Sorted time values as ``datetime64[ns]``.

    Returns
    -------
    np.ndarray
        Missing timestamps as ``datetime64[ns]``.
    """
    times = np.asarray(time_values_ns).astype("datetime64[ns]")
    if times.size < 2:
        return np.array([], dtype="datetime64[ns]")

    diffs = np.diff(times).astype("timedelta64[ns]")
    positive_diffs = diffs[diffs > np.timedelta64(0, "ns")]
    if positive_diffs.size == 0:
        return np.array([], dtype="datetime64[ns]")

    expected_interval = positive_diffs.min()
    expected_times = pd.date_range(
        start=times[0],
        end=times[-1],
        freq=pd.to_timedelta(expected_interval),
    ).values.astype("datetime64[ns]")

    missing_times = np.setdiff1d(expected_times, times).astype("datetime64[ns]")
    return np.unique(missing_times)


def create_empty_geozarr_single_variable_from_inventory(
    store_path: str,
    inventory_parquet: str,
    source_var: str,
    grid: ODIMGridSpec,
    out_var_name: str = "rain_rate",
    time_col: str | None = None,
    require_file: bool = True,
    chunks_time: int = 1,
    shards_time: int | None = 2016,
    consolidated: bool = True,
    clevel: int = 9,
    dtype: Any = np.float32,
    fill_value: float = np.nan,
    use_zarr_fill_value_as_mask: bool | None = None,
) -> xr.Dataset:
    """
    Create an empty Zarr v3 GeoZarr dataset for a single variable.

    The dataset contains:
    - one main data variable with dimensions ``(time, y, x)``
    - coordinate arrays ``time``, ``x``, ``y``
    - 2D ``lat`` and ``lon`` fields
    - scalar ``spatial_ref`` CRS variable
    - optional ``missing_times`` coordinate for regular time series gaps

    Parameters
    ----------
    store_path
        Output path of the Zarr store.
    inventory_parquet
        Path to the inventory parquet file.
    source_var
        Column name in the inventory containing source file paths.
    grid
        Grid specification extracted from an ODIM file.
    out_var_name
        Output variable name in the Zarr store.
    time_col
        Optional explicit time column in the inventory parquet.
    require_file
        Retained for backward compatibility. Currently unused.
    chunks_time
        Time chunk length for the main variable.
    shards_time
        Optional time shard length for Zarr v3 sharding.
        Set to ``None`` to disable sharding.
    consolidated
        Whether to consolidate Zarr metadata after writing.
    clevel
        Zstandard compression level.
    dtype
        Data type of the main output variable.
    fill_value
        Fill value used for unwritten cells.
    use_zarr_fill_value_as_mask
        Passed through to ``xr.open_zarr``.

    Returns
    -------
    xr.Dataset
        Opened xarray dataset backed by the newly created Zarr store.
    """
    del require_file  # kept only for API compatibility

    ny, nx = grid.ny, grid.nx

    times = extract_available_times_from_inventory(
        inventory_parquet=inventory_parquet,
        source_var=source_var,
        time_col=time_col,
    )
    if len(times) == 0:
        raise ValueError(
            "No timestamps found for the selected variable in the inventory parquet."
        )

    time_size = len(times)

    store = zarr.storage.LocalStore(store_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    root.attrs.update(
        {
            "Conventions": "CF-1.10",
            "license": "CC-BY-4.0",
            "institute": "Royal Meteorological Institute of Belgium",
            "mlcast_created_on": pd.Timestamp.now(tz="UTC").isoformat(),
            "mlcast_created_by": (
                "Simon De Kock, Lesley De Cruz "
                "<simon.de.kock@vub.be,lesley.decruz@meteo.be>"
            ),
            "mlcast_created_with": (
                "https://github.com/mlcast-community/mlcast-dataset-BE-RMI-radclim"
            ),
            "mlcast_dataset_version": "1.0",
            "mlcast_dataset_identifier": f"BE-radclim-{out_var_name}",
            "consistent_timestep_start": pd.Timestamp(times[0]).isoformat(),
            "coordinates": "time y x spatial_ref lat lon",
        }
    )

    def create_array(
        name: str,
        *,
        data: Any = None,
        shape: tuple[int, ...] | tuple[()] | None = None,
        dtype: Any = None,
        chunks: tuple[int, ...] | tuple[()] | None = None,
        fill_value: Any = None,
        dimension_names: tuple[str, ...] | tuple[()] | None = None,
        attrs: dict[str, Any] | None = None,
        compressors: tuple[Any, ...] | None = None,
        shards: tuple[int, ...] | None = None,
    ) -> Any:
        """
        Create a Zarr array while respecting the Zarr v3 rule that ``dtype``
        must not be passed when ``data`` is provided.
        """
        kwargs = {
            "name": name,
            "chunks": chunks,
            "fill_value": fill_value,
            "dimension_names": dimension_names,
            "compressors": compressors,
            "shards": shards,
            "overwrite": True,
        }

        if data is not None:
            kwargs["data"] = data
        else:
            kwargs["shape"] = shape
            kwargs["dtype"] = dtype

        array = root.create_array(**kwargs)

        if attrs:
            array.attrs.update(attrs)

        if dimension_names is not None:
            array.attrs.setdefault("_ARRAY_DIMENSIONS", list(dimension_names))

        return array

    create_array(
        "time",
        data=times.astype("datetime64[ns]"),
        chunks=(time_size,),
        dimension_names=("time",),
        attrs={"axis": "T"},
    )

    missing_times = infer_missing_timestamps_regular(times)
    if missing_times.size > 0:
        create_array(
            "missing_times",
            data=missing_times.astype("datetime64[ns]"),
            chunks=(min(len(missing_times), 1024),),
            dimension_names=("missing_times",),
            attrs={"long_name": "Missing timestamps in regular period"},
        )

    create_array(
        "y",
        data=np.asarray(grid.y),
        chunks=(ny,),
        dimension_names=("y",),
        attrs={"standard_name": "projection_y_coordinate", "axis": "Y"},
    )

    create_array(
        "x",
        data=np.asarray(grid.x),
        chunks=(nx,),
        dimension_names=("x",),
        attrs={"standard_name": "projection_x_coordinate", "axis": "X"},
    )

    lat, lon = compute_lat_lon_coordinates(grid.x, grid.y, grid.projdef)

    create_array(
        "lat",
        data=np.asarray(lat),
        chunks=(ny, nx),
        dimension_names=("y", "x"),
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )

    create_array(
        "lon",
        data=np.asarray(lon),
        chunks=(ny, nx),
        dimension_names=("y", "x"),
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )

    crs_wkt = grid.crs.to_wkt()
    spatial_ref = create_array(
        "spatial_ref",
        shape=(),
        dtype="i8",
        chunks=(),
        fill_value=0,
        dimension_names=(),
        attrs={
            **grid.crs.to_cf(),
            "spatial_ref": crs_wkt,
            "crs_wkt": crs_wkt,
            "projdef_original": grid.projdef,
            "GeoTransform": " ".join(map(str, grid.geotransform)),
        },
    )
    spatial_ref[...] = 0

    variable_attrs = build_variable_attributes(source_var)
    compressors = (ZstdCodec(level=clevel),)
    shards = (int(shards_time), ny, nx) if shards_time is not None else None

    create_array(
        out_var_name,
        shape=(time_size, ny, nx),
        dtype=dtype,
        chunks=(int(chunks_time), ny, nx),
        shards=shards,
        fill_value=fill_value,
        compressors=compressors,
        dimension_names=("time", "y", "x"),
        attrs={
            **variable_attrs,
            "grid_mapping": "spatial_ref",
            "radclim_source_variable": source_var,
        },
    )

    if consolidated:
        try:
            consolidate_metadata(store, zarr_format=3)
        except TypeError:
            consolidate_metadata(store)

    return xr.open_zarr(
        store_path,
        consolidated=consolidated,
        zarr_format=3,
        use_zarr_fill_value_as_mask=use_zarr_fill_value_as_mask,
    )
