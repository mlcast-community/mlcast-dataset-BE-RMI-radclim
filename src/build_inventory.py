from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ARCHIVE_ROOT = Path("/RADCLIM/archive")

# Inclusive start, exclusive end
START_5M = "2016-01-01 00:00:00"
END_5M = "2024-01-01 00:00:00"

# Deliberate broad range to capture possibility of data outside the expected range
START_1H = "2016-01-01 00:00:00"
END_1H = "2024-01-01 00:00:00"

OUT_5M = Path("radclim_inventory_5min.parquet")
OUT_1H = Path("radclim_inventory_1h.parquet")


@dataclass(frozen=True)
class DatasetSpec:
    """
    Description of one RADCLIM dataset to track in the inventory.

    Attributes
    ----------
    column_name
        Name of the column in the output inventory table.
    product_kind
        Product group used in the archive path, typically ``rate`` or ``acrr``.
    dataset_name
        Dataset token used in both the directory name and filename.
    """

    column_name: str
    product_kind: str
    dataset_name: str


DATASETS_5MIN: list[DatasetSpec] = [
    DatasetSpec("rate_cap", "rate", "cap"),
    DatasetSpec("rate_qpe", "rate", "qpe"),
    DatasetSpec("rate_cap_edk", "rate", "cap_edk"),
    DatasetSpec("rate_qpe_edk", "rate", "qpe_edk"),
    DatasetSpec("rate_cap_mfb", "rate", "cap_mfb"),
    DatasetSpec("rate_qpe_mfb", "rate", "qpe_mfb"),
    DatasetSpec("acrr_cap_5m", "acrr", "cap_5m"),
    DatasetSpec("acrr_qpe_5m", "acrr", "qpe_5m"),
    DatasetSpec("acrr_cap_edk_5m", "acrr", "cap_edk_5m"),
    DatasetSpec("acrr_qpe_edk_5m", "acrr", "qpe_edk_5m"),
    DatasetSpec("acrr_cap_mfb_5m", "acrr", "cap_mfb_5m"),
    DatasetSpec("acrr_qpe_mfb_5m", "acrr", "qpe_mfb_5m"),
]

DATASETS_1H: list[DatasetSpec] = [
    DatasetSpec("acrr_cap_1h", "acrr", "cap_1h"),
    DatasetSpec("acrr_qpe_1h", "acrr", "qpe_1h"),
    DatasetSpec("acrr_cap_edk_1h", "acrr", "cap_edk_1h"),
    DatasetSpec("acrr_qpe_edk_1h", "acrr", "qpe_edk_1h"),
    DatasetSpec("acrr_cap_mfb_1h", "acrr", "cap_mfb_1h"),
    DatasetSpec("acrr_qpe_mfb_1h", "acrr", "qpe_mfb_1h"),
]


def build_filename(
    timestamp: pd.Timestamp, product_kind: str, dataset_name: str
) -> str:
    """
    Build the expected RADCLIM filename for a timestamp and dataset.

    Example
    -------
    ``20231101085500.rad.best.comp.rate.cap.hdf``
    """
    return f"{timestamp:%Y%m%d%H%M%S}.rad.best.comp.{product_kind}.{dataset_name}.hdf"


def build_expected_path(
    timestamp: pd.Timestamp,
    product_kind: str,
    dataset_name: str,
) -> Path:
    """
    Build the full expected archive path for one RADCLIM file.

    Expected layout
    ---------------
    archive/YYYY/MM/DD/best/comp/<kind>/<dataset>/hdf/<filename>
    """
    return (
        ARCHIVE_ROOT
        / f"{timestamp:%Y}"
        / f"{timestamp:%m}"
        / f"{timestamp:%d}"
        / "best"
        / "comp"
        / product_kind
        / dataset_name
        / "hdf"
        / build_filename(timestamp, product_kind, dataset_name)
    )


def build_inventory_parquet(
    start: str,
    end: str,
    freq: str,
    datasets: list[DatasetSpec],
    output_path: Path,
) -> None:
    """
    Build a parquet inventory containing expected RADCLIM file paths.

    For each timestamp in the requested time range, this function checks whether
    the expected file exists for each dataset. Existing files are stored as full
    paths; missing files are stored as empty strings.

    Parameters
    ----------
    start
        Inclusive start timestamp.
    end
        Exclusive end timestamp.
    freq
        Timestamp frequency, e.g. ``"5min"`` or ``"1h"``.
    datasets
        Dataset specifications to include as columns.
    output_path
        Output parquet path.
    """
    timestamps = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    inventory = pd.DataFrame(index=timestamps)
    inventory.index.name = "timestamp"

    total_datasets = len(datasets)

    for dataset_idx, dataset in enumerate(datasets, start=1):
        print(
            f"[{dataset_idx}/{total_datasets}] checking "
            f"{dataset.column_name} ({dataset.product_kind}/{dataset.dataset_name}) ..."
        )

        inventory[dataset.column_name] = [
            (
                str(file_path)
                if (
                    file_path := build_expected_path(
                        ts, dataset.product_kind, dataset.dataset_name
                    )
                ).is_file()
                else ""
            )
            for ts in timestamps
        ]

    inventory.to_parquet(output_path, index=True)
    print(f"Wrote: {output_path} (rows={len(inventory):,}, cols={inventory.shape[1]})")


def main() -> None:
    """Build both the 5-minute and 1-hour RADCLIM inventory parquet files."""
    build_inventory_parquet(
        start=START_5M,
        end=END_5M,
        freq="5min",
        datasets=DATASETS_5MIN,
        output_path=OUT_5M,
    )

    build_inventory_parquet(
        start=START_1H,
        end=END_1H,
        freq="1h",
        datasets=DATASETS_1H,
        output_path=OUT_1H,
    )


if __name__ == "__main__":
    main()
