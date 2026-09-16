"""Convert an ND2 time-lapse stack to 3DeeCellTracker TIFF slices.

The default output layout is:

    output_dir/
      data/
        raw_t0001_z0001.tif
        raw_t0001_z0002.tif
        raw_t0002_z0001.tif
        ...

For multi-channel ND2 files, export one channel with --channel, or export every
channel into separate channel folders with --all-channels. Files with a Z axis
are exported as full z-stacks. Files without a Z axis are exported as one
z-slice per timepoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from tifffile import imwrite


TRACKER_AXES = ("T", "C", "Z", "Y", "X")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ND2 file to 3DeeCellTracker-compatible TIFF slices."
    )
    parser.add_argument("input_nd2", type=Path, help="Path to the input .nd2 file.")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output dataset folder. TIFF slices are written under output_dir/data.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Zero-based channel index to export. Ignored when --all-channels is used.",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Export each channel to output_dir/channel_000/data, channel_001/data, etc.",
    )
    parser.add_argument(
        "--prefix",
        default="raw",
        help="Filename prefix. Default: raw, producing raw_t0001_z0001.tif.",
    )
    parser.add_argument(
        "--time-digits",
        type=int,
        default=4,
        help="Number of digits for time indices. Default: 4.",
    )
    parser.add_argument(
        "--z-digits",
        type=int,
        default=4,
        help="Number of digits for z-slice indices. Default: 4.",
    )
    parser.add_argument(
        "--time-start",
        type=int,
        default=1,
        help="First exported time index in filenames. Default: 1.",
    )
    parser.add_argument(
        "--z-start",
        type=int,
        default=1,
        help="First exported z index in filenames. Default: 1.",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=None,
        help="Optional limit for quick tests, e.g. --max-timepoints 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIFF files.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print ND2 dimension metadata and exit without exporting TIFF files.",
    )
    return parser.parse_args()


def normalize_to_tczyx(array: np.ndarray, axes: Sequence[str]) -> np.ndarray:
    """Return image data in T,C,Z,Y,X order, adding singleton missing axes."""
    axes = list(axes)
    data = np.asarray(array)

    if data.ndim != len(axes):
        raise ValueError(
            f"ND2 axis metadata has {len(axes)} axes {axes}, but array has "
            f"{data.ndim} dimensions with shape {data.shape}."
        )

    for axis in list(axes):
        if axis not in TRACKER_AXES:
            axis_index = axes.index(axis)
            if data.shape[axis_index] != 1:
                raise ValueError(
                    f"Unsupported ND2 axis {axis!r} with size {data.shape[axis_index]}. "
                    "Split this dimension before conversion, or extend the converter."
                )
            data = np.squeeze(data, axis=axis_index)
            axes.pop(axis_index)

    for axis in TRACKER_AXES:
        if axis not in axes:
            data = np.expand_dims(data, axis=data.ndim)
            axes.append(axis)

    transpose_order = [axes.index(axis) for axis in TRACKER_AXES]
    return np.transpose(data, transpose_order)


def write_channel(
    data_tzyx: np.ndarray,
    output_data_dir: Path,
    prefix: str,
    time_digits: int,
    z_digits: int,
    time_start: int,
    z_start: int,
    overwrite: bool,
) -> int:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for t_index in range(data_tzyx.shape[0]):
        t_label = t_index + time_start
        for z_index in range(data_tzyx.shape[1]):
            z_label = z_index + z_start
            output_path = output_data_dir / (
                f"{prefix}_t{t_label:0{time_digits}d}_z{z_label:0{z_digits}d}.tif"
            )
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace files."
                )
            imwrite(output_path, data_tzyx[t_index, z_index])
            written += 1

    return written


def main() -> None:
    args = parse_args()

    try:
        import nd2
    except ImportError as exc:
        raise SystemExit(
            "The 'nd2' package is required. Install it with:\n"
            "  python -m pip install nd2\n"
            "or update the Conda environment from environment.yml."
        ) from exc

    with nd2.ND2File(args.input_nd2) as nd2_file:
        sizes = dict(nd2_file.sizes)
        axes = tuple(sizes.keys())
        if args.inspect:
            print("ND2 axes:", axes)
            print("ND2 sizes:", json.dumps(sizes, indent=2))
            if "Z" not in sizes:
                print("Warning: no Z axis was found. Conversion will create one z-slice per timepoint.")
            print("Use --channel N to export one channel, or --all-channels to export each channel separately.")
            return
        if "Z" not in sizes:
            print("Warning: no Z axis was found. Conversion will create one z-slice per timepoint.")
        data = normalize_to_tczyx(nd2_file.asarray(), axes)

    if args.output_dir is None:
        raise SystemExit("output_dir is required unless --inspect is used.")

    if args.max_timepoints is not None:
        data = data[: args.max_timepoints]

    channel_count = data.shape[1]
    if args.all_channels:
        channel_indices = list(range(channel_count))
    else:
        if args.channel < 0 or args.channel >= channel_count:
            raise ValueError(
                f"Channel {args.channel} is out of range. ND2 has {channel_count} channel(s)."
            )
        channel_indices = [args.channel]

    summary = {
        "input_nd2": str(args.input_nd2),
        "original_sizes": sizes,
        "normalized_shape_tczyx": list(data.shape),
        "exported_channels": channel_indices,
        "filename_pattern": f"{args.prefix}_t%0{args.time_digits}d_z%0{args.z_digits}d.tif",
    }

    total_written = 0
    for channel_index in channel_indices:
        if args.all_channels:
            output_data_dir = args.output_dir / f"channel_{channel_index:03d}" / "data"
        else:
            output_data_dir = args.output_dir / "data"

        total_written += write_channel(
            data[:, channel_index],
            output_data_dir=output_data_dir,
            prefix=args.prefix,
            time_digits=args.time_digits,
            z_digits=args.z_digits,
            time_start=args.time_start,
            z_start=args.z_start,
            overwrite=args.overwrite,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "conversion_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {total_written} TIFF files.")
    print(f"Metadata: {metadata_path}")
    if args.all_channels:
        print("Notebook path pattern example: output_dir/channel_000/data/*t%04d*.tif")
    else:
        print("Notebook path pattern example: output_dir/data/*t%04d*.tif")


if __name__ == "__main__":
    main()
