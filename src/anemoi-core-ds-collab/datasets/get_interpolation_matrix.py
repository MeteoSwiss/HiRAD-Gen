"""This script computes an interpolation matrix between two grids, and stores
it in a .mat.npz file.

Scipy's nearest neighbour or linear interpolation method can be used.
In the case of the nearest neighbour interpolation, the number of neighbour
points in the source grid, and the power involved in the weighting can be
chosen by the user.

The user can provide a lonlat_box, that will allow to interpolate the source
field only to a subdomain of the target grid. Note that in this case, the
target cropped matrix is not stored to a file (no avoid storing multiple
files). This lonlat_box option has no effect if this script is used in
stand-alone mode, it is only useful when the user wants to get a cropped
interpolation matrix to interpolate on-the-fly.

Important notice : all the longitudes / latitudes should be in degrees.
The longitude convention (-180~180 or 0~360) should be the same for all
grids and for the specified lonlat box, if any. The convention can be
changed for all longitude fields that are used. This can be useful if you
need to interpolate in a domain containing the international date line
(if the convention used is initially -180~180), or if you need to
interpolate in a domain containing the Greenwich meridian (if the
convention is initially 0~360)

Usage:
puv run get-interpolation-matrix --gridf_in --gridf_out
    [--wgt_dir --num_nn --power_nn --lonlat_box --lon_conv --overwrite]

Arguments:
  --gridf_in            Name of the source grid file (mandatory). This should be
                        a .npz file, with name beginning with 'grid_'
  --gridf_out           Name of the target grid file (mandatory). This should be
                        a .npz file, with name beginning with 'grid_'
  --wgt_dir             Name of the directory where matrix files can be found
                        (default : where the script is run)
  --wgt_file            Name of the output matrix .mat.npz file (default is
                        {gridn_in}_to_{gridn_out}_linear.mat.npz)
  --method              Interpolation method used: 'nearest' or 'linear' (default)
  --num_nn              If the nearest neighbours interpolation is used, number
                        of neighbours taken into account (default : 1)
  --power_nn            The weights of the interpolation are proportional
                        to 1/d**power, where d is the distance of target grid
                        (nearest neighbours interpolation method)
                        points to source grid points (default : 1)
  --lonlat_box          The limits of the lon/lat box should be provided
                        in degrees, and in the following order :
                        lon_min, lon_max, lat_min, lat_max
  --lon_conv            Force a convention change for the longitudes
                        (-180~180 or 0~360).
  --overwrite           Overwrite matrix file (default : False)
  -h, --help            Show the help message and exit (facultative)

Example:
  puv run get-interpolation-matrix
    --gridf_in grid_arpege-eurat01-0p1.npz --gridf_out grid_arome-1s40.npz
    --method nearest --num_nn 10 --power_nn 1.5 --lonlat_box -12 16 37.5 55.4
"""

# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------------
# IMPORT PACKAGES
# -----------------------------------------------------------------------------
import argparse
import os
import warnings

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial import Delaunay, cKDTree

FIELD_NAMES = {
    "latitude": ["latitude", "latitudes", "lat", "lats", "y", "Y"],
    "longitude": ["longitude", "longitudes", "lon", "lons", "x", "X"],
}


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------
# Search for a field in a dictionary
def search_field(
    d: dict, field: str, field_names: dict = FIELD_NAMES, filename: str = ""
):
    avail_field_names = list(
        set(d.keys()).intersection(field_names[field])
    )  # finding which field names are in the keys of d
    if len(avail_field_names) == 0:
        raise KeyError(
            f"No field name among {field_names[field]} can be found "
            f"in the input {filename} keys."
        )
    if len(avail_field_names) > 1:
        warnings.warn(
            f"Several field names ({avail_field_names}) are found in "
            f"the input {filename} keys.\nUsing {avail_field_names[0]}.",
            stacklevel=2,
        )
    return d[avail_field_names[0]]


# Get grid name from grid file name
def get_grid_name(gridf):
    # Relative file name
    gridrf = os.path.basename(gridf)
    if not gridrf.startswith("grid_"):
        print(f"Error. Wrong file name {gridrf} : should start with 'grid_'")
        raise SystemExit
    if not gridrf.endswith(".npz"):
        print(f"Error. Wrong file name {gridrf} : should end with '.npz'")
        raise SystemExit
    gridn = gridrf.replace("grid_", "").replace(".npz", "")
    return gridn


# Read lon, lat from an npz file
def read_grid(gridf: str, lon_conv: str | None = None):
    g = np.load(gridf, allow_pickle=True)
    # Search for a latitude field
    lats = search_field(g, "latitude", filename=gridf)
    if lats.max() > 90:
        raise ValueError(f"Error. {gridf} : found latitude > 90°")
    if lats.min() < -90:
        raise ValueError(f"Error. {gridf} : found latitude < -90°")
    # Search for a longitude field
    lons = search_field(g, "longitude", filename=gridf)
    if lons.min() < 0:
        if lons.max() > 180:
            raise ValueError("❌ Longitude convention unclear: found longitude values both < 0 and > 180.")
        else:
            print(f'⚠️ Detecting -180~180° longitude convention (min: {lons.min()}, max {lons.max()})')
    elif lons.max() > 180:
        print(f'⚠️ Detecting 0~360° longitude convention (min: {lons.min()}, max {lons.max()})')
    else:
        print(f'⚠️ Detecting -180~180° longitude convention (min: {lons.min()}, max {lons.max()})')

    if lon_conv is not None:
        print(f"Changing longitude convention to {lon_conv}")
        if lon_conv == "0~360":
            lons[lons < -180.0] = lons[lons < -180.0] + 360.0
        elif lon_conv == "-180~180":
            lons[lons >= 180.0] = lons[lons >= 180.0] - 360.0
    points = np.stack((lons, lats), axis=-1)

    return points


# Check that target domain is inside source domain
# (this check is not rigorous in the case of non-regular grids, just informative)
def check_target_in_source_domain(points_in, points_out):
    """Check that target domain is inside source domain.
    This is useful in the case of 'linear' interpolation method
    (there is a risk that the script fails at computing the interpolation
    matrix if the target domain is bigger than the source domain)
    """

    # Input grid
    lon1d_in, lat1d_in = np.split(points_in, 2, axis=-1)

    # Output grid
    lon1d_out, lat1d_out = np.split(points_out, 2, axis=-1)

    if lon1d_out.min() < lon1d_in.min():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The western boundary of the target grid "
            f"({lon1d_out.min()}) is further west than that of the "
            f"source grid ({lon1d_in.min()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if lon1d_out.max() > lon1d_in.max():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The eastern boundary of the target grid "
            f"({lon1d_out.max()}) is further east than that of the "
            f"source grid ({lon1d_in.max()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if lat1d_out.min() < lat1d_in.min():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The southern boundary of the target grid "
            f"({lat1d_out.min()}) is further south than that of the "
            f"source grid ({lat1d_in.min()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if lat1d_out.max() > lat1d_in.max():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The northern boundary of the target grid "
            f"({lat1d_out.max()}) is further north than that of the "
            f"source grid ({lat1d_in.max()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )


def check_lonlat_box(gridf_in, gridf_out, lonlat_box, lon_conv):
    # Cropping lon/lat box
    lon_min, lon_max, lat_min, lat_max = lonlat_box

    # Input grid
    lon1d_in, lat1d_in = read_grid(gridf_in, lon_conv).T

    # Output grid
    lon1d_out, lat1d_out = read_grid(gridf_out, lon_conv).T

    # Message about domain cropping
    print(
        "Source domain - lonlat_box: "
        f"\n\tlon_min = {lon1d_in.min()}\n\tlon_max = {lon1d_in.max()}"
        f"\n\tlat_min = {lat1d_in.min()}\n\tlat_max = {lat1d_in.max()}"
    )

    print(
        "Target domain - lonlat_box: "
        f"\n\tlon_min = {lon1d_out.min()}\n\tlon_max = {lon1d_out.max()}"
        f"\n\tlat_min = {lat1d_out.min()}\n\tlat_max = {lat1d_out.max()}"
    )

    print(
        "Cropping domain - lonlat_box: "
        f"\n\tlon_min = {lon_min}\n\tlon_max = {lon_max}"
        f"\n\tlat_min = {lat_min}\n\tlat_max = {lat_max}"
    )

    # Check lonlat box
    assert lon_max > lon_min, f"lon_max ({lon_max}) should be >= lon_min ({lon_min})."
    assert lat_max > lat_min, f"lat_max ({lat_max}) should be >= lat_min ({lat_min})."
    assert lat_min >= -90, f"lat_min ({lat_min}) should be >= -90°."
    assert lat_max <= 90, f"lat_max ({lat_max}) should be <= 90°"

    # Issue warnings about the lonlat box
    for (lon, lat), dom in zip(
        [(lon1d_in, lat1d_in), (lon1d_out, lat1d_out)],
        ["source", "target"],
        strict=False,
    ):
        if lon_min < lon.min():
            warnings.warn(
                f"\nCrop warning: lon_min ({lon_min}) is further west than "
                f"the western boundary of the {dom} grid ({lon.min()})",
                stacklevel=2,
            )
        if lon_max > lon.max():
            warnings.warn(
                f"\nCrop warning: lon_max ({lon_max}) is further east than "
                f"the eastern boundary of the {dom} grid ({lon.max()})",
                stacklevel=2,
            )
        if lat_min < lat.min():
            warnings.warn(
                f"\nCrop warning: lat_min ({lat_min}) is further south than "
                f"the southern boundary of the {dom} grid ({lat.min()})",
                stacklevel=2,
            )
        if lat_max > lat.max():
            warnings.warn(
                f"\nCrop warning: lat_max ({lat_max}) is further north than "
                f"the northern boundary of the {dom} grid ({lat.max()})",
                stacklevel=2,
            )


# Computes crop mask
def cropmsk(gridf, lonlat_box, lon_conv):
    # Cropping lon/lat box
    lon_min, lon_max, lat_min, lat_max = lonlat_box

    # Coordinates from the grid
    lon1d, lat1d = read_grid(gridf, lon_conv).T

    # Define crop mask
    cropmsk = (
        (lon1d > lon_min) & (lon1d <= lon_max) & (lat1d >= lat_min) & (lat1d <= lat_max)
    )

    return cropmsk


# Create a sparse interpolation matrix, to be compliant with Anemoi
# (and to save memory !)
def build_sparse_interpolation_matrix(indices, weights, N_in):
    """Build a sparse interpolation matrix.

    Parameters:
    - indices: np.ndarray of shape (N_out, N_neighbours)
    - weights: np.ndarray of shape (N_out, N_neighbours)
    - N_in: int, total number of source points

    Returns:
    - csr_matrix of shape (N_out, N_in)
    """
    N_out, N_neighbours = indices.shape
    rows = np.repeat(np.arange(N_out), N_neighbours)
    cols = indices.flatten()
    data = weights.flatten()

    # Keep only valid column indices
    valid_mask = (cols >= 0) & np.isfinite(data) & (data != 0.0)

    rows = rows[valid_mask]
    cols = cols[valid_mask]
    data = data[valid_mask]

    matrix = csr_matrix((data, (rows, cols)), shape=(N_out, N_in))

    return matrix


# In the nearest-neighbour case, read the interpolation weights file
# or compute them and save them in a file.
def get_iw_nn(
    gridf_in: str,
    gridf_out: str,
    wgt_dir: str,
    wgt_file: str,
    lonlat_box: list,
    lon_conv: str,
    overwrite: bool = False,
    method: str = "nearest",
    nn: int = 1,
    power: float = 1,
):
    # Source and target grid names
    gridn_in = get_grid_name(gridf_in)
    gridn_out = get_grid_name(gridf_out)

    # Interpolation matrix file name
    power_str = f"{power}".replace(".", "p")
    if wgt_file is None:
        radix = f"{wgt_dir}/{gridn_in}_to_{gridn_out}"
        if method == "nearest":
            wgt_file = f"{radix}_nearest_nn{nn}_power{power_str}.mat.npz"
        elif method == "linear":
            wgt_file = f"{radix}_linear.mat.npz"
        else:
            print("Error, method should be 'linear' or 'nearest'")
            raise SystemExit
    # Look for interpolation weights
    interpolation_matrix = None
    if os.path.exists(wgt_file) and not overwrite:
        print(f"Error. Interpolation matrix file {wgt_file} already exists.")
        raise SystemExit

    else:
        print(f" => Creating {wgt_file}")
        # Tolerance parameter
        epsilon = 1.0e-12
        # Input grid
        points_in = read_grid(gridf_in, lon_conv)
        # Output grid
        points_out = read_grid(gridf_out, lon_conv)

        # Check if target domain is likely to be in source domain
        check_target_in_source_domain(points_in, points_out)

        if method == "nearest":
            # Get index of source grid closest to every target point
            tree = cKDTree(points_in)
            distances, indices = tree.query(points_out, k=nn)
            weights = 1.0 / (distances**power + epsilon)
            # Add degenerate dimension in case nn = 1
            if indices.ndim == 1:
                indices, weights = indices[:, None], weights[:, None]
            # Normalizing of weights
            weights /= weights.sum(axis=1)[:, None]
        elif method == "linear":
            # Look for points in source grid that are closest to the target points
            tri = Delaunay(points_in)
            simplex = tri.find_simplex(points_out)
            # Manage points outside the domain : simplex = -1
            mask = simplex >= 0
            d = points_in.shape[1]
            indices = np.full((len(points_out), d + 1), -1, dtype=int)
            weights = np.zeros((len(points_out), d + 1))

            if np.any(mask):
                x = tri.transform[simplex[mask], :d]
                y = points_out[mask] - tri.transform[simplex[mask], d]
                bary = np.einsum("ijk,ik->ij", x, y)
                bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]
                vertices = tri.simplices[simplex[mask]]
                indices[mask, :] = vertices
                weights[mask, :] = bary_coords

        # Save the weights. To be done :
        # - extract just part of the matrix, if interpolation is needed
        # to a sudbdomain of the original target grid ?
        interpolation_matrix = build_sparse_interpolation_matrix(
            indices, weights, np.shape(points_in)[0]
        )

    # Cropping if required
    if lonlat_box is not None:
        # Check lon/lat box
        check_lonlat_box(gridf_in, gridf_out, lonlat_box, lon_conv)

        # Compute crop mask
        cropmsk_in = cropmsk(gridf_in, lonlat_box, lon_conv)
        cropmsk_out = cropmsk(gridf_out, lonlat_box, lon_conv)

        # Apply crop mask to interpolation matrix
        print(f"\t > Original interp matrix shape = {np.shape(interpolation_matrix)}")
        interpolation_matrix = interpolation_matrix[cropmsk_out, :][:, cropmsk_in]
        print(f"\t > Cropped interp matrix shape = {np.shape(interpolation_matrix)}")

    save_npz(wgt_file, interpolation_matrix)


# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    # -----------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Computes or read weights to interpolate a field "
        "(nearest neighbour method).",
        usage="puv run get-interpolation-matrix --gridf_in --gridf_out --method"
        "[--wgt_dir --num_nn --power_nn --lonlat_box --lon_conv --overwrite]",
    )
    parser.add_argument(
        "--gridf_in",
        type=str,
        required=True,
        help="Source grid file name (format : grid_[source_grid_name].npz)",
    )
    parser.add_argument(
        "--gridf_out",
        type=str,
        required=True,
        help="Target grid file name (format : grid_[target_grid_name].npz)",
    )
    parser.add_argument(
        "--wgt_dir",
        type=str,
        default=".",
        help="Where the weights file can be found (default : where the script in run).",
    )
    parser.add_argument(
        "--wgt_file",
        type=str,
        help="Name of the output matrix .mat.npz file. ",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["linear", "nearest"],
        help="Interpolation method used. Either 'linear' or 'nearest'.",
    )
    parser.add_argument(
        "--num_nn",
        type=int,
        default=1,
        help="Number of nearest neighbours (default is 1).",
    )
    parser.add_argument(
        "--power_nn",
        type=float,
        default=1.0,
        help="The weights of the interpolation are proportional to 1/d**power, "
        "where d is the distance of target grid points to source grid points.",
    )
    parser.add_argument(
        "--lonlat_box",
        nargs=4,
        type=float,
        help="The limits of the lon/lat box should be provided in degrees, "
        "and in the following order : lon_min, lon_max, lat_min, lat_max",
    )
    parser.add_argument(
        "--lon_conv_360",
        action="store_true",
        help="Convert to 0~360° longitude convention if used,s -180~180° otherwise",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite interpolation matrix file "
        "(e.g. in case you modified an source grid file)",
    )

    args = parser.parse_args()

    # Interpolation method
    kwargs = {"method": args.method}

    if args.method == "nearest":
        # Add nearest neighbour interpolation options
        # - nn : number of considered neighbours
        # - power: power involved in weights computation (weight = 1/distance**power)
        kwargs.update({"nn": args.num_nn, "power": args.power_nn})

    # Get interpolation indices & weights (or compute them)
    lon_conv = "-180~180"
    if args.lon_conv_360:
        lon_conv = "0~360"
    get_iw_nn(
        args.gridf_in,
        args.gridf_out,
        args.wgt_dir,
        args.wgt_file,
        args.lonlat_box,
        lon_conv,
        args.overwrite,
        **kwargs,
    )


if __name__ == "__main__":
    main()
