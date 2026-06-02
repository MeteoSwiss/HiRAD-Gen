import torch
import numpy as np
from scipy.spatial import Delaunay
from typing import Optional


def regrid_icon_to_rotlatlon(
    data: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    nx: int = 1170,
    ny: int = 786,
    ) -> torch.Tensor:
    """Regrid ICON unstructured data to a rotated lat-lon grid.

    Parameters
    ----------
    data : torch.Tensor
        Input data with the last axis being the unstructured grid dimension.
    indices : torch.LongTensor
        Remap indices of shape (n_target, n_stencil).
    weights : torch.Tensor
        Remap weights of shape (n_target, n_stencil).
    nx, ny : int
        Grid dimensions.

    Returns
    -------
    torch.Tensor
        Regridded data of shape (*(batch,channel), ny, nx).
    """
    out_shape = data.shape[:-1] + (ny, nx)

    # Gather stencil values: (..., n_target, n_stencil)
    # indices: (n_target, n_stencil) -> expand to match data batch dims
    values = data[..., indices]                     # (..., n_target, n_stencil)

    # Weighted sum: multiply then reduce over stencil dim
    result = (values * weights).sum(dim=-1)         # (..., n_target)

    # Clamp to stencil min/max to avoid extrapolation
    vmin = values.amin(dim=-1)
    vmax = values.amax(dim=-1)
    result = result.clamp(min=vmin, max=vmax)

    return result.reshape(out_shape)

def coarsen_2x(data: torch.Tensor) -> torch.Tensor:
    """Subsample the last two spatial dimensions by a factor of 2 (every other point)."""
    return data[..., ::2, ::2]

class GridData():
    """
    Performs interpolation from irregular points to target grid using Delaunay triangulation.
    
    This class uses barycentric coordinate interpolation within Delaunay triangles to map
    values from an original set of scattered points to a target set of points. It has same
    behavior as scipy.interpolate.griddata with method='linear', but is optimized for
    repeated interpolations on the same set of points.
    
    Attributes:
        longitudes_orig (np.ndarray): Longitude coordinates of original points.
        latitudes_orig (np.ndarray): Latitude coordinates of original points.
        longitudes_target (np.ndarray): Longitude coordinates of target points.
        latitudes_target (np.ndarray): Latitude coordinates of target points.
        is_torch (bool): Whether tensors are prepared for PyTorch operations.
        device (torch.device | None): Device for PyTorch tensors if applicable.
    """

    def __init__(
        self,
        longitudes_orig: np.ndarray,
        latitudes_orig: np.ndarray,
        longitudes_target: np.ndarray,
        latitudes_target: np.ndarray
    ) -> None:
        """
        Initialize the GridData interpolator.
        
        Args:
            longitudes_orig: 1D array of longitude values for original points.
            latitudes_orig: 1D array of latitude values for original points.
            longitudes_target: 1D array of longitude values for target points.
            latitudes_target: 1D array of latitude values for target points.
            
        Raises:
            ValueError: If input arrays have incompatible shapes.
        """
        # Validate inputs
        if len(longitudes_orig) != len(latitudes_orig):
            raise ValueError("Original longitude and latitude arrays must have same length")
        if len(longitudes_target) != len(latitudes_target):
            raise ValueError("Target longitude and latitude arrays must have same length")

        self.longitudes_orig = np.asarray(longitudes_orig)
        self.latitudes_orig = np.asarray(latitudes_orig)
        
        self.longitudes_target = np.asarray(longitudes_target)
        self.latitudes_target = np.asarray(latitudes_target)

        self.is_torch = False
        self.device = None
        
        self._prepare_interpolation()

    def _prepare_interpolation(self):
        """
        Prepare interpolation by computing Delaunay triangulation and barycentric coordinates.
        
        This method:
        1. Computes Delaunay triangulation of original points
        2. Finds which simplex each target point belongs to
        3. Precomputes barycentric coordinates (lambda1, lambda2, lambda3) for interpolation
        """
        # Compute Delaunay triangulation
        coords_orig = np.stack([self.longitudes_orig, self.latitudes_orig], axis=-1)
        self._tri = Delaunay(coords_orig)

        # Find simplex indices for target points
        coords_target = np.stack([self.longitudes_target, self.latitudes_target], axis=-1)
        self._simplex_id = self._tri.find_simplex(coords_target)

        # Check for points outside convex hull
        if np.any(self._simplex_id == -1):
            n_outside = np.sum(self._simplex_id == -1)
            print(f"Warning: {n_outside} target points are outside the convex hull of original points")

        # Get corner coordinates of simplices
        longitudes_corners = self.longitudes_orig[self._tri.simplices]
        latitudes_corners = self.latitudes_orig[self._tri.simplices]
        
        # Get corner coordinates for each target point's simplex
        longitude_corners_per_target = longitudes_corners[self._simplex_id]
        latitude_corners_per_target = latitudes_corners[self._simplex_id]
        
        # Extract traingle vertices
        x1, y1 = longitude_corners_per_target[:, 0], latitude_corners_per_target[:, 0]
        x2, y2 = longitude_corners_per_target[:, 1], latitude_corners_per_target[:, 1]
        x3, y3 = longitude_corners_per_target[:, 2], latitude_corners_per_target[:, 2]
        
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        self._lambda1 = ((y2 - y3) * (self.longitudes_target - x3) + 
                        (x3 - x2) * (self.latitudes_target - y3)) / denominator
        
        self._lambda2 = ((y3 - y1) * (self.longitudes_target - x3) + 
                        (x1 - x3) * (self.latitudes_target - y3)) / denominator
        
        self._lambda3 = 1 - self._lambda1 - self._lambda2


    def to(self, device: str | torch.device) -> None:
        """
        Prepare barycentric coordinates and simplex indices for PyTorch operations.
        
        This method converts the precomputed numpy arrays to PyTorch tensors
        and moves them to the specified device.
        
        Args:
            device: The torch device to move tensors to (e.g., 'cpu' or 'cuda').
        """
        if isinstance(device, str):
            device = torch.device(device)
        if self.is_torch and self.device == device:
            return  # Already on the correct device
        elif self.is_torch and self.device != device:
            self.device = device
            self._lambda1 = self._lambda1.to(device)
            self._lambda2 = self._lambda2.to(device)
            self._lambda3 = self._lambda3.to(device)
            self._simplex_id = self._simplex_id.to(device)
            self._tri.simplices = self._tri.simplices.to(device)
        elif not self.is_torch:
            self.to_torch(device)

    def to_torch(self, device: torch.device | str ='cpu') -> None:
        """
        Prepare barycentric coordinates and simplex indices for PyTorch operations.
        
        This method converts the precomputed numpy arrays to PyTorch tensors
        and moves them to the specified device.
        
        Args:
            device: The torch device to move tensors to (e.g., 'cpu' or 'cuda').
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self._lambda1 = torch.from_numpy(self._lambda1).to(device)
        self._lambda2 = torch.from_numpy(self._lambda2).to(device)
        self._lambda3 = torch.from_numpy(self._lambda3).to(device)

        # Convert indexing arrays
        self._simplex_id = torch.from_numpy(self._simplex_id).to(device)
        self._tri.simplices = torch.from_numpy(self._tri.simplices).to(device)
        
        self.is_torch = True

    def to_numpy(self) -> None:
        """
        Convert barycentric coordinates and simplex indices back to numpy arrays.
        
        This method converts the precomputed PyTorch tensors back to numpy arrays.
        """
        if not self.is_torch:
            return  # Already in numpy format

        self._lambda1 = self._lambda1.cpu().numpy()
        self._lambda2 = self._lambda2.cpu().numpy()
        self._lambda3 = self._lambda3.cpu().numpy()

        self._simplex_id = self._simplex_id.cpu().numpy()
        self._tri.simplices = self._tri.simplices.cpu().numpy()
        
        self.is_torch = False
        self.device = None
    
    def interpolate(self, values: np.ndarray | torch.Tensor, fill_value: Optional[float] = np.nan) -> np.ndarray:
        """
        Interpolate values from original points to target points.
        
        Args:
            values: Array of shape (n_channels, n_original_points) or (batch, n_channels, n_original_points) containing values
                   at original point locations to be interpolated.
            fill_value: Value to use for target points outside the convex hull.
                       Defaults to np.nan.
        
        Returns:
            Array of shape (n_channels, n_target_points) or (batch, n_channels, n_original_points) with interpolated values.
            
        Raises:
            ValueError: If values shape is incompatible with original points.
        """
        if values.shape[-1] != len(self.longitudes_orig):
            raise ValueError(
                f"Expected values with shape (..., {len(self.longitudes_orig)}), "
                f"got shape {values.shape}"
            )

        # Save original shape for reshaping later
        orig_shape = values.shape

        # In case that there is a batch dimension, flatten it for easier indexing
        values = values.reshape(-1, values.shape[-1])  # shape (batch*n_channels, n_original_points)
        
        # Find the corner values of each simplice
        values_simplices = values[:,self._tri.simplices]
        
        # Find the simplice corner values for each target point
        values_per_target_simplices = values_simplices[:,self._simplex_id]
        
        # Perform barycentric interpolation
        out = (self._lambda1 * values_per_target_simplices[:,:,0] +
               self._lambda2 * values_per_target_simplices[:,:,1] +
               self._lambda3 * values_per_target_simplices[:,:,2])

        # Handle points outside convex hull
        if (not self.is_torch and np.any(self._simplex_id == -1)) or (self.is_torch and torch.any(self._simplex_id == -1)):
            out[::, self._simplex_id == -1] = fill_value

        return out.reshape(orig_shape[:-1] + (out.shape[-1],))
    
    def __call__(self, values: np.ndarray | torch.Tensor, fill_value: Optional[float] = np.nan) -> np.ndarray:
        """Alias for forward method to make class callable."""
        return self.interpolate(values, fill_value)