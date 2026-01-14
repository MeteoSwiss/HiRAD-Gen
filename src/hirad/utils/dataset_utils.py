import numpy as np
from scipy.spatial import Delaunay
from typing import Optional


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
        tri (Delaunay): Delaunay triangulation of original points.
        simplex_id (np.ndarray): Simplex indices for each target point.
        lambda1 (np.ndarray): Barycentric coordinate 1 for each target point.
        lambda2 (np.ndarray): Barycentric coordinate 2 for each target point.
        lambda3 (np.ndarray): Barycentric coordinate 3 for each target point.
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
        self.tri = Delaunay(coords_orig)

        # Find simplex indices for target points
        coords_target = np.stack([self.longitudes_target, self.latitudes_target], axis=-1)
        self.simplex_id = self.tri.find_simplex(coords_target)

        # Check for points outside convex hull
        if np.any(self.simplex_id == -1):
            n_outside = np.sum(self.simplex_id == -1)
            print(f"Warning: {n_outside} target points are outside the convex hull of original points")

        # Get corner coordinates of simplices
        longitudes_corners = self.longitudes_orig[self.tri.simplices]
        latitudes_corners = self.latitudes_orig[self.tri.simplices]
        
        # Get corner coordinates for each target point's simplex
        longitude_corners_per_target = longitudes_corners[self.simplex_id]
        latitude_corners_per_target = latitudes_corners[self.simplex_id]
        
        # Extract traingle vertices
        x1, y1 = longitude_corners_per_target[:, 0], latitude_corners_per_target[:, 0]
        x2, y2 = longitude_corners_per_target[:, 1], latitude_corners_per_target[:, 1]
        x3, y3 = longitude_corners_per_target[:, 2], latitude_corners_per_target[:, 2]
        
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        self.lambda1 = ((y2 - y3) * (self.longitudes_target - x3) + 
                        (x3 - x2) * (self.latitudes_target - y3)) / denominator
        
        self.lambda2 = ((y3 - y1) * (self.longitudes_target - x3) + 
                        (x1 - x3) * (self.latitudes_target - y3)) / denominator
        
        self.lambda3 = 1 - self.lambda1 - self.lambda2
    
    def interpolate(self, values: np.ndarray, fill_value: Optional[float] = np.nan) -> np.ndarray:
        """
        Interpolate values from original points to target points.
        
        Args:
            values: Array of shape (n_samples, n_original_points) containing values
                   at original point locations to be interpolated.
            fill_value: Value to use for target points outside the convex hull.
                       Defaults to np.nan.
        
        Returns:
            Array of shape (n_samples, n_target_points) with interpolated values.
            
        Raises:
            ValueError: If values shape is incompatible with original points.
        """
        if values.shape[-1] != len(self.longitudes_orig):
            raise ValueError(
                f"Expected values with shape (..., {len(self.longitudes_orig)}), "
                f"got shape {values.shape}"
            )
        
        # Find the corner values of each simplice
        values_simplices = values[:,self.tri.simplices]
        
        # Find the simplice corner values for each target point
        values_per_target_simplices = values_simplices[:,self.simplex_id]
        
        # Perform barycentric interpolation
        out = (self.lambda1 * values_per_target_simplices[:,:,0] +
               self.lambda2 * values_per_target_simplices[:,:,1] +
               self.lambda3 * values_per_target_simplices[:,:,2])

        # Handle points outside convex hull
        if np.any(self.simplex_id == -1):
            out[:, self.simplex_id == -1] = fill_value

        return out
    
    def __call__(self, values: np.ndarray, fill_value: Optional[float] = np.nan) -> np.ndarray:
        """Alias for forward method to make class callable."""
        return self.interpolate(values, fill_value)