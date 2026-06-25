# Surface class to represent a surface defined by a set of functions
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

class Surface:
    def __init__(self, name, functions):
        self.name = name
        self.functions = functions


    def is_inside(self, point):
        # Check if the point is inside the surface defined by the functions
        for func in self.functions:
            if func(point) > 0:
                return False
        return True
    

    def exit_surface(self, point):
        if self.is_inside(point):
            return None

        for i, func in enumerate(self.functions):
            if func(point) > 0:
                return i
            
    
    def boundary_segments_2d(self, xlim, ylim, n_grid=400, tol=1e-8):
        """
        Extract approximate boundary segments for a 2D implicit surface.

        Each boundary is defined by func(point) = 0.

        Parameters
        ----------
        xlim : tuple
            (xmin, xmax)

        ylim : tuple
            (ymin, ymax)

        n_grid : int
            Number of grid points in each direction.

        tol : float
            Tolerance used to decide if a segment belongs to the real boundary
            of the admissible domain.

        Returns
        -------
        boundary_segments : list of np.ndarray
            List of segments. Each segment has shape (2, 2).
        """

        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)

        X, Y = np.meshgrid(x, y)

        all_segments = []

        # Temporary figure used only to extract contour lines
        fig_tmp, ax_tmp = plt.subplots()

        for i, func in enumerate(self.functions):
            Z = np.empty_like(X)

            for row in range(n_grid):
                for col in range(n_grid):
                    point = np.array([X[row, col], Y[row, col]])
                    Z[row, col] = func(point)

            contour = ax_tmp.contour(X, Y, Z, levels=[0.0])

            # contour.allsegs[0] contains all curves for level 0
            for curve in contour.allsegs[0]:
                if len(curve) < 2:
                    continue

                for k in range(len(curve) - 1):
                    p_a = curve[k]
                    p_b = curve[k + 1]
                    midpoint = 0.5 * (p_a + p_b)

                    # Keep only boundary parts that really belong to the domain boundary.
                    # This matters if the surface is defined by several inequalities.
                    valid = True

                    for j, other_func in enumerate(self.functions):
                        if j == i:
                            continue

                        if other_func(midpoint) > tol:
                            valid = False
                            break

                    if valid:
                        all_segments.append(np.array([p_a, p_b]))

        plt.close(fig_tmp)

        return all_segments
    
    

    def plot_boundary_2d(
        self,
        ax,
        xlim,
        ylim,
        escape_checker=None,
        n_grid=400,
        reflective_color="black",
        escape_color="orange",
        reflective_linewidth=2.0,
        escape_linewidth=4.0
    ):
        """
        Plot the 2D boundary of the surface.

        If escape_checker is given, boundary segments satisfying escape_checker
        are plotted as escape windows.
        """

        boundary_segments = self.boundary_segments_2d(
            xlim=xlim,
            ylim=ylim,
            n_grid=n_grid
        )

        reflective_segments = []
        escape_segments = []

        for segment in boundary_segments:
            midpoint = 0.5 * (segment[0] + segment[1])

            if escape_checker is not None and escape_checker(midpoint):
                escape_segments.append(segment)
            else:
                reflective_segments.append(segment)

        if reflective_segments:
            reflective_collection = LineCollection(
                reflective_segments,
                colors=reflective_color,
                linewidths=reflective_linewidth,
                label="Reflective boundary"
            )
            ax.add_collection(reflective_collection)

        if escape_segments:
            escape_collection = LineCollection(
                escape_segments,
                colors=escape_color,
                linewidths=escape_linewidth,
                label="Escape window"
            )
            ax.add_collection(escape_collection)