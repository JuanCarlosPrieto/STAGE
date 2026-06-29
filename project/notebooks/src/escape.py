import numpy as np

class Escape:
    def __init__(self, conditions):
        self.conditions = conditions
    
    def is_valid_escape(self, point):
        """
        Check if the point satisfies all escape conditions.
        
        Parameters
        ----------
        point : array-like
            The point to check.
        
        Returns
        -------
        bool
            True if the point satisfies all conditions, False otherwise.
        """
        for condition in self.conditions:
            if not condition(point):
                return False
        return True


    def plot_escape(
        self,
        ax,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        n_grid=400,
        escape_color="orange",
        region_alpha=0.25
    ):
        """
        Paint the 2D region where all escape conditions are satisfied.

        Parameters
        ----------
        ax : matplotlib axis
            Axis where the escape region is plotted.

        xlim : tuple
            Limits in the x direction, for example (-1, 1).

        ylim : tuple
            Limits in the y direction, for example (-1, 1).

        n_grid : int
            Number of grid points in each direction.

        escape_color : str
            Color used for the escape region.

        region_alpha : float
            Transparency of the painted region.
        """

        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)

        X, Y = np.meshgrid(x, y)
        mask = np.zeros_like(X, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                point = np.array([X[i, j], Y[i, j]])

                if self.is_valid_escape(point):
                    mask[i, j] = True

        ax.contourf(
            X,
            Y,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=[escape_color],
            alpha=region_alpha
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")