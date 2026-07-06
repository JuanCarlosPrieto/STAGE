from .geometry_utilities import find_intersection
from .visualization import add_colored_path_2d
import matplotlib.pyplot as plt
import numpy as np

class NaiveNarrowEscape:
    def __init__(self, brownian_motion, surface, escapes):
        self.brownian_motion = brownian_motion
        self.surface = surface
        self.escapes = escapes

        if brownian_motion.positions[0] is None:
            raise ValueError("Initial position of Brownian motion cannot be None.")
        
        try:
            if not surface.is_inside(brownian_motion.positions[0]):
                raise ValueError("Initial position of Brownian motion must be inside the surface.")
        
        except Exception as e:
            raise ValueError(f"Error checking if initial position is inside the surface: {e}, check the surface functions and the initial position.")
        
        if len(brownian_motion.positions) <= 1:
            brownian_motion.num_steps = 1000  # Default number of steps if not set
            brownian_motion.simulate()  # Simulate the Brownian motion if not already done
        

    def check_escape(self, intersection_point):
        """
        Check if the intersection point satisfies any of the escape conditions.
        
        Parameters
        ----------
        intersection_point : array-like
            The point to check for escape conditions.
        
        Returns
        -------
        bool
            True if the point satisfies any escape condition, False otherwise.
        """
        for escape in self.escapes:
            if escape.is_valid_escape(intersection_point):
                return True
        return False
        
    

    def run_simulation_straight_exit(self, max_steps=1e6):
        """
        Run the naive narrow escape simulation.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps to simulate before stopping.
        
        Returns
        -------
        escape_point : np array
            point where the Brownian motion escapes, or None if no escape occurs.
        escape_time : float
            time at which the escape occurs, or None if no escape occurs.
        """
        curr = 0  # Start checking from the first position
        while len(self.brownian_motion.positions) <= max_steps + self.brownian_motion.num_steps:
            for i in range(curr, len(self.brownian_motion.positions) - 1):            
                a = self.brownian_motion.positions[i]
                b = self.brownian_motion.positions[i + 1]

                if self.surface.is_inside(b):
                    continue  # No escape, continue to the next step

                index = self.surface.exit_surface(b)
                intersection_point = find_intersection(self.surface.functions[index], a, b)
                
                if intersection_point is not None:
                    if self.check_escape(intersection_point):
                        del self.brownian_motion.positions[i + 2:]  # Remove all positions after the exit
                        return intersection_point, i * self.brownian_motion.delta_t  # Return the escape point if found and the corresponding time
                    else:
                        del self.brownian_motion.positions[i + 1:]  # Remove all positions after the exit (no more valid)
                        self.brownian_motion.positions.append(self.brownian_motion.positions[-1]) # Append the last valid position (Metroplis algorithm)
                        curr = i
                        break  # Break to restart the loop with the updated positions

            self.brownian_motion.simulate()  # Continue the simulation from the last valid position
            
        return None, None  # No escape occurred
    

    def plot_narrow_escape_problem(
        self,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        point_stride=1000,
        point_size=50,
        point_alpha=0.8,
        show_points=True,
    ):
        """
        Plot the Brownian motion path along with the surface and escape regions.
        Currently supports 2D paths.
        """

        path = np.asarray(self.brownian_motion.positions, dtype=float)

        if path.ndim != 2:
            raise ValueError("Brownian path must have shape (N, dim).")

        n_steps, dim = path.shape

        if dim != 2:
            raise ValueError("This plotting function only supports 2D paths.")

        fig, ax = plt.subplots(figsize=(7, 7))

        _, cmap, norm = add_colored_path_2d(
            ax=ax,
            path=path,
            point_stride=point_stride,
            point_size=point_size,
            point_alpha=point_alpha,
            show_points=show_points,
        )

        self.surface.plot_boundary_2d(ax=ax, escape_checker=self.check_escape, xlim=xlim, ylim=ylim)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.autoscale()
        ax.legend()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Trajectory progression")

        plt.tight_layout()
        plt.show()