from ..visualization import add_colored_path_2d
import matplotlib.pyplot as plt
import numpy as np


# For this case we'll have escape sections, as soon as the particle gets to one of this section
# it will be considered it scaped
class ABPEquivalentNarrowEscape:
    def __init__(self, abp, surface, escapes):
        self.abp = abp
        self.surface = surface
        self.escapes = escapes


        if abp.positions[0] is None:
            raise ValueError("Initial position of Brownian motion cannot be None.")
        
        try:
            if not surface.is_inside(abp.positions[0]):
                raise ValueError("Initial position of Brownian motion must be inside the surface.")
        
        except Exception as e:
            raise ValueError(f"Error checking if initial position is inside the surface: {e}, check the surface functions and the initial position.")
        
        if len(abp.positions) <= 1:
            abp.num_steps = 10  # Default number of steps if not set
            abp.simulate(max_iters=abp.num_steps + 1)  # Simulate the Brownian motion if not already done
        

    def check_escape(self, point):
        """
        Check if the intersection point satisfies any of the escape conditions.
        
        Parameters
        ----------
        point : array-like
            The point to check for escape conditions.
        
        Returns
        -------
        bool
            True if the point satisfies any escape condition, False otherwise.
        """
        for escape in self.escapes:
            if escape.is_valid_escape(point):
                return True
        return False
    

    def escape_index(self, point):
        '''
        Return index of the escape used by the particle

        Parameters
        ----------
        point : array-like
            The point to check for escape conditions.}
        
        Returns
        -------
        int
            index
        '''
        if not self.check_escape(point):
            return None
        
        for i, escape in enumerate(self.escapes):
            if escape.is_valid_escape(point):
                return i
        
        return None
        
    

    def run_simulation(self, max_steps=1e6):
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
        while len(self.abp.positions) <= max_steps + self.abp.num_steps:
            for i in range(curr, len(self.abp.positions) - 1):            
                a = self.abp.positions[i]
                b = self.abp.positions[i + 1]

                # Particle got to one of the escapes
                if self.check_escape(b):
                        del self.abp.positions[i + 2:]  # Remove all positions after the exit
                        return self.escape_index(b), self.abp.real_time  # Return the escape point if found and the corresponding time

                if self.surface.is_inside(b):
                    continue  # No escape, continue to the next step

                
                print('I am here')
                input()
                del self.abp.positions[i + 1:]  # Remove all positions after the exit (no more valid)
                self.abp.positions.append(self.abp.positions[-1]) # Append the last valid position (Metroplis algorithm)
                curr = i
                break  # Break to restart the loop with the updated positions

            self.abp.simulate(max_iters=self.abp.num_steps + 1)  # Continue the simulation from the last valid position
            
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

        path = np.asarray(self.abp.positions, dtype=float)

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

        self.surface.plot_boundary_2d(ax=ax, xlim=xlim, ylim=ylim)
        for escape in self.escapes:
            escape.plot_escape(ax=ax)

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