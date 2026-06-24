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