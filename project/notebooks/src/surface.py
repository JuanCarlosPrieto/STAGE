# Surface class to represent a surface defined by a set of functions

class Surface:
    def __init__(self, name, functions):
        self.name = name
        self.functions = functions


    def isInside(self, point):
        # Check if the point is inside the surface defined by the functions
        for func in self.functions:
            if func(point) > 0:
                return False
        return True