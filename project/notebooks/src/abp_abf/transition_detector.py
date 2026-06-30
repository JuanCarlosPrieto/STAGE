class TransitionDetector:
    def __init__(self, positions, cv, threshold):
        self.positions = positions
        self.cv = cv
        self.threshold = threshold


    def detect_transition(self):
        for i, position in enumerate(self.positions):
            if self.cv(position) > self.threshold:
                return i  # Return the index of the transition point
        
        return None  # Return None if no transition is detected