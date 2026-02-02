import random

class FillerGenerator:
    def __init__(self):
        self.fillers = [
            "Let me check the manual for that.",
            "One moment, looking that up.",
            "Checking the specifications.",
            "Let me find the details on that.",
            "Searching the documentation.",
            "Just a second, retrieving that info."
        ]
        
    def get_filler(self):
        return random.choice(self.fillers)
