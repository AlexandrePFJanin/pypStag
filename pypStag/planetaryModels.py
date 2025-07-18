# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Planetary models data
"""


# ----------------- CLASS -----------------


class Planet():
    """
    Main class for planetary models
    """
    def __init__(self, radius=0, mantle_bottom_depth=0, mantle_up_depth=0):
        self.radius = radius # mean radius in [km]
        self.mantle_bottom_depth = mantle_bottom_depth #depth CMB [km]
        self.mantle_up_depth = mantle_up_depth # surface [km]
        self.check()
    
    def check(self):
        if self.mantle_up_depth >= self.mantle_bottom_depth:
            raise ValueError()


class EarthModel(Planet):
    def __init__(self):
        super().__init__(radius=6371.0088, mantle_bottom_depth=2890, mantle_up_depth=0)

class VenusModel(Planet):
    # from: https://doi.org/10.1029/JB095iB09p14105
    def __init__(self):
        super().__init__(radius=6051.8, mantle_bottom_depth=2942, mantle_up_depth=0)

class MarsModel(Planet):
    # from: https://doi.org/10.1029/JB095iB09p14105
    def __init__(self):
        super().__init__(radius=3389.5, mantle_bottom_depth=1628, mantle_up_depth=190)


# ----------------- INSTANCES -----------------


Earth = EarthModel()
Venus = VenusModel()
Mars  = MarsModel()
