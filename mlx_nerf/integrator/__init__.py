"""Base Integrator class 


"""

from abc import abstractmethod

class Integrator:
    def __init__(self, config) -> None:
        self.config = config
        pass

    @abstractmethod
    def train(self, rays, target):
        ...

    