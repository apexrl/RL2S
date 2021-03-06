import abc
import numpy as np

from rlkit.core.base_algorithm import BaseAlgorithm
from rlkit.core.offline_algorithm import OfflineAlgorithm


class TorchBaseAlgorithm(BaseAlgorithm, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device):
        for net in self.networks:
            net.to(device)

class TorchOfflineAlgorithm(OfflineAlgorithm, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device):
        for net in self.networks:
            net.to(device)
