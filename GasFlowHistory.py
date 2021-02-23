import os
import numpy as np
from numpy.lib.index_tricks import MGridClass
from matplotlib import pyplot as plt

class GasFlowHistory:

    def __init__(self, fps, nf, gas_region_shape, global_frame_time):
        self.fps = fps
        self.nf = nf
        self.max_gas_region_hist = int(10 * fps / nf)
        self.index_gas_region_hist = int(0)
        self.gas_region_shape = gas_region_shape
        self.gas_region_hist = np.zeros(
            (self.max_gas_region_hist, gas_region_shape[0], gas_region_shape[1]))
        self.gas_ended = False

        # Gas information
        self.gas_id = os.getuid()
        self.start_time = global_frame_time
        self.stop_time = None
        self.rate_FCS = None
        self.gas_peak_index = None

        # Start-Stop Event Report
        self.start_event_reported = False
        self.stop_event_reported = False


    def IncreaseTimeStep(self, global_frame_time):
        if self.gas_ended:
            return
        self.index_gas_region_hist += 1
        if self.index_gas_region_hist >= self.max_gas_region_hist:
            self.index_gas_region_hist = 0
        self.gas_region_hist[self.index_gas_region_hist, :, :] = np.zeros(self.gas_region_shape)
        self.last_gas_region_mean = np.mean(self.gas_region_hist, axis=0)
        if np.sum(self.last_gas_region_mean) == 0:
            self.gas_ended = True
            self.stop_time = global_frame_time


    def IsInThisRegion(self, gas_region):
        if self.last_gas_region_mean is None:
            return 0
        same = np.sum(self.last_gas_region_mean * gas_region)
        return same

    def AddToThisRegion(self, gas_region, cur_rate_FCS):
        self.gas_region_hist[self.index_gas_region_hist, :, :] = np.maximum(
            self.gas_region_hist[self.index_gas_region_hist, :, :], gas_region)
        self.last_gas_region_mean = np.mean(self.gas_region_hist, axis=0)
        if self.rate_FCS is None:
            self.rate_FCS = cur_rate_FCS
        self.rate_FCS = self.rate_FCS * 0.80 + cur_rate_FCS * 0.20
        ind = np.unravel_index(np.argmax(self.last_gas_region_mean), self.last_gas_region_mean.shape)
        if self.gas_peak_index is None:
            self.gas_peak_index = np.float32(ind)
        self.gas_peak_index = self.gas_peak_index * 0.8 + np.float32(ind) * 0.2
        return 0

    def GetGasInfo(self):
        if (self.start_event_reported == True) and (self.stop_event_reported == True):
            return None

        # smoke started but It continues (in next period)
        if self.stop_time is None:
            info = {
                "id": self.gas_id,
                "start time": self.start_time,
                "rate FCM": np.round(self.rate_FCS, 2),
                "x": np.int0(self.gas_peak_index[1]),
                "y": np.int0(self.gas_peak_index[0])}
            self.start_event_reported = True
        else:
            # smoke started in previous period
            if self.start_event_reported == True:
                info = {
                    "id": self.gas_id,
                    "stop time": self.stop_time,
                    "rate FCM": np.round(self.rate_FCS, 2),
                    "x": np.int0(self.gas_peak_index[1]),
                    "y": np.int0(self.gas_peak_index[0])}
                self.stop_event_reported = True
            else:
                # smoke started and has stoped in this period
                info = {
                    "id": self.gas_id,
                    "start time": self.start_time,
                    "stop time": self.stop_time,
                    "rate FCM": np.round(self.rate_FCS, 2),
                    "x": np.int0(self.gas_peak_index[1]),
                    "y": np.int0(self.gas_peak_index[0])}
                self.start_event_reported = True
                self.stop_event_reported = True

        return info
