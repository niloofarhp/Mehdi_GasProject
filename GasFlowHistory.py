import uuid
import numpy as np
import cv2 as cv
import imutils as imutils
import numpy as np
from numpy.lib.index_tricks import MGridClass
from matplotlib import pyplot as plt

class GasFlowHistory:

    def __init__(self, fps, fps_down_sample, nf_step, gas_region_shape, global_frame_time, min_time_detect):
        self.fps = fps
        self.fps_down_sample = fps_down_sample
        self.nf_step = nf_step
        self.min_time_detect = min_time_detect
        self.max_gas_region_hist = int(min_time_detect * fps / fps_down_sample / nf_step)
        self.index_gas_region_hist = int(0)
        self.gas_region_shape = gas_region_shape
        self.gas_region_hist = np.zeros(
            (self.max_gas_region_hist, gas_region_shape[0], gas_region_shape[1]))
        self.gas_ended = False

        # Gas information
        self.gas_uuid = uuid.uuid1()
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

        if same == 0:
            s1 = np.sum(self.last_gas_region_mean)
            s2 = np.sum(gas_region)
            k1 = int(np.floor(np.sqrt(s1))) + 1
            k2 = int(np.floor(np.sqrt(s2))) + 1
            kernel1 = np.ones((k1, k1)) / (k1*k1)
            dst1 = cv.filter2D(self.last_gas_region_mean, -1, kernel1)
            kernel2 = np.ones((k2, k2)) / (k2*k2)
            dst2 = cv.filter2D(self.last_gas_region_mean, -1, kernel2)
            same = np.sum(dst1 * dst2)
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
                "id": self.gas_uuid,
                "start time": self.start_time,
                "rate FCM": np.round(self.rate_FCS * self.fps_down_sample, 2),
                "x": np.int0(self.gas_peak_index[1]),
                "y": np.int0(self.gas_peak_index[0])}
            self.start_event_reported = True
        else:
            # smoke started in previous period
            if self.start_event_reported == True:
                info = {
                    "id": self.gas_uuid,
                    "stop time": self.stop_time,
                    "rate FCM": np.round(self.rate_FCS * self.fps_down_sample, 2),
                    "x": np.int0(self.gas_peak_index[1]),
                    "y": np.int0(self.gas_peak_index[0])}
                self.stop_event_reported = True
            else:
                diff_time = self.stop_time - self.start_time
                if diff_time.total_seconds() < self.min_time_detect:
                    return None

                # smoke started and has stoped in this period
                info = {
                    "id": self.gas_uuid,
                    "start time": self.start_time,
                    "stop time": self.stop_time,
                    "rate FCM": np.round(self.rate_FCS * self.fps_down_sample, 2),
                    "x": np.int0(self.gas_peak_index[1]),
                    "y": np.int0(self.gas_peak_index[0])}
                self.start_event_reported = True
                self.stop_event_reported = True

        return info
