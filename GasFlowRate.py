import cv2 as cv
import imutils as imutils
import numpy as np
from numpy.lib.index_tricks import MGridClass
from matplotlib import pyplot as plt

class GasFlowRate:

    def __init__(self, fps, nf):

        self.fps = fps
        self.nf = nf

        self.fgbg = cv.createBackgroundSubtractorMOG2()
        self.prev_gray = None
        self.gas_flow_rate_hist = None
        self.rate_FCS = None

    def CalcGasFlowRate(self, gray_frames, bin_gas_region, flow_method_median=True):

        step = 10
        kernel = np.ones((step, step), np.uint8)
        bin_gas_region_erode = cv.erode(1.0 * bin_gas_region, kernel)
        bin_gas_region_erode = np.multiply(bin_gas_region_erode > 0.1, 1)

        height = np.shape(bin_gas_region)[0]
        width = np.shape(bin_gas_region)[1]

        contours = cv.findContours(np.uint8(bin_gas_region_erode), cv.RETR_EXTERNAL,
                                   cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        result_list = []
        for contour in contours:
            abs_contour = np.zeros_like(bin_gas_region)
            abs_contour = cv.drawContours(abs_contour, [contour], -1, 1, thickness=-1)
            boundingRect = cv.boundingRect(contour)
            w1 = boundingRect[0]
            w2 = w1 + boundingRect[2]
            h1 = boundingRect[1]
            h2 = h1 + boundingRect[3]

            # Create flow hist specialy for this zone
            flow_hist = np.zeros((self.nf, height, width, 2))
            for f in range(self.nf-1):
                flow_hist[f,h1:h2,w1:w2,:] = 2.0 * cv.calcOpticalFlowFarneback(
                    gray_frames[f,h1:h2,w1:w2], gray_frames[f+1,h1:h2,w1:w2],
                                                     None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_all_frame = np.zeros((2,))

            for f in range(self.nf):
                flow_sel_zone = flow_hist[f,:,:,:]
                flow_sel_zone[:,:,0] *= abs_contour
                flow_sel_zone[:,:,1] *= abs_contour

                flow_abs = np.abs(flow_sel_zone[:, :, 0] + 1j * flow_sel_zone[:, :, 1])
                ind = np.unravel_index(np.argmax(flow_abs), flow_abs.shape)

                if flow_method_median:
                    cond = flow_abs > 1.0
                    if np.max(cond) > 0:
                        flow_sel = flow_sel_zone[cond]
                        shift_smoke = np.median(flow_sel, axis=0)
                    else:
                        shift_smoke = np.zeros((2,))
                else:
                    shift_smoke = flow_sel_zone[ind[0], ind[1]]

                flow_all_frame = flow_all_frame + shift_smoke

            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)

            cross_section = np.min(rect[1])
            pixel_flow = np.abs(flow_all_frame[0] + 1j * flow_all_frame[1])
            pixel_flow_per_frame = (pixel_flow / self.nf)

            rate_pixel = cross_section * pixel_flow_per_frame * self.fps
            rate_MM = rate_pixel * (20.0 / width) * (20.0 / height)

            # 1 MCM = 35.3 MCF
            rate_FCS = rate_MM * 35.3

            center = np.int0(np.mean(box, axis=0))
            result_list.append([rate_FCS, center[0], center[1], rect])

        return result_list


    def ClacGasFlowRate_single(self, color_frame, bin_gas_region):

        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray_frame)
        gray = fgmask

        if self.prev_gray is None:
            self.prev_gray = gray

        flow = 2.0 * cv.calcOpticalFlowFarneback(self.prev_gray, gray,
                                           None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray

        if self.flow_hist is None:
            self.flow_hist = flow
        self.flow_hist = flow * 0.05 + self.flow_hist * 0.95
        flow = self.flow_hist


        mgrid = MGridClass()
        step = 5
        len_thresh = 0.1

        #--------------------------------------------------------
        #--------------------------------------------------------
        #--------------------------------------------------------
        bin = np.multiply(bin_gas_region[:,:,2] > 1, 1)
        kernel = np.ones((step, step), np.uint8)
        bin = cv.erode(1.0 * bin, kernel)
        bin = np.multiply(bin > 0.1, 1)

        #plt.imshow(bin)
        #plt.show()

        contours = cv.findContours(np.uint8(bin.copy()), cv.RETR_EXTERNAL,
                                   cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) == 0:
            return color_frame
        sel_contour = max(contours, key=cv.contourArea)

        pts = sel_contour

        abs_flow = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        abs_contour = np.zeros_like(abs_flow)
        abs_contour = cv.drawContours(abs_contour, [sel_contour], -1, 1, thickness=-1)


        #--------------------------------------------------
        #--------------------------------------------------
        #--- Cut ------------------------------------------
        #--------------------------------------------------
        #--------------------------------------------------
        flow_cut = flow.copy()
        flow_cut[:,:,0] = abs_contour * flow_cut[:,:,0]
        flow_cut[:,:,1] = abs_contour * flow_cut[:,:,1]

        h, w = flow_cut.shape[:2]
        # create and flatten a meshgrid
        y, x = mgrid[step / 2: h: step, step / 2: w: step].reshape(2, -1)
        fx, fy = flow_cut[y.astype(int), x.astype(int)].T
        #max_abs = np.max(np.sqrt(fx ** 2 + fy ** 2))
        len_thresh = 0.01#max(max_abs / 5.0, 0.1)
        cond = np.sqrt(fx ** 2 + fy ** 2) > len_thresh
        x, y, fx, fy = (x[cond], y[cond], fx[cond], fy[cond])

        #np.average(a)

        abs = np.sqrt(fx ** 2 + fy ** 2)
        cx = np.sum(np.multiply(abs, x)) / np.sum(abs)
        cy = np.sum(np.multiply(abs, y)) / np.sum(abs)

        cfx = 10 * np.sum(np.multiply(abs, fx)) / np.sum(abs)
        cfy = 10 * np.sum(np.multiply(abs, fy)) / np.sum(abs)

        cfx_angle = np.angle(cfx + 1j * cfy)

        th = -cfx_angle
        #e = np.array([[np.cos(th), np.sin(th)]]).T
        es = np.array([
            [np.cos(th), np.sin(th)],
            [-np.sin(th), np.cos(th)],
        ])
        dists = np.dot(pts, es)
        #sh = np.shape(dists)
        #dists = np.reshape(dists, (sh[0], sh[2]))
        center = (cx, cy)
        rot_center = np.dot(center, es)
        #wh = dists.max(axis=0) - dists.min(axis=0)
        #wh = dists.max(axis=0) - dists.min(axis=0)
        cond = dists[:,:, 0] > rot_center[0]
        dists[cond] = rot_center

        wh_max = dists.max(axis=0)[0]
        wh_min = dists.min(axis=0)[0]

        th = cfx_angle
        ies = np.array([
            [np.cos(th), np.sin(th)],
            [-np.sin(th), np.cos(th)],
        ])
        p = np.zeros((4,2))
        # limit to center arrow
        wh_max[0] = min(rot_center[0], wh_max[0])
        p[0,:] = np.dot((wh_max[0], wh_min[1]), ies)
        p[1,:] = np.dot((wh_max[0], wh_max[1]), ies)
        p[2,:] = np.dot((wh_min[0], wh_max[1]), ies)
        p[3,:] = np.dot((wh_min[0], wh_min[1]), ies)

        rect_emit = np.int32(p)
        dists2 = np.int32(np.dot(dists, ies))



        #--------------------------------------------------
        #--------------------------------------------------
        #--- After Cut ------------------------------------
        #--------------------------------------------------
        #--------------------------------------------------
        abs_contour_cut = np.zeros_like(abs_flow)
        abs_contour_cut = cv.drawContours(abs_contour_cut, [dists2], -1, 1, thickness=-1)

        flow_cn = flow.copy()
        flow_cn[:,:,0] = abs_contour_cut * flow_cn[:,:,0]
        flow_cn[:,:,1] = abs_contour_cut * flow_cn[:,:,1]

        h, w = flow_cn.shape[:2]
        # create and flatten a meshgrid
        y, x = mgrid[step / 2: h: step, step / 2: w: step].reshape(2, -1)
        fx, fy = flow_cn[y.astype(int), x.astype(int)].T
        #max_abs = np.max(np.sqrt(fx ** 2 + fy ** 2))
        len_thresh = 0.01#max(max_abs / 5.0, 0.01)
        cond = np.sqrt(fx ** 2 + fy ** 2) > len_thresh
        x, y, fx, fy = (x[cond],
                        y[cond],
                        fx[cond],
                        fy[cond])

        abs = np.sqrt(fx ** 2 + fy ** 2)
        cx_n = np.sum(np.multiply(abs, x)) / np.sum(abs)
        cy_n = np.sum(np.multiply(abs, y)) / np.sum(abs)

        cfx_n = np.sum(np.multiply(abs, fx)) / np.sum(abs)
        cfy_n = np.sum(np.multiply(abs, fy)) / np.sum(abs)


        #--------------------------------------------------
        #--- Emittion Rate --------------------------------
        #--------------------------------------------------
        wide = wh_max[0] - wh_min[0]
        mean_flow_rate = np.abs(cfx_n + cfy_n*1j)
        emit_rate_cur = wide * mean_flow_rate * self.fps

        if self.gas_flow_rate_hist is None:
            gas_flow_rate_hist = emit_rate_cur
        gas_flow_rate_hist = gas_flow_rate_hist * 0.99 + emit_rate_cur * 0.01


        #--------------------------------------------------
        #--- Emittion Rate --------------------------------
        #--------------------------------------------------
        # create line endpoints
        try:
            lines = np.int32(np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2))
        except BaseException:
            val = np.sqrt(fx ** 2 + fy ** 2).max()
            raise ValueError("No flow vectors longer than %.2f were "
                             "detected, longest flow vector: %.2f"
                             % (len_thresh, val))

        bin_gas = np.zeros_like(bin_gas_region)
        bin_gas[:,:,2] = bin * 200
        merge_frame = np.maximum(color_frame, bin_gas)

        # lines
        vis = merge_frame
        '''
        for (x1, y1), (x2, y2) in lines:
            cv.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv.circle(vis, (x2, y2), 1, (255, 0, 0), -1)

        if not np.isnan(cx):
            cv.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), thickness=3)
            cv.line(vis, (int(cx), int(cy)), (int(cx + cfx), int(cy + cfy)), (0, 0, 255), thickness=3)
        if not np.isnan(cx_n):
            cv.circle(vis, (int(cx_n), int(cy_n)), 3, (0, 255, 0), thickness=3)
            cv.line(vis, (int(cx_n), int(cy_n)), (int(cx_n + cfx_n), int(cy_n + cfy_n)), (0, 127, 255), thickness=2)
        '''
        if gas_flow_rate_hist is not None and not np.isnan(cx):
            # 1 MCM = 35.3 MCF
            rate_FCS = gas_flow_rate_hist * (2.0 / w) * (2.0 / h) * 35.3
            str = '{:.2f} MCF'.format(rate_FCS)
            cv.putText(vis, str, (int(cx), int(cy - 15)), cv.FONT_ITALIC, fontScale=1,
                       thickness=2, lineType=cv.LINE_AA, color=(255, 100, 50))


        xp, yp = None, None
        for c in dists2:  # sel_contour:
            x1, y1 = c[0, 0], c[0, 1]
            if xp is None:
                xp, yp = x1, y1
            cv.line(vis, (x1, y1), (xp, yp), (64, 64, 255), 3)
            xp, yp = x1, y1

        cv.line(vis, (rect_emit[0, 0], rect_emit[0, 1]), (rect_emit[1, 0], rect_emit[1, 1]), (255, 255, 255),
                thickness=1)
        cv.line(vis, (rect_emit[1, 0], rect_emit[1, 1]), (rect_emit[2, 0], rect_emit[2, 1]), (255, 255, 255),
                thickness=1)
        cv.line(vis, (rect_emit[2, 0], rect_emit[2, 1]), (rect_emit[3, 0], rect_emit[3, 1]), (255, 255, 255),
                thickness=1)
        cv.line(vis, (rect_emit[3, 0], rect_emit[3, 1]), (rect_emit[0, 0], rect_emit[0, 1]), (255, 255, 255),
                thickness=1)

        #cv.imshow("Optical flow live view", vis)
        return vis
