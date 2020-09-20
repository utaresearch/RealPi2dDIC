#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
RealPi2dDIC is a free and open source Digital Image Correlation software which generates in-situ two dimensional strain field of any structure under loading. This software while
running captures images using a Picamera v2 module, registers the image data, correlates with a reference image using computer vision. This software is based
on Lucas-Kanade Method, a well established technique for measurement of optical flow based upon image correlation. It returns the local strain field over a
region of interest selected by the user in both x and y directions.

To know about the operational prcedures, please read Readme.md

This software is developed at the Institute of Predictive Performance Methodologies (IPPM), Univerisity of Texas at Arlington
Research Institute (UTARI), Texas, USA.

Authors: Partha Pratim Das [parthapratim.das@mavs.uta.edu] , Muthu Ram Prabhu Elenchezhian, Md Rassel Raihan [mdrassel.raihan@uta.edu], Vamsee Vadlamudi, Kenneth Reifsnider

This is a free software. It can be distributed/modified under the terms of MIT Licence. See the LICENSE.md file for more details.

Modules Dependencies:
i. picamera
ii. opencv-python
iii. numpy
iv. scipy
v. matplotlib

Hardware Dependencies:
i. Raspberry Pi + Raspbian OS
ii. Picamera Module v2
'''

import time
import math
import glob
import copy
import os
import shutil
import keyboard

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.interpolate
import picamera
camera = picamera.PiCamera()

surface_grid = []  # points on surface are stored in this array
# correlation window size (in pixels) for Lucas Kanade Method for optical
# flow measurement
lk_window_size = (65, 65)
# the size in pixel of the interval (dx,dy) of the correlation grid
lk_grid_size = (10, 10)
area = []
cropping = False


class grid:
    """A class used for many important features starting from
    capturing photos to image correlation and processing rawdata."""

    def __init__(self, Points_X, Points_Y, size_x, size_y):
        """
        This generates new grid object with x coordinate (Points_X)
        y coordinate (Points_Y), number of point along x (size_x) and
        number of point along y (size_y)"""

        self.Points_X = Points_X
        self.Points_Y = Points_Y
        self.size_x = size_x
        self.size_y = size_y
        self.disp_x = self.Points_X.copy().fill(0.)
        self.disp_y = self.Points_Y.copy().fill(0.)
        self.strain_longitudinal = None
        self.strain_transverse = None
        self.strain_xy = None

    def init_Raw_Data(self, winsize, reference_image, image, reference_point,
                      correlated_point, disp):
        """Saves raw image data using opencv to current grid object"""

        self.winsize = winsize
        self.reference_image = reference_image
        self.image = image
        self.reference_point = reference_point
        self.correlated_point = correlated_point
        self.disp = disp

    def directory_Name(self, prefix, extension):
        """Makes specific directories for storing specific results including
        surface image, grid displacement, raw data in CSV format"""

        folder = os.path.dirname(self.image)
        folder = folder + '/rawdata/' + prefix
        if not os.path.exists(folder):
            os.makedirs(folder)
        base = os.path.basename(self.image)
        name = folder + '/' + \
            os.path.splitext(base)[0] + '_' + prefix + '.' + extension
        return name

    def save_Points(self, pref, poInt):
        """saves correlated points and reference points array to a txt file """
        folderLocation = os.path.dirname(self.image)
        folderLocation = folderLocation + '/rawdata/%s' % pref
        if not os.path.exists(folderLocation):
            os.makedirs(folderLocation)
        num = os.path.splitext(os.path.basename(self.image))[0]
        np.savetxt('%s/%s_%s.txt' % (folderLocation, pref, num),
                   poInt,
                   delimiter=',')

    def correlation_Image(self):
        """Draws image correlation data points on reference image for each data point"""

        name = self.directory_Name('marker', 'png')
        display_Image(self.image,
                      point=self.correlated_point,
                      l_color=(0, 120, 255),
                      p_color=(140, 130, 0),
                      file_Name=name,
                      text=name)
        self.save_Points('correlatedPoints', self.correlated_point)

    def displacement_Image(self, scale):
        """Draws displacement image on reference image for each data point.
        Parameters
        ----------
        scale : int
             to amplify the displacement
        """

        name = self.directory_Name('disp', 'png')
        display_Image(self.reference_image,
                      point=self.reference_point,
                      pointf=self.correlated_point,
                      l_color=(125, 0, 125),
                      p_color=(125, 125, 125),
                      scale=scale,
                      file_Name=name,
                      text=name)
        self.save_Points('DisplacementPoints', self.reference_point)

    def surface_Image(self, scale):
        """Draws mesh deformations on reference image for each data point.
        Parameters
        ----------
        scale : int
             to amplify the deformation
        """
        name = self.directory_Name('surface', 'png')
        display_Image(self.reference_image,
                      grid=self,
                      scale=scale,
                      surfColor=(255, 0, 250),
                      file_Name=name,
                      text=name)

    def raw_Data_CSV(self):
        """writes a csv file for displacement, strain and other parameters.
        This data can be used for post processing"""

        name = self.directory_Name('result', 'csv')
        f = open(name, 'w')
        f.write("index" + ',' + "index_x" + ',' + "index_y" + ',' + "x (px)" +
                ',' + "y (px)" + ',' + "x_Displacement" + ',' +
                "y_Displacement" + ',' + "x_Strain" + ',' + "y_Strain" + ',' +
                "xy_Strain" + '\n')
        index = 0
        for i in range(self.size_x):
            for j in range(self.size_y):
                f.write(
                    str(index) + ',' + str(i) + ',' + str(j) + ',' +
                    str(self.Points_X[i, j]) + ',' + str(self.Points_Y[i, j]) +
                    ',' + str(self.disp_x[i, j]) + ',' +
                    str(self.disp_y[i, j]) + ',' + str(self.strain_longitudinal[i, j]) +
                    ',' + str(self.strain_transverse[i, j]) + ',' +
                    str(self.strain_xy[i, j]) + '\n')
                index = index + 1
        f.close()

    def insitu_Plot(self, field, title, fig_No, img_file_Name):
        """Connects with Plot Class. Plots in-situ strain field using matplotlib interactive mapping"""

        img_folder = './Test_%s/*.jpg' % img_file_Name
        img_listt = sorted(glob.glob(img_folder),
                           key=lambda t: os.stat(t).st_mtime)
        image_ref = cv2.imread(img_listt[0], 0)
        Plot(image_ref, self, field, title, fig_No)

    def bivariate_Interpolation(self, point, disp, *args, **kwargs):
        """Interpolates the displacement field. Bivariate B-spline interpolation algorithm from scipy is used here.
        for no interpolation option this can use raw method"""

        dx = np.array([d[0] for d in disp])
        dy = np.array([d[1] for d in disp])
        method = 'raw' if 'method' not in kwargs else kwargs['method']
        if method == 'raw':
            self.disp_x = self.Points_X.copy()
            self.disp_y = self.Points_Y.copy()
            count = 0
            for i in range(self.disp_x.shape[0]):
                for j in range(self.disp_x.shape[1]):
                    self.disp_x[i, j] = dx[count]
                    self.disp_y[i, j] = dy[count]
                    count = count + 1

        elif method == 'bivar_Spline':
            tck_x = scipy.interpolate.bisplrep(self.Points_X,
                                               self.Points_Y,
                                               dx,
                                               kx=5,
                                               ky=5)
            self.disp_x = scipy.interpolate.bisplev(self.Points_X[:, 0],
                                                    self.Points_Y[0, :], tck_x)
            tck_y = scipy.interpolate.bisplrep(self.Points_X,
                                               self.Points_Y,
                                               dy,
                                               kx=5,
                                               ky=5)
            self.disp_y = scipy.interpolate.bisplev(self.Points_X[:, 0],
                                                    self.Points_Y[0, :], tck_y)

    def strain_Field_Compute(self):
        """Computes Green-Langragian strain field from interpolated displacement data using numpy"""

        dx = self.Points_X[1][0] - self.Points_X[0][0]
        dy = self.Points_Y[0][1] - self.Points_Y[0][0]

        strain_longitudinal, strain_xy = np.gradient(
            self.disp_x, dx, dy, edge_order=2)
        strain_yx, strain_transverse = np.gradient(
            self.disp_y, dx, dy, edge_order=2)

        self.strain_longitudinal = strain_longitudinal + .5 * \
            (np.power(strain_longitudinal, 2) + np.power(strain_xy, 2))
        self.strain_transverse = strain_transverse + .5 * \
            (np.power(strain_transverse, 2) + np.power(strain_yx, 2))
        self.strain_xy = .5 * (strain_xy + strain_yx + strain_longitudinal * strain_xy +
                               strain_yx * strain_transverse)


class Plot:
    """ Plot class includes all the required operations for interactive plotting of strain field data on the reference image
    using matplotlib """

    def __init__(self, image, grid, data, title, fig):
        self.data = np.ma.masked_invalid(data)
        self.data_copy = np.copy(self.data)
        self.Points_X = grid.Points_X
        self.Points_Y = grid.Points_Y
        self.data = np.ma.array(self.data, mask=self.data == np.nan)

        self.fig = plt.figure(fig)
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=1.00)

        self.ax.imshow(image, cmap='Greys')

        self.im = self.ax.contourf(grid.Points_X,
                                   grid.Points_Y,
                                   self.data,
                                   50,
                                   cmap='rainbow',
                                   alpha=0.75)

        self.ax.set_title(title)
        self.ax.set_ylabel('Height (Pixels)')
        self.ax.set_xlabel('Width (Pixels)')
        self.cb = self.fig.colorbar(self.im)


def surface_Generate(area, num_point, *args, **kwargs):
    """Generates a surface grid using points from region of interest"""

    xmin = area[0][0]
    xmax = area[1][0]
    dx = xmax - xmin
    ymin = area[0][1]
    ymax = area[1][1]
    dy = ymax - ymin
    point_surface = dx * dy / num_point
    point_line = math.sqrt(point_surface)
    ratio = 1. if 'ratio' not in kwargs else kwargs['ratio']
    num_x = int(ratio * dx / point_line) + 1
    num_y = int(ratio * dy / point_line) + 1
    Points_X, Points_Y = np.mgrid[xmin:xmax:num_x * 1j, ymin:ymax:num_y * 1j]
    return grid(Points_X, Points_Y, num_x, num_y)


def display_Image(image, *args, **kwargs):
    """Draws opencv images for raw data visualization
    Parameters
    ----------
    - point : np.array
    - p_color : (r,g,b) value
         arg to choose the color of point
    - pointf : np.array
         draw lines between point and pointf
    - l_color : (r,g,b) value
         arg to choose the color of lines
    - grid : to display a grid, the grid must be a grid object
    - surfColor : to choose the grid color"""

    if isinstance(image, str):
        image = cv2.imread(image, 0)

    if 'text' in kwargs:
        text = kwargs['text']
        image = cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 4)

    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if 'point' in kwargs:
        p_color = (160, 120,
                   0) if 'p_color' not in kwargs else kwargs['p_color']
        for pt in kwargs['point']:
            if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                x = int(pt[0])
                y = int(pt[1])
                frame = cv2.circle(frame, (x, y), 4, p_color, -1)

    scale = 1. if 'scale' not in kwargs else kwargs['scale']
    if 'pointf' in kwargs and 'point' in kwargs:
        assert len(kwargs['point']) == len(kwargs['pointf']), 'bad size'
        l_color = (255, 120,
                   255) if 'l_color' not in kwargs else kwargs['l_color']
        for i, pt0 in enumerate(kwargs['point']):
            pt1 = kwargs['pointf'][i]
            if np.isnan(
                pt0[0]) == False and np.isnan(
                pt0[1]) == False and np.isnan(
                pt1[0]) == False and np.isnan(
                    pt1[1]) == False:
                disp_x = (pt1[0] - pt0[0]) * scale
                disp_y = (pt1[1] - pt0[1]) * scale
                frame = cv2.line(frame, (pt0[0], pt0[1]),
                                 (int(pt0[0] + disp_x), int(pt0[1] + disp_y)),
                                 l_color, 2)

    if 'grid' in kwargs:
        gr = kwargs['grid']
        surfColor = (255, 255,
                     255) if 'surfColor' not in kwargs else kwargs['surfColor']
        for i in range(gr.size_x):
            for j in range(gr.size_y):
                if (not math.isnan(gr.Points_X[i, j]) and
                        not math.isnan(gr.Points_Y[i, j]) and
                        not math.isnan(gr.disp_x[i, j]) and
                        not math.isnan(gr.disp_y[i, j])):
                    x = int(gr.Points_X[i, j]) + int(gr.disp_x[i, j] * scale)
                    y = int(gr.Points_Y[i, j]) + int(gr.disp_y[i, j] * scale)

                    if i < (gr.size_x - 1):
                        if (not math.isnan(gr.Points_X[i + 1, j]) and
                                not math.isnan(gr.Points_Y[i + 1, j]) and
                                not math.isnan(gr.disp_x[i + 1, j]) and
                                not math.isnan(gr.disp_y[i + 1, j])):
                            x1 = int(gr.Points_X[i + 1, j]) + \
                                int(gr.disp_x[i + 1, j] * scale)
                            y1 = int(gr.Points_Y[i + 1, j]) + \
                                int(gr.disp_y[i + 1, j] * scale)
                            frame = cv2.line(
                                frame, (x, y), (x1, y1), surfColor, 2)

                    if j < (gr.size_y - 1):
                        if (not math.isnan(gr.Points_X[i, j + 1]) and
                                not math.isnan(gr.Points_Y[i, j + 1]) and
                                not math.isnan(gr.disp_x[i, j + 1]) and
                                not math.isnan(gr.disp_y[i, j + 1])):
                            x1 = int(gr.Points_X[i, j + 1]) + \
                                int(gr.disp_x[i, j + 1] * scale)
                            y1 = int(gr.Points_Y[i, j + 1]) + \
                                int(gr.disp_y[i, j + 1] * scale)
                            frame = cv2.line(
                                frame, (x, y), (x1, y1), surfColor, 4)
    if 'file_Name' in kwargs:
        cv2.imwrite(kwargs['file_Name'], frame)
        return

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', frame.shape[1], frame.shape[0])
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_img_data(raw_Image_Data_File, image, points):
    """Writes image data points to the designated txt file."""
    raw_Image_Data_File.write(image + '\t')
    for p in points:
        raw_Image_Data_File.write(str(p[0]) + ',' + str(p[1]) + '\t')
    raw_Image_Data_File.write('\n')


def capture_Parameters():
    """UI to check Brightness, Contrast, ISO and Exposure compensation of picamera and define values"""

    def nothing(x):
        pass

    cv2.namedWindow('Press_C_to_Confirm')
    cv2.createTrackbar('Brightness', 'Press_C_to_Confirm', 50, 100, nothing)
    cv2.createTrackbar('Contrast', 'Press_C_to_Confirm', 50, 100, nothing)
    cv2.createTrackbar('Exposure', 'Press_C_to_Confirm', 25, 50, nothing)
    cv2.createTrackbar('ISO', 'Press_C_to_Confirm', 1, 1600, nothing)
    camera.exposure_mode = 'beach'  # Can be changed. Follow picamera documentation
    camera.awb_mode = 'tungsten'  # Can be changed. Follow picamera documentation
    camera.start_preview(fullscreen=False, window=(600, 400, 640, 480))
    while True:
        brightness = cv2.getTrackbarPos("Brightness", "Press_C_to_Confirm")
        camera.brightness = brightness
        contrast = cv2.getTrackbarPos("Contrast", "Press_C_to_Confirm")
        camera.contrast = contrast
        expos = cv2.getTrackbarPos("Exposure", "Press_C_to_Confirm")
        expos = int(expos - 25)
        camera.exposure_compensation = expos
        iso = cv2.getTrackbarPos("ISO", "Press_C_to_Confirm")
        camera.iso = iso
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            camera.stop_preview()
            cv2.destroyAllWindows()
            break
    # This can be changed based on the processing power and speed of the device
    # For maximum resolution, check picamera documentation
    camera.resolution = (1920, 1080)


def displacement_Compute(point, pointf):
    """Computes displacement between two image point arrays"""

    assert len(point) == len(pointf)
    values = []
    for i, pt0 in enumerate(point):
        pt1 = pointf[i]
        values.append((pt1[0] - pt0[0], pt1[1] - pt0[1]))
    return values


def select_ROI(captured_Img):
    """UI to select the Region of Interest on the 2D surface"""

    global area, cropping
    image = cv2.putText(
        captured_Img,
        "Choose Region of Interest (ROI) | Press c to Confirm, r to Reset",
        (50, 1000), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)

    def select_Point(event, x, y, flags, param):
        global area, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            area = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            area.append((x, y))
            cropping = False
            draw_Image = cv2.rectangle(
                image, area[0], area[1], (125, 115, 65), 2)
            cv2.imshow('image', draw_Image)

    clone = image.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640, 480)
    cv2.setMouseCallback("image", select_Point)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
        elif key == ord("c"):
            break
    return area


def list_Point_Final(points, area, *args, **kwargs):
    """An internal function to organize the image points for further calculation"""

    xmin = area[0][0]
    xmax = area[1][0]
    ymin = area[0][1]
    ymax = area[1][1]
    res = []
    for p in points:
        x = p[0]
        y = p[1]
        if ((x >= xmin) and (x <= xmax) and (y >= ymin) and (y <= ymax)):
            res.append(p)
    return np.array(res)


def displacement_Measure(p1, p2):
    """A supporting function for displacement_Compute to measure displacement"""

    A = []
    B = []
    removed_indices = []
    for i in range(len(p1)):
        if np.isnan(p1[i][0]):
            assert np.isnan(p1[i][0]) and np.isnan(p1[i][1]) and np.isnan(
                p2[i][0]) and np.isnan(p2[i][1])
            removed_indices.append(i)
        else:
            A.append(p1[i])
            B.append(p2[i])
    A = np.matrix(A)
    B = np.matrix(B)
    assert len(A) == len(B)
    N = A.shape[0]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = np.matrix(A - np.tile(centroid_A, (N, 1)))
    BB = np.matrix(B - np.tile(centroid_B, (N, 1)))

    H = np.transpose(AA) * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    n = len(A)
    T = -R * centroid_A.T + centroid_B.T
    A2 = (R * A.T) + np.tile(T, (1, n))
    A2 = np.array(A2.T)
    out = []
    j = 0
    for i in range(len(p1)):
        if np.isnan(p1[i][0]):
            out.append(p1[i])
        else:
            out.append(A2[j])
            j = j + 1
    out = np.array(out)
    return displacement_Compute(p2, out)


def raw_Data_Process(raw_Image_Data_File, number, *args, **kwargs):
    """the raw_Data_Process is a a wrapper function that reads the raw result txt file, calculates displacements and strain fields
    and generate interpolated/non-interpolated data for using in Plot Class. This also generates raw outputs if user prompts.
    Parameters
    ----------
    - 'raw_Image_Data_File' : Path to the file of the result txt file where raw image data is stored.
    - 'number' :  iteration step value, acquired from the DIC_Run function
    * Other Arguments:
    - 'interpolation' the allowed vals are 'raw', 'bivar_Spline'
        raw : No Interpolation
        bivar_Spline : interpolates displacement data using bivariate spline interpolation
    - 'export_RAW' : True or False ; True to get RAW displacement, strain and other important outputs for post processing.
    - 'scale_disp' : float; used to amplify the displacement values on displacement output images
    - 'scale_mesh' : float; used to amplify the pointwise mesh displacement values on mesh output images

    """

    interpolation = 'raw' if 'interpolation' not in kwargs else kwargs[
        'interpolation']
    export_RAW = True if 'export_RAW' not in kwargs else kwargs['export_RAW']
    scale_disp = 1. if 'scale_disp' not in kwargs else float(
        kwargs['scale_disp'])
    scale_mesh = 1. if 'scale_mesh' not in kwargs else float(
        kwargs['scale_mesh'])

    with open(raw_Image_Data_File) as f:
        head = f.readlines()[0:2]
    (xmin, xmax, xnum, window__Size_x) = [float(x) for x in head[0].split()]
    (ymin, ymax, ynum, window__Size_y) = [float(x) for x in head[1].split()]
    window__Size = (window__Size_x, window__Size_y)

    Points_X, Points_Y = np.mgrid[xmin:xmax:int(xnum) * 1j,
                                  ymin:ymax:int(ynum) * 1j]
    surface_Maps = grid(Points_X, Points_Y, int(xnum), int(ynum))

    list_Points_arr = []
    list_Image_arr = []
    list_Displacement_arr = []

    with open(raw_Image_Data_File) as f:
        res = f.readlines()[2:-1]
        for line in res:
            val = line.split('\t')
            list_Image_arr.append(val[0])
            point = []
            for pair in val[1:-1]:
                (x, y) = [float(x) for x in pair.split(',')]
                point.append(np.array([np.float32(x), np.float32(y)]))
            list_Points_arr.append(np.array(point))
            surface_grid.append(copy.deepcopy(surface_Maps))
    f.close()

    k = number
    for p, surface_Maps in enumerate(surface_grid):
        print('.')  # for registering values of k, do not comment
    print("computing strain field of", list_Image_arr[k], "...")
    disp = displacement_Measure(list_Points_arr[k], list_Points_arr[0])
    surface_Maps.init_Raw_Data(
        window__Size,
        list_Image_arr[0],
        list_Image_arr[k],
        list_Points_arr[0],
        list_Points_arr[k],
        disp)
    list_Displacement_arr.append(disp)
    surface_Maps.bivariate_Interpolation(
        list_Points_arr[0], disp, method=interpolation)
    surface_Maps.strain_Field_Compute()

    if (export_RAW):
        surface_Maps.correlation_Image()
        surface_Maps.displacement_Image(scale_disp)
        surface_Maps.surface_Image(scale_mesh)
        surface_Maps.raw_Data_CSV()


def DIC_Run(image_Path, file_Name, window_Size_px, mesh_size_px,
            raw_Image_Data_File, *args, **kwargs):
    """
    This DIC_Run function is a simple wrapper function that captures images in a specific time interval.
    Region of interest is selected by the user and displacements are computed with each increment. A result txt file
    is written for image correlation.

    - 'image_Path' : the path where images will be captured
    - 'window_Size_px' : size in pixel of your correlation windows for LK Method
    - 'mesh_size_px' : the size of your correlation grid or mesh. The smaller the better, but requires more processing
    power
    - 'raw_Image_Data_File' : locates your result txt file
    """

    capture_Parameters()
    camera.start_preview()
    time.sleep(1)
    file_location = './Test_%s/' % file_Name
    shutil.rmtree(file_location)
    if not os.path.exists(file_location):
        os.mkdir(file_location)

    camera.capture(file_location + '0.jpg')
    camera.stop_preview()
    img_ref = cv2.imread(file_location + '0.jpg', 0)
    img_list = sorted(glob.glob(image_Path), key=lambda t: os.stat(t).st_mtime)

    area = select_ROI(img_ref)
    points = []
    points_x = np.float64(np.arange(area[0][0], area[1][0], mesh_size_px[0]))
    points_y = np.float64(np.arange(area[0][1], area[1][1], mesh_size_px[1]))

    for x in points_x:
        for y in points_y:
            points.append(np.array([np.float32(x), np.float32(y)]))
    points = np.array(points)
    point_List_Final = list_Point_Final(points, area, shape='box')

    img_ref = cv2.imread(img_list[0], 0)
    img_ref = cv2.putText(img_ref, "Press any button to continue", (50, 1000),
                          cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    display_Image(img_ref, point=point_List_Final)

    f = open(raw_Image_Data_File, 'w')
    xmin = points_x[0]
    xmax = points_x[-1]
    xnum = len(points_x)
    ymin = points_y[0]
    ymax = points_y[-1]
    ynum = len(points_y)
    f.write(
        str(xmin) + '\t' + str(xmax) + '\t' + str(int(xnum)) + '\t' +
        str(int(window_Size_px[0])) + '\n')
    f.write(
        str(ymin) + '\t' + str(ymax) + '\t' + str(int(ynum)) + '\t' +
        str(int(window_Size_px[1])) + '\n')

    # Lucas-Kanade Method Paramaters from OpenCV
    lk_params = dict(winSize=window_Size_px,
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    ref_Points = point_List_Final
    write_img_data(f, img_list[0], ref_Points)
    f.close()

    i = 0
    plt.ion()
    while True:
        f = open(raw_Image_Data_File, 'a')
        img_list = sorted(glob.glob(image_Path),
                          key=lambda t: os.stat(t).st_mtime)
        image_ref = cv2.imread(img_list[i], 0)
        camera.capture(file_location + '%i.jpg' % (i + 1))
        img_list = sorted(glob.glob(image_Path),
                          key=lambda t: os.stat(t).st_mtime)
        image_str = cv2.imread(img_list[i + 1], 0)
        ref_Points, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_str,
                                                       ref_Points, None,
                                                       **lk_params)
        write_img_data(f, img_list[i + 1], ref_Points)
        f.close()

        img_list = sorted(glob.glob(image_Path),
                          key=lambda t: os.stat(t).st_mtime)
        raw_Data_Process('./rawdata_%s.txt' % file_Name,
                         i,
                         interpolation='bivar_Spline',
                         export_RAW=False)

        surfaceArr = surface_grid[-1]
        os.mkdir(file_location + 'fig_%s' % (i))

        surfaceArr.insitu_Plot(surfaceArr.strain_longitudinal, 'x strain', 1,
                               file_Name)
        plt.savefig(file_location + 'fig_%s/x_strain_%s.png' % (i, i), dpi=300)
        surfaceArr.insitu_Plot(
            surfaceArr.strain_transverse,
            'y strain',
            2,
            file_Name)
        plt.savefig(file_location + 'fig_%s/y_strain_%s.png' % (i, i), dpi=300)
        print("Saved Strain Field for Image_%s..." % i)

        plt.show(block=False)
        plt.pause(1)
        i = i + 1
    f.write('\n')
    f.close()


""" This part is executed and user information is to be given here. """

while True:
    file_Name = input("Folder Name: ")
    if os.path.exists('./Test_%s' % file_Name):
        print('Folder exists already\nwarning: Data will be overwritten \nChoose different name')
        continue
    else:
        os.mkdir('./Test_%s' % file_Name)
        break

DIC_Run('./Test_%s/*.jpg' % file_Name, file_Name,
        lk_window_size, lk_grid_size, "rawdata_%s.txt" % file_Name)
