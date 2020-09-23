# RealPi2dDIC


### *A low-cost and open-source approach to in-situ 2D DIC applications*

-----
-----

Digital Image Correlation (DIC) is a non-contact optical technique that can be used to monitor the shape deformation and motion of rigid objects. Here we present *RealPi2dDIC*, a Python based open source real time DIC software, which provides a solution to the real time DIC applications using a very low-cost and mobile hardware interface. It can effectively compute in-plane full-field surface displacement and strain to sub-pixel accuracy by continuously capturing images of any rigid body in motion using a Pi camera module. This program is simple to use and can come up with highly accurate DIC results.

The application of Real Time DIC is limitless. To express it in a nutshell, this prevalent method can be used to get full field displacement and strain data of any rigid structure under tensile or compressive loading. It can be used in:

- The application in experimental solid mechanics (eg. Mechanical testing and characterization of metallic/composite materials, etc. ) 
- Monitoring of engineering components under mechanical or thermal loading
- Monitoring displacements in rail and road bridges 
- Crack opening in civil engineering structures, particularly in the nuclear industry. 

It also has the potential to be the viable alternative of manual inspection techniques of large civil engineering structures like buildings and power generation infrastructures and this list goes on.

As, the computation of gradual displacement from image data requires enormous amount of computational capability of the hardware interface, typically commercial in-situ DIC measurement equipment are very costly. Mostly a number of commercial vendors provide such services which are not relatively affordable for most research groups and companies. Hereby, we have developed this low-cost approach to in-situ DIC, which uses a *Pi camera module v2 (30$)* and a *Raspberry Pi system (35$)* to capture images and process them in an efficient algorithm to produce quality in-plane DIC measurements. This can be used in any indoor or outdoor applications and can be controlled remotely, thanks to the form factor of Raspberry Pi and its camera module. To add to its value, the program can be developed more to comply with nVidia Jetson NX or similar single board computers and Digital SLR cameras for faster computation.

## How to use?

## Surface Preparation

The basic operation of DIC is nothing but tracking a unique pattern, namely [Speckle Pattern](https://www.osti.gov/servlets/purl/1408942), on the object surface. Creation of speckle pattern is easy but it has to be good as the results significantly depend on the quality of this pattern. In order to provide unique information to track every data point or subset, the pattern should be non-repetitive, isotropic and of high contrast. The following qualities- the pattern must have:

- The pattern should be random, should not have a bias in any specific orientation
- Speckles should be 3-5 pixels in size to optimize spatial resolution and should be consistent
- Speckles should not be too sparse or too dense. The black and white portion each should cover roughly 50% of the surface. For example, if the speckles are 5 pixels in size they should be roughly 5 pixels apart from each other

<img src="https://i.ibb.co/x5H9Zw9/image-20200812192927805.png" style="zoom:100%;" align="center" />

#### Bad Pattern Types

<img src="https://i.ibb.co/R7YJ8BP/image-20200812195542852.png" style="zoom:100%;" align="center" />

#### Good Pattern Types

 <img src="https://i.ibb.co/FzM1NTz/image-20200812200104846.png" style="zoom:100%" align="center" />

### Procedure

Speckle Pattern can be applied in [a number of ways](https://youtu.be/U9FTmAZK6Yo?t=93). In our experiment, we have used Rust Oleum black spray on white spray print over the surface. This is one of the easiest and most used methods to apply speckle patterns, mostly in small to intermediate scale applications. It is a good choice for metallic, polymeric or composite materials specimens. 

In the following picture, you can see the speckle pattern on an Aluminum specimen.

![0](https://i.ibb.co/mGJcc1J/image.jpg)

Considerations regarding applying spray paints include:

- Use matte/flat paints. Do not use glossy paints which can cause specular reflections.
- White coating over the surface should be light enough not to cause any dripping. Otherwise, it will change the shape of the surface
- Black spray should only be used after the white paint is dried. If the base coat is wet, the white and black colors may get mixed and create grey color spots which will affect the accuracy of the results
- To create small speckles, the spray mist should be farther away from the specimen and stream should be moved continuously across the specimen.

For much smaller scale samples, toner powder, carbon black or graphite powder can be used. For very large-scale applications, like civil structures, one can literally print each dot and the dots can be huge compared to the position of the camera and size of the structure. You can Know more about the types of speckle patterns and application [here](https://www.correlatedsolutions.com/support/index.php?/Knowledgebase/Article/GetAttachment/80/14750) provided by Correlated Solutions Ltd.

## Hardware Setup

RealPi2dDIC application requires two specific hardware:

- Raspberry Pi (any model having a CSI port) running on Raspbian OS
- Pi Camera module v2

### Schematic Diagram

Here a schematic diagram of a [quasi static test](https://aboutcivil.org/quasi-static-test.html) of a flat Aluminum dog-bone shaped specimen incorporated with RealPi2dDIC setup is shown: 

![](https://i.ibb.co/y4rfpDJ/schem1.png)

### Actual Setup

![actual2](https://i.ibb.co/WpBPtnB/actual12.jpg)

We have attached the Pi Camera module to the MTS Testing Equipment as shown in the above picture. An external light source is used instead of the room light for better exposure only on top of the specimen and to have less environmental noise. The camera should be parallel to the plane of the specimen to get accurate results. Any out of plane motion should be avoided.

## Software Setup

### Install Modules

RealPi2dDIC is based on **Python3**. You have to install Python3 if it is not installed already ([instructions](https://projects.raspberrypi.org/en/projects/generic-python-install-python3)) . It also requires the following modules:

| Module Name   | Installation Instructions                                    |
| ------------- | ------------------------------------------------------------ |
| numpy         | [Click Here](http://helloraspberrypi.blogspot.com/2015/03/install-numpy-matplotlib-and-drawnow.html) |
| opencv-python | [Click Here](https://www.learnopencv.com/install-opencv-4-on-raspberry-pi/) |
| matplotlib    | [Click Here](http://michals-diy-electronics.blogspot.com/2018/04/matplotlib-and-raspberry-pi-3-show-my.html) |
| scipy         | [Click Here](https://www.scipy.org/install.html)             |
| picamera      | [Click Here](https://picamera.readthedocs.io/en/release-1.13/install.html) |

### Application Instructions

After installing all of the modules you can run RealPi2dDIC. To run it, you need to clone it from the GitHub Repository or you can download from [here](https://github.com/utaresearch/RealPi2dDIC). To clone the repository from GitHub, you need [Git](https://git-scm.com/). 

From your raspberry pi command line:

```sh
# install git
$ sudo apt install git-all

# clone the repository
$ git clone https://github.com/utaresearch/RealPi2dDIC.git

# change directory to RealPi2dDIC
# letter case matters
$ cd RealPi2dDIC

# run with Python 3 from the ./src/ directory
# try `python3` instead of `python` if the following does not work
$ python RealPi2dDIC.py
```

#### Step 1

At first give a name to a folder where all the raw images and result files will be saved. 

![DeepinScreenshot_select-area_20200813144858](https://i.ibb.co/r2B7m3P/Deepin-Screenshot-select-area-20200813144858.png)



#### Step 2

A camera preview screen and a window consisting the image capture parameters will appear. Select the best settings based on your lighting conditions. `Then press c to confirm`. Then it will capture the reference image and convert it to hsv mode.

![DeepinScreenshot_select-area_20200813145112](https://i.ibb.co/tYN6Xnr/Deepin-Screenshot-select-area-20200813145112.png)

*Recommended settings:*

- Brightness should be adjusted so that the image does not get over exposed.
- Contrast should be very high ( ~100).
- Exposure should be adjusted based on the lighting arrangements. *25* is the recommended value. *0* provides the least exposure while *50* provides the maximum exposure.
- ISO increases the sensitivity of the camera sensor, but also increases noise grain. Noise grains affects the results adversely to a great extent. So it is recommended to keep the ISO value around or below 10 based on the lighting arrangements

#### Step 3

In this step, you have to draw a rectangular Region of Interest on the specimen surface. If you want to reset the area and select it again `press r`.  `Then press c to confirm`. 

![Fig_roi](https://i.ibb.co/hYLM40Y/Fig-roi.png)

The software will divide the surface in small mesh grids (10 pixels x10 pixels)*. After you press any button the application will start collecting and processing the image data. It will plot in-situ full field strain on the reference image until you stop the process. To stop the process, `press ctrl+c`. 

All the raw images, raw calculation data and strain field plots will be stored in the folder you selected which can be used for post processing.

**you can change this size inside the source code*

## Output Example

![examplee2](https://i.ibb.co/LCnB0mL/illu-exm.jpg)

Figure: In-situ true strain field of a specimen at different time of a quasi-static test (a. Transverse, b. Longitudinal), c. Specimen breaks in the place where the strain is maximum.

<img src="https://i.ibb.co/0Q29RDg/ezgif-4-3eba52a3b50b.gif" style="zoom:50%;" />

## Video Demonstration

video: https://youtu.be/79wRLBrXLA4

## Documentation & Theory

Documentation can be found [here](https://utaresearch.github.io/RealPi2dDIC/docs.html). The theory and algorithm of the code is explained in a journal article (to be submitted for review soon)

## Contributing

You can fork the RealPi2dDIC repo, develop new algorithms and send us a pull request. To ensure great software quality please follow the following guidelines:

- The changes should be documented properly
- The changes should be reviewed by an objective party
- Use descriptive commit messages

## Authors

- Partha Pratim Das - *PhD Student, Mechanical Engineering, University of Texas at Arlington*
  parthapratim.das@mavs.uta.edu

- Muthu Ram Prabhu Elenchezhian - *PhD Student, Aerospace Engineering, University of Texas at Arlington*
  muthuramprabhu.elenchezhian@mavs.uta.edu

- Md Rassel Raihan - *Assistant Professor, Mechanical and Aerospace Engineering, University of Texas at Arlington*
  mdrassel.raihan@uta.edu

- Vamsee Vadlamudi - *Research Engineer II, Institute of Predictive Performance and Methodologies, University of Texas at Arlington Research Institute*
  vamsee.vadlamudi@uta.edu

- Kenneth Reifsnider - *Founding Director, Institute of Predictive Performance and Methodologies, University of Texas at Arlington Research Institute* 
  kenneth.reifsnider@uta.edu

## Acknowledgements

This work has been carried out at the Institute of Predictive Performance Methodologies (IPPM), the University of Texas at Arlington Research Institute (UTARI). The authors acknowledge the support from IPPM students and staff for their enormous support throughout the development of the application. 

Thanks to the other open source DIC projects like [pydic](https://gitlab.com/damien.andre/pydic/blob/master/pydic.py), [muDIC](https://github.com/PolymerGuy/muDIC), [PReDIC](https://github.com/texm/PReDIC) [py2dic](https://github.com/Geod-Geom/py2DIC), [DICe](https://github.com/dicengine/dice), etc. for their intense effort in developing different algorithms for DIC applications. These have supported primarily to develop the overall concept and algorithm of RealPi2dDIC. 

## Support

For any bug reporting and suggestions, please reach out to parthapratim.das@mavs.uta.edu or mdrassel.raihan@uta.edu

## License

This software is licensed under the MIT License. See LICENSE.md for more details.
