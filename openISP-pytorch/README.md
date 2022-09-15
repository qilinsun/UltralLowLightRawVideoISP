
### Defective pixel mask 
     - CMOS have pixels that are defective
     - Dead pixel masks are pre-calibrated at the factory
        - Using “dark current” calibration
        - Take an image with no light
        - Record locations reporting values to make “mask”
    - Bad pixels in the mask are interpolated

![](https://i.imgur.com/XLEQa1L.png)


### black level correction
    Subtract the level from the “black” pixels in Optical Black (OB)
    <!-- caused by the dark current and circuit design -->
![](https://i.imgur.com/BCpQo0b.png)


### Lens shading mask
    vignetting 
    generate a lens shading map
![](https://i.imgur.com/KxiQC0i.png)


### anti-aliasing 
    a local low pass filter for removing moire pattern (caused by low sampling rate)
![](https://i.imgur.com/pKc0Xi4.jpg)


### tone mapping 
    HDR rendering (higher bits to lower bits) e.x. [0 4096] --> [0 255]
    todo

### AWB 
    "Gray world" algorithm assume R_mean == B_mean == G_mean to obtain the b_gain and r_gain

     - To realize auto white balance function, it requires to estimate the color temperature

    AWB is difficult since color temperature is hard to be recognized.
![](https://i.imgur.com/UOhBgrm.jpg)

### CFA Demosaicing
    #!!! https://github.com/guochengqian/TENet/blob/013608976e1c1f2b0e7a2d6cb832554ce081d61a/datasets/processdnd.py#L54
    

    https://github.com/mushfiqulalam/isp/blob/6fc30fce500d97cb2091387bf2742b8e4cc4495d/debayer.py#L35
     - Demosaicing can be combined with additional processing  
        - Highlight clipping
        - Sharpening
        - Noise reduction

![](https://i.imgur.com/s9m07Ca.jpg)



### noise reduction
    - Most cameras apply additional NR after A/D conversion
    - For high-end cameras, it is likely that cameras apply different strategies depending on the ISO settings, e.g. **high ISO will result in more noise, so a more aggressive NR could be used**
    - Smartphone cameras, because the sensor is small, apply aggressive noise reduction

## RGB domain
### gamma correction
    # https://zhuanlan.zhihu.com/p/79203830
    # https://pytorch.org/vision/master/generated/torchvision.transforms.functional.adjust_gamma.html#torchvision.transforms.functional.adjust_gamma
    gamma larger than 1 make the shadows darker,
    while gamma smaller than 1 make dark regions lighter.

    adaptive to the human vision system, which are more sensitive to the illumination than chrominance. [0 255] --> [0 255]
    
    Be used to enhance image contrast and dynamic range and dehaze
    
    Gamma correction is often realized in hardware by a look-up table, which is easy to change and program.
    
![](https://i.imgur.com/DFstwUx.jpg)


### color space conversion
    #! https://github.com/guochengqian/TENet/blob/013608976e1c1f2b0e7a2d6cb832554ce081d61a/datasets/processdnd.py#L115 

     - XYZ : canonical color ( or “device independent”) space to describe Spectral power distribution (SPD)
     - raw-RGB (or camera-RGB) represents the physical world's SPD "projected" onto the sensor's spectral filters.
     - sRGB has built in the assumed viewing condition (6500K daylight or D65 illuminant).
     - color space matrix is dedicated to transfer raw-RGB color space to sRGB color space.
     - 3D Lookup table is also used to map one color space to another (Often 33×33×33 cubes are used as 3D LUTs. Most products use trilinear interpolation)

