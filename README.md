# CANN-YOLO26-Atlas

<p align="center">
  <img alt="platform" src="https://img.shields.io/badge/Platform-Huawei%20Atlas-blue">
  <img alt="toolkit" src="https://img.shields.io/badge/Toolkit-CANN-orange">
  <img alt="language" src="https://img.shields.io/badge/C%2B%2B-17-brightgreen">
  <img alt="opencv" src="https://img.shields.io/badge/OpenCV-required-yellowgreen">
  <img alt="status" src="https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey">
</p>

## Overview

This repository contains a practical C++ deployment project for **YOLO26** on **Huawei Atlas** devices using **CANN / AscendCL**.

The project currently supports:

- **FP16 inference**
- **AIPP-based preprocessing**
- **pure detection**
- **detection + tracking**
- **video-file inference**
- **USB camera inference**

## Repository Structure

```text
cann-yolo26-atlas/
├── CANN-CPP-DETECTION/
│   ├── CMakeLists.txt
│   └── main.cpp
├── CANN-CPP-TRACK/
│   ├── CMakeLists.txt
│   └── main.cpp
├── .gitignore
└── README.md
```


## Results

### Detection
![detection demo](assets/detection_demo.gif)

### Tracking
![tracking demo](assets/tracking_demo.gif)

### USB Camera
![usb camera demo](assets/usb_camera_demo.gif)

## Requirements

This project is intended to run on a **Huawei Atlas** device with a local **CANN Toolkit** installation. Huawei’s Ascend documentation portal is the official entry point for CANN and Atlas-related manuals and setup references. ([昇腾社区][2])

Typical environment:

* Huawei Atlas device
* CANN Toolkit
* OpenCV
* CMake
* GCC / G++
* Ubuntu-based runtime environment on device

## Model Weights

This project uses **YOLO26** weights from Ultralytics, for example:

* `yolo26n.pt`

Ultralytics documents YOLO26 as the latest edge-oriented YOLO family member with DFL removal and end-to-end NMS-free inference. ([Ultralytics Docs][1])

Typical export example on PC:

```bash
yolo export model=yolo26n.pt format=onnx imgsz=640 batch=1 simplify=True opset=13
```

## COCO Validation Reference

For validation / calibration experiments, you can use **COCO 2017 val**.

References:

* COCO official homepage: [https://cocodataset.org/](https://cocodataset.org/) ([cocodataset.org][3])
* COCO val2017 zip: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip) ([Gist][4])

Ultralytics’ COCO dataset page also summarizes that COCO val2017 contains 5K validation images. ([Ultralytics Docs][5])

## Build

### Detection

```bash
cd CANN-CPP-DETECTION
mkdir -p build
cd build
cmake ..
make -j
```

### Tracking

```bash
cd CANN-CPP-TRACK
mkdir -p build
cd build
cmake ..
make -j
```

## Quick Start

### 1. Video-file inference

In `main.cpp`, set:

```cpp
const std::string kVideoSource = "/path/to/your/video.mp4";
```

Then run:

```bash
./yolo26_acl_demo
```

### 2. USB camera inference

In `main.cpp`, set:

```cpp
const std::string kVideoSource = "0";
```

This uses the default webcam device, typically `/dev/video0`.

If needed, camera resolution can be set after `cap.open()` in OpenCV.

## AIPP Notes

The current deployment path uses static AIPP preprocessing. A common configuration for this project is:

```text
aipp_op {
    aipp_mode: static
    input_format: RGB888_U8
    csc_switch: false
    src_image_size_w: 640
    src_image_size_h: 640

    mean_chn_0: 0
    mean_chn_1: 0
    mean_chn_2: 0
    var_reci_chn_0: 0.003921568627
    var_reci_chn_1: 0.003921568627
    var_reci_chn_2: 0.003921568627
}
```

This corresponds to mapping input RGB values from `[0, 255]` into `[0, 1]`.

## CMake Notes

The project expects:

* OpenCV from the local system installation
* Ascend headers and libraries from local CANN installation

A typical path looks like:

```text
/usr/local/Ascend/ascend-toolkit/latest
```

The repository does **not** include system runtime libraries such as OpenCV shared libraries or `libascendcl.so`. These should come from the target device environment.

## Current Status

Implemented:

* [x] CANN / AscendCL inference
* [x] FP16 deployment
* [x] AIPP preprocessing
* [x] pure detection demo
* [x] tracking demo
* [x] video-file inference
* [x] USB camera inference

In progress / planned:

* [ ] cleaner project modularization
* [ ] INT8 deployment optimization
* [ ] DVPP / VPC preprocessing
* [ ] improved tracking module
* [ ] official ByteTrack integration
* [ ] trajectory visualization refinement
* [ ] better camera pipeline for realtime deployment

## TODO

* [ ] split reusable inference utilities from `main.cpp`
* [ ] add model conversion scripts
* [ ] add AIPP config examples
* [ ] add calibration dataset notes
* [ ] add deployment screenshots / GIFs
* [ ] add INT8 benchmark results
* [ ] add USB camera benchmark results
* [ ] add README in Chinese


