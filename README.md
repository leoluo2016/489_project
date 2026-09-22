# CUDA-Accelerated Golf Ball Tracking and Trajectory Simulation

Final project for **ECEN 489 / VIST 489 - GPU Programming** at Texas A&M University.

This project implements a GPU-accelerated pipeline for estimating golf-ball launch conditions from high-speed face-on video and simulating the resulting flight trajectory.

## Overview

The project consists of two primary stages:

1. **Computer Vision / Ball Tracking**
   - Processes high-speed golf swing video using OpenCV
   - Uses HSV color segmentation and frame differencing
   - Restricts detection to a launch-region wedge
   - Detects the golf ball using contour/blob analysis
   - Estimates launch speed and launch angle from consecutive ball positions
   - Accelerates per-pixel image processing using custom CUDA kernels

2. **Golf Ball Trajectory Simulation**
   - Simulates 2D golf-ball flight using launch speed, launch angle, and estimated spin
   - Models gravity, aerodynamic drag, Magnus lift, and spin decay
   - Performs spin-rate parameter sweeps to compare predicted carry distance
   - Uses CUDA to evaluate multiple trajectories in parallel

## CPU vs. CUDA Results

For a 1280 × 720 slow-motion video:

| Metric | CPU | CUDA | Speedup |
| --- | ---: | ---: | ---: |
| Pixel/kernel processing per frame | 1.234 ms | 0.164 ms | 7.52x |
| Dilation per frame | 0.166 ms | 0.023 ms | 7.22x |
| Total application runtime | 8.985 s | 8.298 s | 1.08x |

The CUDA implementation significantly accelerated the highly parallel pixel-processing stages. End-to-end speedup was smaller because video I/O, GPU memory transfers, OpenCV contour detection, visualization, and video output remained CPU-side.

For trajectory simulation, GPU acceleration became advantageous as the number of simultaneously evaluated trajectories increased, reaching approximately **1.72x speedup at 80 trajectories**.

## Source Files

- `face_on_timed.cpp` — CPU implementation of golf-ball tracking
- `two_blob_tracking_cuda_timed.cu` — CUDA-accelerated golf-ball tracking
- `tracer.cpp` — CPU golf-ball trajectory simulation
- `tracer_cuda.cu` — CUDA-accelerated trajectory simulation
- `ECEN489_Final_Project_Report.pdf` — complete project report

## Technologies

- CUDA
- C++
- OpenCV
- GPU Programming
- Parallel Computing
- Computer Vision
- Numerical Simulation
- SFML

## Building

### CPU Tracker

```bash
g++ face_on_timed.cpp -o ball_tracker $(pkg-config --cflags --libs opencv4)
./ball_tracker
```

### CUDA Tracker 

```bash
nvcc --expt-relaxed-constexpr -std=c++17 two_blob_tracking_cuda_timed.cu \
    -o ball_tracker_cuda $(pkg-config --cflags --libs opencv4)

./ball_tracker_cuda
