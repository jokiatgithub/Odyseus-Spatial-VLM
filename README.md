# Blinkin VLM

![Blinkin VLM demo](media/SpatialVLM-demo2-low-res.gif)

Blinkin VLM combines monocular depth estimation with a vision-language model to turn an uploaded image and natural-language prompt into a 3D point-cloud view with labeled targets.


Follow the setup below for custom deployment.

## Setup

This repo is currently set up primarily for Linux.

If you clone this as a git repo, prefer pulling the external DA3 dependency as a submodule:

```bash
git clone --recurse-submodules <repo-url>
cd <repo-folder>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

If you are packaging this repo yourself, `Depth-Anything-3/` is intended to track the upstream project as a submodule.

Set up the VLM environment:

```bash
./setup-vlm.sh
```

Set up the depth demo environment:

```bash
./setup.sh
```

## Run

Start the VLM server:

```bash
./run-vlm.sh
```

Start the depth demo:

```bash
./run.sh
```

Then open:

```text
http://localhost:8080
```

## Hosted Demo


The local repo remains the reference implementation for running and modifying the demo yourself.

## Use

1. Upload an image.
2. Enter a prompt like `select the chair near the desk and the closest door`.
3. Click `Run Demo`.
4. Inspect:
   - the 2D target overlay
   - the 3D point cloud
   - labeled 3D targets
   - the camera frustum and guide vectors

## Flow

```mermaid
flowchart LR
    A[User Prompt + Image] --> B[VLM]
    B --> C[2D Target Coordinates]
    A --> D[DA3 Metric Depth]
    C --> E[Depth Sampling]
    D --> E
    E --> F[3D Projection]
    F --> G[Three.js Viewer]
```

## Notes

- Linux is the best-supported path right now.
- PowerShell / Windows setup help is welcome. Contributions for improving `setup-vlm.ps1` or adding fuller Windows support are encouraged.
