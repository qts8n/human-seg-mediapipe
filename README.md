# human-seg-mediapipe

> Human segmentation on video/stream using Mediapipe

## Requirements

- Python v3.9 or newer (works on v3.12 as well)
- vc_redist latest **(windows only)**
- CUDA v10.2 or newer **(optional)**

The script uses background and foreground animations stored in their
dedicated directories in the `assets` directory.

Every directory should contain animation frames in `.png`
format.

The script should automatically determine your default
camera and connect to it.

The script will automaticall use your Nvidia GPU
if you have one and all the necessery drivers are installed.

## Setup (Linux / MacOSX)

Setup virtual environment and install dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run (Linux / MacOSX)

```sh
python main.py
```

## Setup and Run (Windows)

```sh
init-camera-1.bat
```

## Keyboard Controls

- `Q` - quit the application.
- `F` - enter fullscreen mode.
- `N` - enter normal mode.
