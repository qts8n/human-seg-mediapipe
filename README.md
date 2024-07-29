# human-seg-mediapipe

> Human segmentation on video/stream using Mediapipe

## Requirements

- Python v3.9 or newer (works on v3.12 as well)
- CUDA v10.2 or newer **(optional)**

## Setup

Setup virtual environment and install dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

The script uses background animations stored in their
dedicated directories in `assets` directory:

- `assets/idle` - Background idle animation.
- `assets/in` - Background animation-in.
- `assets/out` - Background animation-out.

Every directory should contain animation frames in `.png`
format. You can also change the foreground PNG by replacing
`assets/foreground.png`.

Run the script:

```sh
python main.py
```

The script should automatically determine you default
camera and connect to it.

The script will automaticall use your Nvidia GPU
if you have one and all the necessery drivers are installed.
