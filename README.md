<p align="center">
  <img src="./assets/terra_nova_logo.png" width=50% />
</p>

<p align="center">
  <img src="./assets/viewer_demo.gif" width=90% />
</p>

Terra Nova is a comprehensive challenge environment for reinforcement learning research.
Terra Nova is based on and contains the challenges of Civilization V.

## Table of Contents
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Maps](#maps)
- [Documentation](#documentation)
- [Recording Games](#recording-games)
- [Replay Viewer](#replay-viewer)

## Installation
We provide a short requirements file that will auto-install everything all prerequisites using your preferred package manager.

Terra Nova is compatible with both older and newer combinations of Python and JAX. 

We've tested Terra Nova with the following combinations:
- Pyhton 3.9.18 + JAX 0.43.0 and 
- Python 3.13.9 + JAX 0.8.0


For example, with `mamba`:
```bash
mamba create -n terra_nova python=3.9.18
mamba activate terra_nova
pip install -r requirements.txt
```

or

```bash
mamba create -n terra_nova python=3.13.9
mamba activate terra_nova
pip install -r requirements_311.txt
```



## Quickstart
Terra Nova comes with utilities to automatically distribute environments across any arbitrary mesh of GPUs, allowing for increased throughput via horizontal scaling.

## Maps
Terra Nova games take place on procedurally-generated maps. We provide 10k maps you can download [here](https://huggingface.co/datasets/trevormcinroe/terra_nova_maps).
For more information on map contents and mechanics, see [here](https://trevormcinroe.github.io/terra_nova_environment#maps-mech).

We also provide utilities for generating your own maps, which can be read about [here](https://trevormcinroe.github.io/terra_nova_documentation#mapgen).


## Documentation
For documentation on environment mechanics, see [here](https://trevormcinroe.github.io/terra_nova_environment).

For documentation on the environment API, the action space, the observation space, and more, see [here](https://trevormcinroe.github.io/terra_nova_documentation).

## Recording Games
We highly recommend using CPU-only context for recording Terra Nova games. This is because the recorder captures the entirety of the gamestate on every played turn, which requires a significant amount of memory. 

You can accomplish this by using the following preamble:

```bash
CUDA_VISIBLE_DEVICES= python3 your_recording_script.py
```

## Replay Viewer
After a game has been recorded, you can watch the recording in your browser of choice. We recommend Chrome.

The Viewer uses node, which can be installed on Linux systems like:

```bash
sudo apt install nodejs
```

The Viewer also provides plots of various metrics over the course of the game:

<p align="center">
  <img src="./assets/pop_demos.png" width=75% />
</p>
<p align="center">
  <img src="./assets/prod_demos.png" width=75% />
</p>
<p align="center">
  <img src="./assets/gold_demos.png" width=75% />
</p>
