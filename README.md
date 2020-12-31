<img src="rel3d.gif" align="right" width="30%"/>

[**Rel3D: A Minimally Contrastive Benchmark for Grounding Spatial Relations in 3D**](https://arxiv.org/pdf/2012.01634.pdf)
[Ankit Goyal](http://imankgoyal.github.io), [Kaiyu Yang](https://www.cs.princeton.edu/~kaiyuy/), [Dawei Yang](http://www-personal.umich.edu/~ydawei/), [Jia Deng](https://www.cs.princeton.edu/~jiadeng/) <br/>
***Neural Information Processing Systems (NeuRIPS), 2020 (Spotlight)***

*For downloading the dataset and reproducing the results in the paper, use [the main repository](https://github.com/princeton-vl/Rel3D). This repository contains code for rendering images using the 3D data.*

## Getting Started

#### Clone Repository
First clone the repository. We would refer to the directory containing the code as `Rel3D_Render`.

```
git clone git@github.com:princeton-vl/Rel3D_Render.git
```

#### Requirements
The code is tested on Linux OS with Python version **3.7**. We recommend using a machine with a GPU for faster rendering.

#### Install Libraries
We recommend you to first install [Anaconda](https://anaconda.org/) and create a virtual environment.
```
conda create -n rel3d_render python=3.7 -y
```

Activate the virtual environment and install standard libraries. Make sure you are in `Rel3D_Render`.
```
conda activate rel3d_render
conda install opencv -y
conda install scipy -y
conda install pillow -y
```

Install third-party dependencies including [trimesh](https://github.com/mikedh/trimesh) and [fcl](https://github.com/BerkeleyAutomation/python-fcl):
```bash
conda install trimesh -c conda-forge -y
pip install fcl
pip install networkx
```

Download [blender](https://www.blender.org). Our code is tested with blender version 2.81. You can either use the following commands or download blender manually. In case you are using the following commands, make sure you are in the `Rel3D_Render` folder. In case you download blender manually (i.e. by not using the following commands), you would need to update this line (https://github.com/princeton-vl/Rel3D_Render/blob/master/config.py#L4) with the path where you downloaded blender.
```bash
wget https://download.blender.org/release/Blender2.81/blender-2.81-linux-glibc217-x86_64.tar.bz2
tar -jxvf blender-2.81-linux-glibc217-x86_64.tar.bz2
rm -rf blender-2.81-linux-glibc217-x86_64.tar.bz2
```

## Download Data
Make sure you are in `Rel3D_Render`. download.sh script can be used for downloading the data. It also places them at the correct locations. First, use the following command to provide execute permission to the download.sh script.

```
chmod +x download.sh
```

The shapes are required for rendering. The following command downloads them and places them at the correct location.
```
./download.sh shapes
```

(Optional) You can download our pre-generated data using the following command. It places the data in the `data/20200223`. For each sample there is a `.pkl`, `.png` and `.tiff` file. The `.png` and `.tiff` files store rgb and depth respectively at 720X1280 resolution. Information about object masks, bounding box, and surface normal is stored in the `.pkl` file.
```
./download.sh data_raw
```
**If you get error while executing the above command, you can manually download the data using this [link](https://drive.google.com/uc?id=1MSMwnX0znCfgEisj7zJ4ohFWJDrsxeme). After downloading the zip file, you need to extract it and place the extracted `20200223` folder inside the `data` folder.**

## Code
The main script for rendering is `render_scene.py`. The following command can be used to render the entire dataset.
```
python render_scene.py --output-folder <output_path> render --denoising
```

The `render_scene.py` can take the following arguments. They are defined [here](https://github.com/princeton-vl/Rel3D_Render/blob/master/render_scene.py#L216-L239). Note that one can use any suitable combination of these arguments.

- Whether to use image denoising is decided by the `denoising` argument. We recommend always using the `--denoising` argument.

- The height and width of the generated images can be changed using the `height` and `width` arguments which can be passed to the `render_scene.py`. For example, the following command can be used to render images of size 100X100.
  ```
  python render_scene.py --output-folder <output_path> --height 100 --width 100 render --denoising
  ```

- Since rendering the entire dataset on a single machine can take a long time, one can parallelize the process by dividing the dataset into chunks and running multiple processes at the same time. To do so, one can use the `array-index` and `array-total` arguments. `array-total` specifies the number of chunks and  `array-index` specifies the index of the current chunk. `array-index` should change from `0` to `array-total - 1`. For example, the following command can be used to render images of chunk `0` when the data is divided into `10` chunks.
  ```
  python render_scene.py --output-folder <output_path> render --denoising --array-total 10 --array-index 0
  ```

- To skip pre-generated images, one can use the `skip` argument. For example,
  ```
  python render_scene.py --output-folder <output_path> render --denoising --skip
  ```

- `start` and `end` can be used to specify which chunk of the data to render. If they are unspecified, the entire dataset is rendered.

- `sample-count` decides the number of samples `blender` uses for rendering. We recommend using the default value. A larger sample count improves the image quality but reduces rendering speed.

We also provide code for extracting the `3D features` which we used in our MLP baseline (Table 1, Column 8-9). These features can be extracted using the `transforms.py` script with the following command.
```
python transforms.py --output-folder <output_path> --img-path <img_path>
```
Here, `<output_path>` is the folder where the `.pkl` files are stored and `<img_path>` is the path of the image for which we want to extract the features. For example, in case one downloaded our pre-generated data using `./download.sh data_raw`, one could use the command `python transforms.py --output-folder ./data/20200223 --img-path "./data/20200223/behind - Wheel_wheel_3 - Bike_97bb8c467f0efadaf2b5d20b1742ee75 - initialState cam10 rgb.png"` to extract features for `behind - Wheel_wheel_3 - Bike_97bb8c467f0efadaf2b5d20b1742ee75 - initialState cam10 rgb.png` image.


If you find our research useful, consider citing it:
```
@article{goyal2020rel3d,
  title={Rel3D: A Minimally Contrastive Benchmark for Grounding Spatial Relations in 3D},
  author={Goyal, Ankit and Yang, Kaiyu and Yang, Dawei and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
