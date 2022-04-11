# Body2Hands: Learning to Infer 3D Hands from Conversational Gesture Body Dynamics (CVPR 2021)

[![report](https://img.shields.io/badge/arXiv-2007.12287-b31b1b.svg)](https://arxiv.org/abs/2007.12287#)

[Project page](http://people.eecs.berkeley.edu/~evonne_ng/projects/body2hands/)

This repository contains a pytorch implementation of "Body2Hands: Learning to Infer 3D Hands from Conversational Gesture Body Dynamics"

SOTA vs. Ours:

![](video_data/b2h_mtc_preview.gif)

This codebase provides:
- train code
- test code
- visualization code
- plug-in for smplx test code


## Overview:
We provide models and code to train/test the *body only* prior model and the *body + image* model

Body only vs. Body + image as input:

![](video_data/b2h_wi_preview.gif)

See below for sections:
- **Installation**: environment setup and installation for visualization
- **Download data and models**: download annotations and pre-trained models
- **Training from scratch**: scripts to get the training pipeline running from scratch
- **Testing with pretrianed models**: scripts to test pretrained models and to visualize outputs
- **SMPLx Plug-in Demo**: scripts to run models on SMPLx body model outputs from FrankMocap

## Installation:
 with Cuda version 10.1

```
conda create -n venv_b2h python=3.7
conda activate venv_b2h
pip install -r requirements.txt
```

#### Visualization (Optional):
Please follow the installation instructions outlined in the [MTC repo](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture).


## Download data and models:
Downloading data described [here](https://github.com/facebookresearch/body2hands/tree/master/scripts).

## Training from scratch:
```
## Flag definitions:
## --models: path to directory save checkpoints
## --base_path: path to directory where the directory video_data/ is stored
## --require_image: whether or not to include resnet feature as input

## to train with body only as input
python train_gan.py --model models/ --base_path ./`

## to train with body and image as input
python train_gan.py --model models/ --base_path ./` --require_image
```


## Testing with pretrained models:
To test with provided pretrained models (see above section "Download"). If training from scratch, replace `--checkpoint` as necessary.
```
## Flag definitions:
## --checkpoint: path to saved pretrained model
## --data_dir: directory of the test data where the .npy files are saved
## --base_path: path to directory where the directory video_data/ is stored
## --require_image: whether or not to include resnet feature as input
## --tag: (optional) naming prefix for saving results

## testing model with body only as input
python sample.py --checkpoint models/ours_wb_arm2wh_checkpoint.pth \
                 --data_dir video_data/Multi/sample/ \
                 --base_path ./ \
                 --tag 'test_'


## testing model with body and image as input
python sample.py --checkpoint models/ours_wbi_arm2wh_checkpoint.pth \
                 --data_dir video_data/Multi/sample/ \
                 --base_path ./ \
                 --tag 'test_wim_' \
                 --require_image
```

After running the above code, you can check to see if your outputs match the provided outputs we provide under `video_data/Multi/sample/chemistry_test/seq1/sample_results/test_predicted_body_3d_frontal/<%04d.txt>` or `video_data/Multi/sample/chemistry_test/seq1/sample_results/test_wim_predicted_body_3d_frontal/<%04d.txt>` depending on if you run with body only input or body+image input respectively.


#### Visualization (Optional)
Once you have run the above test script, the output will be saved to a `<path_to_sequence>/results/<tag>_predicted_body_3d_frontal/` directory as .txt files for each frame. We can then visualize the results as follows (see [MTC repo](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture) for installation instructions):

```
## to visualize results from above test commands, run ./wrapper.sh <sequence path> <tag>

cd visualization/

## visualize model with body only as input
./wrapper.sh ../video_data/Multi/sample/chemistry_test/seq1/ test_

## visualize model with body and image as input
./wrapper.sh ../video_data/Multi/sample/chemistry_test/seq1/ test_wim_
```

After running the above visualization code, you can check to see if the first few generated visualizations match ours by checking `video_data/Multi/sample/chemistry_test/seq1/sample_results/test_predicted_body_3d_frontal/<%04d.png>` or `video_data/Multi/sample/chemistry_test/seq1/sample_results/test_wim_predicted_body_3d_frontal/<%04d.png>` depending on if you run with body only input or body+image input respectively.


## SMPLx Plug-in Demo
We also provide a quick plug-in script for SMPLx body model compatibility using FrankMocap to obtain the 3D body poses (as opposed to Adam body model using MTC).
Please refer to [Frankmocap repo](https://github.com/facebookresearch/frankmocap/blob/master/docs/INSTALL.md) for installation instructions. Once the package is properly installed, follow the instructions `python -m demo.demo_frankmocap --input_path <path_to_mp4> --out_dir <path_to_output>` to generate files: `<path_to_output>_prediction_result.pkl`.

For the purposes of this demo, we provide a short example of expected FrankMocap outputs under `video_data/Multi/conan_frank/mocap/`


```
## Flag definitions:
## -- checkpoint: path to saved pretrained model
## --data_dir:  directory where all of the `*_prediction_result.pkl` files are saved
## --tag (optional) naming prefix for saving results

## to run on output smplx files from frankmocap
python -m smplx_plugin.demo --checkpoint models/ours_wb_arm2wh_checkpoint.pth \
                            --data_dir video_data/Multi/conan_frank/mocap/ \
                            --tag 'mocap_'

## to visualize
cd visualization && ./wrapper.sh ../video_data/Multi/conan_frank/mocap/ mocap_
```

## Coming Soon!
- Saved results from our methods as additional downloadable data

## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
See the [LICENSE](LICENSE) file.
