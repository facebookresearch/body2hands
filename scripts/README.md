# Download data:
## Instructions and dataset structure
```
sh ./scripts/download.sh
```

Please note, the above script will download all of the test, train, and demo data. However, if you want to download only a part of the dataset, please look inside the `download_data.sh` script for individual download URLs.

The fully downloaded data will unpack into the following directory structure:

```
|-- video_data/
    |-- Chemistry/
        |-- train/
            |-- filepaths.npy
            |-- full_bodies2.npy
            |-- fully_hands2.npy
            |-- full_resnet.npy
    |-- Conan/
    |-- Multi/
        |-- sample/
        |-- conan_frank/
    |-- Oliver/
    |-- Seth/
    |-- Test/
```

Above, we only list the 4 individuals we train on: `Chemistry, Conan, Oliver, Seth`. However, please note we also include annotations for 5 additional speakers in the tar: `Almaram, Angelica, Ellen, Rock, Shelly`.

The test data and follow-along demo data we provide are located in `Test/` and `Multi/` respectively.

## Annotation information

N = number of sequences

T = number of frames in each sequence. For our purposes, it is 64.

#### filepaths.npy: (N x T x 3) array.
- filepaths[i,j,0]: video name of the sequence.
- filepaths[i,j,2]: frame number indexing into the original full video.

Note: We only provide 3D full body and resnet feature annotations. To obtain the associated videos, please refer to [speech2gesture](https://github.com/amirbar/speech2gesture/blob/master/data/dataset.md).

#### full_bodies2.npy: (N x T x 36) array.
- each 36D feature can be broken down into a 6x6 matrix, where the 6 joints referring to the arms are expressed in 6D rotation space.
- The 6 joints are as follows:
	- 0: left collar,
	- 1: right collar,
	- 2: left shoulder,
	- 3: right shoulder,
	- 4: left elbow,
	- 5: right elbow
- To convert back to 3D axis-angle representation, you may use something like this [function](https://github.com/facebookresearch/body2hands/blob/master/utils/load_utils.py#L60).

#### full_hands2.npy: (N x T x 252) array.
- each 252D feature can be broken down into a (2 x 21 x 6) matrix, where there are 2 hands, each composed of 21 joints represented in 6D rotation space.
- you can separate out the hands via:

```
## assuming some arbitrary video and frame number index (i,j)
hands = np.reshape(full_hands2, (N, T, 42, 6))
left_hand = full_hands2[i, j, :21, :]
right_hand = full_hands2[i, j, 21:, :]
```

#### full_resnet.npy: (N x T x 1024) array.
- each 1024D feature can be broken down into a (2 x 512) matrix, where there are 2 hand image features, each of 512D.
- these are simply pre-processed image features, where both hand images (left, right) is fed through a pretrained resnet34 individually, which outputs a 512D feature per hand.


# Download models (Coming soon!):
```
sh ./scripts/download_models
tar xf b2h_models.tar
```
We provide the pre-trained models for both our body-only model **(Ours w/ B)** and the body with image model **(Ours w/ B+I)**.
