# Copyright (c) Facebook, Inc. and its affiliates.

## Download visualization models
wget posefs1.perception.cs.cmu.edu/mtc/adam_blendshapes_348_delta_norm.json
mv -v adam_blendshapes_348_delta_norm.json ./visualization/FitAdam/model/

wget posefs1.perception.cs.cmu.edu/mtc/InitializeAdamData.h
mv -v InitializeAdamData.h ./visualization/FitAdam/include/

# Download train data for each individual speaker
wget https://dl.fbaipublicfiles.com/body2hands/chemistry.tar
wget https://dl.fbaipublicfiles.com/body2hands/conan.tar
wget https://dl.fbaipublicfiles.com/body2hands/oliver.tar
wget https://dl.fbaipublicfiles.com/body2hands/seth.tar

# Extra train data not used for training body2hands due to noisiness
wget https://dl.fbaipublicfiles.com/body2hands/almaram.tar
wget https://dl.fbaipublicfiles.com/body2hands/angelica.tar
wget https://dl.fbaipublicfiles.com/body2hands/ellen.tar
wget https://dl.fbaipublicfiles.com/body2hands/rock.tar
wget https://dl.fbaipublicfiles.com/body2hands/shelly.tar

# Test files
wget https://dl.fbaipublicfiles.com/body2hands/body2hands_test.tar

# Demo files used as part of walk through demo in README
wget https://dl.fbaipublicfiles.com/body2hands/body2hands_demo.tar

## untar files downloaded in current directory
for file in *.tar; do
   tar -xvf "${file}" ; done
