#!/usr/bin/bash

set -e
# input param 
# $1: sequence name
# $2: whether the video is upper body only (false by default, enable by -f)

seqName=$1
dataDir=$2
sampleNum=$3
speaker=$4

if [[ $sampleNum -eq 500 ]]; 
then
	tag=$5
fi

# Assume that you already have a video in $dataDir/(seqName)/(seqName).mp4 
# dataDir=/home/evonneng/Documents/research/ssp/ssp_expression/video_data/monologue/Oliver/test/

# # Git clone openpose to ../openpose and compile with cmake
# openposeDir=../openpose/

# convert to absolute path
dataDir=$(readlink -f $dataDir)
MTCDir=$(readlink -f .)

if [ ! -f $dataDir/$seqName/calib.json ]; then
        echo "Camera intrinsics not specified, use default."
        cp -v POF/calib.json $dataDir/$seqName
fi


cd $MTCDir/FitAdam/
if [ ! -f ./build/run_fitting ]; then
	echo "C++ project not correctly compiled. Please check your setting."
fi


./build/run_fitting --root_dirs $dataDir --seqName $seqName --start 0 --end 1600 --stage 1 --imageOF --sample_num $sampleNum --tag $tag --speakerName $speaker

