#!/bin/bash

# Download software
wget http://www.vlfeat.org/sandbox-matconvnet/models/imagenet-vgg-verydeep-16.mat \
    --output-document=data/cnn/imagenet-vgg-verydeep-16.mat --continue

if test ! -e vlfeat
then
    wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz --output-document=data/vlfeat-0.9.20-bin.tar.gz --continue
    tar xzvf data/vlfeat-0.9.20-bin.tar.gz
    mv vlfeat-0.9.20 vlfeat
fi

if test ! -e matconvnet
then
    wget http://www.vlfeat.org/sandbox-matconvnet/download/matconvnet-1.0-beta17.tar.gz \
         --output-document=data/matconvnet.tar.gz --continue
    tar xzvf data/matconvnet.tar.gz
    mv matconvnet-1.0-beta23 matconvnet
fi

# Download images
voc=data/tmp/VOCdevkit/VOC2007

mkdir -p data/tmp
mkdir -p data/myImages

if test ! -d data/tmp/VOCdevkit
then
    (
        cd data/tmp
        wget -c -nc http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        tar xzvf VOCtrainval_06-Nov-2007.tar
    )
fi

for j in train val
do
    for i in aeroplane person car motorbike horse
    do
        grep -E '.*\ 1$' $voc/ImageSets/Main/${i}_${j}.txt | \
            cut -f1 -d\  > data/${i}_${j}.txt
        grep -E '.*\-1$' $voc/ImageSets/Main/${i}_${j}.txt | \
            cut -f1 -d\  > data/${i}_background_${j}.txt
    done

    # intersect negative sets and create indexes
    cp data/aeroplane_background_${j}.txt data/background_${j}.txt
    for i in aeroplane person car motorbike horse
    do
        sort data/background_${j}.txt data/${i}_background_${j}.txt | \
            uniq -d > data/background_${j}_inters.txt
        rm -f data/${i}_background_${j}.txt
        mv data/background_${j}_inters.txt data/background_${j}.txt
    done

    # make all index
    sort data/*_{train,val}.txt | uniq > data/all.txt

    # copy images
    mkdir -p data/images
    cat data/all.txt | sed 's/\(.*\)/\1.jpg/' > data/temp.txt
    rsync --delete -v $voc/JPEGImages/ --files-from=data/temp.txt data/images/
    # rm data/temp.txt
done
