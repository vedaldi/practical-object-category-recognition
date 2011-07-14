voc=/data/pascal/VOCdevkit_2007/VOC2007

mkdir -p data

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
