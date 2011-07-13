mkdir -p data/download
wget -nc http://www.robots.ox.ac.uk/~vgg/data/motorbikes_side/motorbikes_side.tar -o data/download/motorbike.tar
wget -nc http://www.robots.ox.ac.uk/~vgg/data/airplanes_side/airplanes_side.tar -o data/download/airplane.tar
wget -nc http://www.robots.ox.ac.uk/~vgg/data/faces/faces.tar -o data/download/face.tar
wget -nc http://www.robots.ox.ac.uk/~vgg/data/background/background.tar -o data/download/background.tar
wget -nc http://www.robots.ox.ac.uk/~vgg/data/cars_brad/cars_brack.tar -o data/car.tar

for i in car face airplane motorbike background
do
    mkdir -p data/$i
    tar -C data/$i -xvf data/download/$i.tar --include='*.jpg' 
done

