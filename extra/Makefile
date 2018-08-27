# Build practical

name ?= practical-category-recognition-cnn
ver ?= pytorch-2018a

code=\
README.md \
practical.ipynb \
lab.py

doc=\
doc/images \
doc/base.css \
doc/instructions.html

data=\
data/conv{1,2,3,4,5,6,7}.pth \
data/all.txt \
data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
data/images \
data/alexnet.pth

include extra/practical/Makefile

clean: nbstripout

.PHONY: nbstripout
nbstripout:
	nbstripout -f practical.ipynb
