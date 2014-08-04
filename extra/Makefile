name = practical-category-recognition-2014a
DST = vgg@login.robots.ox.ac.uk:
DSTDOC = vgg@login.robots.ox.ac.uk:WWW/practicals/category-recognition

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

code=\
computeFeatures.m \
trainEncoder.m \
encodeImage.m \
displayRankedImageList.m \
displayRelevantVisualWords.m \
exercise1.m \
exercise2.m \
getImageSet.m \
removeSpatialInformation.m \
setup.m \
standardizeImage.m \
trainLinearSVM.m \
README.md \
vlfeat

doc=\
doc/images \
doc/instructions.html

data=\
/data/{aeroplane,background,car,horse,motorbike,person}_train_{bovw,vlad,fv}.mat \
/data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
/data/{aeroplane,background,car,horse,motorbike,person}_val_{bovw,vlad,fv}.mat \
/data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
/data/images \
/data/myImages \
/data/encoder_{bovw,vlad,fv}.mat

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))
deps:=$(shell find $(code) $(doc) $(data) -type f | sed "s/ /\\\\ /g")

pack: data/$(name).tar.gz
pack-data: data/$(name)-data-only.tar.gz
pack-code: data/$(name)-code-only.tar.gz

data/$(name).tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(data) data/$(name)/data/
	ln -sf $(doc) data/$(name)/doc/
	ln -sf $(code) data/$(name)/
	tar -C data -czvhf data/$(name).tar.gz $(name)/

data/$(name)-data-only.tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(data) data/$(name)/
	tar -C data -czvhf data/$(name)-data-only.tar.gz $(name)/

data/$(name)-code-only.tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(doc) data/$(name)/doc/
	ln -sf $(code) data/$(name)/
	tar -C data -czvhf data/$(name)-code-only.tar.gz $(name)/

post-doc:
	rsync -rv doc/images $(DSTDOC)/
	rsync -v doc/instructions.html $(DSTDOC)/index.html

post: pack-all
	rsync -v data/$(name).tar.gz $(DST)/
	rsync -v data/$(name)-data-only.tar.gz $(DST)/
	rsync -v data/$(name)-code-only.tar.gz $(DST)/

clean:
	find . -name '*~' -delete

distclean: clean
	rm -f data/$(name)*.tar.gz
