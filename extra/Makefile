name = practical-category-recognition-2013a
DST = vgg@login.robots.ox.ac.uk:WWW/share
%DST = ~/d/Dropbox/Public

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

data=\
/data/{aeroplane,background,car,horse,motorbike,person}_train_hist.mat \
/data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
/data/{aeroplane,background,car,horse,motorbike,person}_val_hist.mat \
/data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
/data/images/ \
/data/myImages \
/data/vocabulary.mat

code=\
computeFeatures.m \
computeHistogramFromImage.m \
computeHistogram.m \
computeHistogramsFromImageList.m \
computeVocabularyFromImageList.m \
displayRankedImageList.m \
displayRelevantVisualWords.m \
exercise1.m \
exercise2.m \
exercise3.m \
getImageSet.m \
quantizeDescriptors.m \
removeSpatialInformation.m \
setup.m \
standardizeImage.m \
trainLinearSVM.m

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))

prepack:
	rm -rf data/$(name)
	mkdir -p data/$(name)/data

pack: prepack
	ln -sft data/$(name)/data/ $(data)
	ln -sft data/$(name)/ $(code)
	tar -C data -czvhf data/$(name).tar.gz $(name)/

pack-data: prepack
	ln -sft data/$(name)/data/ $(data)
	tar -C data -czvhf data/$(name)-data-only.tar.gz $(name)/

pack-code: prepack
	ln -sft data/$(name)/ $(code)
	tar -C data -czvhf data/$(name)-code-only.tar.gz $(name)/

post: pack-all
	rsync -v data/$(name).tar.gz $(DST)/
	rsync -v data/$(name)-data-only.tar.gz $(DST)/
	rsync -v data/$(name)-code-only.tar.gz $(DST)/

clean:
	find -name '*~' -delete

distclean: clean
	rm -f data/$(name)*.tar.gz
