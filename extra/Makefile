name = practical-category-recognition-2015a
DST = vgg@login.robots.ox.ac.uk:WWW/share
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
data/{aeroplane,background,car,horse,motorbike,person}_train_{bovw,vlad,fv}.mat \
data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
data/{aeroplane,background,car,horse,motorbike,person}_val_{bovw,vlad,fv}.mat \
data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
data/images \
data/myImages \
data/encoder_{bovw,vlad,fv}.mat

tarflags=--exclude='.git*'

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))
doc:=$(addprefix $(CURDIR)/,$(doc))
deps:=$(shell find $(code) $(doc) $(data) -type f | sed "s/ /\\\\ /g")

pack: $(TMPDIR)/$(name).tar.gz
pack-data: $(TMPDIR)/$(name)-data-only.tar.gz
pack-code: $(TMPDIR)/$(name)-code-only.tar.gz

$(TMPDIR)/$(name).tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(data) $(TMPDIR)/$(name)/data/
	ln -sf $(doc) $(TMPDIR)/$(name)/doc/
	ln -sf $(code) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name).tar.gz $(tarflags) $(name)/

$(TMPDIR)/$(name)-data-only.tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(data) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name)-data-only.tar.gz $(tarflags) $(name)/

$(TMPDIR)/$(name)-code-only.tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(doc) $(TMPDIR)/$(name)/doc/
	ln -sf $(code) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name)-code-only.tar.gz $(tarflags) $(name)/

post-doc:
	rsync -rvt doc/images $(DSTDOC)/
	rsync -vt doc/instructions.html $(DSTDOC)/index.html

post: pack-all
	rsync -vt $(TMPDIR)/$(name).tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-data-only.tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-code-only.tar.gz $(DST)/

clean:
	find . -name '*~' -delete

distclean: clean
	rm -f $(TMPDIR)/$(name)*.tar.gz
