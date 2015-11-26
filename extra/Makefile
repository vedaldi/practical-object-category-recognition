name = practical-category-recognition-cnn-2015b
DST = vgg@login.robots.ox.ac.uk:WWW/share
DSTDOC = vgg@login.robots.ox.ac.uk:WWW/practicals/category-recognition-cnn
TMPDIR = /tmp

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

code=\
encodeImage.m \
displayRankedImageList.m \
displaySaliencyMap.m \
exercise1.m \
exercise2.m \
getImageSet.m \
setup.m \
standardizeImage.m \
trainLinearSVM.m \
README.md \
vlfeat \
matconvnet

doc=\
doc/images \
doc/instructions.html

data=\
data/{aeroplane,background,car,horse,motorbike,person}_train_vggm128-{fc7,conv5,conv4}.mat \
data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
data/{aeroplane,background,car,horse,motorbike,person}_val_vggm128-{fc7,conv5,conv4}.mat \
data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
data/images \
data/myImages \
data/cnn/imagenet-vgg-m-128.mat

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

doc/instructions.html : doc/instructions.md doc/base.html extra/Makefile
	cp doc/base.html "$@"
	python -m markdown \
	  -x toc -x footnotes -x tables \
	  "$<" >> "$@"
	cat doc/end.html >> "$@"

post-doc:
	rsync -rvt doc/images $(DSTDOC)/
	rsync -vt doc/instructions.html doc/base.css $(DSTDOC)/index.html

post: pack-all
	rsync -vt $(TMPDIR)/$(name).tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-data-only.tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-code-only.tar.gz $(DST)/

clean:
	find . -name '*~' -delete

distclean: clean
	rm -f $(TMPDIR)/$(name)*.tar.gz
