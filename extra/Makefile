name = practical-image-classification
DST = vgg@login.robots.ox.ac.uk:WWW/share
%DST = ~/d/Dropbox/Public

pack-all: pack pack-data pack-code

pack:
	tar -C .. -czvf $(name).tar.gz \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_train_hist.mat \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_val_hist.mat \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
	  $(name)/data/images/ \
	  $(name)/data/myImages \
	  $(name)/data/vocabulary.mat \
	  $$(git ls-tree -r HEAD --name-only | sed "s/^/$(name)\//") \
	  $(name)/vlfeat
	rsync -v $(name).tar.gz $(DST)

pack-data:
	tar -C .. -czvf $(name)-data-only.tar.gz \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_train_hist.mat \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_train.txt \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_val_hist.mat \
	  $(name)/data/{aeroplane,background,car,horse,motorbike,person}_val.txt \
	  $(name)/data/images/ \
	  $(name)/data/myImages \
	  $(name)/data/vocabulary.mat
	rsync -v $(name)-data-only.tar.gz $(DST)

pack-code:
	tar -C .. -czvf $(name)-code-only.tar.gz \
	  $$(git ls-tree -r HEAD --name-only | sed "s/^/$(name)\//") \
	  $(name)/vlfeat
	rsync -v $(name)-code-only.tar.gz $(DST)
