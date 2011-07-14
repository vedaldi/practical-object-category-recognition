name = practical-image-classification

pack:
	tar -C .. -czvf $(name).tar.gz \
	  --exclude '*/.git' \
	  --exclude '*/data/cache' \
	  --exclude '*/data/myImages' \
	  --exclude '*/*.tar.gz' \
	  $(name)
	cp $(name).tar.gz ~/Desktop/Dropbox/Public

pack-no-data:
	tar -C .. -czvf $(name)-no-data.tar.gz \
	  --exclude '*/.git' \
	  --exclude '*/data' \
	  --exclude '*/*.tar.gz' \
	  $(name)
	cp $(name)-no-data.tar.gz ~/Desktop/Dropbox/Public

