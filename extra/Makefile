name = practical-image-classification

pack:
	tar -C .. -czvf $(name).tar.gz \
	  --exclude '*/.git' \
	  --exclude '*/data/cache' \
	  --exclude '*/data/myOtherImages' \
	  --exclude '*/*.tar.gz' \
	  --exclude '*DS_Store*' \
	  $(name)
	cp $(name).tar.gz ~/Desktop/Dropbox/Public

pack-data:
	tar -czvf $(name)-data-only.tar.gz \
	  --exclude '*/cache' \
	  --exclude '*/myOtherImages' \
	  --exclude '*DS_Store*' \
	  data
	cp $(name)-data-only.tar.gz ~/Desktop/Dropbox/Public

pack-code:
	tar -C .. -czvf $(name)-code-only.tar.gz \
	  --exclude '*/.git' \
	  --exclude '*/data' \
	  --exclude '*/*.tar.gz' \
	  --exclude '*DS_Store*' \
	  $(name)
	cp $(name)-code-only.tar.gz ~/Desktop/Dropbox/Public
