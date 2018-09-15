LOGS=$(wildcard results/*.txt) $(wildcard oldresults/*/*.txt)
FIGURES=$(LOGS:txt=pdf)

figures: $(FIGURES)

push:
	rsync -avz data Makefile vfm.py fm.py run.sh fm.sh raiden:vae

retrieve:
	# rsync -avz raiden:vae/results/* oldresults/backup-sep-4/
	rsync -avz raiden:vae/rlog*.csv .

clean:
	rm -f logs/*
	rm -f results/*

%.pdf: %.txt
	python rule.py $<
