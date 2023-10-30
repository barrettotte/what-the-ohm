PROJECT_NAME := what-the-ohm

env_init:
	@conda create -f environment.yml

env_export:
	@conda env export | grep -v "^prefix: " > environment.yml

build:
	@echo 'build not implemented'

train:
	@python train.py

run:
	@echo 'train not implemented'

clean:
	@echo 'clean not implemented'

jupyter:
	@jupyter lab
