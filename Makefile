PROJECT_NAME := what-the-ohm

env_init:
	@conda create -f environment.yml

env_export:
	@conda env export | grep -v "^prefix: " > environment.yml

train:
	@python train.py

jupyter:
	@jupyter lab
