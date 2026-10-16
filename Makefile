#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml_atmospheric_emission
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Pipeline 1: raw (excel) -> data/interim
.PHONY: data
data:
	$(PYTHON_INTERPRETER) ml_atmospheric_emission/dataset.py

## Pipeline 2: interim -> data/processed (features + targets of the heads)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) ml_atmospheric_emission/features.py

## Pipeline 3: processed -> models/ (heads that use a model)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) ml_atmospheric_emission/modeling/train.py

## Pipeline 4: probability of plume of each head at the last instant
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) ml_atmospheric_emission/modeling/predict.py

## Pipeline 5: recommendation of changes in the tags for the last instant
.PHONY: recommend
recommend:
	$(PYTHON_INTERPRETER) ml_atmospheric_emission/modeling/recommend.py

## Package to install anywhere (Databricks, another machine): dist/*.whl
.PHONY: wheel
wheel:
	$(PYTHON_INTERPRETER) -m pip wheel . --no-deps -w dist

## Whole pipeline (1 to 5)
.PHONY: pipeline
pipeline: data features train predict recommend



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
