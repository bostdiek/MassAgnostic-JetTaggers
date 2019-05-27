.PHONY: clean data lint requirements base_nn uBoost

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = massagnosticjettaggers
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifeq ($(CONDA_DEFAULT_ENV),$(PROJECT_NAME))
    ACTIVATE_ENV := true
else
    ACTIVATE_ENV := source activate $(PROJECT_NAME)
endif

# Execute python related functionalities from within the project's environment
define execute_in_env
    $(ACTIVATE_ENV) && $1
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Set up python interpreter environment and python dependencies
create_environment:
	(\
	if conda info -e | grep -q $(PROJECT_NAME); \
		then $(call execute_in_env, conda env update -f environment.yml);\
		else conda env create;\
	fi;\
	)

## Install Python Dependencies
requirements: test_environment

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Train Base networks
base_nn: data
	$(PYTHON_INTERPRETER) src/models/train_base_nn.py --prong=2
	$(PYTHON_INTERPRETER) src/models/train_base_nn.py --prong=3
	$(PYTHON_INTERPRETER) src/models/train_base_nn.py --prong=4

## Train networks on Planed data
planed_nn: data
	$(PYTHON_INTERPRETER) src/models/train_planed_nn.py --prong=2
	$(PYTHON_INTERPRETER) src/models/train_planed_nn.py --prong=3
	$(PYTHON_INTERPRETER) src/models/train_planed_nn.py --prong=4

## Train networks on PCA scaled data
pca_nn: data
	$(PYTHON_INTERPRETER) src/models/train_PCA_nn.py --prong=2
	$(PYTHON_INTERPRETER) src/models/train_PCA_nn.py --prong=3
	$(PYTHON_INTERPRETER) src/models/train_PCA_nn.py --prong=4

## Train uBoost classifiers
uBoost: data
	$(PYTHON_INTERPRETER) src/models/train_uBoost.py --prong=2
	$(PYTHON_INTERPRETER) src/models/train_uBoost.py --prong=3
	$(PYTHON_INTERPRETER) src/models/train_uBoost.py --prong=4

## Train Gradient Boosting Classifiers
GBC: data
	$(PYTHON_INTERPRETER) src/models/train_AdaBoost.py --prong=2
	$(PYTHON_INTERPRETER) src/models/train_AdaBoost.py --prong=3
	$(PYTHON_INTERPRETER) src/models/train_AdaBoost.py --prong=4

## Make the histograms from the predictions
predictions: data
	$(PYTHON_INTERPRETER) src/models/predict_models.py --prong=2
	$(PYTHON_INTERPRETER) src/models/predict_models.py --prong=3
	$(PYTHON_INTERPRETER) src/models/predict_models.py --prong=4

## Make the metrics from the predictions
metrics:
	$(PYTHON_INTERPRETER) src/test_metrics/run_metrics.py --prong=2
	$(PYTHON_INTERPRETER) src/test_metrics/run_metrics.py --prong=3
	$(PYTHON_INTERPRETER) src/test_metrics/run_metrics.py --prong=4

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
