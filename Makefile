setup:
	python3 -m venv .env
	@echo "Virtual environment created."

install:
	. .env/bin/activate && \
	pip install -r requirements.txt
	sudo apt-get update
	sudo apt-get install -y libgl1-mesa-glx
	@echo "Requirements installed."

# test:
# 	export PYTHONPATH=$(PWD):$$PYTHONPATH; pytest tests/

all: setup install #test