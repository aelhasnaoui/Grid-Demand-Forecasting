.PHONY: install train clean

install:
	pip install -e .

train:
	@echo "Starting training process..."
	@echo "Please ensure you have the dataset (e.g., PJM_Load_hourly.csv) ready."
	@echo "Running TFT.py..."
	python TFT.py

clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
