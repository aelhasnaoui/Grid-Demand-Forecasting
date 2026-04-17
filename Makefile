.PHONY: install train clean

install:
	pip install -e .

train:
	@echo "Starting training process..."
	@echo "Live data will be fetched using 'gridstatus' within the Jupyter Notebook."
	@echo "Please open and run 'GridForecast.ipynb'."


clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
