.PHONY: clean

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf __pycache
	rm -rf .venv

clean_data:
	rm -rf data/
