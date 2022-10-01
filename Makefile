.PHONY: clean

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	./.venv/bin/pip3 install -r requirements.txt

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf __pycache
	rm -rf .venv
