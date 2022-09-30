.PHONY: clean

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt

clean:
	rm -rf __pycache
	rm -rf .venv