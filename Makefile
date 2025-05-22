install:
	pip install -v .

reinstall:
	pip uninstall -y af3_kernels
	python setup.py clean
	rm -rf build dist
	python setup.py install

clean:
	python setup.py clean
