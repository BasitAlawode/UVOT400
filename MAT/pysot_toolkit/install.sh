rm utils/*.so
python setup.py clean --all
python setup.py build_ext --inplace
