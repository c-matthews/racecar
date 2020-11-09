python setup.py install

rm dist/*
python3 setup.py sdist bdist_wheel
cd docs
make clean;
make html;

echo "  "
echo "Now run:  twine upload dist/*"
