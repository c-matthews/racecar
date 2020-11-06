rm dist/*
python3 setup.py sdist bdist_wheel
 
echo "Now run:  twine upload dist/*" 
