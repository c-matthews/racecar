import setuptools
import src.racecar._version as v

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="racecar",
    version=v.version_number,
    author="Charles Matthews",
    author_email="mail@cmatthe.ws",
    description="A lightweight Bayesian sampling package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/c-matthews/racecar",
    install_requires=['numpy', 'scipy'],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
