import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="racecar", # Replace with your own username
    version="0.0.1",
    author="Charles Matthews",
    author_email="mail@cmatthe.ws",
    description="A lightweight Bayesian sampling package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/c-matthews/racecar",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
