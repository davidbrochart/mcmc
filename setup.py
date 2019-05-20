import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcmc",
    version="0.0.1",
    author="David Brochart",
    author_email="david.brochart@gmail.com",
    description="MCMC algorithms (MH, SMC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidbrochart/mcmc",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
