import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abstcal-ycui1",
    version="0.3",
    author="Yong Cui",
    author_email="yong.cui01@gmail.com",
    description="Calculate abstinence using the timeline follow back data in addiction research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ycui1/abstcal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)