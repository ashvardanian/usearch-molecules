from setuptools import setup, find_packages

setup(
    name="usearch-molecules",
    version="1.0.0",
    description="Library for molecule exploration using Unum's USearch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ash Vardanian",
    author_email="ash.vardanian@unum.cloud",
    url="https://github.com/ashvardanian/usearch-molecules",
    license="LICENSE",
    packages=find_packages(where="usearch_molecules"),
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
