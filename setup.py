from setuptools import setup, find_packages

setup(
    name="sagemaker_examples",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
