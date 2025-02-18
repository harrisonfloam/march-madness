from setuptools import setup, find_packages

setup(
    name="march_madness",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
