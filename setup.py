from setuptools import setup, find_packages

setup(
    name="SkyalyticAI",
    version="0.2.1",
    packages=find_packages(include=["skyalyticAI*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.9.0",
    ],
)
