from setuptools import setup
import setuptools

setup(
    name="AILabs",
    author="Nikhil",
    packages=setuptools.find_packages(),
    version='1.0',
    description='This package contains libraries related to CV',
    license='MIT',
    install_requires=[
        'tqdm',
    ],
)
