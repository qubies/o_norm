from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call

setup(
    name='o_norm',
    version='1.0.1',
    url='https://github.com/qubies/onorm.git',
    author='Tobias Renwick',
    author_email='tobias@renwick.tech',
    description='O Norm can be used either to ofuscate words or to de-obfuscates hidden offensive words using a character level transformer network',
    packages=find_packages(),
    include_package_data=True)

