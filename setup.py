#!/usr/bin/env python3
from setuptools import setup
from setuptools import find_packages
from pathlib import Path

REQUIREMENTS_TXT = Path(__file__).parent / "requirements.txt"

setup(
    name='diallama',
    version='0.0.1dev',
    description='Framework for a dialogue system using pretrained language models',
    author='ÚFAL Dialogue Systems Group, Charles University',
    author_email='odusek@ufal.mff.cuni.cz',
    url='https://gitlab.com/ufal/dsg/diallama',
    download_url='https://gitlab.com/ufal/dsg/diallama.git',
    license='Apache 2.0',
    include_package_data=True,
    install_requires=[l for l in open(REQUIREMENTS_TXT).read().splitlines() if not l.startswith("#")],
    packages=find_packages()
)

