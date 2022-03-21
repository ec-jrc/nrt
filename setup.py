#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from setuptools import setup, find_packages
import os

# Parse the version from the main __init__.py
with open('nrt/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue


with codecs.open('README.rst', encoding='utf-8') as f:
    readme = f.read()

extra_reqs = {'tests': ['pytest'],
              'docs': ['sphinx', 'sphinx_rtd_theme']}

setup(name='nrt',
      version=version,
      description=u"Online monitoring with xarray",
      long_description_content_type="text/x-rst",
      long_description=readme,
      keywords='sentinel2, xarray, datacube, monitoring, change',
      author=u"Loic Dutrieux, Jonas Viehweger, Chris Holden",
      author_email='loic.dutrieux@ec.europa.eu',
      url='https://jeodpp.jrc.ec.europa.eu/apps/gitlab/use_cases/canhemon/nrt.git',
      license='EUPL-v1.2',
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      packages=find_packages(),
      package_data={'nrt': ['data/*.nc', 'data/*.tif', 'data/*.json']},
      install_requires=[
          'numpy',
          'scipy',
          'xarray',
          'rasterio',
          'netCDF4',
          'numba',
          'pandas',
          'affine'
      ],
      extras_require=extra_reqs)

