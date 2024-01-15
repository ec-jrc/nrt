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
              'docs': ['sphinx',
                       'dask',
                       'sphinx_rtd_theme',
                       'matplotlib',
                       'sphinx-gallery']}

setup(name='nrt',
      version=version,
      description=u"Online monitoring with xarray",
      long_description_content_type="text/x-rst",
      long_description=readme,
      keywords='sentinel2, xarray, datacube, monitoring, change',
      author=u"Loic Dutrieux, Jonas Viehweger, Chris Holden",
      author_email='loic.dutrieux@ec.europa.eu',
      url='https://github.com/ec-jrc/nrt.git',
      license='EUPL-v1.2',
      classifiers=[
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
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
      python_requires=">=3.9",
      extras_require=extra_reqs)

