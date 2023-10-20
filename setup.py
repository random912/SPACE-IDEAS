#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()


requirements = [s.strip() for s in open('requirements.txt', 'r').readlines()]
test_requirements = []


setup(
    name='ideas_annotation',
    version='0.1.0',
    description="Research project for...",
    long_description=readme,
    author="",
    author_email='',
    url='https://github.com/random912/SPACE-IDEAS',
    python_requires='>=3.5',
    packages=find_packages(include=[
        'ideas_annotation', 'ideas_annotation.*'
    ]),
    package_dir={'ideas_annotation':
                 'ideas_annotation'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='ideas_annotation',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
