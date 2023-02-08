from setuptools import setup

setup(
   name='pypStag',
   version='0.5.0',
   author='Alexandre JANIN',
   author_email='alexandre.janin@protonmail.com',
   packages=['pypStag'],
   url='https://github.com/AlexandrePFJanin/pypStag',
   license='LICENSE.txt',
   description='pypStag is a python package for managing StagYY outputs. This package offers you an efficient solution for basic reading/processing operations and is associated with a complete visualization package for 2D and 3D representation of your StagYY data.',
   long_description=open('README.rst').read(),
   install_requires=[
        'numpy>=1.12',
        'scipy>=1.0',
        'h5py>=2.7.1',
        'matplotlib>=3.0',
	'Cartopy>=0.18',
	'scipy>=1.5.2'
   ],
)
