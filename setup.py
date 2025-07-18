from setuptools import setup

setup(
   name='pypStag',
   version='1.0.0',
   author='Alexandre JANIN',
   author_email='alexandre.janin@protonmail.com',
   url='https://github.com/AlexandrePFJanin/pypStag',
   packages=['pypStag'],
   data_files=[('pypStag', ['pypStag/fields/stagyy-fields-defaults', 'pypStag/fields/stagyy-fields-local'])],     # to include data samples
   include_package_data=True,   # to include data samples
   license='LICENSE.md',
   description='A python package for post-processing and analysing StagYY outputs.',
   long_description=open('README.md').read(),
   install_requires=[
        'ipython>=8.15.0',
        'numpy>=1.12',
	    'scipy>=1.5.2',
	    'h5py>=3.9.0',
	    'termcolor'
   ],
)
