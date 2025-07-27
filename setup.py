from setuptools import setup

setup(
    name='pypStag',
    version='1.0.1',
    author='Alexandre JANIN',
    author_email='alexandre.janin@protonmail.com',
    url='https://github.com/AlexandrePFJanin/pypStag',
    description='A python package for post-processing and analysing StagYY outputs.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='LICENSE.md',
    packages=['pypStag'],
    include_package_data=True,
    package_data={
        'pypStag': ['fields/*'],
    },
    install_requires=[
        'ipython>=8.15.0',
        'numpy>=1.12',
        'scipy>=1.5.2',
        'h5py>=3.9.0',
        'termcolor',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
