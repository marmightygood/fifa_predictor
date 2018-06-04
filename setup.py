from setuptools import setup, find_packages

setup (
       name='russia_fifa_wc',
       version='0.1',
       packages=find_packages(),

       # Declare your packages' dependencies here, for eg:
       install_requires=['pandas', 'keras'],

       # Fill in these to make your Egg ready for upload to
       # PyPI
       author='ollie',
       author_email='oliver@jonette.co.nz',

       #summary = 'Just another Python package for the cheese shop',
       url='',
       license='',
       long_description='Predicts the FIFA world cup 2018',

       # could also include long_description, download_url, classifiers, etc.

  
       )