from setuptools import setup, find_packages

setup(
        name='dpfa',
        packages=find_packages('src'),
        package_dir={'': 'src'})