from setuptools import setup, find_packages

setup(
    name='linselect',
    version='0.1.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A fast, flexible, and performant feature selection package for python.',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/EFavDB/linselect',
    download_url='https://github.com/efavdb/linselect/archive/0.1.1.tar.gz',
    author='Jonathan Landy',
    author_email='jslandy@gmail.com'
)
