from setuptools import setup, find_packages

setup(
    name='vxs',
    version='0.1.0',
    description='Vocal percussion transcription system',
    author='Dmitrii Mukhutdinov',
    author_email='flyingleafe@gmail.com',
    packages=find_packages(),
    install_requires=(
        'numpy==1.18.*',
        'pandas==1.0.*',
        'torch==1.4.*',
        'aubio==0.4.*'
    )
)
