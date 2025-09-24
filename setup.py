import setuptools

# python setup.py develop --user
setuptools.setup(
    name='ucasr',
    version='0.1dev',
    description='Unsupervised contrastive audio-sheet music retriever',
    packages=setuptools.find_packages(),
    author='Luis Carvalho',
)
