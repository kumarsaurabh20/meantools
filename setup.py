from setuptools import setup, find_packages

setup(
    name='meantools',
    version='1.1.4',
    description='Integration of multi-omics data for biosynthetic pathway discovery',
    long_description=open('README.rst').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kumarsaurabh20/meantools',
    author='Kumar Saurabh Singh',
    author_email='kumar.singh@wur.nl',
    license='MIT',
    packages=['data'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'rdkit', 'networkx', 'pandas', 'tqdm', 'svgutils', 'svgelements', 'scipy', 'markov_clustering', 'seaborn'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    
    include_package_data=True,
    package_data={'data': ['data/*.csv', 'data/chem_prop.tsv', 'data/lotus_v101023.sqlite', 'data/mvc.db', 'cluster_one-1.0.jar']},
    python_requires='>=3.8',
    scripts=['corrMultiomics_mod.py', 'format_databases.py', 'gizmos.py', 'heraldPathways_mod.py', 'mutual_ranks.py', 'pathMassTransitions_mod.py', 'paveWays.py', 'queryMassNPDB_mod.py', 'clusters.py', 'graphics.py']
)

