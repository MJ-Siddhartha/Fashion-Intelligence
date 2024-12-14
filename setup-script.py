from setuptools import setup, find_packages

setup(
    name='universal-fashion-feature-extraction',
    version='0.1.0',
    description='AI-driven Universal Fashion Feature Extraction System',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Will be dynamically populated from requirements.txt
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'fashion-feature-extraction=run:main',
        ],
    },
)
