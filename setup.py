from setuptools import setup, find_packages

setup(
    name="electrical_test_analyzer",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib"
    ],
    author="Sunil Mair",
    author_email="smair@mit.edu",
    description="Tools for analyzing electrical test data.",
    url="https://github.com/yourusername/mypackage",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)