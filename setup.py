from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nfl-playoff-predictor",
    version="0.1.0",
    author="Eli Spiller, Zion Tippetts",
    author_email="",
    description="A Python package for predicting NFL postseason wins using advanced passing statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/espiller602/stat-386-final-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.xls", "*.xlsx"],
        "nfl_playoff_predictor": ["streamlit_app.py"],
    },
    entry_points={
        "console_scripts": [
            "nfl-playoff-predictor-app=nfl_playoff_predictor.app:launch_app",
        ],
    },
)

