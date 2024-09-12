from setuptools import setup, find_packages
from typing import List
import os

def requirements() -> List[str]:
    """
    Reads the requirements.txt file and returns a list of dependencies.

    :return: List of dependencies.
    """
    requirements_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as requirements_file:
            requirements_list = requirements_file.read().splitlines()
            # Optionally remove empty lines or comments
            requirements_list = [req for req in requirements_list if req and not req.startswith('#')]
    except FileNotFoundError:
        print("requirements.txt not found. Please ensure it exists in the root directory.")
    except Exception as e:
        print(f"An error occurred while reading requirements.txt: {e}")
    return requirements_list

# Reading the long description from README.md
def read_long_description() -> str:
    """
    Reads the README.md file to use as the long description.

    :return: Long description as a string.
    """
    try:
        with open('README.md', 'r', encoding='utf-8') as readme_file:
            return readme_file.read()
    except FileNotFoundError:
        return "Long description not available. Please see the GitHub repository."
    except Exception as e:
        print(f"An error occurred while reading README.md: {e}")
        return "Long description not available. Please see the GitHub repository."

setup(
    name='PropBots',
    version='0.0.1',
    author='Mohit Verma',
    author_email='mohitvvermaa@outlook.com',
    description='A Generative AI project for Property Bots',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',  # This ensures the README is interpreted as Markdown
    url='https://github.com/mohitvverma/PropBot',
    packages=find_packages(),
    install_requires=requirements(),
    extras_require={
        'dev': ['pytest', 'flake8', 'black'],
        'gpu': ['tensorflow-gpu>=2.5.0'],  # Example of optional GPU support
        'docs': ['sphinx', 'sphinx_rtd_theme'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
    keywords='generative-ai chatbot NLP',
    python_requires='>=3.7',  # Specify the Python version compatibility
    license='MIT',  # Specify the license
    include_package_data=True,  # Includes non-Python files specified in MANIFEST.in
    zip_safe=False,  # To allow the package to be unpacked and run directly

)
