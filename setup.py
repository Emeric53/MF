from setuptools import setup, find_packages

setup(
    name="matched_filter_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
