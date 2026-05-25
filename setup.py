from setuptools import setup, find_packages

setup(
    name="cascade-pid",
    version="0.1.0",
    description="Cascade-PID: two-stage prompt injection detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
