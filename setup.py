from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

setup(
    name="llm_doctor",
    packages=find_packages(),
    install_requires=requirements,
)