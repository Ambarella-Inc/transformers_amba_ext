from setuptools import setup, find_packages
from src import transformers_amba_ext

with open("README.md", "r", encoding="utf-8") as readme_file:
	readme = readme_file.read()

with open("requirements.txt") as reqs_file:
	requirements = reqs_file.read().split("\n")

setup(
	name="transformers_amba_ext",
	version=f"{transformers_amba_ext.__version__}.{transformers_amba_ext.__mod_time__}",
	author="Ambarella",
	author_email='UNKNOWN',
	description="A framework named transformers_amba_ext to run LLM on Ambarella's Chip",
	long_description=readme,
	long_description_content_type="text/markdown",
	license="Apache 2.0",
	keywords="transformers",
	package_dir={"": "src"},
	packages=find_packages("src"),
	install_requires=requirements,
	python_requires='>=3.8',
)
