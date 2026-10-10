from setuptools import find_packages,setup
from typing import List

HYPEN_E_COT="-e ."


def get_requirements(file_path:str)->List[str]:
    requirements={}
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("/n","") for req in requirements]

        if HYPEN_E_COT in requirements:
            requirements.remove(HYPEN_E_COT)

    return requirements
        




setup(
name = "ml project",
version= "0.0.1",
author="Mustafa Sherzad",
author_email="sherzadmustafa1111@gmail.com",
packages= find_packages(),
install_requires=get_requirements("requirements.txt")





)