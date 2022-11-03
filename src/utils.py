import yaml
import sys

with open("requirements.yml") as file_handle:
    environment_data = yaml.load(file_handle, Loader=yaml.FullLoader)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        print(dependency, dependency.split("=")[0:-1])
        package_name, package_version = dependency.split("=")[0:-1]
        file_handle.write("{} == {}\n".format(package_name, package_version))
