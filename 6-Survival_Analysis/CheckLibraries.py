import subprocess
import pkg_resources

# Check if pipreqs is installed
try:
    import pipreqs
except ImportError:
    subprocess.check_call(['pip', 'install', 'pipreqs'])

file_list = input("Enter the file names (space-separated) to scan for dependencies: ")
files_to_scan = [filename.strip() for filename in file_list.split()]

subprocess.check_call(['pipreqs', '--force', '--print'] + files_to_scan)

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

installed_packages = [pkg.key for pkg in pkg_resources.working_set]

packages_to_install = [package for package in requirements if package not in installed_packages]

for package in packages_to_install:
    subprocess.check_call(['pip', 'install', package])
