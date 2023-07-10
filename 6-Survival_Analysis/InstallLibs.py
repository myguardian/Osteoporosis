import subprocess

# Check if pipreqs is installed
try:
    import pipreqs
except ImportError:
    subprocess.check_call(['pip', 'install', 'pipreqs'])

# Get a list of files to scan for dependencies from user input
file_list = input("Enter the file names (comma-separated) to scan for dependencies: ")
files_to_scan = [filename.strip() for filename in file_list.split(",")]

# Generate the requirements.txt file using pipreqs with --print option
subprocess.check_call(['pipreqs', '--force', '--print'] + files_to_scan)

# Read the requirements.txt file and extract the package names
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

# Install the packages
for package in requirements:
    subprocess.check_call(['pip', 'install', package])
