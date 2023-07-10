import subprocess
import pkg_resources

# Check if pipreqs is installed
try:
    import pipreqs
except ImportError:
    subprocess.check_call(['pip', 'install', 'pipreqs'])

# Define the command to execute
command = ['python', '-m', 'pipreqs.pipreqs', '--force']

# Execute the command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the process to finish and get the output
stdout, stderr = process.communicate()

# Check the return code
if process.returncode == 0:
    print("pipreqs command executed successfully.")
else:
    print("An error occurred while executing pipreqs command.")
    print("Error message:", stderr.decode('utf-8'))

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

installed_packages = [pkg.key for pkg in pkg_resources.working_set]

packages_to_install = [package for package in requirements if package not in installed_packages]

for package in packages_to_install:
    subprocess.check_call(['pip', 'install', package])
