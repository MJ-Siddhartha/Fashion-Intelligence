import os
import shutil
import zipfile
from datetime import datetime

def create_submission_package():
    """
    Create comprehensive submission package
    """
    # Timestamp for unique package name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"universal_fashion_feature_extraction_{timestamp}"
    
    # Create temporary directory
    os.makedirs(package_name, exist_ok=True)
    
    # Copy project structure
    project_dirs = [
        'src', 'tests', 'docs', 'data', 'models', 
        'reports', 'scripts'
    ]
    
    for directory in project_dirs:
        shutil.copytree(directory, os.path.join(package_name, directory))
    
    # Copy key files
    key_files = [
        'requirements.txt', 'setup.py', 'README.md', 
        'run.py', 'LICENSE'
    ]
    
    for file in key_files:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(package_name, file))
    
    # Generate documentation
    os.system(f'sphinx-quickstart -q -p "Universal Fashion Feature Extraction" -a "Your Name" -v 0.1.0 -r 0.1.0 -l en --ext-autodoc --ext-viewcode {package_name}/docs')
    
    # Create PDF documentation
    os.system(f'sphinx-build -b pdf {package_name}/docs {package_name}/docs/_build/pdf')
    
    # Create PPT presentation
    # (Note: This would typically require manual creation or a dedicated tool)
    
    # Run tests and generate reports
    os.system(f'pytest {package_name}/tests > {package_name}/reports/test_results.txt')
    
    # Create zip package
    shutil.make_archive(package_name, 'zip', package_name)
    
    print(f"Submission package created: {package_name}.zip")

if __name__ == '__main__':
    create_submission_package()
