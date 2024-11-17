
## Listing Packages
1. List all installed packages:
```
conda list
```
2. List installed packages matching a regular expression:
Example: List packages with names starting with z:
```
conda list ^z
```

3. List all versions of all packages (all channels):
```
conda search
```
4. List all versions of a specific package (all channels):
 Example: Search for scikit-learn:
```
conda search scikit-learn
```
5. List specific versions of a package (all channels): Example: Find versions of scikit-learn greater than or equal to 1:
```
conda search 'scikit-learn>=1'
```
6. List package versions for a specific channel:
 Example: Search for scikit-learn in the conda-forge channel:
```
conda search conda-forge::scikit-learn
```

## Installing & Managing Packages
1. Install packages:
Example: Install numpy and pandas:
```
conda install numpy pandas
```
2. Install a specific version of a package:
Example: Install version 1.10.1 of scipy:
```
conda install scipy=1.10.1
```

3. Update all packages:
```
conda update --all
```

4. Uninstall a package:
Example: Uninstall pycaret:
```
conda uninstall pycaret
```

## Working with Channels

1. List available channels:
```
conda config --show channels
```

2. Add a channel with the highest priority:
Example: Add conda-forge as a high-priority channel:
```
conda config --prepend channels conda-forge
```
3. Add a channel with the lowest priority:
Example: Add bioconda as a low-priority channel:
```
conda config --append channels bioconda
```


## Working with Environments
1. List all available environments:
```
conda env list
```
2. Restrict command scope to a specific environment:
Use the --name {envname} option to target a specific environment.
* Example: List packages in the base environment:
```
conda list --name base
```

* Example: Install scikit-learn in the myenv environment:
```
conda install scikit-learn --name myenv
```


## Managing Environments
1. Create a new environment:
Example: Create an environment named my_python_env:
```
conda create --name my_python_env
```
2. Create new environment name {.env} in same folder
```
conda create --prefix ./env python==version -y
```

3. Clone an existing environment:
Example: Clone template_env into a new environment named project_env:
```
conda create --clone template_env --name project_env
```

4. Create an environment and auto-accept prompts:
For non-interactive setups:
```
conda create --yes --name my_env
```

5. Activate an environment:
Set an environment as the default for your current session:
```
conda activate my_env
```

6. Deactivate the current environment:
Revert back to the base environment or unset the current environment:
```
conda deactivate
```


## Sharing Environments
1. Export an active environment to a YAML file:
For maximum reproducibility, including all dependencies:
```
conda env export > environment.yml
```

2. Export only explicitly installed packages:
For better portability:
```
conda env export --from-history > environment.yml
```

3. Import an environment from a YAML file:
Example: Create an environment from environment.yml:
```
conda create --name my_env --file environment.yml
```

4. Export a list of installed packages to a plain text file:
Example: Export to requirements.txt (suitable for manual editing or use with pip):
```
conda list --export > requirements.txt
```

5. Import an environment from a plain text file:
Example: Create an environment using requirements.txt:
```
conda create --name my_env --file requirements.txt
```




