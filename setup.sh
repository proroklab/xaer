# echo 'Create a virtual environment'
# python3 -m venv .env
# echo 'Activate the virtual environment'
# source .env/bin/activate
# echo 'Update the virtual environment'
# pip install -U pip setuptools wheel
echo "Install other environments' dependencies"
pip install -r environments/requirements.txt
echo 'Install XARL'
pip install -e ./package # cmake is needed
# echo 'Fixing environments rendering'
# pip install pyglet==1.5.11 # fix for rendering environments