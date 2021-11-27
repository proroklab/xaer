MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR


# if [ ! -d ".env" ]; then
# 	sh $MY_DIR/setup_python_env.sh 3.8 6 $MY_DIR .env
# fi
# . .env/bin/activate	

virtualenv .env -p python3
source .env/bin/activate

# echo 'Create a virtual environment'
# python3 -m venv .env
# echo 'Activate the virtual environment'
# source .env/bin/activate
# echo 'Update the virtual environment'
# pip install -U pip setuptools wheel
echo "Installing other environments' dependencies.."
pip install -r environments/requirements.txt
# echo 'Fixing environments rendering'
# pip install pyglet==1.5.11 # fix for rendering environments
echo 'Installing XARL..'
pip install -e ./package # cmake is needed
# pip install ray[rllib]==1.2.0 aioredis==1.3.1
