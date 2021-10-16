echo 'Create a virtual environment'
virtualenv .env -p python3
echo 'Activate the virtual environment'
source .env/bin/activate
# echo 'Update the virtual environment'
# pip install -U pip setuptools wheel
echo "Installing other environments' dependencies.."
pip install -r environments/requirements.txt
# echo 'Fixing environments rendering'
# pip install pyglet==1.5.11 # fix for rendering environments
echo 'Installing XARL..'
pip install -e ./package # cmake is needed

echo 'Installing Atari ROMs..'
mkdir Atari-ROM
cd Atari-ROM
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
unzip ROMS.zip
python -m atari_py.import_roms ROMS
cd ..
