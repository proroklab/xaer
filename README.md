# XARL
Repo for Explanation-Aware Reinforcement Learning Agents

## Installing CARLA on Ubuntu 20.04
If you have a desktop/machine with a dedicated graphics
card, you may install the vanilla version of [CARLA 0.9.9.](https://github.com/carla-simulator/carla/releases/tag/0.9.9)

This bundled version comes with all the required Unreal Engine libraries
and does not require building from source.

### Running the Server
1. Extract the `CARLA_0.9.9.tar.gz` file to `~/carla`.
2. Update the assets with `./ImportAssets.sh`.
3. Run the server with `./CARLAUE4.sh`.
4. Graphics options can be: `-quality-level=Epic` or `-quality-level=Low`.

The server needs to be running before we make calls to the Python API.
This API will create a detached game window running a client.

### Running a Python API Example

1. Install Python 3.7. It is possible to run it with Python 3.8 if you adjust
 the pygame requirement to `pygame==2.0.0.dev6`.
2. Run `python3 -m pip install pygame numpy future` to install dependencies.
3. Install the CARLA egg file in `~/carla/PythonAPI/carla/dist` with 
`python3 -m easy_install carla-0.9.9-py3.7-linux-x86_64.egg`.
4. Go to `~/carla/PythonAPI/examples` and try `python3 automatic_control.py` to see
a self-driving example.

### Disabling Vulkan (Integrated Graphics)

If your machine does not have dedicated drivers, it is likely that it will not support Vulkan.
In my case, running the vanilla version resulted in the machine freezing completely, requiring a power cycle.
It is possible to make CARLA work by disabling Vulkan and enabling OpenGL rendering.
Contrary to what is stated in the CARLA docs, the `-opengl` flag does not work,
so we will need to amend files manually.

1. Edit file `~/carla/Engine/Config/BaseEngine.ini`.
2. Comment out line 2153: `;+TargetedRHIs=SF_VULKAN_SM5`.
3. Uncomment line 2156: `+TargetedRHIs=GLSL_430`.
4. Edit file `~/carla/CarlaUE4/Config/DefaultEngine.ini`.
5. Comment out lines 51 and 53: `;+TargetedRHIs=SF_VULKAN_SM5`.
6. Run `~/carla/CarlaUE4.sh -quality-level=Low` to see if it works.
7. To increase frame rate, reduce the size of the server window as much as possible.

