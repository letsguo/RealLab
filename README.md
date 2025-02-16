## Installation
1. Use the instructions in the mushr repo to install and build `mushr_base` and all of its dependancies. [https://github.com/prl-mushr/mushr](https://github.com/prl-mushr/mushr)
2. Clone this repo into the src directory of your catkin workspace.
3. If you would like to run a policy that uses mocap, clone the odom branch of the mushr mocap repo with `git clone https://github.com/prl-mushr/mushr_mocap.git -b odom`. Then follow the [installation instructions](https://github.com/prl-mushr/mushr_mocap/tree/odom?tab=readme-ov-file#mushr-mocap) in mushr_mocap to get mocap set up.
4. Run `catkin_make` to build all packages
##

## Running the Vehicle
1. Upload your model weights into the `src/models` folder. For more information about training models refer to [https://github.com/UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab)
2. Go to `config/policies/<POLICY_TO_RUN>.yaml` and configure the model parameters
3. Use the following command to launch the policy:
```bash
roslaunch hound_core offroad_irl.launch policy:=<POLICY_TO_RUN> robot_name:=<ROBOT_NAME>
```
4. Arm the mushr (typically by pressing the right trigger)
5. To log data run the launch command with the `data:=True` arguement

## References

### This work

```
@misc{2502.07380,
Author = {Tyler Han and Preet Shah and Sidharth Rajagopal and Yanda Bao and Sanghun Jung and Sidharth Talia and Gabriel Guo and Bryan Xu and Bhaumik Mehta and Emma Romig and Rosario Scalise and Byron Boots},
Title = {Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics},
Year = {2025},
Eprint = {arXiv:2502.07380},
}
```

### Mushr

[1] Siddhartha S. Srinivasa, Patrick Lancaster, Johan Michalove, Matt Schmittle, Colin Summers, Matthew Rockett, Rosario Scalise, Joshua R. Smith, Sanjiban Choudhury, Christoforos Mavrogiannis, and Fereshteh Sadeghi.MuSHR: A Low-Cost, Open-Source Robotic Racecar for Education and Research, December 2023.URL http://arxiv.org/abs/1908.08031.arXiv:1908.08031 [cs].