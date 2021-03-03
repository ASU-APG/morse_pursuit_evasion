# Online Prediction for Vision-based Active Pursuit using a Domain Agnostic Offline Motion Model 
<hr>
- OS: Ubuntu 16/18 <br/>
- Language: Python <br/>
- Dependencies: <br/>  

1) https://www.openrobots.org/morse/doc/stable/user/installation.html (MORSE Simulator) <br/>
2) Using anaconda/miniconda to install all dependencies: `conda env create -f env.yml` <br/>

### Using the code <hr>
- Each spawn file has `world_name, pursuer_spawn, evader_spawn, pursuer_ori, evader_ori, goal_id` columns. <br/>
`world_name` - World environment used for specific run <br/>
`pursuer_spawn, pursuer_ori` - Spawn location and orientation of the pursuer <br/>
`evader_spawn, evader_ori` - Spawn location and orientation of the evader <br/>
`goal_id` - Initial goal location of the evader <br/>
- In the root folder, `run_pursuer.py` change the `pursuer_type` variable to use different pursuers. <br/>
   `1-camera_only; 2-kalman; 3-lstm`
- In the root folder, `python run_simulation.py` will run with `max_runs` experiments.


Submitted to IROS-2021
