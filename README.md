# AI agent for source detection with SoFiA-2  

## Dependence

```Python
numpy
astropy
matplotlib
pandas
joblib
torch
gymnasium
stable-baselines3
```
and  
[SoFiA-2](https://gitlab.com/SoFiA-Admin/SoFiA-2)

## Files

`SAC` includes the following scripts:  
- `sofia_env_multiple_regions.py`: The environment for SoFiA-2.    
- `train_agent_multiple_regions.py`: Main training script.  
- `evaluate_models.py`: Evaluate agent's models.  

`scripts` includes utils for analysis and plotting:  
- `ska_sdc`: The scripts used for calculating the SDC2 scores. Modified from original scripts [ska-sdc](https://gitlab.com/ska-telescope/sdc/ska-sdc).  
- `physical_parameter_conversion.py`: Convert source properties from catalog to correspond to the ones in truth catalog.  Modified from original scripts by Team SoFiA in [here](https://gitlab.com/SoFiA-Admin/SKA-SDC2-SoFiA/-/blob/master/scripts/physical_parameter_conversion.py?ref_type=heads).  
- `utils.py`: Utility functions for analysis and plotting.  

`dataset` includes relevant datasets:  
- `development`: Development sky map (0.25 square degrees) and truth catalog from SKA-SDC2 (replace the placeholders with actual files).  
- `development_large`: Large development sky map (1 square degrees) and truth catalog from SKA-SDC2 (replace the placeholders with actual files).  
- `temp_par_file.par`: Template par file for SoFiA-2.  

## Run

Train the agent:  
```bash
python train_agent_multiple_regions.py
```

Evaluate saved models:  
```bash
python evaluate_models.py
```