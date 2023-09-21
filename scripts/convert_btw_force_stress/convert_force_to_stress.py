
import argparse
import torch
import yaml
import os

def convert(ckpt_path,out_path):
    
    ckpt_force=torch.load(os.path.join(ckpt_path,'checkpoint.pt'),'cpu')
    ckpt_force['config']['model_attributes']['regress_stress']=True
    ckpt_force['config']['model_attributes']['model_builders'][-2]='StressForceOutput'
    ckpt_force['config']['task']['metrics'] +=['stress_mae','stress_mse']
    for k in ckpt_force['state_dict'].keys():

        if(k.startswith("module.module")):
            ckpt_force['state_dict'][".".join(k.split(".")[:3])+'.model._empty']=torch.tensor([])
        else:
            ckpt_force['state_dict'][".".join(k.split(".")[:2])+'.model._empty']=torch.tensor([])
        break
    
    with open (os.path.join(ckpt_path,'config_train.yml'),'r')as f:
        config=yaml.load(f,yaml.FullLoader)
    config['model']['regress_stress']=True
    config['model']['model_builders'][-2]='StressForceOutput'
    config['task']['metrics'] +=['stress_mae','stress_mse']
    config['use_stress']=True
    os.makedirs(out_path,exist_ok=True)

    torch.save(ckpt_force,os.path.join(out_path,"checkpoint.pt"))
    with open (os.path.join(out_path,'config_train.yml'),'w')as f:
        yaml.dump(config,f)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Converting ForceNequip to StressForceNequip")
    parser.add_argument(
            "--ckpt-dir",
            type=str,
            default=None,
            help="Path of trained checkpoint folder",
        )
    parser.add_argument(
            "--out-dir",
            type=str,
            default=None,
            help="Path of converted output checkpoint folder",
        )
    args=parser.parse_args()
    convert(args.ckpt_dir,args.out_dir)