import datetime
import json
import os
import time

import torch
import torch.distributed as dist
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

import utils.eval as eval
from utils.data_loader_multifiles import get_data_loader

from networks.PanguLite import PanguModel as PanguModelLite
from networks.PanguLite2DAttention import PanguModel as PanguLite2D

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_distributed(params):
    """
    Initialize DistributedDataParallel or set to CPU.
    
    params: Dict
        dictionary specifying run parameters
    
    Returns
    -------
    device: String
    slurm_localid: int
    gpus_per_node: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    #gpus_per_node = torch.cuda.device_count()


    # Initialize GPUs and dataloaders
    if slurm_job_gpus is not None:
        #gpu = rank % gpus_per_node
        #assert gpu == slurm_localid
        device = f"cuda:{slurm_localid}"
        torch.cuda.set_device(device)
        # Initialize DistributedDataParallel.
        if params['data_distributed']:
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    else:
        # Initialize DistributedDataParallel.
        if params['data_distributed']:
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method="env://")
        # set device to CPU
        device = 'cpu'

    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    else:
        print("Running in serial")

    return device, slurm_localid, rank, world_size

def training_loop(params):
    """
    train the model.

    params: Dict
    device: String
    slurm_localid: int
    """
    # Set seeds for reproducability
    set_all_seeds(1)
    
    device, slurm_localid, rank, world_size = init_distributed(params)

    
    # Define patch size, data loader, model
    dim          = params['C']
    two_dimensional = False
    if params['model'] == '2D':
        two_dimensional = True
    
    train_data_loader = get_data_loader(params, params['train_data_path'], dist.is_initialized(), mode='train', patch_size=params['patch_size'], subset_size=params['subset_size'], two_dimensional=two_dimensional)
   
    valid_data_loader = get_data_loader(params, params['valid_data_path'], dist.is_initialized(), mode='validation', patch_size=params['patch_size'], subset_size=params['validation_subset_size'], two_dimensional=two_dimensional)

    if params['model'] == 'panguLite':
        model = PanguModelLite(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == '2D':
        model = PanguLite2D(device=device, dim=int(1.5*dim), patch_size=params['patch_size'][1:])
    else: 
        raise NotImplementedError(params['model'] + ' currently not supported.')

    model = model.to(torch.float32).to(device)

    # DistributedDataParallel wrapper if GPUs are available
    if dist.is_initialized():
        model = DistributedDataParallel( # Wrap model with DistributedDataParallel.
                model, 
                device_ids=[slurm_localid], 
                output_device=slurm_localid,
                find_unused_parameters=False
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)
    scheduler = CosineLRScheduler(optimizer, t_initial=params['num_epochs'], warmup_t=5, warmup_lr_init=1e-5)


    start_epoch = 0

    # Training loop
    train_loss_history = []
    valid_loss_history = []
    best_val_loss = 100000           # Arbitrary large number

    loss1 = torch.nn.L1Loss()
    loss2 = torch.nn.L1Loss()

    save_counter = 0

    # Z, Q, T, U, V
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).view(1, 5, 1, 1, 1).to(device)
    surface_weights  = torch.tensor([1.5, 0.77, 0.66, 3.0]).view(1, 4, 1, 1).to(device)

    scaler = GradScaler()

    early_stopping = 0 # tracks how many epochs have passed since the validation loss has improved

    if rank == 0:
        print("Model will be saved in", params['save_dir'][params['model']])
        # dump parameters into json directory
        with open(params['save_dir'][params['model']] + "params" + '.json', 'w') as params_file:
            json.dump(params, params_file)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if rank == 0 and epoch == start_epoch:
            lr = optimizer.param_groups[0]['lr'] # write to variable so that we can print...
            print("learning rate:", lr)
        start_epoch_time = time.perf_counter()
        model.train()
        
        train_data_loader.sampler.set_epoch(epoch)
        epoch_average_loss = 0
        loss = 0
        for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
            input, input_surface, target, target_surface = data[0], data[1], data[2][0], data[3][0]
            input = input.to(torch.float32).to(device)
            input_surface = input_surface.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            target_surface = target_surface.to(torch.float32).to(device)

            with torch.autocast(device_type="cuda"):
                optimizer.zero_grad()
                # Call the model and get the output
                output, output_surface = model(input, input_surface)
                if params['model'] == '2D':
                    output = output.reshape(-1, 5, 13, output.shape[-2], output.shape[-1])
                    target = target.reshape(-1, 5, 13, target.shape[-2], target.shape[-1])

                # We use the MAE loss to train the model
                # The weight of surface loss is 0.25
                # Different weights can be applied for different fields if needed
                output, output_surface = output * pressure_weights, output_surface * surface_weights
                target, target_surface = target * pressure_weights, target_surface * surface_weights
                loss = 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)

            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            # Call the backward algorithm and calculate the gradient of parameters
            scaler.step(optimizer)
            scaler.update()
            
            dist.all_reduce(loss) # Allreduce rank-local mini-batch losses.
            
            if rank == 0:
                loss /= world_size    # Average allreduced rank-local mini-batch losses over all ranks.
                train_loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.
                epoch_average_loss += loss.item()
            
                print(f'Epoch: {epoch+1:03d}/{int(start_epoch+num_epochs):03d} '
                    f'| Batch {i:04d}/{len(train_data_loader):04d} '
                    f'| Averaged Loss: {loss:.4f}')
        
        epoch_average_loss /= len(train_data_loader)
        
        end_epoch_time = time.perf_counter()

        if rank == 0:
            print(f'Epoch: {int(epoch+1):03d}/{int(start_epoch+num_epochs):03d} '
                f'Elapsed training time: {end_epoch_time - start_epoch_time:04f}')
        
        model.eval()
        with torch.no_grad():
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            val_loss, mse, acc, total_samples, dt_validation = eval.get_loss(model, valid_data_loader, device, loss1, loss2, lat_crop=params['lat_crop'], lon_crop=params['lon_crop'], world_size=world_size)

            if rank == 0:
                valid_loss_history.append(val_loss[0])
                if (val_loss[0] < best_val_loss):
                    early_stopping = 0
                    best_val_loss = val_loss[0]
                    # save model to path
                    save_path = params['save_dir'][params['model']] + str(save_counter) + "_" + params['model'] + '.pt'
                    state = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, save_path)
                    save_counter += 1
                else:
                    early_stopping += 1
                
                lr = optimizer.param_groups[0]['lr'] # write to variable so that we can print...

                print(f'Epoch Average: {int(epoch)+1:03d}/{int(num_epochs):03d} ')
                print(f'| Mean L1 Training Loss: {epoch_average_loss :.5f} ')
                print(f'| Mean L1 Validation Loss: {val_loss[0] :.5f} ')
                print(f'| MSE T850: {mse[0][0] :.3f} ')
                print(f'| MSE Z500: {mse[1][0] :.3f} ')
                print(f'| MSE U850: {mse[2][0] :.3f} ')
                print(f'| MSE V850: {mse[3][0] :.3f} ')
                print(f'| MSE Q850: {mse[4][0]*1000 :.3f} ')
                print(f'| MSE T2M:  {mse[5][0] :.3f} ')
                print(f'| MSE U10:  {mse[6][0] :.3f} ')
                print(f'| MSE V10:  {mse[7][0] :.3f} ')
                print(f'| MSE MSL:  {mse[8][0] :.3f} ')
                print(f'| ACC T850: {acc[0][0] :.3f} ')
                print(f'| ACC Z500: {acc[1][0] :.3f} ')
                print(f'| ACC U850: {acc[2][0] :.3f} ')
                print(f'| ACC V850: {acc[3][0] :.3f} ')
                print(f'| ACC Q850: {acc[4][0] :.3f} ')
                print(f'| ACC T2M:  {acc[5][0] :.3f} ')
                print(f'| ACC U10:  {acc[6][0] :.3f} ')
                print(f'| ACC V10:  {acc[7][0] :.3f} ')
                print(f'| ACC MSL:  {acc[8][0] :.3f} ')
                print(f'| Validation time: {dt_validation : .5f}')
                print(f'| Total samples: {total_samples :.3f}')
                print(f'| Learning Rate: {lr :.7f}')

            scheduler.step(epoch+1)
                    
    # How can we verify that at least one model will be saved? Currently only saves when in case of the best validation loss
    dist.destroy_process_group()

if __name__ == '__main__':
    params = {}
    params['train_data_path'] =  '/lsdf/kit/scc/projects/SmartWeater21/era_subset.zarr'#/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr' # CHANGE TO YOUR DATA DIRECTORY
    params['valid_data_path'] =  '/lsdf/kit/scc/projects/SmartWeater21/era_subset.zarr'#/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr' # CHANGE TO YOUR DATA DIRECTORY
    params['pressure_static_data_path'] = '/lsdf/kit/scc/projects/SmartWeater21/constant_masks/pressure_zarr.npy' 
    params['surface_static_data_path'] =  '/lsdf/kit/scc/projects/SmartWeater21/constant_masks/surface_zarr.npy'  
    
    params['dt'] = 24
    params['num_data_workers'] = 2  # pyTorch parameter
    params['data_distributed'] = True
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['num_epochs'] = 5
    num_epochs = params['num_epochs']
    params['C'] = 192
    params['subset_size'] = 8
    params['validation_subset_size'] = 6
    params['Lite'] = True
    params['daily'] = False

    
    
    params['model'] = 'panguLite' # Specify model: panguLite = light model; 2D = 2D transformer
    base_save_dir = 'trained_models/' # Save directory
    params['save_dir'] = {
        'panguLite':    base_save_dir + 'panguLite/' ,
        '2D':           base_save_dir + 'twoDimensionalLite/' ,
    }
            
        
    # initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite.
    # patch size is (2, 4, 4) for all other sizes.
    params['patch_size'] = (2, 8, 8)  
    params['batch_size'] = 1
    params['lat_crop']   = (3, 4) # Do not change if input image size of (721, 1440)
    params['lon_crop']   = (0, 0) # Do not change if input image size of (721, 1440)
    params['delta_T_divisor'] = 6 # Baseline assumption is 6-hourly subsampled data

    training_loop(params)
    

    
