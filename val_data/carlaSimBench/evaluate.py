import torch


dir = '/home/moresweet/gitCloneZone/DMGTS/visualization/new/'

nbrs_pred = torch.load(dir+'all_nbr_predictions.pt', map_location=torch.device('cpu'))
batch_nbrs_size = torch.load(dir+'batch_nbrs_count.pt', map_location=torch.device('cpu'))
nbrs_start = torch.load(dir+'batch_nbrs_start.pt', map_location=torch.device('cpu'))

# Load trajectory data
past_traj = torch.load(dir+'past.pt', map_location=torch.device('cpu'))
fut_traj = torch.load(dir+'future.pt', map_location=torch.device('cpu'))
nbrs_traj = torch.load(dir+'nbrs.pt', map_location=torch.device('cpu'))
traj = torch.cat((past_traj, fut_traj), dim=1)
traj_mask = torch.load(dir+'traj_mask.pt', map_location=torch.device('cpu'))
nbrs_fut = torch.load(dir + 'nbrs_fut.pt', map_location=torch.device('cpu'))


# ego的原始点为(0,0)，他车的在nbrs_start中
print("over")

