import torch

#creates a small test dataset with two users, each having 4 different locations. 
#formats data into sequences where model sees one location and learns to predict the next location
users = [0, 1]  
times = [[1000000, 1000020, 1000040, 1000060], [2000000, 2000030, 2000060, 2000090]]  
time_slots = [[10, 11, 12, 13], [20, 21, 22, 23]]  
coords = [[(40.7128, -74.0060), (40.7130, -74.0070), (40.7140, -74.0080), (40.7150, -74.0090)],  
          [(34.0522, -118.2437), (34.0525, -118.2440), (34.0528, -118.2445), (34.0531, -118.2450)]]  
locs = [[101, 102, 103, 104], [201, 202, 203, 204]] 

#intialize test data
dataset = PoiDataset(
    users=users,
    times=times,
    time_slots=time_slots,
    coords=coords,
    locs=locs,
    sequence_length=1,  
    batch_size=1,  # process one user at a time
    split=Split.TRAIN,  
    usage=Usage.MAX_SEQ_LENGTH,
    loc_count=300,  # total number of unique locations
    custom_seq_count=1
)

print("Total Users:", len(dataset.users))
print("Total Sequences:", len(dataset.sequences))

x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users = dataset.__getitem__(0)

print("Input Sequences (x):", x)
print("Timestamps (t):", t)
print("Time Slots (t_slot):", t_slot)
print("Coordinates (s):", s)
print("Labels (y - next locations):", y)
print("Active Users:", active_users)
