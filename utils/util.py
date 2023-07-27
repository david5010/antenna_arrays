import torch
import matplotlib.pyplot as plt

def show_arrays(inp):
    data = inp.cpu().detach().numpy()
    reshaped_data = np.reshape(data, (2,1024))
    x = reshaped_data[0]
    y = reshaped_data[1]

    nonzero_ind = np.nonzero((x != 0) | (y != 0))
    x = x[nonzero_ind]
    y = y[nonzero_ind]

    plt.scatter(x,y, s=0.75)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Antenna Array Pattern')
    plt.show()

def calculate_min_distance(ants, unit = 1):
    """
    Checks the smallest Euclidean distance, if it violates then return True
    """
    reshaped_tensor = torch.reshape(ants, (2, 1024))

# Transpose the reshaped tensor
    reshaped_tensor = torch.transpose(reshaped_tensor, 0, 1)
    nonzero_rows = torch.any(reshaped_tensor != 0, dim=1)
    filtered_tensor = reshaped_tensor[nonzero_rows]
    distances = torch.cdist(filtered_tensor, filtered_tensor)
    distances.fill_diagonal_(float('inf'))
    closest_neighbor_distances, _ = torch.min(distances, dim=1)
    smallest = torch.min(closest_neighbor_distances)
    return smallest.item()
    

def gradient_wrt_input(model, ants, lr, noise_std, iters):
    model.eval()
    last = None
    for i in range(iters):
        if ants.grad is not None:
            ants.requires_grad_()
            ants.grad.zero_()
        output = model(ants)
        output.backward()

        with torch.no_grad():
            last = ants.clone()
            noise = torch.rand_like(ants) * noise_std
            ants -= (lr * ants.grad * (ants != 0).float())
            if calculate_min_distance(ants) < 0.5:
                continue
        ants.grad.zero_()
    return ants

def get_optimized_cost():
    pass