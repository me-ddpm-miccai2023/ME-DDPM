import torch

def dense_image_warp(input_image, input_flow):
    permuted_image = torch.permute(input_image, [0, 3, 1, 2])

    batch_size, channels, height, width = permuted_image.shape
    height_indices = torch.arange(height, dtype=permuted_image.dtype, device=permuted_image.device)
    width_indices = torch.arange(width, dtype=permuted_image.dtype, device=permuted_image.device)

    height_indices, width_indices = torch.meshgrid(height_indices, width_indices)
    hw_indices = torch.stack((width_indices, height_indices), 2).unsqueeze(0)
    input_flow = input_flow.flip(-1)

    HW_tensor = torch.tensor([[[[width, height]]]], dtype=permuted_image.dtype, device=permuted_image.device)
    sample_grid = (2 * hw_indices - 2 * input_flow + 1.0) / HW_tensor - 1.0

    warped_image = torch.nn.functional.grid_sample(
        permuted_image, sample_grid, mode='bilinear', padding_mode='border', align_corners=False
    )

    return torch.permute(warped_image, [0, 2, 3, 1])
