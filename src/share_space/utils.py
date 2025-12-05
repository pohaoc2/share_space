import torch

def inverse_transform(img):
    return (img*0.5) + 0.5

def inverse_transform_batch(imgs: torch.Tensor):
    imgs = imgs.permute(0, 2, 3, 1).cpu().detach()
    imgs = (imgs*0.5) + 0.5
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs

def get_all_samples_from_loader(loader):
    sim_tensor = torch.cat([b['simulation'] for b in loader], dim=0)
    exp_tensor = torch.cat([b['experimental'] for b in loader], dim=0)

    return sim_tensor, exp_tensor