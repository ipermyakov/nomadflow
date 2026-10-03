import matplotlib.pyplot as plt


def format_list(l, fmt_str):
    return [float(fmt_str.format(i)) for i in l]


def gen_pbar_str(loss_dict, include_keys=None):
    if include_keys is None:
        include_keys = loss_dict.keys()
    pbar_str = []
    for key, value in loss_dict.items():
        if key in include_keys:
            if len(value.shape) == 0:
                value = value[None]
            pbar_str.append(f"{key}: {format_list(value.tolist(), '{:.4f}')}")
    pbar_str = ", ".join(pbar_str)
    return pbar_str


def update_epoch_loss_dict(epoch_dict, loss_dict):
    for key, value in loss_dict.items():
        if len(value.shape) == 0:
            value = value.unsqueeze(0)
        if key not in epoch_dict.keys():
            epoch_dict[key] = value.tolist()
        else:
            epoch_dict[key] += value.tolist()

    return epoch_dict
