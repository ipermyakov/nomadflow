import torch.nn.functional as F

def seq_losses_dense_batch(
    batch,
    model_outputs,
    seqwise_weight=1,
):
    token_data = batch['token']

    seq = token_data['seq']
    # seq_mask = token_data['seq_mask']
    seq_mask = token_data['token_mask']
    token_mask = token_data['token_mask']
    seq_noising_mask = token_data['seq_noising_mask']

    seq_loss_mask = token_mask & seq_mask & seq_noising_mask
    seq_logits = model_outputs['decoded_seq_logits']

    flat_masked_seq = (seq * seq_loss_mask).flatten(0, 1)
    flat_seq_logits = seq_logits.flatten(0, 1)
    seq_loss = F.cross_entropy(
        flat_seq_logits,
        flat_masked_seq,
        reduction='none'
    )
    seq_loss = seq_loss.view(seq.shape)
    seq_loss = seq_loss * seq_loss_mask

    if (~seq_loss_mask).all(dim=-1).any():
        # print("seq_loss_mask all False", batch['name'], (~seq_loss_mask).all(dim=-1))
        print("seq_loss_mask all False", (~seq_loss_mask).all(dim=-1))

    seq_loss = seq_loss * seqwise_weight
    seq_loss = seq_loss.sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    per_seq_recov = (seq == seq_logits[..., :-1].argmax(dim=-1))
    per_seq_recov = (per_seq_recov * seq_loss_mask).sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
    }

    return out_dict