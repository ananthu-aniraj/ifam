# Ref: https://github.com/MadryLab/backgrounds_challenge/blob/master/tools/model_utils.py
import torch as ch
from tqdm import tqdm as tqdm


class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''

    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


class NormalizedModel(ch.nn.Module):
    """
    """

    def __init__(self, model, dataset):
        super(NormalizedModel, self).__init__()
        self.normalizer = InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, inp):
        """
        """
        normalized_inp = self.normalizer(inp)
        output = self.model(normalized_inp)
        return output


def eval_model(loader, model, map_to_in9, map_in_to_in9=True, device=None, eval_baseline=False):
    """
    *Internal function*
    Args:
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model: model to evaluate
        map_in_to_in9: whether or not to map model outputs from
        device: device to run the model on
        ImageNet class labels to ImageNet9 class labels
        eval_baseline: whether to evaluate the baseline model
    Returns:
        The average top1 accuracy across the epoch.
    """

    model = model.eval()

    iterator = tqdm(enumerate(loader), total=len(loader))
    correct = 0
    for i, (inp, target) in iterator:
        inp = inp.to(device)
        with ch.no_grad():
            if eval_baseline:
                output = model(inp)
            else:
                output = model(inp)[0]

        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        if map_in_to_in9:
            if map_to_in9 is None:
                raise ValueError('Need to pass in mapping from IN to IN9')
            pred_list = list(pred.numpy())
            pred = ch.LongTensor([map_to_in9[str(x)] for x in pred_list])
        correct += (pred == target).sum().item()

    return correct / len(loader.dataset)


def adv_bgs_eval_model(bg_loader, model, im, mask, fg_class, batch_size, map_to_in9, map_in_to_in9=True, device = None, eval_baseline=False):
    """
    *Internal function*
    Args:
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model: model to evaluate
        use_mapping: whether or not to map model outputs from
        ImageNet class labels to ImageNet9 class labels
    Returns:
        The average top1 accuracy across the epoch.
    """

    model = model.eval()

    big_im = im.repeat(batch_size, 1, 1, 1)
    big_mask = mask.repeat(batch_size, 1, 1, 1)

    # iterator = tqdm(enumerate(bg_loader), total=len(bg_loader))
    for i, (inp, target) in enumerate(bg_loader):
        # for i, (inp, target) in iterator:
        if inp.shape[0] != batch_size:  # For handling the last batch
            big_im = im.repeat(inp.shape[0], 1, 1, 1)
            big_mask = mask.repeat(inp.shape[0], 1, 1, 1)
        combined = inp * (1 - big_mask) + big_mask * big_im

        # #         Uncomment the next 5 lines to look at images
        #         import matplotlib.pyplot as plt
        #         for_viz = transforms.ToPILImage()(combined[0])
        #         plt.imshow(for_viz)
        #         plt.show()
        #         import pdb; pdb.set_trace()
        combined = combined.to(device)
        with ch.no_grad():
            if eval_baseline:
                output = model(combined)
            else:
                output = model(combined)[0]

        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        if map_in_to_in9:
            pred_list = list(pred.numpy())
            pred = ch.LongTensor([map_to_in9[str(x)] for x in pred_list])

        has_adversarial = (pred != fg_class).any().item()
        if has_adversarial:
            return True
    return False
