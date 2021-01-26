
import numpy as np
import matplotlib.pyplot as plt

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def crf_refine(img, label, n_classes, softmax=False, n_iter=5):

    assert len(img.shape) == 2  # only grayscaled image

    # Setup the CRF model ##############################################################################################
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_classes)

    # get unary potentials (neg log probability)
    if softmax:
        U = unary_from_softmax(label)
    else:
        U = unary_from_labels(label, n_classes, gt_prob=0.8, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    # Smoothing factor
    feats = create_pairwise_gaussian(sdims=(2, 2), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    # Appearance factor
    feats = create_pairwise_bilateral(sdims=(3, 3), schan=(2,), img=img, chdim=-1)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Do inference and compute refined map #############################################################################
    # Run five inference steps.
    Q = d.inference(n_iter)

    # Find out the most probable class for each pixel.
    refined_map = np.argmax(Q, axis=0)

    return refined_map


