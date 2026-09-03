"""The two segmentation arms the whole package is parameterised by.

Its own module because every layer needs the names -- the pipeline runs them, the statistics
calibrate on one and score both, the figures colour by them, the commands take them as flags -- and
none of those should have to import another layer to learn a pair of strings.

    reference    the segmentation carried by the cloud itself: a human labelling
    predicted    the segmentation a model produced, read from per-point label files

The model is fitted on `reference` alone and applied to both, so the `predicted` arm's extra error
measures the segmentation rather than a recalibration that would absorb it. A run may use only the
reference arm, in which case the comparison simply is not made.
"""

REFERENCE = 'reference'
PREDICTED = 'predicted'
ARMS = (REFERENCE, PREDICTED)

# What each arm is called in a figure legend or a printed table
ARM_LABEL = {REFERENCE: 'Manual segmentation', PREDICTED: 'Model segmentation'}
