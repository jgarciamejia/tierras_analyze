import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip 

def find_consecutive_sequences(arr):
    consecutive_sequences = []
    current_sequence = []

    for i in range(len(arr)):
        if i == 0 or arr[i] == arr[i - 1] + 1:
            current_sequence.append(arr[i])
        else:
            if len(current_sequence) > 1:
                consecutive_sequences.append(current_sequence)
            current_sequence = [arr[i]]

    if len(current_sequence) > 1:  # Check if the last sequence is consecutive
        consecutive_sequences.append(current_sequence)

    return consecutive_sequences

def indices_not_in_array(arr1, arr2):
    set2 = set(arr2)
    return [index for index, _ in enumerate(arr1) if index not in set2]

def flare_masker(x_list, x, y, threshold=2):
    mask = np.ones_like(y, dtype='bool')

    print('Masking flares...')
    for i in range(len(x_list)):

        # grab the ith night of data
        night_inds = np.where((x>=x_list[i][0])&(x<=x_list[i][-1]))[0]
        x_ = x[night_inds]
        y_ = y[night_inds]

        # first do a 6-sigma clipping to remove any single major outliers
        v, l, h = sigmaclip(y_, 6, 6)

        y_[np.where((y_>h)|(y_<l))[0]] = np.nan

        use_inds = np.arange(len(y_))

        # now sigmaclip the flux with the major outliers removed 
        v2, l2, h2 = sigmaclip(y_[~np.isnan(y_)], threshold, threshold)

        # identify points above the high treshold
        hi_inds = np.where(y_ > h2)[0] 
        flare_inds = []
        if len(hi_inds) != 0:
            # look for consecutive sequences of points above the high threshold
            consecutive_sequences = find_consecutive_sequences(hi_inds)
            if len(consecutive_sequences) > 0:
                for j in range(len(consecutive_sequences)):
                    flare_inds.extend([consecutive_sequences[j][0]-1]) # flag the exposure before the flare starts as well
                    flare_inds.extend(consecutive_sequences[j])
                    ind = consecutive_sequences[j][-1]+1
                    v3, l3, h3 = sigmaclip(v2,2,2) # sigma clip on the data that have had the 3-sigma outliers removed
                    # keep adding points to the flare indices until the flux has returned to baseline
                    if ind < len(y_):
                        while y_[ind] > h3:
                            flare_inds.extend([ind])
                            ind += 1
                            if ind == len(y_):
                                break


        flare_inds = np.unique(flare_inds).astype('int') # fixes issue if multiple flares in have been identified in a single sequence
        mask[night_inds[flare_inds]] = False

    # plt.figure()
    # plt.plot(x,y)
    # plt.plot(x[mask],y[mask])
    # breakpoint()
    return mask