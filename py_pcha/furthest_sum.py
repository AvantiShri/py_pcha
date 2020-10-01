"""FurthestSum algorithm to efficiently generate initial seeds/archetypes."""

import numpy as np
from numpy.matlib import repmat


def furthest_sum(K, noc, i, exclude=[]):
    """Furthest sum algorithm, to efficiently generat initial seed/archetypes.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).
    
    Parameters
    ----------
    K : numpy 2d-array
        Either a data matrix or a kernel matrix.

    noc : int
        Number of candidate archetypes to extract.

    i : int
        inital observation used for to generate the FurthestSum.

    exclude : numpy.1darray
        Entries in K that can not be used as candidates.

    Output
    ------
    i : int
        The extracted candidate archetypes
    """
    def max_ind_val(l): #max idx and corresponding value
        return max(zip(range(len(l)), l), key=lambda x: x[1])

    I, J = K.shape #I = num features, J = number of examples
    index = np.array(range(J))
    index[exclude] = 0
    index[i] = -1
    ind_t = i #ind_t is the index of the next candidatearchetype picked
    sum_dist = np.zeros((1, J), np.complex128)

    #if J (i.e. number of examples) > number of archetypes * number
    # of features (usually true)...
    if J > noc * I: 
        Kt = K #K is the data matrix - features x examples
        Kt2 = np.sum(Kt**2, axis=0) #Kt2 is the magnitude-squared of each example
        #the +11 is to give some room to update the early-chosen candidate archetypes
        for k in range(1, noc + 11):
            if k > noc - 1: #If we already have a full list of archetypes
                #'boot-out' the early chosen candidate archetypes to make room for
                # potentially better ones.
                #correct sum_dist to remove out the early-chosen archetype
                Kq = np.dot(Kt[:, i[0]], Kt) 
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                #restore the early-chosen archetype to list of potential
                # indices to pick as candidate archetypes
                index[i[0]] = i[0]
                #delete that early-chosen candidate archetype from the list of selected
                # archetypes.
                i = i[1:]
            #t = all indices that haven't been selected as archetypes yet
            t = np.where(index != -1)[0] 
            #Kq = dot product of vec at index ind_t (next candidate archetype) with all examples
            Kq = np.dot(Kt[:, ind_t].T, Kt)
            #np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t]) is the
            # distance between the vector at ind_t and all the examples
            #sum_dist keeps track of the sum of the distances from each point
            # to the candidate archetypes picked so far
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
            #find the not-yet-chosen example for which the sum of distances
            # to selected candidatearchetypes is the furthest.
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind] #set that as the next candidate archetype
            i.append(ind_t) #and append to the list of candidate achetypes
            index[ind_t] = -1 #mark that index as having been selected.
    else:
        if I != J or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = np.dot(Kt.T, Kt)
            K = np.lib.scimath.sqrt(
                repmat(np.diag(K), J, 1) - 2 * K + \
                repmat(np.mat(np.diag(K)).T, 1, J)
            )

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, noc + 11):
            if k > noc - 1:
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1

    return i
