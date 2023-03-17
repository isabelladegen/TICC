from TICC_solver import TICC, analyse_segments
import numpy as np

fname = "example_data.txt"
number_of_clusters = 8
# dataset has n = 10 variates
window_size = 3

# ORIGINAL settings as used in the paper
# this is in the viterbi path where when there seems to be too many clusters the algorithm just cycles between pushing
# front and back between the last and second last cluster resulting in it not converging.
# This only is a problem if not starting with gmm for this dataset (but surely would happen for other datasets too)
# allow_zero_cluster_inbetween = True

# this is a bit at the end of fit before the next iteration where the algorithm assigns some points to all the clusters
# and avoids zero clusters. Again it's not described in the paper why this needs to happen and I don't think it does
# but for it to converge with the above variable set to true it does need to happen
# reassign_points_to_zero_clusters = True

# the paper seems to have been done running TICC on top of a GMM cluster assignment. It's not clear to me how this
# does not impact its outcome and is probably the reason why it does so much better than GMM. However with the goal
# to keep temporal consistency it makes more sense to me to start off with all points assigned to cluster zero.
# Impact of this has not been evaluated yet.
# use_gmm_initialisation = True

# NEW SETTING that I feel make more sense and the produce half as big a BIC on the example data, k = 6 would in this
# case be the resulting number of clusters?
# without the reassigning points to zero cluster it becomes possible to have zero clusters. To me this means that
# too many clusters were given and the algorithm can converge with less. For that to happen and not get stuck in an
# endless loop where the last segment gets pushed between cluster k-1 and k both of these variables need to be false
# this avoids zero clusters inbetween clusters, but allows them at the end, and it stops reassigning points to zero
# clusters
allow_zero_cluster_inbetween = False
reassign_points_to_zero_clusters = False
# this initialises TICC with all observations being assigned to the first cluster at the beginning
use_gmm_initialisation = False

ticc = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=11e-2, beta=600,
            maxIters=100, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1, compute_BIC=True,
            allow_zero_cluster_inbetween=allow_zero_cluster_inbetween)
cluster_assignment, cluster_MRFs, bic = ticc.fit(input_file=fname,
                                                 reassign_points_to_zero_clusters=reassign_points_to_zero_clusters,
                                                 use_gmm_initialisation=use_gmm_initialisation)

# MY EDITS TO BETTER UNDERSTAND THE RESULTS
analyse_segments(cluster_assignment, number_of_clusters)
print('\n')
print("BIC: " + str(bic))

# shape is a dictionary with a key for each of the 8 clusters with each an array of w*n x w*n where w=window size, n = no variants
cluster_one_mfr = cluster_MRFs[1]
print("\n")
print("Shape of MFRs:")
print(cluster_one_mfr.shape)
no_time_series = 10


def assert_two_matrices_are_the_same(m1, m2, msg):
    # without rounding they are not equal which might just be due to np's equality
    if (np.around(m1, decimals=8) != np.around(m2, decimals=4)).all():
        print(msg + " are not equal")


print("\n")
print("List all the block matrices of each cluster's MFR that are not equal")
# check which clusters have proper Toeplitz Matrices
for cluster_id, mfr in cluster_MRFs.items():
    # check diagonal
    print("Cluster " + str(cluster_id) + ":")
    first_a0 = mfr[0:no_time_series, 0:no_time_series]
    second_a0 = mfr[no_time_series:2 * no_time_series, no_time_series:2 * no_time_series]
    third_a0 = mfr[2 * no_time_series:, 2 * no_time_series:]
    assert_two_matrices_are_the_same(first_a0, second_a0, "first a0 and second a0")
    assert_two_matrices_are_the_same(first_a0, third_a0, "first a0 and third a0")

    # check lower half diagonal
    first_a1 = mfr[no_time_series:2 * no_time_series, 0:no_time_series]
    second_a1 = mfr[2 * no_time_series:, no_time_series:2 * no_time_series]
    assert_two_matrices_are_the_same(first_a1, second_a1, "first a1 and second a1")

    first_a1t = mfr[0:no_time_series, no_time_series:2 * no_time_series]
    second_a1t = mfr[no_time_series:2 * no_time_series, 2 * no_time_series:]
    assert_two_matrices_are_the_same(first_a1t, second_a1t, "first a1t and second a1t")
    assert_two_matrices_are_the_same(first_a1.T, first_a1t, "first a1 transposed and first a1t")

    a2 = mfr[2 * no_time_series:, 0:no_time_series]
    a2t = mfr[0:no_time_series, 2 * no_time_series:]
    assert_two_matrices_are_the_same(a2.T, a2t, "a2 transposed and a2t")

np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
