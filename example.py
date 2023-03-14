from TICC_solver import TICC
import numpy as np

fname = "example_data.txt"
number_of_clusters = 8
# dataset has n = 10 variates
window_size = 3
ticc = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=11e-2, beta=600,
            maxIters=100, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1)
(cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

# MY EDITS TO BETTER UNDERSTAND THE RESULTS
list_of_lists = []
current_cluster = int(cluster_assignment[0])
new_list = []
# some more useful output than cluster assignment
for i in range(len(cluster_assignment)):
    current_observations_cluster = int(cluster_assignment[i])
    if current_observations_cluster != current_cluster:  # create new list
        current_cluster = current_observations_cluster
        list_of_lists.append(new_list)
        new_list = []
    new_list.append(current_cluster)
list_of_lists.append(new_list)  # the last one
print("Cluster assignment")
print("Number of segments: " + str(len(list_of_lists)))
repetition_of_cluster = [0] * number_of_clusters
for clusterx_list in list_of_lists:
    cluster_index = clusterx_list[0]
    repetition_of_cluster[int(cluster_index)] += 1
    print("Segment for cluster: " + str(cluster_index) + ", with number of observations: " + str(len(clusterx_list)))
print("\n")
print("Number of times cluster repeats (list is in order of cluster):")
print(repetition_of_cluster)

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
