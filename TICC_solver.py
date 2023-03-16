import errno
import os

import matplotlib

import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd

from src.TICC_helper import *
from src.admm_solver import ADMMSolver

matplotlib.use('Agg')


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5, lambda_parameter=11e-2,
                 beta=400, maxIters=1000, threshold=2e-5, write_out_file=False,
                 prefix_string="", num_proc=1, compute_BIC=False, cluster_reassignment=20, biased=False):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased
        # training data (left over design): 2d ndarray of dimension (no_train-observations, w*n), example (19605, 30)
        self.complete_d_train = None  # gets assigned in fit

        # this is a weird design, but it is how it's been done. Predict uses this data, fit sets it
        self.trained_model = {'computed_covariance': None,
                              'cluster_mean_stacked_info': None,
                              'time_series_col_size': None}
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit(self, input_file):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format
        # in the example times_series_arr = (19607,10)
        # rows and cols not defined
        times_series_arr, time_series_rows_size, time_series_col_size = self.load_data(input_file)

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        # list of length 19605
        training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks,
                                             self.window_size)  # indices of the training samples
        # 19605
        num_train_points = len(training_indices)

        # Stack the training data
        # 2d ndarray of dimension (no_train-observations, w*n), example (19605, 30)
        self.complete_d_train = self.stack_training_data(times_series_arr, time_series_col_size, num_train_points,
                                                         training_indices)

        # Initialization
        enforce_all_clusters_not_zero = False  # should 20 points be allocated to 0 clusters from it 2 onwards?
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
        gmm.fit(self.complete_d_train)
        # clustered_points = gmm.predict(self.complete_d_train)  # -> this is the original starting point

        # ndarray of length observations and values index of cluster the observation belongs to, example (19605)
        # clustered_points = np.random.randint(self.number_of_clusters, size=num_train_points)
        clustered_points = np.zeros(num_train_points)
        # clustered_points = np.ones(num_train_points)

        old_clustered_points = None  # points from last iteration

        # PERFORM TRAINING ITERATIONS
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            # {cluster_num: [indices of observation in that cluster]}
            train_clusters_arr = self.transform_into_dictionary_of_clusters_and_indices_of_points(clustered_points)

            # dictionary of key=cluster index and value=total number of observations in that cluster
            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # opt_res is list of k (8) ndarrays of length 465
            # this is the M part
            opt_res, cluster_mean_stacked_info, empirical_covariances = self.train_clusters(self.complete_d_train,
                                                                                            len_train_clusters,
                                                                                            time_series_col_size,
                                                                                            train_clusters_arr)

            computed_covariance, train_cluster_inverse = self.optimize_clusters(opt_res)

            print("Length of clusters from train_cluster_array:")
            for cluster in range(self.number_of_clusters):
                print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])

            # update old computed covariance
            old_computed_covariance = computed_covariance

            # this is the E part self.smoothen and update clusters
            # SMOOTHENING
            lle_all_points_clusters = self.smoothen_clusters(computed_covariance, cluster_mean_stacked_info,
                                                             self.complete_d_train, time_series_col_size)
            # Update cluster points - using NEW smoothening
            clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

            # save training result
            self.trained_model = {'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'time_series_col_size': time_series_col_size}

            # recalculate lengths
            new_train_clusters = self.transform_into_dictionary_of_clusters_and_indices_of_points(clustered_points)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            # missing from the paper but this code is assigning points to empty cluster, not quite sure why this
            # is needed and how it's done
            before_empty_cluster_assign = clustered_points.copy()
            print("Length of clusters from clustered_points BEFORE EMPTY ASSIGN:")
            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            if enforce_all_clusters_not_zero:
                if iters != 0:
                    old_cluster_num__used = [x[1] for x in list(old_computed_covariance.keys())]
                    cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                     old_cluster_num__used]
                    norms_sorted = sorted(cluster_norms, reverse=True)
                    # clusters that are not 0 as sorted by norm
                    valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                    # Add a point to the empty clusters
                    # assuming more non empty clusters than empty ones
                    counter = 0
                    for cluster_num in range(self.number_of_clusters):
                        if len_new_train_clusters[cluster_num] == 0:
                            cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                            counter = (counter + 1) % len(valid_clusters)
                            print("cluster that is zero is:", cluster_num, "selected cluster instead is:",
                                  cluster_selected)
                            start_point = np.random.choice(
                                new_train_clusters[cluster_selected])  # random point number from that cluster
                            for i in range(0, self.cluster_reassignment):
                                # put cluster_reassignment points from point_num in this cluster
                                point_to_move = start_point + i
                                if point_to_move >= len(clustered_points):
                                    break
                                clustered_points[point_to_move] = cluster_num
                                computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                    self.number_of_clusters, cluster_selected]
                                cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = self.complete_d_train[
                                                                                                  point_to_move, :]

            print("Length of clusters from clustered_points:")
            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            # self.write_plot(clustered_points, str_NULL, training_indices)

            # TEST SETS STUFF
            # LLE + swtiching_penalty
            # Segment length
            # Create the F1 score from the graphs from k-means and GMM
            # Get the train and test points
            train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
                                                                 training_indices)
            # compute the matchings
            # matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
            #                                                                   train_confusion_matrix_GMM,
            #                                                                   train_confusion_matrix_kmeans)

            print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):  # stop if nothing changes
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            # PRINT DIFFERENCES IN SEGMENTS
            # else:
            #     if old_clustered_points is not None:
            #         print("PREVIOUS CLUSTERED POINTS")
            #         analyse_segments(old_clustered_points, self.number_of_clusters)
            #     print("CURRENT CLUSTERED POINTS")
            #     analyse_segments(clustered_points, self.number_of_clusters)

            old_clustered_points = before_empty_cluster_assign
            # end of training
        # train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
        #                                                      training_indices)
        # train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
        #                                                       training_indices)
        # train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, clustered_points_kmeans,
        #                                                          training_indices)
        #
        # self.compute_f_score(matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
        #                      train_confusion_matrix_GMM, train_confusion_matrix_kmeans)

        bic = None
        if self.compute_BIC:
            bic = compute_bic(time_series_rows_size, clustered_points, train_cluster_inverse, empirical_covariances)

        # train cluster inverse is np.linalg.inv(computed_covariance), it's not used in training!!!

        return clustered_points, train_cluster_inverse, bic

    def transform_into_dictionary_of_clusters_and_indices_of_points(self, clustered_points):
        """
        Transforms ndarray of clustered points into dictionary of clusters and lists of incices for points in that cluster
        :param clustered_points: ndarray of length TS length
        :return: {cluster_id:[indices of points in cluster]}

        """
        train_clusters_arr = {k: [] for k in range(self.number_of_clusters)}  # {cluster: [point indices]}
        for point_idx, cluster_num in enumerate(clustered_points):
            train_clusters_arr[cluster_num].append(point_idx)
        return train_clusters_arr

    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
        print("\n\n")
        print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        print("Not implemented")
        # # matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        # # matching_GMM = find_matching(train_confusion_matrix_GMM)
        # matching_EM = find_matching(train_confusion_matrix_EM)
        # correct_e_m = 0
        # correct_g_m_m = 0
        # correct_k_means = 0
        # for cluster in range(self.number_of_clusters):
        #     matched_cluster_e_m = matching_EM[cluster]
        #     matched_cluster_g_m_m = matching_GMM[cluster]
        #     matched_cluster_k_means = matching_Kmeans[cluster]
        #
        #     correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
        #     correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
        #     correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        # return matching_EM, matching_GMM, matching_Kmeans

    def write_plot(self, clustered_points, str_NULL, training_indices):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.number_of_clusters + 0.5))
        plt.show()
        if self.write_out_file: plt.savefig(
            str_NULL + "TRAINING_EM_lam_sparse=" + str(self.lambda_parameter) + "switch_penalty = " + str(
                self.switch_penalty) + ".jpg")
        plt.close("all")
        print("Done writing the figure")

    def smoothen_clusters(self, computed_covariance, cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        print("Smoothen cluster computed covariance keys:")
        print(computed_covariance.keys())
        cluster_num_currently_used = [x[1] for x in list(computed_covariance.keys())]
        for cluster in cluster_num_currently_used:
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(self.num_blocks - 1) * n,
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in cluster_num_currently_used:
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, opt_res):
        # initialise
        # dictionary with key=cluster index and value 2d ndarray of Toeplitz Matrices for each cluster
        train_cluster_inverse = {}
        # dictionary with key=(k,cluster index) and value covariance matrix, if you calculate np.linalg.inv(computed_covariance) you get train_cluster_inverse, this is sparse
        computed_covariance = {}

        for cluster in range(self.number_of_clusters):
            if opt_res[cluster] is None:
                continue
            opt_res_for_cluster = opt_res[cluster]
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upperToFull(opt_res_for_cluster, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2

        return computed_covariance, train_cluster_inverse

    def train_clusters(self, complete_D_train, len_train_clusters, n, train_clusters_arr):
        # initialise
        optRes = [None for i in range(self.number_of_clusters)]
        # dictionary with key=(k,cluster index) and value a list of length w*n not sure what the values mean
        cluster_mean_stacked_info = {}
        empirical_covariances = [None for i in range(self.number_of_clusters)]

        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.window_size * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)
                # Fit a model - OPTIMIZATION
                probSize = self.window_size * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                empirical_covariance = np.cov(np.transpose(D_train), bias=self.biased)

                rho = 1
                solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, empirical_covariance)  # M step
                verbose = False
                optRes[cluster] = solver(1000, 1e-6, 1e-6, verbose)
                empirical_covariances[cluster] = empirical_covariance
        return optRes, cluster_mean_stacked_info, empirical_covariances

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def prepare_out_directory(self):
        str_NULL = self.prefix_string + "lam_sparse=" + str(self.lambda_parameter) + "maxClusters=" + str(
            self.number_of_clusters + 1) + "/"
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return str_NULL

    def load_data(self, input_file):
        Data = np.loadtxt(input_file, delimiter=",")
        (m, n) = Data.shape  # m: num of observations, n: size of observation vector
        print("completed getting the data")
        return Data, m, n

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.number_of_clusters)
        print("num stacked", self.window_size)

    def predict_clusters(self, test_data):
        """
        Given the current trained model, predict clusters.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        """
        if not isinstance(test_data, np.ndarray):
            raise TypeError(
                "test data must be a numpy array with rows observation at time i and columns different variates!")

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'], test_data,
                                                         self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

        return clustered_points


def analyse_segments(cluster_assignment, number_of_clusters):
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
        print(
            "Segment for cluster: " + str(cluster_index) + ", with number of observations: " + str(len(clusterx_list)))
    print("\n")
    print("Number of times cluster repeats (list is in order of cluster):")
    print(repetition_of_cluster)
