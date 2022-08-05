import numpy as np
from shape_model.LPCA.LPCALib import utils as lpca_utils, subspacemodels as subspacemodels, dists as lpca_dists, \
    kernels as lpca_kernels


class LPCA():
    def __init__(self, num_levels, target_variation, fixed_modes=None):
        assert 0 < target_variation <= 1
        assert num_levels > 0
        self.target_variation = target_variation
        self.fixed_modes = fixed_modes
        self.mean_vector = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.num_modes = None
        self.percent_of_variance = None
        self.number_of_levels = num_levels
        self.distance_matrix = None
        self.std_dev = None

    def pca(self, data_matrix: np.ndarray):
        """Performs a principal component analysis of the variance found in each coordinate of the given shapes

        :param train_shapes: expected to be a 2-dimensional matrix, where each column represents one shape
        :return: None
        """
        assert len(data_matrix.shape) == 2
        print('Fit locality SSM (PCA) to training data: ')

        pca_model = subspacemodels.SubspaceModelGenerator.compute_pca_subspace(data_matrix, self.target_variation,
                                                                               debug=True)
        # convert from matrix type to ndarray
        self.mean_vector = np.asarray(pca_model.translation_vector)
        self.eigenvectors = np.asarray(pca_model.basis)
        self.eigenvalues = np.asarray(pca_model.eigenvalues).squeeze()
        self.num_modes = pca_model.basis.shape[1]
        self.percent_of_variance = self.target_variation  # todo: this is incorrect
        # compute std-dev to predict distribution
        self.std_dev = np.std(self.project_shapes(data_matrix).transpose(), axis=1)

        return self.mean_vector, self.eigenvectors, self.eigenvalues, self.num_modes, self.percent_of_variance

    def lpca(self, data_matrix: np.ndarray):
        """Performs a principal component analysis of the variance found in each coordinate of the given shapes

        :param train_shapes: expected to be a 2-dimensional matrix, where each column represents one shape
        :return: None
        """
        assert len(data_matrix.shape) == 2
        print('Fit locality SSM (Kernel) to training data: ')
        # compute a schedule for the locality levels, i.e. the maximum distances for each level
        max_distance = self.compute_max_distance(data_matrix)
        distance_schedule = max_distance * np.power(0.5, np.array(range(0, self.number_of_levels)))
        # compute a shape distance -- we use euclidean point distance
        mean_shape = np.mean(data_matrix, axis=1, keepdims=True)
        mean_matrix = np.hstack((mean_shape[0::3], mean_shape[1::3], mean_shape[2::3]))
        shape_distance = lpca_dists.ShapeNDEuclideanDist(np.repeat(mean_matrix, 3, axis=0))
        # compute kernels for distance schedules
        N = data_matrix.shape[1]  # number of samples/shapes
        cov_kernel = lpca_kernels.CovKernel(1 / (N - 1))
        gamma = 1 / (2 * ((2 * distance_schedule) ** 2))
        exponent = 2
        kernel_list = []
        kernels_lvl_1 = [(cov_kernel, None, 'data', None, 1)]
        kernel_list.append(kernels_lvl_1)
        for lvl in range(1, self.number_of_levels):
            kernels_lvl_i = [(cov_kernel, None, 'data', None, 1),
                             (lpca_kernels.ExponentialKernel(gamma[lvl], exponent), np.multiply, 'dist', shape_distance,
                              1)]
            kernel_list.append(kernels_lvl_i)
        max_rank = min(N * 10, 200)
        max_ranks = [max_rank] * self.number_of_levels

        lambda_eig_method = lambda x, y, z: lpca_utils.eig_fast_spsd_kernel(x, y, z, 10)

        localized_model, tmp1, tmp2 = subspacemodels.SubspaceModelGenerator.compute_localized_subspace_kernel(
            data_matrix, self.target_variation, distance_schedule, kernel_list, max_ranks, None,
            debug=True, eig_method=lambda_eig_method,
            merge_method=lpca_utils.merge_subspace_models_closest_rotation_decorr_kernel,
            test_method=None)

        # localized_model = subspacemodels.SubspaceModelGenerator.compute_localized_subspace_media(data_matrix,
        #                                                                                          self.target_variation,
        #                                                                                          self.distance_matrix,
        #                                                                                          distance_schedule,
        #                                                                                          test_data=None,
        #                                                                                          merge_method=lpca_utils.merge_subspace_models_closest_rotation_decorr,
        #                                                                                          debug=True)
        # convert from matrix type to ndarray
        self.mean_vector = np.asarray(localized_model.translation_vector)
        self.eigenvectors = np.asarray(localized_model.basis)
        self.eigenvalues = np.asarray(localized_model.eigenvalues).squeeze()
        self.num_modes = localized_model.basis.shape[1]
        self.percent_of_variance = self.target_variation  # todo: this is incorrect
        # compute std-dev to predict distribution
        self.std_dev = np.std(self.project_shapes(data_matrix).transpose(), axis=1)

        return self.mean_vector, self.eigenvectors, self.eigenvalues, self.num_modes, self.percent_of_variance

    def project_shapes(self, shapes):
        return np.matmul(self.eigenvectors.transpose(), shapes - self.mean_vector).transpose()

    def generate_shapes(self, weights):
        generated = self.mean_vector + np.matmul(self.eigenvectors, weights.transpose())
        return generated

    def sample_shapes(self, num_samples, alpha):
        # stdev = np.sqrt(self.eigenvalues)
        stdev = self.std_dev
        weights = np.random.randn(num_samples, self.num_modes) * stdev

        # restrict samples to range +-alpha*stddev
        weights = np.maximum(np.minimum(weights, alpha * stdev), -alpha * stdev)
        return self.generate_shapes(weights)

    def get_object_indicator(self) -> np.array:
        """Returns an array of the same length as the vectorized object contours with an index for each object

        :return: obj_indicator: np.array with indices for each object
        """
        indicator = np.zeros(2048)
        indicator[1024:] = 1
        return indicator

    def compute_distance_matrix(self, data_matrix: np.ndarray) -> None:
        """Computes a modified (multi-object) geodesic distance on shapes as a matrix of N x N, where N = #pts*d

        :param data_matrix: expected to be a 2-dimensional matrix, where each column represents one shape
        :return: None
        """
        # ONLY VALID FOR SCR DATA !!!
        obj_indicator = self.get_object_indicator()
        mean_shape = np.mean(data_matrix, axis=1, keepdims=True)
        mean_shape = mean_shape * 0.35  # account for spacing to get nearly same results as our code
        assert len(obj_indicator) == len(mean_shape), "Length of object indicator and mean shape does not fit!"

        self.distance_matrix = lpca_utils.compute_multi_object_pseudo_euclidean_geodesic_shortest_path_3d_point_distance_matrix(
            mean_shape, obj_indicator, eta=10, kappa=150)

    def compute_max_distance(self, data_matrix: np.ndarray) -> float:
        """Computes the maximum distance between two points of the mean shape of the training data

        :param data_matrix: expected to be a 2-dimensional matrix, where each column represents one shape
        :return: distance
        """
        mean_shape = np.mean(data_matrix, axis=1, keepdims=True)
        # ONLY VALID FOR 3D DATA !!!
        # compute max distance on 'mean shape' --> approximated by a minimum bbox
        mean_matrix = np.hstack((mean_shape[0::3], mean_shape[1::3], mean_shape[2::3]))
        min_point = np.min(mean_matrix, axis=0)
        max_point = np.max(mean_matrix, axis=0)
        max_distance = np.sqrt(np.sum(np.squeeze(np.array(min_point - max_point)) ** 2))

        return max_distance
