class SOMModel(keras.Model):
    def __init__(self, h, w, d, n_epochs, lr, **kwargs):
        super().__init__(**kwargs)
        self.w = w
        self.h = h
        self.d = d
        self.epoch = 1
        self.n_epochs = n_epochs
        self.lr = lr
        self.initial_radius = tf.math.divide(tf.minimum(self.h, self.w), 2)

        self.grid_weights = tf.Variable(initial_value=tf.random.normal((self.h * self.w, self.d)))
        self.x, self.y = tf.meshgrid(tf.range(self.h), tf.range(self.w), indexing='ij')
        self.grid_coordinates = tf.convert_to_tensor([[i, j] for i, j in zip(tf.reshape(self.x, (self.h * self.w,)).numpy(), tf.reshape(self.y, (self.h * self.w)).numpy())])

        self.qe = None
        self.te = None

    def call(self, inputs, **kwargs):
        distances = tf.pow(tf.subtract(tf.expand_dims(self.grid_weights, axis=0),
                                       tf.expand_dims(inputs, axis=1)), 2)
        distances = tf.reduce_sum(distances, axis=2)

        bmu_indices = tf.argmin(distances, axis=1)
        bmu_coordinates = tf.gather(self.grid_coordinates, bmu_indices)

        # quantization error
        bmu_vecs = np.array([self.grid_weights.numpy()[i].tolist() for i in bmu_indices])
        self.qe = round(np.mean(np.abs(bmu_vecs - inputs)), 4)

        # topographic error
        bmus_ind1 = np.argpartition(distances, 3, axis=1)[:, 0]
        bmus_ind2 = np.argpartition(distances, 3, axis=1)[:, 1]
        bmu_coords1 = np.array([self.grid_coordinates.numpy()[i].tolist() for i in bmus_ind1])
        bmu_coords2 = np.array([self.grid_coordinates.numpy()[i].tolist() for i in bmus_ind2])
        bmus_gap = np.abs(bmu_coords1 - bmu_coords2).sum(axis=1)
        self.te = round(np.mean(bmus_gap != 1), 4)

        neighbourhood_radius = tf.math.multiply(self.initial_radius,
                                                tf.math.exp(tf.math.divide(tf.negative(self.epoch), self.n_epochs)))

        bmu_dist = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.grid_coordinates, axis=0),
                                                    tf.expand_dims(bmu_coordinates, axis=1)), 2), 2)

        neighbourhood_fn = tf.exp(tf.divide(tf.negative(tf.cast(bmu_dist, dtype=tf.float32)),
                                            tf.multiply(tf.square(
                                                tf.cast(neighbourhood_radius, dtype=tf.float32), 2), 2)))

        current_lr = tf.math.multiply(self.lr,
                                      tf.math.exp((tf.math.divide(self.epoch,
                                                                  self.n_epochs))))
        lr_matrix = tf.multiply([current_lr], neighbourhood_fn)

        numerator = tf.reduce_sum(tf.multiply(
            tf.expand_dims(lr_matrix, axis=-1), tf.expand_dims(inputs, axis=1)), axis=0)
        denominator = tf.expand_dims(tf.reduce_sum(lr_matrix, axis=0) + float(1e-20), axis=-1)

        self.grid_weights.assign(tf.math.divide(numerator, denominator))
