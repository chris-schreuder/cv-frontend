import cv2
import pickle
import warnings
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# from base import Stream
# from plotting import plot_tracks, plot_deleted_tracks
from scipy.stats.distributions import chi2

warnings.filterwarnings('ignore')

ELLIPSE = True
ONLY_VALID = True
ALPHA = 0.99
STEPS = 350
COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


class Observation:
    def __init__(self, step, id, validity, z, covariance, colour):
        self.id = id
        self.step = step

        self.z = z
        self.R = covariance
        self.valid = validity

        self.colour = colour
        self.status = 'non-matched'  # False = not-matched, True is matched

    def print(self):
        print("id: " + str(self.id))
        print("measurement: " + str(self.z))
        print("covariance: " + str(self.R))
        print("validity: " + str(self.valid))


class KalmanFilter:
    def __init__(self, dt, x=None, sigma=None, A=None, H=None, Q=None, **kwargs):
        self.updated = False
        # always 2 dimensions when tracking in the image frame
        self.dim = 2

        self.z = None
        self.v = None
        self.S = None
        self.d = None

        # mode
        model = kwargs.get('model', 'constant_acceleration')

        if model == 'constant_velocity':
            # states
            self.x = x or np.zeros(self.dim * 2, dtype=float)
            self.sigma = sigma or np.diag(10000 * np.ones(len(self.x)))

            # models
            self.A = A or np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=float)

            self.H = H or np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], dtype=float)

            # measurement prediction
            self.zp = np.dot(self.H, self.x)

            # noise matrices
            self.Q = Q or np.diag([10000.0, 500.0, 10000.0, 500.0])
            # self.Q = Q or np.diag([0.001, 0.001, 0.1, 0.1])
            # self.Q = Q or 10000*np.array([[0.25 * dt ** 4, 0., 0.5 * dt ** 3, 0.],
            #                              [0., 0.25 * dt ** 4, 0., 0.5 * dt ** 3],
            #                              [0.5 * dt ** 3, 0., dt ** 2, 0.],
            #                              [0., 0.5 * dt ** 3, 0., dt ** 2]], dtype='float')

        elif model == 'constant_acceleration':
            # states
            self.x = x or np.zeros(self.dim * 3, dtype=float)
            self.sigma = sigma or np.diag(100. * np.ones(len(self.x)))

            # models
            self.A = np.array([[1, 0, dt, 0, 0.5 * dt ** 2, 0.],
                               [0, 1, 0, dt, 0., 0.5 * dt ** 2],
                               [0, 0, 1, 0, dt, 0.],
                               [0, 0, 0, 1, 0., dt],
                               [0, 0, 0, 0., 1., 0.],
                               [0, 0, 0, 0., 0., 1.]], dtype=float)

            self.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0]], dtype=float)

            # measurement prediction
            self.zp = np.dot(self.H, self.x)

            # noise matrices
            # self.Q = np.diag([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
            self.Q = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 500.0])

    def prediction(self):
        self.x = np.dot(self.A, self.x)
        self.sigma = np.dot(self.A, np.dot(self.sigma, self.A.T)) + self.Q

        self.zp = np.dot(self.H, self.x)

        return self.x, self.sigma

    def measurement(self, z, R):
        self.z = z
        self.v = z - self.zp
        self.S = np.dot(self.H, np.dot(self.sigma, self.H.T)) + R
        self.d = np.dot(self.v.T, np.dot(np.linalg.inv(self.S), self.v))

    def correction(self):
        K = np.dot(self.sigma, np.dot(self.H.T, np.linalg.inv(self.S)))
        self.x = self.x + np.dot(K, self.v)
        self.sigma = self.sigma - np.dot(K, np.dot(self.H, self.sigma))

        return self.x, self.sigma

    def run_filter(self, z, z_valid, cov):
        self.prediction()

        if z_valid:
            self.measurement(z, cov)

            if self.d < chi2.ppf(ALPHA, df=2):
                self.correction()


class KFTracks(KalmanFilter):
    def __init__(self, dt, id, x=None, sigma=None, A=None, H=None, Q=None):
        self.iobs = None
        self.id = id
        self.status = 'non-matched'
        self.lost = 0

        KalmanFilter.__init__(self, dt, x=x, sigma=sigma, A=A, H=H, Q=Q, model='constant_acceleration')

        self.x_hist = np.zeros((1, self.x.shape[0]), dtype=float)
        self.xp_hist = np.zeros((1, self.x.shape[0]), dtype=float)
        self.z_hist = np.zeros((1, self.zp.shape[0]), dtype=float)
        self.sigma_hist = np.zeros((1, self.sigma.shape[0], self.sigma.shape[1]), dtype=float)
        self.sigmap_hist = np.zeros((1, self.sigma.shape[0], self.sigma.shape[1]), dtype=float)

    def record(self, pred=False):
        if pred:
            self.xp_hist = np.append(self.xp_hist, self.x.reshape((1, self.x.shape[0])), axis=0)
            self.sigmap_hist = np.append(self.sigmap_hist, self.sigma.reshape((1,
                                                                               self.sigma.shape[0],
                                                                               self.sigma.shape[1])), axis=0)

        else:
            self.x_hist = np.append(self.x_hist, self.x.reshape((1, self.x.shape[0])), axis=0)
            self.z_hist = np.append(self.z_hist, self.z.reshape((1, self.z.shape[0])), axis=0)

            self.sigma_hist = np.append(self.sigma_hist, self.sigma.reshape((1,
                                                                             self.sigma.shape[0],
                                                                             self.sigma.shape[1])), axis=0)


class KFTracker:
    def __init__(self, dt, track_index, file_name=None, colour=(0, 0, 255)):
        self.dt = dt
        self.t = -1  # number of measurements instances

        self.tracks = []
        self.deleted_tracks = []
        self.measurements = {}

        self.colour = colour

        self.maxLost = 5
        self.track_index = track_index
        self.n_tracks = 0
        self.total_tracks = 0

        self.offline = False

        if file_name:
            self.offline = True
            self.mat = scipy.io.loadmat(file_name)
            self.dt = self.mat['dt']
            self.process_measurements(self.mat['obsdata'])

    def process_measurements(self, data):
        for time in range(data[0, 0][1].shape[1]):
            self.measurements[time] = []

            for sensors in range(data.shape[1]):
                sensor_data = data[0, sensors]
                id = sensor_data[0][0, 0]
                z = sensor_data[1][:, time]
                R = sensor_data[2]
                validity = sensor_data[3][0, time]
                colour = sensor_data[4]

                if validity and ONLY_VALID:
                    self.measurements[time].append(Observation(time, id, validity, z, R, colour))

    def process_camera_measurements(self, data, time):
        for obs in data:
            R = np.diag([1., 1.])
            z = np.asarray(obs)
            z[0] = z[0]
            z[1] = z[1]
            self.measurements[time].append(Observation(time, 0, True, z, R, None))

    def nearest_neighbour_association(self, obs):
        n_tracks = self.n_tracks  # len(self.tracks)
        n_obs = len(obs)

        D = np.zeros((n_tracks, n_obs), dtype=float)

        for i in range(n_tracks):
            for j in range(n_obs):
                v = obs[j].z - self.tracks[i].zp
                S = np.dot(self.tracks[i].H, np.dot(self.tracks[i].sigma, self.tracks[i].H.T)) + obs[j].R
                D[i, j] = np.dot(v.T, np.dot(np.linalg.inv(S), v))
                self.tracks[i].iobs = -1

        n_match = 0
        terminate = False

        matched_targets = []
        matched_obs = []

        while not terminate:
            d_min = np.min(D)
            d_args = np.where(D == np.min(D))
            i, j = d_args[0][0], d_args[1][0]

            if (d_min < 100000) & (d_min <= chi2.ppf(ALPHA, df=len(obs[j].z))):
                self.tracks[i].iobs = obs[j].id
                self.tracks[i].status = 'matched'
                obs[j].status = 'matched'

                matched_targets.append(i)
                matched_obs.append(j)

                D[i, :] = 100000
                D[:, j] = 100000

                n_match += 1

            else:
                terminate = True

        return matched_obs, matched_targets

    def run(self, centroids):
        measure = False

        # propagate states for existing tracks
        self.propagate_states()

        if True:
            self.t += 1
            self.measurements[self.t] = []
            # _, centroids, frame = cam.measurement()

            if len(centroids) > 0:
                self.process_camera_measurements(centroids, self.t)
                measure = True

        matched_obs = []
        matched_tracks = []

        if measure:
            # associate measurements
            matched_obs, matched_tracks = self.measurement_association(self.t)

        # update existing tracks
        self.update_tracks(self.t, matched_tracks, matched_obs)

        if measure:
            # spawn new tracks for unassigned measurements
            self.spawn_tracks(self.t)

        # if ax:
        #     plot_tracks(tracker, ax)

    def propagate_states(self):
        for track in self.tracks:
            track.prediction()

            # record the predicted state for plotting
            track.record(pred=True)

    def measurement_association(self, index):
        matched_obs = []
        matched_tracks = []

        # associate measurements with existing tracks
        if (len(self.measurements[index]) > 0) & (len(self.tracks) > 0):
            matched_obs, matched_tracks = self.nearest_neighbour_association(self.measurements[index])

        return matched_obs, matched_tracks

    def update_tracks(self, index, matched_tracks, matched_obs):
        to_delete = []
        for track_index in range(self.n_tracks):
            if track_index in matched_tracks:
                self.tracks[track_index].lost = 0
                j = matched_tracks.index(track_index)
                i = matched_obs[j]
                self.tracks[track_index].measurement(self.measurements[index][i].z, self.measurements[index][i].R)
                self.tracks[track_index].correction()
                self.tracks[track_index].record(pred=False)

            else:
                self.tracks[track_index].lost += 1
                if self.tracks[track_index].lost > self.maxLost:
                    to_delete.append(track_index)

                else:
                    self.tracks[track_index].z = self.tracks[track_index].zp
                    self.tracks[track_index].measurement(self.tracks[track_index].z, np.array([[10000., 0.],
                                                                                               [0., 10000.]]))
                    self.tracks[track_index].correction()
                    self.tracks[track_index].record(pred=False)

        for delete in to_delete:
            self.deleted_tracks.append(self.tracks[delete])
            del self.tracks[delete]
            self.n_tracks -= 1

    def spawn_tracks(self, index):
        for obs in range(len(self.measurements[index])):
            if self.measurements[index][obs].valid:  # TODO: move to sample measurement step
                if self.measurements[index][obs].status == 'non-matched':
                    self.tracks.append(KFTracks(self.dt, self.track_index+self.total_tracks))
                    self.tracks[self.n_tracks].x[:2] = self.measurements[index][obs].z
                    self.tracks[self.n_tracks].z = self.measurements[index][obs].z
                    self.tracks[self.n_tracks].colour = COLOURS[self.total_tracks % len(COLOURS)]
                    self.tracks[self.n_tracks].status = 'created'
                    self.n_tracks += 1
                    self.total_tracks += 1
                    # print("n_tacks: " + str(self.n_tracks))
                    # print("total_tracks: " + str(self.total_tracks))
                    # print("len(self.tracks): " + str(self.tracks))


def create_data_from_cam(file_name='camera_measurements2.pickle'):
    i = 0
    measurements = {}

    cam = Stream(src='/Users/aidan-landsberg/Movies/exit-enter2.mp4')
    cam.start()

    while i < STEPS:
        measurements[i] = []
        detections, centroid, frame = cam.measurement()

        if len(centroid) > 0:
            print(centroid)
            measurements[i] = centroid

        # if len(centroid) > 0:
        #     for obs in centroid:
        #         measurements[i] = []
        #         id = 0
        #         z = np.asarray(obs)
        #         R = np.diag([10., 10.])
        #         validity = True
        #         colour = None
        #         measurements[i].append(Observation(i, id, validity, z, R, colour))
        #
        #     # cv2.imshow('feed', cam.frame)

        i += 1
        print(i)

        cv2.imshow('feed', cam.frame)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    # Store data (serialize)
    with open(file_name, 'wb') as handle:
        pickle.dump(measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("done!")
    # cv2.destroyAllWindows()
    return


def create_measurements_from_file(file_name='camera_measurements2.pickle'):
    # Load data(deserialize)
    with open(file_name, 'rb') as handle:
        unserialized_data = pickle.load(handle)

    x = []
    y = []
    xdot = []
    ydot = []

    for i, j in unserialized_data.items():
        if len(j) > 0 and i < 99:
            for k in j:
                x.append(k.z[0])
                y.append(k.z[1])
        # else:
        #     x.append(None)
        #     y.append(None)

    for l in range(len(x)):
        if l > 0:
            if x[l] or x[l-1]:
                xdot.append((x[l] - x[l-1])/0.2)
                ydot.append((y[l] - y[l-1])/0.2)

            else:
                xdot.append(None)
                ydot.append(None)

    # plt.figure()
    # plt.scatter(np.array(x), np.array(y), s=1, c='k')
    return unserialized_data, x, y, xdot, ydot


if __name__ == "__main__":
    FILE = 'enter-exit.pickle'

    # create_data_from_cam(file_name=FILE)

    # data, x, y, xdot, ydot = create_measurements_from_file(file_name=FILE)

    with open(FILE, 'rb') as handle:
        unserialized_data = pickle.load(handle)

    # # Load data(deserialize)
    # with open(FILE, 'rb') as handle:
    #     data = pickle.load(handle)

    # # tracker = KFTracker(file_name='/Users/aidan-landsberg/Downloads/code-archives/robotics
    # # /People_Detection_and_Tracking/dataobs3.mat')

    fig1, ax = plt.subplots(1, 1)
    ax.axis([0, 1920, 0, 1080])

    tracker = KFTracker(0, file_name=None)
    for i, j in unserialized_data.items():
        tracker.run(j)

    # plot_tracks(tracker, ax)
    # plot_deleted_tracks(tracker, ax)
    # plt.show()
