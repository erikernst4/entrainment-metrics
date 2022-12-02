from unittest import TestCase

import numpy as np
from scipy.io import wavfile
from sklearn.neighbors import KNeighborsRegressor

from speechalyze import InterPausalUnit
from speechalyze.continuous import TimeSeries, calculate_metric


class KNNTestCase(TestCase):
    def setUp(self):
        self.cases = {
            'long_100-200-300_x2': {
                'audio': wavfile.read("./data/100-200-300_long_x2.wav"),
                'ipus': [
                    InterPausalUnit(0.0, 4.0, {'F0_MAX': 100.003}),
                    InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                    InterPausalUnit(28.0, 32.0, {'F0_MAX': 100.003}),
                    InterPausalUnit(36.0, 40.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(44.0, 52.0, {'F0_MAX': 300.002}),
                ],
                'ipus_middle_points_in_time': [2.0, 10.0, 20.0, 30.0, 38.0, 48.0],
                'ipus_feature_values': [
                    100.003,
                    200.002,
                    300.002,
                    100.003,
                    200.002,
                    300.002,
                ],
            },
            'long_300-200-100_x2': {
                'ipus': [
                    InterPausalUnit(0.0, 8.0, {'F0_MAX': 300.002}),
                    InterPausalUnit(12.0, 16.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(20.0, 24.0, {'F0_MAX': 100.003}),
                    InterPausalUnit(28.0, 36.0, {'F0_MAX': 300.002}),
                    InterPausalUnit(40.0, 44.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(48.0, 52.0, {'F0_MAX': 100.003}),
                ],
            },
        }

    def test_calculate_knn_time_series_longx2(self):
        case = self.cases['long_100-200-300_x2']
        model = KNeighborsRegressor(n_neighbors=4)
        X = np.array(case['ipus_middle_points_in_time'])
        X = X.reshape(-1, 1)
        y = case['ipus_feature_values']
        model.fit(X, y)

        samplerate, data = case['audio']
        audio_lenght = data.shape[0] // samplerate
        values_to_predict = [i for i in range(0, audio_lenght, int(0.01 * samplerate))]
        values_to_predict = np.array(values_to_predict)
        values_to_predict = values_to_predict.reshape(-1, 1)

        time_series = TimeSeries(
            feature='F0_MAX', interpausal_units=case['ipus'], method='knn', k=4
        )
        np.testing.assert_almost_equal(
            model.predict(values_to_predict),
            time_series.predict(values_to_predict),
        )

    def test_calculate_knn_time_series_warnings_longx2(self):
        case = self.cases['long_100-200-300_x2']
        time_series = TimeSeries(
            feature='F0_MAX', interpausal_units=case['ipus'], method='knn', k=4
        )
        values_before_start_to_predict = np.array([-1.0]).reshape(-1, 1)
        values_after_end_to_predict = np.array([53.0]).reshape(-1, 1)

        self.assertWarns(Warning, time_series.predict, values_before_start_to_predict)

        self.assertWarns(Warning, time_series.predict, values_after_end_to_predict)

    def test_calculate_proximity_with_itself(self):
        case = self.cases['long_100-200-300_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case['ipus'], method='knn', k=4
        )

        self.assertEqual(
            calculate_metric("proximity", time_series_a, time_series_a), 0.0
        )

    def test_calculate_convergence_with_itself(self):
        case = self.cases['long_100-200-300_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case['ipus'], method='knn', k=4
        )
        self.assertTrue(
            np.isnan(
                calculate_metric("convergence", time_series_a, time_series_a),
            )
        )

    def test_calculate_synchrony_with_itself(self):
        case = self.cases['long_100-200-300_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case['ipus'], method='knn', k=4
        )

        self.assertEqual(
            calculate_metric("synchrony", time_series_a, time_series_a), 1.0
        )

    def test_calculate_proximity_oposites(self):
        case_a = self.cases['long_100-200-300_x2']
        case_b = self.cases['long_300-200-100_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case_a['ipus'], method='knn', k=4
        )
        time_series_b = TimeSeries(
            feature='F0_MAX', interpausal_units=case_b['ipus'], method='knn', k=4
        )

        np.testing.assert_almost_equal(
            calculate_metric("proximity", time_series_a, time_series_b),
            -0.0096134877908014,
        )

    def test_calculate_convergence_oposites(self):
        case_a = self.cases['long_100-200-300_x2']
        case_b = self.cases['long_300-200-100_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case_a['ipus'], method='knn', k=4
        )
        time_series_b = TimeSeries(
            feature='F0_MAX', interpausal_units=case_b['ipus'], method='knn', k=4
        )

        np.testing.assert_almost_equal(
            calculate_metric("convergence", time_series_a, time_series_b),
            0.0001704841413878858,
        )

    def test_calculate_synchrony_oposites(self):
        case_a = self.cases['long_100-200-300_x2']
        case_b = self.cases['long_300-200-100_x2']

        time_series_a = TimeSeries(
            feature='F0_MAX', interpausal_units=case_a['ipus'], method='knn', k=4
        )
        time_series_b = TimeSeries(
            feature='F0_MAX', interpausal_units=case_b['ipus'], method='knn', k=4
        )

        np.testing.assert_almost_equal(
            calculate_metric("synchrony", time_series_a, time_series_b),
            -0.9380342373191826,
        )
