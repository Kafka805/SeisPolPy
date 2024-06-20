# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:26:23 2024

@author: austin.abreu
"""
import unittest as ut
from SeisPolPy import eigSort as eg
from SeisPolPy import filtermerge as fm
from numpy import vstack, linalg, sin, linspace, pi, random, fft, abs, log10
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from scipy import signal


class seisTest(ut.TestCase):
    def test_eigSort(self):
        # NIST Test Data
        # [https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc5.htm]

        a = [4.0, 4.2, 3.9, 4.3, 4.1]
        b = [2.0, 2.1, 2.0, 2.1, 2.2]
        c = [0.6, 0.59, 0.58, 0.62, 0.63]

        vals, vecs = eg.eigSort(a, b, c)

        S1 = [0.025, 0.0075, 0.00175]
        S2 = [0.0075, 0.007, 0.00135]
        S3 = [0.00175, 0.00135, 0.00043]

        S = vstack((S1, S2, S3))

        vim, vigor = linalg.eig(S)

        first_case = assert_allclose(vals, vim, rtol=1e-10)
        second_case = assert_allclose(vecs, vigor, rtol=1e-10)

        self.assertEqual()

    def test_filterMerge(self):
        # Set parameters
        sampling_rate = 1000  # samples per second
        duration = 10  # seconds
        time = linspace(0, duration, sampling_rate * duration)
        nyquist_freq = sampling_rate / 2

        # Generate sine waves
        freq1 = 2  # Hz
        freq2 = 3  # Hz
        freq3 = 10  # Hz
        sine_wave1 = sin(2 * pi * freq1 * time)
        sine_wave2 = sin(2 * pi * freq2 * time)
        modulator = sin(2 * pi * freq3 * time)

        # Modulate and combine sine waves
        expected_signal = sine_wave1 + sine_wave2
        modulated_signal = expected_signal + modulator

        # Add noise
        noise_level = 0.1  # adjust this value to control the noise level
        noise = random.normal(0, noise_level, len(time))
        noisy_signal = modulated_signal + noise

        # Define filtering parameters
        low = 2
        high = 8
        dt = 1 / sampling_rate
        order = 2

        # Apply the filter and retrieve the parameters of the equation
        b, a, filtered_signal = fm.butterworth(
            noisy_signal, dt, low, high, order, test=True
        )

        # Calculate expected frequency response
        fft_expected = fft.fft(expected_signal)
        freq_expected = fft.fftfreq(len(fft_expected), dt)
        expected_freq_response = abs(freq_expected)

        # Calculate actual frequency response
        fft_filtered = fft.fft(filtered_signal)
        freq_retrieved = fft.fftfreq(len(fft_filtered), dt)
        actual_freq_response = abs(freq_retrieved)

        # Compare manufactured signals and actual signals
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(time, expected_signal)
        plt.title("Expected Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        plt.plot(time, filtered_signal)
        plt.title("Filtered Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        # Make assertion
        for expected, actual in zip(
            expected_freq_response, actual_freq_response
        ):
            self.assertAlmostEqual(expected, actual, places=4)

    def test_polarity(self):
        pass

    def test_dStruct(self):
        pass

    def test_SeisPol(self):
        pass


if __name__ == "__main__":
    ut.main(verbosity=0)
