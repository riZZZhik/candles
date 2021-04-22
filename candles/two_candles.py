from math import exp

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class TwoCandles:
    def __init__(self, u1=1, u2=1, v1=0.2, v2=0.15, t=2, dt=0.001,
                 au=2.7, av=1, e=0.001, o=1, c=5, m0=0.05):
        """ Initialize main class variables.

        :param u1: Relative temperature of the first candle
        :type u1: float
        :param u2: Relative temperature of the second candle
        :type u2: float
        :param v1: Relative oxygen concentration of the first candle
        :type v1: float
        :param v2: Relative oxygen concentration of the second candle
        :type v2: float
        :param t: End calculation time
        :type t: float
        :param dt: Time step
        :type dt: float
        :param au: Dimensionless coefficient %of something%  # TODO: Of what?
        :type au: float
        :param av: Dimensionless coefficient %of something%  # TODO: Of what?
        :type av: float
        :param e: # TODO
        :type e: float
        :param o: # TODO
        :type o: float
        :param c: # TODO
        :type c: float
        :param m0: # TODO
        :type m0: float
        """

        # Initialize input variables
        self.u1 = u1
        self.v1 = v1
        self.u2 = u2
        self.v2 = v2

        self.t = t
        self.dt = dt

        self.au = au
        self.av = av
        self.e = e
        self.o = o
        self.c = c
        self.m0 = m0

        # Initialize class variables
        self.data_lists = {
            "u1": [],
            "u2": [],
            "v1": [],
            "v2": []
        }

        self.norm_data = []
        self.limits = []

        logger.info("Initialized Two Candles class")

    def calculate_equation(self):
        """Calculation of the differential equation"""
        for _ in np.arange(0, self.t, self.dt):
            # Calculate variables
            delta_u1 = (1 / self.e * (-self.u1 + self.au * self.v1 * exp(self.u1 / (1 + self.u1 / self.c))) -
                        self.o * (1 + self.u1 / self.c) + self.o * self.m0 * (1 + self.u2 / self.c) ** 4) * self.dt
            delta_u2 = (1 / self.e * (-self.u2 + self.au * self.v2 * exp(self.u2 / (1 + self.u2 / self.c))) -
                        self.o * (1 + self.u2 / self.c) + self.o * self.m0 * (1 + self.u1 / self.c) ** 4) * self.dt

            delta_v1 = (1 - self.v1 - self.av * self.v1 * exp(self.u1 / (1 + self.u1 / self.c))) * self.dt
            delta_v2 = (1 - self.v2 - self.av * self.v2 * exp(self.u2 / (1 + self.u2 / self.c))) * self.dt

            self.u1 += delta_u1
            self.u2 += delta_u2
            self.v1 += delta_v1
            self.v2 += delta_v2

            # Save data to lists
            self.data_lists["u1"].append(self.u1)
            self.data_lists["u2"].append(self.u2)
            self.data_lists["v1"].append(self.v1)
            self.data_lists["v2"].append(self.v2)

        # Create time list
        time = [self.dt]
        for i in range(len(self.data_lists["u1"]) - 1):
            time.append(time[-1] + self.dt)

        # Normalize data
        data1, limits1 = self.data_preprocessing([[time, self.data_lists["u1"]], [time, self.data_lists["u2"]]])
        data2, limits2 = self.data_preprocessing([[time, self.data_lists["v1"]], [time, self.data_lists["v2"]]])
        self.norm_data = [data1, data2]
        self.limits = [limits1, limits2]

        logger.info("Calculated differential equation")

    @staticmethod
    def data_preprocessing(data):
        limits = [(min(data[1][i]), max(data[1][i])) for i in range(2)]

        for a in range(2):
            span = max(data[0][a]) - min(data[0][a])
            min_ = min(data[0][a])
            for idx in range(len(data)):
                preprocessed = (max(data[idx][a]) - min(data[idx][a])) / span
                data[idx][a] = [i / preprocessed + min_ - min([i / preprocessed
                                                               for i in data[idx][a]]) for i in data[idx][a]]

        return data, limits

    @staticmethod
    def show_plot(data, limits, title):
        """Create and show plot"""
        fig, ax = plt.subplots()

        for x, y in data:
            ax.plot(x, y)

        ax2, ax3 = ax.twinx(), ax.twiny()
        ax2.set_ylim(limits[1])
        ax3.set_xlim(limits[0])
        plt.title(title)
        plt.show()

    def show_graphs(self):
        """Show graphs using matplotlib"""
        self.show_plot(self.norm_data[0], self.limits[0], "Time dependence of relative temperature")
        logger.info('Showed "Time dependence of relative temperature" graph')
        self.show_plot(self.norm_data[1], self.limits[1], "Dependence of the relative concentration of oxygen")
        logger.info('Showed "Dependence of the relative concentration of oxygen" graph')

    def calculate_oscillation_period(self):
        max_value_ids = np.where(np.isin(self.norm_data[0][0][1], [23.214873619199803, 23.223769484885555]))[0]
        logger.debug(f"Max value ids are {max_value_ids} "
                     f"and distance between them is {abs(max_value_ids[0] - max_value_ids[1])}")
