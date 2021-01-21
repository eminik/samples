import numpy as np
import matplotlib.pyplot as plt


def in_circle(xy, centre, radius):
    return (xy[0] - centre[0]) ** 2 + (xy[1] - centre[1]) ** 2 < radius ** 2


def calculate_pi(n=10000, shrink=1):
    centre = np.array([0.5, 0.5]) / shrink
    radius = 0.5 / shrink
    box_area = (1 / shrink) ** 2
    points = np.random.uniform(0, 1 / shrink, size=(n, 2))

    points_classifications = np.apply_along_axis(lambda xy: in_circle(xy, centre, radius), axis=1, arr=points)
    area = np.mean(points_classifications) * box_area

    plt.plot(points[points_classifications == 0, 0], points[points_classifications == 0, 1], 'b*')
    plt.plot(points[points_classifications, 0], points[points_classifications, 1], 'r*')
    plt.show()

    return area / radius ** 2
