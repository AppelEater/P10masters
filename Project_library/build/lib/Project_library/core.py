import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import jit, vmap



def test():
    print("This is a test function from the core module.")
    return True

@jit
def spherical_to_cartesian(latitude, longitude):
    """
    Converts spherical coordinates (latitude and longitude) to Cartesian coordinates (x, y, z).

    Parameters:
    latitude (float or ndarray): Latitude in radians. Values range from -π/2 to π/2.
    longitude (float or ndarray): Longitude in radians. Values range from -π to π.

    Returns:
    tuple: A tuple containing the Cartesian coordinates (x, y, z).
        - x (float or ndarray): Cartesian x-coordinate.
        - y (float or ndarray): Cartesian y-coordinate.
        - z (float or ndarray): Cartesian z-coordinate.
    """
    x = jnp.cos(longitude) * jnp.cos(latitude)  # Longitude
    y = jnp.cos(latitude) * jnp.sin(longitude)
    z = jnp.sin(latitude)  # Example function for z value
    return x, y, z


def generate_latitude_longitude_points(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude, range_of_latitude = [-jnp.pi / 2, jnp.pi / 2], range_of_longitude = [-jnp.pi, jnp.pi]):
    """
    Generates a meshgrid of latitude and longitude points.

    Parameters:
    NumberOfPointsAlonglatitude (int): Number of points to generate along the latitude axis.
    NumberOfPointsAlonglongitude (int): Number of points to generate along the longitude axis.
    range_of_latitude (list, optional): Range of latitude values in radians. Defaults to [-π/2, π/2].
    range_of_longitude (list, optional): Range of longitude values in radians. Defaults to [-π, π].

    Returns:
    tuple: A tuple containing two 2D arrays representing the latitude and longitude meshgrid.
        - latitude (ndarray): 2D array of latitude values.
        - longitude (ndarray): 2D array of longitude values.
    """
    latitude = jnp.linspace(range_of_latitude[0], range_of_latitude[1], NumberOfPointsAlonglatitude)
    longitude = jnp.linspace(range_of_longitude[0], range_of_longitude[1], NumberOfPointsAlonglongitude)
    cell_mesh = jnp.meshgrid(latitude, longitude)
    return cell_mesh


def three_dimensional_plot_latitude_longitude_points(cell_mesh):
    """
    Creates a 3D scatter plot of latitude and longitude points converted to Cartesian coordinates.

    Parameters:
    cell_mesh (tuple): A tuple containing two 2D arrays representing the latitude and longitude meshgrid.
        - cell_mesh[0] (ndarray): 2D array of latitude values.
        - cell_mesh[1] (ndarray): 2D array of longitude values.

    Returns:
    tuple: A tuple containing the matplotlib figure and axes objects.
        - fig (Figure): The matplotlib figure object.
        - ax (Axes3D): The 3D axes object for the plot.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Value')
    ax.view_init(30, 45)

    calculate_along_row = vmap(spherical_to_cartesian, in_axes=(0, 0))

    for i in range(cell_mesh[0].shape[0]):
        x, y, z = calculate_along_row(cell_mesh[0][i], cell_mesh[1][i])
        ax.scatter(x, y, z, color='b', marker='o')
    
    return fig, ax


def generate_latitude_longitude_divisions(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude, range_of_latitude = [-jnp.pi / 2, jnp.pi / 2], range_of_longitude = [-jnp.pi, jnp.pi]):
    """
    Generates latitude and longitude divsions with adjusted ranges to be able to know the boundary.

    Parameters:
    NumberOfPointsAlonglatitude (int): Number of points to generate along the latitude axis.
    NumberOfPointsAlonglongitude (int): Number of points to generate along the longitude axis.
    range_of_latitude (list, optional): Range of latitude values in radians. Defaults to [-π/2, π/2].
    range_of_longitude (list, optional): Range of longitude values in radians. Defaults to [-π, π].

    Returns:
    tuple: A tuple containing two 1D arrays representing the latitude and longitude points.
        - latitude (ndarray): 1D array of latitude values.
        - longitude (ndarray): 1D array of longitude values.
    """
    off_set_latitude = (range_of_latitude[1] - range_of_latitude[0]) / (NumberOfPointsAlonglatitude * 2)
    off_set_longitude = (range_of_longitude[1] - range_of_longitude[0]) / (NumberOfPointsAlonglongitude * 2)

    range_of_latitude = [range_of_latitude[0] + off_set_latitude, range_of_latitude[1] - off_set_latitude]
    range_of_longitude = [range_of_longitude[0] + off_set_longitude, range_of_longitude[1] - off_set_longitude]

    latitude = jnp.linspace(range_of_latitude[0], range_of_latitude[1], NumberOfPointsAlonglatitude)
    longitude = jnp.linspace(range_of_longitude[0], range_of_longitude[1], NumberOfPointsAlonglongitude)
    return latitude, longitude