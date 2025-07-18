import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import jit, vmap

import typing


def test():
    print("This is a test function from the core module.")
    

@jit
def spherical_to_cartesian(radius,latitude, longitude):
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
    x = radius*jnp.cos(longitude) * jnp.cos(latitude)  # Longitude
    y = radius*jnp.cos(latitude) * jnp.sin(longitude)
    z = radius*jnp.sin( latitude)  # Example function for z value
    return x, y, z

@jit
def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (radius, latitude, longitude).

    Parameters:
    x (float or ndarray): Cartesian x-coordinate.
    y (float or ndarray): Cartesian y-coordinate.
    z (float or ndarray): Cartesian z-coordinate.

    Returns:
    tuple: A tuple containing the spherical coordinates (radius, latitude, longitude).
        - radius (float or ndarray): Distance from the origin.
        - latitude (float or ndarray): Latitude in radians. Range: [-π/2, π/2]
        - longitude (float or ndarray): Longitude in radians. Range: [-π, π]
    """
    radius = jnp.sqrt(x**2 + y**2 + z**2)
    latitude = jnp.arcsin(z / radius)
    longitude = jnp.arctan2(y, x)
    return radius, latitude, longitude


@jax.jit
def calculate_distance(x_user, y_user, z_user, satellite_position: jax.typing.ArrayLike) -> float:
    """Calculate the distance between a user and a satellite."""
    # Calculate the distance using the Euclidean formula
    return jnp.sqrt((satellite_position[0] - x_user) ** 2 + (satellite_position[1] - y_user) ** 2 + (satellite_position[2] - z_user) ** 2)

# Calculate the elevation angle of the satellite
@jax.jit
def calculate_elevation(x_user, y_user, z_user, satellite_position : jax.typing.ArrayLike) -> float:
    user_pos = jnp.asarray((x_user, y_user, z_user))
    elev = jnp.arcsin(jnp.dot(satellite_position-user_pos, user_pos/jnp.linalg.norm(user_pos))/jnp.linalg.norm(satellite_position-user_pos))
    return elev

calculate_distances = vmap(calculate_distance, in_axes=(0, 0, 0, None))
calculate_elevations = vmap(calculate_elevation, in_axes=(0, 0, 0, None))

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
    latitude = jnp.linspace(range_of_latitude[0], range_of_latitude[1], NumberOfPointsAlonglatitude*2+1)
    longitude = jnp.linspace(range_of_longitude[0], range_of_longitude[1], NumberOfPointsAlonglongitude*2+1)
    
    cell_mesh = jnp.meshgrid(latitude[1::2], longitude[1::2], indexing='ij')
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
    latitude_divisions = jnp.linspace(range_of_latitude[0], range_of_latitude[1], NumberOfPointsAlonglatitude*2+1)
    longitude_divisions = jnp.linspace(range_of_longitude[0], range_of_longitude[1], NumberOfPointsAlonglongitude*2+1)


    return latitude_divisions[0::2], longitude_divisions[0::2]

@jit
def calculate_area_on_sphere(radius, latitude_divisions, longitude_divisions):
    """
    Calculates the area on a sphere defined by the latitude and longitude divisions.

    Parameters:
    radius (float): Radius of the sphere.
    latitude_divisions (ndarray): 1D array of latitude divisions in radians.
    longitude_divisions (ndarray): 1D array of longitude divisions in radians.

    Returns:
    float: Area on the sphere defined by the latitude and longitude divisions.
    """
    # Calculate the area of each cell on the sphere

    area = radius**2 * jnp.diff(jnp.asarray(longitude_divisions)) * (jnp.sin( latitude_divisions[1])-jnp.sin(latitude_divisions[0]))
    
    return area

@jit
def locate_user_cell(lat_user, long_user, latitude_divsions, longitude_divisions):
    """
    Locates the cell in which a user is located based on their latitude and longitude.

    Parameters:
    lat_user (float): Latitude of the user in radians.
    long_user (float): Longitude of the user in radians.
    latitude_divsions (ndarray): 1D array of latitude divisions in radians.
    longitude_divisions (ndarray): 1D array of longitude divisions in radians.

    Returns:
    tuple: A tuple containing the indices of the cell in which the user is located.
        - lat_index (int): Index of the latitude division.
        - long_index (int): Index of the longitude division.
    """
    lat_index = jnp.searchsorted(latitude_divsions,lat_user) - 1
    long_index = jnp.searchsorted(longitude_divisions,long_user) - 1

    return lat_index, long_index

@jax.jit
def visible_angle(min_observation, satellite_position : jnp.ndarray) -> tuple[float, float, float, float, float, float]:
    """Calculate the visible angle area of a satellite given its position.

    Args:
        min_observation (float): The minimum observational angle in radians.
        satellite_position (jnp.ndarray): The position of the satellite in Cartesian coordinates.

    Returns:
        tuple: A tuple containing the visible area and the radius of the visible area.
    """
    # Calculate the distance from the satellite to the center of the Earth
    distance = jnp.linalg.norm(satellite_position)
    # Calculate the radius of the Earth
    earth_radius = 6371  # in km

    #Intermediate calculation
    bing = jnp.sin(min_observation+jnp.pi/2)

    # Calculate the angle of the satellite
    alpha = jnp.arcsin(earth_radius*bing/distance)
    # Calculate the angle from the center of the earth
    lat_width = jnp.pi/2 - alpha - min_observation

    # Calculate the spherical value of the satellite
    radius, longitude, latitude = cartesian_to_spherical(*satellite_position)
    
    # Define the ranges of the visible area
    lat_range = (latitude - lat_width, latitude + lat_width)
    lon_range = (longitude - lat_width, longitude + lat_width)    


    return lat_range, lon_range, radius, alpha, distance, lat_width

@jax.jit
def calculate_if_cells_within_visible_area(
        cell_latitude_list: float,
        cell_longitude_list: float,
        satellite_latitude: float,
        satellite_longitude: float,
        satellite_height: float,
        alpha : float,
        earth_radius: float = 6371.0) -> bool:
    """Check if a cell is within the visible area of a satellite.
        cell_latitude_list: Latitude of the cells in degrees.
        cell_longitude_list: Longitude of the cell in degrees.
        satellite_latitude: Latitude of the satellite in degrees.
        satellite_longitude: Longitude of the satellite in degrees.
        satellite_height: Height of the satellite in kilometers.
        alpha: Angle of visibility in degrees.
    """
    
    phi_s, lam_s, alpha = jnp.deg2rad(jnp.asarray([satellite_latitude, satellite_longitude, alpha]))
    cos_alpha  = jnp.cos(alpha)

    # Make into a tensor
    phi_c  = jnp.deg2rad(cell_latitude_list)[:, None]          # (N,1)
    lam_c  = jnp.deg2rad(cell_longitude_list)[None, :]          # (1,M)

    sin_phi_s, cos_phi_s = jnp.sin(phi_s), jnp.cos(phi_s)
    sin_phi_c, cos_phi_c = jnp.sin(phi_c), jnp.cos(phi_c)

    # Calculate the angle from the center of the earth
    cos_dsigma = (sin_phi_s * sin_phi_c +
                  cos_phi_s * cos_phi_c * jnp.cos(lam_c - lam_s))
    
    # Calculate the conditional
    return cos_dsigma >= cos_alpha

@jax.jit
def disconnect_time(v: jnp.ndarray, time_step) -> jnp.ndarray:
    # Boolean mask for zeros
    is_zero = (v == 0)

    # Scanning function to compute lengths of consecutive zeros
    def scan_fn(carry, x):
        current_run = jnp.where(x, carry + 1, 0)
        return current_run, current_run

    # Scan over the boolean mask to get lengths of runs of zeros
    _, runs = jax.lax.scan(scan_fn, jnp.array(0), is_zero)

    
    # Check the shifted version of the boolean mask to find run-ends, where a run of zeros ends
    shifted = jnp.concatenate([is_zero[1:], jnp.array([False])])
    run_ends = is_zero & (~shifted)

    # Zero out everything except at run-ends
    run_lengths = runs * run_ends

    return run_lengths*time_step

disconnect_times = jax.jit(jax.vmap(disconnect_time, in_axes=(0, None)))


def calculate_pmf(A, active_users, M):
    alpha = A[0, 1]
    beta = A[1, 0]

    M1 = active_users
    M2 = M - M1

    k1 = jnp.arange(M1+1)
    k2 = jnp.arange(M2+1)

    # Binomial Distribution of each case
    Binom_from_on_to_on = jax.scipy.stats.binom.pmf(k1, n=M1, p=1-beta)
    Binom_from_off_to_on = jax.scipy.stats.binom.pmf(k2, n=M2, p=alpha)

    # Convolve the two distributions
    convolved = jax.scipy.signal.convolve(Binom_from_on_to_on, Binom_from_off_to_on, mode='full')

    # Normalize the convolved distribution
    convolved = convolved / jnp.sum(convolved)

    return convolved