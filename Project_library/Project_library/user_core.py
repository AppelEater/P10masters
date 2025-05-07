import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import typing

from jax import jit, vmap

import Project_library.core as pl


class User(typing.NamedTuple):
    """A user class that represents a user with a unique ID and a list of items."""
    latitude: float
    longitude: float
    id: int
    demand: float = 0.0 # Consider making a secondary class that is only for all QoS. You kinda just need to add more information to the state of the user.
    position: jax.typing.ArrayLike = jnp.array([0.0, 0.0, 0.0])

    def __repr__(self):
        return f"User(id={self.id} \n lat={self.latitude} \n lon={self.longitude} \n demand={self.demand} \n position={self.position})"

class square_cell(typing.NamedTuple):
    """A class representing a square cell."""
    # Define the fields of the named tuple
    lat: float  # Latitude of the cell center
    longi: float  # Longitude of the cell center
    lat_width : typing.List[float]  # Latitude range of the cell
    longi_width : typing.List[float]  # Longitude range of the cell
    density: float  # Density of users in the cell
    id: int  # ID of the cell
    users_amount : int = 0  # Number of users in the cell
    users: User = 0  # List of users in the cell

@jax.jit
def create_user(id, lat_range, longi_range, key : jax.typing.ArrayLike) -> User:
    """Creates a list of users with random latitude and longitude."""
    earth_radius = 6378 # km

    # Generate a random key for the user
    key, subkey = jax.random.split(key, 2)

    latitude = jax.random.uniform(key, minval=lat_range[0], maxval=lat_range[1])
    key = jax.random.split(key, 1)[0]
    # Split the key for the next random number generation
    longitude = jax.random.uniform(key, minval=longi_range[0], maxval=longi_range[1])

    position = pl.spherical_to_cartesian(earth_radius,latitude, longitude)
    return pl.User(
        latitude=latitude,
        longitude=longitude,
        position=position,
        id=id
    )

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

    #
    bing = jnp.sin(min_observation+jnp.pi/2)

    # Calculate the angle of the satellite
    alpha = jnp.arcsin(earth_radius*bing/distance)
    # Calculate the angle from the center of the earth
    lat_width = jnp.pi/2 - alpha - min_observation

    # Calculate the spherical value of the satellite
    radius, longitude, latitude = pl.cartesian_to_spherical(*satellite_position)
    
    # Define the ranges of the visible area
    lat_range = (latitude.item() - lat_width, latitude.item() + lat_width)
    lon_range = (longitude.item() - lat_width, longitude.item() + lat_width)

    return lat_range, lon_range, radius, alpha, distance, lat_width