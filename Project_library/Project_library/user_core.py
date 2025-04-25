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
