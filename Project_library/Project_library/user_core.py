import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import typing

from jax import jit, vmap




class User(typing.NamedTuple):
    """A user class that represents a user with a unique ID and a list of items."""
    latitude: float
    longitude: float
    id: int
    throughput: float = 0.0 # Consider making a secondary class that is only for all QoS. You kinda just need to add more information to the state of the user.
    position: jax.typing.ArrayLike = jnp.array([0.0, 0.0, 0.0])

    def __repr__(self):
        return f"User(id={self.id} \n lat={self.latitude} \n lon={self.longitude})"


