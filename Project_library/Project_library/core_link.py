import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax import jit, vmap

import typing

# Caculate the path loss
@jax.jit
def calculate_path_loss(distance: float, frequency: float) -> float:
    """Calculate the path loss using the free space path loss formula.
    
    Args:
        distance (float): The distance between the user and the satellite in m.
        frequency (float): The frequency of the signal in MHz.
    """
    # Calculate the path loss using the free space path loss formula
    path_loss = 32.45 + 20 * jnp.log10(frequency) + 20 * jnp.log10(distance*1000)

    return path_loss

# Calculate the gain from the user to the satellite
@jax.jit
def calculate_gain_fixed(elevation:float, frequency:float) -> float:
    """Calculate the gain from the user to the satellite.
    
    Args:
        elevation (float): The elevation angle of the satellite in degrees.
        frequency (float): The frequency of the signal in MHz.
        
    return:
        float: The gain from the user to the satellite in dB.
    """

    user_gain = 0.0
    satellite_gain = 30.0
    # Values from 3gpp TR 38.811 and 38.821

    return satellite_gain, user_gain

def calculate_noise(sky_temperature : float, Bandwidth :float , Kb : float) -> float:
    """Calculate the noise using the formula N = k * T * B.
    
    Args:
        sky_temperature (float): The sky temperature in Kelvin.
        Bandwidth (float): The bandwidth in Hz.
        Kb (float): The Boltzmann constant in J/K.
        
    return:
        float: The noise in dBm.
    """
    # Calculate the noise using the formula N = k * T * B
    noise = 10 * jnp.log10(Kb * sky_temperature * Bandwidth)
    return noise

# Calculate the SNR
@jax.jit
def calculate_snr(power : float, frequency : float, bandwidth : float, distance: float, elevation: float) -> float:
    """Calculate the SNR using the path loss and gain."""
    # Calculate the SNR using the formula SNR = gain - path_loss - nois

    gain_tx, gain_rx = calculate_gain_fixed(elevation, None)
    path_loss = calculate_path_loss(distance, frequency)
    noise = calculate_noise(300, bandwidth, 1.38e-23)

    power_db = 10 * jnp.log10(power)

    snr = power_db + gain_tx + gain_rx - path_loss - noise

    return snr

# Calculate the capacity of each channel with a uniform distribution of bandwidths
@jax.jit
def calculate_capacity(snr: float, bandwidth: float) -> float:
    """Calculate the capacity of a channel using the Shannon formula.
    
    args:
        snr (float): The signal-to-noise ratio in dB.
        bandwidth (float): The bandwidth of the channel in Hz.
    """
    # Calculate the capacity using the Shannon formula
    capacity = bandwidth * jnp.log2(1 + jnp.power(10,snr/10))

    return capacity

