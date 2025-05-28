import gurobipy as gp
from gurobipy import GRB
import jax.random as jrandom
import jax.numpy as jnp
import jax
import Project_library as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

##### Optimisation and logger configuration #####
node_limit = 1000 # Number of iterations for the simulation 
configuration_period = 30 # Configuration period in seconds
time_step = 0.25 # Time step in seconds
SHOW_FIGURES = False# Show figures or not
PRINT_RESULTS = False# Print results or not

earth_radius = 6371 # km

##### Sweep paramters #####
sweeping_parameters = {
    "reconfig_period" : [1],
    "time_step" : [0.001],
    "rmin" : [0, 50000, 10000],
    "iterations" : 10 # Number of iterations for the sweep
}


##### Define the satellite parameters #####
satellite_height = 600 # km
satellite_transmit_power_per_beam = 75.4 # W
beam_bandwidth = 30e6 # Hz
beam_gain = 30 # dBi   
sky_temperature = 290 # K
satellite_central_frequency = 2 # GHz
minimum_observation_angle = 40 # degrees
cell_size = 50 # km
satellite_position = jnp.asarray([earth_radius + satellite_height, 0, 0]) # km
satellite_spherical_coordinates = pl.cartesian_to_spherical(*satellite_position) # km
number_of_beams = 19 # Number of beams


##### User paramters #####
probability_vector_for_high_and_low_density = [0.2, 0.8] # Probability of high and low density. [p_high, p_low] 
user_turn_on_probability = 0.8 # Probability of user turning on from off state
user_turn_off_probability = 0.2 # Probability of user turning off from on state
start_of_on_state = 0.6
user_state_transition_probability = [[1-user_turn_on_probability, user_turn_on_probability],
                                      [user_turn_off_probability, 1-user_turn_off_probability]] # Transition probability matrix
user_state_transition_probability = np.array(user_state_transition_probability)
high_density_user_amount = 50 # Number of users in high density
low_density_user_amount = 10 # Number of users in low density
user_density = [high_density_user_amount, low_density_user_amount] # Number of users in high and low density
user_density = np.array(user_density)
base_user_demand = 100e3

##### Some asserts #####
assert probability_vector_for_high_and_low_density[0] + probability_vector_for_high_and_low_density[1] == 1, "The sum of the probabilities must be equal to 1"
assert user_state_transition_probability[0][0] + user_state_transition_probability[0][1] == 1, "The sum of the probabilities must be equal to 1"
assert user_state_transition_probability[1][0] + user_state_transition_probability[1][1] == 1, "The sum of the probabilities must be equal to 1"
print("All asserts passed.\n\n")

##### Print the parameters when running the script #####
print("Satellite parameters used in the simulation:")
print("Satellite height:            ", satellite_height, "km")
print("Satellite transmit power:    ", satellite_transmit_power_per_beam, "W")
print("Beam bandwidth:              ", beam_bandwidth, "MHz")
print("Beam gain:                   ", beam_gain, "dBi")
print("Sky temperature:             ", sky_temperature, "K")
print("Satellite central frequency: ", satellite_central_frequency, "GHz")
print("Minimum observation angle:   ", minimum_observation_angle, "degrees")
print("Base user demand:            ", base_user_demand, "bps")
print("Number of beams:             ", number_of_beams, "beams")

def optimise_allocation_of_beams(satellite_position, satellite_transmit_power_per_beam, beam_gain, beam_bandwidth, configuration_period, time_step, rmin, key):
    """
    Optimise the allocation of beams to cells.
    
    Args:
        satellite_position (jnp.ndarray): The position of the satellite in Cartesian coordinates """
    ##### Intialise the cell area based on the cell size and satellite parameters #####

    # Calculate the visble latitude and longitude range of the satellite

    lat_range, lon_range, _, _,_, angle_of_interest = pl.visible_angle(jnp.deg2rad(minimum_observation_angle), satellite_position)

    earth_angle_cell_size = cell_size / (earth_radius) # radians
    # Calculate the number of cells in the grid
    num_cells_lat = int((lat_range[1] - lat_range[0]) /earth_angle_cell_size)
    num_cells_lon = int((lon_range[1] - lon_range[0]) /earth_angle_cell_size)

    # Create a grid of cells
    mesh_grid = pl.generate_latitude_longitude_points(num_cells_lat, num_cells_lon, lat_range, lon_range)

    # Recover list of lat and lon points
    lat_points, long_points = mesh_grid[0][:,0], mesh_grid[1][0,:]

    # Check which cells are visible from the satellite (It generates a mask)
    visibility_mask = pl.calculate_if_cells_within_visible_area(jnp.rad2deg(lat_points),jnp.rad2deg(long_points), 0, 0, 6000, jnp.rad2deg(angle_of_interest))

    mesh_grid_comb = jnp.stack(mesh_grid, axis=-1)
    mesh_grid_comb.shape

    print("Generating cells with users...")
    # Generate the cells with users
    cells = []
    for i in range(mesh_grid_comb.shape[0]):
        cells_row = []
        for j in range(mesh_grid_comb.shape[1]):
            
            density = jrandom.choice(key, jnp.array(user_density), p=jnp.array(probability_vector_for_high_and_low_density))

            key, subkey = jrandom.split(key)
            user_list = []
            #print([pl.User(mesh_grid_comb[i,j,0], mesh_grid_comb[i,j,1], x, jrandom.choice(key, jnp.array([0,1]), p=jnp.array([0.1,0.9]))) for x in range(density)])
            for x in range(density):
                key, subkey = jrandom.split(key)
                #print(jrandom.choice(key, jnp.array([0,1]), p=jnp.array([0.1,0.9])))
                user_list.append(pl.User(mesh_grid_comb[i,j,0], mesh_grid_comb[i,j,1], x, jrandom.choice(key, jnp.array([0,1])*base_user_demand, p=jnp.array([1-start_of_on_state,start_of_on_state]))))
            
            cells_row.append(pl.square_cell(
                lat = mesh_grid_comb[i,j,0], 
                longi = mesh_grid_comb[i,j,1],
                lat_width=[],
                longi_width=[],
                density=density,
                id=i*mesh_grid_comb.shape[1]+j,
                users= user_list,
                users_amount=density
            ))

            key, subkey = jrandom.split(key)
        cells.append(cells_row)


    # Calculate the distance from the satellite to the center of the earth
    print()
    print("Calculating the distance from the satellite to the cells and demand from the cells...")
    distance = []
    initial_demand = []
    for x in range(len(cells)):
        for y in range(len(cells[x])):
            if visibility_mask[x,y]:
                position = pl.spherical_to_cartesian(earth_radius, cells[x][y].lat, cells[x][y].longi)
                distance.append(pl.calculate_distance(position[0], position[1],position[2], satellite_position))
                initial_demand.append(pl.calculate_demand_of_cell(cells[x][y]))

    # Calculate the predicted demand of the cells
      


    distance = jnp.asarray(distance)

    # Calculate the rates
    calculate_multiple_snr = jax.vmap(pl.calculate_snr, in_axes=(None,None,None,0,None))

    snr = calculate_multiple_snr(satellite_transmit_power_per_beam, satellite_central_frequency, beam_bandwidth, distance, 0)

    rates = jax.vmap(pl.calculate_capacity, in_axes=(0,None))(snr, beam_bandwidth)


    R = np.array(rates)
    D = np.array(initial_demand)

    T = int(configuration_period/time_step)
    I = list(range(T)) 
    B = number_of_beams 
    K = list(range(len(distance)))

    problem = gp.Model("Satellite_optimization")

    t = problem.addVar(name="t", lb=0, vtype=gp.GRB.CONTINUOUS)
    x = problem.addVars(I, K, vtype=gp.GRB.BINARY, name="x")

    # Objective
    problem.setObjective(t, gp.GRB.MAXIMIZE)

    # precompute the weight for each cell k
    weights = {k: R[k]/T for k in K}

    
    # Constraints
    for k in K:
        problem.addConstr(t * jnp.maximum(D[k], rmin) <= weights[k] * gp.quicksum(x[i, k] for i in I),
                        name=f"Demand Constraint {k}")
    for i in I:
        problem.addConstr(B >= gp.quicksum(x[i, k] for k in K) ,
                        name=f"beam_capacity_time_{i}")
        

    # Gurobi parameters
    problem.Params.MIPGap    = 1e-4     # tighten or loosen tolerance
    problem.Params.TimeLimit = 600      # seconds
    problem.Params.OutputFlag = 1       # suppress output
    problem.Params.Threads = 0 # use all available threads
    problem.Params.MIPGap = 0.005

    print("Starting Optimisation")

    problem.optimize()

    print("Optimisation finished with status: ", problem.Status)

    schedule = np.zeros((len(I), len(K)))  # rows = time steps, columns = cells
    # Extract solution, if there are multiple solution, take the first one 
    if problem.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SUBOPTIMAL}:
        for i_idx, i in enumerate(I):
            for k_idx, k in enumerate(K):
                if x[i, k].X > 0.5:
                    schedule[i_idx, k_idx] = 1
    else:
        print("No feasible solution found.")

    print("Rates : ", rates)


    if SHOW_FIGURES:
        plt.figure(figsize=(12, 6))
        plt.imshow(schedule.T, aspect='auto', cmap='Greys', interpolation='none')
        plt.xlabel("Time index (i)")
        plt.ylabel("Cell ID (k)")
        plt.title("Beam allocation schedule")
        plt.colorbar(label="Allocated (1 = yes)")
        plt.show()


    print("Calculating the measures of the optimisation...")
    ##### Calculate the measures of the optimisation #####
    # Calculate the minimum capacity to demand
    schedule = jnp.asarray(schedule)
    demand = jnp.asarray(initial_demand)
    achieved_capacity = jnp.multiply(jnp.sum(schedule, axis=0)/T, rates)
    min_capacity = jnp.min(achieved_capacity)
    avg_capacity = jnp.mean(achieved_capacity)
    min_CD = jnp.min(jnp.divide(achieved_capacity, demand))
    avg_CD = jnp.mean(jnp.divide(achieved_capacity, demand))

    print("Calculating dismetrics...")
    # Dismetrics
    unmet_demand = jnp.maximum(jnp.zeros_like(demand), demand - achieved_capacity)
    Average_unmet_demand = jnp.mean(unmet_demand)
    maximum_unmet_demand = jnp.max(unmet_demand)

    # Calculate the disconnect time
    print("Calculating disconnect time...")
    disconnect_times = pl.disconnect_times(schedule.T, time_step)
    
         
    for i in range(schedule.shape[0]):
        pass
    if PRINT_RESULTS:
        print("Achieved capacity:                ", achieved_capacity)
        print("Minimum capacity:                 ", min_capacity)
        print("Average capacity to demand:       ", avg_capacity)
        print("Minimum capacity to demand ratio: ", min_CD)
        print("Average capacity to demand ratio: ", avg_CD)
        print("Average unmet demand:             ", Average_unmet_demand)
        print("Maximum unmet demand:             ", maximum_unmet_demand)
        print("Disconnect time:                  ", disconnect_times)

    return demand, schedule, min_capacity, avg_capacity, min_CD, avg_CD, Average_unmet_demand, maximum_unmet_demand, disconnect_times

if __name__ == "__main__":    

    key = jrandom.PRNGKey(11)

    # What to pickle
    pickle_data = {}
    pickle_data["Config"] = sweeping_parameters

    with open(f"results_sweep{int(time.time())}", "ab") as f: 
        # Run the optimisation,
        pkl.dump(pickle_data, f)

        for reconfig_period in sweeping_parameters["reconfig_period"]:
            for time_step in sweeping_parameters["time_step"]: 
                for rmin in sweeping_parameters["rmin"]:
                    
                    con_sweep = {"reconfig_period": reconfig_period, "time_step": time_step, "rmin": rmin}
                    data = []
                                        
                    for iter in range(sweeping_parameters["iterations"]):

                        key, subkey = jrandom.split(key)    
                        demand, schedule, min_capacity, avg_capacity, min_CD, avg_CD, Average_unmet_demand, maximum_unmet_demand, disconnect_times = optimise_allocation_of_beams(satellite_position,
                                                     satellite_transmit_power_per_beam,
                                                       beam_gain,
                                                         beam_bandwidth,
                                                           reconfig_period,
                                                             time_step,
                                                               rmin,
                                                                 key)
                    
                        # Save the results
                        data.append({
                            "demand": demand,
                            "schedule": schedule,
                            "min_capacity": min_capacity,
                            "avg_capacity": avg_capacity,
                            "min_CD": min_CD,
                            "avg_CD": avg_CD,
                            "Average_unmet_demand": Average_unmet_demand,
                            "maximum_unmet_demand": maximum_unmet_demand,
                            "disconnect_time": disconnect_times
                        })
                
                    con_sweep["data"] = data
                    # Save the results
                    pkl.dump(con_sweep, f)


    plt.figure(figsize=(10, 6))
    plt.hist(demand, bins=20)
    print("Print the sum of beams over time",jnp.sum(schedule, axis=1))

    # Plot histogram of the disconnect time
    plt.figure(figsize=(10, 6)) 
    plt.hist(disconnect_times, bins=20)
    plt.xlabel("Disconnect time (s)")
    plt.ylabel("Frequency")
    plt.title("Disconnect time histogram")
    plt.show()