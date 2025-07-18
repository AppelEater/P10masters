from Project_library import *

import jax 
import jax.numpy as jnp

def test_spherical_to_cartesian():
	radius = 1.0
	longitude = 0.0
	latitude = 0.0
	expected_x = 1.0
	expected_y = 0.0
	expected_z = 0.0

	x, y, z = spherical_to_cartesian(radius,latitude, longitude)
	assert jnp.isclose(x,expected_x).item(), f"Expected {expected_x}, but got {x}"

	x,y,z = spherical_to_cartesian(radius,jnp.pi/2, longitude)
	assert jnp.isclose(x, 0, atol = 1e-7).item(), f"Expected {0.0}, but got {x}"
	assert jnp.isclose(y, 0, atol = 1e-7).item() , f"Expected {0.0}, but got {y}"
	assert jnp.isclose(z, 1, atol = 1e-7).item(), f"Expected {1.0}, but got {z}"


def test_generate_latitude_longitude_points():
	NumberOfPointsAlonglatitude = 3
	NumberOfPointsAlonglongitude = 3
	cell_mesh = generate_latitude_longitude_points(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	assert cell_mesh[0].shape == (NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude), "Latitude meshgrid shape mismatch"
	assert cell_mesh[1].shape == (NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude), "Longitude meshgrid shape mismatch"

	assert cell_mesh[0][1][1] == 0, "Latitude meshgrid value mismatch"

	print(cell_mesh[0])
	print(cell_mesh[1])

	cell_border =(cell_mesh[0][0][0]-cell_mesh[0][1][0])/2

	difference_to_cell_edge = (-jnp.pi/2-cell_mesh[0][0][0])

	# Determine that the meshgrid have the proper distance between them.
	assert jnp.isclose(cell_border, difference_to_cell_edge).item(), "Latitude meshgrid value mismatch"

	# check in the longitude direction
	cell_border =(cell_mesh[1][0][0]-cell_mesh[1][0][1])/2

	difference_to_cell_edge = (-jnp.pi-cell_mesh[1][0][0])

	assert jnp.isclose(cell_border, difference_to_cell_edge).item(), "Longitude meshgrid value mismatch"




def test_generate_latitude_longitude_divisions():
	NumberOfPointsAlonglatitude = 10
	NumberOfPointsAlonglongitude = 10
	cell_mesh = generate_latitude_longitude_points(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	assert cell_mesh[0].shape == (NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude), "Latitude meshgrid shape mismatch"
	assert cell_mesh[1].shape == (NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude), "Longitude meshgrid shape mismatch"

	# Generate the latitude and longitude divisions
	lat_divsion, longi_division = generate_latitude_longitude_divisions(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	# Check the shape of the divisions, the lat division should have one less point than the NumberOfPointsAlonglatitude
	# As the divisions between a set of points will be less than the points themselves by one.
	assert lat_divsion.shape == (NumberOfPointsAlonglatitude+1,), "Latitude divisions shape mismatch"    
	assert longi_division.shape == (NumberOfPointsAlonglongitude+1,), "Longitude divisions shape mismatch"

	# Check wheter the divsions are at the half point between the meshgrid points
	cell_mesh_lat = cell_mesh[0]
	cell_mesh_long = cell_mesh[1]

	expected_lat_division = (cell_mesh_lat[0][0] + cell_mesh_lat[1][0]) / 2
	assert jnp.isclose(expected_lat_division, lat_divsion[1]).item(), "Latitude divisions mismatch"
	# Grab the second division, as this the first is the edge of coverage.


def test_calculate_area_on_sphere():
	# Calculate the area of an entire sphere to compare
	# The area of a sphere is given by the formula 4 * pi * r^2
	# Assuming a unit sphere (r = 1)
	# area = 4 * jnp.pi * (1**2)
	area = 4 * jnp.pi

	radius = 1.0

	calc_area = calculate_area_on_sphere(1,jnp.asarray([-jnp.pi/2,jnp.pi/2]), jnp.asarray([-jnp.pi, jnp.pi]))

	print(calc_area, area)

	assert jnp.isclose(calc_area, area).item(), "Area calculation mismatch"
	# Half sphere 
	
	calc_area = calculate_area_on_sphere(1,jnp.asarray([-jnp.pi/2,0]), jnp.asarray([-jnp.pi, jnp.pi]))
	assert jnp.isclose(calc_area, area/2).item(), "Area calculation mismatch"

def test_locate_user_cell():
	# Split the sphere into nine cells
	NumberOfPointsAlonglatitude = 3
	NumberOfPointsAlonglongitude = 3

	mesh_grid = generate_latitude_longitude_points(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	lat_divisions, long_divisions = generate_latitude_longitude_divisions(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	# Test the function with a user located at the center of the first cell
	lat_user = mesh_grid[0][0][0]
	long_user = mesh_grid[1][0][0]
	print(lat_divisions, long_divisions)
	print(lat_user, long_user)
	lat_index, long_index = locate_user_cell(lat_user, long_user, lat_divisions, long_divisions)

	assert lat_index == 0, "Latitude index mismatch"
	assert long_index == 0, "Longitude index mismatch"

	# Test the function with a user located at the center of the middle cell
	lat_user = mesh_grid[0][1][1]
	long_user = mesh_grid[1][1][1]
	lat_index, long_index = locate_user_cell(lat_user, long_user, lat_divisions, long_divisions)

	assert lat_index == 1, "Latitude index mismatch"
	assert long_index == 1, "Longitude index mismatch"
	
	# Test the function with a user located at the center of the last cell
	lat_user = mesh_grid[0][2][2]
	long_user = mesh_grid[1][2][2]
	lat_index, long_index = locate_user_cell(lat_user, long_user, lat_divisions, long_divisions)

	assert lat_index == 2, "Latitude index mismatch"
	assert long_index == 2, "Longitude index mismatch"

def test_calculate_cells_within_area():
	# Split the sphere into nine cells
	NumberOfPointsAlonglatitude = 3
	NumberOfPointsAlonglongitude = 3

	mesh_grid = generate_latitude_longitude_points(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	lat_divisions, long_divisions = generate_latitude_longitude_divisions(NumberOfPointsAlonglatitude, NumberOfPointsAlonglongitude)

	# Test the function with a user located at the center of the first cell
	lat_user = mesh_grid[0][0][0]
	long_user = mesh_grid[1][0][0]

	# Calculate the area of the first cell
	cell_area = calculate_area_on_sphere(1,jnp.asarray([lat_divisions[0], lat_divisions[1]]), jnp.asarray([long_divisions[0], long_divisions[1]]))

	sat_pos = jnp.array([0,0,600+6371])
	_, _ ,_ , _, _, alpha= visible_angle(jnp.deg2rad(30), sat_pos)

	#print(mesh_grid)

	# Satellite position to spherical coordinates
	r, lat, long = cartesian_to_spherical(*sat_pos)

	cell_lat_list = jnp.rad2deg(mesh_grid[0].T[0]) 
	cell_long_list = jnp.rad2deg(mesh_grid[1][0]) 
	lat = jnp.rad2deg(lat-jnp.pi/2)
	long = jnp.rad2deg(long)
	alpha = jnp.rad2deg(alpha)


	cells_within_area = calculate_if_cells_within_visible_area(cell_lat_list,cell_long_list, lat, long, 600,alpha)
	assert cells_within_area.shape == (3,3), "Cells within area shape mismatch"
	print(lat, long)
	print(alpha)
	print(cell_lat_list, cell_long_list)
	print(cells_within_area)
	assert cells_within_area[1][1] == 1, "Cells within area value mismatch"
	assert cells_within_area[0][0] == 0, "Cells within area value mismatch"


def test_prediction():
	# Test the prediction function

	# Create a dummy set of users
	s = jnp.array([0,0,1,0])

	A = jnp.array([[0.8, 0.2], [0.1, 0.9]])

	O = 3 # Number of time‐steps

	N = s.shape[0] # Number of users

	# Call the prediction function
	predicted_probabilties = prediction_of_activity(s, A, O, N)

	assert predicted_probabilties.shape == (O, N+1), "Prediction shape mismatch"