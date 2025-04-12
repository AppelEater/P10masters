from Project_library import spherical_to_cartesian, generate_latitude_longitude_points, three_dimensional_plot_latitude_longitude_points

def test_spherical_to_cartesian():
    longitude = 0.0
    latitude = 0.0
    expected_x = 1.0
    expected_y = 0.0
    expected_z = 0.0

    x, y, z = spherical_to_cartesian(latitude, longitude)
    assert x == expected_x, f"Expected {expected_x}, but got {x}"