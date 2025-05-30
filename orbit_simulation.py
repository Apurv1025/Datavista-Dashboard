# Orbit Simulation Module
# Uses Skyfield to simulate satellite orbits and calculate positions

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from skyfield.api import load, wgs84
from skyfield.sgp4lib import EarthSatellite
import math

def create_satellite_from_tle(satellite_data, ts=None):
    """
    Create a Skyfield EarthSatellite object from TLE data
    
    Args:
        satellite_data (dict): Dictionary with name, line1, line2
        ts (Timescale): Skyfield timescale object
    
    Returns:
        EarthSatellite: Skyfield satellite object
    """
    
    if ts is None:
        ts = load.timescale()
    
    satellite = EarthSatellite(
        satellite_data['line1'],
        satellite_data['line2'],
        satellite_data['name'],
        ts
    )
    
    return satellite

def calculate_orbit_positions(satellite_data, start_date, end_date, time_step_minutes=10):
    """
    Calculate satellite positions over a time period
    
    Args:
        satellite_data (dict): TLE data for the satellite
        start_date (datetime.date): Start date for calculations
        end_date (datetime.date): End date for calculations
        time_step_minutes (int): Time step in minutes
    
    Returns:
        dict: Orbit position data
    """
    
    ts = load.timescale()
    satellite = create_satellite_from_tle(satellite_data, ts)
    
    # Create time array
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    time_points = []
    current_time = start_dt
    
    while current_time <= end_dt:
        time_points.append(current_time)
        current_time += timedelta(minutes=time_step_minutes)
    
    # Convert to Skyfield time format
    t = ts.utc([tp.year for tp in time_points],
              [tp.month for tp in time_points],
              [tp.day for tp in time_points],
              [tp.hour for tp in time_points],
              [tp.minute for tp in time_points],
              [tp.second for tp in time_points])
    
    # Calculate positions
    geocentric = satellite.at(t)
    
    # Get position in km
    position_km = geocentric.position.km
    
    # Get lat/lon/alt
    subpoint = wgs84.subpoint(geocentric)
    
    # Calculate velocity (approximate)
    if len(t) > 1:
        dt = (t[1].tt - t[0].tt) * 86400  # Convert to seconds
        velocity_km_s = np.gradient(position_km, dt, axis=1)
    else:
        velocity_km_s = np.zeros_like(position_km)
    
    orbit_data = {
        'times': time_points,
        'positions_km': position_km,
        'velocities_km_s': velocity_km_s,
        'latitudes_deg': subpoint.latitude.degrees,
        'longitudes_deg': subpoint.longitude.degrees,
        'altitudes_km': subpoint.elevation.km,
        'satellite_name': satellite_data['name']
    }
    
    return orbit_data

def simulate_conjunction(primary_satellite, secondary_satellite, analysis_time):
    """
    Simulate a conjunction between two satellites
    
    Args:
        primary_satellite (dict): Primary satellite TLE data
        secondary_satellite (dict): Secondary satellite TLE data
        analysis_time (datetime): Time for conjunction analysis
    
    Returns:
        dict: Conjunction analysis results
    """
    
    ts = load.timescale()
    
    # Create satellite objects
    sat1 = create_satellite_from_tle(primary_satellite, ts)
    sat2 = create_satellite_from_tle(secondary_satellite, ts)
    
    # Convert analysis time to Skyfield format
    t = ts.utc(analysis_time.year, analysis_time.month, analysis_time.day,
               analysis_time.hour, analysis_time.minute, analysis_time.second)
    
    # Calculate positions at analysis time
    pos1 = sat1.at(t)
    pos2 = sat2.at(t)
    
    # Calculate relative position
    relative_pos = pos2 - pos1
    distance_km = relative_pos.distance().km
    
    # Calculate relative velocity (approximate)
    dt = 1.0  # 1 second
    t_next = ts.tt_jd(t.tt + dt/86400)
    
    pos1_next = sat1.at(t_next)
    pos2_next = sat2.at(t_next)
    
    vel1 = (pos1_next.position.km - pos1.position.km) / dt
    vel2 = (pos2_next.position.km - pos2.position.km) / dt
    relative_velocity = np.linalg.norm(vel2 - vel1)
    
    conjunction_data = {
        'time': analysis_time,
        'distance_km': distance_km,
        'relative_velocity_km_s': relative_velocity,
        'primary_position': pos1.position.km,
        'secondary_position': pos2.position.km,
        'primary_name': primary_satellite['name'],
        'secondary_name': secondary_satellite['name']
    }
    
    return conjunction_data

def find_closest_approaches(primary_satellite, secondary_satellite, start_date, end_date, time_step_minutes=10):
    """
    Find closest approaches between two satellites over a time period
    
    Args:
        primary_satellite (dict): Primary satellite TLE data
        secondary_satellite (dict): Secondary satellite TLE data
        start_date (datetime.date): Start date for search
        end_date (datetime.date): End date for search
        time_step_minutes (int): Time step in minutes
    
    Returns:
        list: List of closest approach events
    """
    
    ts = load.timescale()
    
    sat1 = create_satellite_from_tle(primary_satellite, ts)
    sat2 = create_satellite_from_tle(secondary_satellite, ts)
    
    # Create time array
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    time_points = []
    current_time = start_dt
    
    while current_time <= end_dt:
        time_points.append(current_time)
        current_time += timedelta(minutes=time_step_minutes)
    
    # Convert to Skyfield time format
    t = ts.utc([tp.year for tp in time_points],
              [tp.month for tp in time_points],
              [tp.day for tp in time_points],
              [tp.hour for tp in time_points],
              [tp.minute for tp in time_points],
              [tp.second for tp in time_points])
    
    # Calculate relative distances
    pos1 = sat1.at(t)
    pos2 = sat2.at(t)
    
    distances = []
    for i in range(len(t)):
        rel_pos = pos2.position.km[:, i] - pos1.position.km[:, i]
        distance = np.linalg.norm(rel_pos)
        distances.append(distance)
    
    # Find local minima (closest approaches)
    closest_approaches = []
    
    for i in range(1, len(distances) - 1):
        if distances[i] < distances[i-1] and distances[i] < distances[i+1]:
            # This is a local minimum
            approach = {
                'time': time_points[i],
                'miss_distance_km': distances[i],
                'primary_satellite': primary_satellite['name'],
                'secondary_satellite': secondary_satellite['name']
            }
            closest_approaches.append(approach)
    
    return closest_approaches

def calculate_orbital_elements(satellite_data):
    """
    Calculate current orbital elements from TLE data
    
    Args:
        satellite_data (dict): TLE data
    
    Returns:
        dict: Orbital elements
    """
    
    from data_acquisition import parse_tle_data
    
    elements = parse_tle_data(satellite_data['line1'], satellite_data['line2'])
    
    # Calculate additional derived parameters
    mu_earth = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    # Semi-major axis from mean motion
    n = elements['mean_motion'] * 2 * np.pi / 86400  # Convert to rad/s
    a = (mu_earth / (n**2))**(1/3)  # Semi-major axis in km
    
    # Apogee and perigee
    e = elements['eccentricity']
    apogee = a * (1 + e) - 6371  # Altitude above Earth surface
    perigee = a * (1 - e) - 6371  # Altitude above Earth surface
    
    # Orbital period
    period_seconds = 2 * np.pi * np.sqrt(a**3 / mu_earth)
    period_minutes = period_seconds / 60
    
    orbital_elements = {
        'semi_major_axis_km': a,
        'apogee_altitude_km': apogee,
        'perigee_altitude_km': perigee,
        'period_minutes': period_minutes,
        'inclination_deg': elements['inclination'],
        'eccentricity': elements['eccentricity'],
        'right_ascension_deg': elements['right_ascension'],
        'argument_of_perigee_deg': elements['argument_of_perigee'],
        'mean_anomaly_deg': elements['mean_anomaly'],
        'mean_motion_rev_per_day': elements['mean_motion']
    }
    
    return orbital_elements

def propagate_orbit_uncertainty(satellite_data, time_hours, position_uncertainty_km=1.0):
    """
    Propagate orbital uncertainty over time
    
    Args:
        satellite_data (dict): TLE data
        time_hours (float): Time to propagate (hours)
        position_uncertainty_km (float): Initial position uncertainty
    
    Returns:
        dict: Uncertainty propagation results
    """
    
    # Simple uncertainty propagation model
    # In reality, this would involve complex covariance matrix propagation
    
    orbital_elements = calculate_orbital_elements(satellite_data)
    period_hours = orbital_elements['period_minutes'] / 60
    
    # Uncertainty grows with time and orbital characteristics
    time_factor = time_hours / period_hours
    eccentricity_factor = 1 + orbital_elements['eccentricity']
    
    # Simple model: uncertainty grows roughly quadratically with time
    position_uncertainty = position_uncertainty_km * (1 + 0.1 * time_factor + 0.01 * time_factor**2) * eccentricity_factor
    velocity_uncertainty = position_uncertainty * 0.001  # km/s, rough estimate
    
    uncertainty_data = {
        'time_hours': time_hours,
        'position_uncertainty_km': position_uncertainty,
        'velocity_uncertainty_km_s': velocity_uncertainty,
        'position_uncertainty_1sigma': position_uncertainty,
        'position_uncertainty_3sigma': position_uncertainty * 3
    }
    
    return uncertainty_data

def calculate_ground_track(satellite_data, start_date, end_date, time_step_minutes=5):
    """
    Calculate satellite ground track (subsatellite points)
    
    Args:
        satellite_data (dict): TLE data
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        time_step_minutes (int): Time step in minutes
    
    Returns:
        dict: Ground track data
    """
    
    orbit_data = calculate_orbit_positions(satellite_data, start_date, end_date, time_step_minutes)
    
    ground_track = {
        'times': orbit_data['times'],
        'latitudes': orbit_data['latitudes_deg'],
        'longitudes': orbit_data['longitudes_deg'],
        'altitudes': orbit_data['altitudes_km'],
        'satellite_name': satellite_data['name']
    }
    
    return ground_track

# Example usage and testing
if __name__ == "__main__":
    from data_acquisition import SAMPLE_TLE_DATA
    
    # Test orbit calculation
    iss_data = SAMPLE_TLE_DATA["ISS (ZARYA)"]
    start_date = datetime.now().date()
    end_date = start_date + timedelta(days=1)
    
    print("Testing orbit calculation...")
    orbit_data = calculate_orbit_positions(iss_data, start_date, end_date)
    print(f"Calculated {len(orbit_data['times'])} orbit points")
    
    # Test orbital elements calculation
    print("\nTesting orbital elements...")
    elements = calculate_orbital_elements(iss_data)
    print(f"Semi-major axis: {elements['semi_major_axis_km']:.2f} km")
    print(f"Period: {elements['period_minutes']:.2f} minutes")
    
    # Test conjunction simulation
    print("\nTesting conjunction simulation...")
    starlink_data = SAMPLE_TLE_DATA["STARLINK-1007"]
    conjunction = simulate_conjunction(iss_data, starlink_data, datetime.now())
    print(f"Distance: {conjunction['distance_km']:.2f} km")