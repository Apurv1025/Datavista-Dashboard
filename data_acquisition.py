# Data Acquisition Module
# Handles TLE and CDM data fetching and parsing

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

# Sample TLE data for demo purposes
SAMPLE_TLE_DATA = {
    "ISS (ZARYA)": {
        "name": "ISS (ZARYA)",
        "line1": "1 25544U 98067A   21123.45678901  .00002182  00000-0  40864-4 0  9992",
        "line2": "2 25544  51.6461 339.0518 0003068  83.4820 276.7047 15.48919103281553"
    },
    "STARLINK-1007": {
        "name": "STARLINK-1007",
        "line1": "1 44713U 19074A   21123.12345678  .00001234  00000-0  12345-4 0  9998",
        "line2": "2 44713  53.0534 123.4567 0001234  45.6789 314.5432 15.06123456123456"
    },
    "COSMOS 2251 DEB": {
        "name": "COSMOS 2251 DEB",
        "line1": "1 34454U 93036SX  21123.87654321  .00000567  00000-0  56789-5 0  9997",
        "line2": "2 34454  74.0123 234.5678 0012345 123.4567 236.7890 14.12345678234567"
    }
}

def fetch_tle_data(satellite_name, use_spacetrack=False):
    """
    Fetch TLE data for a specific satellite
    
    Args:
        satellite_name (str): Name of the satellite
        use_spacetrack (bool): Whether to use Space-Track.org API (requires credentials)
    
    Returns:
        dict: TLE data with name, line1, line2
    """
    
    if not use_spacetrack or satellite_name in SAMPLE_TLE_DATA:
        # Use sample data for demo
        return SAMPLE_TLE_DATA.get(satellite_name, SAMPLE_TLE_DATA["ISS (ZARYA)"])
    
    # In a real implementation, you would fetch from Space-Track.org
    # This requires authentication and proper API calls
    try:
        # Placeholder for Space-Track API integration
        # You would need to register at https://www.space-track.org/
        # and implement proper authentication
        
        # For now, return sample data
        return SAMPLE_TLE_DATA.get(satellite_name, SAMPLE_TLE_DATA["ISS (ZARYA)"])
        
    except Exception as e:
        print(f"Error fetching TLE data: {e}")
        return SAMPLE_TLE_DATA["ISS (ZARYA)"]

def parse_tle_data(tle_line1, tle_line2):
    """
    Parse TLE data and extract orbital parameters
    
    Args:
        tle_line1 (str): First line of TLE
        tle_line2 (str): Second line of TLE
    
    Returns:
        dict: Parsed orbital parameters
    """
    
    # Parse Line 1
    satellite_number = int(tle_line1[2:7])
    classification = tle_line1[7]
    intl_designator = tle_line1[9:17].strip()
    epoch_year = int(tle_line1[18:20])
    epoch_day = float(tle_line1[20:32])
    first_derivative = float(tle_line1[33:43])
    second_derivative = float(tle_line1[44:50]) if tle_line1[44:52].strip() else 0.0
    bstar = float(tle_line1[53:59]) if tle_line1[53:61].strip() else 0.0
    ephemeris_type = int(tle_line1[62])
    element_number = int(tle_line1[64:68])
    
    # Parse Line 2
    inclination = float(tle_line2[8:16])
    right_ascension = float(tle_line2[17:25])
    eccentricity = float("0." + tle_line2[26:33])
    argument_of_perigee = float(tle_line2[34:42])
    mean_anomaly = float(tle_line2[43:51])
    mean_motion = float(tle_line2[52:63])
    revolution_number = int(tle_line2[63:68])
    
    return {
        'satellite_number': satellite_number,
        'classification': classification,
        'intl_designator': intl_designator,
        'epoch_year': epoch_year,
        'epoch_day': epoch_day,
        'inclination': inclination,
        'right_ascension': right_ascension,
        'eccentricity': eccentricity,
        'argument_of_perigee': argument_of_perigee,
        'mean_anomaly': mean_anomaly,
        'mean_motion': mean_motion,
        'revolution_number': revolution_number,
        'first_derivative': first_derivative,
        'second_derivative': second_derivative,
        'bstar': bstar
    }

def generate_sample_cdm_data():
    """
    Generate sample CDM (Conjunction Data Message) data for demonstration
    
    Returns:
        list: List of CDM dictionaries
    """
    
    # Generate realistic but synthetic CDM data
    cdm_data = []
    
    base_time = datetime.now()
    
    for i in range(10):
        cdm = {
            'message_id': f'CDM_{i+1:03d}',
            'creation_date': (base_time + timedelta(hours=i)).isoformat(),
            'emergency_reportable': False,
            'tca': (base_time + timedelta(days=i+1, hours=12)).isoformat(),
            'miss_distance': np.random.uniform(0.1, 50.0),  # km
            'relative_speed': np.random.uniform(1.0, 15.0),  # km/s
            'object1_name': 'PRIMARY_SATELLITE',
            'object2_name': f'DEBRIS_OBJECT_{i+1}',
            'collision_probability': np.random.uniform(1e-6, 1e-3),
            'position_covariance': {
                'xx': np.random.uniform(100, 1000),
                'yy': np.random.uniform(100, 1000),
                'zz': np.random.uniform(100, 1000),
                'xy': np.random.uniform(-500, 500),
                'xz': np.random.uniform(-500, 500),
                'yz': np.random.uniform(-500, 500)
            },
            'object1_position': {
                'x': np.random.uniform(-7000, 7000),
                'y': np.random.uniform(-7000, 7000),
                'z': np.random.uniform(-7000, 7000)
            },
            'object2_position': {
                'x': np.random.uniform(-7000, 7000),
                'y': np.random.uniform(-7000, 7000),
                'z': np.random.uniform(-7000, 7000)
            }
        }
        cdm_data.append(cdm)
    
    return cdm_data

def parse_cdm_data(cdm_file_path=None, cdm_text=None):
    """
    Parse CDM data from file or text
    
    Args:
        cdm_file_path (str): Path to CDM file
        cdm_text (str): CDM text content
    
    Returns:
        dict: Parsed CDM data
    """
    
    if cdm_file_path:
        try:
            with open(cdm_file_path, 'r') as f:
                cdm_content = f.read()
        except FileNotFoundError:
            # Return sample data if file not found
            return generate_sample_cdm_data()[0]
    elif cdm_text:
        cdm_content = cdm_text
    else:
        # Return sample data
        return generate_sample_cdm_data()[0]
    
    # Simple CDM parsing (in reality, this would be more complex)
    # For demonstration, return sample data
    return generate_sample_cdm_data()[0]

def get_sample_satellites():
    """
    Get sample satellite data for dashboard display
    
    Returns:
        pd.DataFrame: Sample satellite data
    """
    
    satellites = []
    for name, data in SAMPLE_TLE_DATA.items():
        parsed = parse_tle_data(data['line1'], data['line2'])
        satellites.append({
            'Name': name,
            'Catalog Number': parsed['satellite_number'],
            'Inclination (deg)': parsed['inclination'],
            'Eccentricity': parsed['eccentricity'],
            'Mean Motion (rev/day)': parsed['mean_motion'],
            'Epoch Year': parsed['epoch_year'],
            'Classification': parsed['classification']
        })
    
    return pd.DataFrame(satellites)

def download_celestrak_data(satellite_group='stations'):
    """
    Download TLE data from CelestTrak
    
    Args:
        satellite_group (str): Group of satellites to download
    
    Returns:
        list: List of TLE data
    """
    
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={satellite_group}&FORMAT=tle"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tle_lines = response.text.strip().split('\n')
            tle_data = []
            
            for i in range(0, len(tle_lines), 3):
                if i + 2 < len(tle_lines):
                    tle_data.append({
                        'name': tle_lines[i].strip(),
                        'line1': tle_lines[i+1].strip(),
                        'line2': tle_lines[i+2].strip()
                    })
            
            return tle_data
        else:
            print(f"Failed to download data: HTTP {response.status_code}")
            return []
    
    except Exception as e:
        print(f"Error downloading CelestTrak data: {e}")
        return []

def validate_tle_data(tle_line1, tle_line2):
    """
    Validate TLE data format and checksums
    
    Args:
        tle_line1 (str): First line of TLE
        tle_line2 (str): Second line of TLE
    
    Returns:
        bool: True if valid, False otherwise
    """
    
    def calculate_checksum(line):
        """Calculate TLE checksum"""
        checksum = 0
        for char in line[:-1]:  # Exclude the checksum digit
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return checksum % 10
    
    try:
        # Check line lengths
        if len(tle_line1) != 69 or len(tle_line2) != 69:
            return False
        
        # Check line numbers
        if tle_line1[0] != '1' or tle_line2[0] != '2':
            return False
        
        # Check satellite numbers match
        if tle_line1[2:7] != tle_line2[2:7]:
            return False
        
        # Check checksums
        if int(tle_line1[-1]) != calculate_checksum(tle_line1):
            return False
        
        if int(tle_line2[-1]) != calculate_checksum(tle_line2):
            return False
        
        return True
    
    except (ValueError, IndexError):
        return False

# Example usage and testing functions
if __name__ == "__main__":
    # Test TLE data fetching
    iss_data = fetch_tle_data("ISS (ZARYA)")
    print("ISS TLE Data:", iss_data)
    
    # Test TLE parsing
    parsed = parse_tle_data(iss_data['line1'], iss_data['line2'])
    print("Parsed TLE:", parsed)
    
    # Test CDM generation
    cdm_data = generate_sample_cdm_data()
    print(f"Generated {len(cdm_data)} CDM records")
    
    # Test TLE validation
    is_valid = validate_tle_data(iss_data['line1'], iss_data['line2'])
    print(f"TLE validation: {is_valid}")