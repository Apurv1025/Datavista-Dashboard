# Risk Assessment Module
# Machine learning models for collision risk assessment

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CollisionRiskAssessor:
    """
    Machine learning-based collision risk assessment system
    """
    
    def __init__(self):
        self.miss_distance_model = None
        self.collision_probability_model = None
        self.risk_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_training_data(self, n_samples=10000):
        """
        Generate synthetic training data for ML models
        
        Args:
            n_samples (int): Number of training samples to generate
        
        Returns:
            tuple: (features, miss_distances, collision_probs, risk_labels)
        """
        
        np.random.seed(42)  # For reproducibility
        
        # Generate features
        features = []
        miss_distances = []
        collision_probs = []
        risk_labels = []
        
        for i in range(n_samples):
            # Orbital parameters
            semi_major_axis = np.random.uniform(6700, 42000)  # km
            eccentricity = np.random.uniform(0, 0.8)
            inclination = np.random.uniform(0, 180)  # degrees
            relative_velocity = np.random.uniform(0.1, 15)  # km/s
            
            # Time to closest approach
            time_to_tca = np.random.uniform(0.1, 168)  # hours
            
            # Position uncertainties
            position_uncertainty_1 = np.random.uniform(0.1, 10)  # km
            position_uncertainty_2 = np.random.uniform(0.1, 10)  # km
            
            # Velocity uncertainties
            velocity_uncertainty_1 = np.random.uniform(0.001, 0.1)  # km/s
            velocity_uncertainty_2 = np.random.uniform(0.001, 0.1)  # km/s
            
            # Cross-correlation factor
            correlation_factor = np.random.uniform(-0.5, 0.5)
            
            # Object sizes (approximated as spheres)
            object_size_1 = np.random.uniform(0.1, 100)  # meters
            object_size_2 = np.random.uniform(0.1, 100)  # meters
            
            feature_vector = [
                semi_major_axis, eccentricity, inclination, relative_velocity,
                time_to_tca, position_uncertainty_1, position_uncertainty_2,
                velocity_uncertainty_1, velocity_uncertainty_2, correlation_factor,
                object_size_1, object_size_2
            ]
            
            # Calculate synthetic miss distance
            # Higher uncertainties and lower time to TCA lead to higher miss distances
            base_miss_distance = np.sqrt(position_uncertainty_1**2 + position_uncertainty_2**2)
            time_factor = 1 + 0.1 * np.log(time_to_tca + 1)
            velocity_factor = 1 + relative_velocity * 0.1
            
            miss_distance = base_miss_distance * time_factor * velocity_factor
            miss_distance += np.random.normal(0, miss_distance * 0.1)  # Add noise
            miss_distance = max(0.001, miss_distance)  # Ensure positive
            
            # Calculate collision probability
            combined_size = (object_size_1 + object_size_2) / 1000  # Convert to km
            combined_uncertainty = np.sqrt(position_uncertainty_1**2 + position_uncertainty_2**2)
            
            # Simple probability model based on miss distance and uncertainties
            if miss_distance < combined_size:
                collision_prob = 0.5 + 0.4 * np.random.random()
            else:
                prob_factor = combined_size / (miss_distance + combined_uncertainty)
                collision_prob = min(0.5, prob_factor * 0.1) * np.random.random()
            
            # Risk classification
            if collision_prob > 0.1 or miss_distance < 1.0:
                risk_label = 2  # High risk
            elif collision_prob > 0.01 or miss_distance < 5.0:
                risk_label = 1  # Medium risk
            else:
                risk_label = 0  # Low risk
            
            features.append(feature_vector)
            miss_distances.append(miss_distance)
            collision_probs.append(collision_prob)
            risk_labels.append(risk_label)
        
        return np.array(features), np.array(miss_distances), np.array(collision_probs), np.array(risk_labels)
    
    def train_models(self):
        """
        Train the machine learning models
        """
        
        # Generate training data
        features, miss_distances, collision_probs, risk_labels = self.generate_training_data()
        
        # Split data
        X_train, X_test, y_miss_train, y_miss_test = train_test_split(
            features, miss_distances, test_size=0.2, random_state=42
        )
        
        _, _, y_prob_train, y_prob_test = train_test_split(
            features, collision_probs, test_size=0.2, random_state=42
        )
        
        _, _, y_risk_train, y_risk_test = train_test_split(
            features, risk_labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train miss distance model
        self.miss_distance_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.miss_distance_model.fit(X_train_scaled, y_miss_train)
        
        # Train collision probability model
        self.collision_probability_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.collision_probability_model.fit(X_train_scaled, y_prob_train)
        
        # Train risk classifier
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.risk_classifier.fit(X_train_scaled, y_risk_train)
        
        # Evaluate models
        miss_pred = self.miss_distance_model.predict(X_test_scaled)
        prob_pred = self.collision_probability_model.predict(X_test_scaled)
        risk_pred = self.risk_classifier.predict(X_test_scaled)
        
        miss_rmse = np.sqrt(mean_squared_error(y_miss_test, miss_pred))
        prob_rmse = np.sqrt(mean_squared_error(y_prob_test, prob_pred))
        risk_accuracy = accuracy_score(y_risk_test, risk_pred)
        
        print(f"Miss Distance RMSE: {miss_rmse:.3f} km")
        print(f"Collision Probability RMSE: {prob_rmse:.6f}")
        print(f"Risk Classification Accuracy: {risk_accuracy:.3f}")
        
        self.is_trained = True
    
    def predict_collision_risk(self, feature_vector):
        """
        Predict collision risk for given features
        
        Args:
            feature_vector (list): Feature vector for prediction
        
        Returns:
            dict: Prediction results
        """
        
        if not self.is_trained:
            self.train_models()
        
        # Scale features
        features_scaled = self.scaler.transform([feature_vector])
        
        # Make predictions
        miss_distance = self.miss_distance_model.predict(features_scaled)[0]
        collision_prob = self.collision_probability_model.predict(features_scaled)[0]
        risk_class = self.risk_classifier.predict(features_scaled)[0]
        risk_probabilities = self.risk_classifier.predict_proba(features_scaled)[0]
        
        # Calculate composite risk score
        risk_score = min(1.0, collision_prob * 10 + (1.0 / max(0.1, miss_distance)) * 0.1)
        
        risk_levels = ['Low', 'Medium', 'High']
        
        return {
            'miss_distance_km': max(0.001, miss_distance),
            'collision_probability': max(0.0, min(1.0, collision_prob)),
            'risk_classification': risk_levels[risk_class],
            'risk_score': risk_score,
            'risk_probabilities': {
                'low': risk_probabilities[0],
                'medium': risk_probabilities[1],
                'high': risk_probabilities[2]
            }
        }

def extract_features_from_orbit_data(orbit_data, cdm_data=None):
    """
    Extract features from orbit data for ML prediction
    
    Args:
        orbit_data (dict): Orbit simulation data
        cdm_data (dict): Optional CDM data
    
    Returns:
        list: Feature vector
    """
    
    # Calculate orbital characteristics
    positions = orbit_data['positions_km']
    velocities = orbit_data['velocities_km_s']
    
    # Calculate semi-major axis approximation
    distances = np.sqrt(np.sum(positions**2, axis=0))
    avg_distance = np.mean(distances)
    
    # Calculate velocity magnitude
    velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=0))
    avg_velocity = np.mean(velocity_magnitudes)
    
    # Estimate eccentricity from distance variation
    eccentricity = (np.max(distances) - np.min(distances)) / (np.max(distances) + np.min(distances))
    
    # Calculate inclination approximation
    z_positions = positions[2, :]
    inclination = np.arcsin(np.std(z_positions) / np.std(distances)) * 180 / np.pi
    
    # Default values for unknowns
    relative_velocity = avg_velocity * 0.1  # Rough estimate
    time_to_tca = 24.0  # Default 24 hours
    position_uncertainty_1 = 1.0  # Default 1 km
    position_uncertainty_2 = 1.0  # Default 1 km
    velocity_uncertainty_1 = 0.01  # Default 0.01 km/s
    velocity_uncertainty_2 = 0.01  # Default 0.01 km/s
    correlation_factor = 0.0
    object_size_1 = 10.0  # Default 10 meters
    object_size_2 = 10.0  # Default 10 meters
    
    # Use CDM data if available
    if cdm_data:
        time_to_tca = (datetime.fromisoformat(cdm_data['tca']) - datetime.now()).total_seconds() / 3600
        position_uncertainty_1 = np.sqrt(cdm_data['position_covariance']['xx']) / 1000  # Convert to km
        position_uncertainty_2 = np.sqrt(cdm_data['position_covariance']['yy']) / 1000
        relative_velocity = cdm_data.get('relative_speed', relative_velocity)
    
    feature_vector = [
        avg_distance, eccentricity, inclination, relative_velocity,
        max(0.1, time_to_tca), position_uncertainty_1, position_uncertainty_2,
        velocity_uncertainty_1, velocity_uncertainty_2, correlation_factor,
        object_size_1, object_size_2
    ]
    
    return feature_vector

def assess_collision_risk(orbit_data, risk_threshold=0.1):
    """
    Assess collision risk for orbit data
    
    Args:
        orbit_data (dict): Orbit simulation data
        risk_threshold (float): Risk threshold for alerts
    
    Returns:
        list: List of risk assessment results
    """
    
    # Initialize risk assessor
    assessor = CollisionRiskAssessor()
    
    # Extract features
    features = extract_features_from_orbit_data(orbit_data)
    
    # Make prediction
    risk_result = assessor.predict_collision_risk(features)
    
    # Generate multiple risk scenarios (simulating multiple debris objects)
    risk_results = []
    
    # Create variations for different potential collision scenarios
    for i in range(5):
        # Vary some parameters to simulate different debris objects
        varied_features = features.copy()
        varied_features[5] += np.random.uniform(-0.5, 0.5)  # Position uncertainty
        varied_features[6] += np.random.uniform(-0.5, 0.5)  # Position uncertainty
        varied_features[3] *= (1 + np.random.uniform(-0.3, 0.3))  # Relative velocity
        varied_features[4] = np.random.uniform(1, 168)  # Time to TCA
        
        prediction = assessor.predict_collision_risk(varied_features)
        
        # Create risk event
        tca_time = datetime.now() + timedelta(hours=varied_features[4])
        
        risk_event = {
            'event_id': f'CONJ_{i+1:03d}',
            'time_to_closest_approach': tca_time,
            'miss_distance': prediction['miss_distance_km'],
            'collision_probability': prediction['collision_probability'],
            'risk_score': prediction['risk_score'],
            'risk_classification': prediction['risk_classification'],
            'alert_level': 'HIGH' if prediction['risk_score'] > risk_threshold else 'LOW'
        }
        
        risk_results.append(risk_event)
    
    return risk_results

def predict_miss_distance(orbital_params, covariance_data):
    """
    Predict miss distance using orbital parameters and covariance data
    
    Args:
        orbital_params (dict): Orbital parameters
        covariance_data (dict): Position/velocity covariance data
    
    Returns:
        dict: Miss distance prediction
    """
    
    # Simple miss distance prediction based on uncertainties
    position_uncertainty = np.sqrt(
        covariance_data.get('xx', 1000) + 
        covariance_data.get('yy', 1000) + 
        covariance_data.get('zz', 1000)
    ) / 1000  # Convert to km
    
    # Account for relative velocity and time
    relative_velocity = orbital_params.get('relative_velocity', 10)  # km/s
    time_to_tca = orbital_params.get('time_to_tca', 24)  # hours
    
    # Simple prediction model
    base_miss_distance = position_uncertainty
    velocity_factor = 1 + relative_velocity * 0.1
    time_factor = 1 + np.log(time_to_tca + 1) * 0.1
    
    predicted_miss_distance = base_miss_distance * velocity_factor * time_factor
    
    # Calculate uncertainty bounds
    uncertainty = predicted_miss_distance * 0.3  # 30% uncertainty
    
    return {
        'predicted_miss_distance_km': predicted_miss_distance,
        'uncertainty_km': uncertainty,
        'lower_bound_km': max(0.001, predicted_miss_distance - uncertainty),
        'upper_bound_km': predicted_miss_distance + uncertainty,
        'confidence_level': 0.68  # 1-sigma confidence
    }

def calculate_risk_metrics(miss_distance, relative_velocity, object_sizes):
    """
    Calculate various risk metrics
    
    Args:
        miss_distance (float): Miss distance in km
        relative_velocity (float): Relative velocity in km/s
        object_sizes (list): List of object sizes in meters
    
    Returns:
        dict: Risk metrics
    """
    
    # Combined object size
    combined_size = sum(object_sizes) / 1000  # Convert to km
    
    # Collision probability (simplified model)
    if miss_distance <= combined_size:
        collision_prob = 0.8
    else:
        collision_prob = min(0.5, (combined_size / miss_distance)**2 * 0.1)
    
    # Kinetic energy (rough estimate)
    # Assuming equal masses of 1000 kg each
    kinetic_energy = 0.5 * 1000 * (relative_velocity * 1000)**2  # Joules
    
    # Risk score (0-1 scale)
    distance_factor = max(0, 1 - miss_distance / 10)  # Risk increases as distance decreases
    velocity_factor = min(1, relative_velocity / 15)  # Risk increases with velocity
    size_factor = min(1, combined_size / 0.1)  # Risk increases with size
    
    risk_score = (distance_factor * 0.5 + velocity_factor * 0.3 + size_factor * 0.2)
    
    return {
        'collision_probability': collision_prob,
        'kinetic_energy_joules': kinetic_energy,
        'risk_score': risk_score,
        'distance_factor': distance_factor,
        'velocity_factor': velocity_factor,
        'size_factor': size_factor
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Risk Assessment Module...")
    
    # Test model training
    assessor = CollisionRiskAssessor()
    assessor.train_models()
    
    # Test prediction
    test_features = [7000, 0.1, 51.6, 7.5, 12.0, 1.0, 1.5, 0.01, 0.015, 0.1, 10, 5]
    prediction = assessor.predict_collision_risk(test_features)
    
    print(f"Test prediction:")
    print(f"  Miss distance: {prediction['miss_distance_km']:.3f} km")
    print(f"  Collision probability: {prediction['collision_probability']:.6f}")
    print(f"  Risk classification: {prediction['risk_classification']}")
    print(f"  Risk score: {prediction['risk_score']:.3f}")
    
    print("Risk Assessment Module test completed!")