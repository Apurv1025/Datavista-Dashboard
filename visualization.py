# Visualization Module
# 3D trajectory visualization and collision point plotting

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime

def create_3d_orbit_plot(orbit_data, risk_results=None, show_earth=True):
    """
    Create 3D orbit visualization with Earth and collision points
    
    Args:
        orbit_data (dict): Orbit position data
        risk_results (list): Risk assessment results
        show_earth (bool): Whether to show Earth sphere
    
    Returns:
        plotly.graph_objects.Figure: 3D plot figure
    """
    
    fig = go.Figure()
    
    # Add Earth if requested
    if show_earth:
        # Create Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        earth_radius = 6371  # km
        
        x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            showscale=False,
            opacity=0.7,
            name='Earth'
        ))
    
    # Plot satellite orbit
    positions = orbit_data['positions_km']
    
    fig.add_trace(go.Scatter3d(
        x=positions[0, :],
        y=positions[1, :],
        z=positions[2, :],
        mode='lines+markers',
        line=dict(color='cyan', width=3),
        marker=dict(size=2, color='cyan'),
        name=orbit_data.get('satellite_name', 'Satellite Orbit'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'X: %{x:.1f} km<br>' +
                     'Y: %{y:.1f} km<br>' +
                     'Z: %{z:.1f} km<br>' +
                     '<extra></extra>'
    ))
    
    # Add risk points if available
    if risk_results:
        high_risk_x, high_risk_y, high_risk_z = [], [], []
        medium_risk_x, medium_risk_y, medium_risk_z = [], [], []
        low_risk_x, low_risk_y, low_risk_z = [], [], []
        
        for risk in risk_results:
            # Generate random position near orbit for demonstration
            # In reality, you'd use actual conjunction positions
            idx = np.random.randint(0, positions.shape[1])
            x, y, z = positions[:, idx]
            
            # Add some offset for visibility
            offset = np.random.uniform(-500, 500, 3)
            x, y, z = x + offset[0], y + offset[1], z + offset[2]
            
            if risk['risk_classification'] == 'High':
                high_risk_x.append(x)
                high_risk_y.append(y)
                high_risk_z.append(z)
            elif risk['risk_classification'] == 'Medium':
                medium_risk_x.append(x)
                medium_risk_y.append(y)
                medium_risk_z.append(z)
            else:
                low_risk_x.append(x)
                low_risk_y.append(y)
                low_risk_z.append(z)
        
        # Add high risk points
        if high_risk_x:
            fig.add_trace(go.Scatter3d(
                x=high_risk_x, y=high_risk_y, z=high_risk_z,
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                name='High Risk Events',
                hovertemplate='<b>High Risk Conjunction</b><br>' +
                             'X: %{x:.1f} km<br>' +
                             'Y: %{y:.1f} km<br>' +
                             'Z: %{z:.1f} km<br>' +
                             '<extra></extra>'
            ))
        
        # Add medium risk points
        if medium_risk_x:
            fig.add_trace(go.Scatter3d(
                x=medium_risk_x, y=medium_risk_y, z=medium_risk_z,
                mode='markers',
                marker=dict(size=6, color='orange', symbol='circle'),
                name='Medium Risk Events',
                hovertemplate='<b>Medium Risk Conjunction</b><br>' +
                             'X: %{x:.1f} km<br>' +
                             'Y: %{y:.1f} km<br>' +
                             'Z: %{z:.1f} km<br>' +
                             '<extra></extra>'
            ))
        
        # Add low risk points
        if low_risk_x:
            fig.add_trace(go.Scatter3d(
                x=low_risk_x, y=low_risk_y, z=low_risk_z,
                mode='markers',
                marker=dict(size=4, color='yellow', symbol='circle'),
                name='Low Risk Events',
                hovertemplate='<b>Low Risk Conjunction</b><br>' +
                             'X: %{x:.1f} km<br>' +
                             'Y: %{y:.1f} km<br>' +
                             'Z: %{z:.1f} km<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title='3D Satellite Orbit and Collision Risk Visualization',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='black'
        ),
        showlegend=True,
        width=800,
        height=600
    )
    
    return fig

def create_collision_risk_plot(risk_results):
    """
    Create timeline plot of collision risk events
    
    Args:
        risk_results (list): Risk assessment results
    
    Returns:
        plotly.graph_objects.Figure: Risk timeline plot
    """
    
    if not risk_results:
        # Return empty plot
        fig = go.Figure()
        fig.update_layout(title="No Risk Events Found")
        return fig
    
    # Prepare data
    times = [r['time_to_closest_approach'] for r in risk_results]
    risk_scores = [r['risk_score'] for r in risk_results]
    miss_distances = [r['miss_distance'] for r in risk_results]
    classifications = [r['risk_classification'] for r in risk_results]
    
    # Color mapping
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    colors = [color_map[c] for c in classifications]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Risk Score Timeline', 'Miss Distance Timeline'),
        vertical_spacing=0.15
    )
    
    # Risk score plot
    fig.add_trace(
        go.Scatter(
            x=times,
            y=risk_scores,
            mode='markers+lines',
            marker=dict(size=10, color=colors),
            line=dict(color='blue', width=2),
            name='Risk Score',
            hovertemplate='<b>Risk Event</b><br>' +
                         'Time: %{x}<br>' +
                         'Risk Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Miss distance plot
    fig.add_trace(
        go.Scatter(
            x=times,
            y=miss_distances,
            mode='markers+lines',
            marker=dict(size=8, color=colors),
            line=dict(color='green', width=2),
            name='Miss Distance',
            hovertemplate='<b>Miss Distance</b><br>' +
                         'Time: %{x}<br>' +
                         'Distance: %{y:.2f} km<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add risk threshold line
    fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                  annotation_text="Risk Threshold", row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title='Collision Risk Timeline Analysis',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time to Closest Approach", row=2, col=1)
    fig.update_yaxes(title_text="Risk Score", row=1, col=1)
    fig.update_yaxes(title_text="Miss Distance (km)", row=2, col=1)
    
    return fig

def create_ground_track_plot(orbit_data):
    """
    Create ground track visualization
    
    Args:
        orbit_data (dict): Orbit data with lat/lon information
    
    Returns:
        plotly.graph_objects.Figure: Ground track plot
    """
    
    fig = go.Figure()
    
    # Add world map
    fig.add_trace(go.Scattergeo(
        lon=orbit_data.get('longitudes_deg', []),
        lat=orbit_data.get('latitudes_deg', []),
        mode='markers+lines',
        marker=dict(size=4, color='cyan'),
        line=dict(width=2, color='cyan'),
        name=orbit_data.get('satellite_name', 'Satellite Ground Track'),
        hovertemplate='<b>Ground Track</b><br>' +
                     'Latitude: %{lat:.2f}°<br>' +
                     'Longitude: %{lon:.2f}°<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Satellite Ground Track',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='lightblue',
            showlakes=True,
            lakecolor='lightblue'
        ),
        height=500
    )
    
    return fig

def create_orbital_elements_plot(orbital_elements):
    """
    Create visualization of orbital elements
    
    Args:
        orbital_elements (dict): Orbital elements data
    
    Returns:
        plotly.graph_objects.Figure: Orbital elements plot
    """
    
    # Create radar chart for orbital elements
    categories = [
        'Inclination (norm)',
        'Eccentricity (norm)',
        'Semi-major axis (norm)',
        'Period (norm)',
        'Apogee altitude (norm)',
        'Perigee altitude (norm)'
    ]
    
    # Normalize values to 0-1 scale for radar chart
    values = [
        orbital_elements['inclination_deg'] / 180,
        orbital_elements['eccentricity'],
        (orbital_elements['semi_major_axis_km'] - 6371) / 35000,  # Normalize altitude
        orbital_elements['period_minutes'] / 1440,  # Normalize to daily period
        orbital_elements['apogee_altitude_km'] / 35000,
        orbital_elements['perigee_altitude_km'] / 35000
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Orbital Elements',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Normalized Orbital Elements",
        height=500
    )
    
    return fig

def create_risk_distribution_plot(risk_results):
    """
    Create distribution plots for risk metrics
    
    Args:
        risk_results (list): Risk assessment results
    
    Returns:
        plotly.graph_objects.Figure: Risk distribution plot
    """
    
    if not risk_results:
        fig = go.Figure()
        fig.update_layout(title="No Risk Data Available")
        return fig
    
    # Extract data
    risk_scores = [r['risk_score'] for r in risk_results]
    miss_distances = [r['miss_distance'] for r in risk_results]
    collision_probs = [r.get('collision_probability', 0) for r in risk_results]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Score Distribution', 'Miss Distance Distribution',
                       'Risk Score vs Miss Distance', 'Collision Probability Distribution'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Risk score histogram
    fig.add_trace(
        go.Histogram(x=risk_scores, nbinsx=20, name='Risk Score', 
                    marker_color='blue', opacity=0.7),
        row=1, col=1
    )
    
    # Miss distance histogram
    fig.add_trace(
        go.Histogram(x=miss_distances, nbinsx=20, name='Miss Distance', 
                    marker_color='green', opacity=0.7),
        row=1, col=2
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=miss_distances, y=risk_scores, mode='markers',
                  marker=dict(size=8, color='red', opacity=0.6),
                  name='Risk vs Distance'),
        row=2, col=1
    )
    
    # Collision probability histogram
    fig.add_trace(
        go.Histogram(x=collision_probs, nbinsx=20, name='Collision Probability', 
                    marker_color='orange', opacity=0.7),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Risk Metrics Distribution Analysis',
        height=700,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_xaxes(title_text="Miss Distance (km)", row=1, col=2)
    fig.update_xaxes(title_text="Miss Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Collision Probability", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Risk Score", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_uncertainty_visualization(uncertainty_data):
    """
    Create visualization of position uncertainty over time
    
    Args:
        uncertainty_data (list): List of uncertainty data points
    
    Returns:
        plotly.graph_objects.Figure: Uncertainty visualization
    """
    
    times = [d['time_hours'] for d in uncertainty_data]
    pos_uncertainty = [d['position_uncertainty_km'] for d in uncertainty_data]
    pos_1sigma = [d['position_uncertainty_1sigma'] for d in uncertainty_data]
    pos_3sigma = [d['position_uncertainty_3sigma'] for d in uncertainty_data]
    
    fig = go.Figure()
    
    # Add uncertainty bands
    fig.add_trace(go.Scatter(
        x=times + times[::-1],
        y=pos_3sigma + [0] * len(times),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='3σ Uncertainty',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=times + times[::-1],
        y=pos_1sigma + [0] * len(times),
        fill='toself',
        fillcolor='rgba(255,165,0,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='1σ Uncertainty',
        showlegend=True
    ))
    
    # Add mean uncertainty line
    fig.add_trace(go.Scatter(
        x=times,
        y=pos_uncertainty,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Mean Position Uncertainty'
    ))
    
    fig.update_layout(
        title='Position Uncertainty Propagation Over Time',
        xaxis_title='Time (hours)',
        yaxis_title='Position Uncertainty (km)',
        height=400
    )
    
    return fig

def create_mission_planning_plot(orbit_data, maneuver_windows=None):
    """
    Create mission planning visualization with maneuver windows
    
    Args:
        orbit_data (dict): Orbit data
        maneuver_windows (list): List of maneuver opportunity windows
    
    Returns:
        plotly.graph_objects.Figure: Mission planning plot
    """
    
    times = orbit_data.get('times', [])
    altitudes = orbit_data.get('altitudes_km', [])
    
    fig = go.Figure()
    
    # Plot altitude over time
    fig.add_trace(go.Scatter(
        x=times,
        y=altitudes,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Altitude',
        hovertemplate='<b>Satellite Altitude</b><br>' +
                     'Time: %{x}<br>' +
                     'Altitude: %{y:.1f} km<br>' +
                     '<extra></extra>'
    ))
    
    # Add maneuver windows if provided
    if maneuver_windows:
        for i, window in enumerate(maneuver_windows):
            fig.add_vrect(
                x0=window['start_time'],
                x1=window['end_time'],
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f"Maneuver Window {i+1}",
                annotation_position="top left"
            )
    
    fig.update_layout(
        title='Mission Planning - Altitude Profile and Maneuver Windows',
        xaxis_title='Time',
        yaxis_title='Altitude (km)',
        height=400
    )
    
    return fig

# Example usage and testing
if __name__ == "__main__":
    print("Testing Visualization Module...")
    
    # Create sample orbit data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample orbit positions
    t = np.linspace(0, 2*np.pi, 100)
    r = 7000  # km
    orbit_positions = np.array([
        r * np.cos(t),
        r * np.sin(t),
        1000 * np.sin(3*t)  # Some inclination variation
    ])
    
    sample_orbit_data = {
        'positions_km': orbit_positions,
        'times': [datetime.now() + timedelta(minutes=i*10) for i in range(100)],
        'altitudes_km': [400 + 50*np.sin(i*0.1) for i in range(100)],
        'satellite_name': 'Test Satellite'
    }
    
    # Create sample risk results
    sample_risk_results = [
        {
            'time_to_closest_approach': datetime.now() + timedelta(hours=12),
            'miss_distance': 2.5,
            'risk_score': 0.15,
            'risk_classification': 'High'
        },
        {
            'time_to_closest_approach': datetime.now() + timedelta(hours=24),
            'miss_distance': 5.0,
            'risk_score': 0.08,
            'risk_classification': 'Medium'
        }
    ]
    
    # Test 3D orbit plot
    fig_3d = create_3d_orbit_plot(sample_orbit_data, sample_risk_results)
    print("3D orbit plot created successfully")
    
    # Test risk timeline plot
    fig_risk = create_collision_risk_plot(sample_risk_results)
    print("Risk timeline plot created successfully")
    
    print("Visualization Module test completed!")