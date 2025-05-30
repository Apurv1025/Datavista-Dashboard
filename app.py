# Satellite Collision Risk Assessment Dashboard
# Author: Created for educational purposes
# Date: 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import local modules
from data_acquisition import fetch_tle_data, parse_cdm_data, get_sample_satellites
from orbit_simulation import calculate_orbit_positions, simulate_conjunction
from risk_assessment import assess_collision_risk, predict_miss_distance
from visualization import create_3d_orbit_plot, create_collision_risk_plot
from reporting import generate_risk_report, suggest_mitigation_strategies

# Page configuration
st.set_page_config(
    page_title="Satellite Collision Risk Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.sidebar-section {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛰️ Satellite Collision Risk Assessment Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Control Panel")
        
        # Data acquisition section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("1. Data Acquisition")
        
        # Satellite selection
        satellite_option = st.selectbox(
            "Select Primary Satellite",
            ["ISS (ZARYA)", "STARLINK-1007", "COSMOS 2251 DEB", "Custom TLE"]
        )
        
        if satellite_option == "Custom TLE":
            tle_line1 = st.text_input("TLE Line 1", value="1 25544U 98067A   21123.45678901  .00002182  00000-0  40864-4 0  9992")
            tle_line2 = st.text_input("TLE Line 2", value="2 25544  51.6461 339.0518 0003068  83.4820 276.7047 15.48919103281553")
        
        # Time range selection
        start_date = st.date_input("Analysis Start Date", datetime.now().date())
        end_date = st.date_input("Analysis End Date", (datetime.now() + timedelta(days=7)).date())
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk assessment parameters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("2. Risk Parameters")
        
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.1, 0.01)
        miss_distance_threshold = st.number_input("Miss Distance Threshold (km)", 1.0, 100.0, 10.0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis controls
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("3. Analysis Controls")
        
        if st.button("🚀 Run Analysis", type="primary"):
            st.session_state.run_analysis = True
        
        if st.button("📊 Generate Report"):
            st.session_state.generate_report = True
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        run_collision_analysis(satellite_option, start_date, end_date, risk_threshold, miss_distance_threshold)
    else:
        show_dashboard_info()

def show_dashboard_info():
    """Display dashboard information and instructions"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dashboard Overview")
        st.markdown("""
        This dashboard provides comprehensive satellite collision risk assessment capabilities:
        
        **Key Features:**
        - 🛰️ TLE data acquisition and processing
        - 📡 CDM (Conjunction Data Message) analysis
        - 🔄 Skyfield-based orbit simulation
        - 🤖 Machine learning risk assessment
        - 📊 3D trajectory visualization
        - 📄 Automated reporting and mitigation strategies
        """)
        
        st.subheader("🎯 Analysis Steps")
        st.markdown("""
        1. **Data Acquisition**: Fetch TLE and CDM data
        2. **Orbit Simulation**: Calculate satellite trajectories
        3. **Risk Assessment**: Apply ML models for collision prediction
        4. **Visualization**: Display 3D orbits and collision points
        5. **Reporting**: Generate findings and mitigation strategies
        """)
    
    with col2:
        st.subheader("📊 Sample Data")
        
        # Display sample satellite data
        sample_data = get_sample_satellites()
        st.dataframe(sample_data)
        
        st.subheader("⚠️ Risk Metrics")
        st.markdown("""
        **Collision Probability**: Statistical likelihood of collision based on position uncertainties
        
        **Miss Distance**: Closest approach distance between objects
        
        **Time to Closest Approach (TCA)**: When objects will be nearest
        
        **Risk Score**: ML-derived composite risk assessment
        """)

def run_collision_analysis(satellite_option, start_date, end_date, risk_threshold, miss_distance_threshold):
    """Run the collision risk analysis"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Acquisition
        status_text.text("Step 1/5: Acquiring TLE and CDM data...")
        progress_bar.progress(20)
        
        if satellite_option == "Custom TLE":
            # Use custom TLE data
            primary_satellite = {
                'name': 'Custom Satellite',
                'line1': st.session_state.get('tle_line1', ''),
                'line2': st.session_state.get('tle_line2', '')
            }
        else:
            # Fetch real TLE data
            primary_satellite = fetch_tle_data(satellite_option)
        
        # Step 2: Orbit Simulation
        status_text.text("Step 2/5: Simulating satellite orbits...")
        progress_bar.progress(40)
        
        orbit_data = calculate_orbit_positions(primary_satellite, start_date, end_date)
        
        # Step 3: Risk Assessment
        status_text.text("Step 3/5: Analyzing CDMs and assessing risks...")
        progress_bar.progress(60)
        
        risk_results = assess_collision_risk(orbit_data, risk_threshold)
        
        # Step 4: Visualization
        status_text.text("Step 4/5: Creating visualizations...")
        progress_bar.progress(80)
        
        # Step 5: Display Results
        status_text.text("Step 5/5: Displaying results...")
        progress_bar.progress(100)
        
        display_analysis_results(orbit_data, risk_results, primary_satellite)
        
        status_text.text("Analysis complete!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)

def display_analysis_results(orbit_data, risk_results, satellite_info):
    """Display the analysis results"""
    
    st.header("🔍 Analysis Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Conjunctions", len(risk_results))
    
    with col2:
        high_risk_count = sum(1 for r in risk_results if r['risk_score'] > 0.5)
        st.metric("High Risk Events", high_risk_count)
    
    with col3:
        min_distance = min([r['miss_distance'] for r in risk_results]) if risk_results else 0
        st.metric("Min Miss Distance (km)", f"{min_distance:.2f}")
    
    with col4:
        max_risk = max([r['risk_score'] for r in risk_results]) if risk_results else 0
        st.metric("Max Risk Score", f"{max_risk:.3f}")
    
    # 3D Orbit Visualization
    st.subheader("🌍 3D Orbit Visualization")
    orbit_fig = create_3d_orbit_plot(orbit_data, risk_results)
    st.plotly_chart(orbit_fig, use_container_width=True)
    
    # Risk Assessment Results
    st.subheader("⚠️ Risk Assessment Results")
    
    if risk_results:
        risk_df = pd.DataFrame(risk_results)
        st.dataframe(risk_df)
        
        # Risk timeline
        st.subheader("📈 Risk Timeline")
        risk_timeline_fig = create_collision_risk_plot(risk_results)
        st.plotly_chart(risk_timeline_fig, use_container_width=True)
    else:
        st.info("No collision risks detected in the analysis period.")
    
    # Download results
    st.subheader("💾 Download Results")
    if st.button("Download Analysis Report"):
        report = generate_risk_report(orbit_data, risk_results, satellite_info)
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"collision_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()