import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Chicago Crime Analytics Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Color palette for crime types
crime_color_map = {
    "THEFT": "#5D6D7E",
    "OTHER OFFENSE": "#566573",
    "OFFENSE INVOLVING CHILDREN": "#8E44AD",
    "DECEPTIVE PRACTICE": "#B7950B",
    "CRIMINAL DAMAGE": "#1F618D",
    "CRIM SEXUAL ASSAULT": "#c95edb",
    "SEX OFFENSE": "#BA68C8",
    "ASSAULT": "#922B21",
    "BATTERY": "#7B8D8E",
    "MOTOR VEHICLE THEFT": "#2E86C1",
    "BURGLARY": "#2874A6",
    "WEAPONS VIOLATION": "#A04000",
    "CRIMINAL TRESPASS": "#1ABC9C",
    "ROBBERY": "#616A6B",
    "NARCOTICS": "#6084a7",
    "LIQUOR LAW VIOLATION": "#CA6F1E",
    "PUBLIC PEACE VIOLATION": "#99A3A4",
    "INTERFERENCE WITH PUBLIC OFFICER": "#196F3D",
    "STALKING": "#7FB3D5",
    "HOMICIDE": "#641E16",
    "GAMBLING": "#A55D00",
    "KIDNAPPING": "#7D6608",
    "PROSTITUTION": "#2C3E50",
    "ARSON": "#BF6157",
    "RITUALISM": "#b82ed0",
    "DOMESTIC VIOLENCE": "#CB4335",
    "INTIMIDATION": "#27AE60",
    "OTHER NARCOTIC VIOLATION": "#5B2C6F",
    "PUBLIC INDECENCY": "#CE93D8",
    "CRIMINAL SEXUAL ASSAULT": "#e4aeed",
    "OBSCENITY": "#8E44AD",
    "HUMAN TRAFFICKING": "#7D3C20",
    "CONCEALED CARRY LICENSE VIOLATION": "#F49F54",
    "NON-CRIMINAL": "#6DF4F4"
}
default_color = "#7D8C8C"

# Arrest status colors
ARREST_COLORS = {
    'Arrest Made': '#1f77b4',
    'No Arrest': '#d62728',
    True: '#d62728',
    False: '#1f77b4'
}

# Severity mapping
SEVERITY_MAPPING = {
    "HOMICIDE": 5, "CRIM SEXUAL ASSAULT": 5, "SEX OFFENSE": 5, "KIDNAPPING": 5,
    "WEAPONS VIOLATION": 5, "DOMESTIC VIOLENCE": 5, "CRIMINAL SEXUAL ASSAULT": 5,
    "ASSAULT": 4, "BATTERY": 4, "ROBBERY": 4, "BURGLARY": 4, "ARSON": 4, "MOTOR VEHICLE THEFT": 4,
    "CRIMINAL DAMAGE": 3, "DECEPTIVE PRACTICE": 3, "CRIMINAL TRESPASS": 3, "THEFT": 3,
    "NARCOTICS": 2, "LIQUOR LAW VIOLATION": 2, "PUBLIC PEACE VIOLATION": 2, "PROSTITUTION": 2,
    "GAMBLING": 2, "INTERFERENCE WITH PUBLIC OFFICER": 2, "OTHER OFFENSE": 2, "STALKING": 2,
    "INTIMIDATION": 1, "RITUALISM": 1, "OTHER NARCOTIC VIOLATION": 1, "PUBLIC INDECENCY": 1
}

def get_crime_color(crime_type):
    """Get color for a crime type."""
    return crime_color_map.get(crime_type, default_color)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the crime data."""
    try:
        dtype_map = {
            "ID": "string",
            "Primary Type": "category",
            "Description": "category",
            "Location Description": "category",
            "Arrest": "boolean",
            "Domestic": "boolean"
        }
        
        df = pd.read_csv("chicago_crimes_clean.csv", dtype=dtype_map, low_memory=False)
        
        # Convert types for analysis - ensure proper integer types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])  # Remove any NaN years
        df['Year'] = df['Year'].astype('int64')  # Convert to int64
        
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df['Month'] = df['Month'].fillna(0).astype('int64')
        
        df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
        df['Hour'] = df['Hour'].fillna(0).astype('int64')
        df['Primary Type'] = df['Primary Type'].astype(str)
        df['Location Description'] = df['Location Description'].astype(str)
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Handle DayOfWeek - ensure it's numeric
        if 'DayOfWeek' in df.columns:
            df['DayOfWeek'] = pd.to_numeric(df['DayOfWeek'], errors='coerce')
            df['DayOfWeek'] = df['DayOfWeek'].fillna(-1).astype('int64')
        
        if 'District' in df.columns:
            df['District'] = pd.to_numeric(df['District'], errors='coerce')
            df['District'] = df['District'].fillna(0).astype('int64')
        
        # Add severity score
        df['SeverityScore'] = df['Primary Type'].map(SEVERITY_MAPPING).fillna(1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
try:
    df = load_data()
    
    # Main title
    st.markdown('<p class="main-header">🚔 Chicago Crime Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Comprehensive Crime Data Analysis
    Understanding the dynamics of crime in Chicago through interactive visualizations and statistical analysis.
    """)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    st.sidebar.markdown("Use these filters to drill down into specific data subsets:")
    
    # Year range filter
    min_year = int(df['Year'].dropna().min())
    max_year = int(df['Year'].dropna().max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Crime type filter
    crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
    selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)
    
    # Additional filters
    arrest_filter = st.sidebar.radio(
        "Arrest Status",
        options=['All', 'Arrest Made', 'No Arrest'],
        index=0
    )
    
    domestic_filter = st.sidebar.radio(
        "Domestic Incident",
        options=['All', 'Domestic', 'Non-Domestic'],
        index=0
    )
    
    # Apply filters - ensure Year is numeric and drop NaN
    df_filtered = df.dropna(subset=['Year']).copy()
    df_filtered['Year'] = pd.to_numeric(df_filtered['Year'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Year'])
    # Convert Year to int for comparison
    df_filtered['Year'] = df_filtered['Year'].astype('int64')
    df_filtered = df_filtered[(df_filtered['Year'] >= year_range[0]) & (df_filtered['Year'] <= year_range[1])]
    if selected_crime != 'All':
        df_filtered = df_filtered[df_filtered['Primary Type'] == selected_crime]
    
    # Apply arrest filter
    if arrest_filter == 'Arrest Made':
        df_filtered = df_filtered[df_filtered['Arrest'] == True]
    elif arrest_filter == 'No Arrest':
        df_filtered = df_filtered[df_filtered['Arrest'] == False]
    
    # Apply domestic filter
    if domestic_filter == 'Domestic':
        df_filtered = df_filtered[df_filtered['Domestic'] == True]
    elif domestic_filter == 'Non-Domestic':
        df_filtered = df_filtered[df_filtered['Domestic'] == False]
    
    # Show filter summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filtered Data")
    st.sidebar.info(f"""
    **Records:** {len(df_filtered):,}  
    **% of Total:** {len(df_filtered)/len(df)*100:.1f}%  
    **Years:** {int(df_filtered['Year'].min())} - {int(df_filtered['Year'].max())}
    """)
    
    # Key metrics
    st.markdown('<p class="section-header">📊 Key Metrics</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crimes", f"{len(df_filtered):,}")
    with col2:
        arrest_rate = (df_filtered['Arrest'].astype(str).str.lower().isin(['true', 't', '1', 'yes']).sum() / len(df_filtered) * 100)
        st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
    with col3:
        domestic_rate = (df_filtered['Domestic'].astype(str).str.lower().isin(['true', 't', '1', 'yes']).sum() / len(df_filtered) * 100)
        st.metric("Domestic Rate", f"{domestic_rate:.1f}%")
    with col4:
        st.metric("Crime Types", len(df_filtered['Primary Type'].unique()))
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Temporal Analysis",
        "🔍 Crime Types",
        "👮 Arrests & Domestic",
        "🌍 Spatial Analysis",
        "⚠️ Violent Crimes",
        "💔 Sexual Offenses",
        "📊 Key Insights"
    ])
    
    # Tab 1: Temporal Analysis
    with tab1:
        st.markdown('<p class="section-header">📈 Temporal Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        #### Understanding Crime Patterns Over Time
        Temporal patterns reveal when crimes are most likely to occur, helping law enforcement optimize resource allocation.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crimes by Hour")
            hour_counts = df_filtered.dropna(subset=['Hour']).groupby('Hour').size().reset_index(name='Count')
            fig = px.bar(hour_counts, x='Hour', y='Count', 
                        title='<b>Crime Distribution by Hour of Day</b>',
                        color='Count',
                        color_continuous_scale='Blues')
            fig.update_xaxes(tickmode='array', tickvals=list(range(0, 24)))
            fig.update_traces(hovertemplate='<b>Hour:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Most crimes occur during afternoon and evening hours (12 PM - 8 PM)")
        
        with col2:
            st.subheader("Crimes by Day of Week")
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if 'DayOfWeek' in df_filtered.columns:
                dow_counts = df_filtered.dropna(subset=['DayOfWeek']).copy()
                dow_counts = dow_counts[dow_counts['DayOfWeek'] >= 0]  # Filter out invalid days
                dow_counts = dow_counts.groupby('DayOfWeek').size().reset_index(name='Count')
                dow_counts['DayOfWeek'] = dow_counts['DayOfWeek'].astype('int64')
                dow_counts['Day'] = dow_counts['DayOfWeek'].apply(lambda x: day_names[x] if 0 <= x < 7 else 'Unknown')
                fig = px.bar(dow_counts, x='Day', y='Count',
                            title='<b>Crime Distribution by Day of Week</b>',
                            color='Count',
                            color_continuous_scale='Greens')
                fig.update_traces(hovertemplate='<b>%{x}</b><br><b>Count:</b> %{y:,}<extra></extra>')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Crime peaks during weekends when social interaction is highest")
            else:
                st.warning("Day of week data not available")
        
        st.subheader("Crimes by Month")
        if 'Month' in df_filtered.columns:
            month_counts = df_filtered.dropna(subset=['Month']).copy()
            month_counts = month_counts[month_counts['Month'] > 0]  # Filter out invalid months
            month_counts = month_counts.groupby('Month').size().reset_index(name='Count')
            month_counts['Month'] = month_counts['Month'].astype('int64')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_counts['MonthName'] = month_counts['Month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else 'Unknown')
            fig = px.line(month_counts, x='MonthName', y='Count', markers=True,
                         title='Crime Distribution by Month',
                         color_discrete_sequence=['#d62728'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Month data not available")
        
        st.subheader("Crime Severity Analysis Over Years")
        st.markdown("**Crime Severity Contribution** - Higher scores indicate more serious crimes")
        
        top10 = df_filtered["Primary Type"].value_counts().head(10).index
        df_top = df_filtered[df_filtered["Primary Type"].isin(top10)]
        
        severity_year_type = df_top.groupby(["Year", "Primary Type"]).apply(
            lambda x: x["SeverityScore"].sum()
        ).reset_index(name="Severity Index")
        
        pivot_data = severity_year_type.pivot(
            index="Year",
            columns="Primary Type",
            values="Severity Index"
        ).fillna(0)
        
        fig = go.Figure()
        
        for crime in pivot_data.columns:
            fig.add_trace(go.Scatter(
                x=pivot_data.index,
                y=pivot_data[crime],
                mode='lines',
                stackgroup='one',
                name=crime,
                line=dict(color=get_crime_color(crime)),
                fillcolor=get_crime_color(crime),
                hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Severity: %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="<b>Crime Severity Contribution (Top 10)</b>",
            xaxis_title="Year",
            yaxis_title="Cumulative Severity Index",
            height=500,
            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
            plot_bgcolor='white',
            hovermode='x unified',
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Severity scores: Homicide/Assault=5, Robbery/Burglary=4, Theft/Damage=3, Narcotics=2, Other=1")
    
    # Tab 2: Crime Types
    with tab2:
        st.markdown('<p class="section-header">🔍 Crime Types Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        #### Understanding Crime Categories
        Different crime types have different patterns, arrest rates, and severity levels.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top Crime Types")
            top_n = st.slider("Number of crime types to display", 5, 20, 15, key='top_crimes')
            top_crimes = df_filtered['Primary Type'].value_counts().head(top_n).reset_index()
            top_crimes.columns = ['Crime Type', 'Count']
            fig = px.bar(top_crimes, x='Crime Type', y='Count',
                        color='Crime Type',
                        color_discrete_map=crime_color_map,
                        title=f'<b>Top {top_n} Crime Types</b>')
            fig.update_xaxes(tickangle=-45)
            fig.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Crime Distribution")
            top_crimes_pie = df_filtered['Primary Type'].value_counts().head(10).reset_index()
            top_crimes_pie.columns = ['Crime Type', 'Count']
            fig = px.pie(top_crimes_pie, names='Crime Type', values='Count',
                        color='Crime Type',
                        color_discrete_map=crime_color_map,
                        title='<b>Top 10 Crimes</b>',
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"💡 Top crime: **{top_crimes_pie.iloc[0]['Crime Type']}** ({top_crimes_pie.iloc[0]['Count']:,} incidents)")
        
        st.subheader("Crime Types Over Years (Stacked)")
        pivot = df_filtered.groupby(['Year', 'Primary Type']).size().reset_index(name='Count')
        fig = px.bar(pivot, x='Year', y='Count', color='Primary Type',
                    color_discrete_map=crime_color_map,
                    title='Crime Types Stacked by Year',
                    barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Heatmap: Crime Types by Month")
        heatmap_data = df_filtered.dropna(subset=['Month']).groupby(['Month', 'Primary Type']).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index='Primary Type', columns='Month', values='Count').fillna(0)
        fig = px.imshow(heatmap_pivot.values,
                       x=heatmap_pivot.columns,
                       y=heatmap_pivot.index,
                       labels=dict(x='Month', y='Primary Type', color='Count'),
                       title='Crime Types by Month Heatmap',
                       aspect='auto',
                       color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Arrests & Domestic
    with tab3:
        st.markdown('<p class="section-header">👮 Arrests & Domestic Violence Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Arrest Distribution")
            arrest_norm = df_filtered['Arrest'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            arrest_data = arrest_norm.value_counts().reset_index()
            arrest_data.columns = ['Arrest', 'Count']
            arrest_data['Label'] = arrest_data['Arrest'].map({True: 'Arrested', False: 'Not Arrested'})
            fig = px.pie(arrest_data, names='Label', values='Count',
                        color='Label',
                        color_discrete_map={'Arrested': 'seagreen', 'Not Arrested': 'indianred'},
                        title='Arrest Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Domestic Violence Distribution")
            domestic_norm = df_filtered['Domestic'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            domestic_data = domestic_norm.value_counts().reset_index()
            domestic_data.columns = ['Domestic', 'Count']
            domestic_data['Label'] = domestic_data['Domestic'].map({True: 'Domestic', False: 'Non-Domestic'})
            fig = px.pie(domestic_data, names='Label', values='Count',
                        color='Label',
                        color_discrete_map={'Domestic': 'mediumpurple', 'Non-Domestic': 'lightslategray'},
                        title='Domestic Violence Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Arrests by Crime Type (Top 20)")
        arrest_mask = df_filtered['Arrest'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
        arrest_by_type = df_filtered[arrest_mask].groupby('Primary Type').size().reset_index(name='Count').sort_values('Count', ascending=False).head(20)
        fig = px.bar(arrest_by_type, y='Primary Type', x='Count',
                    orientation='h',
                    color='Primary Type',
                    color_discrete_map=crime_color_map,
                    title='Top 20 Crime Types by Arrest Count')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Arrest Rate by Crime Type (Top 15)")
        arrest_rate_by_type = df_filtered.groupby('Primary Type').agg({
            'Arrest': lambda x: (x.astype(bool).sum() / len(x) * 100)
        }).reset_index()
        arrest_rate_by_type.columns = ['Crime Type', 'Arrest Rate']
        
        # Add count for reference
        crime_counts = df_filtered.groupby('Primary Type').size().reset_index(name='Total Count')
        crime_counts.columns = ['Crime Type', 'Total Count']  # Rename to match
        arrest_rate_by_type = arrest_rate_by_type.merge(crime_counts, on='Crime Type')
        arrest_rate_by_type = arrest_rate_by_type.sort_values('Arrest Rate', ascending=False).head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=arrest_rate_by_type['Crime Type'],
            y=arrest_rate_by_type['Arrest Rate'],
            marker_color=[get_crime_color(ct) for ct in arrest_rate_by_type['Crime Type']],
            text=[f'{v:.1f}%' for v in arrest_rate_by_type['Arrest Rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Arrest Rate: %{y:.1f}%<br>Total Incidents: %{customdata:,}<extra></extra>',
            customdata=arrest_rate_by_type['Total Count']
        ))
        fig.update_layout(
            title='<b>Top 15 Crime Types by Arrest Rate (%)</b>',
            xaxis_title='Crime Type',
            yaxis_title='Arrest Rate (%)',
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Higher arrest rates often correlate with crimes involving direct victim-offender contact")
        
        # Funnel Chart - Arrests by Year
        st.markdown("---")
        st.subheader("Arrest Funnel Analysis by Year")
        st.markdown("**Crime types with arrests - use dropdown to switch years**")
        
        arrest_true = df_filtered[df_filtered['Arrest'] == True].copy()
        
        if len(arrest_true) > 0:
            available_years = sorted(arrest_true['Year'].dropna().unique())
            
            if len(available_years) > 0:
                selected_funnel_year = st.selectbox("Select Year for Funnel Chart", available_years, index=len(available_years)-1)
                
                year_data = arrest_true[arrest_true['Year'] == selected_funnel_year]
                crime_counts = year_data.groupby('Primary Type').size().sort_values(ascending=False).head(20).reset_index(name='Count')
                
                colors = [crime_color_map.get(crime, default_color) for crime in crime_counts['Primary Type']]
                
                fig = go.Figure(go.Funnel(
                    y=crime_counts['Primary Type'],
                    x=crime_counts['Count'],
                    textinfo="value+percent initial",
                    marker=dict(color=colors),
                    connector={"line": {"color": "rgba(150, 150, 150, 0.3)", "width": 2}}
                ))
                
                fig.update_layout(
                    title=f"<b>Crime Types with Arrests - {int(selected_funnel_year)} (Top 20)</b>",
                    height=700,
                    margin={"r": 0, "t": 60, "l": 0, "b": 0}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Funnel narrows from most to least arrested crime types")
            else:
                st.warning("No arrest data available for the selected filters")
        else:
            st.info("No arrests recorded for the selected filters")
    
    # Tab 4: Spatial Analysis
    with tab4:
        st.markdown('<p class="section-header">🌍 Spatial Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        #### Geographic Crime Distribution
        Understanding where crimes occur helps identify hotspots and allocate police resources effectively.
        """)
        
        # Hotspot Map with Animation Toggle
        st.subheader("Crime Hotspot Map")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Interactive map showing crime concentration across Chicago**")
        with col2:
            show_animation = st.checkbox("🎬 Animate by Year", value=False, key='map_animation')
        
        df_map = df_filtered.dropna(subset=['Latitude', 'Longitude']).copy()
        
        if len(df_map) > 0:
            # Sample data for better performance
            max_points = st.slider("Number of points to display", 1000, min(50000, len(df_map)), min(10000, len(df_map)), 1000, key='hotspot_map_points')
            if len(df_map) > max_points:
                df_map_sample = df_map.sample(n=max_points, random_state=42)
            else:
                df_map_sample = df_map
            
            if show_animation and 'Year' in df_map_sample.columns:
                # Animated map by year
                df_map_sample['lat_bin'] = df_map_sample['Latitude'].round(2)
                df_map_sample['lon_bin'] = df_map_sample['Longitude'].round(2)
                hotspots = df_map_sample.groupby(['Year', 'lat_bin', 'lon_bin']).size().reset_index(name='count')
                
                fig = px.density_mapbox(
                    hotspots, lat='lat_bin', lon='lon_bin', z='count',
                    animation_frame='Year',
                    radius=10,
                    center=dict(lat=df_map_sample['Latitude'].mean(), lon=df_map_sample['Longitude'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    color_continuous_scale=[[0, '#ECCCCF'], [0.5, '#FF0013'], [1, '#B2000E']],
                    title=f'<b>Crime Hotspot Map (Animated - {len(df_map_sample):,} points)</b>'
                )
            else:
                fig = px.density_mapbox(
                    df_map_sample, lat='Latitude', lon='Longitude',
                    radius=10,
                    center=dict(lat=df_map_sample['Latitude'].mean(), lon=df_map_sample['Longitude'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    color_continuous_scale=[[0, '#ECCCCF'], [0.5, '#FF0013'], [1, '#B2000E']],
                    title=f'<b>Crime Hotspot Map ({len(df_map_sample):,} points)</b>'
                )
            
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Darker/larger circles indicate higher crime concentrations")
        else:
            st.warning("No geographic data available for the selected filters")
        
        st.markdown("---")
        
        st.subheader("Top Locations")
        location_counts = df_filtered['Location Description'].value_counts().head(20).reset_index()
        location_counts.columns = ['Location', 'Count']
        fig = px.bar(location_counts, x='Count', y='Location',
                    orientation='h',
                    title='Top 20 Crime Locations',
                    color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Crime Density Map")
        df_map = df_filtered.dropna(subset=['Latitude', 'Longitude']).copy()
        
        # Sample data for better performance
        max_points = st.slider("Number of points to display", 1000, 50000, 10000, 1000, key='density_map_points')
        if len(df_map) > max_points:
            df_map = df_map.sample(n=max_points, random_state=42)
        
        fig = px.density_mapbox(df_map, lat='Latitude', lon='Longitude',
                               radius=10,
                               center=dict(lat=df_map['Latitude'].mean(), lon=df_map['Longitude'].mean()),
                               zoom=10,
                               mapbox_style="open-street-map",
                               title=f'Crime Density Map ({len(df_map):,} points)')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Crimes by District")
        district_counts = df_filtered['District'].value_counts().head(20).reset_index()
        district_counts.columns = ['District', 'Count']
        fig = px.bar(district_counts, x='District', y='Count',
                    title='Top 20 Districts by Crime Count',
                    color_discrete_sequence=['#9467bd'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Violent Crimes
    with tab5:
        st.markdown('<p class="section-header">⚠️ Violent Crimes Analysis</p>', unsafe_allow_html=True)
        
        violent_types = ['HOMICIDE', 'BATTERY', 'ASSAULT', 'ROBBERY', 'KIDNAPPING', 'WEAPONS VIOLATION', 'ARSON']
        df_violent = df_filtered[df_filtered['Primary Type'].str.upper().isin(violent_types)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Violent Crimes", f"{len(df_violent):,}")
        with col2:
            violent_rate = (len(df_violent) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("Violent Crime Rate", f"{violent_rate:.1f}%")
        
        st.subheader("Violent Crime Types")
        violent_counts = df_violent['Primary Type'].value_counts().reset_index()
        violent_counts.columns = ['Crime Type', 'Count']
        fig = px.bar(violent_counts, x='Crime Type', y='Count',
                    color='Crime Type',
                    color_discrete_map=crime_color_map,
                    title='Violent Crime Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Violent Crimes by Location (Top 15)")
        violent_locations = df_violent['Location Description'].value_counts().head(15).reset_index()
        violent_locations.columns = ['Location', 'Count']
        fig = px.bar(violent_locations, x='Count', y='Location',
                    orientation='h',
                    title='Top 15 Locations for Violent Crimes',
                    color_discrete_sequence=['#d62728'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Violent Crimes by Hour")
        violent_hour = df_violent.dropna(subset=['Hour']).groupby('Hour').size().reset_index(name='Count')
        fig = px.line(violent_hour, x='Hour', y='Count', markers=True,
                     title='Violent Crimes by Hour',
                     color_discrete_sequence=['#d62728'])
        fig.update_xaxes(tickmode='array', tickvals=list(range(0, 24)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Homicide Trends")
        homicide_data = df_filtered[df_filtered['Primary Type'].str.upper() == 'HOMICIDE'].groupby('Year').size().reset_index(name='Count')
        if not homicide_data.empty:
            homicide_data['MA3'] = homicide_data['Count'].rolling(3, center=True).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=homicide_data['Year'], y=homicide_data['Count'], mode='lines+markers', name='Homicides'))
            fig.add_trace(go.Scatter(x=homicide_data['Year'], y=homicide_data['MA3'], mode='lines', name='3-year Moving Average', line=dict(dash='dash')))
            fig.update_layout(title='Homicide Trends Over Years', xaxis_title='Year', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Sexual Offenses
    with tab6:
        st.markdown('<p class="section-header">💔 Sexual Offenses Analysis</p>', unsafe_allow_html=True)
        
        sexual_types = ['SEX OFFENSE', 'CRIM SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT']
        df_sexual = df_filtered[df_filtered['Primary Type'].str.upper().isin([t.upper() for t in sexual_types])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Sexual Offenses", f"{len(df_sexual):,}")
        with col2:
            sexual_rate = (len(df_sexual) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("Sexual Offense Rate", f"{sexual_rate:.2f}%")
        
        st.subheader("Sexual Offense Types")
        sexual_counts = df_sexual['Primary Type'].value_counts().reset_index()
        sexual_counts.columns = ['Crime Type', 'Count']
        fig = px.bar(sexual_counts, x='Crime Type', y='Count',
                    color='Crime Type',
                    color_discrete_map=crime_color_map,
                    title='Sexual Offense Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Domestic vs Non-Domestic")
            domestic_norm = df_sexual['Domestic'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            domestic_data = domestic_norm.value_counts().reset_index()
            domestic_data.columns = ['Domestic', 'Count']
            domestic_data['Label'] = domestic_data['Domestic'].map({True: 'Domestic', False: 'Non-Domestic'})
            fig = px.pie(domestic_data, names='Label', values='Count',
                        color='Label',
                        color_discrete_map={'Domestic': 'mediumpurple', 'Non-Domestic': 'lightslategray'},
                        title='Domestic Sexual Offenses')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Arrest Rate")
            arrest_norm = df_sexual['Arrest'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            arrest_data = arrest_norm.value_counts().reset_index()
            arrest_data.columns = ['Arrest', 'Count']
            arrest_data['Label'] = arrest_data['Arrest'].map({True: 'Arrested', False: 'Not Arrested'})
            fig = px.pie(arrest_data, names='Label', values='Count',
                        color='Label',
                        color_discrete_map={'Arrested': 'seagreen', 'Not Arrested': 'indianred'},
                        title='Sexual Offense Arrests')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sexual Offenses by Hour")
        sexual_hour = df_sexual.dropna(subset=['Hour']).groupby('Hour').size().reset_index(name='Count')
        fig = px.bar(sexual_hour, x='Hour', y='Count',
                    title='Sexual Offenses by Hour',
                    color_discrete_sequence=['#e377c2'])
        fig.update_xaxes(tickmode='array', tickvals=list(range(0, 24)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sexual Offenses by Location (Top 15)")
        sexual_locations = df_sexual['Location Description'].value_counts().head(15).reset_index()
        sexual_locations.columns = ['Location', 'Count']
        fig = px.bar(sexual_locations, x='Count', y='Location',
                    orientation='h',
                    title='Top 15 Locations for Sexual Offenses',
                    color_discrete_sequence=['#e377c2'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sexual Offenses Trend Over Years")
        sexual_year = df_sexual.groupby('Year').size().reset_index(name='Count')
        fig = px.line(sexual_year, x='Year', y='Count', markers=True,
                     title='Sexual Offenses Trend',
                     color_discrete_sequence=['#e377c2'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: Key Insights
    with tab7:
        st.markdown('<p class="section-header">📊 Key Insights & Analysis Summary</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Executive Summary
        
        This dashboard provides a comprehensive analysis of Chicago crime data, revealing patterns across time, 
        geography, and crime categories.
        """)
        
        # Create insight cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Geographic Patterns
            - **Crime Concentration**: Crime forms clusters in high-traffic areas
            - **Top Locations**: Streets account for majority of incidents
            - **Hotspot Areas**: Central and southern neighborhoods show highest concentrations
            
            ### 📈 Temporal Trends  
            - **Daily**: Crime peaks 12 PM - 8 PM
            - **Weekly**: Weekends show higher rates
            - **Long-term**: Steady decline from 2000s through 2015
            
            ### 👮 Arrest Patterns
            - **Overall Rate**: {:.1f}% of crimes result in arrests
            - **Theft**: Lowest arrest rate despite being most common
            - **Narcotics**: Highest arrest rate (direct police intervention)
            """.format(df_filtered['Arrest'].astype(bool).mean() * 100 if len(df_filtered) > 0 else 0))
        
        with col2:
            st.markdown("""
            ### 🔍 Crime Type Analysis
            - **Most Common**: {} ({:,} incidents)
            - **Domestic Violence**: {:.1f}% of crimes
            - **Severity**: Violent crimes account for significant share
            
            ### 💡 Key Recommendations
            1. **Resource Allocation**: Focus on hotspot areas during peak hours
            2. **Prevention**: Target high-risk time periods
            3. **Community Engagement**: Address social factors
            4. **Data-Driven Policing**: Continue analytics use
            5. **Process Improvement**: Focus on low-arrest-rate crimes
            """.format(
                df_filtered['Primary Type'].mode().iloc[0] if len(df_filtered) > 0 else "N/A",
                df_filtered['Primary Type'].value_counts().iloc[0] if len(df_filtered) > 0 else 0,
                df_filtered['Domestic'].astype(bool).mean() * 100 if len(df_filtered) > 0 else 0
            ))
        
        st.markdown("---")
        st.markdown("### 📊 Comparative Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        overall_arrest = df['Arrest'].astype(bool).mean() * 100
        filtered_arrest = df_filtered['Arrest'].astype(bool).mean() * 100 if len(df_filtered) > 0 else 0
        
        with col1:
            st.metric(
                "Filtered Arrest Rate",
                f"{filtered_arrest:.1f}%",
                f"{filtered_arrest - overall_arrest:+.1f}% vs overall"
            )
        
        with col2:
            st.metric(
                "Data Coverage",
                f"{len(df_filtered)/len(df)*100:.1f}%",
                f"{len(df_filtered):,} records"
            )
        
        with col3:
            if 'SeverityScore' in df_filtered.columns and len(df_filtered) > 0:
                avg_severity = df_filtered['SeverityScore'].mean()
                st.metric("Avg Severity", f"{avg_severity:.2f}", "Scale 1-5")
        
        with col4:
            st.metric("Unique Crime Types", len(df_filtered['Primary Type'].unique()))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Chicago Crime Analytics Dashboard</strong> | Data: Chicago Police Department</p>
        <p>Built with Streamlit | Updated December 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("❌ Error: 'chicago_crimes_clean.csv' not found. Please ensure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
