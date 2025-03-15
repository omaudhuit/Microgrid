import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os

# Set style for better visualization
plt.style.use('ggplot')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300

class MicrogridReportGenerator:
    """
    Generate static reports and visualizations for microgrid analysis.
    """
    def __init__(self, scenario_name="Base Scenario"):
        self.scenario_name = scenario_name
        
        # Simulation parameters
        self.days = 7  # One week simulation
        self.hours = self.days * 24
        self.time_range = np.linspace(0, self.days, self.hours)
        self.dates = [dt.datetime(2025, 1, 1) + dt.timedelta(hours=h) for h in range(self.hours)]
        
        # System parameters
        self.pv_capacity = 150  # kW
        self.wind_capacity = 100  # kW
        self.battery_capacity = 300  # kWh
        self.battery_power = 75  # kW
        self.battery_efficiency = 0.9
        self.battery_soc_min = 0.1
        self.battery_soc_initial = 0.5
        self.peak_load = 180  # kW
        
        # Financial parameters
        self.pv_cost_per_kw = 1000  # $/kW
        self.wind_cost_per_kw = 1500  # $/kW
        self.battery_cost_per_kwh = 400  # $/kWh
        self.om_cost_percent = 2  # % of capital cost per year
        self.project_lifetime = 25  # years
        self.discount_rate = 0.06  # 6%
        self.grid_electricity_price = 0.12  # $/kWh
        self.grid_export_price = 0.05  # $/kWh
        
        # Generate data
        self.generate_data()
        
        # Create report directory
        self.report_dir = f"microgrid_report_{scenario_name.replace(' ', '_')}"
        os.makedirs(self.report_dir, exist_ok=True)
    
    def generate_data(self):
        """Generate all data needed for the report"""
        # Generate solar profile
        self.generate_solar_profile()
        
        # Generate wind profile
        self.generate_wind_profile()
        
        # Generate load profile
        self.generate_load_profile()
        
        # Run simulation
        self.run_simulation()
        
        # Calculate LCOE
        self.calculate_lcoe()
    
    def generate_solar_profile(self):
        """Generate realistic solar profile with daily and hourly patterns"""
        # Time factors
        hours_of_day = np.array([h % 24 for h in range(self.hours)])
        days = np.array([h // 24 for h in range(self.hours)])
        
        # Solar daily pattern (bell curve)
        solar_factor = np.maximum(0, np.sin(np.pi * (hours_of_day - 6) / 12))
        
        # Add day-to-day variability
        daily_factor = 1.0 - 0.3 * np.sin(days * 1.5)
        daily_factor = np.repeat(daily_factor, 24)[:self.hours]
        
        # Add some cloud events
        cloud_events = np.random.randint(0, self.hours, 5)  # 5 random cloud events
        for event in cloud_events:
            # Create a dip for cloud cover
            duration = np.random.randint(2, 6)  # 2-6 hours
            intensity = np.random.uniform(0.3, 0.8)  # How much solar is reduced
            
            # Apply cloud effect
            for i in range(duration):
                if event + i < self.hours:
                    daily_factor[event + i] *= intensity
        
        # Combine factors
        self.pv_profile = self.pv_capacity * solar_factor * daily_factor
    
    def generate_wind_profile(self):
        """Generate realistic wind profile with daily and hourly patterns"""
        # Time factors
        hours_of_day = np.array([h % 24 for h in range(self.hours)])
        days = np.array([h // 24 for h in range(self.hours)])
        
        # Wind tends to be stronger at night
        wind_daily = 0.7 + 0.3 * np.sin(np.pi * (hours_of_day - 18) / 12)
        
        # Add day-to-day variability (with some periodicity)
        daily_factor = 0.6 + 0.4 * np.sin(days * 0.8 + 2)
        daily_factor = np.repeat(daily_factor, 24)[:self.hours]
        
        # Add random gusts and lulls
        for _ in range(10):  # 10 wind events
            start = np.random.randint(0, self.hours - 12)
            duration = np.random.randint(3, 12)
            if np.random.rand() > 0.5:  # Gust
                factor = np.random.uniform(1.2, 1.5)
            else:  # Lull
                factor = np.random.uniform(0.3, 0.7)
            
            for i in range(duration):
                if start + i < self.hours:
                    daily_factor[start + i] *= factor
        
        # Add some smoothing (wind changes aren't instant)
        daily_factor = np.convolve(daily_factor, np.ones(3)/3, mode='same')
        
        # Combine factors and ensure non-negative
        self.wind_profile = np.maximum(0, self.wind_capacity * wind_daily * daily_factor)
    
    def generate_load_profile(self):
        """Generate load profile with daily and weekly patterns"""
        hours_of_day = np.array([h % 24 for h in range(self.hours)])
        days = np.array([h // 24 for h in range(self.hours)])
        days_of_week = np.array([d % 7 for d in days])
        
        # Create control points for the daily load curve (hour, load factor)
        # Updated control points for periodicity
        control_points_x = [0, 6, 9, 12, 15, 18, 21, 24]
        control_points_y = [0.4, 0.35, 0.65, 0.8, 0.75, 0.95, 0.7, 0.4]
        
        # Create spline for daily pattern with periodic boundary conditions
        cs = CubicSpline(control_points_x, control_points_y, bc_type='periodic')
        daily_pattern = cs(hours_of_day)
        
        # Apply weekday/weekend pattern
        weekday_factor = np.ones(self.hours)
        for h in range(self.hours):
            if days_of_week[h] >= 5:  # Weekend
                weekday_factor[h] = 0.8  # Lower load on weekends
        
        # Apply some random noise
        noise = 0.05 * np.random.randn(self.hours)
        
        # Calculate final load
        self.load_profile = self.peak_load * daily_pattern * weekday_factor * (1 + noise)
        self.load_profile = np.maximum(0, self.load_profile)  # Ensure non-negative

    
    def run_simulation(self):
        """Run the microgrid simulation"""
        # Initialize arrays
        self.grid_import = np.zeros(self.hours)
        self.grid_export = np.zeros(self.hours)
        self.battery_charge = np.zeros(self.hours)
        self.battery_discharge = np.zeros(self.hours)
        self.battery_soc = np.zeros(self.hours)
        
        # Set initial battery SOC
        self.battery_soc[0] = self.battery_soc_initial * self.battery_capacity
        
        # Run energy balance for each hour
        for h in range(self.hours):
            # Calculate renewable generation
            renewable_gen = self.pv_profile[h] + self.wind_profile[h]
            
            # Calculate net load (negative means excess generation)
            net_load = self.load_profile[h] - renewable_gen
            
            # Apply battery operation
            if h > 0:
                self.battery_soc[h] = self.battery_soc[h-1]
            
            if net_load > 0:  # Need more power
                # Discharge battery
                max_discharge = min(
                    self.battery_soc[h] - self.battery_soc_min * self.battery_capacity,
                    self.battery_power
                )
                
                discharge = min(max_discharge, net_load)
                if discharge > 0:
                    self.battery_discharge[h] = discharge
                    self.battery_soc[h] -= discharge
                    net_load -= discharge
                
                # Import from grid if needed
                if net_load > 0:
                    self.grid_import[h] = net_load
            else:  # Excess generation
                # Charge battery
                max_charge = min(
                    (self.battery_capacity - self.battery_soc[h]) / self.battery_efficiency,
                    self.battery_power
                )
                
                charge = min(max_charge, -net_load)
                if charge > 0:
                    self.battery_charge[h] = charge
                    self.battery_soc[h] += charge * self.battery_efficiency
                    net_load += charge
                
                # Export to grid if still excess
                if net_load < 0:
                    self.grid_export[h] = -net_load
        
        # Calculate key performance metrics
        self.total_load = np.sum(self.load_profile)
        self.pv_generation = np.sum(self.pv_profile)
        self.wind_generation = np.sum(self.wind_profile)
        self.total_grid_import = np.sum(self.grid_import)
        self.total_grid_export = np.sum(self.grid_export)
        
        # Calculate renewable fraction
        total_renewable_used = (self.pv_generation + self.wind_generation - self.total_grid_export)
        self.renewables_fraction = min(100, 100 * total_renewable_used / self.total_load)
        
        # Calculate self-sufficiency
        self.self_sufficiency = 100 * (self.total_load - self.total_grid_import) / self.total_load
        
        # Battery utilization
        self.battery_cycles = np.sum(self.battery_discharge) / self.battery_capacity
        
    def calculate_lcoe(self):
        """Calculate Levelized Cost of Energy (LCOE)"""
        # Calculate capital costs
        pv_capex = self.pv_capacity * self.pv_cost_per_kw
        wind_capex = self.wind_capacity * self.wind_cost_per_kw
        battery_capex = self.battery_capacity * self.battery_cost_per_kwh
        total_capex = pv_capex + wind_capex + battery_capex
        
        # Calculate annual O&M costs
        annual_om = total_capex * (self.om_cost_percent / 100)
        
        # Calculate grid costs and revenues (annualized)
        annual_grid_cost = self.total_grid_import * self.grid_electricity_price * (365 / self.days)
        annual_export_revenue = self.total_grid_export * self.grid_export_price * (365 / self.days)
        annual_net_grid = annual_grid_cost - annual_export_revenue
        
        # Calculate NPV of all costs
        npv_factor = sum(1 / ((1 + self.discount_rate) ** year) for year in range(1, self.project_lifetime + 1))
        npv_costs = total_capex + (annual_om + annual_net_grid) * npv_factor
        
        # Calculate lifetime energy production
        annual_energy = self.total_load * (365 / self.days)
        lifetime_energy = annual_energy * self.project_lifetime
        
        # Calculate LCOE
        self.lcoe = npv_costs / lifetime_energy if lifetime_energy > 0 else 0
        
        # Store components for reporting
        self.cost_components = {
            'PV Capital': pv_capex,
            'Wind Capital': wind_capex,
            'Battery Capital': battery_capex,
            'Annual O&M': annual_om,
            'Annual Grid Cost': annual_grid_cost,
            'Annual Export Revenue': annual_export_revenue,
            'NPV Total': npv_costs,
            'Annual Energy': annual_energy,
            'LCOE': self.lcoe
        }
    
    def create_energy_flow_plot(self):
        """Create a plot of the energy flows over time"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert to dates for better x-axis formatting
        dates = self.dates
        
        # Plot load
        ax.plot(dates, self.load_profile, 'k-', label='Load', linewidth=2)
        
        # Plot renewable generation
        ax.plot(dates, self.pv_profile, color='orange', label='Solar PV')
        ax.plot(dates, self.wind_profile, color='blue', label='Wind')
        ax.plot(dates, self.pv_profile + self.wind_profile, 'g--', label='Total Renewable', alpha=0.7)
        
        # Plot grid exchanges
        ax.plot(dates, self.grid_import, 'r-', label='Grid Import')
        ax.plot(dates, -self.grid_export, 'g-', label='Grid Export')
        
        # Fill areas for visual appeal
        ax.fill_between(dates, 0, self.pv_profile, color='orange', alpha=0.3)
        ax.fill_between(dates, self.pv_profile, self.pv_profile + self.wind_profile, color='blue', alpha=0.3)
        
        # Add battery charge/discharge indicators
        charge_mask = self.battery_charge > 0
        discharge_mask = self.battery_discharge > 0
        
        if np.any(charge_mask):
            ax.scatter(np.array(dates)[charge_mask], self.battery_charge[charge_mask],
                      marker='^', color='green', s=50, label='Battery Charging')
        
        if np.any(discharge_mask):
            ax.scatter(np.array(dates)[discharge_mask], self.battery_discharge[discharge_mask],
                      marker='v', color='purple', s=50, label='Battery Discharging')
        
        # Set axis labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'Microgrid Energy Flows - {self.scenario_name}')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Add summary statistics as text
        stats_text = (
            f"Total Load: {self.total_load:.1f} kWh\n"
            f"PV Generation: {self.pv_generation:.1f} kWh\n"
            f"Wind Generation: {self.wind_generation:.1f} kWh\n"
            f"Grid Import: {self.total_grid_import:.1f} kWh\n"
            f"Grid Export: {self.total_grid_export:.1f} kWh\n"
            f"Renewable Fraction: {self.renewables_fraction:.1f}%\n"
            f"Self-sufficiency: {self.self_sufficiency:.1f}%\n"
            f"Battery Cycles: {self.battery_cycles:.2f}"
        )
        
        # Add text box
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f"{self.report_dir}/energy_flows.png")
        
        return fig
    
    def create_battery_soc_plot(self):
        """Create a plot of the battery state of charge"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Convert to dates for better x-axis formatting
        dates = self.dates
        
        # Calculate SOC percentage
        soc_percent = 100 * self.battery_soc / self.battery_capacity
        
        # Create colormap for SOC
        cmap = plt.cm.viridis
        norm = plt.Normalize(0, 100)
        
        # Plot SOC with gradient fill
        for i in range(len(dates)-1):
            ax.fill_between([dates[i], dates[i+1]], [0, 0], [soc_percent[i], soc_percent[i+1]],
                           color=cmap(norm(soc_percent[i])))
        
        # Plot SOC line
        ax.plot(dates, soc_percent, 'k-', linewidth=2, label='SOC')
        
        # Add min SOC line
        ax.axhline(y=self.battery_soc_min * 100, color='r', linestyle='--', alpha=0.7, label=f'Min SOC ({self.battery_soc_min*100:.0f}%)')
        
        # Highlight charge/discharge events
        charge_mask = self.battery_charge > 0
        discharge_mask = self.battery_discharge > 0
        
        if np.any(charge_mask):
            ax.scatter(np.array(dates)[charge_mask], soc_percent[charge_mask],
                      marker='^', color='lime', s=50, label='Charging')
        
        if np.any(discharge_mask):
            ax.scatter(np.array(dates)[discharge_mask], soc_percent[discharge_mask],
                      marker='v', color='red', s=50, label='Discharging')
        
        # Set axis labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('State of Charge (%)')
        ax.set_title(f'Battery State of Charge - {self.scenario_name}')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add SOC stats
        min_soc = np.min(soc_percent)
        max_soc = np.max(soc_percent)
        avg_soc = np.mean(soc_percent)
        
        stats_text = (
            f"Battery Capacity: {self.battery_capacity:.0f} kWh\n"
            f"Battery Power: {self.battery_power:.0f} kW\n"
            f"Min SOC: {min_soc:.1f}%\n"
            f"Max SOC: {max_soc:.1f}%\n"
            f"Avg SOC: {avg_soc:.1f}%\n"
            f"Total Charge: {np.sum(self.battery_charge):.1f} kWh\n"
            f"Total Discharge: {np.sum(self.battery_discharge):.1f} kWh\n"
            f"Equivalent Cycles: {self.battery_cycles:.2f}"
        )
        
        # Add text box
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='SOC (%)')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f"{self.report_dir}/battery_soc.png")
        
        return fig
    
    def create_daily_profile_plot(self):
        """Create a plot showing average daily profiles"""
        # Reshape data to get daily profiles (hours x days)
        hours_per_day = 24
        days = self.hours // hours_per_day
        
        load_daily = self.load_profile.reshape(days, hours_per_day)
        pv_daily = self.pv_profile.reshape(days, hours_per_day)
        wind_daily = self.wind_profile.reshape(days, hours_per_day)
        grid_import_daily = self.grid_import.reshape(days, hours_per_day)
        grid_export_daily = self.grid_export.reshape(days, hours_per_day)
        
        # Calculate averages
        load_avg = np.mean(load_daily, axis=0)
        pv_avg = np.mean(pv_daily, axis=0)
        wind_avg = np.mean(wind_daily, axis=0)
        grid_import_avg = np.mean(grid_import_daily, axis=0)
        grid_export_avg = np.mean(grid_export_daily, axis=0)
        
        # Calculate standard deviations for uncertainty bands
        load_std = np.std(load_daily, axis=0)
        pv_std = np.std(pv_daily, axis=0)
        wind_std = np.std(wind_daily, axis=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Time range
        hours = np.arange(24)
        
        # Plot averages with uncertainty bands
        ax.plot(hours, load_avg, 'k-', linewidth=3, label='Load')
        ax.fill_between(hours, load_avg - load_std, load_avg + load_std, color='k', alpha=0.2)
        
        ax.plot(hours, pv_avg, color='orange', linewidth=2, label='Solar PV')
        ax.fill_between(hours, pv_avg - pv_std, pv_avg + pv_std, color='orange', alpha=0.2)
        
        ax.plot(hours, wind_avg, color='blue', linewidth=2, label='Wind')
        ax.fill_between(hours, wind_avg - wind_std, wind_avg + wind_std, color='blue', alpha=0.2)
        
        ax.plot(hours, pv_avg + wind_avg, 'g--', linewidth=2, label='Total Renewable')
        
        ax.plot(hours, grid_import_avg, 'r-', linewidth=2, label='Grid Import')
        ax.plot(hours, -grid_export_avg, 'g-', linewidth=2, label='Grid Export')
        
        # Set labels and title
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Power (kW)')
        ax.set_title(f'Average Daily Profiles - {self.scenario_name}')
        
        # Set x-axis ticks to show hours
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xlim(0, 23)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Time labels for better readability
        time_labels = ['Midnight', '2 AM', '4 AM', '6 AM', '8 AM', '10 AM', 
                      'Noon', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM']
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticklabels(time_labels)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f"{self.report_dir}/daily_profiles.png")
        
        return fig
    
    def create_energy_balance_diagram(self):
        """Create an energy balance diagram showing the overall energy flows"""
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set background color
        ax.set_facecolor('#f9f9f9')
        
        # Hide axes
        ax.axis('off')
        
        # Define node positions
        node_pos = {
            'PV': [0.2, 0.8],
            'Wind': [0.2, 0.5],
            'Battery': [0.5, 0.5],
            'Grid': [0.2, 0.2],
            'Load': [0.8, 0.5]
        }
        
        # Define node sizes based on capacity
        node_sizes = {
            'PV': self.pv_capacity / 5,
            'Wind': self.wind_capacity / 5,
            'Battery': self.battery_capacity / 15,
            'Grid': max(self.total_grid_import, self.total_grid_export) / 50,
            'Load': self.peak_load / 5
        }
        
        # Cap node sizes for visualization
        for node in node_sizes:
            node_sizes[node] = max(15, min(50, node_sizes[node]))
        
        # Define node colors
        node_colors = {
            'PV': 'orange',
            'Wind': 'blue',
            'Battery': 'gray',
            'Grid': 'red',
            'Load': 'black'
        }
        
        # Draw nodes
        for node, pos in node_pos.items():
            circle = plt.Circle(pos, node_sizes[node]/100, color=node_colors[node], alpha=0.8)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], node, ha='center', va='center', color='white', 
                   fontweight='bold', fontsize=12)
        
        # Calculate energy flows
        flows = {
            'PV->Load': min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)),
            'Wind->Load': min(self.wind_generation, self.total_load - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)) - self.total_grid_import - np.sum(self.battery_discharge)),
            'PV->Battery': min(np.sum(self.battery_charge), self.pv_generation - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge))),
            'Wind->Battery': min(np.sum(self.battery_charge) - min(np.sum(self.battery_charge), self.pv_generation - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge))), 
                               self.wind_generation - min(self.wind_generation, self.total_load - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)) - self.total_grid_import - np.sum(self.battery_discharge))),
            'Battery->Load': np.sum(self.battery_discharge),
            'Grid->Load': self.total_grid_import,
            'PV->Grid': max(0, self.pv_generation - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)) - min(np.sum(self.battery_charge), self.pv_generation - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)))),
            'Wind->Grid': max(0, self.wind_generation - min(self.wind_generation, self.total_load - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)) - self.total_grid_import - np.sum(self.battery_discharge)) - min(np.sum(self.battery_charge) - min(np.sum(self.battery_charge), self.pv_generation - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge))), self.wind_generation - min(self.wind_generation, self.total_load - min(self.pv_generation, self.total_load - self.total_grid_import - np.sum(self.battery_discharge)) - self.total_grid_import - np.sum(self.battery_discharge))))
        }
        
        # Define flow paths
        flow_paths = {
            'PV->Load': [node_pos['PV'], node_pos['Load']],
            'Wind->Load': [node_pos['Wind'], node_pos['Load']],
            'PV->Battery': [node_pos['PV'], node_pos['Battery']],
            'Wind->Battery': [node_pos['Wind'], node_pos['Battery']],
            'Battery->Load': [node_pos['Battery'], node_pos['Load']],
            'Grid->Load': [node_pos['Grid'], node_pos['Load']],
            'PV->Grid': [node_pos['PV'], node_pos['Grid']],
            'Wind->Grid': [node_pos['Wind'], node_pos['Grid']]
        }
        
        # Define flow colors
        flow_colors = {
            'PV->Load': 'orange',
            'Wind->Load': 'blue',
            'PV->Battery': 'orange',
            'Wind->Battery': 'blue',
            'Battery->Load': 'gray',
            'Grid->Load': 'red',
            'PV->Grid': 'orange',
            'Wind->Grid': 'blue'
        }
        
        # Normalize flows for visualization
        max_flow = max(flows.values()) if flows.values() else 1
        normalized_flows = {k: max(1, v / max_flow * 20) for k, v in flows.items()}
        
        # Draw flows
        for flow_name, width in normalized_flows.items():
            if flows[flow_name] > 0.01 * self.total_load:  # Only show significant flows
                start, end = flow_paths[flow_name]
                
                # Create curved arrows
                ax.annotate('', xy=end, xytext=start,
                          arrowprops=dict(arrowstyle='->', color=flow_colors[flow_name], 
                                        lw=width, connectionstyle='arc3,rad=0.1', alpha=0.7))
        
        # Add legend for flow quantities
        legend_items = []
        for flow_name, value in flows.items():
            if value > 0.01 * self.total_load:  # Only show significant flows
                color = flow_colors[flow_name]
                legend_items.append((flow_name, f"{value:.1f} kWh", color))
        
        # Sort legend by value
        legend_items.sort(key=lambda x: flows[x[0]], reverse=True)
        
        # Create legend patches
        legend_patches = []
        for name, value, color in legend_items:
            patch = Patch(color=color, alpha=0.7, label=f"{name}: {value}")
            legend_patches.append(patch)
        
        # Add legend
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                 ncol=2, fontsize=10)
        
        # Add title
        ax.set_title(f'Energy Flow Diagram - {self.scenario_name}', fontsize=16, pad=20)
        
        # Add system metrics
        metrics_text = (
            f"Total Load: {self.total_load:.1f} kWh\n"
            f"PV Generation: {self.pv_generation:.1f} kWh\n"
            f"Wind Generation: {self.wind_generation:.1f} kWh\n"
            f"Battery Throughput: {np.sum(self.battery_discharge):.1f} kWh\n"
            f"Grid Import: {self.total_grid_import:.1f} kWh\n"
            f"Grid Export: {self.total_grid_export:.1f} kWh\n"
            f"Renewable Fraction: {self.renewables_fraction:.1f}%\n"
            f"LCOE: ${self.lcoe:.4f}/kWh"
        )
        
        plt.figtext(0.02, 0.97, metrics_text, fontsize=10, ha='left', va='top',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Save figure
        fig.savefig(f"{self.report_dir}/energy_balance.png")
        
        return fig
    
    def create_lcoe_breakdown(self):
        """Create a breakdown of LCOE components"""
        # Ensure grid cost is non-negative for visualization
        grid_cost = (self.cost_components['Annual Grid Cost'] - self.cost_components['Annual Export Revenue']) * self.project_lifetime
        grid_cost = max(0, grid_cost)  # Set to zero if negative

        # Extract cost components
        components = [
            ('PV Capital', self.cost_components['PV Capital']),
            ('Wind Capital', self.cost_components['Wind Capital']),
            ('Battery Capital', self.cost_components['Battery Capital']),
            ('O&M (NPV)', self.cost_components['Annual O&M'] * self.project_lifetime),
            ('Grid Costs (NPV)', grid_cost)
        ]
        
        # Sort by value
        components.sort(key=lambda x: x[1], reverse=True)
        
        # Extract labels and values
        labels = [c[0] for c in components]
        values = [c[1] for c in components]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Create pie chart of capital costs
        ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
                wedgeprops=dict(width=0.5, edgecolor='w'),
                textprops={'fontsize': 12})
        ax1.set_title('Cost Components Breakdown')
        
        # Create bar chart for LCOE components
        # Convert to LCOE by dividing by lifetime energy
        lifetime_energy = self.cost_components['Annual Energy'] * self.project_lifetime
        lcoe_components = [(label, val / lifetime_energy * 100) for label, val in components]  # cents/kWh
        
        # Sort by value
        lcoe_components.sort(key=lambda x: x[1], reverse=True)
        
        # Extract labels and values
        lcoe_labels = [c[0] for c in lcoe_components]
        lcoe_values = [c[1] for c in lcoe_components]
        
        # Plot horizontal bar chart
        bars = ax2.barh(lcoe_labels, lcoe_values, color=plt.cm.tab10.colors)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}¢', ha='left', va='center')
        
        ax2.set_xlabel('Cost (¢/kWh)')
        ax2.set_title('LCOE Components')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add total LCOE line
        ax2.axvline(x=self.lcoe * 100, color='red', linestyle='--', 
                    label=f'Total LCOE: {self.lcoe*100:.2f}¢/kWh')
        ax2.legend()
        
        # Add financial summary
        summary_text = (
            f"System Costs:\n"
            f"PV: ${self.cost_components['PV Capital']:,.0f} (${self.pv_cost_per_kw:,.0f}/kW)\n"
            f"Wind: ${self.cost_components['Wind Capital']:,.0f} (${self.wind_cost_per_kw:,.0f}/kW)\n"
            f"Battery: ${self.cost_components['Battery Capital']:,.0f} (${self.battery_cost_per_kwh:,.0f}/kWh)\n"
            f"Total Capital: ${sum(values):,.0f}\n\n"
            f"Annual O&M: ${self.cost_components['Annual O&M']:,.0f}/year\n"
            f"Annual Energy: {self.cost_components['Annual Energy']:,.0f} kWh/year\n"
            f"Project Lifetime: {self.project_lifetime} years\n"
            f"Discount Rate: {self.discount_rate*100:.1f}%\n"
            f"LCOE: ${self.lcoe:.4f}/kWh"
        )
        
        plt.figtext(0.5, 0.02, summary_text, fontsize=10, ha='center', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        plt.subplots_adjust(wspace=0.3)
        
        # Add title
        plt.suptitle(f'Cost Analysis - {self.scenario_name}', fontsize=16, y=0.98)
        
        # Save figure
        fig.savefig(f"{self.report_dir}/cost_analysis.png")
        
        return fig

    
    def create_comprehensive_report(self):
        """Create a comprehensive report with all visualizations"""
        # Generate all plots
        energy_flow_fig = self.create_energy_flow_plot()
        battery_soc_fig = self.create_battery_soc_plot()
        daily_profile_fig = self.create_daily_profile_plot()
        energy_balance_fig = self.create_energy_balance_diagram()
        lcoe_breakdown_fig = self.create_lcoe_breakdown()
        
        # Create summary text file
        with open(f"{self.report_dir}/summary.txt", 'w') as f:
            f.write(f"MICROGRID ANALYSIS REPORT - {self.scenario_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SYSTEM CONFIGURATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"PV Capacity: {self.pv_capacity} kW\n")
            f.write(f"Wind Capacity: {self.wind_capacity} kW\n")
            f.write(f"Battery Capacity: {self.battery_capacity} kWh\n")
            f.write(f"Battery Power: {self.battery_power} kW\n")
            f.write(f"Battery Efficiency: {self.battery_efficiency*100}%\n")
            f.write(f"Peak Load: {self.peak_load} kW\n\n")
            
            f.write("ENERGY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Simulation Period: {self.days} days\n")
            f.write(f"Total Load: {self.total_load:.1f} kWh\n")
            f.write(f"PV Generation: {self.pv_generation:.1f} kWh ({100*self.pv_generation/self.total_load:.1f}% of load)\n")
            f.write(f"Wind Generation: {self.wind_generation:.1f} kWh ({100*self.wind_generation/self.total_load:.1f}% of load)\n")
            f.write(f"Total Renewable Generation: {self.pv_generation+self.wind_generation:.1f} kWh ({100*(self.pv_generation+self.wind_generation)/self.total_load:.1f}% of load)\n")
            f.write(f"Grid Import: {self.total_grid_import:.1f} kWh ({100*self.total_grid_import/self.total_load:.1f}% of load)\n")
            f.write(f"Grid Export: {self.total_grid_export:.1f} kWh\n")
            f.write(f"Battery Charge: {np.sum(self.battery_charge):.1f} kWh\n")
            f.write(f"Battery Discharge: {np.sum(self.battery_discharge):.1f} kWh\n")
            f.write(f"Battery Cycles: {self.battery_cycles:.2f}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Renewable Fraction: {self.renewables_fraction:.1f}%\n")
            f.write(f"Self-sufficiency: {self.self_sufficiency:.1f}%\n")
            f.write(f"Battery Utilization: {100*self.battery_cycles/self.days:.1f}% (cycles per day)\n\n")
            
            f.write("FINANCIAL METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"PV Cost: ${self.pv_capacity*self.pv_cost_per_kw:,.0f} (${self.pv_cost_per_kw:,.0f}/kW)\n")
            f.write(f"Wind Cost: ${self.wind_capacity*self.wind_cost_per_kw:,.0f} (${self.wind_cost_per_kw:,.0f}/kW)\n")
            f.write(f"Battery Cost: ${self.battery_capacity*self.battery_cost_per_kwh:,.0f} (${self.battery_cost_per_kwh:,.0f}/kWh)\n")
            f.write(f"Total Capital Cost: ${self.pv_capacity*self.pv_cost_per_kw + self.wind_capacity*self.wind_cost_per_kw + self.battery_capacity*self.battery_cost_per_kwh:,.0f}\n")
            f.write(f"Annual O&M Cost: ${self.cost_components['Annual O&M']:,.0f}\n")
            f.write(f"Annual Grid Cost: ${self.cost_components['Annual Grid Cost']:,.0f}\n")
            f.write(f"Annual Export Revenue: ${self.cost_components['Annual Export Revenue']:,.0f}\n")
            f.write(f"Levelized Cost of Energy (LCOE): ${self.lcoe:.4f}/kWh\n\n")
            
            f.write("CONCLUSION\n")
            f.write("-" * 30 + "\n")
            f.write(f"The {self.scenario_name} microgrid configuration achieves a renewable energy fraction of {self.renewables_fraction:.1f}% ")
            f.write(f"with an LCOE of ${self.lcoe:.4f}/kWh. ")
            
            if self.renewables_fraction > 90:
                f.write("The system achieves very high renewable penetration, ")
            elif self.renewables_fraction > 70:
                f.write("The system achieves good renewable penetration, ")
            else:
                f.write("The system achieves moderate renewable penetration, ")
                
            if self.battery_cycles / self.days > 0.8:
                f.write("with high battery utilization. ")
            elif self.battery_cycles / self.days > 0.4:
                f.write("with moderate battery utilization. ")
            else:
                f.write("with low battery utilization. ")
                
            if self.lcoe < 0.10:
                f.write("The LCOE is very competitive compared to typical grid prices.")
            elif self.lcoe < 0.15:
                f.write("The LCOE is competitive with typical grid prices.")
            else:
                f.write("The LCOE is higher than typical grid prices but may be justified by other benefits.")
        
        # Return paths to all generated files
        return {
            'summary': f"{self.report_dir}/summary.txt",
            'energy_flow': f"{self.report_dir}/energy_flows.png",
            'battery_soc': f"{self.report_dir}/battery_soc.png",
            'daily_profile': f"{self.report_dir}/daily_profiles.png",
            'energy_balance': f"{self.report_dir}/energy_balance.png",
            'cost_analysis': f"{self.report_dir}/cost_analysis.png"
        }
    
    def compare_scenarios(self, other_scenarios):
        """Compare multiple scenarios and create comparative visualizations"""
        # This assumes other_scenarios is a list of MicrogridReportGenerator instances
        all_scenarios = [self] + other_scenarios
        scenario_names = [s.scenario_name for s in all_scenarios]
        
        # Create comparison directory
        comparison_dir = "microgrid_comparison"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Extract key metrics for comparison
        metrics = {
            'PV Capacity (kW)': [s.pv_capacity for s in all_scenarios],
            'Wind Capacity (kW)': [s.wind_capacity for s in all_scenarios],
            'Battery Capacity (kWh)': [s.battery_capacity for s in all_scenarios],
            'Renewable Fraction (%)': [s.renewables_fraction for s in all_scenarios],
            'Self-sufficiency (%)': [s.self_sufficiency for s in all_scenarios],
            'Battery Cycles': [s.battery_cycles for s in all_scenarios],
            'LCOE ($/kWh)': [s.lcoe for s in all_scenarios],
            'Grid Import (kWh)': [s.total_grid_import for s in all_scenarios],
            'Grid Export (kWh)': [s.total_grid_export for s in all_scenarios]
        }
        
        # Create comparison bar charts
        for metric, values in metrics.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            bars = ax.bar(scenario_names, values, color=plt.cm.tab10.colors)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                if 'LCOE' in metric:
                    label = f'${height:.4f}'
                elif 'Capacity' in metric or 'Cycles' in metric:
                    label = f'{height:.1f}'
                else:
                    label = f'{height:.1f}'
                
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * max(values),
                       label, ha='center', va='bottom', rotation=0)
            
            # Set labels and title
            ax.set_xlabel('Scenario')
            ax.set_ylabel(metric)
            ax.set_title(f'Comparison of {metric} Across Scenarios')
            
            # Adjust y-axis to start from 0
            ax.set_ylim(0, max(values) * 1.15)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45 if len(scenario_names) > 3 else 0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            metric_filename = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_per_')
            fig.savefig(f"{comparison_dir}/comparison_{metric_filename}.png")
        
        # Create spider chart for key metrics
        spider_metrics = {
            'Renewable %': [s.renewables_fraction / 100 for s in all_scenarios],
            'Self-sufficiency': [s.self_sufficiency / 100 for s in all_scenarios],
            'Battery Util.': [s.battery_cycles / (s.days * 1.5) for s in all_scenarios],  # Normalized to max 1.5 cycles per day
            'Cost Efficiency': [min(1.5, 0.2 / s.lcoe) for s in all_scenarios],  # Normalized, higher is better
            'Grid Independence': [1 - (s.total_grid_import / s.total_load) for s in all_scenarios]
        }
        
        # Extract labels and values
        categories = list(spider_metrics.keys())
        N = len(categories)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set category labels
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw category labels
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Plot each scenario
        for i, scenario in enumerate(all_scenarios):
            values = [spider_metrics[metric][i] for metric in categories]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=scenario.scenario_name, color=plt.cm.tab10.colors[i])
            ax.fill(angles, values, plt.cm.tab10.colors[i], alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title("Performance Comparison Across Scenarios", size=15, y=1.1)
        
        # Save figure
        plt.tight_layout()
        fig.savefig(f"{comparison_dir}/performance_spider_chart.png")
        
        # Create LCOE breakdown comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        lcoe_components = {
            'PV Capital': [s.pv_capacity * s.pv_cost_per_kw / (s.cost_components['Annual Energy'] * s.project_lifetime) for s in all_scenarios],
            'Wind Capital': [s.wind_capacity * s.wind_cost_per_kw / (s.cost_components['Annual Energy'] * s.project_lifetime) for s in all_scenarios],
            'Battery Capital': [s.battery_capacity * s.battery_cost_per_kwh / (s.cost_components['Annual Energy'] * s.project_lifetime) for s in all_scenarios],
            'O&M': [s.cost_components['Annual O&M'] / s.cost_components['Annual Energy'] for s in all_scenarios],
            'Grid Costs': [(s.cost_components['Annual Grid Cost'] - s.cost_components['Annual Export Revenue']) / s.cost_components['Annual Energy'] for s in all_scenarios]
        }
        
        # Convert to DataFrame for easier stacked bar plotting
        df = pd.DataFrame(lcoe_components, index=scenario_names)
        
        # Create stacked bar chart
        ax = df.plot(kind='bar', stacked=True, ax=ax, figsize=(12, 8))
        
        # Add total LCOE values
        total_lcoe = [s.lcoe for s in all_scenarios]
        for i, lcoe in enumerate(total_lcoe):
            ax.text(i, sum(df.iloc[i]), f'Total: ${lcoe:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Format y-axis as dollars
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:.4f}'))
        
        # Set labels and title
        ax.set_xlabel('Scenario')
        ax.set_ylabel('LCOE ($/kWh)')
        ax.set_title('LCOE Component Breakdown by Scenario')
        
        # Adjust legend
        ax.legend(title='Cost Component', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45 if len(scenario_names) > 3 else 0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f"{comparison_dir}/lcoe_breakdown_comparison.png")
        
        # Return paths to all comparison files
        comparison_files = {
            'spider_chart': f"{comparison_dir}/performance_spider_chart.png",
            'lcoe_breakdown': f"{comparison_dir}/lcoe_breakdown_comparison.png"
        }
        
        for metric in metrics:
            metric_filename = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_per_')
            comparison_files[metric_filename] = f"{comparison_dir}/comparison_{metric_filename}.png"
        
        return comparison_files

# Example usage
if __name__ == "__main__":
    # Create base scenario
    base_scenario = MicrogridReportGenerator("Base Scenario")
    base_scenario.create_comprehensive_report()
    
    # Create alternative scenarios
    high_pv = MicrogridReportGenerator("High PV Scenario")
    high_pv.pv_capacity = 250
    high_pv.wind_capacity = 50
    high_pv.generate_data()
    high_pv.create_comprehensive_report()
    
    high_wind = MicrogridReportGenerator("High Wind Scenario")
    high_wind.pv_capacity = 50
    high_wind.wind_capacity = 250
    high_wind.generate_data()
    high_wind.create_comprehensive_report()
    
    high_storage = MicrogridReportGenerator("High Storage Scenario")
    high_storage.battery_capacity = 600
    high_storage.battery_power = 150
    high_storage.generate_data()
    high_storage.create_comprehensive_report()
    
    # Compare scenarios
    comparison_files = base_scenario.compare_scenarios([high_pv, high_wind, high_storage])
    
    print("Reports generated successfully!")
    print(f"Base scenario report: {base_scenario.report_dir}")
    print(f"Comparison report: {comparison_files}")
