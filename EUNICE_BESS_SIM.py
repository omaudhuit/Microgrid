import math
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

##############################
# AFT_01 FUNCTIONS (Excel Calculator)
##############################
def IF(condition, true_val, false_val):
    return true_val if condition else false_val

def PMT(rate, nper, pv):
    if rate == 0:
        return pv / nper
    return (rate * pv) / (1 - (1 + rate) ** (-nper))

def compute_residential():
    # Input Values
    annual_consumption = 11536         # kWh - hard coded
    consumption_time = "Απόγευμα-Βράδυ"  # options: "Πρωί-Μεσημέρι" or "Απόγευμα-Βράδυ"
    supply_power = 35                  # kVA
    tariff_category = "Γ1"             # e.g., "Γ1" or "Γ1Ν"
    normal_tariff = 0.3                # €/kWh
    night_tariff = 0.08                # €/kWh
    region = "Δυτική Ελλάδα"           # region name
    roof_type = "Στέγη"                # roof type
    orientation = "ΝΔ"                 # orientation
    available_area = 75                # available roof area
    battery_choice = "Όχι"             # battery-saving effect flag
    battery_capacity_choice = 12       # user battery capacity selection
    storage_decision = "Ναι"           # whether storage is used

    # Intermediate Calculations
    I7 = IF(consumption_time == "Πρωί-Μεσημέρι", 0.7, 0.5)
    I8 = 1 - I7

    if region in ["Νότιο Αιγαίο", "Κρήτη"]:
        I11 = 1600
    elif region in ["Αττική", "Δυτική Ελλάδα", "Ιόνιοι Νήσοι"]:
        I11 = 1550
    elif region == "Πελοπόννησος":
        I11 = 1500
    elif region in ["Ήπειρος", "Θεσσαλία", "Στερεά Ελλάδα"]:
        I11 = 1450
    else:
        I11 = 1400

    if roof_type == "Στέγη":
        if orientation in ["Α", "Δ"]:
            I10 = 0.85
        elif orientation in ["ΝΑ", "ΝΔ"]:
            I10 = 0.93
        else:
            I10 = 0.95
    else:
        I10 = 1.0

    I12 = I11 * I10
    L7 = annual_consumption / I12
    L8 = 5 if supply_power <= 12 else 10.8
    L9 = min(annual_consumption / I12, L8)
    I9 = 0.7 if consumption_time == "Πρωί-Μεσημέρι" else 0.3
    I13 = 7 if roof_type == "Ταράτσα" else 5
    L10 = available_area / I13
    L11 = min(L9, L10)
    L12 = L11 * I12
    L14 = L12 * (1 - I9) / 365
    L15 = battery_capacity_choice
    L16 = L15 if battery_choice == "Όχι" else L14

    if tariff_category == "Γ1":
        L17 = (normal_tariff + 0.046) * 1.06 * annual_consumption
    elif tariff_category == "Γ1Ν":
        L17 = ((normal_tariff + 0.046) * I7 + (night_tariff + 0.01707) * I8) * 1.06 * annual_consumption
    else:
        L17 = None

    if tariff_category == "Γ1Ν":
        O7 = (normal_tariff * I7 + night_tariff * I8)
    else:
        O7 = normal_tariff
    O8 = 0.046
    O9 = (L12 * O7 + O8 * L12 * I9) * 1.06
    O10 = L16 * 365 * O8 * 1.06
    O11 = (O9 + O10) if storage_decision == "Ναι" else O9
    O15 = L17
    O16 = O15 - O9
    O17 = O11

    results = {
        "Annual Consumption (kWh)": annual_consumption,
        "I7 (Normal consumption fraction)": I7,
        "I8 (Reduced consumption fraction)": I8,
        "I11 (PV production factor by region)": I11,
        "I10 (Production coefficient)": I10,
        "I12 (Combined production factor)": I12,
        "L7 (Required inverter power)": L7,
        "L8 (Upper power limit)": L8,
        "L9 (Selected inverter power)": L9,
        "I9 (PV production factor based on time)": I9,
        "I13 (Area coefficient)": I13,
        "L10 (Area-based capacity)": L10,
        "L11 (Final inverter capacity)": L11,
        "L12 (Annual PV production, kWh)": L12,
        "L14 (Daily production for battery sizing)": L14,
        "L15 (Battery capacity selection)": L15,
        "L16 (Final battery capacity)": L16,
        "L17 (Baseline electricity cost)": L17,
        "O7 (Effective tariff)": O7,
        "O8 (Cost factor)": O8,
        "O9 (Estimated PV saving)": O9,
        "O10 (Additional saving with battery)": O10,
        "O11 (Total saving with system)": O11,
        "O15 (Baseline cost)": O15,
        "O16 (Cost saving difference)": O16,
        "O17 (Final expected saving)": O17
    }
    return results

def compute_loan():
    power = 8                # kW
    interest_rate = 0.06     # Annual interest rate
    repayment_years = 5      # Loan term in years
    battery_kWh = 12         # kWh
    annual_saving = 656      # €/year saving
    prepayment_fraction = 0.5

    if power <= 5:
        cost_per_kw = 2000
    elif power <= 10:
        cost_per_kw = 1500
    elif power <= 20:
        cost_per_kw = 1200
    elif power <= 50:
        cost_per_kw = 1000
    elif power <= 100:
        cost_per_kw = 850
    else:
        cost_per_kw = 750

    cost_per_kWh = 800  # for the battery

    pv_cost = power * cost_per_kw
    monthly_saving = annual_saving / 12
    battery_cost = cost_per_kWh * battery_kWh
    estimated_loan = pv_cost + battery_cost
    principal_for_loan = estimated_loan - prepayment_fraction * estimated_loan
    monthly_payment = PMT(interest_rate / 12, repayment_years * 12, principal_for_loan)
    half_loan = estimated_loan / 2

    results = {
        "Power (kW)": power,
        "Interest Rate": interest_rate,
        "Repayment Years": repayment_years,
        "Battery Capacity (kWh)": battery_kWh,
        "Annual Saving (€)": annual_saving,
        "Prepayment Fraction": prepayment_fraction,
        "Cost per kW (€)": cost_per_kw,
        "PV Cost (€)": pv_cost,
        "Cost per kWh (Battery, €)": cost_per_kWh,
        "Battery Cost (€)": battery_cost,
        "Estimated Loan (€)": estimated_loan,
        "Monthly Payment (€)": monthly_payment,
        "Half of Loan (€)": half_loan
    }
    return results

def compute_kwh_estimation():
    annual_electricity_cost_known = 900  # €/year
    tariff = 0.07                        # €/kWh
    annual_electricity_cost_unknown = 900
    consumer_category_known = "Κατοικία"
    consumer_category_unknown = "Κατοικία"

    if consumer_category_known == "Κατοικία":
        estimated_price_known = tariff + 0.05
    elif consumer_category_known == "Επιχείρηση χαμηλής τάσης":
        estimated_price_known = tariff + 0.04
    else:
        estimated_price_known = tariff + 0.035

    if consumer_category_unknown == "Κατοικία":
        estimated_price_unknown = 0.17
    elif consumer_category_unknown == "Επιχείρηση χαμηλής τάσης":
        estimated_price_unknown = 0.15
    else:
        estimated_price_unknown = 0.12

    consumption_known = annual_electricity_cost_known / estimated_price_known
    consumption_unknown = annual_electricity_cost_unknown / estimated_price_unknown

    results = {
        "Annual Electricity Cost (known, €)": annual_electricity_cost_known,
        "Tariff (€/kWh)": tariff,
        "Estimated Price (known, €/kWh)": estimated_price_known,
        "Estimated Annual Consumption (known, kWh)": consumption_known,
        "Annual Electricity Cost (unknown, €)": annual_electricity_cost_unknown,
        "Estimated Price (unknown, €/kWh)": estimated_price_unknown,
        "Estimated Annual Consumption (unknown, kWh)": consumption_unknown
    }
    return results

def compute_corporate():
    annual_consumption = 240000          # kWh
    consumption_time = "Απόγευμα-Βράδυ"
    tariff_category = "Γ1"
    normal_tariff = 0.3
    night_tariff = 0.08
    region = "Δυτική Ελλάδα"
    roof_type = "Στέγη"
    orientation = "ΝΔ"

    if consumption_time == "Πρωί-Μεσημέρι":
        I7 = 0.7
    elif consumption_time == "Απόγευμα-Βράδυ":
        I7 = 0.3
    else:
        I7 = 0.6
    I8 = 1 - I7

    if region in ["Νότιο Αιγαίο", "Κρήτη"]:
        I11 = 1600
    elif region in ["Αττική", "Δυτική Ελλάδα", "Ιόνιοι Νήσοι"]:
        I11 = 1550
    elif region == "Πελοπόννησος":
        I11 = 1500
    elif region in ["Ήπειρος", "Θεσσαλία", "Στερεά Ελλάδα"]:
        I11 = 1450
    else:
        I11 = 1400

    if roof_type == "Στέγη":
        if orientation in ["Α", "Δ"]:
            I10 = 0.85
        elif orientation in ["ΝΑ", "ΝΔ"]:
            I10 = 0.93
        else:
            I10 = 0.95
    else:
        I10 = 1.0

    I12 = I11 * I10
    L7 = annual_consumption / I12

    if tariff_category == "Γ1Ν":
        O7 = normal_tariff * I7 + night_tariff * I8
    else:
        O7 = normal_tariff

    O8 = 0.046

    if tariff_category == "Γ1":
        L17 = (normal_tariff + 0.046) * 1.06 * annual_consumption
    else:
        L17 = None

    results = {
        "Annual Consumption (kWh)": annual_consumption,
        "I7": I7,
        "I8": I8,
        "I11": I11,
        "I10": I10,
        "I12": I12,
        "L7": L7,
        "O7 (Effective Tariff)": O7,
        "O8": O8,
        "Estimated Baseline Cost (L17)": L17
    }
    return results

##############################
# MICROGRID REPORT GENERATOR CLASS (from EUNICE_BESS_SIM)
##############################
class MicrogridReportGenerator:
    """
    Generate reports and visualizations for microgrid analysis.
    """
    def __init__(self, scenario_name="Base Scenario", load_data_file=None):
        self.scenario_name = scenario_name
        self.days = 7  # default simulation period (days)
        self.hours = self.days * 24
        self.time_range = np.linspace(0, self.days, self.hours)
        self.dates = [dt.datetime(2025, 1, 1) + dt.timedelta(hours=h) for h in range(self.hours)]
        
        # System parameters
        self.pv_capacity = 150   # kW
        self.wind_capacity = 100 # kW
        self.battery_capacity = 300  # kWh
        self.battery_power = 75  # kW
        self.battery_efficiency = 0.9
        self.battery_soc_min = 0.1
        self.battery_soc_initial = 0.5
        self.peak_load = 180      # kW
        self.pv_cost_per_kw = 1000
        self.wind_cost_per_kw = 1500
        self.battery_cost_per_kwh = 400
        self.om_cost_percent = 2
        self.project_lifetime = 25
        self.discount_rate = 0.06
        self.grid_electricity_price = 0.12
        self.grid_export_price = 0.05

        self.original_daily_load = None
        self.df_load = None

        if load_data_file is not None:
            try:
                df_load = pd.read_excel(load_data_file, engine='openpyxl')
                st.write("Excel file preview:", df_load.head())
                self.df_load = df_load.copy()
                daily_load = df_load.iloc[:, 2:].sum(axis=0).values
                date_index = pd.date_range(start="2025-01-01", periods=24, freq="H")
                self.original_daily_load = pd.DataFrame({'load': daily_load}, index=date_index)
            except Exception as e:
                st.error(f"Error reading load data file: {e}")
                self.original_daily_load = None
        self.generate_data()
        self.report_dir = f"microgrid_report_{scenario_name.replace(' ', '_')}"
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_data(self):
        self.generate_solar_profile()
        self.generate_wind_profile()
        self.generate_load_profile()
        self.run_simulation()
        self.calculate_lcoe()

    def generate_solar_profile(self):
        hours_of_day = np.array([h % 24 for h in range(self.hours)])
        days = np.array([h // 24 for h in range(self.hours)])
        solar_factor = np.maximum(0, np.sin(np.pi * (hours_of_day - 6) / 12))
        daily_factor = 1.0 - 0.3 * np.sin(days * 1.5)
        daily_factor = np.repeat(daily_factor, 24)[:self.hours]
        cloud_events = np.random.randint(0, self.hours, 5)
        for event in cloud_events:
            duration = np.random.randint(2, 6)
            intensity = np.random.uniform(0.3, 0.8)
            for i in range(duration):
                if event + i < self.hours:
                    daily_factor[event + i] *= intensity
        self.pv_profile = self.pv_capacity * solar_factor * daily_factor

    def generate_wind_profile(self):
        hours_of_day = np.array([h % 24 for h in range(self.hours)])
        days = np.array([h // 24 for h in range(self.hours)])
        wind_daily = 0.7 + 0.3 * np.sin(np.pi * (hours_of_day - 18) / 12)
        daily_factor = 0.6 + 0.4 * np.sin(days * 0.8 + 2)
        daily_factor = np.repeat(daily_factor, 24)[:self.hours]
        for _ in range(10):
            start = np.random.randint(0, self.hours - 12)
            duration = np.random.randint(3, 12)
            factor = np.random.uniform(1.2, 1.5) if np.random.rand() > 0.5 else np.random.uniform(0.3, 0.7)
            for i in range(duration):
                if start + i < self.hours:
                    daily_factor[start + i] *= factor
        daily_factor = np.convolve(daily_factor, np.ones(3)/3, mode='same')
        self.wind_profile = np.maximum(0, self.wind_capacity * wind_daily * daily_factor)

    def generate_load_profile(self):
        if self.original_daily_load is not None:
            daily_load = self.original_daily_load['load'].values
            self.load_profile = np.tile(daily_load, self.days)
        else:
            hours_of_day = np.array([h % 24 for h in range(self.hours)])
            days = np.array([h // 24 for h in range(self.hours)])
            days_of_week = np.array([d % 7 for d in days])
            control_points_x = [0, 6, 9, 12, 15, 18, 21, 24]
            control_points_y = [0.4, 0.35, 0.65, 0.8, 0.75, 0.95, 0.7, 0.4]
            cs = CubicSpline(control_points_x, control_points_y, bc_type='periodic')
            daily_pattern = cs(hours_of_day)
            weekday_factor = np.ones(self.hours)
            for h in range(self.hours):
                if days_of_week[h] >= 5:
                    weekday_factor[h] = 0.8
            noise = 0.05 * np.random.randn(self.hours)
            self.load_profile = self.peak_load * daily_pattern * weekday_factor * (1 + noise)
            self.load_profile = np.maximum(0, self.load_profile)

    def run_simulation(self):
        self.grid_import = np.zeros(self.hours)
        self.grid_export = np.zeros(self.hours)
        self.battery_charge = np.zeros(self.hours)
        self.battery_discharge = np.zeros(self.hours)
        self.battery_soc = np.zeros(self.hours)
        self.battery_soc[0] = self.battery_soc_initial * self.battery_capacity

        for h in range(self.hours):
            renewable_gen = self.pv_profile[h] + self.wind_profile[h]
            net_load = self.load_profile[h] - renewable_gen
            if h > 0:
                self.battery_soc[h] = self.battery_soc[h-1]
            if net_load > 0:
                max_discharge = min(self.battery_soc[h] - self.battery_soc_min * self.battery_capacity, self.battery_power)
                discharge = min(max_discharge, net_load)
                if discharge > 0:
                    self.battery_discharge[h] = discharge
                    self.battery_soc[h] -= discharge
                    net_load -= discharge
                if net_load > 0:
                    self.grid_import[h] = net_load
            else:
                max_charge = min((self.battery_capacity - self.battery_soc[h]) / self.battery_efficiency, self.battery_power)
                charge = min(max_charge, -net_load)
                if charge > 0:
                    self.battery_charge[h] = charge
                    self.battery_soc[h] += charge * self.battery_efficiency
                    net_load += charge
                if net_load < 0:
                    self.grid_export[h] = -net_load

        self.total_load = np.sum(self.load_profile)
        self.pv_generation = np.sum(self.pv_profile)
        self.wind_generation = np.sum(self.wind_profile)
        self.total_grid_import = np.sum(self.grid_import)
        self.total_grid_export = np.sum(self.grid_export)
        total_renewable_used = (self.pv_generation + self.wind_generation - self.total_grid_export)
        self.renewables_fraction = min(100, 100 * total_renewable_used / self.total_load)
        self.self_sufficiency = 100 * (self.total_load - self.total_grid_import) / self.total_load
        self.battery_cycles = np.sum(self.battery_discharge) / self.battery_capacity

    def calculate_lcoe(self):
        pv_capex = self.pv_capacity * self.pv_cost_per_kw
        wind_capex = self.wind_capacity * self.wind_cost_per_kw
        battery_capex = self.battery_capacity * self.battery_cost_per_kwh
        total_capex = pv_capex + wind_capex + battery_capex
        annual_om = total_capex * (self.om_cost_percent / 100)
        annual_grid_cost = self.total_grid_import * self.grid_electricity_price * (365 / self.days)
        annual_export_revenue = self.total_grid_export * self.grid_export_price * (365 / self.days)
        annual_net_grid = annual_grid_cost - annual_export_revenue
        npv_factor = sum(1 / ((1 + self.discount_rate) ** year) for year in range(1, self.project_lifetime + 1))
        npv_costs = total_capex + (annual_om + annual_net_grid) * npv_factor
        annual_energy = self.total_load * (365 / self.days)
        lifetime_energy = annual_energy * self.project_lifetime
        self.lcoe = npv_costs / lifetime_energy if lifetime_energy > 0 else 0
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

    # The following methods create various plots.
    def create_energy_flow_plot(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        dates = self.dates
        ax.plot(dates, self.load_profile, 'k-', label='Load', linewidth=2)
        ax.plot(dates, self.pv_profile, color='orange', label='Solar PV')
        ax.plot(dates, self.wind_profile, color='blue', label='Wind')
        ax.plot(dates, self.pv_profile + self.wind_profile, 'g--', label='Total Renewable', alpha=0.7)
        ax.plot(dates, self.grid_import, 'r-', label='Grid Import')
        ax.plot(dates, -self.grid_export, 'g-', label='Grid Export')
        ax.fill_between(dates, 0, self.pv_profile, color='orange', alpha=0.3)
        ax.fill_between(dates, self.pv_profile, self.pv_profile + self.wind_profile, color='blue', alpha=0.3)
        charge_mask = self.battery_charge > 0
        discharge_mask = self.battery_discharge > 0
        if np.any(charge_mask):
            ax.scatter(np.array(dates)[charge_mask], self.battery_charge[charge_mask],
                       marker='^', color='green', s=50, label='Battery Charging')
        if np.any(discharge_mask):
            ax.scatter(np.array(dates)[discharge_mask], self.battery_discharge[discharge_mask],
                       marker='v', color='purple', s=50, label='Battery Discharging')
        ax.set_xlabel('Date')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'Microgrid Energy Flows - {self.scenario_name}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig(f"{self.report_dir}/energy_flows.png")
        return fig

    def create_battery_soc_plot(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        dates = self.dates
        soc_percent = 100 * self.battery_soc / self.battery_capacity
        cmap = plt.cm.viridis
        norm = plt.Normalize(0, 100)
        for i in range(len(dates)-1):
            ax.fill_between([dates[i], dates[i+1]], [0, 0], [soc_percent[i], soc_percent[i+1]],
                            color=cmap(norm(soc_percent[i])))
        ax.plot(dates, soc_percent, 'k-', linewidth=2, label='SOC')
        ax.axhline(y=self.battery_soc_min * 100, color='r', linestyle='--', alpha=0.7, 
                   label=f'Min SOC ({self.battery_soc_min*100:.0f}%)')
        charge_mask = self.battery_charge > 0
        discharge_mask = self.battery_discharge > 0
        if np.any(charge_mask):
            ax.scatter(np.array(dates)[charge_mask], soc_percent[charge_mask],
                      marker='^', color='lime', s=50, label='Charging')
        if np.any(discharge_mask):
            ax.scatter(np.array(dates)[discharge_mask], soc_percent[discharge_mask],
                      marker='v', color='red', s=50, label='Discharging')
        ax.set_xlabel('Date')
        ax.set_ylabel('State of Charge (%)')
        ax.set_title(f'Battery State of Charge - {self.scenario_name}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig(f"{self.report_dir}/battery_soc.png")
        return fig

    def perform_sensitivity_analysis(self, pv_range, wind_range, battery_range):
        results = []
        for pv in pv_range:
            for wind in wind_range:
                for battery in battery_range:
                    sim = MicrogridReportGenerator("Sensitivity")
                    sim.days = self.days
                    sim.hours = self.days * 24
                    sim.time_range = np.linspace(0, self.days, sim.hours)
                    sim.dates = [dt.datetime(2025, 1, 1) + dt.timedelta(hours=h) for h in range(sim.hours)]
                    sim.pv_capacity = pv
                    sim.wind_capacity = wind
                    sim.battery_capacity = battery
                    sim.battery_power = self.battery_power
                    sim.peak_load = self.peak_load
                    sim.generate_data()
                    results.append((pv, wind, battery, sim.lcoe))
        df = pd.DataFrame(results, columns=['PV Capacity', 'Wind Capacity', 'Battery Capacity', 'LCOE'])
        return df

    def create_load_data_plot(self):
        if self.original_daily_load is not None:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(self.original_daily_load.index, self.original_daily_load['load'], 'b-', label='Total Daily Load')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Load (kW)')
            ax.set_title('Total Daily Load Data from Excel')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(f"{self.report_dir}/load_data_total.png")
            return fig
        else:
            return None

    def create_station_load_plot(self):
        if self.df_load is not None:
            fig, ax = plt.subplots(figsize=(14, 6))
            hours = np.arange(24)
            total_load = np.zeros(24)
            for i, row in self.df_load.iterrows():
                station_name = row.iloc[0]
                station_load = row.iloc[2:].astype(float).values
                total_load += station_load
                ax.plot(hours, station_load, label=str(station_name), alpha=0.6, linestyle='--')
            ax.plot(hours, total_load, label='Total Load', color='black', linewidth=3)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Load (kW)")
            ax.set_title("24-Hour Load Curves by Station and Total")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.xticks(hours)
            plt.tight_layout()
            fig.savefig(f"{self.report_dir}/station_load_curves.png")
            return fig
        else:
            return None

##############################
# MERGED STREAMLIT APPLICATION
##############################
def main():
    st.title("Merged Streamlit Application: Microgrid Analysis & AFT_01 Calculations")
    
    # Sidebar common inputs for the microgrid report section
    load_file = st.sidebar.file_uploader("Upload Load Data Excel (with station data)", type=['xlsx'])
    scenario_option = st.sidebar.selectbox(
        "Select Scenario",
        ["Custom", "Base Scenario", "High PV Scenario", "High Wind Scenario", "High Storage Scenario"]
    )
    simulation_days = st.sidebar.number_input("Simulation Days", min_value=1, value=7)
    
    # Create four main tabs
    tabs = st.tabs(["Load Data", "Simulation", "Sensitivity Analysis", "AFT_01"])
    
    # ---- Load Data Tab ----
    with tabs[0]:
        st.subheader("Load Data Overview")
        scenario_for_loads = MicrogridReportGenerator("Custom", load_data_file=load_file)
        if scenario_for_loads.df_load is not None:
            st.write("### Full Excel Load Data")
            st.dataframe(scenario_for_loads.df_load)
            st.write("### Total Daily Load Data Curve")
            total_load_fig = scenario_for_loads.create_load_data_plot()
            if total_load_fig is not None:
                st.pyplot(total_load_fig)
            st.write("### Station Load Curves")
            station_load_fig = scenario_for_loads.create_station_load_plot()
            if station_load_fig is not None:
                st.pyplot(station_load_fig)
        else:
            st.info("No load data provided or loading failed.")
    
    # ---- Simulation Tab ----
    with tabs[1]:
        st.subheader("Run Simulation and Generate Report")
        if st.button("Run Simulation"):
            with st.spinner("Running simulation and generating report..."):
                scenario = MicrogridReportGenerator("Custom", load_data_file=load_file)
                scenario.days = simulation_days
                scenario.hours = simulation_days * 24
                scenario.time_range = np.linspace(0, simulation_days, scenario.hours)
                scenario.dates = [dt.datetime(2025, 1, 1) + dt.timedelta(hours=h) for h in range(scenario.hours)]
                scenario.generate_data()
                st.write("### Energy Flow Diagram")
                fig1 = scenario.create_energy_flow_plot()
                st.pyplot(fig1)
                st.write("### Battery State of Charge")
                fig2 = scenario.create_battery_soc_plot()
                st.pyplot(fig2)
    
    # ---- Sensitivity Analysis Tab ----
    with tabs[2]:
        st.subheader("Sensitivity Analysis")
        if st.button("Run Sensitivity Analysis"):
            pv_range = st.slider("PV Capacity Range", 50, 300, (150, 250))
            wind_range = st.slider("Wind Capacity Range", 50, 300, (100, 200))
            battery_range = st.slider("Battery Capacity Range", 100, 600, (300, 500))
            scenario = MicrogridReportGenerator("Custom", load_data_file=load_file)
            scenario.days = simulation_days
            scenario.hours = simulation_days * 24
            scenario.time_range = np.linspace(0, simulation_days, scenario.hours)
            scenario.dates = [dt.datetime(2025, 1, 1) + dt.timedelta(hours=h) for h in range(scenario.hours)]
            scenario.generate_data()
            df_sens = scenario.perform_sensitivity_analysis(
                range(pv_range[0], pv_range[1]+1, 50),
                range(wind_range[0], wind_range[1]+1, 50),
                range(battery_range[0], battery_range[1]+1, 100)
            )
            st.write("### Sensitivity Analysis Results")
            st.dataframe(df_sens)
    
    # ---- AFT_01 Tab ----
    with tabs[3]:
        st.subheader("AFT_01 Calculations")
        calc_type = st.radio("Select Calculation", ("Residential", "Loan", "kWh Estimation", "Corporate"))
        if calc_type == "Residential":
            st.write("### Residential Consumers Data")
            results = compute_residential()
            st.write(results)
        elif calc_type == "Loan":
            st.write("### Loan Calculation Data")
            results = compute_loan()
            st.write(results)
        elif calc_type == "kWh Estimation":
            st.write("### kWh Estimation Data")
            results = compute_kwh_estimation()
            st.write(results)
        elif calc_type == "Corporate":
            st.write("### Corporate Consumers Data")
            results = compute_corporate()
            st.write(results)

if __name__ == "__main__":
    main()
