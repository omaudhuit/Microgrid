# Microgrid

User's Manual for MicrogridReportGenerator
Overview
The MicrogridReportGenerator Python class generates comprehensive analytical reports and visualizations for microgrid systems, facilitating detailed analysis of energy flows, renewable energy performance, battery storage dynamics, and financial evaluations. It's designed for users aiming to simulate, assess, and optimize microgrid scenarios based on customizable parameters.
Prerequisites
Ensure the following libraries are installed:
numpy
pandas
matplotlib
seaborn
scipy
Install these packages using pip:
pip install numpy pandas matplotlib seaborn scipy
Initialization and Parameters
Instantiate the class by specifying a scenario name:
report = MicrogridReportGenerator(scenario_name="Custom Scenario")
Configurable Parameters
System Parameters:
pv_capacity (kW): Solar photovoltaic system capacity.
wind_capacity (kW): Wind turbine capacity.
battery_capacity (kWh): Total battery storage capacity.
battery_power (kW): Maximum battery charge/discharge power.
battery_efficiency: Round-trip battery efficiency (default 0.9).
battery_soc_min: Minimum allowed battery state-of-charge (SOC).
peak_load (kW): Peak system load.
Financial Parameters:
pv_cost_per_kw ($/kW): Installation cost for PV.
wind_cost_per_kw ($/kW): Installation cost for wind power.
battery_cost_per_kwh ($/kWh): Installation cost for battery.
om_cost_percent (%): Annual operations and maintenance cost as a percentage of capital.
project_lifetime (years): Expected project lifetime.
discount_rate: Discount rate for financial analysis.
grid_electricity_price ($/kWh): Electricity import price.
grid_export_price ($/kWh): Electricity export price.
Generating Reports
Generate all data and reports automatically:
report.create_comprehensive_report()
This will produce:
Energy flow analysis (energy_flows.png)
Battery SOC analysis (battery_soc.png)
Average daily profiles (daily_profiles.png)
Energy balance diagram (energy_balance.png)
Financial and LCOE breakdown (cost_analysis.png)
A summary text report (summary.txt)
Reports are stored in a dedicated directory named after the scenario, e.g., microgrid_report_Custom_Scenario.
Scenario Comparison
Compare multiple scenarios to evaluate variations in system performance:
base_scenario = MicrogridReportGenerator("Base Scenario")
high_pv = MicrogridReportGenerator("High PV")
high_pv.pv_capacity = 250
high_pv.generate_data()

comparison_files = base_scenario.compare_scenarios([high_pv])
This produces comparative visualizations saved in the microgrid_comparison folder, including spider charts, stacked bar charts for cost breakdown, and scenario comparisons of key metrics.
Customizing Visualizations
Modify plots by adjusting class attributes or directly editing plotting functions to refine visualization aesthetics or include additional details.
Troubleshooting
Missing directories or permissions: Ensure the program has the necessary permissions to create directories and save files.
Missing dependencies: Verify all required libraries are installed.
Data generation errors: Check input parameters for realistic values (e.g., avoid negative capacities).
Best Practices
Regularly back up scenario reports for comparison.
Clearly name scenarios to reflect parameter variations.
Utilize generated summary reports for quick assessments and decision-making.
This manual serves as a comprehensive guide for leveraging the MicrogridReportGenerator to effectively analyze and optimize microgrid systems.
