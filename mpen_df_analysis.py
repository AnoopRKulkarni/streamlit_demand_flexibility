#!/usr/bin/env python
# coding: utf-8

#
# MPEN Demand Flexibility Tool Analysis Page
# Authors: Anoop R Kulkarni, Kaustubh Arekar
# Version 1.0
# Dec 11, 2023

#@title Import libraries
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd

from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.subplots as sp

import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def app(site):
   #@title Define file paths
   INPUT_PATH = 'input/'+site['prefix']+'/'
   OUTPUT_PATH = 'output/'+site['prefix']+'/'
    
   #if not os.path.exists(INPUT_PATH):
   #   os.makedirs(INPUT_PATH) 
   #if not os.path.exists(OUTPUT_PATH):
   #   os.makedirs(OUTPUT_PATH)
    
   #@title Read Data 
   # Read end use assumption of all consumer types
   st.header("Consumer stratification")

   base_base = {}
   base_base['residential'] = pd.read_excel(INPUT_PATH + 'residential_asp.xlsx').iloc[:,:]
   base_base['hospital'] = pd.read_excel(INPUT_PATH + 'hospital_asp.xlsx').iloc[:,:]
   base_base['hotel'] = pd.read_excel(INPUT_PATH + 'hotel_asp.xlsx').iloc[:,:]
   base_base['mall'] = pd.read_excel(INPUT_PATH + 'mall_asp.xlsx').iloc[:,:]
   base_base['office'] = pd.read_excel(INPUT_PATH + 'office_asp.xlsx').iloc[:,:]
   base_base['IT'] = pd.read_excel(INPUT_PATH + 'IT_asp.xlsx').iloc[:,:]
   base_base['coldstorage'] = pd.read_excel(INPUT_PATH + 'cold_asp.xlsx').iloc[:,:]
   base_base['pww'] = pd.read_excel(INPUT_PATH + 'pww_asp.xlsx').iloc[:,:]
   
   pu_load_res=base_base['residential']['Total load'].max()
   pu_load_hospital=base_base['hospital']['Total load'].max()
   pu_load_hotel=base_base['hotel']['Total load'].max()
   pu_load_mall=base_base['mall']['Total load'].max()
   pu_load_office=base_base['office']['Total load'].max()
   pu_load_IT=base_base['IT']['Total load'].max()
   pu_load_coldstorage=base_base['coldstorage']['Total load'].max()
   pu_load_pww=base_base['pww']['Total load'].max()
    
   rsd = st.slider('Peak load residential consumer (MW): ', 5000, 10000, 5600)
   hsp = st.slider('Peak load hospital consumer (MW): ', 5000, 10000, 5600)
   ht = st.slider('Peak load hotel consumer (MW): ', 5000,10000,7700)
   mll = st.slider('Peak load mall consumer (MW): ', 1000, 5000, 2000) 
   off = st.slider('Peak load office consumer (MW): ', 1000, 5000, 2500)
   ito = st.slider('Peak load IT office consumer (MW): ', 1000, 5000, 2600)
   cso = st.slider('Peak load cold storage consumer (MW): ', 1000, 5000, 1000)
   pwwo = st.slider('Peak load pww consumer (MW): ', 2000, 5000, 2600)
   
   # Total number of consumers part of the feeder/ utility
   num_consumers = pd.DataFrame([{'residential': rsd*1000/pu_load_res, 
                                 'hospital': hsp*1000/pu_load_hospital, 
                                  'hotel': ht*1000/pu_load_hotel, 
                                  'mall': mll*1000/pu_load_mall, 
                                  'office': off*1000/pu_load_mall, 
                                  'IT': ito*1000/pu_load_IT,
                                  'coldstorage':cso*1000/pu_load_coldstorage,
                                  'pww':pwwo*1000/pu_load_pww}
                               ])
    
   RE_PPA = pd.read_excel(INPUT_PATH + 'RE power PPA.xlsx') 
   
   # PPA data for conventional plants   
   ppa =pd.read_excel(INPUT_PATH + 'MODCON1.xlsx')
    
   # Read tariff plan
   tariff = pd.read_excel(INPUT_PATH + 'tariffplan.xlsx')
   
   # Solar Rooftop
   solarperkw = pd.read_excel(INPUT_PATH + 'solarrooftop.xlsx')

   #RE integration level for base case it is 1
   RE_integration =1
   
   # Tariff types used in analysis
   tariff_types = ['residential', 'commercial']
    
   # tariff plans
   tariff_plan = pd.DataFrame()
   tariff_plan['residential'] = tariff[tariff.columns[tariff.columns.str.contains('Residential')]]
   tariff_plan['commercial'] = tariff[tariff.columns[tariff.columns.str.contains('Commercial')]]
    
   sectors = ['LT-COMMERCIAL', 'LT-DOMESTIC', 'HT Commercial 22 KV']
   # Only first feeder
   # num_consumers = [cp.loc[cp['Feeder Name'].str.contains(s)].values[0,2] for s in sectors]
    
   customer_segments = ['residential','hospital','hotel','mall','office','IT','coldstorage','pww']
    
   seg_tariff = {
       'residential': 'residential',
       'hospital': 'commercial',
       'hotel': 'commercial',
       'mall': 'commercial',
       'office': 'commercial',
       'IT': 'commercial',
       'coldstorage':'commercial',
       'pww': 'commercial'
       
   }
    
   rs=0.05
   hss=0.1
   hts=0.1
   ms=0.1
   os=0.1
   iss=0.1
   cs=0.05
   pwws = 0.05
    
   # Fractions of consumers with solar installations
   fs = pd.DataFrame({'solar': {'residential':rs,
                                'hospital':hss,
                                'hotel':hts,
                                'mall':ms,
                                'office':os,
                                'IT':iss,
                                'coldstorage':cs,
                                'pww':pwws},
                                  
                      'nonsolar': {'residential':1-rs,
                                   'hospital':1-hss,
                                   'hotel':1-hts,
                                   'mall':1-ms,
                                   'office':1-os,
                                   'IT':1-iss,
                                   'coldstorage':1-cs,
                                   'pww':1-pwws}
                       })
    
   solar_df= 0.3
   
   rpd=solar_df
   hspd=solar_df
   htpd=solar_df
   mpd=solar_df
   opd=solar_df
   ispd=solar_df
   csd=solar_df
   pwwd=solar_df
   
   non_solar_df = 0.1
   
   rnpd=non_solar_df
   hnspd=non_solar_df
   hntpd=non_solar_df
   mnpd=non_solar_df
   onpd=non_solar_df
   isnpd=non_solar_df
   csnd=non_solar_df
   pwwnd=non_solar_df
   
   # Fractions of consumers participating in the DF program
   fp = pd.DataFrame({'solar': {'residential':rpd,
                                'hospital':hspd,
                                'hotel':htpd,
                                'mall':mpd,
                                'office':opd,
                                'IT':ispd,
                                'coldstorage':csd,
                                'pww':pwwd},
                     'nonsolar': {'residential':rnpd,
                                'hospital':hnspd,
                                'hotel':hntpd,
                                'mall':mnpd,
                                'office':onpd,
                                'IT':isnpd,
                                'coldstorage':csnd,
                                'pww':pwwnd},
                       })
    
   # Assumed installed solar capacity in KW
   solar_capacity = pd.DataFrame([{'residential': 1, 
                                   'hospital': 100, 
                                   'hotel':100, 
                                   'mall':100, 
                                   'office':100, 
                                   'IT':100,
                                   'coldstorage':50,
                                   'pww':10}
                               ])
    
   # T&D and other losses
   losses = 0.2
    
   # Factors for incremental steps during iterations
   lambda_t = {'residential': 0.05, 'commercial': 0.07}
    
   # Tariff threholds at 20% of max and min
   tariff_max = {'residential': 1.2*max(tariff_plan['residential']), 
                 'commercial':1.2*max(tariff_plan['commercial'])}
   tariff_min = {'residential': 0.8*min(tariff_plan['residential']), 
                 'commercial':0.8*min(tariff_plan['commercial'])}
    
   max_iters = 20
   epsilon = 0.005

   base_loads = {}
   base_solar = {}
   base_loads_solar = {}

   for segment in customer_segments: 
       base_loads[segment] = base_base[segment]['Total load'].copy()
       base_solar[segment] = solar_capacity[segment].values[0]*solarperkw.iloc[:,1]
       base_loads_solar[segment] = base_loads[segment].subtract(base_solar[segment], axis=0)

   # Convert lists to Pandas dataframes
   base_loads_df = pd.DataFrame(base_loads)
   base_loads_solar_df = pd.DataFrame(base_loads_solar)

   # compute number of consumers in each category, solar/nonsolar and participating solar/nonsolar
   solar_consumers = fs['solar']*num_consumers
   nonsolar_consumers = fs['nonsolar']*num_consumers
   participating_solar_consumers = fp['solar']*solar_consumers
   participating_nonsolar_consumers = fp['nonsolar']*nonsolar_consumers

   #@title Define analytical model
   # Functions of the formulation to compute score, capex, opex and margin
   
   def aggregate_base_load():    
       df = {}
   
       df['nonsolar'] = {}
       df['solar'] = {}
       df['total'] = pd.DataFrame()
   
       total_bill = 0
   
       for segment in customer_segments:
           df['nonsolar'][segment] = base_loads[segment].copy()
           df['solar'][segment] = base_loads_solar[segment].copy()
           df['nonsolar'][segment] *= nonsolar_consumers[segment].values[0]
           df['solar'][segment] *= solar_consumers[segment].values[0]
           df['total'][segment] = df['nonsolar'][segment] + df['solar'][segment]
           
           # compute the bill
           total_bill += sum(df['solar'][segment] * tariff_plan[seg_tariff[segment]])
           total_bill += sum(df['nonsolar'][segment] * tariff_plan[seg_tariff[segment]])
           
           # convert total units from KW to MW
           df['total'][segment] /= 1e3
       df['total']['agg'] = df['total'].sum(axis=1)
       return df, total_bill

   def compute_gen_curve(projected_demand, n=0):    
       #RE Power generation profiles
       
       RE1=pd.read_excel(INPUT_PATH + 'RE.xlsx')
       solar=RE_integration*RE1['Solar']
       wind=RE1['Wind']
       Solar_pu_cost = RE_PPA.iloc[0,0]
       Wind_pu_cost = RE_PPA.iloc[0,1]
       
       RE=solar+wind
   
       # Net demand for conventional genration
       demand=projected_demand.values-RE
   
       min_demand=min(demand)
       max_demand=max(demand)
       num_plants=len(ppa['Capacity'])
       i=1
       capacity =0
       #technical minimum % considered in the analysis
   
       for p in range(num_plants):
           capacity = ppa.iloc[p,1]+capacity
           if capacity>=max_demand:
               break
           else:
               i=i+1
   
       # print(i,min_demand/max_demand) # should be greater than technical minimum considered for analysis
       ppa.iloc[i:len(ppa)-1,1]=0
       
       #...............................
       #print(sum(ppa.iloc[0:num_plants,1]))
   
       generation_capacity = ppa['Capacity']
       technical_minimum_threshold = .55
       technical_minimum=generation_capacity*technical_minimum_threshold
       technical_minimum.iloc[-1]=0 # for the last unconstrained plan
       #print(technical_minimum)
       fixed_cost = ppa['Fixed cost']  # Fixed cost for each plant
       Variable_cost=ppa['Variable cost']
   
       ramping_up = ppa['Ramping_up'] # Ramping up limit for each plant
       ramping_down = ppa['Ramping down']  # Ramping down limit for each plant
       
       # print(technical_minimum)
       num_plants=len(generation_capacity)
       num_hours=len(projected_demand.values)
   
       problem = LpProblem("Power Generation Optimization", LpMinimize)
   
       # Define the decision variables
       schedule = [[LpVariable(f"Schedule_{t}_{p}", lowBound=0) for p in range(num_plants)] 
                   for t in range(num_hours)]
       
       # Set the objective function
       problem += lpSum(schedule[t][p] * Variable_cost[p] 
                       for t in range(num_hours) for p in range(num_plants))
   
       # Add the constraints
       #ramping up and down constraints 1% for thermal, 3% for gas, 10% for hydro
       for t in range(num_hours - 1):
           for p in range(num_plants):
               del_schedule = schedule[t+1][p] - schedule[t][p]
               problem += ramping_down[p] <= del_schedule <= ramping_up[p]
   
       # technical Minimum
       for t in range(num_hours):
           for p in range(num_plants):
               x=schedule[t][p]
               problem += x>=technical_minimum[p]
   
       # Total generation in slot i = Total demand in slot i
       for t in range(num_hours):
           problem += lpSum(schedule[t][p] for p in range(num_plants)) ==1* demand[t]
   
       # maximum generation limitaion max generation by plant i 
       # in any slot < = max generation capacity of plant i
       for t in range(num_hours):
           for p in range(num_plants):
               problem += schedule[t][p] <= ppa['Capacity'][p]
   
       # Solve the problem
       problem.solve()
   
       # Check the status of the solution
       print('-----------------')
       print("Generation optimisation Status:", LpStatus[problem.status])  
   
       # saving variable values in dataframe
       schedule_gen = pd.DataFrame(index=range(num_hours), columns=range(num_plants))
       for t in range(num_hours):
           for p in range(num_plants):
               schedule_gen.at[t, p] = schedule[t][p].varValue
   
       # dataframe to excel
       schedule_gen.to_csv(OUTPUT_PATH + "schedule_output_" + str(n) + ".csv", index=False)
   
       # cost of generation
       daily_generation=schedule_gen.sum()*(24/num_hours)
       daily_generation_2=schedule_gen.sum(axis=1)
   
       # RE cost
       solar_cost=Solar_pu_cost*solar
       wind_cost=wind*Wind_pu_cost
       RE_cost = solar_cost+wind_cost
   
       ## Required output ............................
       # Slot wise total cost INR / conventional + RE
       slot_wise_cost=(schedule_gen.dot(np.array(Variable_cost)) +RE_cost)
       slot_wise_cost1 = slot_wise_cost*1000*24/num_hours
       #...................................................................
       # Slot wise PU cost INR/kWh
       slot_wise_pu=slot_wise_cost1/(1000*(demand+RE)*24/num_hours)      
   
       return slot_wise_cost, slot_wise_cost1, slot_wise_pu 

   def optimal_load_curve1(tariff1, load):
   
       tariff=pd.DataFrame()
       wh_load = load.iloc[:, 1:6]
       wp_load = load.iloc[:, 6:11]
       b_charge = load.iloc[:, 11:17]
       b_discharge = load.iloc[:, 17:23]
       hvac_max = load.iloc[:, 23]
       hvac_min = 0.8 * hvac_max
       fixed_load = load.iloc[:, 24]
      
       tariff['Tariff']=tariff1
       
       hours = len(fixed_load)
       wpp = len(wp_load.columns)
       whp = len(wh_load.columns)
       bcp = len(b_charge.columns)
       bdp = len(b_discharge.columns)
   
       # Create the LP problem
       Bill_optimization = LpProblem("Bill_optimization", LpMinimize)
   
       # Variables
       hvac = [LpVariable(f"{i}_hvac", lowBound=hvac_min[i]) for i in range(hours)]
   
       shift_pump = [LpVariable(f"Shift_pump_{i}", lowBound=0, cat='Binary') for i in range(wpp)]
       shift_wh = [LpVariable(f"Shift_wh_{i}", lowBound=0, cat='Binary') for i in range(whp)]
       shift_bc = [LpVariable(f"Shift_bc_{i}", lowBound=0, cat='Binary') for i in range(bcp)]
       shift_bd = [LpVariable(f"Shift_bd_{i}", lowBound=0, cat='Binary') for i in range(bdp)]
   
       # Constraints
       Bill_optimization += lpSum(shift_pump) == 1
       Bill_optimization += lpSum(shift_wh) == 1
       Bill_optimization += lpSum(shift_bc) == 1
       Bill_optimization += lpSum(shift_bd) == 1
   
       for t in range(hours):
           Bill_optimization += hvac_min[t] <= hvac[t] <= 1.2*hvac_max[t]
   
       Bill_optimization += lpSum(hvac[t] for t in range(hours)) >= 1* hvac_max.sum()
   
       # Set the objective function
       Bill_optimization += lpSum((hvac[t] + wh_load.iloc[t, :].dot(shift_wh) + wp_load.iloc[t, :].dot(shift_pump) + b_charge.iloc[t, :].dot(shift_bc) + b_discharge.iloc[t, :].dot(shift_bd)) * tariff['Tariff'][t] for t in range(hours))
   
       # Solve the problem
       Bill_optimization.solve()
   
       # Check the status of the solution
       #print("Status:", LpStatus[Bill_optimization.status])
   
       # Variables to DataFrames
       hvac_sc = pd.DataFrame(index=range(len(hvac)), columns=range(1))
       wp_sc = pd.DataFrame(index=range(len(shift_pump)), columns=range(1))
       wh_sc = pd.DataFrame(index=range(len(shift_wh)), columns=range(1))
       bc_sc = pd.DataFrame(index=range(len(shift_bc)), columns=range(1))
       bd_sc = pd.DataFrame(index=range(len(shift_bd)), columns=range(1))
   
       for i in range(len(hvac)):
           hvac_sc.iloc[i,0]=hvac[i].varValue
       
       for i in range(len(shift_pump)):
           wp_sc.iloc[i,0]=shift_pump[i].varValue
           if wp_sc.iloc[i,0]==1:
               wpsc_n = i
               
       for i in range(len(shift_wh)):
           wh_sc.iloc[i,0]=shift_wh[i].varValue
           if wh_sc.iloc[i,0]==1:
               whsc_n = i
               
       for i in range(len(shift_bc)):
           bc_sc.iloc[i,0]=shift_bc[i].varValue
           if bc_sc.iloc[i,0]==1:
               bcsc_n = i
               
       for i in range(len(shift_bd)):
           bd_sc.iloc[i,0]=shift_bd[i].varValue
           if bd_sc.iloc[i,0]==1:
               bdsc_n = i
       #-------------------------------------------------------------------
       ## output = load profile    
       load_profile =fixed_load.iloc[:]+b_charge.iloc[:,bcsc_n]+b_discharge.iloc[:,bdsc_n]+wp_load.iloc[:,wpsc_n]+wh_load.iloc[:,whsc_n]+hvac_sc.iloc[0:24,0]
       return load_profile
   
   def compute_optimal_load_curve1(pricing_signal):
       optimal_load_curve = pd.DataFrame()
       total_bill = 0
   
       #print(base_loads_df, nonsolar_consumers)
       a = base_loads_df * (nonsolar_consumers - participating_nonsolar_consumers).iloc[0] + \
           base_loads_solar_df * (solar_consumers - participating_solar_consumers).iloc[0]
   
       b = pd.DataFrame()
       c = pd.DataFrame()
   
       for s in customer_segments:   
           base_assumptions = base_base[s]
           tarrif= pricing_signal[seg_tariff[s]]
           #print(tarrif, base_assumptions)
           optimal_load=optimal_load_curve1(tarrif,base_assumptions)
          
           b[s] = optimal_load * participating_nonsolar_consumers[s].values[0]
           c[s] = (optimal_load - base_solar[s]) * participating_solar_consumers[s].values[0]
           
           # compute the bill
           total_bill += sum(a[s] * tariff_plan[seg_tariff[s]])
           total_bill += sum(b[s] * pricing_signal[seg_tariff[s]])
           total_bill += sum(c[s] * pricing_signal[seg_tariff[s]])
   
       # aggregate the fixed plus optimal load
       optimal_load_curve = a + b + c
       #print(optimal_load_curve.iloc[0,:].sum()/1e3)
       
       optimal_load_curve['agg'] = optimal_load_curve.sum(axis=1)
       optimal_load_curve /= 1e3 # convert KW to MW for generation cost computations
       return optimal_load_curve, total_bill

   def similar_costs(curr,prev):
       cost_diff = (prev-curr)/prev
       if cost_diff < 0:
           return True
       if cost_diff < epsilon:
           return True
       else: 
           return False

   n = 0

   # set initial pricing signal
   pricing_signal = {}
   iterative_load = {}
   iterative_gen_cost = {}
   iterative_bill = {}
   iterative_gen_cost_total = {}

   # Aggregate base loads and add losses to compute projected_demand
   total_base_load, total_bill = aggregate_base_load()
   projected_demand = total_base_load['total']['agg'] / (1 - losses)

   # Compute generation_costs for aggregate base loads
   slot_wise_cost, slot_wise_cost1, slot_wise_pu = compute_gen_curve(projected_demand)
   iterative_gen_cost[n] = slot_wise_pu
   iterative_load[n] = projected_demand
   iterative_bill[n] = total_bill
   iterative_gen_cost_total[n] =slot_wise_cost1
   # print(slot_wise_cost[2])

   pricing_signal[n] = tariff_plan

   prev_gen_cost1 = slot_wise_cost1

   while n < max_iters:        
       #1 - send pricing signal to consumers
       n += 1
       pricing_signal[n] = pricing_signal[n-1].copy()

       #2 - consumers send optimal load curves
       optimal_load_curve, optimal_bill = compute_optimal_load_curve1(pricing_signal[n])
       #print(optimal_load_curve)
    
       #3 - aggregate load curve, add losses and compute gen curve
       projected_optimal_demand = optimal_load_curve['agg'] / (1 - losses)
       #print(projected_optimal_demand.max())
    
       #4 - compute lowest generation cost
       _,curr_gen_cost,curr_slot_pu = compute_gen_curve(projected_optimal_demand, n)
       mean_pu_cost = curr_slot_pu.mean()
       #curr_gen_cost = curr_slot_pu
       #Store intermediate variables through iterations
       iterative_load[n] = projected_optimal_demand
       iterative_bill[n] = optimal_bill
       iterative_gen_cost[n] = curr_slot_pu
       iterative_gen_cost_total[n]=curr_gen_cost
    
       #print(curr_gen_cost.sum()/1e7,prev_gen_cost1.sum()/1e7)
       #5 - Check if algorithm converges, else update the pricing signal (cost diff less than 0.5% of prev cost)
       if similar_costs(curr_gen_cost.sum(), prev_gen_cost1.sum()): break  
       #print(curr_gen_cost - prev_gen_cost1)
       #print((curr_gen_cost.sum()-prev_gen_cost.sum())/1e7)
       for t in lambda_t.keys():
           for j in range(24):
               if curr_slot_pu[j]<mean_pu_cost:
                   pricing_signal[n][t][j] += 1.2*lambda_t[t] * (curr_slot_pu[j]-mean_pu_cost)
               else:
                   pricing_signal[n][t][j] += 0.8*lambda_t[t] * (curr_slot_pu[j]-mean_pu_cost)
                
            
       prev_gen_cost = curr_slot_pu
       prev_gen_cost1  = curr_gen_cost

   nn=n

   st.header("Iterative analysis")
   n = st.slider("Iteration:", 0, nn, 0)

   def update_plot(n):
      # Your existing code for generating the plot with the given value of n
      fig = sp.make_subplots(rows=3, cols=2, subplot_titles=('Utility Load Curve (GW)', 'Generation PU cost (INR/kWH)', 'Up and down flexibility (MW)','Utility benefit', 'Commercial - Pricing signals', 'Residential - Pricing signals'),horizontal_spacing=0.2, vertical_spacing=0.2)
   
      x_values = np.linspace(1, 24, 24)
      x1 = x_values
   
      flexibility = iterative_load[n] - iterative_load[0]
   
      gen_df = pd.DataFrame(pd.DataFrame(iterative_gen_cost_total).sum(axis=0))
      bill_df = pd.DataFrame(iterative_bill, index=[0])
   
      profit_df = bill_df.T - gen_df
       
      # Plot the data
      fig.add_trace(go.Scatter(x=x1, y=iterative_load[0] / 1000, mode='lines', name=f'Iteration: {0} load'), row=1, col=1)
      fig.add_trace(go.Scatter(x=x1, y=iterative_gen_cost[0], mode='lines', name=f'Iteration: {0} gen cost'), row=1, col=2)
      fig.add_trace(go.Scatter(x=x1, y=iterative_load[n] / 1000, mode='lines', name=f'Iteration: {n} load'), row=1, col=1)
      fig.add_trace(go.Scatter(x=x1, y=iterative_gen_cost[n], mode='lines', name=f'Iteration: {n} gen cost'), row=1, col=2)
   
      fig.add_trace(go.Scatter(x=x1, y=flexibility / 1000, mode='lines', name='Flexibility'), row=2, col=1)
      fig.add_trace(go.Scatter(x=x1[0:n+1], y=profit_df.iloc[0:n+1, 0] / 1e7, name='Profit'), row=2, col=2)
       
      fig.add_trace(go.Scatter(x=x1, y=pricing_signal[0]['residential'], name='Residential TOD Tariff'), row=3, col=2)
      fig.add_trace(go.Scatter(x=x1, y=pricing_signal[n]['residential'], name='Pricing signal for Residential'), row=3, col=2)
       
      fig.add_trace(go.Scatter(x=x1, y=pricing_signal[0]['commercial'], name='Commercial TOD Tariff'), row=3, col=1)
      fig.add_trace(go.Scatter(x=x1, y=pricing_signal[n]['commercial'], name='Pricing signal for Commercial'), row=3, col=1)
   
      fig.update_layout(
          height=600,
          width=1100,
          margin=dict(l=3, r=2, b=0, t=100),
          boxgroupgap=0.00,
          title_text=f'Demand flexibility analysis - Iteration: {n}',
       )
   
      # Add axis titles for each subplot
      fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
      fig.update_yaxes(title_text="Load (GW)", row=1, col=1)
      fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
      fig.update_yaxes(title_text="Cost (INR/kWH)", row=1, col=2)
      fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
      fig.update_yaxes(title_text="Flexibility (GW)", row=2, col=1)
      fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
      fig.update_yaxes(title_text="Crore INR", row=2, col=2)
      fig.update_xaxes(title_text="Time (hours)", row=3, col=2)
      fig.update_yaxes(title_text="INR/kWh", row=3, col=2)
      fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
      fig.update_yaxes(title_text="INR/kWh", row=3, col=1)
       
      st.plotly_chart(fig, use_container_width=True)
   update_plot(n)

   ## Load shift in different categories of load

   consumer_list = ['pww','coldstorage','IT','office','mall','hotel','hospital','residential']
 
   st.header ("Consumer load profiles")
   consumer_category = st.selectbox("Choose a consumer category", options=consumer_list)

   def update_plot1(consumer_category):
       # Your existing code for generating the plot with the given value of n
       fig = sp.make_subplots(rows=1, cols=1)
       x_values = np.linspace(1, 24, 24)
       x1 = x_values
       if consumer_category=='residential':
           fps = pricing_signal[nn]['residential']
       else:
           fps = pricing_signal[nn]['commercial']
        
        
       load_cs = base_base[consumer_category]
       optimal_load=optimal_load_curve1(fps,load_cs)
       base_load =load_cs.iloc[:, 25]
    
    
       # Plot the data
       fig.add_trace(go.Scatter(x=x1, y=base_load, mode='lines', name=f'Base load'), row=1, col=1)
       fig.add_trace(go.Scatter(x=x1, y=optimal_load, mode='lines', name=f'Optimal load - iteration: {0}'), row=1, col=1)
   
       fig.update_layout(
           height=400,
           #width=1100,
           margin=dict(l=3, r=2, b=0, t=100),
           boxgroupgap=0.00,
           title_text=f'Consumer load curves: {n}',
        )

       # Add axis titles for each subplot
       fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
       fig.update_yaxes(title_text="Load (kW)", row=1, col=1)

       st.plotly_chart(fig, use_container_width=True)
   update_plot1(consumer_category)

   ## Load shift in different categories of load
   
   file_path = OUTPUT_PATH + 'initial_output.xlsx'
   #with open(file_path, 'rb') as f:
   #    st.download_button(label = 'Download Initial Output', data = f, file_name = 'initial_output.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')    
