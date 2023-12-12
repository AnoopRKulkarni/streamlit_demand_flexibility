#!/usr/bin/env python
# coding: utf-8

#
# MPEN Demand Flexibility Tool Analysis Page
# Authors: Anoop R Kulkarni, Kaustubh Arekar
# Version 1.0
# Dec 11, 2023

#@title Import libraries
import matplotlib.pyplot as plt
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def app(site):
   st.header("Overall objectives")
   st.markdown(
   """
   The demand flexibility model is designed to work for following objectives:
      - Renewable energy utilization maximization
      - Consumer bill minimization
      - Utility power purchase cost minimization
   """
   )

   st.header("Modeling framework")
   st.markdown(
   """
   The demand flexibility model is built around following framework
      - Input
        - Generation (capacity, variable cost, fixed cost, ramping constraints, type, slot wise RE)
        - Substation level load (aggregation of flexible, RE, adjustable, battery loads etc)
        - Feeder level load (aggregation of flexible, RE, adjustable, battery loads etc)
        - Tariff category wise
        - End use scenarios
      - Model
        - Objective function
        - Problem formulation
      - Output
        - Dynamic pricing signal
        - Tariff structure
        - Base load curves
        - Benefit to utility
        - Benefit to consumers
   """
   )

   st.header("Data")
   st.markdown(
   """
   The following data is used as input for the model
      - Generation
        - MODCON.xlsx, RE power PPA.xlsx, RE.xlsx
      - Demand
        - Consumer indexing, end user assumptions, solar rooftop, tariff plan
      - Assumptions
        - Fraction of solar adoption, DF participation, number of consumers, solar capacity, losses etc
   """
   )

   st.subheader("Generation", divider='rainbow')
   st.text("Power plants")
   st.image('res/generation_1.png')
   st.text("Renewable energy")
   st.image('res/generation_2.png')
   st.text("RE power PPA")
   st.image('res/generation_3.png')

   st.subheader("Demand", divider='rainbow')
   st.text("Substation")
   st.image('res/demand_1.png')
   st.text("Feeder")
   st.image('res/demand_2.png')
   st.text("Demand side solar generation per KW")
   st.image('res/demand_3.png')

   st.subheader("Assumptions", divider='rainbow')
   st.text("Residential usage - base scenario")
   st.text("Note: Similar base scenarios are assumed for hospital, hotel, mall, IT park, commercial office, PWW, cold storage")
   st.image('res/residential_base.png')
   st.text ("Tariff plan")
   st.image('res/tariff_plan.png')
