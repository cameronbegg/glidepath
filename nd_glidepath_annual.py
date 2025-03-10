import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# non deterministic
def nd_model(start_age, end_age, initial_investment, contributions, contributions_growth_rate, growth_rate, growth_vol, de_risk_start, de_risk_length, retirement_rate, retirement_vol, iterations, decum_rate, withdraw_25_percent):
    import pandas as pd 
    import sys
    import numpy as np


    # Will be list of Dataframes
    results = []


    for _ in range(iterations):
        # Create new Dataframe for each iteration
        df = pd.DataFrame(columns=['Age', 'Investment Value'])

        # Take the inital value as pot starting point
        investment_value = initial_investment
        current_conts = contributions

        # loop through the age range -> this is for the glidepath range
        for age in range(start_age, end_age + 1):

            
            # removes 25% of pot for members at retirement if selected
            if age == ((de_risk_start + de_risk_length)+1) and withdraw_25_percent == True:
                investment_value *= 0.75
                
            # Growth Stage
            if age < de_risk_start:
                # Applies contribution
                investment_value += current_conts
                # finds volatility
                random_growth_rate_1 = np.random.normal(growth_rate, growth_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_1)
    
            # Glide Path
            elif age <= de_risk_start + de_risk_length:
                # Applies contribution
                investment_value += current_conts

                # Blended growth rate of two portfolios at current age/year
                blended_growth_rate = growth_rate - (growth_rate - retirement_rate) * (age - de_risk_start) / de_risk_length
                # blend vol rate of two portfolios at current age / year
                blended_vol = growth_vol - (growth_vol - retirement_vol) * (age - de_risk_start) / de_risk_length
            
                # finds volatility
                random_growth_rate_2 = np.random.normal(blended_growth_rate, blended_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_2)

            # Post Glide
            else:
                investment_value *= (1-decum_rate)
                random_growth_rate_3 = np.random.normal(retirement_rate, retirement_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_3)

    
            df = pd.concat([df, pd.DataFrame({'Age': [age], 'Investment Value': [investment_value]})], ignore_index=True)
            current_conts *= (1+contributions_growth_rate)
        
        results.append(df)

    return results

###############################################
###### Visualisation ##########################
###############################################

st.title('Investment Growth Over Time')

# Define the number of investment pots
num_pots = st.number_input('Number of Investment Pots', min_value=1, max_value=10, value=1)

# Initialize a list to store the dataframes for each pot
dfs = []

# Loop through each investment pot and get the parameters
for i in range(num_pots):
    st.header(f'Investment Pot {i+1}')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        start_age = st.number_input(f'Start Age {i+1}', min_value=0, max_value=100, value=18)
        growth_rate = st.number_input(f'Growth Rate {i+1}', min_value=0.000, max_value=1.000, value=0.070, format="%.3f")
        initial_investment = st.number_input(f'Initial Investment {i+1}', min_value=0.0, value=1000.000, format="%.3f")
      
    with col2:
        end_age = st.number_input(f'End Age {i+1}', min_value=0, max_value=100, value=95)
        growth_volatility = st.number_input(f'Growth Volatility {i+1}', min_value=0.000, max_value=1.000, value=0.030, format="%.3f")
        annual_contribution = st.number_input(f'Annual Cont. {i+1}', min_value=0.0, value=4000.000, format="%.3f")
    with col3:
        de_risk_start = st.number_input(f'De-risk Start Age {i+1}', min_value=0, max_value=100, value=55)
        retirement_rate = st.number_input(f'Ret. Growth Rate {i+1}', min_value=0.000, max_value=1.000, value=0.030, format="%.3f")
        contribution_growth_rate = st.number_input(f'Cont. Growth Rate {i+1}', min_value=0.000, max_value=1.000, value=0.020, format="%.3f")        
    with col4:
        de_risk_length = st.number_input(f'De-risk Length {i+1}', min_value=0, max_value=100, value=10)
        retirement_volatility = st.number_input(f'Ret. Volatility {i+1}', min_value=0.000, max_value=1.000, value=0.010, format="%.3f")
    with col5:
        lower_bound = st.number_input(f'Lower Bound (above 10 iterations) {i+1} (%)', min_value=0, max_value=100, value=25)
        upper_bound = st.number_input(f'Upper Bound (above 10 iterations){i+1} (%)', min_value=0, max_value=100, value=75)
        decum_rate = st.number_input(f'Decum. Rate {i+1}', min_value=0.000, max_value=1.000, value=0.040, format="%.3f")
            
    iterations = st.number_input(f'Iterations {i+1}', min_value=1, max_value=5000, value=10)

    # Toggle to withdraw 25% at retirement date
    withdraw_25_percent = st.checkbox(f'Withdraw 25% at Retirement Date for Pot {i+1}')

    # Calculate the investment growth for the current pot
    results = nd_model(start_age, end_age, initial_investment, annual_contribution, contribution_growth_rate, growth_rate, growth_volatility, de_risk_start, de_risk_length, retirement_rate, retirement_volatility, iterations, decum_rate, withdraw_25_percent)
              
    # Append the dataframe to the list
    dfs.append(results)


col1, col2 = st.columns(2)
with col1:
    # Toggle to switch between mean and median
    highlight_option = st.radio("Highlight Mid-Point:", ('Median', 'Mean'))

with col2:
    # Toggle to only show mean/median
    brightness = st.slider("Non Median / Mean Opacity:", 0.0, 1.0, 0.1)

# Plot the investment growth over time for all pots

# List of colors to use for each investment pot
colours = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan']

fig, ax = plt.subplots(figsize=(10, 6))
final_values_texts = []
mid_point_values = []

for i in range(num_pots):
    colour = colours[i % len(colours)]  # loop through the colours list
    
    if iterations > 10:
        final_values = [result['Investment Value'].iloc[-1] for result in dfs[i]]
        sorted_indices = np.argsort(final_values)
        lower_bound_index = int(lower_bound / 100 * iterations)
        upper_bound_index = int(upper_bound / 100 * iterations)
        
        filtered_results = [dfs[i][j] for j in sorted_indices[lower_bound_index:upper_bound_index]]
    else:
        filtered_results = dfs[i]
    
    # Calculate retirement age
    retirement_age = de_risk_start + de_risk_length

    for result in filtered_results:
        result['Age'] = pd.to_numeric(result['Age'])
        ax.plot(result['Age'], result['Investment Value'], color=colour, alpha=brightness)
    
    if highlight_option == 'Median':
        # Highlight the median performance
        median_index = len(filtered_results) // 2
        median_result = filtered_results[median_index]
        ax.plot(median_result['Age'], median_result['Investment Value'], color=colour, marker='o', label=f'Investment Pot {i+1} Median')
        
        # Append the final investment value of the median to the list
        final_value = median_result['Investment Value'].iloc[-1]
        final_values_texts.append(f"Investment Pot {i+1} median: £{final_value:.2f}")
        mid_point_values.append(final_value)
    else:
        # Highlight the mean performance
        mean_result = pd.concat(filtered_results).groupby('Age').mean().reset_index()
        ax.plot(mean_result['Age'], mean_result['Investment Value'], color=colour, linestyle='--', label=f'Investment Pot {i+1} Mean')
        
        # Append the final investment value of the mean to the list
        final_value = mean_result['Investment Value'].iloc[-1]
        final_values_texts.append(f"Investment Pot {i+1} mean: £{final_value:.2f}")
        mid_point_values.append(final_value)

      # Add a vertical line at the retirement age
    ax.axvline(x=retirement_age, color='black', linestyle='--', label=f'Retirement Age {i+1}')

# Calculate the difference between the two pot sizes
if num_pots == 2:
    difference = mid_point_values[1] - mid_point_values[0]
    final_values_texts.append(f"Difference between Pot 2 and Pot 1: £{difference:.2f}")

ax.set_title('Investment Growth Over Time')
ax.set_xlabel('Age')
ax.set_ylabel('Investment Value')
ax.legend()
ax.grid(True)

# Display the final investment values below the chart
for i, text in enumerate(final_values_texts):
    plt.figtext(0.5, -0.05 - i * 0.05, text, ha="center", fontsize=12, color=colours[i % len(colours)])

st.pyplot(fig)

