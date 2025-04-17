import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# to do:
# Add in standard deviate figure of returns
# Add in post-retirement glidepath
# Add in annuity taking features, around 7% conversion of pot to annual income when taken at age 65
# Tidy up app design


# non deterministic
def nd_model(start_age, # simulation starting age
            end_age, # simulaiton ending age
            ret_age, # retirement age
            pre_ret_glide, # length of glide to retirement
            post_ret_glide, # length of post retirement glide
            growth_rate, #
            growth_vol,
            glide_rate,
            glide_vol,
            ret_rate,
            ret_vol,
            initial_investment,
            contributions, # inital annual contributions
            contributions_growth_rate, # % that conts. will increase each year
            decum_rate, # post retirement decumulation rate
            withdraw_25_percent, # boolean, take 25% lump sum
            withdrawl_amount, # post retirement annual withdrawal amount
            iterations, # number of simulation runs
            ):
    
    #test = []

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
            if age == (ret_age+1) and withdraw_25_percent == True:
                investment_value *= 0.75
                
            # Growth Stage
            # if the age is less the glidepath start age
            if age < (ret_age - pre_ret_glide):
                # Applies contribution
                investment_value += current_conts
                # finds volatility
                random_growth_rate_1 = np.random.normal(growth_rate, growth_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_1)

                
            # Glide Path
            # if the age is within the glidepath 
            elif age <= ret_age:
                # Applies contribution
                investment_value += current_conts

                # Blended growth rate of two portfolios at current age/year
                blended_growth_rate = growth_rate - (growth_rate - glide_rate) * (age - (ret_age - pre_ret_glide)) / pre_ret_glide
            
                # blend vol rate of two portfolios at current age / year
                blended_vol = growth_vol - (growth_vol - glide_vol) * (age - (ret_age - pre_ret_glide)) / pre_ret_glide
               
            
                # finds volatility
                random_growth_rate_2 = np.random.normal(blended_growth_rate, blended_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_2)
                

            # Post ret glide
            # if the age is post retirement age and less than post ret glidepath age
            elif age <= (ret_age + post_ret_glide):
                # blend glidepath asset allocation with in retirement glidepath

                # Blended growth rate of two portfolios at current age/year
                blended_growth_rate_2 = glide_rate - (glide_rate - ret_rate) * (age - ((ret_age + post_ret_glide) - post_ret_glide)) / post_ret_glide
            
                # blend vol rate of two portfolios at current age / year
                blended_vol_2 = glide_vol - (glide_vol - ret_vol) * (age - ((ret_age + post_ret_glide) - post_ret_glide)) / post_ret_glide
               

                # finds volatility
                random_growth_rate_3 = np.random.normal(blended_growth_rate_2, blended_vol_2)

                investment_value *= (1-decum_rate)
                investment_value -= withdrawl_amount
                
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_3)

                
            # Post all glides
            else:

                investment_value *= (1-decum_rate)
                investment_value -= withdrawl_amount
                random_growth_rate_4 = np.random.normal(ret_rate, ret_vol)
                # combines growth rate & vol
                investment_value *= (1+random_growth_rate_4)

            df = pd.concat([df, pd.DataFrame({'Age': [age], 'Investment Value': [investment_value]})], ignore_index=True)
            current_conts *= (1+contributions_growth_rate)
            

        results.append(df)

    return results

###############################################
###### Visualisation ##########################
###############################################
# Set the page layout to wide mode

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded")

st.title('Glidepath Modelling')

# Define the number of investment pots
num_pots = st.number_input('Number of Investment Pots', min_value=1, max_value=10, value=1)

st.markdown("---")

# Initialize a list to store the dataframes for each pot
dfs = []

# Loop through each investment pot and get the parameters
for i in range(num_pots):
    st.header(f'Investment Pot {i+1}')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.header("Ages & Lengths")
        start_age = st.number_input(f'Start Age {i+1}', min_value=0, max_value=100, value=18)
        end_age = st.number_input(f'End Age {i+1}', min_value=0, max_value=100, value=95)
        ret_age = st.number_input(f'Retirement Age {i+1}', min_value=0, max_value=100, value=65)
        pre_ret_glide = st.number_input(f'Pre Ret. Glide Length {i+1}', min_value=0, max_value=100, value=10)
        post_ret_glide = st.number_input(f'Post Ret. Glide Length {i+1}', min_value=0, max_value=100, value=10)
    with col2:
        st.header("Portfolios")
        growth_rate = st.number_input(f'Growth Rate {i+1}', min_value=0.000, max_value=1.000, value=0.080, format="%.3f")
        growth_vol = st.number_input(f'Growth Volatility {i+1}', min_value=0.000, max_value=1.000, value=0.015, format="%.3f")
        glide_rate = st.number_input(f'Glide Rate {i+1}', min_value=0.000, max_value=1.000, value=0.060, format="%.3f")
        glide_vol = st.number_input(f'Glide Volatility {i+1}', min_value=0.000, max_value=1.000, value=0.01, format="%.3f")
        ret_rate = st.number_input(f'Ret Rate {i+1}', min_value=0.000, max_value=1.000, value=0.040, format="%.3f")
        ret_vol = st.number_input(f'Ret Volatility {i+1}', min_value=0.000, max_value=1.000, value=0.060, format="%.3f")
    with col3:
        st.header("Income/withdrawals")
        initial_investment = st.number_input(f'Initial Investment {i+1}', min_value=0.0, value=1000.000, format="%.3f")
        contributions = st.number_input(f'Annual Cont. {i+1}', min_value=0.0, value=3000.000, format="%.3f")
        contributions_growth_rate = st.number_input(f'Cont. Growth Rate {i+1}', min_value=0.000, max_value=1.000, value=0.020, format="%.3f")  
        decum_rate = st.number_input(f'Decum. Rate {i+1}', min_value=0.000, max_value=1.000, value=0.040, format="%.3f")
        withdrawl_amount = st.number_input(f'Annual withdrawal amount {i+1}', min_value=0, max_value=1000000, value= 0)
        withdraw_25_percent = st.checkbox(f'Withdraw 25% at Retirement Date for Pot {i+1}')
    with col4:     
        st.header("Data")
        lower_bound = st.number_input(f'Lower Bound (above 10 iterations) {i+1} (%)', min_value=0, max_value=100, value=20)
        upper_bound = st.number_input(f'Upper Bound (above 10 iterations){i+1} (%)', min_value=0, max_value=100, value=80)
        iterations = st.number_input(f'Iterations {i+1}', min_value=1, max_value=5000, value=50)
        
    
    st.markdown("---")


    # Calculate the investment growth for the current pot
    results = nd_model(start_age, # simulation starting age
                    end_age, # simulaiton ending age
                    ret_age, # retirement age
                    pre_ret_glide, # length of glide to retirement
                    post_ret_glide, # length of post retirement glide
                    growth_rate, #
                    growth_vol,
                    glide_rate,
                    glide_vol,
                    ret_rate,
                    ret_vol,
                    initial_investment,
                    contributions, # inital annual contributions
                    contributions_growth_rate, # % that conts. will increase each year
                    decum_rate, # post retirement decumulation rate
                    withdraw_25_percent, # boolean, take 25% lump sum
                    withdrawl_amount, # post retirement annual withdrawal amount
                    iterations, # number of simulation runs
                    )

    # Append the dataframe to the list
    dfs.append(results)

st.header("Chart Features")
col1, col2 = st.columns(2)
with col1:
    # Toggle to switch between mean and median
    highlight_option = st.radio("Highlight Mid-Point:", ('Mean', 'Median'))

with col2:
    # Toggle to only show mean/median
    brightness = st.slider("Non Median / Mean Opacity:", 0.0, 1.0, 0.03)


    
st.markdown("---")

# Plot the investment growth over time for all pots

# List of colors to use for each investment pot
colours = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan']

fig, ax = plt.subplots(figsize=(10, 6))
final_values_texts = []
mid_point_values = []


#### TO DO ! ###
# 1. allow for std dev to be calculated when multiple portfolios are used
# 1.1 current code only works when single portfolio is created
# 2. toggle for quartile lines to be added in - use different colour / make smaller

for i in range(num_pots):
    colour = colours[i % len(colours)]  # loop through the colours list
    
    if iterations > 10:
        # Ensure that dfs[i] is not empty and has the 'Investment Value' column
        if dfs[i] and all('Investment Value' in result.columns for result in dfs[i]):
            final_values = [result['Investment Value'].iloc[-1] for result in dfs[i]]
            sorted_indices = np.argsort(final_values)
            lower_bound_index = int(lower_bound / 100 * iterations)
            upper_bound_index = int(upper_bound / 100 * iterations)
            
            filtered_results = [dfs[i][j] for j in sorted_indices[lower_bound_index:upper_bound_index]]
            
            # Calculate and display standard deviation of results - mean should also match
            final_pot_size = []
            for result in filtered_results:
                temp = result['Investment Value'].iloc[-1]
                final_pot_size.append(temp)
            
            final_pot_size.sort()
            lower = int(len(final_pot_size) * (lower_bound / 100))
            upper = int(len(final_pot_size) * (upper_bound / 100))
            final_pot_size_trimmed = final_pot_size[lower:upper]
            mean = np.mean(final_pot_size_trimmed)
            std_dev = np.std(final_pot_size_trimmed)
            
            final_values_texts.append(f"Investment Pot {i+1} final mean: £{mean:.2f}")
            final_values_texts.append(f"Investment Pot {i+1} final std dev: £{std_dev:.2f}")
        else:
            st.error(f"Investment Pot {i+1} has no valid data.")
        
    else:
        filtered_results = dfs[i]
    
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
        #final_values_texts.append(f"Investment Pot {i+1} median: £{final_value:.2f}")
        mid_point_values.append(final_value)
    else:
        # Highlight the mean performance
        mean_result = pd.concat(filtered_results).groupby('Age').mean().reset_index()
        ax.plot(mean_result['Age'], mean_result['Investment Value'], color=colour, linestyle='--', label=f'Investment Pot {i+1} Mean')
        
        # Append the final investment value of the mean to the list
        final_value = mean_result['Investment Value'].iloc[-1]
        #final_values_texts.append(f"Investment Pot {i+1} mean: £{final_value:.2f}")
        mid_point_values.append(final_value)

      # Add a vertical line at the retirement age
    ax.axvline(x=ret_age, color='black', linestyle='--', label=f'Retirement Age {i+1}')
    

# Calculate the difference between the two pot sizes
if num_pots == 2:
    difference = mid_point_values[1] - mid_point_values[0]
    final_values_texts.append(f"Difference between Pot 2 and Pot 1: £{difference:.2f}")

col1, col2 = st.columns(2)
with col1:
    st.header("Chart:")
    # Calculate the difference between the two pot sizes
    ax.set_title('Investment Growth Over Time')
    ax.set_xlabel('Age')
    ax.set_ylabel('Investment Value')
    ax.legend()
    ax.grid(True)

    # Display the final investment values below the chart
    #for i, text in enumerate(final_values_texts):
        #plt.figtext(0.5, -0.05 - i * 0.05, text, ha="center", fontsize=12, color=colours[i % len(colours)])
    st.pyplot(fig)

with col2:
    st.header("Stats:")

    for i, text in enumerate(final_values_texts):
         
        st.header(text) 



