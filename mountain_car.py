# --STEP 0--
# Import and initialize Mountain Car Environment



# --STEP 1--

# Set the variables


# Determine size of discretized state space




# --STEP 2--

# Initialize variables to track rewards


# Initialize Q table



# --STEP 3--

# Initialize parameters (done,state,rewards)


# Discretize state



# --STEP 4--

#Create a loop that is terminated when the game is won
#MAKE SURE THAT YOUR CODE IS ALLIGNED CORRECTLY

#while

    # Render environment


    # Determine next action - epsilon greedy strategy


    # Get next state and reward



    # Discretize new state



    #Allow for terminal states



    # Adjust Q value for current state / Apply the Q-Learning function





    # Update variables


#Close the environment after the loop


# --STEP 5--

# Calculate episodic reduction in epsilon



#Create the loop for the episodes assigned in the beginning of the code and put inside the code from STEP 3
#AND STEP 4 but now close the environment at the end of for loop and
#Rememer to render the environment every 200 episode this time


#MAKE SURE THAT YOUR CODE IS ALLIGNED CORRECTLY


#for

    #PUT YOUR CODE HERE , RENDER EVERY 200 EPISODES

    # Inside the for loop you need to decay epsilon


    # Track rewards




 #After creating the loop remember to CLOSE the environment of mountain car



# --STEP 6--

# Plot Rewards
