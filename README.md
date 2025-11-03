# eco-focused-health-assistant
This is an Eco-Focused Health Assistant AI Agent, that works by using a lightweight TensorFlow personalizer model and a small sentiment
analysis model to provide personalized eco-friendly suggestions based on user input. Users input their daily activities such as commute type,
distance, diet, meals, and electricity usage. The program estimates their daily CO₂ emissions and recommends actions to reduce their footprint.

Included Files:
    - `eco_agent_tf.py` is the main program used for CO₂ estimation, sentiment analysis, and personalized suggestion generation.
    - `user_history.json` is automatically created to store user inputs and feedback.
    - `eco_model_tf/` is automatically created to store the Keras personalizer model.

User Requirements:
    - Python version 3.8 or newer
    - Access to TensorFlow and NumPy

Features:
    - Estimates daily CO₂ emissions for commute, diet, and electricity usage.
    - Generates eco-friendly suggestions ranked by estimated CO₂ savings and personalized acceptance probability.
    - Reads user mood with a small sentiment analysis model.
    - Interactive terminal interface allowing repeated inputs.
    - Persists user history and personalizer model for ongoing personalization.
    - Lightweight and self-contained; no large datasets or NLP libraries required.

Structure of the Program:

    Section 0: Import all libraries needed for numerical computation, TensorFlow model building, and file management.

    Section 1: Define emission factors for commute, diet, and electricity usage.

    Section 2: Define eco-friendly suggestions and helper functions to estimate emissions and CO₂ savings.

    Section 3: Build and train a small personalizer model with synthetic data for warm-starting.

    Section 4: Build and train a tiny sentiment analysis model on embedded example sentences.

    Section 5: Main interactive loop for user input, CO₂ estimation, suggestion ranking, and personalized feedback.

    Section 6: Persistence functions for saving/loading user history and the personalizer model.

    Section 7: Online update function to fine-tune the personalizer model based on user feedback.

Efficiency:

    CO₂ Estimation Functions:
        Time Complexity: O(1) per input
        Space Complexity: O(1) per input

    Personalizer Model Training (Synthetic Warm-Start):
        Time Complexity: O(m * n), where m is the number of synthetic samples and n is the feature dimension
        Space Complexity: O(m * n)
    Sentiment Model Training:
        Time Complexity: O(m * n), where m is the number of example sentences and n is the vocabulary size
        Space Complexity: O(m * n)

    Prediction / Online Update:
        Time Complexity: O(n)
        Space Complexity: O(n)

How to Run:
    - use the command: python agent.py
    - enter your mood, commute, diet, meals, and estimated electricity usage from the day when prompted
    - receive personalized eco-friendly suggestions with estimated CO₂ savings
    - Type "quit" at any point during the program runtime to exit. All progress will be saved locally in the files listed aboveThis is an Eco-Focused Health Assistant AI Agent, that works by using a lightweight TensorFlow personalizer model and a small sentiment
analysis model to provide personalized eco-friendly suggestions based on user input. Users input their daily activities such as commute type,
distance, diet, meals, and electricity usage. The program estimates their daily CO₂ emissions and recommends actions to reduce their footprint.

Included Files:
    - `eco_agent_tf.py` is the main program used for CO₂ estimation, sentiment analysis, and personalized suggestion generation.
    - `user_history.json` is automatically created to store user inputs and feedback.
    - `eco_model_tf/` is automatically created to store the Keras personalizer model.

User Requirements:
    - Python version 3.8 or newer
    - Access to TensorFlow and NumPy

Features:
    - Estimates daily CO₂ emissions for commute, diet, and electricity usage.
    - Generates eco-friendly suggestions ranked by estimated CO₂ savings and personalized acceptance probability.
    - Reads user mood with a small sentiment analysis model.
    - Interactive terminal interface allowing repeated inputs.
    - Persists user history and personalizer model for ongoing personalization.
    - Lightweight and self-contained; no large datasets or NLP libraries required.

Structure of the Program:

    Section 0: Import all libraries needed for numerical computation, TensorFlow model building, and file management.

    Section 1: Define emission factors for commute, diet, and electricity usage.

    Section 2: Define eco-friendly suggestions and helper functions to estimate emissions and CO₂ savings.

    Section 3: Build and train a small personalizer model with synthetic data for warm-starting.

    Section 4: Build and train a tiny sentiment analysis model on embedded example sentences.

    Section 5: Main interactive loop for user input, CO₂ estimation, suggestion ranking, and personalized feedback.

    Section 6: Persistence functions for saving/loading user history and the personalizer model.

    Section 7: Online update function to fine-tune the personalizer model based on user feedback.

Efficiency:

    CO₂ Estimation Functions:
        Time Complexity: O(1) per input
        Space Complexity: O(1) per input

    Personalizer Model Training (Synthetic Warm-Start):
        Time Complexity: O(m * n), where m is the number of synthetic samples and n is the feature dimension
        Space Complexity: O(m * n)
    Sentiment Model Training:
        Time Complexity: O(m * n), where m is the number of example sentences and n is the vocabulary size
        Space Complexity: O(m * n)

    Prediction / Online Update:
        Time Complexity: O(n)
        Space Complexity: O(n)

How to Run:
    - use the command: python agent.py
    - enter your mood, commute, diet, meals, and estimated electricity usage from the day when prompted
    - receive personalized eco-friendly suggestions with estimated CO₂ savings
    - Type "quit" at any point during the program runtime to exit. All progress will be saved locally in the files listed above