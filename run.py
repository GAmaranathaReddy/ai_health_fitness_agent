from ai_health_fitness_agent.helath_fitness_agent import main as app_main

# Run Streamlit app directly when executing: streamlit run run.py
# Avoid spawning a second Streamlit process (previous os.system approach caused blank page / port issues).
if __name__ == "__main__":
    app_main()
