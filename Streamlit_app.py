# Thin wrapper for Streamlit Cloud entrypoint.
# Delegates to module: ai_health_fitness_agent/helath_fitness_agent.py

import importlib

# Dynamically import the main app module (typo in filename preserved)
app_mod = importlib.import_module(
    "ai_health_fitness_agent.helath_fitness_agent"
)

if hasattr(app_mod, "main"):
    app_mod.main()
else:
    raise SystemExit("Expected a function `main()` in helath_fitness_agent.py")
