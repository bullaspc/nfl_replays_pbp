---
name: run-app
description: Launch the Streamlit app and check for startup errors. Use after any edit to nfl_replay_app.py to verify the app starts cleanly.
---

Run the following steps:

1. Kill any existing Streamlit process on port 8501:
   ```bash
   pkill -f "streamlit run" 2>/dev/null; sleep 1
   ```

2. Start the app in the background and capture output for 8 seconds:
   ```bash
   streamlit run nfl_replay_app.py --server.headless true 2>&1 &
   sleep 8
   ```

3. Check the output for any Python tracebacks, `ModuleNotFoundError`, `ImportError`, or `SyntaxError`.

4. Verify the process is still running:
   ```bash
   pgrep -f "streamlit run" && echo "App is running on http://localhost:8501" || echo "App failed to start"
   ```

5. Report the result:
   - If the app started cleanly: confirm it's running on port 8501
   - If there were errors: quote the relevant lines and describe what needs to be fixed
