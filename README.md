# Autonomous Clustering Agent

This project provides an automated clustering pipeline that preprocesses a dataset, selects the optimal number of clusters, runs multiple clustering algorithms, and visualizes the results. The final output includes labeled data and plots.

## Run it
### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```
### Run from the command line
```bash
python parte1.py <input_file.csv> --output clustered_output.csv --show-plot --save-plot
```
Where,
`input_file.csv`: Path to your input CSV file.

`--output`: (Optional) Output file name (default: output.csv)
`--seed`: (Optional) Random seed for reproducibility.
`--show-plot`: Show the cluster visualization.
`--save-plot`: Save the cluster plot as figure_output.png

### Output
`output.csv` (default): Input data with a new cluster column.
`figure_output.png`: Visualization of clusters (if --save-plot is used).

Standard output shows the chosen method, optimal number of clusters, and Silhouette Score.


## Design Decisions
 - Missing Value Handling: Missing values are imputed using the mean of each numeric column. This general-purpose strategy avoids deleting data, especially when missingness exceeds 5%.
 - Standardization: All numeric features are standardized using StandardScaler to ensure that features contribute equally to distance-based clustering.
 - The number of clusters is determined via Silhouette Score, applied to both KMeans and Agglomerative Clustering.
 - DBSCAN is also evaluated using default parameters (eps=0.5, min_samples=5).
 - Model Selection: Best Silhouette Score is used to label the data.

## Assumptions
 - Input data is CSV format and primarily numeric.
 - Imputation with mean is appropriate for the current dataset, though not tailored by feature or group.
 - Clustering methods assume standardized numerical data.
 - DBSCAN uses fixed parameters, which may not be optimal for all datasets.
 - No categorical or text features are handled in this version.
 - The dataset contains a sufficient number of rows (i.e., clustering is meaningful).

## Future works
 - Smarter imputation strategies (e.g., by group or datatype).
 - Support for categorical or mixed-type data.
 - Automatic tuning of DBSCAN parameters.
 

# Simple LLM Clustering Agent

This module adds a simple, task-specific LLM agent built with LangChain that can interpret natural language instructions and trigger the clustering pipeline from the previous section. It wraps the functionality of the `ClusteringAgent` in a lightweight, language-driven interface.

The `SimpleLLM` agent uses `OPENAI_FUNCTIONS` agent type to simulate a chat-based interaction. It detects when a user message intends to cluster a dataset and extracts the relevant file path. It then calls a registered tool that invokes the `ClusteringAgent`, runs the pipeline, and returns a response.

## Run it

```bash
python run_agent.py
```

Example interaction:

```
¿Podrías clusterizar el archivo iris_data.csv?
```

The agent will extract the CSV file name, trigger the clustering tool, and return a message confirming the results were generated and saved.

## How It Works

- The `SimpleLLM` class implements the LangChain `BaseLanguageModel` interface.
- The `predict_messages` method inspects user inputs to determine intent and file path using regular expressions (this is probably a simplification and a more sophisticade predictive module should be implemented in the future if we want to extend the agent functionality).
- If a valid clustering request is detected, it returns a `function_call` that triggers the `clusterize_dataset` tool.
- The tool wraps the `ClusteringAgent` pipeline and returns a text summary similar than the previous Agent.
- The agent then replies with a final message thanking the user.

## Assumptions and Simplifications

- Only one task is supported: clustering a CSV dataset.
- The model looks for Spanish trigger word like "clusteriza" or the English word "clustering".
- CSV file paths are extracted using a regular expression and must match the pattern `*.csv`.
- If multiple CSV files are mentioned, only the first is used.
- If a result has already been received (as a `FunctionMessage`), the agent replies with a fixed message: `"`Thanks!"
- Inputs that do not meet the conditions are answered with: `"Sorry, I don't understand the instruction"`


## Architecture Overview

The diagram below outlines the message flow within the LangChain agent system:

```
User Input (e.g., "clusteriza iris.csv")
        │
LangChain Agent (OPENAI_FUNCTIONS)
        │
SimpleLLM.predict_messages(messages)
        │
        ├─ If FunctionMessage: return static "received" response (Thanks!)
        ├─ If HumanMessage:
        │     ├─ Extract CSV filename
        │     └─ Return function_call to 'clusterize_dataset'
Tool Call Executed: clusterize_dataset(path_csv)
        │
the tool calls ClusteringAgent, runs pipeline and saves results
        │
Tool Result (as FunctionMessage) sent back to SimpleLLM
        │
Final AIMessage generated and returned to user
```

## Future Improvements

- Support for more general-purpose intent parsing.
- Multi-task or multi-step conversations.
- Allow multiple file handling or prompt disambiguation.

