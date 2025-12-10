import json
from pathlib import Path

import streamlit as st
from predict import predict

MLFLOW_URL = "http://localhost:5000"

def load_model_meta():
    meta_path = Path(__file__).with_name("model_meta.json")
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


model_meta = load_model_meta()

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a section", ["Home", "Predict", "About"])

if page == "Home":
    st.title("Iris Species Prediction App")
    st.write("""
        Welcome!  
        This interactive Streamlit app predicts the **species of Iris flowers** based on user input.
    
        You can explore:
        - **Predict tab** to test model predictions  
        - **About tab** to learn how it works
        """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/1024px-Blue_Flag%2C_Ottawa.jpg",
        caption="Example of an Iris flower",
        use_container_width=True
    )

elif page == "Predict":
    st.title("Iris Species Prediction")

    st.write("Enter the following features of the Iris flower:")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input('Sepal Length (cm)', 4.0, 8.0, step=0.1)
        petal_length = st.number_input('Petal Length (cm)', 1.0, 7.0, step=0.1)

    with col2:
        sepal_width = st.number_input('Sepal Width (cm)', 2.0, 5.0, step=0.1)
        petal_width = st.number_input('Petal Width (cm)', 0.1, 2.5, step=0.1)

    tab1, tab2 = st.tabs(["Prediction", "Details"])

    with tab1:
        with st.container():
            if st.button('Predict'):
                features = [sepal_length, sepal_width, petal_length, petal_width]
                predicted_species = predict(features)
                st.success(f"Predicted Iris Species: **{predicted_species}**")

    with tab2:
        st.subheader("Model Info")

        if model_meta is None:
            st.warning(
                "No `model_meta.json` found. "
                "Run the training script to generate the model and metadata."
            )
            st.write("""
            By default, the app expects a trained model and metadata file in the `app/` directory:
            - `model.joblib` – serialized model  
            - `model_meta.json` – information about the best model and its metrics
            """)
        else:
            best_model = model_meta.get("best_model", "Unknown model")
            metrics = model_meta.get("metrics", {})
            accuracy = metrics.get("accuracy")
            f1_macro = metrics.get("f1_macro")
            version = model_meta.get("version", "N/A")
            run_id = model_meta.get("mlflow_run_id", "N/A")

            st.markdown(f"- **Best model:** `{best_model}`")
            if accuracy is not None:
                st.markdown(f"- **Accuracy:** `{accuracy:.4f}`")
            if f1_macro is not None:
                st.markdown(f"- **F1 (macro):** `{f1_macro:.4f}`")
            st.markdown(f"- **App / model version:** `{version}`")
            st.markdown(f"- **MLflow run id:** `{run_id}`")

            with st.expander("Raw metadata (JSON)"):
                st.json(model_meta)

        st.markdown("---")
        st.markdown(
            f"📊 **View MLflow experiment dashboard:** "
            f"[Open MLflow UI]({MLFLOW_URL})"
        )
        st.write("""
        The model uses the four numeric features of the Iris dataset:

        - Sepal Length  
        - Sepal Width  
        - Petal Length  
        - Petal Width  

        These are fed into a trained machine learning model that outputs the most likely species
        (*Setosa*, *Versicolor*, or *Virginica*).
        """)

elif page == "About":
    st.title("About This App")
    st.write("""
    This app demonstrates:
    - **Basic layout** and **columns**
    - **Containers** for grouping content
    - **Tabs** for organizing sections
    - **Expandable panels** for hidden details
    - **Sidebar navigation** for multi-page apps
    """)

    if model_meta:
        st.markdown("### Current model")
        st.markdown(f"- Best model: `{model_meta.get('best_model', 'Unknown')}`")
        st.markdown(f"- Version: `{model_meta.get('version', 'N/A')}`")

    st.write("Created using **Streamlit**.")
