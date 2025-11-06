import streamlit as st
from predict import predict

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
        st.write("""
        **Model Info:**
        - Algorithm: Random Forest Classifier  
        - Input features: Sepal and Petal dimensions  
        - Output: Predicted species — *Setosa*, *Versicolor*, or *Virginica*
        """)

        with st.expander("See how prediction works"):
            st.write("""
            The model uses the four numeric features of the Iris dataset:
            - Sepal Length  
            - Sepal Width  
            - Petal Length  
            - Petal Width  

            These are fed into a trained machine learning model that outputs the most likely species.
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
    st.write("Created using **Streamlit**.")
