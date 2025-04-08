import streamlit as st
import pandas as pd
import os

# Configure the page
st.set_page_config(
    page_title="Student Prediction System",
    page_icon="📚",
    layout="wide"
)

# Display logo if available
if os.path.exists("data/logo.png"):
    st.image("data/logo.png", width=200)
    
# Main title
st.title("Student Performance & Behavior Prediction System")
st.markdown("---")

# App description
st.markdown("""
This application helps predict student performance and behavior based on various factors.
Choose your role below to get started:
""")

# Role selection section
st.subheader("Select Your Role")
col1, col2 = st.columns(2)

with col1:
    st.info("### Student")
    st.write("Input your information to get personalized predictions about your academic performance.")
    if st.button("Go to Student Login"):
        st.switch_page("pages/Student_Login.py")

with col2:
    st.info("### Teacher")
    st.write("Analyze student data and generate performance predictions for your class.")
    if st.button("Go to Teacher Login"):
        st.switch_page("pages/Teacher_Login.py")

# Display sample data info
st.subheader("Sample Data Available")
if os.path.exists("data/Students_Grading_Dataset.csv"):
    try:
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
        sample_data = None
        
        for encoding in encodings:
            try:
                sample_data = pd.read_csv("data/Students_Grading_Dataset.csv", encoding=encoding)
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
        
        if sample_data is None:
            st.error("Unable to read dataset with any of the attempted encodings.")
        else:
            # Check if the dataset has the expected columns
            required_columns = ['Student_ID', 'First_Name', 'Last_Name', 'Age', 'Gender', 'Department']
            missing_columns = [col for col in required_columns if col not in sample_data.columns]
            
            if missing_columns:
                st.warning(f"The dataset is missing some expected columns: {', '.join(missing_columns)}")
            
            st.write(f"Sample dataset contains information for {len(sample_data)} students")
            
            # Show a preview of the data
            if st.checkbox("Show sample data preview"):
                try:
                    st.dataframe(sample_data.head(5))
                except Exception as e:
                    st.error(f"Error displaying data preview: {str(e)}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.write("Sample data file not found.")

# About section
st.markdown("---")
st.subheader("About This System")
st.write("""
This student prediction system uses advanced analytics to help students and teachers 
understand academic performance and behavior patterns. The system uses various factors 
including attendance, study habits, participation, and more to provide personalized insights.
""")

# Footer
st.markdown("---")
st.caption("© 2023 Student Prediction System | by Sujoy Kumar Saha") 