import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Student Login",
    page_icon="👨‍🎓",
    layout="wide"
)

st.title("Student Login")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Student Access")
    
    with st.form("student_login_form"):
        student_id = st.text_input("Student ID")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if student_id and password:
                # For demonstration, we'll use a simple validation
                # In a real application, you would verify against a secure database
                if len(student_id) >= 4 and len(password) >= 4:  # Removed isdigit() check to allow alphanumeric IDs
                    # Check if the sample data file exists and load student IDs
                    if os.path.exists("data/Students_Grading_Dataset.csv"):
                        try:
                            # Try multiple encodings
                            encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
                            df = None
                            
                            for encoding in encodings:
                                try:
                                    df = pd.read_csv("data/Students_Grading_Dataset.csv", encoding=encoding)
                                    break  # If successful, exit the loop
                                except UnicodeDecodeError:
                                    continue  # Try the next encoding
                            
                            if df is None:
                                st.error("Unable to read dataset with any of the attempted encodings.")
                                student_ids = []
                            else:
                                # Check if Student_ID column exists
                                if 'Student_ID' not in df.columns:
                                    st.error("Dataset does not contain a Student_ID column.")
                                    student_ids = []
                                else:
                                    student_ids = df['Student_ID'].tolist()  # Keep original format
                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")
                            student_ids = []
                    else:
                        student_ids = []
                    
                    if student_id in student_ids:
                        # Student exists in database
                        st.success("Login successful!")
                        
                        try:
                            # Get student info
                            student_info = df[df['Student_ID'] == student_id].iloc[0]
                            
                            # Check if required columns exist
                            if 'First_Name' not in student_info or 'Last_Name' not in student_info:
                                st.error("Student record is missing name information.")
                                student_name = f"Student {student_id}"
                            else:
                                student_name = f"{student_info['First_Name']} {student_info['Last_Name']}"
                            
                            # Store student info in session state
                            st.session_state.student_id = student_id
                            st.session_state.student_name = student_name
                            
                            # Redirect to student input page
                            st.success(f"Welcome, {st.session_state.student_name}! Please go to the Student Input page.")
                            st.session_state.show_student_input_button = True
                        except Exception as e:
                            st.error(f"Error retrieving student information: {str(e)}")
                    else:
                        st.error("Student ID not found. Please check your ID.")
                else:
                    st.error("Invalid ID or password format.")
            else:
                st.error("Please enter both Student ID and Password.")
    
    # Student input button outside the form
    if st.session_state.get("show_student_input_button", False):
        if st.button("Go to Student Input"):
            st.switch_page("pages/Student_Input.py")
    
    st.info("Don't have an account? Please contact your school administrator.")

with col2:
    st.image("https://img.freepik.com/free-vector/online-education-concept-illustration_114360-4735.jpg", width=400)
    st.markdown("""
    ### Benefits of Student Account
    - Track your academic performance
    - Get personalized recommendations
    - Monitor your progress over time
    - Access resources to improve your studies
    """)

# Navigation options
st.markdown("---")
col3, col4 = st.columns([1, 1])

with col3:
    if st.button("Teacher Login"):
        st.switch_page("pages/Teacher_Login.py")

with col4:
    if st.button("Return to Home"):
        st.switch_page("Home.py")