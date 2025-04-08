import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Teacher Login",
    page_icon="👨‍🏫",
    layout="wide"
)

st.title("Teacher Login")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Teacher Access")
    
    with st.form("teacher_login_form"):
        teacher_id = st.text_input("Teacher ID")
        password = st.text_input("Password", type="password")
        school_code = st.text_input("School Code")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if teacher_id and password and school_code:
                if teacher_id.startswith('T') and len(password) >= 6 and school_code == "DEMO2023":
                    st.success("Login successful!")
                    st.session_state.teacher_id = teacher_id
                    st.session_state.teacher_name = f"Teacher {teacher_id}"  # Set teacher name
                    st.session_state.is_admin = teacher_id in ["T001", "T002", "T003"]  # Example admin users
                    st.success(f"Welcome, {teacher_id}! Please go to the Teacher Dashboard.")
                    st.session_state.show_dashboard_button = True
                else:
                    st.error("Invalid credentials. Please check your ID, password, and school code.")
            else:
                st.error("Please enter Teacher ID, Password, and School Code.")
    
    with st.expander("Demo Credentials (For Testing)"):
        st.code("""
        Teacher ID: T001
        Password: teacher123
        School Code: DEMO2023
        """)
        
    st.info("For account issues, please contact your school administrator.")

with col2:
    st.image("https://img.freepik.com/free-vector/teacher-concept-illustration_114360-1638.jpg", width=400)
    st.markdown("""
    ### Teacher Portal Benefits
    - Access comprehensive student analytics
    - Generate performance predictions
    - Identify at-risk students
    - Create targeted intervention strategies
    - Export data for reporting
    """)

st.markdown("---")
st.warning("This system contains confidential student information. Unauthorized access is prohibited.")


col3, col4 = st.columns([1, 1])

with col3:
    if st.button("Student Login"):
        st.switch_page("pages/Student_Login.py")

with col4:
    if st.button("Return to Home"):
        st.switch_page("Home.py")

if st.session_state.get("show_dashboard_button", False):
    if st.button("Go to Teacher Dashboard"):
        st.switch_page("pages/Teacher_Student_Analysis.py") 