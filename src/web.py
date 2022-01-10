import streamlit as st
import streamlit.components.v1 as stc
from streamlit_autorefresh import st_autorefresh
import os, time
# File Processing Pkgs
import pandas as pd
from os.path import isfile 

def lock():
    if not isfile("access.lock"):
        with open("access.lock", "w") as f:
            f.write("locked")
            
def unlock():
    if isfile("access.lock"):
        os.remove("access.lock")
        
def islocked():
    if isfile("access.lock"):
        return True
    return False

def main():
    st.title("File Upload Tutorial")
    menu   = ["Train","Default Prediction","Custom File Prediction","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Train":
        if islocked():
            st.write("Prediction is under process. Try again after sometime.")
        else:
            lock()
            st.subheader("Train")
            if st.button("Train"):			
                st.info("Training started.")
                def open_file(file_path="src/logs/scraper.log"):
                    while True:
                        if not islocked():
                            break
                        with open(file_path, 'r') as f:
                            return f.read()
                
                prediction()
                
                unlock()
            
                
    elif choice == "Default Prediction":
        if islocked():
            st.write("Prediction is under process. Try again after sometime.")
        else:
            lock()
            st.subheader("Prediction (default)")
            if st.button("Predict"):			
                st.info("Prediction started.")
                st.success("Prediction Successful")
            unlock()
            
                
    elif choice == "Custom File Prediction":
        st.subheader("Prediction (custom)")
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if islocked():
            st.write("Prediction is under process.")
        else:
            lock()
            if st.button("Predict"):
                if data_file is not None:			
                    st.info("Prediction started.")
                    df = pd.read_csv(data_file)
                    st.dataframe(df)
                    st.success("Prediction Successful")
            unlock()
            
                        
    elif choice == "About":
        st.subheader("About")
        st.write("This project demonstrates thyroid prediction using machine learning.")
        st.write("Developed by: Govind Choudhary and Sreejith Subhash")
                
if __name__ == '__main__':
	main()