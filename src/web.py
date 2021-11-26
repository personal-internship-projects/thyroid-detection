import streamlit as st
import streamlit.components.v1 as stc
from streamlit_autorefresh import st_autorefresh
import os, time
# File Processing Pkgs
import pandas as pd

def main():
    st.title("File Upload Tutorial")
    menu   = ["Train","Default Prediction","Custom File Prediction","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Train":
        st.subheader("Train")
        if st.button("Train"):			
            st.info("Training started.")
            def load_logs_file():
                dir_root = os.path.dirname(os.path.abspath(__file__))
                logs_file = open(dir_root + '/src/logs/scraper.log', 'r')
                return logs_file
            st.write('Refresh to see logs.')
            list_of_logs = ""
            while True:     
                z = st.empty()       
                logs_data = load_logs_file()
                time.sleep(3)
                for line in logs_data:
                    list_of_logs += line            
                xz = z.code(list_of_logs)
                st.write(xz)
            st.success("Training Successful")
            
    elif choice == "Default Prediction":
        st.subheader("Prediction (default)")
        if st.button("Predict"):			
            st.info("Prediction started.")
            st.success("Prediction Successful")
            
    elif choice == "Custom File Prediction":
        st.subheader("Prediction (custom)")
        data_file = st.file_uploader("Upload CSV",type=['csv'])

        if st.button("Predict"):
            if data_file is not None:			
                st.info("Prediction started.")
                df = pd.read_csv(data_file)
                st.dataframe(df)
                st.success("Prediction Successful")
                
    elif choice == "About":
        st.subheader("About")
        st.write("This project demonstrates thyroid prediction using machine learning.")
        st.write("Developed by: Govind Choudhary and Sreejith Subhash")
                
if __name__ == '__main__':
	main()