# Prepare pandas DataFrames and downloadable CSVs
import pandas as pd
import streamlit as st

def display_and_download_summary(summary_records, folder_summary_records, all_detections_records):
    if all_detections_records:
        df = pd.DataFrame(all_detections_records)
        st.download_button(
            label="⬇️ Download Confidence of Pinniped Detections (All Images)",
            data=df.to_csv(index=False),
            file_name="all_pinniped_detections.csv",
            mime="text/csv"
        )

    st.subheader("📊 Per-Image Summary")
    summary_df = pd.DataFrame(summary_records)
    st.dataframe(summary_df)
    
    # Generate filename with location and date from folder_summary_records
    if folder_summary_records and len(folder_summary_records) > 0:
        survey_location = folder_summary_records[0].get("survey_location", "unknown").replace(" ", "_")
        survey_date = str(folder_summary_records[0].get("date", "")).replace("-", "")
        image_summary_filename = f"image_summary_{survey_location}_{survey_date}.csv"
    else:
        image_summary_filename = "image_summary.csv"
    
    st.download_button(
        "Download Image Summary CSV", 
        summary_df.to_csv(index=False), 
        image_summary_filename
    )

    st.subheader("📍 Per-Location Summary")
    folder_df = pd.DataFrame(folder_summary_records)
    st.dataframe(folder_df)
    
    # Generate filename with location and date from folder_summary_records
    if folder_summary_records and len(folder_summary_records) > 0:
        survey_location = folder_summary_records[0].get("survey_location", "unknown").replace(" ", "_")
        survey_date = str(folder_summary_records[0].get("date", "")).replace("-", "")
        location_summary_filename = f"location_summary_{survey_location}_{survey_date}.csv"
    else:
        location_summary_filename = "location_summary.csv"
    
    st.download_button(
        "Download Location Summary CSV", 
        folder_df.to_csv(index=False), 
        location_summary_filename
    )
