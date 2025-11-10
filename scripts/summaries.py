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
    st.download_button("Download Image Summary CSV", summary_df.to_csv(index=False), "image_summary.csv")

    st.subheader("📍 Per-Location Summary")
    folder_df = pd.DataFrame(folder_summary_records)
    st.dataframe(folder_df)
    st.download_button("Download Location Summary CSV", folder_df.to_csv(index=False), "unique_counts.csv")
