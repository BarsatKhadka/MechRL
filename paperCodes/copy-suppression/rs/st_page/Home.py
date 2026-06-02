import streamlit as st

import os, sys
from pathlib import Path

# Reason we need this code: if we don't have it, then we default to importing the version of transformer_lens from site-packages instead
# (please correct me if wrong!)

DEBUG = False

import sys, os
# C:/Users/calsm/Documents/AI Alignment/SERIMATS_23/SERI-MATS-2023-Streamlit-pages/rs/st_page
root_dir = __file__.replace("\\", "/").split("/rs/")[0]
st_page_dir = root_dir + "/rs/st_page"
ST_HTML_PATH = Path(st_page_dir + "/media")

if DEBUG:
    st.write(os.getcwd())
    st.write("st_page_dir:", st_page_dir)
    st.write("root_dir:", root_dir, os.path.exists(root_dir))
    st.write("ST_HTML_PATH:", ST_HTML_PATH, os.path.exists(ST_HTML_PATH))


import platform
is_local = (platform.processor() != "")
NEGATIVE_HEADS = sorted([(10, 7), (11, 10)])
HTML_PLOTS_FILENAME = "GZIP_HTML_PLOTS_b51_s61_smaller.pkl"

st.markdown(
r"""
# Explore Prompts

This page was designed to help explore different prompts for GPT-2 Small, as part of a research project regarding copy-suppression in LLMs. We focus on negative behaviour (specifically copy-suppression in heads 10.7 and 11.10 for GPT2-small) and backup behaviour (specifically in the IOI task).

The goals of this page are:

* Help us keep track of our work in an accessible, readable way (rather than having everything dumped into messy notebooks and directories which we'll never return to),
* Provide a sandbox environment to help us spot interesting things about the behaviour of negative heads which we might otherwise have missed (e.g. their behaviour on bigrams),
* Make our work more accesible to others.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/header.png" width="500">
""", unsafe_allow_html=True)
