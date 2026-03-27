import streamlit as st

def apply_custom_css():
    st.sidebar.markdown("---")
    theme = st.sidebar.radio("🎨 Appearance", ["Dark", "Light"], horizontal=True)
    
    if theme == "Dark":
        bg_main = "#0B0F19"
        bg_sec  = "#1E293B"
        text    = "#E2E8F0"
        text_muted = "#94A3B8"
        prim    = "#6366F1"
        header_bg = "linear-gradient(135deg, #111827 0%, #1e1b4b 100%)"
        header_text = "linear-gradient(135deg, #A5B4FC 0%, #818CF8 50%, #C084FC 100%)"
        card_bg = "rgba(30, 41, 59, 0.4)"
        card_hover = "rgba(30, 41, 59, 0.7)"
        border  = "rgba(255, 255, 255, 0.05)"
        shadow  = "rgba(0, 0, 0, 0.2)"
        sidebar_bg = "#0B1120"
        metric_val = "linear-gradient(to right, #F8FAFC, #CBD5E1)"
    else:
        bg_main = "#F8FAFC"
        bg_sec  = "#FFFFFF"
        text    = "#0F172A"
        text_muted = "#64748B"
        prim    = "#4F46E5"
        header_bg = "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%)"
        header_text = "linear-gradient(135deg, #4338CA 0%, #4F46E5 50%, #6D28D9 100%)"
        card_bg = "rgba(255, 255, 255, 0.6)"
        card_hover = "rgba(255, 255, 255, 1)"
        border  = "rgba(0, 0, 0, 0.05)"
        shadow  = "rgba(0, 0, 0, 0.05)"
        sidebar_bg = "#F1F5F9"
        metric_val = "linear-gradient(to right, #0F172A, #334155)"

    st.session_state.current_theme = theme

    st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_main} !important;
    }}
    
    /* Fix top white strip (Streamlit Header) */
    [data-testid="stHeader"] {{
        background-color: {bg_main} !important;
    }}
    
    /* Ensure Deploy button and Menu text is visible */
    .stDeployButton > button span, [data-testid="stHeader"] button * {{
        color: {text} !important;
    }}
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    
    p, span, label {{
        color: {text} !important;
    }}

    .stMarkdown p, .stMarkdown div {{
        color: {text} !important;
    }}

    .header-container {{
        background: {header_bg};
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px -5px {shadow};
        border: 1px solid {border};
        position: relative;
        overflow: hidden;
    }}
    
    .header-title {{
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 10px;
        background: {header_text};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }}
    
    .header-subtitle {{
        font-size: 18px;
        font-weight: 400;
        color: {text_muted} !important;
        position: relative;
        z-index: 1;
    }}

    .section-header {{
        font-size: 24px;
        font-weight: 800;
        color: {text};
        margin-top: 40px;
        margin-bottom: 25px;
        border-bottom: 1px solid {border};
        padding-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .section-header::before {{
        content: '';
        display: inline-block;
        width: 12px;
        height: 24px;
        background: linear-gradient(to bottom, #6366F1, #8B5CF6);
        border-radius: 4px;
    }}

    .metric-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px {shadow};
        border: 1px solid {border};
        border-left: 4px solid #6366F1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px {shadow};
        border-left: 4px solid #A855F7;
        background: {card_hover};
    }}
    
    .metric-label {{
        color: {text_muted};
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }}
    
    .metric-value {{
        font-size: 36px;
        font-weight: 800;
        line-height: 1.2;
        background: {metric_val};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .metric-suffix {{
        color: {text_muted};
        font-size: 14px;
        font-weight: 500;
        margin-top: 8px;
    }}

    [data-testid="stMetric"] {{
        background: {card_bg};
        border: 1px solid {border};
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px {shadow};
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px {shadow};
        border-color: rgba(99,102,241,0.3);
    }}
    [data-testid="stMetricLabel"] {{
        color: {text_muted} !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    [data-testid="stMetricValue"] {{
        color: {text} !important;
        font-weight: 800;
        font-size: 36px;
    }}

    .stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stBaseButton-secondary"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        width: 100%;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover, [data-testid="stBaseButton-secondary"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.5);
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white !important;
    }}

    /* Forms and Input Fields */
    [data-testid="stForm"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }}
    /* Bulletproof Inputs (Text, Number, Select) */
    .stNumberInput > div > div > div, 
    .stTextInput > div > div > div,
    [data-testid="stNumberInput"] > div > div, 
    [data-testid="stTextInput"] > div > div,
    [data-testid="stNumberInputContainer"], 
    [data-testid="stTextInputContainer"] {{
        background-color: {bg_sec} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}
    
    .stNumberInput input, 
    .stTextInput input,
    [data-testid="stNumberInput"] input, 
    [data-testid="stTextInput"] input {{
        background-color: transparent !important;
        color: {text} !important;
    }}

    .stNumberInput button, 
    [data-testid="stNumberInput"] button {{
        background-color: transparent !important;
        color: {text} !important;
        border: none !important;
    }}
    .stNumberInput button:hover, 
    [data-testid="stNumberInput"] button:hover {{
        background-color: {bg_main} !important;
    }}
    
    .stSelectbox > div > div > div, 
    [data-testid="stSelectbox"] > div > div > div {{
        background-color: {bg_sec} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}
    
    .stSlider > div > div > div > div {{
        color: {text} !important;
    }}
    
    div[data-baseweb="popover"], div[role="listbox"], ul[role="listbox"] {{
        background-color: {bg_sec} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    /* File Uploader Dropzone */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {bg_sec} !important;
        border: 1px dashed {text_muted} !important;
        color: {text} !important;
    }}
    [data-testid="stFileUploadDropzone"] * {{
        color: {text} !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: {bg_sec} !important;
    }}
    [data-testid="stFileUploadDropzone"] button {{
        background-color: {prim} !important;
        color: white !important;
    }}

    /* HTML Tables */
    table, .dataframe {{
        width: 100% !important;
        color: {text} !important;
        background-color: {card_bg} !important;
        border-collapse: collapse !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    th, td {{
        padding: 12px 16px !important;
        border-bottom: 1px solid {border} !important;
    }}
    th {{
        background-color: rgba(0, 0, 0, 0.1) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
    }}
    
    [data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {border};
    }}
    
    .block-container {{
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }}
</style>
    """, unsafe_allow_html=True)

def update_plotly_layout(fig):
    theme = st.session_state.get("current_theme", "Dark")
    
    if theme == "Dark":
        bg = 'rgba(0,0,0,0)'
        text = '#CBD5E1'
        title = '#F8FAFC'
        grid = '#1E293B'
        legend_bg = 'rgba(15, 23, 42, 0.8)'
    else:
        bg = 'rgba(0,0,0,0)'
        text = '#475569'
        title = '#0F172A'
        grid = '#E2E8F0'
        legend_bg = 'rgba(255, 255, 255, 0.8)'

    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(family='Inter, sans-serif', color=text),
        title_font=dict(size=20, color=title, family='Inter, sans-serif'),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid),
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=grid,
            borderwidth=1
        )
    )
    return fig

def style_dataframe(df):
    """Applies theme-consistent Styler to DataFrames."""
    theme = st.session_state.get("current_theme", "Dark")
    if theme == "Dark":
        bg = "#1E293B"
        text = "#F8FAFC"
        border = "#334155"
    else:
        bg = "#FFFFFF"
        text = "#0F172A"
        border = "#E2E8F0"
        
    styler = df.style if hasattr(df, "style") else df
    return styler.set_properties(**{
        'background-color': bg,
        'color': text,
        'border-color': border
    })
