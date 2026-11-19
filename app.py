import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Titanic Passenger Insights",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0F2942"
TEAL = "#148F77"
SLATE = "#5D6D7E"
LIGHT_BG = "#F7F9FC"

CATEGORY_COLORS = {
    "Survived": TEAL,
    "Did Not Survive": NAVY,
    "female": TEAL,
    "male": NAVY,
    "1": NAVY,
    "2": TEAL,
    "3": "#5499C7",
    "Child": "#1ABC9C",
    "Teen": "#3498DB",
    "Young Adult": NAVY,
    "Adult": "#2E4053",
    "Older Adult": "#7F8C8D",
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {LIGHT_BG};
            color: {NAVY};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        div[data-testid="stMetric"] {{
            background-color: #FFFFFF;
            border: 1px solid #DCE4EC;
            border-left: 4px solid {NAVY};
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .finding-box {{
            background-color: #FFFFFF;
            border-left: 4px solid {TEAL};
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 8px;
            border-top: 1px solid #E5ECF2;
            border-right: 1px solid #E5ECF2;
            border-bottom: 1px solid #E5ECF2;
        }}
        .note-caption {{
            color: {SLATE};
            font-size: 0.9rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prepare_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    try:
        raw_df = pd.read_csv("titanic.csv")
    except FileNotFoundError:
        raw_df = pd.read_csv(url)

    selected_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    _ = raw_df.loc[:, selected_cols]
    first_five_iloc = raw_df.iloc[:5, :]

    cleaned_df = raw_df.copy()

    missing_before = pd.DataFrame(
        {
            "Column": ["age", "embarked"],
            "Missing Before": [
                int(cleaned_df["age"].isna().sum()),
                int(cleaned_df["embarked"].isna().sum()),
            ],
        }
    )

    age_median = cleaned_df["age"].median()
    embarked_mode = cleaned_df["embarked"].mode()[0]

    cleaned_df["age"] = cleaned_df["age"].fillna(age_median)
    cleaned_df["embarked"] = cleaned_df["embarked"].fillna(embarked_mode)

    missing_after = pd.DataFrame(
        {
            "Column": ["age", "embarked"],
            "Missing After": [
                int(cleaned_df["age"].isna().sum()),
                int(cleaned_df["embarked"].isna().sum()),
            ],
        }
    )
    missing_summary = missing_before.merge(missing_after, on="Column")

    transformed_df = cleaned_df.copy()

    age_min, age_max = transformed_df["age"].min(), transformed_df["age"].max()
    fare_min, fare_max = transformed_df["fare"].min(), transformed_df["fare"].max()

    transformed_df["age_norm"] = (transformed_df["age"] - age_min) / (age_max - age_min)
    transformed_df["fare_norm"] = (transformed_df["fare"] - fare_min) / (fare_max - fare_min)

    bins = [-float("inf"), 12, 18, 35, 60, float("inf")]
    labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
    transformed_df["age_group"] = pd.cut(
        transformed_df["age"], bins=bins, labels=labels, right=True
    ).astype(str)

    sex_dummies = pd.get_dummies(transformed_df["sex"], prefix="sex", drop_first=True, dtype=int)
    embarked_dummies = pd.get_dummies(
        transformed_df["embarked"], prefix="embarked", drop_first=True, dtype=int
    )

    transformed_df = pd.concat([transformed_df, sex_dummies, embarked_dummies], axis=1)
    transformed_df["pclass_str"] = transformed_df["pclass"].astype(str)
    transformed_df["survival_label"] = transformed_df["survived"].map(
        {1: "Survived", 0: "Did Not Survive"}
    )

    inspection_meta = {
        "rows": raw_df.shape[0],
        "cols": raw_df.shape[1],
        "columns": list(raw_df.columns),
        "dtypes": raw_df.dtypes.astype(str).reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        ),
        "raw_missing": raw_df.isna().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        ),
        "first_five": first_five_iloc,
        "age_median": age_median,
        "embarked_mode": embarked_mode,
        "missing_summary": missing_summary,
        "dummy_cols": list(sex_dummies.columns) + list(embarked_dummies.columns),
    }

    return raw_df, cleaned_df, transformed_df, inspection_meta


raw_df, cleaned_df, transformed_df, meta = load_and_prepare_data()
assert meta["rows"] == 891 and meta["cols"] == 15

st.sidebar.header("Filters")
st.sidebar.caption("Filter passengers to update KPIs and charts across tabs.")

class_options = ["All"] + sorted(transformed_df["pclass"].unique().tolist())
selected_class = st.sidebar.selectbox("Passenger class", options=class_options, index=0)

sex_options = ["All"] + sorted(transformed_df["sex"].unique().tolist())
selected_sex = st.sidebar.selectbox("Sex", options=sex_options, index=0)

age_group_order = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
available_age_groups = [g for g in age_group_order if g in transformed_df["age_group"].unique()]
age_options = ["All"] + available_age_groups
selected_age_group = st.sidebar.selectbox("Age group", options=age_options, index=0)

filtered_df = transformed_df.copy()
if selected_class != "All":
    filtered_df = filtered_df[filtered_df["pclass"] == selected_class]
if selected_sex != "All":
    filtered_df = filtered_df[filtered_df["sex"] == selected_sex]
if selected_age_group != "All":
    filtered_df = filtered_df[filtered_df["age_group"] == selected_age_group]

st.title("Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups")

tab_overview, tab_explore, tab_preprocessing, tab_explorer = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

with tab_overview:
    st.subheader("Key Metrics")

    total_passengers = len(filtered_df)
    survivors_count = int(filtered_df["survived"].sum()) if total_passengers > 0 else 0
    survival_rate = (survivors_count / total_passengers * 100) if total_passengers > 0 else 0.0
    avg_fare = float(filtered_df["fare"].mean()) if total_passengers > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Passengers in selection", f"{total_passengers:,}")
    k2.metric("Survivors", f"{survivors_count:,}")
    k3.metric("Survival rate", f"{survival_rate:.1f}%")
    k4.metric("Average fare", f"${avg_fare:.2f}")

    st.divider()

    if total_passengers == 0:
        st.warning("No passengers match the selected filter combination.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Survival Counts")
            surv_counts_df = (
                filtered_df["survival_label"]
                .value_counts()
                .reindex(["Did Not Survive", "Survived"], fill_value=0)
                .reset_index()
            )
            surv_counts_df.columns = ["Outcome", "Passengers"]

            fig_counts = px.bar(
                surv_counts_df,
                x="Outcome",
                y="Passengers",
                color="Outcome",
                color_discrete_map=CATEGORY_COLORS,
                title="Passenger Count by Survival Status",
                labels={"Outcome": "Survival Status", "Passengers": "Number of Passengers"},
                text="Passengers",
            )
            fig_counts.update_traces(textposition="outside")
            fig_counts.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=380,
            )
            st.plotly_chart(fig_counts, use_container_width=True)

        with col_right:
            st.subheader("Survival Rate by Passenger Class")
            class_surv_df = (
                filtered_df.groupby("pclass_str", observed=False)["survived"]
                .agg(["count", "mean"])
                .reset_index()
            )
            class_surv_df["survival_rate_pct"] = class_surv_df["mean"] * 100
            class_surv_df = class_surv_df.sort_values("pclass_str")

            fig_class_ov = px.bar(
                class_surv_df,
                x="pclass_str",
                y="survival_rate_pct",
                color="pclass_str",
                color_discrete_map=CATEGORY_COLORS,
                title="Survival Rate (%) Across Passenger Classes",
                labels={"pclass_str": "Passenger Class", "survival_rate_pct": "Survival Rate (%)"},
                text=class_surv_df["survival_rate_pct"].map(lambda v: f"{v:.1f}%"),
            )
            fig_class_ov.update_traces(textposition="outside")
            fig_class_ov.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, 108]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=380,
            )
            st.plotly_chart(fig_class_ov, use_container_width=True)

    st.markdown(
        '<p class="note-caption">Note: Survival rates and summary metrics describe this dataset only and do not imply causation.</p>',
        unsafe_allow_html=True,
    )

with tab_explore:
    st.subheader("Compare passenger groups")

    if len(filtered_df) == 0:
        st.warning("No passengers match the selected filter combination.")
    else:
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            rate_by_class = (
                filtered_df.groupby("pclass_str", observed=False)["survived"]
                .mean()
                .mul(100)
                .reset_index(name="survival_rate")
                .sort_values("pclass_str")
            )
            fig_exp_class = px.bar(
                rate_by_class,
                x="pclass_str",
                y="survival_rate",
                color="pclass_str",
                color_discrete_map=CATEGORY_COLORS,
                title="Survival Rate by Passenger Class",
                labels={"pclass_str": "Passenger Class", "survival_rate": "Survival Rate (%)"},
                text=rate_by_class["survival_rate"].map(lambda v: f"{v:.1f}%"),
            )
            fig_exp_class.update_traces(textposition="outside")
            fig_exp_class.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, 108]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=370,
            )
            st.plotly_chart(fig_exp_class, use_container_width=True)

        with row1_col2:
            rate_by_sex = (
                filtered_df.groupby("sex", observed=False)["survived"]
                .mean()
                .mul(100)
                .reset_index(name="survival_rate")
            )
            fig_exp_sex = px.bar(
                rate_by_sex,
                x="sex",
                y="survival_rate",
                color="sex",
                color_discrete_map=CATEGORY_COLORS,
                title="Survival Rate by Sex",
                labels={"sex": "Sex", "survival_rate": "Survival Rate (%)"},
                text=rate_by_sex["survival_rate"].map(lambda v: f"{v:.1f}%"),
            )
            fig_exp_sex.update_traces(textposition="outside")
            fig_exp_sex.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, 108]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=370,
            )
            st.plotly_chart(fig_exp_sex, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            fig_age_hist = px.histogram(
                filtered_df,
                x="age",
                nbins=20,
                color="survival_label",
                color_discrete_map=CATEGORY_COLORS,
                barmode="overlay",
                opacity=0.8,
                title="Age Distribution (Years)",
                labels={"age": "Age (Years)", "survival_label": "Survival Status"},
            )
            fig_age_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=370,
            )
            st.plotly_chart(fig_age_hist, use_container_width=True)

        with row2_col2:
            fig_fare_box = px.box(
                filtered_df.sort_values("pclass_str"),
                x="pclass_str",
                y="fare",
                color="pclass_str",
                color_discrete_map=CATEGORY_COLORS,
                title="Fare Distribution by Passenger Class",
                labels={"pclass_str": "Passenger Class", "fare": "Fare ($)"},
            )
            fig_fare_box.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=NAVY),
                height=370,
            )
            st.plotly_chart(fig_fare_box, use_container_width=True)

    st.markdown(
        """
        <div class="finding-box">
            <strong>Finding:</strong> Across the dataset, higher survival rates are observed among female passengers and passengers in higher ticket classes compared with male and third-class passengers.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_preprocessing:
    st.subheader("Dataset Inspection & Data Preparation Steps")

    with st.expander("Dataset Overview & Raw Inspection (.loc / .iloc)", expanded=False):
        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.write(f"**Shape:** {meta['rows']} rows × {meta['cols']} columns")
            st.write("**Columns:**", ", ".join(meta["columns"]))
            st.write("**Data Types:**")
            st.dataframe(meta["dtypes"], use_container_width=True, hide_index=True)
        with c_info2:
            st.write("**Raw Missing Values per Column:**")
            st.dataframe(meta["raw_missing"], use_container_width=True, hide_index=True)
            st.write("**First Five Rows (selected via .iloc):**")
            st.dataframe(
                meta["first_five"][["survived", "pclass", "sex", "age", "fare", "embarked"]],
                use_container_width=True,
            )

    st.markdown("#### 1. Missing values")
    st.caption(
        f"Missing age values were imputed with the median ({meta['age_median']:.2f}) and missing embarked values with the most frequent category ('{meta['embarked_mode']}'), while deck was left out due to extensive missingness."
    )
    st.dataframe(meta["missing_summary"], use_container_width=True, hide_index=True)

    st.markdown("#### 2. Normalization")
    st.caption(
        "Min-Max normalization rescales age and fare to a [0, 1] range while preserving the original age and fare columns."
    )
    norm_preview = transformed_df[["age", "age_norm", "fare", "fare_norm"]].head(10).round(4)
    st.dataframe(norm_preview, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Age discretization")
    st.caption(
        "Continuous age values were grouped into five ordered life-stage categories: Child (0–12), Teen (13–18), Young Adult (19–35), Adult (36–60), and Older Adult (61+)."
    )
    disc_preview = transformed_df[["age", "age_group"]].head(10)
    st.dataframe(disc_preview, use_container_width=True, hide_index=True)

    st.markdown("#### 4. Dummy coding")
    st.caption(
        "Binary indicator columns were created for sex and embarked using drop_first=True (removing one redundant reference column per variable) without dummy-coding survived or normalizing dummy columns."
    )
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.write("**Before (sex, embarked)**")
        st.dataframe(cleaned_df[["sex", "embarked"]].head(8), use_container_width=True, hide_index=True)
    with d_col2:
        st.write("**After (dummy variables, drop_first=True)**")
        st.dataframe(transformed_df[meta["dummy_cols"]].head(8), use_container_width=True, hide_index=True)

with tab_explorer:
    st.subheader("Interactive Data Explorer")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.8, 1])

    with ctrl_col1:
        view_choice = st.radio(
            "Select view",
            options=["Raw", "Cleaned", "Transformed"],
            index=2,
            horizontal=True,
        )

    if view_choice == "Raw":
        base_view_df = raw_df.copy()
        default_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    elif view_choice == "Cleaned":
        base_view_df = cleaned_df.copy()
        default_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    else:
        base_view_df = transformed_df.copy()
        default_cols = [
            "survived",
            "pclass",
            "sex",
            "age",
            "age_group",
            "age_norm",
            "fare",
            "fare_norm",
            "embarked",
        ] + meta["dummy_cols"]

    active_indices = filtered_df.index
    view_filtered_df = base_view_df.loc[base_view_df.index.isin(active_indices)]

    with ctrl_col2:
        selected_cols = st.multiselect(
            "Choose columns to display",
            options=list(base_view_df.columns),
            default=[c for c in default_cols if c in base_view_df.columns],
        )

    with ctrl_col3:
        sample_mode = st.selectbox(
            "Record view",
            options=["Filtered records", "Random sample (up to 25 rows)"],
            index=0,
        )

    display_df = view_filtered_df[selected_cols] if selected_cols else view_filtered_df
    if sample_mode != "Filtered records" and len(display_df) > 0:
        display_df = display_df.sample(n=min(25, len(display_df)), random_state=42)

    st.write(f"**Filtered row count:** {len(view_filtered_df):,} | **Showing:** {len(display_df):,} rows")
    st.dataframe(display_df, use_container_width=True, height=420)
